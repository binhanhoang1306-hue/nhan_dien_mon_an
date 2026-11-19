# -*- coding: utf-8 -*-
"""
train_production_model.py - HUẤN LUYỆN MÔ HÌNH CHÍNH XÁC CAO CHO 18 MÓN ĂN MIỀN BẮC
Pipeline đặc trưng: HSV Color Histogram + HOG + LBP + GLCM
Mô hình: SVM (RBF) + Random Forest + Soft Voting Ensemble
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List

import cv2
import numpy as np
from PIL import Image, ImageOps
import joblib

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline as SkPipeline

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(r"d:\Nhan_dien_mon_an")
RAW_DATA_DIR = BASE_DIR / "dataset" / "Images"
MODELS_DIR = BASE_DIR / "models"
SPLITS_DIR = BASE_DIR / "data" / "splits"
FEATURES_DIR = BASE_DIR / "data" / "features"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HÀM TRÍCH ĐẶC TRƯNG CHUẨN - DÙNG CHUNG CHO CẢ TRAIN VÀ PREDICT
# =============================================================================

def preprocess_image(img_path_or_rgb, target_size=224):
    """Tiền xử lý ảnh: đọc, sửa EXIF, resize giữ tỉ lệ + padding, CLAHE.
    
    Args:
        img_path_or_rgb: Đường dẫn ảnh (str/Path) hoặc numpy array RGB
        target_size: Kích thước đích (mặc định 224)
    Returns:
        enhanced_rgb: Ảnh RGB đã được cân bằng sáng (224x224x3)
    """
    if isinstance(img_path_or_rgb, (str, Path)):
        with Image.open(str(img_path_or_rgb)) as pil_img:
            try:
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception:
                pass
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img_rgb = np.array(pil_img)
    else:
        img_rgb = img_path_or_rgb

    if img_rgb is None or img_rgb.size == 0:
        return None

    # Resize giữ tỉ lệ + padding nền trung tính
    h, w = img_rgb.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    dy = (target_size - nh) // 2
    dx = (target_size - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    img_rgb = canvas

    # CLAHE trên kênh L (không gian LAB) để cân bằng sáng
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_ch)
    enhanced_rgb = cv2.cvtColor(cv2.merge([l_clahe, a_ch, b_ch]), cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def extract_features(enhanced_rgb):
    """Trích xuất đặc trưng đa góc nhìn từ ảnh đã tiền xử lý.
    
    Vector đầu ra: HSV Color Histogram (96D) + HOG (1764D) + LBP (26D) + GLCM (40D)
    Tổng: ~1926 chiều (trước PCA)
    
    Args:
        enhanced_rgb: Ảnh RGB 224x224 đã qua CLAHE
    Returns:
        full_vec: Vector đặc trưng numpy 1-D
    """
    # 1. Color Histogram HSV (96 chiều: 32 bins x 3 kênh)
    hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
    h_hist = cv2.normalize(cv2.calcHist([hsv], [0], None, [32], [0, 180]), None, norm_type=cv2.NORM_L1).flatten()
    s_hist = cv2.normalize(cv2.calcHist([hsv], [1], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
    v_hist = cv2.normalize(cv2.calcHist([hsv], [2], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
    feat_color = np.hstack([h_hist, s_hist, v_hist])

    # 2. HOG (Histogram of Oriented Gradients) - đặc trưng hình dạng/cạnh
    gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
    feat_hog = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )

    # 3. LBP (Local Binary Pattern) - đặc trưng kết cấu bề mặt (26 chiều)
    lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
    n_bins = int(lbp.max() + 1)
    feat_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # 4. GLCM Haralick (40 chiều) - đặc trưng kết cấu thống kê
    gray_sub = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(
        gray_sub, distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32, symmetric=True, normed=True
    )
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    feat_glcm = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

    # Ghép nối tất cả
    full_vec = np.hstack([feat_color, feat_hog, feat_lbp, feat_glcm]).astype(np.float32)
    full_vec = np.nan_to_num(full_vec, nan=0.0, posinf=1.0, neginf=-1.0)
    return full_vec


def process_single_image(img_path_str):
    """Wrapper cho multiprocessing: đọc + tiền xử lý + trích đặc trưng."""
    try:
        enhanced = preprocess_image(img_path_str)
        if enhanced is None:
            return None
        return extract_features(enhanced)
    except Exception:
        return None


# =============================================================================
# PIPELINE CHUẨN HÓA + GIẢM CHIỀU (DÙNG CHUNG CHO TRAIN VÀ PREDICT)
# =============================================================================

class FeaturePipeline:
    """Pipeline StandardScaler + PCA, dùng chung cho train và predict."""
    def __init__(self, n_components=0.95):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        return self.pca.fit_transform(X_scaled)

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)


# =============================================================================
# HUẤN LUYỆN
# =============================================================================

def main():
    print("=" * 75)
    print(" HUẤN LUYỆN MÔ HÌNH CHÍNH XÁC CAO - 18 MÓN ĂN MIỀN BẮC")
    print("=" * 75)

    # 1. Thu thập danh sách ảnh — sử dụng TẤT CẢ ảnh, lấy tối đa 500/lớp
    classes = sorted([d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for i, cls in enumerate(classes)}

    # Lưu label_map chuẩn (key = tên thư mục gốc, có dấu cách)
    label_map_data = {
        'class_to_idx': class_to_idx,
        'idx_to_class': {str(k): v for k, v in idx_to_class.items()},
    }
    with open(SPLITS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu label_map.json cho {len(classes)} lớp: {classes}")

    items = []
    MAX_PER_CLASS = 500  # Tăng lên 500 ảnh/lớp để học tốt hơn

    for cls in classes:
        cls_dir = RAW_DATA_DIR / cls
        files = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.jpeg")) + sorted(cls_dir.glob("*.png"))
        np.random.seed(42)
        if len(files) > MAX_PER_CLASS:
            selected = list(np.random.choice(files, size=MAX_PER_CLASS, replace=False))
        else:
            selected = files

        label = class_to_idx[cls]
        for f in selected:
            items.append((str(f), label))

    logger.info(f"Tổng số ảnh đưa vào huấn luyện: {len(items)}")

    # 2. Trích đặc trưng song song
    logger.info("Đang trích đặc trưng (HSV + HOG + LBP + GLCM) đa tiến trình...")
    t0 = time.time()

    X_list, y_list = [], []
    n_workers = min(os.cpu_count() or 4, 8)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_image, p): lbl for p, lbl in items}
        done = 0
        for fut in as_completed(futures):
            lbl = futures[fut]
            feat = fut.result()
            if feat is not None:
                X_list.append(feat)
                y_list.append(lbl)
            done += 1
            if done % 500 == 0 or done == len(items):
                logger.info(f"Tiến độ: {done}/{len(items)} ({done * 100 // len(items)}%)")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    logger.info(f"Trích đặc trưng xong trong {time.time() - t0:.1f}s. Ma trận: {X.shape}")

    # 3. Chia tập Stratified 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. StandardScaler + PCA (giữ 95% phương sai)
    logger.info("Fit StandardScaler + PCA trên tập train...")
    pipeline = FeaturePipeline(n_components=0.95)
    X_train_pca = pipeline.fit_transform(X_train)
    X_test_pca = pipeline.transform(X_test)
    logger.info(f"Số chiều sau PCA: {X_train_pca.shape[1]}")

    # 5. SVM (RBF) — tối ưu C và gamma
    logger.info("Huấn luyện SVM (RBF, C=10, gamma=scale, balanced)...")
    svm = SVC(C=10.0, kernel='rbf', gamma='scale', probability=True,
              class_weight='balanced', random_state=42, cache_size=1000)
    svm.fit(X_train_pca, y_train)
    svm_pred = svm.predict(X_test_pca)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred, average='macro')
    logger.info(f"SVM Accuracy: {svm_acc * 100:.2f}% | F1-macro: {svm_f1 * 100:.2f}%")

    # 6. Random Forest
    logger.info("Huấn luyện Random Forest (200 trees, balanced)...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=25, min_samples_split=3,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_pca, y_train)
    rf_pred = rf.predict(X_test_pca)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='macro')
    logger.info(f"Random Forest Accuracy: {rf_acc * 100:.2f}% | F1-macro: {rf_f1 * 100:.2f}%")

    # 7. Soft Voting (SVM 2 : RF 1)
    logger.info("Kết hợp Soft Voting Ensemble (SVM:2 + RF:1)...")
    voting = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf)],
        voting='soft', weights=[2, 1], n_jobs=-1
    )
    voting.fit(X_train_pca, y_train)
    vt_pred = voting.predict(X_test_pca)
    vt_acc = accuracy_score(y_test, vt_pred)
    vt_f1 = f1_score(y_test, vt_pred, average='macro')
    logger.info(f"Soft Voting Accuracy: {vt_acc * 100:.2f}% | F1-macro: {vt_f1 * 100:.2f}%")

    # Classification report
    print("\n" + "=" * 75)
    print("CLASSIFICATION REPORT (Soft Voting)")
    print("=" * 75)
    print(classification_report(y_test, vt_pred,
                                target_names=[idx_to_class[i] for i in range(len(classes))]))

    # 8. Lưu mô hình
    logger.info("Lưu mô hình vào models/ ...")
    joblib.dump(pipeline, MODELS_DIR / "feature_pipeline.joblib")
    joblib.dump(svm, MODELS_DIR / "SVM.joblib")
    joblib.dump(rf, MODELS_DIR / "RandomForest.joblib")
    joblib.dump(voting, MODELS_DIR / "SoftVoting.joblib")

    # Lưu features để phân tích
    np.save(FEATURES_DIR / "X_train.npy", X_train_pca)
    np.save(FEATURES_DIR / "y_train.npy", y_train)
    np.save(FEATURES_DIR / "X_test.npy", X_test_pca)
    np.save(FEATURES_DIR / "y_test.npy", y_test)

    print("\n" + "=" * 75)
    print(f" ✅ HUẤN LUYỆN THÀNH CÔNG!")
    print(f" SVM:     {svm_acc * 100:.2f}% acc | {svm_f1 * 100:.2f}% F1")
    print(f" RF:      {rf_acc * 100:.2f}% acc | {rf_f1 * 100:.2f}% F1")
    print(f" Voting:  {vt_acc * 100:.2f}% acc | {vt_f1 * 100:.2f}% F1")
    print(f" Models:  {MODELS_DIR}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
