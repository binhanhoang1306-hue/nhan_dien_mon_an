# -*- coding: utf-8 -*-
"""
train_perfect_model.py - NÂNG CẤP ĐẶC TRƯNG VÀ HUẤN LUYỆN TOÀN DIỆN
- Tăng cường dữ liệu (Augmentation) cho các lớp ít mẫu như Cơm rang dưa bò, Chả cá, Bún chả
- Trích xuất đặc trưng đa tầng: Màu sắc HSV + LAB (lưới 3x3) + LBP Texture + HOG
- Tinh chỉnh trọng số SVM & Random Forest
- Đảm bảo nhận diện chính xác các góc chụp phức tạp như Cơm rang dưa bò
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
import joblib

from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "dataset" / "Images"
MODELS_DIR = BASE_DIR / "models"
SPLITS_DIR = BASE_DIR / "data" / "splits"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def extract_advanced_features(img_rgb: np.ndarray) -> np.ndarray:
    """Trích xuất vector đặc trưng không gian màu sắc, kết cấu và hình dáng."""
    h, w = img_rgb.shape[:2]
    scale = min(224 / w, 224 / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((224, 224, 3), 128, dtype=np.uint8)
    dx = (224 - nw) // 2
    dy = (224 - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = resized
    img = canvas

    # CLAHE LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced_rgb = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2RGB)
    enhanced_hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
    enhanced_lab = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    feature_parts = []

    # 1. Global Color (HSV 96D + LAB 96D = 192D) - Tăng trọng số x1.5
    h_h = cv2.normalize(cv2.calcHist([enhanced_hsv], [0], None, [32], [0, 180]), None, norm_type=cv2.NORM_L1).flatten() * 1.5
    h_s = cv2.normalize(cv2.calcHist([enhanced_hsv], [1], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten() * 1.5
    h_v = cv2.normalize(cv2.calcHist([enhanced_hsv], [2], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten() * 1.5
    feature_parts.extend([h_h, h_s, h_v])

    l_l = cv2.normalize(cv2.calcHist([enhanced_lab], [0], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten() * 1.5
    l_a = cv2.normalize(cv2.calcHist([enhanced_lab], [1], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten() * 1.5
    l_b = cv2.normalize(cv2.calcHist([enhanced_lab], [2], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten() * 1.5
    feature_parts.extend([l_l, l_a, l_b])

    # 2. Spatial Grid 3x3 (Màu sắc và moments theo 9 vùng không gian)
    gh, gw = 224 // 3, 224 // 3
    for r in range(3):
        for c in range(3):
            y1, y2 = r * gh, (r + 1) * gh
            x1, x2 = c * gw, (c + 1) * gw
            patch_hsv = enhanced_hsv[y1:y2, x1:x2]
            patch_lab = enhanced_lab[y1:y2, x1:x2]

            ph_h = cv2.normalize(cv2.calcHist([patch_hsv], [0], None, [16], [0, 180]), None, norm_type=cv2.NORM_L1).flatten()
            ph_s = cv2.normalize(cv2.calcHist([patch_hsv], [1], None, [16], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
            ph_v = cv2.normalize(cv2.calcHist([patch_hsv], [2], None, [16], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()

            pl_l = cv2.normalize(cv2.calcHist([patch_lab], [0], None, [16], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
            pl_a = cv2.normalize(cv2.calcHist([patch_lab], [1], None, [16], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
            pl_b = cv2.normalize(cv2.calcHist([patch_lab], [2], None, [16], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()

            m_hsv = np.mean(patch_hsv, axis=(0, 1)) / 255.0
            s_hsv = np.std(patch_hsv, axis=(0, 1)) / 255.0
            m_lab = np.mean(patch_lab, axis=(0, 1)) / 255.0
            s_lab = np.std(patch_lab, axis=(0, 1)) / 255.0

            feature_parts.extend([ph_h, ph_s, ph_v, pl_l, pl_a, pl_b, m_hsv, s_hsv, m_lab, s_lab])

    # 3. Spatial LBP 2x2 Texture
    lbp_full = local_binary_pattern(gray, P=24, R=3, method="uniform")
    lh, lw = 224 // 2, 224 // 2
    for r in range(2):
        for c in range(2):
            y1, y2 = r * lh, (r + 1) * lh
            x1, x2 = c * lw, (c + 1) * lw
            patch_lbp = lbp_full[y1:y2, x1:x2]
            n_bins = int(lbp_full.max() + 1)
            p_hist, _ = np.histogram(patch_lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            feature_parts.append(p_hist)

    # 4. HOG Shape (32x32)
    feat_hog = hog(
        gray,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    feature_parts.append(feat_hog)

    full_vector = np.hstack(feature_parts).astype(np.float32)
    return np.nan_to_num(full_vector, nan=0.0, posinf=1.0, neginf=-1.0)


def extract_features_from_image_path(p_str: str, augment: bool = False) -> List[np.ndarray]:
    """Đọc ảnh và có thể sinh thêm bản biến thể tăng cường (Augmentation)."""
    try:
        with Image.open(p_str) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            arr = np.array(pil_img)
            
        results = [extract_advanced_features(arr)]
        
        if augment:
            # Lật ngang
            flipped = cv2.flip(arr, 1)
            results.append(extract_advanced_features(flipped))
            # Đổi sáng nhẹ
            bright = np.clip(arr.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
            results.append(extract_advanced_features(bright))
            
        return results
    except Exception:
        return []


class SpatialFeaturePipeline:
    def __init__(self, n_components=0.95):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        
    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        return self
        
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
        
    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        return self.pca.fit_transform(X_scaled)


setattr(sys.modules['__main__'], 'SpatialFeaturePipeline', SpatialFeaturePipeline)


def main():
    print("="*75)
    print(" BẮT ĐẦU HUẤN LUYỆN HOÀN CHỈNH: CÂN BẰNG MẪU & SPATIAL PYRAMID ")
    print("="*75)

    classes = sorted([d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
    class_to_idx = {cls.lower().replace(" ", "_"): i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls.lower().replace(" ", "_") for i, cls in enumerate(classes)}
    raw_to_norm = {cls: cls.lower().replace(" ", "_") for cls in classes}
    
    label_map_data = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'raw_to_norm': raw_to_norm
    }
    with open(SPLITS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_data, f, ensure_ascii=False, indent=2)

    image_tasks = []
    # Các lớp ít ảnh cần augment: com_rang_dua_bo, cha_ca_la_vong, bun_cha, banh_da_cua, banh_gai
    augmented_classes = {'Com rang dua bo', 'Cha ca La Vong', 'Bun cha', 'Banh da cua', 'Banh gai'}
    
    for cls in classes:
        cls_dir = RAW_DATA_DIR / cls
        files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))
        np.random.seed(42)
        
        should_aug = cls in augmented_classes
        if len(files) > 160:
            selected_files = np.random.choice(files, size=160, replace=False)
        else:
            selected_files = files
            
        norm_label = class_to_idx[raw_to_norm[cls]]
        for f in selected_files:
            image_tasks.append((str(f), norm_label, should_aug))
            
    logger.info(f"Tổng số tệp ảnh: {len(image_tasks)}")

    # Trích xuất đặc trưng
    X_list = []
    y_list = []
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
        futures = {executor.submit(extract_features_from_image_path, p, aug): lbl for p, lbl, aug in image_tasks}
        count = 0
        for fut in as_completed(futures):
            lbl = futures[fut]
            feats = fut.result()
            for feat in feats:
                X_list.append(feat)
                y_list.append(lbl)
            count += 1
            if count % 300 == 0 or count == len(image_tasks):
                logger.info(f"Tiến độ: {count}/{len(image_tasks)} ({count*100//len(image_tasks)}%) - Tổng vector: {len(X_list)}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    logger.info(f"Hoàn thành trích xuất trong {time.time() - t0:.1f}s. Kích thước: {X.shape}")

    # Phân chia Train/Test 80/20 Stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # PCA 95%
    logger.info("Fit StandardScaler + PCA (95% phương sai)...")
    pipeline = SpatialFeaturePipeline(n_components=0.95)
    X_train_pca = pipeline.fit_transform(X_train)
    X_test_pca = pipeline.transform(X_test)
    logger.info(f"Số chiều PCA: {X_train_pca.shape[1]}")

    # SVM C=15.0
    logger.info("Huấn luyện SVM RBF (C=15.0)...")
    svm = SVC(C=15.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_pca, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_pca))
    logger.info(f"SVM Test Accuracy: {svm_acc*100:.2f}%")

    # Random Forest 200 trees
    logger.info("Huấn luyện Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=25, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_pca, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_pca))
    logger.info(f"Random Forest Test Accuracy: {rf_acc*100:.2f}%")

    # Soft Voting
    voting = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf)],
        voting='soft',
        weights=[3, 1],
        n_jobs=-1
    )
    voting.fit(X_train_pca, y_train)
    y_pred = voting.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    logger.info(f"⭐ SOFT VOTING TEST ACCURACY: {acc*100:.2f}% | F1-MACRO: {f1*100:.2f}%")

    # Lưu models
    joblib.dump(pipeline, MODELS_DIR / "spatial_pipeline.joblib")
    joblib.dump(svm, MODELS_DIR / "SVM.joblib")
    joblib.dump(rf, MODELS_DIR / "RandomForest.joblib")
    joblib.dump(voting, MODELS_DIR / "SoftVoting.joblib")
    logger.info("Đã lưu toàn bộ mô hình tối ưu vào models/.")


if __name__ == "__main__":
    main()
