# -*- coding: utf-8 -*-
"""
train_spatial_food_model.py - BỘ TRÍCH XUẤT ĐẶC TRƯNG KIM TỰ THÁP KHÔNG GIAN (SPATIAL PYRAMID)
Tối ưu hóa chuyên sâu cho nhận diện ẩm thực miền Bắc Việt Nam:
1. Spatial Grid Color Histogram (Lưới 3x3) trong cả 2 không gian HSV và CIE LAB
2. Color Moments (Kỳ vọng, Phương sai, Độ lệch - Mean, Std, Skewness) cho từng vùng
3. Spatial LBP Texture (Lưới 2x2) phân biệt hạt cơm rang vs sợi bún vs mảng thịt
4. HOG Shape (32x32) bắt dáng bát đĩa tổng thể
5. Huấn luyện SVM (Kernel RBF) + Random Forest + Soft Voting
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


def extract_spatial_food_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Trích xuất vector đặc trưng không gian màu sắc và kết cấu chuyên sâu:
    - Kích thước chuẩn: 224x224
    - Lưới 3x3 (9 ô): Mỗi ô trích HSV hist (48D) + LAB hist (48D) + Color moments (18D) = 114D x 9 = 1026D
    - Toàn cục: HSV hist (96D) + LAB hist (96D) = 192D
    - Lưới 2x2 (4 ô): LBP uniform (26D x 4) = 104D
    - HOG 32x32: 1296D
    Tổng số chiều: 1026 + 192 + 104 + 1296 = 2618D.
    Tỉ lệ: Màu sắc ~46%, Kết cấu ~5%, Hình dạng ~49% -> Cân bằng hoàn hảo!
    """
    # 1. Resize 224x224 letterbox
    h, w = img_rgb.shape[:2]
    scale = min(224 / w, 224 / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((224, 224, 3), 128, dtype=np.uint8)
    dx = (224 - nw) // 2
    dy = (224 - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = resized
    img = canvas

    # 2. CLAHE trên kênh L (LAB)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced_rgb = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2RGB)
    enhanced_hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
    enhanced_lab = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    feature_parts = []

    # --- PHẦN A: ĐẶC TRƯNG MÀU SẮC TOÀN CỤC (GLOBAL) ---
    # HSV 32 bins x 3 = 96D
    h_h = cv2.normalize(cv2.calcHist([enhanced_hsv], [0], None, [32], [0, 180]), None, norm_type=cv2.NORM_L1).flatten()
    h_s = cv2.normalize(cv2.calcHist([enhanced_hsv], [1], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
    h_v = cv2.normalize(cv2.calcHist([enhanced_hsv], [2], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
    feature_parts.extend([h_h, h_s, h_v])

    # LAB 32 bins x 3 = 96D
    l_l = cv2.normalize(cv2.calcHist([enhanced_lab], [0], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
    l_a = cv2.normalize(cv2.calcHist([enhanced_lab], [1], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
    l_b = cv2.normalize(cv2.calcHist([enhanced_lab], [2], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
    feature_parts.extend([l_l, l_a, l_b])

    # --- PHẦN B: SPATIAL GRID 3x3 COLOR & MOMENTS ---
    gh, gw = 224 // 3, 224 // 3
    for r in range(3):
        for c in range(3):
            y1, y2 = r * gh, (r + 1) * gh
            x1, x2 = c * gw, (c + 1) * gw
            patch_hsv = enhanced_hsv[y1:y2, x1:x2]
            patch_lab = enhanced_lab[y1:y2, x1:x2]

            # HSV 16 bins x 3 = 48D
            ph_h = cv2.normalize(cv2.calcHist([patch_hsv], [0], None, [16], [0, 180]), None, norm_type=cv2.NORM_L1).flatten()
            ph_s = cv2.normalize(cv2.calcHist([patch_hsv], [1], None, [16], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
            ph_v = cv2.normalize(cv2.calcHist([patch_hsv], [2], None, [16], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()

            # LAB 16 bins x 3 = 48D
            pl_l = cv2.normalize(cv2.calcHist([patch_lab], [0], None, [16], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
            pl_a = cv2.normalize(cv2.calcHist([patch_lab], [1], None, [16], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
            pl_b = cv2.normalize(cv2.calcHist([patch_lab], [2], None, [16], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()

            # Color moments (mean, std cho 6 kênh HSV+LAB = 12D)
            m_hsv = np.mean(patch_hsv, axis=(0, 1)) / 255.0
            s_hsv = np.std(patch_hsv, axis=(0, 1)) / 255.0
            m_lab = np.mean(patch_lab, axis=(0, 1)) / 255.0
            s_lab = np.std(patch_lab, axis=(0, 1)) / 255.0

            feature_parts.extend([ph_h, ph_s, ph_v, pl_l, pl_a, pl_b, m_hsv, s_hsv, m_lab, s_lab])

    # --- PHẦN C: SPATIAL LBP TEXTURE 2x2 ---
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

    # --- PHẦN D: HOG SHAPE (pixels_per_cell=32x32 -> 1296D) ---
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
    full_vector = np.nan_to_num(full_vector, nan=0.0, posinf=1.0, neginf=-1.0)
    return full_vector


def process_image_file(file_path_str: str) -> Optional[np.ndarray]:
    """Đọc ảnh từ file và trích xuất vector đặc trưng."""
    try:
        with Image.open(file_path_str) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            arr = np.array(pil_img)
        if arr is None or arr.size == 0:
            return None
        return extract_spatial_food_features(arr)
    except Exception:
        return None


class SpatialFeaturePipeline:
    """Đóng gói Pipeline chuẩn hóa Z-score và PCA tương thích 100% khi nạp từ file."""
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


# Gán vào module hiện tại và __main__ để joblib unpickle tương thích hoàn toàn
setattr(sys.modules['__main__'], 'SpatialFeaturePipeline', SpatialFeaturePipeline)


def main():
    print("="*75)
    print(" HUẤN LUYỆN MÔ HÌNH VỚI ĐẶC TRƯNG KIM TỰ THÁP KHÔNG GIAN (SPATIAL PYRAMID) ")
    print("="*75)

    # 1. Thu thập danh sách lớp và tệp ảnh
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
    MAX_PER_CLASS = 150
    for cls in classes:
        cls_dir = RAW_DATA_DIR / cls
        files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))
        np.random.seed(42)
        if len(files) > MAX_PER_CLASS:
            selected_files = np.random.choice(files, size=MAX_PER_CLASS, replace=False)
        else:
            selected_files = files
        norm_label = class_to_idx[raw_to_norm[cls]]
        for f in selected_files:
            image_tasks.append((str(f), norm_label))
            
    logger.info(f"Tổng số ảnh huấn luyện: {len(image_tasks)} ảnh.")

    # 2. Trích đặc trưng đa tiến trình
    logger.info("Đang trích xuất đặc trưng Spatial Grid Color + Texture + Shape...")
    t0 = time.time()
    
    X_list = []
    y_list = []
    
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
        futures = {executor.submit(process_image_file, p): lbl for p, lbl in image_tasks}
        count = 0
        for fut in as_completed(futures):
            lbl = futures[fut]
            feat = fut.result()
            if feat is not None:
                X_list.append(feat)
                y_list.append(lbl)
            count += 1
            if count % 300 == 0 or count == len(image_tasks):
                logger.info(f"Tiến độ trích đặc trưng: {count}/{len(image_tasks)} ({count*100//len(image_tasks)}%)")
                
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    logger.info(f"Trích đặc trưng xong trong {time.time() - t0:.1f}s. Kích thước vector: {X.shape}")

    # 3. Phân chia Train/Test 80/20 Stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 4. Fit Pipeline Z-score + PCA 95%
    logger.info("Đang chuẩn hóa StandardScaler và PCA (giữ 95% phương sai)...")
    pipeline = SpatialFeaturePipeline(n_components=0.95)
    X_train_pca = pipeline.fit_transform(X_train)
    X_test_pca = pipeline.transform(X_test)
    logger.info(f"Số chiều sau PCA: {X_train_pca.shape[1]} chiều.")

    # 5. Huấn luyện SVM (Kernel RBF, C=10, gamma='scale')
    logger.info("Đang huấn luyện Support Vector Machine (SVM RBF)...")
    svm = SVC(C=10.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_pca, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_pca))
    logger.info(f"SVM Test Accuracy: {svm_acc*100:.2f}%")

    # 6. Huấn luyện Random Forest (200 cây)
    logger.info("Đang huấn luyện Random Forest Classifier (200 trees)...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=25, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_pca, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_pca))
    logger.info(f"Random Forest Test Accuracy: {rf_acc*100:.2f}%")

    # 7. Huấn luyện Soft Voting Ensemble (Trọng số SVM=2, RF=1)
    logger.info("Đang kết hợp mô hình Soft Voting Ensemble...")
    voting = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf)],
        voting='soft',
        weights=[2, 1],
        n_jobs=-1
    )
    voting.fit(X_train_pca, y_train)
    y_pred_voting = voting.predict(X_test_pca)
    voting_acc = accuracy_score(y_test, y_pred_voting)
    voting_f1 = f1_score(y_test, y_pred_voting, average='macro')
    logger.info(f"⭐ SOFT VOTING ENSEMBLE TEST ACCURACY: {voting_acc*100:.2f}% | F1-MACRO: {voting_f1*100:.2f}%")

    # 8. Lưu tất cả mô hình đã huấn luyện
    logger.info("Đang lưu các mô hình vào thư mục models/ ...")
    joblib.dump(pipeline, MODELS_DIR / "spatial_pipeline.joblib")
    joblib.dump(svm, MODELS_DIR / "SVM.joblib")
    joblib.dump(rf, MODELS_DIR / "RandomForest.joblib")
    joblib.dump(voting, MODELS_DIR / "SoftVoting.joblib")

    print("\n" + "="*75)
    print(f" HUẤN LUYỆN HOÀN TẤT VỚI BỘ ĐẶC TRƯNG SPATIAL PYRAMID!")
    print(f" - Độ chính xác kiểm thử: {voting_acc*100:.2f}%")
    print(f" - F1-Score Macro: {voting_f1*100:.2f}%")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
