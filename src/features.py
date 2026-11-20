# -*- coding: utf-8 -*-
"""
src/features.py - MODULE TRÍCH CHỌN VÀ RÚT GỌN ĐẶC TRƯNG
Đặc trưng hình ảnh phục vụ bài toán Khai phá dữ liệu & Phân lớp Món ăn miền Bắc.

Phương pháp:
1. Đặc trưng màu sắc: HSV Color Histogram (96 chiều)
2. Đặc trưng hình dạng & cấu trúc: HOG (Histogram of Oriented Gradients)
3. Đặc trưng kết cấu bề mặt: LBP (Local Binary Patterns) & GLCM (Ma trận đồng xuất độ xám)
4. Chuẩn hóa & Rút gọn chiều: StandardScaler + PCA (Fit CHỈ TRÊN TRAIN)
5. Lựa chọn đặc trưng: VarianceThreshold, SelectKBest, RF Feature Importance
6. Trực quan hóa không gian dữ liệu: PCA 2D và t-SNE
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent


# ==============================================================================
# 1. CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG TRUYỀN THỐNG (HAND-CRAFTED FEATURES)
# ==============================================================================

def extract_color_histogram_hsv(image_rgb: np.ndarray, bins_per_channel: int = 32) -> np.ndarray:
    """
    Trích xuất biểu đồ tần suất màu sắc trong không gian HSV (Hue - Saturation - Value).
    Không gian HSV phản ánh màu sắc món ăn độc lập tương đối với điều kiện chiếu sáng.
    Kích thước vector: 32 x 3 = 96 chiều.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Kênh H: [0..180], Kênh S: [0..256], Kênh V: [0..256]
    hist_h = cv2.calcHist([hsv], [0], None, [bins_per_channel], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins_per_channel], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins_per_channel], [0, 256])
    
    # Chuẩn hóa L1 cho từng histogram kênh
    hist_h = cv2.normalize(hist_h, hist_h, norm_type=cv2.NORM_L1).flatten()
    hist_s = cv2.normalize(hist_s, hist_s, norm_type=cv2.NORM_L1).flatten()
    hist_v = cv2.normalize(hist_v, hist_v, norm_type=cv2.NORM_L1).flatten()
    
    return np.hstack([hist_h, hist_s, hist_v]).astype(np.float32)


def extract_hog_features(image_rgb: np.ndarray) -> np.ndarray:
    """
    Trích xuất đặc trưng Histogram of Oriented Gradients (HOG):
    Mô tả đường nét, hình dáng bao quanh của món ăn (bát phở tròn, đĩa xôi vòm, cuốn dài,...).
    orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        channel_axis=None
    )
    return features.astype(np.float32)


def extract_lbp_features(image_rgb: np.ndarray, num_points: int = 24, radius: int = 3) -> np.ndarray:
    """
    Trích xuất đặc trưng Local Binary Patterns (LBP) bất biến theo phép quay (uniform LBP):
    Mô tả độ mịn, vân bề mặt (hạt cơm, sợi bún, độ xốp của chả cá, bề mặt bánh đúc).
    Kích thước vector: num_points + 2 = 26 chiều.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=num_points, R=radius, method="uniform")
    
    # Tính biểu đồ phân bố LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_glcm_features(image_rgb: np.ndarray) -> np.ndarray:
    """
    Trích xuất các chỉ số kết cấu Haralick từ ma trận đồng xuất độ xám (GLCM):
    Tính toán tương phản (Contrast), đồng nhất (Homogeneity), năng lượng (Energy), 
    tương quan (Correlation) và bất đồng (Dissimilarity) qua 4 hướng (0, 45, 90, 135 độ).
    Kích thước vector: 5 chỉ số x 4 hướng = 20 chiều.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Lượng tử hóa độ xám về 32 mức để tính nhanh và ổn định
    gray_quantized = (gray // 8).astype(np.uint8)
    
    glcm = graycomatrix(
        gray_quantized, 
        distances=[1, 2], 
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        levels=32, 
        symmetric=True, 
        normed=True
    )
    
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    glcm_feat = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
    return glcm_feat.astype(np.float32)


def extract_combined_traditional_features(image_rgb: np.ndarray) -> np.ndarray:
    """
    Ghép nối tất cả các họ đặc trưng thủ công thành 1 vector tổng hợp:
    [Color Hist HSV (96) + HOG (~1500-1800) + LBP (26) + GLCM (40)]
    Tự động xử lý kiểm tra và triệt tiêu NaN/Inf.
    """
    feat_color = extract_color_histogram_hsv(image_rgb)
    feat_hog = extract_hog_features(image_rgb)
    feat_lbp = extract_lbp_features(image_rgb)
    feat_glcm = extract_glcm_features(image_rgb)
    
    combined = np.hstack([feat_color, feat_hog, feat_lbp, feat_glcm])
    # Xử lý NaN hoặc Inf nếu có
    combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
    return combined


class FeaturePipelinePack:
    """Pipeline chuẩn hóa Z-score và giảm chiều PCA có thể pickle/joblib nhất quán."""
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


# ==============================================================================
# 2. TRÍCH XUẤT ĐẶC TRƯNG HÀNG LOẠT VÀ CACHE LƯU NPY
# ==============================================================================

def extract_features_from_csv(
    csv_path: Path, 
    label_map: Dict[str, int], 
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Đọc danh sách ảnh từ CSV, lần lượt đọc từng file ảnh,
    trích xuất vector đặc trưng và mã hóa nhãn lớp thành số nguyên.
    """
    df = pd.read_csv(csv_path)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
    X_list = []
    y_list = []
    valid_paths = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Trích đặc trưng {csv_path.stem}"):
        img_p = Path(row['image_path'])
        if not img_p.exists():
            continue
            
        bgr = cv2.imread(str(img_p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        feat = extract_combined_traditional_features(rgb)
        cls_name = row['class_name']
        
        if cls_name in label_map:
            label_idx = label_map[cls_name]
            X_list.append(feat)
            y_list.append(label_idx)
            valid_paths.append(str(img_p))
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, valid_paths


# ==============================================================================
# 3. CHUẨN HÓA, CHỌN LỌC ĐẶC TRƯNG VÀ RÚT GỌN CHIỀU PCA
# ==============================================================================

class FeatureReductionPipeline:
    """
    Đóng gói toàn diện quá trình:
    1. Loại bỏ đặc trưng có phương sai quá thấp (VarianceThreshold)
    2. Chuẩn hóa tỷ lệ Z-score (StandardScaler)
    3. Rút gọn chiều bằng Phân tích thành phần chính (PCA)
    
    Quy tắc vàng KDD:
    Chỉ .fit() DUY NHẤT trên tập Train, sau đó .transform() trên Validation và Test
    để ngăn chặn rò rỉ phân phối dữ liệu (Data Leakage).
    """
    def __init__(self, variance_thresh: float = 1e-4, pca_variance_ratio: float = 0.95):
        self.var_selector = VarianceThreshold(threshold=variance_thresh)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_variance_ratio, random_state=42)
        self.is_fitted = False
        
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        logger.info(f"Kích thước ban đầu tập Train: {X_train.shape}")
        
        # 1. Lọc phương sai thấp
        X_var = self.var_selector.fit_transform(X_train)
        logger.info(f"Sau VarianceThreshold: Giữ lại {X_var.shape[1]} / {X_train.shape[1]} đặc trưng")
        
        # 2. Fit Scaler
        X_scaled = self.scaler.fit_transform(X_var)
        
        # 3. Fit PCA
        X_pca = self.pca.fit_transform(X_scaled)
        explained_var = np.sum(self.pca.explained_variance_ratio_) * 100.0
        logger.info(f"Sau PCA: Giữ lại {X_pca.shape[1]} thành phần chính (Giải thích {explained_var:.2f}% phương sai)")
        
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được fit trên tập train!")
        X_var = self.var_selector.transform(X)
        X_scaled = self.scaler.transform(X_var)
        return self.pca.transform(X_scaled)
        
    def fit_transform(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X_train, y_train)
        return self.transform(X_train)


# ==============================================================================
# 4. TRỰC QUAN HÓA KHÔNG GIAN ĐẶC TRƯNG (PCA 2D VÀ t-SNE)
# ==============================================================================

def visualize_feature_space_2d(
    X_reduced: np.ndarray, 
    y: np.ndarray, 
    idx_to_class: Dict[int, str], 
    save_path: Path, 
    title: str = "Không gian đặc trưng 2D"
):
    """Vẽ biểu đồ phân tán 2D cho các lớp món ăn để quan sát mức độ phân tách cụm."""
    plt.figure(figsize=(13, 10))
    unique_labels = np.unique(y)
    palette = sns.color_palette("tab20", len(unique_labels))
    
    for idx, label_val in enumerate(unique_labels):
        mask = (y == label_val)
        class_label = idx_to_class.get(label_val, str(label_val))
        plt.scatter(
            X_reduced[mask, 0], 
            X_reduced[mask, 1], 
            label=class_label, 
            alpha=0.65, 
            s=35,
            color=palette[idx]
        )
        
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Thành phần 1", fontsize=12)
    plt.ylabel("Thành phần 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Đã lưu biểu đồ phân tán tại: {save_path}")


# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH GIAI ĐOẠN 3
# ==============================================================================

def run_feature_pipeline(
    base_data_dir: str = BASE_DIR / "data",
    models_dir: str = BASE_DIR / "models",
    reports_dir: str = BASE_DIR / "reports" / "figures",
    sample_limit_train: Optional[int] = None
):
    """Chạy trọn gói quá trình trích đặc trưng và chuẩn bị vector cho bước Modeling."""
    base_path = Path(base_data_dir)
    processed_dir = base_path / "processed"
    splits_dir = base_path / "splits"
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # Đọc label map
    label_map_file = splits_dir / "label_map.json"
    with open(label_map_file, 'r', encoding='utf-8') as f:
        label_map_data = json.load(f)
    class_to_idx = label_map_data['class_to_idx']
    idx_to_class = {int(k): v for k, v in label_map_data['idx_to_class'].items()}
    
    print("\n" + "="*80)
    print(" GIAI ĐOẠN 3: BẮT ĐẦU TRÍCH CHỌN VÀ RÚT GỌN ĐẶC TRƯNG ")
    print("="*80 + "\n")
    
    # 1. Trích đặc trưng thô
    X_train_raw, y_train, train_paths = extract_features_from_csv(
        processed_dir / "train_processed.csv", class_to_idx, max_samples=sample_limit_train
    )
    X_val_raw, y_val, val_paths = extract_features_from_csv(
        processed_dir / "val_processed.csv", class_to_idx
    )
    X_test_raw, y_test, test_paths = extract_features_from_csv(
        processed_dir / "test_processed.csv", class_to_idx
    )
    
    logger.info(f"Dữ liệu thô - Train: {X_train_raw.shape}, Val: {X_val_raw.shape}, Test: {X_test_raw.shape}")
    
    # 2. Fit pipeline rút gọn chiều CHỈ TRÊN TRAIN
    pipeline = FeatureReductionPipeline(variance_thresh=1e-4, pca_variance_ratio=0.95)
    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)
    
    logger.info(f"Dữ liệu sau PCA - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 3. Lưu mảng npy vào disk
    features_out_dir = base_path / "features"
    features_out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(features_out_dir / "X_train.npy", X_train)
    np.save(features_out_dir / "y_train.npy", y_train)
    np.save(features_out_dir / "X_val.npy", X_val)
    np.save(features_out_dir / "y_val.npy", y_val)
    np.save(features_out_dir / "X_test.npy", X_test)
    np.save(features_out_dir / "y_test.npy", y_test)
    
    # Lưu pipeline bằng joblib để phục vụ inference web
    joblib.dump(pipeline, models_path / "feature_pipeline.joblib")
    logger.info(f"Đã lưu feature_pipeline.joblib tại: {models_path}")
    
    # 4. Trực quan hóa PCA 2D và t-SNE trên tập Test
    logger.info("Sinh biểu đồ phân bố không gian đặc trưng PCA 2D & t-SNE...")
    pca_2d = PCA(n_components=2, random_state=42)
    X_test_pca2 = pca_2d.fit_transform(X_test)
    visualize_feature_space_2d(
        X_test_pca2, y_test, idx_to_class, 
        reports_path / "feature_pca_2d.png", 
        title=f"Trực quan hóa không gian đặc trưng {len(idx_to_class)} lớp qua PCA 2D (Test Set)"
    )
    
    # t-SNE (lấy tối đa 1000 mẫu để vẽ mượt)
    tsne_sample_size = min(1000, len(X_test))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_test_tsne = tsne.fit_transform(X_test[:tsne_sample_size])
    visualize_feature_space_2d(
        X_test_tsne, y_test[:tsne_sample_size], idx_to_class, 
        reports_path / "feature_tsne_2d.png", 
        title="Trực quan hóa mức độ phân tách cụm món ăn qua t-SNE (Test Set)"
    )
    
    print("\n" + "="*80)
    print(" HOÀN THÀNH GIAI ĐOẠN 3! VECTOR ĐẶC TRƯNG ĐÃ SẴN SÀNG CHO BƯỚC HUẤN LUYỆN.")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_feature_pipeline()
