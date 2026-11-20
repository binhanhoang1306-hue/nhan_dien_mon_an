# -*- coding: utf-8 -*-
"""
notebooks/03_features.py - GIAI ĐOẠN 3: TRỰC QUAN HÓA CÁC HỌ ĐẶC TRƯNG HÌNH ẢNH
Quy trình CRISP-DM: Feature Engineering

Nội dung:
1. Trực quan hóa Histogram of Oriented Gradients (HOG) trên ảnh món ăn mẫu.
2. Trực quan hóa Biểu đồ màu HSV 3 kênh (Hue, Saturation, Value).
3. Trực quan hóa Bản đồ kết cấu Local Binary Patterns (LBP).
4. Phân tích ma trận đồng xuất độ xám (GLCM).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog, local_binary_pattern

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

REPORT_FIGS = BASE_DIR / "reports" / "figures"
REPORT_FIGS.mkdir(parents=True, exist_ok=True)


def visualize_feature_extraction():
    sample_img_path = BASE_DIR / "web" / "static" / "img" / "dishes" / "Pho.jpg"
    if not sample_img_path.exists():
        print("Chưa tìm thấy Pho.jpg")
        return
        
    img_bgr = cv2.imread(str(sample_img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. HOG
    _, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True
    )
    
    # 2. LBP
    lbp_map = local_binary_pattern(gray, P=24, R=3, method="uniform")
    
    # 3. HSV Histogram
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Ảnh gốc
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Ảnh đầu vào đã chuẩn hóa (224x224)", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Bản đồ HOG
    axes[0, 1].imshow(hog_image, cmap='inferno')
    axes[0, 1].set_title("Bản đồ biên dạng & hướng dốc HOG (Shape & Edges)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bản đồ LBP
    axes[1, 0].imshow(lbp_map, cmap='magma')
    axes[1, 0].set_title("Bản đồ kết cấu bề mặt LBP (Texture Patterns)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Biểu đồ HSV
    axes[1, 1].plot(hist_h, color='red', label='Hue (Sắc độ)', lw=2)
    axes[1, 1].plot(hist_s, color='green', label='Saturation (Độ bão hòa)', lw=2)
    axes[1, 1].plot(hist_v, color='blue', label='Value (Độ sáng)', lw=2)
    axes[1, 1].set_title("Phân bố biểu đồ màu HSV (Color Histogram)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Bin index (0-31)")
    axes[1, 1].set_ylabel("Tần suất pixel")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle("Trực quan hóa các họ đặc trưng hình ảnh trích chọn", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_path = REPORT_FIGS / "03_feature_extraction_visuals.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ trích đặc trưng tại: {out_path}")


if __name__ == "__main__":
    visualize_feature_extraction()
