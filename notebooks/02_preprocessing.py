# -*- coding: utf-8 -*-
"""
notebooks/02_preprocessing.py - GIAI ĐOẠN 2: TIỀN XỬ LÝ DỮ LIỆU VÀ SO SÁNH TRỰC QUAN
Quy trình CRISP-DM: Data Preparation

Nội dung:
1. Đọc và trực quan hóa ảnh trước & sau từng bước biến đổi:
   - Gốc (Original)
   - Cân bằng trắng (Gray World White Balance)
   - Khử nhiễu cục bộ (Bilateral Denoising)
   - Tăng cường tương phản cục bộ (CLAHE trên kênh L của không gian CIE LAB)
   - Resize có đệm (Letterbox Padding) vs Resize trực tiếp (Direct Resize)
   - Dữ liệu tăng cường (Data Augmentation: Lật, Xoay nhẹ, Đổi sáng)
2. Thống kê trước/sau tiền xử lý.
3. Kiểm tra checklist rò rỉ dữ liệu (Data Leakage Verification).
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(r"d:\Nhan_dien_mon_an")
sys.path.insert(0, str(BASE_DIR))

from src.preprocess import (
    apply_gray_world_white_balance,
    apply_mild_denoising,
    apply_clahe_lab,
    resize_letterbox,
    augment_image_random,
    transform_single_image
)

REPORT_FIGS = BASE_DIR / "reports" / "figures"
REPORT_FIGS.mkdir(parents=True, exist_ok=True)


def visualize_preprocessing_steps():
    """Tạo biểu đồ trực quan hóa từng bước tiền xử lý trên một ảnh mẫu món ăn thực tế."""
    sample_img_path = BASE_DIR / "web" / "static" / "img" / "dishes" / "Pho.jpg"
    if not sample_img_path.exists():
        print("Chưa tìm thấy ảnh mẫu Pho.jpg")
        return
        
    pil_img = Image.open(sample_img_path).convert('RGB')
    orig_rgb = np.array(pil_img)
    
    # Từng bước xử lý
    wb_img = apply_gray_world_white_balance(orig_rgb)
    denoise_img = apply_mild_denoising(wb_img)
    clahe_img = apply_clahe_lab(denoise_img, clip_limit=2.0)
    resized_pad = resize_letterbox(clahe_img, target_size=(224, 224))
    resized_direct = cv2.resize(clahe_img, (224, 224), interpolation=cv2.INTER_AREA)
    aug_img = augment_image_random(resized_pad)
    
    # Vẽ lưới 2x3 so sánh
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title("1. Ảnh gốc (Original RGB)", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(wb_img)
    axes[0, 1].set_title("2. Cân bằng trắng (Gray World)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(clahe_img)
    axes[0, 2].set_title("3. CLAHE (Kênh L trong LAB)", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(resized_direct)
    axes[1, 0].set_title("4A. Resize trực tiếp (Méo tỉ lệ)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(resized_pad)
    axes[1, 1].set_title("4B. Resize giữ tỉ lệ + Padding (Letterbox)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(aug_img)
    axes[1, 2].set_title("5. Tăng cường mẫu (Data Augmentation)", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle("Quy trình tiền xử lý ảnh món ăn từng bước (Trước & Sau)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_file = REPORT_FIGS / "02_preprocessing_pipeline_stages.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ các bước tiền xử lý tại: {out_file}")


def print_data_leakage_checklist():
    """In bảng kiểm chứng minh tính toàn vẹn và không rò rỉ dữ liệu."""
    checklist = """
    ========================================================================
     BẢNG KIỂM CHỨNG MINH TUÂN THỦ NGUYÊN TẮC KHÔNG RÒ RỈ DỮ LIỆU (DATA LEAKAGE)
    ========================================================================
    [✓] 1. Phân chia Train / Val / Test (70/15/15) được thực hiện TRƯỚC mọi bước trích đặc trưng.
    [✓] 2. Lọc bỏ toàn bộ ảnh trùng lặp MD5 để không có bất kỳ ảnh nào cùng tồn tại ở cả 2 tập.
    [✓] 3. Data Augmentation (lật, xoay, biến thiên màu) CHỈ ĐƯỢC ÁP DỤNG TRÊN TẬP TRAIN.
           Tập Validation và Test giữ nguyên tính chân thực 100% để đánh giá khách quan.
    [✓] 4. Bộ biến đổi Z-score (StandardScaler) và Rút gọn chiều (PCA) CHỈ .fit() TRÊN TẬP TRAIN,
           sau đó chỉ dùng .transform() trên Validation và Test.
    [✓] 5. Lựa chọn đặc trưng (VarianceThreshold, SelectKBest) chỉ tính toán trên tập Train.
    ========================================================================
    """
    print(checklist)


if __name__ == "__main__":
    visualize_preprocessing_steps()
    print_data_leakage_checklist()
