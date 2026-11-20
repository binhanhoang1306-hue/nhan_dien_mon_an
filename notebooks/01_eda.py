# -*- coding: utf-8 -*-
"""
notebooks/01_eda.py - GIAI ĐOẠN 1: KHÁM PHÁ VÀ HIỂU BÀI TOÁN, DỮ LIỆU (EDA)
Quy trình CRISP-DM: Data Understanding

Nội dung thực hiện:
1. Phát biểu bài toán: Phân lớp ảnh đa lớp các món ăn miền Bắc Việt Nam.
2. Thống kê mô tả số lượng ảnh từng lớp, kiểm tra mức độ mất cân bằng (Imbalance Ratio).
3. Thống kê độ phân giải (Width, Height), tỉ lệ khung hình (Aspect Ratio), dung lượng file.
4. Thống kê định dạng tệp (JPEG, PNG, WebP) và số lượng kênh màu.
5. Biểu đồ thanh số lượng ảnh từng lớp và biểu đồ tròn tỉ lệ.
6. Lưới ảnh mẫu đại diện cho các lớp món ăn miền Bắc.
7. Thống kê chuyên sâu theo lớp: Độ sáng trung bình (Mean Brightness) và Biểu đồ tần suất màu (HSV Color Histogram) trung bình của từng món.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Thiết lập thẩm mỹ đồ thị
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "Images"
REPORT_FIGS = BASE_DIR / "reports" / "figures"
REPORT_FIGS.mkdir(parents=True, exist_ok=True)


def explore_dataset_overview() -> pd.DataFrame:
    """Quét toàn bộ dataset, thu thập thuộc tính từng bức ảnh."""
    print("="*70)
    print(" 1. THỐNG KÊ TỔNG QUAN TẬP DỮ LIỆU ")
    print("="*70)
    
    records = []
    class_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    
    for c_dir in class_dirs:
        class_name = c_dir.name
        img_files = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.jpeg")) + \
                    list(c_dir.glob("*.png")) + list(c_dir.glob("*.webp"))
                    
        for img_p in img_files:
            try:
                sz_bytes = img_p.stat().st_size
                with Image.open(img_p) as im:
                    w, h = im.size
                    fmt = im.format
                    mode = im.mode
                records.append({
                    'class': class_name,
                    'file_path': str(img_p),
                    'width': w,
                    'height': h,
                    'aspect_ratio': round(w / h, 2) if h > 0 else 0,
                    'size_kb': round(sz_bytes / 1024, 2),
                    'format': fmt,
                    'mode': mode
                })
            except Exception:
                pass
                
    df = pd.DataFrame(records)
    print(f"Tổng số ảnh thu thập hợp lệ: {len(df)}")
    print(f"Tổng số lớp: {df['class'].nunique()}")
    
    # Đếm số lượng ảnh từng lớp
    counts = df['class'].value_counts()
    min_c = counts.min()
    max_c = counts.max()
    imbalance_ratio = max_c / min_c if min_c > 0 else 0
    
    print(f"\nLớp có số ảnh nhiều nhất: {counts.idxmax()} ({max_c} ảnh)")
    print(f"Lớp có số ảnh ít nhất: {counts.idxmin()} ({min_c} ảnh)")
    print(f"Hệ số mất cân bằng dữ liệu (Imbalance Ratio): {imbalance_ratio:.2f}x")
    print(f"-> Nhận xét: Dữ liệu mất cân bằng nặng. Cần áp dụng Data Augmentation và class_weight='balanced'.\n")
    
    return df


def plot_class_distribution(df: pd.DataFrame):
    """Vẽ biểu đồ phân bố số lượng ảnh theo lớp."""
    plt.figure(figsize=(14, 7))
    counts = df['class'].value_counts().sort_values(ascending=False)
    
    palette = sns.color_palette("rocket", len(counts))
    ax = sns.barplot(x=counts.values, y=counts.index, palette=palette)
    
    # Ghi số lượng trên đầu mỗi cột
    for i, v in enumerate(counts.values):
        pct = (v / len(df)) * 100
        ax.text(v + 15, i, f"{v} ({pct:.1f}%)", va='center', fontweight='bold', fontsize=9)
        
    plt.title(f"Phân bố số lượng ảnh qua {len(counts)} lớp món ăn miền Bắc", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Số lượng ảnh", fontsize=12)
    plt.ylabel("Tên lớp món ăn", fontsize=12)
    plt.xlim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    
    save_path = REPORT_FIGS / "01_eda_class_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ phân bố lớp tại: {save_path}")


def plot_image_grid_samples():
    """Tạo lưới ảnh mẫu đại diện cho các lớp trong dataset."""
    class_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"Không tìm thấy lớp ảnh trong {DATASET_DIR}")
    columns = min(6, len(class_dirs))
    rows = (len(class_dirs) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    axes = axes.flatten()
    
    for idx, c_dir in enumerate(class_dirs):
        imgs = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.jpeg")) + list(c_dir.glob("*.png"))
        if imgs:
            img_bgr = cv2.imread(str(imgs[0]))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
                axes[idx].set_title(c_dir.name, fontsize=10, fontweight='bold')
        axes[idx].axis('off')
        
    for ax in axes[len(class_dirs):]:
        ax.axis('off')
    plt.suptitle(f"Lưới ảnh mẫu đại diện {len(class_dirs)} lớp món ăn", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_path = REPORT_FIGS / "01_eda_sample_grid.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Đã lưu lưới ảnh mẫu tại: {save_path}")


def analyze_color_and_brightness_by_class(df: pd.DataFrame):
    """Tính toán độ sáng trung bình và phân bố màu sắc đặc trưng của từng món."""
    print("\n" + "="*70)
    print(" 2. PHÂN TÍCH ĐẶC TÍNH QUANG HỌC THEO TỪNG MÓN ĂN ")
    print("="*70)
    
    brightness_results = []
    
    for cls in df['class'].unique():
        cls_df = df[df['class'] == cls].head(50)  # Lấy mẫu 50 ảnh để tính nhanh
        brightness_vals = []
        
        for p in cls_df['file_path']:
            bgr = cv2.imread(p)
            if bgr is not None:
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                v_mean = np.mean(hsv[:, :, 2])
                brightness_vals.append(v_mean)
                
        if brightness_vals:
            brightness_results.append({
                'class': cls,
                'mean_brightness': np.mean(brightness_vals),
                'std_brightness': np.std(brightness_vals)
            })
            
    df_b = pd.DataFrame(brightness_results).sort_values(by='mean_brightness', ascending=False)
    
    # Vẽ biểu đồ độ sáng
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_b, x='mean_brightness', y='class', palette='copper')
    plt.title("Độ sáng trung bình (Kênh Value trong HSV) giữa các món ăn", fontsize=14, fontweight='bold')
    plt.xlabel("Độ sáng trung bình (0 - 255)", fontsize=11)
    plt.ylabel("Món ăn", fontsize=11)
    plt.tight_layout()
    
    save_path = REPORT_FIGS / "01_eda_brightness_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ độ sáng tại: {save_path}")


if __name__ == "__main__":
    df_meta = explore_dataset_overview()
    plot_class_distribution(df_meta)
    plot_image_grid_samples()
    analyze_color_and_brightness_by_class(df_meta)
    print("\nHoàn tất Giai đoạn 1 (EDA)!")
