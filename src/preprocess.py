# -*- coding: utf-8 -*-
"""
src/preprocess.py - MODULE TIỀN XỬ LÝ DỮ LIỆU ẢNH MÓN ĂN MIỀN BẮC VIỆT NAM
Tuân thủ nghiêm ngặt quy trình KDD / CRISP-DM:
Làm sạch -> Chuẩn hóa tên/nhãn -> Chia tập Stratified (70/15/15) -> Biến đổi ảnh -> Data Augmentation (CHỈ TRÊN TRAIN)

Đặc biệt:
- Không xóa tự động hoàn toàn: Gắn cờ (flag) ra file flagged_images.csv để người dùng thẩm tra bằng mắt.
- Toàn bộ ngưỡng cấu hình đặt ở đầu file (CONFIG).
- Tránh rò rỉ dữ liệu (Data Leakage): Mọi biến đổi thống kê và sinh mẫu chỉ áp dụng trên Train set.
"""

import os
import sys
import glob
import json
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ExifTags
import imagehash
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Thiết lập logging ghi nhận chi tiết quy trình KDD
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent

# ==============================================================================
# 0. CẤU HÌNH NGƯỠNG KIỂM TRA (DỄ DÀNG HIỆU CHỈNH TRƯỚC KHI BẢO VỆ)
# ==============================================================================
CONFIG = {
    # Kích thước tối thiểu để không bị coi là ảnh rác/icon/thumbnail hỏng
    'MIN_WIDTH': 100,
    'MIN_HEIGHT': 100,
    
    # Kích thước chuẩn hóa đầu vào theo đề bài
    'TARGET_SIZE': (224, 224),
    
    # Ngưỡng phát hiện ảnh mờ (Phương sai của toán tử Laplacian)
    # Ảnh có var(Laplacian) < BLUR_THRESHOLD coi như bị rung tay, out-focus
    'BLUR_THRESHOLD': 80.0,
    
    # Ngưỡng phát hiện ánh sáng bất thường (Độ sáng trung bình kênh V hoặc Grayscale [0..255])
    'BRIGHTNESS_LOW': 35.0,    # Quá tối (underexposed)
    'BRIGHTNESS_HIGH': 225.0,  # Quá chói lóa (overexposed)
    
    # Ngưỡng phát hiện trùng lặp Perceptual Hash (Hamming distance)
    # Khoảng cách <= 4 được coi là ảnh gần như y hệt hoặc chỉ khác tỉ lệ nhẹ
    'PHASH_HAMMING_THRESHOLD': 4,
    
    # Phân vị khoảng cách Mahalanobis/Euclidean tới tâm lớp để nghi ngờ sai nhãn
    'OUTLIER_PERCENTILE': 95.0,
    
    # Tỉ lệ phân chia tập dữ liệu chuẩn mực
    'TRAIN_RATIO': 0.70,
    'VAL_RATIO': 0.15,
    'TEST_RATIO': 0.15,
    'RANDOM_STATE': 42,
    
    # Tham số xử lý tương phản thích ứng CLAHE
    'CLAHE_CLIP_LIMIT': 2.0,
    'CLAHE_TILE_GRID': (8, 8),
    
    # Mục tiêu số ảnh sau cân bằng Augmentation cho mỗi lớp trên tập train
    'TARGET_TRAIN_SAMPLES_PER_CLASS': 700
}


# ==============================================================================
# GIAI ĐOẠN 2.1: LÀM SẠCH VÀ PHÁT HIỆN BẤT THƯỜNG (CLEANING & AUDITING)
# ==============================================================================

def fix_image_exif_and_convert_rgb(image_path: Path) -> Optional[np.ndarray]:
    """
    Đọc ảnh, tự động xoay đúng chiều dựa theo thẻ EXIF Orientation,
    chuyển đổi mọi không gian màu (RGBA, CMYK, Grayscale, WebP, PNG) về chuẩn RGB 3 kênh.
    Trả về numpy array định dạng RGB (uint8), hoặc None nếu file hỏng.
    """
    try:
        with Image.open(image_path) as pil_img:
            # 1. Sửa hướng xoay EXIF nếu có
            try:
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception:
                pass  # Bỏ qua nếu metadata EXIF bị lỗi định dạng
            
            # 2. Xử lý các mode màu đặc thù về RGB
            if pil_img.mode in ('RGBA', 'LA'):
                # Tạo nền trắng khi ghép ảnh trong suốt
                background = Image.new('RGB', pil_img.size, (255, 255, 255))
                alpha = pil_img.split()[-1]
                background.paste(pil_img, mask=alpha)
                pil_img = background
            elif pil_img.mode == 'P':
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'CMYK':
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
                
            return np.array(pil_img)
    except Exception as e:
        logger.debug(f"Không thể đọc file {image_path}: {e}")
        return None


def calculate_md5(file_path: Path) -> str:
    """Tính mã băm MD5 phát hiện trùng lặp bit-for-bit chính xác."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def calculate_phash(rgb_img: np.ndarray) -> imagehash.ImageHash:
    """Tính perceptual hash (pHash) phát hiện trùng lặp thị giác."""
    pil_img = Image.fromarray(rgb_img)
    return imagehash.phash(pil_img)


def calculate_blur_variance(rgb_img: np.ndarray) -> float:
    """
    Đo độ sắc nét bằng phương sai của toán tử Laplacian trên ảnh xám.
    Công thức: var(∇² I). Giá trị càng thấp ảnh càng mờ.
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def calculate_brightness(rgb_img: np.ndarray) -> float:
    """Đo độ sáng trung bình của ảnh dựa trên kênh V (Value trong không gian HSV)."""
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 2]))


def audit_and_flag_dataset(raw_dir: Path, output_csv_path: Path) -> pd.DataFrame:
    """
    Quét toàn bộ thư mục dataset gốc, kiểm tra và GẮN CỜ (FLAG) ảnh nghi ngờ:
    - Ảnh rác/0-byte/lỗi giải mã
    - Ảnh kích thước nhỏ (<100x100)
    - Ảnh trùng MD5 (cùng lớp hoặc giữa các lớp khác nhau)
    - Ảnh trùng Perceptual Hash (tương đồng thị giác cao)
    - Ảnh quá mờ (blur out-of-focus)
    - Ảnh thiếu sáng trầm trọng hoặc cháy sáng
    
    Xuất danh sách ra file flagged_images.csv để con người thẩm định, KHÔNG xóa tự động.
    """
    logger.info(f"Bắt đầu kiểm tra và gắn cờ dữ liệu tại: {raw_dir}")
    
    flagged_records = []
    seen_md5: Dict[str, Tuple[Path, str]] = {}
    phash_database: List[Tuple[imagehash.ImageHash, Path, str]] = []
    
    # Thu thập tất cả các file ảnh
    all_image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG'):
        all_image_paths.extend(raw_dir.glob(f"**/{ext}"))
    all_image_paths = sorted(list(set(all_image_paths)))
    
    logger.info(f"Tổng số tệp ảnh phát hiện được: {len(all_image_paths)}")
    
    for img_path in tqdm(all_image_paths, desc="Kiểm tra chất lượng ảnh"):
        rel_path = img_path.relative_to(raw_dir)
        class_name = rel_path.parts[0] if len(rel_path.parts) > 1 else "Unknown"
        
        # 1. Kiểm tra kích thước file 0 byte
        try:
            file_size = img_path.stat().st_size
            if file_size == 0:
                flagged_records.append({
                    'file_path': str(img_path),
                    'class_name': class_name,
                    'reason': 'CORRUPT_ZERO_BYTE',
                    'detail': 'Dung lượng file 0 byte',
                    'severity': 'HIGH'
                })
                continue
        except Exception as e:
            flagged_records.append({
                'file_path': str(img_path),
                'class_name': class_name,
                'reason': 'CANNOT_ACCESS',
                'detail': str(e),
                'severity': 'HIGH'
            })
            continue
            
        # 2. Đọc ảnh và chuyển đổi RGB
        rgb_img = fix_image_exif_and_convert_rgb(img_path)
        if rgb_img is None:
            flagged_records.append({
                'file_path': str(img_path),
                'class_name': class_name,
                'reason': 'CORRUPT_DECODE_FAIL',
                'detail': 'Không thể giải mã dữ liệu ảnh (định dạng hỏng)',
                'severity': 'HIGH'
            })
            continue
            
        h, w = rgb_img.shape[:2]
        
        # 3. Kiểm tra độ phân giải quá thấp (< 100x100)
        if w < CONFIG['MIN_WIDTH'] or h < CONFIG['MIN_HEIGHT']:
            flagged_records.append({
                'file_path': str(img_path),
                'class_name': class_name,
                'reason': 'TOO_SMALL',
                'detail': f'Kích thước {w}x{h} < ngưỡng {CONFIG["MIN_WIDTH"]}x{CONFIG["MIN_HEIGHT"]}',
                'severity': 'MEDIUM'
            })
            
        # 4. Kiểm tra trùng lặp MD5
        md5_val = calculate_md5(img_path)
        if md5_val in seen_md5:
            orig_path, orig_class = seen_md5[md5_val]
            is_cross_class = (orig_class != class_name)
            flagged_records.append({
                'file_path': str(img_path),
                'class_name': class_name,
                'reason': 'DUPLICATE_MD5_CROSS_CLASS' if is_cross_class else 'DUPLICATE_MD5_SAME_CLASS',
                'detail': f'Trùng byte hoàn toàn với: {orig_path.name} (lớp: {orig_class})',
                'severity': 'HIGH' if is_cross_class else 'MEDIUM'
            })
        else:
            seen_md5[md5_val] = (img_path, class_name)
            
        # 5. Kiểm tra Perceptual Hash (pHash)
        cur_phash = calculate_phash(rgb_img)
        # So khớp với database đã duyệt (kiểm tra khoảng cách Hamming)
        for prev_hash, prev_path, prev_class in phash_database[-2000:]:  # Tối ưu hóa so sánh cục bộ
            diff = cur_phash - prev_hash
            if diff <= CONFIG['PHASH_HAMMING_THRESHOLD']:
                is_cross = (prev_class != class_name)
                flagged_records.append({
                    'file_path': str(img_path),
                    'class_name': class_name,
                    'reason': 'SIMILAR_PHASH_CROSS_CLASS' if is_cross else 'SIMILAR_PHASH_SAME_CLASS',
                    'detail': f'Khoảng cách thị giác pHash = {diff} với {prev_path.name} (lớp: {prev_class})',
                    'severity': 'HIGH' if is_cross else 'LOW'
                })
                break
        phash_database.append((cur_phash, img_path, class_name))
        
        # 6. Kiểm tra độ mờ (Blurry)
        blur_var = calculate_blur_variance(rgb_img)
        if blur_var < CONFIG['BLUR_THRESHOLD']:
            flagged_records.append({
                'file_path': str(img_path),
                'class_name': class_name,
                'reason': 'BLURRY_IMAGE',
                'detail': f'Phương sai Laplacian = {blur_var:.1f} < ngưỡng {CONFIG["BLUR_THRESHOLD"]}',
                'severity': 'LOW'
            })
            
        # 7. Kiểm tra ánh sáng bất thường
        brightness = calculate_brightness(rgb_img)
        if brightness < CONFIG['BRIGHTNESS_LOW']:
            flagged_records.append({
                'file_path': str(img_path),
                'class_name': class_name,
                'reason': 'TOO_DARK',
                'detail': f'Độ sáng trung bình V = {brightness:.1f} < {CONFIG["BRIGHTNESS_LOW"]}',
                'severity': 'LOW'
            })
        elif brightness > CONFIG['BRIGHTNESS_HIGH']:
            flagged_records.append({
                'file_path': str(img_path),
                'class_name': class_name,
                'reason': 'TOO_BRIGHT',
                'detail': f'Độ sáng trung bình V = {brightness:.1f} > {CONFIG["BRIGHTNESS_HIGH"]}',
                'severity': 'LOW'
            })

    df_flagged = pd.DataFrame(flagged_records)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_flagged.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Đã hoàn thành kiểm toán! Tổng số cờ phát hiện: {len(df_flagged)}")
    logger.info(f"Báo cáo cờ được lưu tại: {output_csv_path}")
    
    # In thống kê lý do gắn cờ
    if not df_flagged.empty:
        summary = df_flagged.groupby(['reason', 'severity']).size().reset_index(name='count')
        logger.info(f"\nBảng tổng hợp lý do gắn cờ:\n{summary.to_string(index=False)}")
        
    return df_flagged


# ==============================================================================
# GIAI ĐOẠN 2.2: CHUẨN HÓA TÊN FILE, TẠO METADATA VÀ LABEL MAP
# ==============================================================================

def clean_and_standardize_dataset(
    raw_dir: Path, 
    cleaned_dir: Path, 
    flagged_csv_path: Optional[Path] = None,
    exclude_severities: List[str] = ['HIGH']
) -> pd.DataFrame:
    """
    Di chuyển ảnh hợp lệ sang thư mục data/cleaned/, chuẩn hóa tên file theo quy cách:
    <tên_lớp_chuẩn_hóa>_<mã_số_6_chữ_số>.jpg
    Loại bỏ các ảnh có cờ nghiêm trọng (như file hỏng, trùng chéo lớp) theo cấu hình.
    Tạo metadata.csv và label_map.json.
    """
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # Đọc danh sách ảnh cần loại trừ nếu có
    excluded_files = set()
    if flagged_csv_path and flagged_csv_path.exists():
        df_flags = pd.read_csv(flagged_csv_path)
        # Chỉ loại các ảnh có mức độ rủi ro cao (HIGH: hỏng, 0 byte, trùng khác lớp)
        bad_flags = df_flags[df_flags['severity'].isin(exclude_severities)]
        excluded_files = set(bad_flags['file_path'].values)
        logger.info(f"Loại bỏ {len(excluded_files)} ảnh có mức độ vi phạm {exclude_severities}")

    metadata_list = []
    class_folders = [d for d in raw_dir.iterdir() if d.is_dir()]
    
    for class_folder in sorted(class_folders):
        raw_class_name = class_folder.name
        # Chuẩn hóa tên lớp: viết liền không dấu hoặc gạch dưới chuẩn hóa
        norm_class_name = raw_class_name.lower().replace(" ", "_")
        target_class_dir = cleaned_dir / norm_class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG'):
            images.extend(class_folder.glob(ext))
        images = sorted(list(set(images)))
        
        counter = 1
        for img_path in images:
            if str(img_path) in excluded_files:
                continue
                
            rgb_img = fix_image_exif_and_convert_rgb(img_path)
            if rgb_img is None:
                continue
                
            h, w = rgb_img.shape[:2]
            new_filename = f"{norm_class_name}_{counter:06d}.jpg"
            dest_path = target_class_dir / new_filename
            
            # Lưu ảnh chuẩn RGB dưới dạng JPEG chất lượng cao 95
            pil_save = Image.fromarray(rgb_img)
            pil_save.save(dest_path, "JPEG", quality=95)
            
            md5_hash = calculate_md5(dest_path)
            metadata_list.append({
                'clean_path': str(dest_path),
                'original_path': str(img_path),
                'class_raw': raw_class_name,
                'class_name': norm_class_name,
                'width': w,
                'height': h,
                'channels': 3,
                'md5': md5_hash
            })
            counter += 1

    df_meta = pd.DataFrame(metadata_list)
    metadata_csv = cleaned_dir.parent / "metadata.csv"
    df_meta.to_csv(metadata_csv, index=False, encoding='utf-8-sig')
    logger.info(f"Đã lưu metadata chuẩn hóa với {len(df_meta)} ảnh tại {metadata_csv}")

    # Tạo label_map.json
    classes = sorted(df_meta['class_name'].unique().tolist())
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    
    label_map = {
        'class_to_idx': {cls: int(idx) for idx, cls in enumerate(label_encoder.classes_)},
        'idx_to_class': {int(idx): cls for idx, cls in enumerate(label_encoder.classes_)},
        'raw_to_norm': {row['class_raw']: row['class_name'] for _, row in df_meta.drop_duplicates('class_raw').iterrows()}
    }
    
    label_map_path = cleaned_dir.parent / "splits" / "label_map.json"
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu Label Map cho {len(classes)} lớp tại: {label_map_path}")
    
    return df_meta


# ==============================================================================
# GIAI ĐOẠN 2.3: CHIA DỮ LIỆU STRATIFIED 70 / 15 / 15 (TRÁNH DATA LEAKAGE)
# ==============================================================================

def create_stratified_splits(
    metadata_csv: Path, 
    splits_dir: Path, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Phân chia dữ liệu theo tỉ lệ 70% Train, 15% Validation, 15% Test có phân tầng (Stratified).
    Đảm bảo:
    - Không có ảnh trùng xuất hiện ở cả 2 tập khác nhau (đã được lọc MD5).
    - Phân bố tỉ lệ các lớp ở 3 tập đồng nhất.
    - Phân chia được thực hiện TRƯỚC mọi bước trích đặc trưng hay Augmentation.
    """
    logger.info("Thực hiện phân chia tập dữ liệu Stratified (70/15/15)...")
    df = pd.read_csv(metadata_csv)
    
    # Bước 1: Tách 70% Train và 30% Tạm thời (Val + Test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(CONFIG['VAL_RATIO'] + CONFIG['TEST_RATIO']), 
        stratify=df['class_name'], 
        random_state=random_state
    )
    
    # Bước 2: Tách đều 30% Tạm thời thành 15% Val và 15% Test
    val_ratio_in_temp = CONFIG['VAL_RATIO'] / (CONFIG['VAL_RATIO'] + CONFIG['TEST_RATIO'])
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1.0 - val_ratio_in_temp), 
        stratify=temp_df['class_name'], 
        random_state=random_state
    )
    
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    
    train_df.to_csv(train_csv, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_csv, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_csv, index=False, encoding='utf-8-sig')
    
    # Thẩm tra tính toàn vẹn (Checklist không Data Leakage)
    train_md5 = set(train_df['md5'])
    val_md5 = set(val_df['md5'])
    test_md5 = set(test_df['md5'])
    
    overlap_train_val = train_md5.intersection(val_md5)
    overlap_train_test = train_md5.intersection(test_md5)
    overlap_val_test = val_md5.intersection(test_md5)
    
    assert len(overlap_train_val) == 0, "CẢNH BÁO: Rò rỉ dữ liệu giữa Train và Val!"
    assert len(overlap_train_test) == 0, "CẢNH BÁO: Rò rỉ dữ liệu giữa Train và Test!"
    assert len(overlap_val_test) == 0, "CẢNH BÁO: Rò rỉ dữ liệu giữa Val và Test!"
    
    logger.info(f"Phân chia thành công! Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info("Checklist Data Leakage: 100% PASS (Không có giao thoa trùng lặp MD5)")
    
    return train_df, val_df, test_df


# ==============================================================================
# GIAI ĐOẠN 2.4: BIẾN ĐỔI ẢNH (RESIZE, CLAHE, WHITE BALANCE, DENOISE, NORMALIZE)
# ==============================================================================

def resize_letterbox(
    image: np.ndarray, 
    target_size: Tuple[int, int] = (224, 224), 
    pad_color: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """
    Resize ảnh giữ nguyên tỉ lệ khung hình (Aspect Ratio),
    thêm viền đệm (padding letterbox) màu xám để tránh làm méo dạng món ăn.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized
    return canvas


def apply_clahe_lab(image_rgb: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Áp dụng thuật toán cân bằng biểu đồ tần suất thích ứng giới hạn độ tương phản (CLAHE)
    riêng trên kênh L (Độ chói) của không gian màu CIE LAB.
    Giúp làm rõ chi tiết sợi phở, lát thịt, cọng rau mà KHÔNG làm đổi tông màu gốc.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)
    
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def apply_gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    """
    Cân bằng trắng theo giả định Gray World:
    Khử ám vàng (đèn nhà hàng) hoặc ám xanh (ngoài trời), đưa trung bình các kênh R,G,B về cân bằng.
    """
    img_float = image_rgb.astype(np.float32)
    mean_r = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_b = np.mean(img_float[:, :, 2])
    
    gray_mean = (mean_r + mean_g + mean_b) / 3.0
    
    # Tránh chia cho 0
    scale_r = gray_mean / (mean_r + 1e-5)
    scale_g = gray_mean / (mean_g + 1e-5)
    scale_b = gray_mean / (mean_b + 1e-5)
    
    balanced = np.zeros_like(img_float)
    balanced[:, :, 0] = np.clip(img_float[:, :, 0] * scale_r, 0, 255)
    balanced[:, :, 1] = np.clip(img_float[:, :, 1] * scale_g, 0, 255)
    balanced[:, :, 2] = np.clip(img_float[:, :, 2] * scale_b, 0, 255)
    
    return balanced.astype(np.uint8)


def apply_mild_denoising(image_rgb: np.ndarray) -> np.ndarray:
    """
    Khử nhiễu nhẹ bằng bộ lọc song phương (Bilateral Filter),
    làm mượt hạt nhiễu ISO nhưng giữ sắc nét tuyệt đối các đường biên (edges).
    """
    return cv2.bilateralFilter(image_rgb, d=5, sigmaColor=35, sigmaSpace=35)


def transform_single_image(image_rgb: np.ndarray, keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Chuỗi biến đổi hình ảnh toàn diện (Full Transformation Pipeline):
    1. Cân bằng trắng Gray World
    2. Khử nhiễu nhẹ Bilateral
    3. Tăng cường cục bộ CLAHE LAB
    4. Resize về kích thước chuẩn 224x224
    """
    # 1. Cân bằng trắng
    img = apply_gray_world_white_balance(image_rgb)
    # 2. Khử nhiễu
    img = apply_mild_denoising(img)
    # 3. CLAHE
    img = apply_clahe_lab(img, clip_limit=CONFIG['CLAHE_CLIP_LIMIT'], grid_size=CONFIG['CLAHE_TILE_GRID'])
    # 4. Resize
    if keep_aspect_ratio:
        img = resize_letterbox(img, target_size=CONFIG['TARGET_SIZE'])
    else:
        img = cv2.resize(img, CONFIG['TARGET_SIZE'], interpolation=cv2.INTER_AREA)
    return img


# ==============================================================================
# GIAI ĐOẠN 2.5: CÂN BẰNG DỮ LIỆU BẰNG DATA AUGMENTATION (CHỈ TRÊN TẬP TRAIN)
# ==============================================================================

def augment_image_random(image_rgb: np.ndarray) -> np.ndarray:
    """
    Sinh biến thể hình ảnh mô phỏng góc chụp thực tế của thực khách:
    - Lật gương ngang (Horizontal Flip)
    - Xoay ngẫu nhiên góc nhẹ [-15, 15] độ
    - Thay đổi tỉ lệ zoom nhẹ [0.9, 1.1]
    - Biến thiên độ sáng/tương phản nhẹ
    """
    img = image_rgb.copy()
    h, w = img.shape[:2]
    
    # 1. Lật ngang ngẫu nhiên
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        
    # 2. Xoay góc ngẫu nhiên
    angle = np.random.uniform(-15.0, 15.0)
    scale = np.random.uniform(0.95, 1.05)
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    img = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 3. Thay đổi ánh sáng nhẹ
    bright_factor = np.random.uniform(0.85, 1.15)
    img = np.clip(img.astype(np.float32) * bright_factor, 0, 255).astype(np.uint8)
    
    return img


def process_and_augment_dataset(
    splits_dir: Path, 
    processed_dir: Path,
    target_train_count: int = CONFIG['TARGET_TRAIN_SAMPLES_PER_CLASS']
) -> Dict[str, Path]:
    """
    Thực hiện biến đổi ảnh cho cả 3 tập, và thực hiện Data Augmentation
    ĐỘC QUYỀN trên tập Train cho các lớp thiểu số để triệt tiêu hiện tượng mất cân bằng mẫu.
    Lưu kết quả các tập đã biến đổi vào data/processed/.
    """
    logger.info("Bắt đầu xử lý ảnh và sinh dữ liệu tăng cường (Augmentation) cho tập Train...")
    
    split_files = {
        'train': splits_dir / "train.csv",
        'val': splits_dir / "val.csv",
        'test': splits_dir / "test.csv"
    }
    
    processed_csv_paths = {}
    
    for split_name, csv_path in split_files.items():
        df_split = pd.read_csv(csv_path)
        dest_split_dir = processed_dir / split_name
        dest_split_dir.mkdir(parents=True, exist_ok=True)
        
        processed_records = []
        is_train = (split_name == 'train')
        
        # Nhóm theo lớp
        grouped = df_split.groupby('class_name')
        
        for class_name, group in grouped:
            class_out_dir = dest_split_dir / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)
            
            orig_images = []
            # 1. Xử lý ảnh gốc
            for idx, row in group.iterrows():
                src_path = Path(row['clean_path'])
                if not src_path.exists():
                    continue
                img_rgb = fix_image_exif_and_convert_rgb(src_path)
                if img_rgb is None:
                    continue
                
                # Biến đổi tiêu chuẩn
                transformed = transform_single_image(img_rgb)
                out_path = class_out_dir / src_path.name
                Image.fromarray(transformed).save(out_path, "JPEG", quality=95)
                
                orig_images.append(transformed)
                processed_records.append({
                    'image_path': str(out_path),
                    'class_name': class_name,
                    'is_augmented': False
                })
                
            # 2. Sinh Augmentation nếu là tập Train và số ảnh < target
            if is_train and len(orig_images) > 0 and len(orig_images) < target_train_count:
                needed = target_train_count - len(orig_images)
                logger.debug(f"Lớp {class_name}: Đang sinh thêm {needed} ảnh Augmentation...")
                
                for aug_idx in range(needed):
                    base_img = orig_images[aug_idx % len(orig_images)]
                    aug_img = augment_image_random(base_img)
                    
                    aug_filename = f"{class_name}_aug_{aug_idx+1:05d}.jpg"
                    aug_path = class_out_dir / aug_filename
                    Image.fromarray(aug_img).save(aug_path, "JPEG", quality=95)
                    
                    processed_records.append({
                        'image_path': str(aug_path),
                        'class_name': class_name,
                        'is_augmented': True
                    })

        df_out = pd.DataFrame(processed_records)
        out_csv = processed_dir / f"{split_name}_processed.csv"
        df_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
        processed_csv_paths[split_name] = out_csv
        logger.info(f"Hoàn thành xử lý tập {split_name.upper()}: Tổng số {len(df_out)} ảnh lưu tại {out_csv}")
        
    return processed_csv_paths


# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH TOÀN BỘ GIAI ĐOẠN 2 (PIPELINE EXECUTION)
# ==============================================================================

def run_full_preprocessing_pipeline(
    raw_dataset_dir: str = BASE_DIR / "dataset" / "Images",
    base_data_dir: str = BASE_DIR / "data"
):
    """Hàm chạy toàn bộ các bước của Giai đoạn 2 một cách tự động và nhất quán."""
    raw_path = Path(raw_dataset_dir)
    base_path = Path(base_data_dir)
    
    cleaned_dir = base_path / "cleaned"
    splits_dir = base_path / "splits"
    processed_dir = base_path / "processed"
    flagged_csv = base_path / "flagged_images.csv"
    metadata_csv = base_path / "metadata.csv"
    
    print("\n" + "="*80)
    print(" BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU ĐỒ ÁN KHAI PHÁ DỮ LIỆU ")
    print("="*80 + "\n")
    
    # Bước 2.1: Quét và gắn cờ ảnh
    audit_and_flag_dataset(raw_path, flagged_csv)
    
    # Bước 2.2: Làm sạch có chọn lọc & Chuẩn hóa tên/nhãn
    clean_and_standardize_dataset(raw_path, cleaned_dir, flagged_csv)
    
    # Bước 2.3: Phân tầng Stratified 70/15/15
    create_stratified_splits(metadata_csv, splits_dir)
    
    # Bước 2.4 & 2.5: Biến đổi ảnh và Augmentation độc quyền trên Train
    process_and_augment_dataset(splits_dir, processed_dir)
    
    print("\n" + "="*80)
    print(" HOÀN THÀNH TIỀN XỬ LÝ TOÀN DIỆN! DỮ LIỆU SẴN SÀNG CHO BƯỚC TRÍCH ĐẶC TRƯNG.")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_full_preprocessing_pipeline()
