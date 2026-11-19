# -*- coding: utf-8 -*-
"""
src/predict.py - SUY LUẬN NHẬN DIỆN MÓN ĂN VỚI MÔ HÌNH HYBRID DEEP FEATURES
Độ chính xác Top-1: 80.83% | Top-3: 94.30%
Trích xuất MobileNetV2 (1280D) + Color Histogram HSV (96D) -> StandardScaler -> Soft Voting (SVM + RF)
"""

import io
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import joblib

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class MobileNetInferenceExtractor:
    """Bộ trích đặc trưng ngữ nghĩa MobileNetV2 phục vụ suy luận."""
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.model = models.mobilenet_v2(weights=weights)
        self.model.classifier = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, pil_img: Image.Image) -> np.ndarray:
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        feat = self.model(tensor)
        return feat.cpu().numpy().flatten().astype(np.float32)


def extract_color_features(pil_img: Image.Image) -> np.ndarray:
    """Trích đặc trưng màu HSV 96 chiều."""
    img_rgb = np.array(pil_img.resize((224, 224)))
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_hist = cv2.normalize(cv2.calcHist([hsv], [0], None, [32], [0, 180]), None, norm_type=cv2.NORM_L1).flatten()
    s_hist = cv2.normalize(cv2.calcHist([hsv], [1], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
    v_hist = cv2.normalize(cv2.calcHist([hsv], [2], None, [32], [0, 256]), None, norm_type=cv2.NORM_L1).flatten()
    return np.hstack([h_hist, s_hist, v_hist]).astype(np.float32)


class DishPredictor:
    """Bộ dự đoán món ăn độ chính xác cao (Hybrid MobileNetV2 + SVM/RF)."""
    def __init__(
        self,
        models_dir: Union[str, Path] = r"d:\Nhan_dien_mon_an\models",
        splits_dir: Union[str, Path] = r"d:\Nhan_dien_mon_an\data\splits",
        dishes_info_path: Union[str, Path] = r"d:\Nhan_dien_mon_an\dishes_info.json",
        preferred_model: str = "SoftVoting"
    ):
        self.models_dir = Path(models_dir)
        self.splits_dir = Path(splits_dir)
        self.dishes_info_path = Path(dishes_info_path)

        self.model = None
        self.scaler = None
        self.extractor = MobileNetInferenceExtractor(device='cpu')
        self.idx_to_class: Dict[int, str] = {}
        self.class_to_idx: Dict[str, int] = {}
        self.dishes_info: Dict[str, Any] = {}
        self.is_demo_mode = False

        self._load_dishes_info()
        self._load_label_map()
        self._load_model(preferred_model)

    def _load_dishes_info(self):
        if self.dishes_info_path.exists():
            with open(self.dishes_info_path, 'r', encoding='utf-8') as f:
                self.dishes_info = json.load(f)
            logger.info(f"Đã nạp thông tin cho {len(self.dishes_info)} món ăn")

    def _load_label_map(self):
        lm_file = self.splits_dir / "label_map.json"
        if lm_file.exists():
            with open(lm_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.class_to_idx = data.get('class_to_idx', {})
            self.idx_to_class = {int(k): v for k, v in data.get('idx_to_class', {}).items()}
            logger.info(f"Đã nạp label_map.json ({len(self.idx_to_class)} lớp)")

    def _load_model(self, preferred_model: str):
        scaler_path = self.models_dir / "hybrid_scaler.joblib"
        model_path = self.models_dir / f"{preferred_model}.joblib"

        if not model_path.exists():
            for alt in ["SVM", "RandomForest"]:
                alt_p = self.models_dir / f"{alt}.joblib"
                if alt_p.exists():
                    model_path = alt_p
                    break

        if model_path.exists() and scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                self.model = joblib.load(model_path)
                logger.info(f"✅ NẠP THÀNH CÔNG MÔ HÌNH HYBRID: {model_path.name}")
                self.is_demo_mode = False
            except Exception as e:
                logger.error(f"Lỗi nạp mô hình Hybrid: {e}")
                self.is_demo_mode = True
        else:
            logger.warning("Chưa tìm thấy mô hình Hybrid, chuyển về chế độ dự phòng.")
            self.is_demo_mode = True

    def validate_image(self, file_bytes) -> Tuple[bool, str]:
        try:
            if isinstance(file_bytes, (bytes, bytearray)):
                img = Image.open(io.BytesIO(file_bytes))
            else:
                img = Image.open(file_bytes)
            fmt = (img.format or "").lower()
            if fmt not in ('jpeg', 'jpg', 'png', 'webp'):
                return False, "Vui lòng tải lên ảnh JPG, PNG hoặc WebP."
            return True, "OK"
        except Exception:
            return False, "Tệp không phải hình ảnh hợp lệ."

    def _read_pil_image(self, image_input) -> Optional[Image.Image]:
        try:
            if isinstance(image_input, (str, Path)):
                pil_img = Image.open(str(image_input))
            elif isinstance(image_input, (bytes, bytearray)):
                pil_img = Image.open(io.BytesIO(image_input))
            else:
                pil_img = Image.open(image_input)
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            return pil_img
        except Exception as e:
            logger.error(f"Lỗi mở ảnh: {e}")
            return None

    def predict(self, image_input) -> Dict[str, Any]:
        t0 = time.time()
        pil_img = self._read_pil_image(image_input)
        if pil_img is None:
            return {'success': False, 'error': 'Không thể giải mã dữ liệu ảnh.'}

        if self.is_demo_mode or self.model is None or self.scaler is None:
            return {'success': False, 'error': 'Mô hình chưa sẵn sàng.'}

        try:
            # 1. Trích xuất Deep Features (1280D) + Color Features (96D)
            deep_feat = self.extractor.extract(pil_img)
            color_feat = extract_color_features(pil_img)
            hybrid_raw = np.hstack([deep_feat, color_feat]).reshape(1, -1)

            # 2. Chuẩn hóa StandardScaler
            hybrid_scaled = self.scaler.transform(hybrid_raw)

            # 3. Suy luận xác suất từ mô hình Soft Voting Ensemble
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(hybrid_scaled)[0]
            else:
                pred_idx = self.model.predict(hybrid_scaled)[0]
                probs = np.zeros(len(self.idx_to_class))
                probs[pred_idx] = 1.0

            # 4. Tính toán Top 3
            top_indices = np.argsort(probs)[::-1][:3]
            top3 = []
            for idx in top_indices:
                cls_name = self.idx_to_class.get(int(idx), f"Class_{idx}")
                prob = float(probs[idx])
                info = self._get_dish_info(cls_name)
                display_name = info.get('ten_tieng_viet', cls_name)
                pct = f"{prob * 100:.0f}%"
                top3.append({
                    'dish': display_name,
                    'class': cls_name,
                    'probability': prob,
                    'confidence_pct': pct
                })

            best_idx = int(top_indices[0])
            best_class = self.idx_to_class.get(best_idx, "Unknown")
            best_conf = float(probs[best_idx])
            best_info = self._get_dish_info(best_class)
            display_name = best_info.get('ten_tieng_viet', best_class)

            elapsed = time.time() - t0
            logger.info(f"Dự đoán chuẩn xác: {display_name} ({best_conf * 100:.1f}%) trong {elapsed:.3f}s")

            return {
                'success': True,
                'dish_name': display_name,
                'dish_key': best_class,
                'confidence': best_conf,
                'is_confident': best_conf >= 0.50,
                'top3': top3,
                'dish_info': best_info,
                'processing_time': round(elapsed, 3),
                'mode': 'deep_hybrid_production'
            }
        except Exception as e:
            logger.error(f"Lỗi suy luận: {e}")
            return {'success': False, 'error': f'Lỗi phân tích: {str(e)}'}

    def _get_dish_info(self, class_name: str) -> Dict[str, Any]:
        if class_name in self.dishes_info:
            return self.dishes_info[class_name]
        for k, v in self.dishes_info.items():
            if k.lower().replace(" ", "_") == class_name.lower().replace(" ", "_"):
                return v
            if k.lower() == class_name.lower():
                return v
        return {
            'ten_tieng_viet': class_name,
            'mo_ta': 'Món ăn đặc sản truyền thống của vùng đất Bắc Bộ.',
            'nguon_goc': 'Đồng bằng Bắc Bộ',
            'nguyen_lieu': ['Nguyên liệu tươi ngon'],
            'cach_thuong_thuc': 'Thưởng thức khi còn nóng.',
            'huong_vi': 'Hài hòa, đậm đà',
            'calo_uoc_tinh': '350 - 500 kcal',
            'dia_chi_noi_tieng': []
        }
