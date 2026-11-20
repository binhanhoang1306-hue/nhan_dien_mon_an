# -*- coding: utf-8 -*-
"""Inference for the production MobileNetV2 + HSV food classifier."""

import io
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
from PIL import Image, ImageOps

from src.production_features import (
    FEATURE_CONFIG,
    MobileNetV2FeatureExtractor,
    extract_hybrid_features,
    preprocess_image,
)

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent


class DishPredictor:
    """Loads the trained feature pipeline and selected classifier for inference."""

    def __init__(
        self,
        models_dir: Union[str, Path] = BASE_DIR / "models",
        splits_dir: Union[str, Path] = BASE_DIR / "data" / "splits",
        dishes_info_path: Union[str, Path] = BASE_DIR / "dishes_info.json",
        preferred_model: str = "SoftVoting",
    ):
        self.models_dir = Path(models_dir)
        self.splits_dir = Path(splits_dir)
        self.dishes_info_path = Path(dishes_info_path)
        self.model = None
        self.feature_pipeline = None
        self.feature_extractor = None
        self.idx_to_class: Dict[int, str] = {}
        self.dishes_info: Dict[str, Any] = {}
        self.is_demo_mode = True

        self._load_dishes_info()
        self._load_label_map()
        self._load_model(preferred_model)

    def _load_dishes_info(self):
        if self.dishes_info_path.exists():
            with self.dishes_info_path.open("r", encoding="utf-8") as file:
                self.dishes_info = json.load(file)

    def _load_label_map(self):
        label_map_file = self.splits_dir / "label_map.json"
        if not label_map_file.exists():
            logger.warning("Chưa tìm thấy label map: %s", label_map_file)
            return

        with label_map_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        self.idx_to_class = {
            int(index): name for index, name in data.get("idx_to_class", {}).items()
        }

    def _load_model(self, preferred_model: str):
        pipeline_path = self.models_dir / "feature_pipeline.joblib"
        feature_config_path = self.models_dir / "feature_config.json"
        feature_weights_path = self.models_dir / "mobilenet_v2_features.pt"
        model_path = self.models_dir / f"{preferred_model}.joblib"
        if not model_path.exists():
            model_path = self.models_dir / "SVM.joblib"

        if (
            not self.idx_to_class
            or not pipeline_path.is_file()
            or not model_path.is_file()
            or not feature_config_path.is_file()
            or not feature_weights_path.is_file()
        ):
            logger.warning(
                "Mô hình chưa sẵn sàng; cần feature pipeline, cấu hình MobileNetV2 + HSV, "
                "trọng số trích đặc trưng, %s và label_map.json.",
                model_path.name,
            )
            return

        try:
            with feature_config_path.open("r", encoding="utf-8") as file:
                feature_config = json.load(file)
            if feature_config != FEATURE_CONFIG:
                logger.error(
                    "Cấu hình đặc trưng không tương thích: %s", feature_config_path
                )
                return
            self.feature_pipeline = joblib.load(pipeline_path)
            self.model = joblib.load(model_path)
            self.feature_extractor = MobileNetV2FeatureExtractor(
                weights_path=feature_weights_path
            )
        except (
            OSError,
            EOFError,
            ImportError,
            AttributeError,
            ValueError,
            RuntimeError,
            TypeError,
            pickle.UnpicklingError,
        ):
            logger.exception("Không thể nạp mô hình từ %s", self.models_dir)
            self.feature_pipeline = None
            self.model = None
            self.feature_extractor = None
            return

        self.is_demo_mode = False
        logger.info("Đã nạp pipeline và mô hình: %s", model_path.name)

    def validate_image(self, file_bytes) -> Tuple[bool, str]:
        try:
            with Image.open(io.BytesIO(file_bytes)) as image:
                if (image.format or "").lower() not in ("jpeg", "jpg", "png", "webp"):
                    return False, "Vui lòng tải lên ảnh JPG, PNG hoặc WebP."
            return True, "OK"
        except (OSError, ValueError):
            return False, "Tệp không phải hình ảnh hợp lệ."

    def _read_pil_image(self, image_input) -> Optional[Image.Image]:
        try:
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input)
            elif isinstance(image_input, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image_input))
            else:
                image = Image.open(image_input)
            with image:
                return ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, ValueError):
            logger.exception("Không thể đọc ảnh đầu vào")
            return None

    def predict(self, image_input) -> Dict[str, Any]:
        started_at = time.time()
        pil_image = self._read_pil_image(image_input)
        if pil_image is None:
            return {"success": False, "error": "Không thể giải mã dữ liệu ảnh."}

        if (
            self.is_demo_mode
            or self.model is None
            or self.feature_pipeline is None
            or self.feature_extractor is None
        ):
            return {
                "success": False,
                "error": "Mô hình chưa sẵn sàng. Hãy huấn luyện mô hình trước khi nhận diện.",
            }

        try:
            enhanced_rgb = preprocess_image(pil_image)
            feature_vector = extract_hybrid_features(
                enhanced_rgb, self.feature_extractor
            ).reshape(1, -1)
            transformed = self.feature_pipeline.transform(feature_vector)

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(transformed)[0]
                class_indices = self.model.classes_
            else:
                prediction = int(self.model.predict(transformed)[0])
                class_indices = np.asarray([prediction])
                probabilities = np.asarray([1.0])

            ranked = np.argsort(probabilities)[::-1][:3]
            top3 = []
            for position in ranked:
                class_index = int(class_indices[position])
                class_name = self.idx_to_class.get(class_index, f"Class_{class_index}")
                probability = float(probabilities[position])
                dish_info = self._get_dish_info(class_name)
                top3.append({
                    "dish": dish_info.get("ten_tieng_viet", class_name),
                    "class": class_name,
                    "probability": probability,
                    "confidence_pct": f"{probability * 100:.0f}%",
                })

            best = top3[0]
            elapsed = time.time() - started_at
            return {
                "success": True,
                "dish_name": best["dish"],
                "dish_key": best["class"],
                "confidence": best["probability"],
                "is_confident": best["probability"] >= 0.50,
                "top3": top3,
                "dish_info": self._get_dish_info(best["class"]),
                "processing_time": round(elapsed, 3),
                "mode": "mobilenetv2_hsv",
            }
        except (OSError, ValueError, RuntimeError, AttributeError) as error:
            logger.exception("Lỗi suy luận ảnh")
            return {"success": False, "error": f"Lỗi phân tích: {error}"}

    def _get_dish_info(self, class_name: str) -> Dict[str, Any]:
        if class_name in self.dishes_info:
            return self.dishes_info[class_name]
        normalized_name = class_name.lower().replace(" ", "_")
        for key, info in self.dishes_info.items():
            if key.lower().replace(" ", "_") == normalized_name:
                return info
        return {"ten_tieng_viet": class_name}
