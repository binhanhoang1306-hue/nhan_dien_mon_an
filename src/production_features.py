"""Shared image preprocessing and feature extraction for the production model."""

import os

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import models


FEATURE_CONFIG = {
    "feature_extractor": "MobileNetV2",
    "color_features": "HSV",
    "input_size": 224,
    "weights": "ImageNet",
}


def extract_hsv_features(enhanced_rgb):
    hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
    color_features = []
    for channel, upper_bound in ((0, 180), (1, 256), (2, 256)):
        histogram = cv2.calcHist([hsv], [channel], None, [32], [0, upper_bound])
        color_features.append(
            cv2.normalize(histogram, None, norm_type=cv2.NORM_L1).flatten()
        )
    return np.hstack(color_features).astype(np.float32)


class MobileNetV2FeatureExtractor:
    """Extract ImageNet MobileNetV2 embeddings from preprocessed RGB batches."""

    def __init__(self, device="cpu", weights_path=None):
        self.device = torch.device(device)
        if weights_path is None:
            model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )
        else:
            model = models.mobilenet_v2(weights=None)
        model.classifier = torch.nn.Identity()
        if weights_path is not None:
            state_dict = torch.load(
                weights_path, map_location=self.device, weights_only=True
            )
            model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()
        self.mean = torch.tensor(
            (0.485, 0.456, 0.406), dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            (0.229, 0.224, 0.225), dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

    @torch.inference_mode()
    def extract_batch(self, images_rgb):
        images = np.asarray(images_rgb)
        if images.ndim != 4 or images.shape[1:] != (224, 224, 3):
            raise ValueError("MobileNetV2 expects RGB batches with shape (N, 224, 224, 3).")
        if images.shape[0] == 0:
            return np.empty((0, 1280), dtype=np.float32)

        inputs = torch.from_numpy(np.ascontiguousarray(images)).to(
            device=self.device, dtype=torch.float32
        )
        inputs = inputs.permute(0, 3, 1, 2).div_(255.0)
        inputs = (inputs - self.mean) / self.std
        features = self.model(inputs)
        return features.cpu().numpy().astype(np.float32, copy=False)

    def extract(self, image_rgb):
        return self.extract_batch(np.asarray(image_rgb)[None, ...])[0]

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)


def preprocess_image(img_path_or_rgb, target_size=224):
    if isinstance(img_path_or_rgb, (str, os.PathLike)):
        with Image.open(str(img_path_or_rgb)) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            img_rgb = np.array(pil_img)
    elif isinstance(img_path_or_rgb, Image.Image):
        pil_img = ImageOps.exif_transpose(img_path_or_rgb).convert("RGB")
        img_rgb = np.array(pil_img)
    else:
        img_rgb = img_path_or_rgb

    if img_rgb is None or img_rgb.size == 0:
        return None

    height, width = img_rgb.shape[:2]
    scale = min(target_size / width, target_size / height)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(
        img_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    top = (target_size - resized_height) // 2
    left = (target_size - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized

    lab = cv2.cvtColor(canvas, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    return cv2.cvtColor(
        cv2.merge([enhanced_lightness, a_channel, b_channel]), cv2.COLOR_LAB2RGB
    )


def extract_feature_groups(enhanced_rgb):
    gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        channel_axis=None,
    )

    lbp_points = 24
    lbp = local_binary_pattern(gray, P=lbp_points, R=3, method="uniform")
    bin_count = lbp_points + 2
    lbp_features, _ = np.histogram(
        lbp.ravel(), bins=bin_count, range=(0, bin_count), density=True
    )

    quantized_gray = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(
        quantized_gray,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    glcm_features = np.hstack([
        graycoprops(glcm, prop).flatten()
        for prop in (
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
        )
    ])

    return {
        "color": extract_hsv_features(enhanced_rgb),
        "hog": hog_features.astype(np.float32),
        "lbp": lbp_features.astype(np.float32),
        "glcm": glcm_features.astype(np.float32),
    }


def extract_hybrid_features(enhanced_rgb, feature_extractor):
    """Combine the primary MobileNetV2 embedding with the HSV color histogram."""
    deep_features = feature_extractor.extract(enhanced_rgb)
    features = np.hstack([deep_features, extract_hsv_features(enhanced_rgb)])
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)


def extract_features(enhanced_rgb):
    """Extract the traditional baseline: HSV + HOG + LBP + GLCM."""
    groups = extract_feature_groups(enhanced_rgb)
    features = np.hstack([groups[name] for name in ("color", "hog", "lbp", "glcm")])
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)


class FeaturePipeline:
    """Fits scaling and PCA on training data and reuses them at inference."""

    def __init__(self, n_components=0.95):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit_transform(self, features):
        scaled = self.scaler.fit_transform(features)
        return self.pca.fit_transform(scaled)

    def transform(self, features):
        scaled = self.scaler.transform(features)
        return self.pca.transform(scaled)
