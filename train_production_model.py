# -*- coding: utf-8 -*-
"""
train_production_model.py - Huấn luyện pipeline MobileNetV2 + HSV cho web
Mô hình: SVM (RBF) + Random Forest + Soft Voting Ensemble
"""

import json
import time
import logging
import csv
from pathlib import Path

import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.production_features import (
    FEATURE_CONFIG,
    FeaturePipeline,
    MobileNetV2FeatureExtractor,
    extract_feature_groups,
    preprocess_image,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "dataset" / "Images"
MODELS_DIR = BASE_DIR / "models"
SPLITS_DIR = BASE_DIR / "data" / "splits"
FEATURES_DIR = BASE_DIR / "data" / "features"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def extract_dataset_features(items, feature_extractor, batch_size=32):
    """Extract primary and baseline features from the same valid image set."""
    group_names = ("mobilenet", "color", "hog", "lbp", "glcm")
    feature_lists = {name: [] for name in group_names}
    labels = []

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        images = []
        batch_labels = []
        for path, label in batch:
            try:
                image = preprocess_image(path)
            except (OSError, ValueError):
                logger.warning("Bỏ qua ảnh không đọc được: %s", path, exc_info=True)
                continue
            if image is not None:
                images.append(image)
                batch_labels.append(label)

        if not images:
            continue

        deep_features = feature_extractor.extract_batch(np.stack(images))
        for image in images:
            baseline_features = extract_feature_groups(image)
            for name in ("color", "hog", "lbp", "glcm"):
                feature_lists[name].append(baseline_features[name])
        feature_lists["mobilenet"].extend(deep_features)
        labels.extend(batch_labels)

        batch_end = min(batch_start + batch_size, len(items))
        logger.info("Đã trích đặc trưng: %s/%s", batch_end, len(items))

    if not labels:
        raise RuntimeError("Không trích xuất được đặc trưng từ bất kỳ ảnh nào.")

    feature_groups = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in feature_lists.items()
    }
    return feature_groups, np.asarray(labels, dtype=np.int64)


def run_ablation_study(
    feature_groups,
    y_train,
    y_test,
    train_indices,
    test_indices,
    hybrid_svm_predictions,
    baseline_svm_predictions,
    voting_predictions,
):
    """Compare the primary and traditional feature pipelines on the same test split."""
    configurations = [
        ("HSV + PCA + SVM", ("color",)),
        ("HSV + HOG + PCA + SVM", ("color", "hog")),
    ]
    rows = []

    for name, selected_groups in configurations:
        X_train = np.hstack([
            feature_groups[group][train_indices] for group in selected_groups
        ])
        X_test = np.hstack([
            feature_groups[group][test_indices] for group in selected_groups
        ])
        pipeline = FeaturePipeline(n_components=0.95)
        X_train_pca = pipeline.fit_transform(X_train)
        X_test_pca = pipeline.transform(X_test)
        model = SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            random_state=42,
            cache_size=1000,
        )
        model.fit(X_train_pca, y_train)
        predictions = model.predict(X_test_pca)
        rows.append({
            "Stage": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "F1_Macro": f1_score(y_test, predictions, average="macro"),
        })

    rows.append({
        "Stage": "MobileNetV2 + HSV + PCA + SVM",
        "Accuracy": accuracy_score(y_test, hybrid_svm_predictions),
        "F1_Macro": f1_score(y_test, hybrid_svm_predictions, average="macro"),
    })
    rows.append({
        "Stage": "HSV + HOG + LBP + GLCM + PCA + SVM (baseline)",
        "Accuracy": accuracy_score(y_test, baseline_svm_predictions),
        "F1_Macro": f1_score(y_test, baseline_svm_predictions, average="macro"),
    })
    rows.append({
        "Stage": "MobileNetV2 + HSV + PCA + SoftVoting (web)",
        "Accuracy": accuracy_score(y_test, voting_predictions),
        "F1_Macro": f1_score(y_test, voting_predictions, average="macro"),
    })
    return rows


# =============================================================================
# HUẤN LUYỆN
# =============================================================================

def main():
    print("=" * 75)
    print(" HUẤN LUYỆN PIPELINE MOBILENETV2 + HSV - MÓN ĂN MIỀN BẮC")
    print("=" * 75)

    if not RAW_DATA_DIR.is_dir():
        raise FileNotFoundError(f"Không tìm thấy dataset: {RAW_DATA_DIR}")

    classes = sorted(
        directory.name for directory in RAW_DATA_DIR.iterdir() if directory.is_dir()
    )
    if len(classes) < 2:
        raise ValueError(f"Cần ít nhất 2 lớp ảnh để huấn luyện, chỉ tìm thấy {len(classes)}.")

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for i, cls in enumerate(classes)}

    label_map_data = {
        'class_to_idx': class_to_idx,
        'idx_to_class': {str(k): v for k, v in idx_to_class.items()},
    }
    with open(SPLITS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu label_map.json cho {len(classes)} lớp: {classes}")

    items = []
    rng = np.random.default_rng(42)
    max_per_class = 500

    for cls in classes:
        cls_dir = RAW_DATA_DIR / cls
        files = sorted(
            path for path in cls_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not files:
            logger.warning("Bỏ qua lớp không có ảnh hợp lệ: %s", cls)
            continue
        if len(files) > max_per_class:
            selected = [files[index] for index in rng.choice(
                len(files), size=max_per_class, replace=False
            )]
        else:
            selected = files

        label = class_to_idx[cls]
        for f in selected:
            items.append((str(f), label))

    logger.info(f"Tổng số ảnh đưa vào huấn luyện: {len(items)}")

    if len({label for _, label in items}) < 2:
        raise ValueError("Dataset phải có ít nhất hai lớp chứa ảnh hợp lệ.")

    logger.info("Đang khởi tạo bộ trích đặc trưng MobileNetV2...")
    t0 = time.time()
    feature_extractor = MobileNetV2FeatureExtractor(device="cpu")
    feature_groups, y = extract_dataset_features(items, feature_extractor)
    logger.info(
        "Trích đặc trưng xong trong %.1fs. Kích thước: %s",
        time.time() - t0,
        {name: features.shape for name, features in feature_groups.items()},
    )

    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.20, stratify=y, random_state=42
    )
    y_train, y_test = y[train_indices], y[test_indices]
    primary_groups = ("mobilenet", "color")
    baseline_groups = ("color", "hog", "lbp", "glcm")
    X_train = np.hstack([
        feature_groups[name][train_indices] for name in primary_groups
    ])
    X_test = np.hstack([
        feature_groups[name][test_indices] for name in primary_groups
    ])
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    logger.info("Fit StandardScaler + PCA trên tập train...")
    pipeline = FeaturePipeline(n_components=0.95)
    X_train_pca = pipeline.fit_transform(X_train)
    X_test_pca = pipeline.transform(X_test)
    logger.info(f"Số chiều sau PCA: {X_train_pca.shape[1]}")

    logger.info("Huấn luyện SVM (RBF, C=10, gamma=scale, balanced)...")
    started_at = time.time()
    svm = SVC(C=10.0, kernel='rbf', gamma='scale', probability=True,
              class_weight='balanced', random_state=42, cache_size=1000)
    svm.fit(X_train_pca, y_train)
    svm_train_time = time.time() - started_at
    svm_pred = svm.predict(X_test_pca)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred, average='macro')
    logger.info(f"SVM Accuracy: {svm_acc * 100:.2f}% | F1-macro: {svm_f1 * 100:.2f}%")

    logger.info("Đo baseline HSV + HOG + LBP + GLCM với SVM...")
    baseline_X_train = np.hstack([
        feature_groups[name][train_indices] for name in baseline_groups
    ])
    baseline_X_test = np.hstack([
        feature_groups[name][test_indices] for name in baseline_groups
    ])
    baseline_pipeline = FeaturePipeline(n_components=0.95)
    baseline_X_train_pca = baseline_pipeline.fit_transform(baseline_X_train)
    baseline_X_test_pca = baseline_pipeline.transform(baseline_X_test)
    baseline_svm = SVC(
        C=10.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        random_state=42,
        cache_size=1000,
    )
    baseline_svm.fit(baseline_X_train_pca, y_train)
    baseline_pred = baseline_svm.predict(baseline_X_test_pca)

    logger.info("Huấn luyện Random Forest (200 trees, balanced)...")
    started_at = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=25, min_samples_split=3,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_pca, y_train)
    rf_train_time = time.time() - started_at
    rf_pred = rf.predict(X_test_pca)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='macro')
    logger.info(f"Random Forest Accuracy: {rf_acc * 100:.2f}% | F1-macro: {rf_f1 * 100:.2f}%")

    logger.info("Kết hợp Soft Voting Ensemble (SVM:2 + RF:1)...")
    started_at = time.time()
    voting = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf)],
        voting='soft', weights=[2, 1], n_jobs=-1
    )
    voting.fit(X_train_pca, y_train)
    voting_train_time = time.time() - started_at
    vt_pred = voting.predict(X_test_pca)
    vt_acc = accuracy_score(y_test, vt_pred)
    vt_f1 = f1_score(y_test, vt_pred, average='macro')
    logger.info(f"Soft Voting Accuracy: {vt_acc * 100:.2f}% | F1-macro: {vt_f1 * 100:.2f}%")

    logger.info("Lưu mô hình vào models/ ...")
    joblib.dump(pipeline, MODELS_DIR / "feature_pipeline.joblib")
    feature_extractor.save_weights(MODELS_DIR / "mobilenet_v2_features.pt")
    with (MODELS_DIR / "feature_config.json").open("w", encoding="utf-8") as file:
        json.dump(FEATURE_CONFIG, file, ensure_ascii=False, indent=2)
    joblib.dump(svm, MODELS_DIR / "SVM.joblib")
    joblib.dump(rf, MODELS_DIR / "RandomForest.joblib")
    joblib.dump(voting, MODELS_DIR / "SoftVoting.joblib")
    with (MODELS_DIR / "training_times.json").open("w", encoding="utf-8") as file:
        json.dump({
            "SVM": svm_train_time,
            "RandomForest": rf_train_time,
            "SoftVoting": voting_train_time,
        }, file, indent=2)

    # Lưu features để phân tích
    np.save(FEATURES_DIR / "X_train.npy", X_train_pca)
    np.save(FEATURES_DIR / "y_train.npy", y_train)
    np.save(FEATURES_DIR / "X_test.npy", X_test_pca)
    np.save(FEATURES_DIR / "y_test.npy", y_test)

    model_predictions = {
        "SVM": svm_pred,
        "RandomForest": rf_pred,
        "SoftVoting": vt_pred,
    }
    comparison_rows = []
    for model_name, predictions in model_predictions.items():
        comparison_rows.append({
            "Model": f"MobileNetV2 + HSV / {model_name}",
            "Accuracy": accuracy_score(y_test, predictions),
            "F1_Macro": f1_score(y_test, predictions, average="macro"),
        })
    comparison_rows.append({
        "Model": "HSV + HOG + LBP + GLCM baseline / SVM",
        "Accuracy": accuracy_score(y_test, baseline_pred),
        "F1_Macro": f1_score(y_test, baseline_pred, average="macro"),
    })
    with open(REPORTS_DIR / "model_comparison.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["Model", "Accuracy", "F1_Macro"])
        writer.writeheader()
        writer.writerows(comparison_rows)

    ablation_rows = run_ablation_study(
        feature_groups,
        y_train,
        y_test,
        train_indices,
        test_indices,
        svm_pred,
        baseline_pred,
        vt_pred,
    )
    with open(REPORTS_DIR / "ablation_results.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["Stage", "Accuracy", "F1_Macro"])
        writer.writeheader()
        writer.writerows(ablation_rows)

    print("\n" + "=" * 75)
    print(f" Số lớp đã huấn luyện: {len(classes)}")
    print(f" ✅ HUẤN LUYỆN THÀNH CÔNG!")
    print(f" SVM:     {svm_acc * 100:.2f}% acc | {svm_f1 * 100:.2f}% F1")
    print(f" RF:      {rf_acc * 100:.2f}% acc | {rf_f1 * 100:.2f}% F1")
    print(f" Voting:  {vt_acc * 100:.2f}% acc | {vt_f1 * 100:.2f}% F1")
    print(f" Models:  {MODELS_DIR}")
    print(f" Test metrics: {REPORTS_DIR / 'model_comparison.csv'}")
    print(f" Ablation metrics: {REPORTS_DIR / 'ablation_results.csv'}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
