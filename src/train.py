# -*- coding: utf-8 -*-
"""
src/train.py - MODULE HUẤN LUYỆN VÀ TỐI ƯU HÓA MÔ HÌNH PHÂN LỚP
Đồ án Khai phá dữ liệu: Nhận diện món ăn miền Bắc Việt Nam

Các mô hình triển khai:
1. Baselines: K-Nearest Neighbors (KNN), Decision Tree (DT), Gaussian Naive Bayes (GNB)
2. Support Vector Machine (SVM): Tinh chỉnh siêu tham số qua 5-Fold Cross Validation
3. Random Forest (RF): Ensemble Bagging với cân bằng trọng số lớp (class_weight='balanced')
4. Mô hình kết hợp (Ensemble Learning):
   - Soft Voting Ensemble với đa bộ trọng số (Weights Tuning)
   - Stacking Classifier: SVM (RBF) + Random Forest + KNN với meta-learner Logistic Regression
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. HUẤN LUYỆN CÁC MÔ HÌNH CƠ SỞ (BASELINES)
# ==============================================================================

def train_baseline_knn(X_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 5) -> Tuple[Any, float]:
    """Mô hình K-Nearest Neighbors (KNN) cơ sở."""
    start_t = time.time()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)
    elapsed = time.time() - start_t
    logger.info(f"[Baseline] Huấn luyện KNN hoàn tất trong {elapsed:.2f}s")
    return knn, elapsed


def train_baseline_decision_tree(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, float]:
    """Mô hình Cây quyết định (Decision Tree) cơ sở."""
    start_t = time.time()
    dt = DecisionTreeClassifier(max_depth=15, min_samples_split=5, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_train)
    elapsed = time.time() - start_t
    logger.info(f"[Baseline] Huấn luyện Decision Tree hoàn tất trong {elapsed:.2f}s")
    return dt, elapsed


def train_baseline_naive_bayes(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, float]:
    """Mô hình Gaussian Naive Bayes cơ sở."""
    start_t = time.time()
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    elapsed = time.time() - start_t
    logger.info(f"[Baseline] Huấn luyện Naive Bayes hoàn tất trong {elapsed:.2f}s")
    return gnb, elapsed


# ==============================================================================
# 2. HUẤN LUYỆN VÀ TINH CHỈNH SVM (GRID SEARCH 5-FOLD CV)
# ==============================================================================

def train_svm_grid_search(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    cv_folds: int = 5,
    quick_tune: bool = False
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Huấn luyện Support Vector Machine (SVM) kết hợp GridSearchCV:
    - Tìm kiếm Kernel: 'rbf', 'linear'
    - Tinh chỉnh tham số phạt C và hệ số ảnh hưởng gamma
    - CV 5-Fold Stratified, tối ưu hàm mục tiêu F1-Macro
    - Bật probability=True để sinh xác suất (hỗ trợ Soft Voting & Web confidence)
    """
    logger.info("Bắt đầu tinh chỉnh siêu tham số SVM qua 5-Fold Cross Validation...")
    start_t = time.time()
    
    if quick_tune:
        param_grid = {
            'C': [1.0, 10.0],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
    else:
        param_grid = {
            'C': [0.1, 1.0, 10.0, 50.0],
            'gamma': ['scale', 'auto', 0.01],
            'kernel': ['rbf', 'linear']
        }
        
    base_svc = SVC(probability=True, class_weight='balanced', random_state=42)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        base_svc, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='f1_macro', 
        n_jobs=-1, 
        verbose=1
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - start_t
    
    logger.info(f"SVM Tối ưu: Best Params = {grid.best_params_}, Best 5-Fold F1-Macro = {grid.best_score_:.4f}")
    logger.info(f"Thời gian tinh chỉnh SVM: {elapsed:.2f}s")
    return grid.best_estimator_, elapsed, grid.best_params_


# ==============================================================================
# 3. HUẤN LUYỆN VÀ TINH CHỈNH RANDOM FOREST (GRID SEARCH 5-FOLD CV)
# ==============================================================================

def train_random_forest_grid_search(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    cv_folds: int = 5,
    quick_tune: bool = False
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Huấn luyện Random Forest Classifier kết hợp GridSearchCV:
    - Tinh chỉnh n_estimators, max_depth, min_samples_split, max_features
    - class_weight='balanced' chống lệch về lớp nhiều mẫu
    """
    logger.info("Bắt đầu tinh chỉnh siêu tham số Random Forest qua 5-Fold Cross Validation...")
    start_t = time.time()
    
    if quick_tune:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 25],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt']
        }
    else:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
        
    base_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        base_rf, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='f1_macro', 
        n_jobs=-1, 
        verbose=1
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - start_t
    
    logger.info(f"Random Forest Tối ưu: Best Params = {grid.best_params_}, Best 5-Fold F1-Macro = {grid.best_score_:.4f}")
    logger.info(f"Thời gian tinh chỉnh RF: {elapsed:.2f}s")
    return grid.best_estimator_, elapsed, grid.best_params_


# ==============================================================================
# 4. KẾT HỢP MÔ HÌNH (ENSEMBLE: SOFT VOTING & STACKING)
# ==============================================================================

def train_soft_voting_ensemble(
    svm_model: Any, 
    rf_model: Any, 
    X_train: np.ndarray, 
    y_train: np.ndarray
) -> Tuple[Any, float, Tuple[int, int]]:
    """
    Kết hợp SVM và Random Forest theo cơ chế Soft Voting (bình chọn theo xác suất).
    Thử nghiệm nhiều cấu hình trọng số (weights tuning): (1,1), (2,1), (1,2), (3,2), (2,3)
    Chọn ra bộ trọng số có điểm Cross-validation cao nhất.
    """
    logger.info("Bắt đầu tìm kiếm bộ trọng số tối ưu cho Soft Voting Ensemble (SVM + RF)...")
    start_t = time.time()
    
    weight_candidates = [(1, 1), (2, 1), (1, 2), (3, 2), (2, 3)]
    best_weights = (1, 1)
    best_cv_score = -1.0
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for w in weight_candidates:
        voting_clf = VotingClassifier(
            estimators=[('svm', svm_model), ('rf', rf_model)],
            voting='soft',
            weights=list(w),
            n_jobs=-1
        )
        scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='f1_macro')
        mean_score = float(np.mean(scores))
        logger.debug(f"Trọng số {w}: F1-Macro CV = {mean_score:.4f}")
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_weights = w
            
    logger.info(f"Bộ trọng số tốt nhất được chọn: SVM={best_weights[0]}, RF={best_weights[1]} (CV={best_cv_score:.4f})")
    
    final_voting = VotingClassifier(
        estimators=[('svm', svm_model), ('rf', rf_model)],
        voting='soft',
        weights=list(best_weights),
        n_jobs=-1
    )
    final_voting.fit(X_train, y_train)
    elapsed = time.time() - start_t
    return final_voting, elapsed, best_weights


def train_stacking_classifier(
    svm_model: Any, 
    rf_model: Any, 
    knn_model: Any, 
    X_train: np.ndarray, 
    y_train: np.ndarray
) -> Tuple[Any, float]:
    """
    Mô hình Xếp chồng (Stacking Classifier):
    - Base Learners cấp 1: SVM (RBF) + Random Forest + KNN
    - Meta Learner cấp 2: Logistic Regression (học cách phối hợp dự đoán xác suất)
    - 5-Fold Cross-Validation sinh out-of-fold predictions ngăn overfitting.
    """
    logger.info("Bắt đầu huấn luyện mô hình Stacking Classifier...")
    start_t = time.time()
    
    base_estimators = [
        ('svm', svm_model),
        ('rf', rf_model),
        ('knn', knn_model)
    ]
    meta_learner = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    stacking_clf.fit(X_train, y_train)
    elapsed = time.time() - start_t
    logger.info(f"Stacking Classifier huấn luyện xong trong {elapsed:.2f}s")
    return stacking_clf, elapsed


# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH TOÀN BỘ GIAI ĐOẠN 4
# ==============================================================================

def run_full_training_pipeline(
    features_dir: str = r"d:\Nhan_dien_mon_an\data\features",
    models_dir: str = r"d:\Nhan_dien_mon_an\models",
    quick_mode: bool = False
):
    """Nạp ma trận đặc trưng .npy, huấn luyện toàn diện các mô hình và lưu vào thư mục models/."""
    feat_path = Path(features_dir)
    out_models_path = Path(models_dir)
    out_models_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(" GIAI ĐOẠN 4: BẮT ĐẦU HUẤN LUYỆN TOÀN DIỆN CÁC MÔ HÌNH PHÂN LỚP ")
    print("="*80 + "\n")
    
    # 1. Đọc dữ liệu đặc trưng đã lưu
    logger.info("Đang nạp ma trận đặc trưng từ disk...")
    X_train = np.load(feat_path / "X_train.npy")
    y_train = np.load(feat_path / "y_train.npy")
    X_val = np.load(feat_path / "X_val.npy")
    y_val = np.load(feat_path / "y_val.npy")
    
    logger.info(f"Kích thước X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    trained_models = {}
    train_times = {}
    
    # 2. Huấn luyện Baselines
    knn, t_knn = train_baseline_knn(X_train, y_train)
    trained_models['KNN'] = knn
    train_times['KNN'] = t_knn
    
    dt, t_dt = train_baseline_decision_tree(X_train, y_train)
    trained_models['DecisionTree'] = dt
    train_times['DecisionTree'] = t_dt
    
    gnb, t_gnb = train_baseline_naive_bayes(X_train, y_train)
    trained_models['NaiveBayes'] = gnb
    train_times['NaiveBayes'] = t_gnb
    
    # 3. Huấn luyện SVM và Random Forest
    svm_best, t_svm, svm_params = train_svm_grid_search(X_train, y_train, cv_folds=5, quick_tune=quick_mode)
    trained_models['SVM'] = svm_best
    train_times['SVM'] = t_svm
    
    rf_best, t_rf, rf_params = train_random_forest_grid_search(X_train, y_train, cv_folds=5, quick_tune=quick_mode)
    trained_models['RandomForest'] = rf_best
    train_times['RandomForest'] = t_rf
    
    # 4. Huấn luyện các mô hình kết hợp Ensemble
    voting, t_voting, best_w = train_soft_voting_ensemble(svm_best, rf_best, X_train, y_train)
    trained_models['SoftVoting'] = voting
    train_times['SoftVoting'] = t_voting
    
    stacking, t_stack = train_stacking_classifier(svm_best, rf_best, knn, X_train, y_train)
    trained_models['Stacking'] = stacking
    train_times['Stacking'] = t_stack
    
    # 5. Lưu trữ tất cả mô hình và thời gian vào disk
    logger.info("Tiến hành lưu trữ tất cả mô hình đã huấn luyện (.joblib)...")
    for name, model_obj in trained_models.items():
        save_file = out_models_path / f"{name}.joblib"
        joblib.dump(model_obj, save_file)
        logger.info(f"Đã lưu: {save_file}")
        
    # Lưu metadata thời gian huấn luyện
    time_log_file = out_models_path / "training_times.json"
    with open(time_log_file, 'w', encoding='utf-8') as f:
        json.dump(train_times, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*80)
    print(" HOÀN THÀNH GIAI ĐOẠN 4! TẤT CẢ MÔ HÌNH ĐÃ ĐƯỢC HUẤN LUYỆN VÀ LƯU THÀNH CÔNG.")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_full_training_pipeline()
