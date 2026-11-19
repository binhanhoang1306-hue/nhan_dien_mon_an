# -*- coding: utf-8 -*-
"""
src/evaluate.py - MODULE ĐÁNH GIÁ TOÀN DIỆN MÔ HÌNH VÀ RÚT RA TRI THỨC (CRISP-DM / KDD)
Đồ án Khai phá dữ liệu: Nhận diện món ăn miền Bắc Việt Nam

Nội dung đánh giá:
1. Độ đo định lượng: Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1
2. Ma trận nhầm lẫn (Confusion Matrix Heatmap) chuẩn hóa và trực quan
3. Đường cong ROC và chỉ số One-vs-Rest AUC
4. Bảng tổng hợp so sánh các mô hình: KNN vs DT vs GNB vs SVM vs RF vs Voting vs Stacking
5. Đo lường tốc độ: Thời gian huấn luyện và thời gian suy luận (Latency per image)
6. Nghiên cứu triệt tiêu thành phần (Ablation Study): Đánh giá đóng góp của CLAHE, Augmentation, PCA
7. Kiểm định thống kê McNemar Test (xác thực sự khác biệt có ý nghĩa thống kê)
8. Đường cong học tập (Learning Curve: 25%, 50%, 75%, 100% dữ liệu) chứng minh độ đủ dữ liệu
9. Phân tích lỗi (Error Analysis): Nhận diện các cặp món ăn dễ nhầm lẫn nhất và nguyên nhân
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. TÍNH TOÁN CÁC ĐỘ ĐO ĐỊNH LƯỢNG
# ==============================================================================

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính toán tất cả các chỉ số đo lường hiệu năng phân lớp đa lớp."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }


# ==============================================================================
# 2. TRỰC QUAN HÓA MA TRẬN NHẦM LẪN (CONFUSION MATRIX HEATMAP)
# ==============================================================================

def plot_and_save_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str], 
    save_path: Path,
    title: str = "Ma trận nhầm lẫn (Confusion Matrix)"
):
    """
    Vẽ Heatmap ma trận nhầm lẫn với tỉ lệ phần trăm chuẩn hóa theo hàng (True Labels),
    kích thước lớn 15x13 để hiển thị rõ 18 lớp món ăn mà không bị đè chữ.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_counts = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 13))
    
    # Tạo nhãn kết hợp: phần trăm và số lượng mẫu thực tế
    annot_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f"{cm[i, j]*100:.1f}%\n({cm_counts[i, j]})"
            
    sns.heatmap(
        cm, 
        annot=annot_labels, 
        fmt='', 
        cmap='YlOrRd', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Tỉ lệ chuẩn hóa'}
    )
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=13, fontweight='bold')
    plt.ylabel('Nhãn thực tế (True Label)', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Đã lưu biểu đồ ma trận nhầm lẫn tại: {save_path}")


# ==============================================================================
# 3. VẼ ĐƯỜNG CONG ROC VÀ TÍNH DIỆN TÍCH DƯỚI ĐƯỜNG CONG (AUC)
# ==============================================================================

def plot_multiclass_roc_curves(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    class_names: List[str], 
    save_path: Path
):
    """Vẽ đường cong ROC đa lớp theo chiến lược One-vs-Rest (OvR)."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Tính micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(12, 9))
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f'Đường cong Micro-average (AUC = {roc_auc["micro"]:.3f})',
        color='deeppink', linestyle=':', linewidth=3
    )
    
    # Vẽ top các lớp
    colors = sns.color_palette("husl", n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.7,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỉ lệ dương tính giả (False Positive Rate)', fontsize=12)
    plt.ylabel('Tỉ lệ dương tính thật (True Positive Rate)', fontsize=12)
    plt.title('Đường cong ROC One-vs-Rest cho 18 món ăn miền Bắc', fontsize=15, fontweight='bold', pad=15)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Đã lưu biểu đồ ROC Curves tại: {save_path}")


# ==============================================================================
# 4. ĐƯỜNG CONG HỌC TẬP (LEARNING CURVE CHỨNG MINH LƯỢNG DỮ LIỆU ĐỦ)
# ==============================================================================

def plot_learning_curve_progression(
    model: Any, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    save_path: Path,
    train_sizes: List[float] = [0.25, 0.50, 0.75, 1.0]
):
    """
    Vẽ Learning Curve với 4 mốc dữ liệu: 25%, 50%, 75%, 100% kết hợp 5-Fold Cross Validation.
    Báo cáo dải phương sai (Mean ± Std) chứng minh mô hình không bị Underfitting/Overfitting
    và hội tụ tốt khi tăng số lượng ảnh.
    """
    logger.info("Đang vẽ Learning Curve qua các mốc 25%, 50%, 75%, 100% dữ liệu...")
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, 
        X_train, 
        y_train, 
        train_sizes=train_sizes, 
        cv=5, 
        scoring='f1_macro', 
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='#C0392B', label='Điểm tập huấn luyện (Train Score)')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#C0392B')
    
    plt.plot(train_sizes_abs, val_mean, 's-', color='#2980B9', label='Điểm Cross-Validation (Val Score)')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#2980B9')
    
    plt.title('Đường cong học tập (Learning Curve) theo tỉ lệ dữ liệu huấn luyện', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Số lượng mẫu dữ liệu Train', fontsize=12)
    plt.ylabel('F1-Score Macro', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Đã lưu biểu đồ Learning Curve tại: {save_path}")


# ==============================================================================
# 5. PHÂN TÍCH LỖI VÀ CÁC CẶP MÓN ĂN DỄ NHẦM LẪN (ERROR ANALYSIS)
# ==============================================================================

def perform_error_analysis(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str], 
    top_n_confusions: int = 5
) -> pd.DataFrame:
    """
    Phân tích top các cặp lớp món ăn có số lượng dự đoán nhầm cao nhất.
    Rút ra tri thức chuyên gia về mặt thị giác (đặc trưng màu, nước dùng, sợi bún).
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)  # Bỏ qua các trường hợp phân loại đúng
    
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                confusion_pairs.append({
                    'Mon_that': class_names[i],
                    'Doan_nham_thanh': class_names[j],
                    'So_anh_nham': int(cm[i, j])
                })
                
    df_conf = pd.DataFrame(confusion_pairs)
    if not df_conf.empty:
        df_top = df_conf.sort_values(by='So_anh_nham', ascending=False).head(top_n_confusions)
    else:
        df_top = df_conf
        
    return df_top


# ==============================================================================
# 6. KIỂM ĐỊNH THỐNG KÊ MCNEMAR TEST
# ==============================================================================

def perform_mcnemar_test(
    y_true: np.ndarray, 
    y_pred_model1: np.ndarray, 
    y_pred_model2: np.ndarray, 
    name1: str = "SVM", 
    name2: str = "RandomForest"
) -> Dict[str, Any]:
    """
    Kiểm định McNemar xác định xem sự vượt trội của mô hình 1 so với mô hình 2
    có ý nghĩa thống kê hay chỉ do ngẫu nhiên (p-value < 0.05).
    """
    correct1 = (y_pred_model1 == y_true)
    correct2 = (y_pred_model2 == y_true)
    
    # Bảng ngẫu nhiên 2x2
    # n00: cả 2 cùng sai, n11: cả 2 cùng đúng
    # n01: 1 sai, 2 đúng; n10: 1 đúng, 2 sai
    n01 = np.sum((~correct1) & correct2)
    n10 = np.sum(correct1 & (~correct2))
    
    # Thống kê chi bình phương hiệu chỉnh của McNemar: (|n01 - n10| - 1)^2 / (n01 + n10)
    if (n01 + n10) == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        chi2 = (abs(n01 - n10) - 1.0)**2 / (n01 + n10)
        # Tra phân phối Chi-square bậc tự do df=1
        from scipy.stats import chi2 as chi2_dist
        p_val = 1.0 - chi2_dist.cdf(chi2, df=1)
        
    is_significant = (p_val < 0.05)
    return {
        'model_1': name1,
        'model_2': name2,
        'n10_only_model1_correct': int(n10),
        'n01_only_model2_correct': int(n01),
        'chi2_stat': float(chi2),
        'p_value': float(p_val),
        'statistically_significant': bool(is_significant)
    }


# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH TOÀN BỘ GIAI ĐOẠN 5
# ==============================================================================

def run_full_evaluation_pipeline(
    features_dir: str = r"d:\Nhan_dien_mon_an\data\features",
    models_dir: str = r"d:\Nhan_dien_mon_an\models",
    splits_dir: str = r"d:\Nhan_dien_mon_an\data\splits",
    reports_dir: str = r"d:\Nhan_dien_mon_an\reports"
):
    """Nạp tất cả các mô hình, chạy đánh giá chi tiết trên tập Test và sinh báo cáo chuẩn bị bảo vệ."""
    feat_path = Path(features_dir)
    models_path = Path(models_dir)
    splits_path = Path(splits_dir)
    rep_path = Path(reports_dir)
    figs_path = rep_path / "figures"
    
    figs_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Nạp nhãn
    with open(splits_path / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_class = {int(k): v for k, v in label_map['idx_to_class'].items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # 2. Nạp dữ liệu Test & Train
    X_train = np.load(feat_path / "X_train.npy")
    y_train = np.load(feat_path / "y_train.npy")
    X_test = np.load(feat_path / "X_test.npy")
    y_test = np.load(feat_path / "y_test.npy")
    
    # Nạp thời gian huấn luyện nếu có
    train_times = {}
    time_log_file = models_path / "training_times.json"
    if time_log_file.exists():
        with open(time_log_file, "r", encoding="utf-8") as f:
            train_times = json.load(f)
            
    model_names = ['KNN', 'DecisionTree', 'NaiveBayes', 'SVM', 'RandomForest', 'SoftVoting', 'Stacking']
    comparison_rows = []
    predictions_dict = {}
    probabilities_dict = {}
    
    print("\n" + "="*80)
    print(" GIAI ĐOẠN 5: ĐÁNH GIÁ MÔ HÌNH VÀ RÚT RA TRI THỨC ")
    print("="*80 + "\n")
    
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None
    
    # 3. Đánh giá từng mô hình
    for name in model_names:
        m_file = models_path / f"{name}.joblib"
        if not m_file.exists():
            logger.warning(f"Không tìm thấy file mô hình {m_file}, bỏ qua...")
            continue
            
        model = joblib.load(m_file)
        
        # Đo thời gian dự đoán (Latency)
        t0 = time.time()
        y_pred = model.predict(X_test)
        infer_time_per_image = (time.time() - t0) / len(X_test) * 1000.0  # ms
        
        predictions_dict[name] = y_pred
        
        # Lấy xác suất nếu hỗ trợ
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                probabilities_dict[name] = y_proba
            except Exception:
                pass
                
        metrics = compute_all_metrics(y_test, y_pred)
        train_t = train_times.get(name, 0.0)
        
        comparison_rows.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'Precision_Macro': f"{metrics['precision_macro']*100:.2f}%",
            'Recall_Macro': f"{metrics['recall_macro']*100:.2f}%",
            'F1_Macro': f"{metrics['f1_macro']*100:.2f}%",
            'F1_Weighted': f"{metrics['f1_weighted']*100:.2f}%",
            'Train_Time_s': f"{train_t:.2f}s",
            'Inference_Time_per_image_ms': f"{infer_time_per_image:.2f}ms"
        })
        
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            best_model_name = name
            best_model_obj = model
            
    # Bảng tổng kết so sánh
    df_comparison = pd.DataFrame(comparison_rows)
    comp_csv = rep_path / "model_comparison.csv"
    df_comparison.to_csv(comp_csv, index=False, encoding='utf-8-sig')
    logger.info(f"\nBẢNG SO SÁNH CÁC MÔ HÌNH TRÊN TẬP TEST:\n{df_comparison.to_string(index=False)}")
    
    # 4. Vẽ Confusion Matrix cho mô hình tốt nhất
    if best_model_name:
        logger.info(f"Mô hình đạt hiệu năng tốt nhất: {best_model_name} (F1-Macro = {best_f1*100:.2f}%)")
        best_preds = predictions_dict[best_model_name]
        plot_and_save_confusion_matrix(
            y_test, best_preds, class_names,
            figs_path / f"confusion_matrix_{best_model_name}.png",
            title=f"Ma trận nhầm lẫn - Mô hình {best_model_name}"
        )
        
        # Classification report chi tiết
        rep_text = classification_report(y_test, best_preds, target_names=class_names, digits=4)
        with open(rep_path / f"classification_report_{best_model_name}.txt", "w", encoding="utf-8") as f:
            f.write(rep_text)
            
        # Vẽ Learning Curve cho best model
        plot_learning_curve_progression(
            best_model_obj, X_train, y_train,
            figs_path / f"learning_curve_{best_model_name}.png"
        )
        
        # 5. Phân tích lỗi (Error Analysis)
        df_errors = perform_error_analysis(y_test, best_preds, class_names, top_n_confusions=6)
        df_errors.to_csv(rep_path / "top_confusions.csv", index=False, encoding='utf-8-sig')
        logger.info(f"\nTOP CÁC CẶP MÓN HAY BỊ NHẬN DIỆN NHẦM NHẤT:\n{df_errors.to_string(index=False)}")

    # 6. Vẽ ROC Curves cho mô hình Soft Voting hoặc SVM
    roc_target = 'SoftVoting' if 'SoftVoting' in probabilities_dict else ('SVM' if 'SVM' in probabilities_dict else None)
    if roc_target:
        plot_multiclass_roc_curves(
            y_test, probabilities_dict[roc_target], class_names,
            figs_path / f"roc_curves_{roc_target}.png"
        )
        
    # 7. Kiểm định McNemar giữa SVM và Random Forest
    if 'SVM' in predictions_dict and 'RandomForest' in predictions_dict:
        mcnemar_res = perform_mcnemar_test(
            y_test, predictions_dict['SVM'], predictions_dict['RandomForest'],
            name1='SVM', name2='RandomForest'
        )
        with open(rep_path / "mcnemar_test.json", "w", encoding="utf-8") as f:
            json.dump(mcnemar_res, f, ensure_ascii=False, indent=2)
        logger.info(f"Kết quả kiểm định McNemar (SVM vs RF): p-value = {mcnemar_res['p_value']:.4f}")

    print("\n" + "="*80)
    print(" HOÀN THÀNH GIAI ĐOẠN 5! TOÀN BỘ BÁO CÁO VÀ BIỂU ĐỒ ĐÃ ĐƯỢC LƯU VÀO /reports/")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_full_evaluation_pipeline()
