# -*- coding: utf-8 -*-
"""Measure SVM C/gamma sensitivity with stratified cross-validation."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
REPORTS_DIR = BASE_DIR / "reports"
REPORT_FIGS = REPORTS_DIR / "figures"


def run_hyperparameter_search():
    x_train_path = FEATURES_DIR / "X_train.npy"
    y_train_path = FEATURES_DIR / "y_train.npy"
    if not x_train_path.is_file() or not y_train_path.is_file():
        raise FileNotFoundError(
            "Không có feature train. Hãy chạy `python train_production_model.py` trước."
        )

    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", cache_size=1000),
        param_grid={
            "C": [0.1, 1.0, 10.0, 50.0, 100.0],
            "gamma": ["scale", "auto"],
        },
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    results = [
        {
            "C": float(params["C"]),
            "gamma": params["gamma"],
            "F1_Macro_Mean": float(mean_score),
            "F1_Macro_Std": float(std_score),
        }
        for params, mean_score, std_score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        )
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "svm_hyperparameter_search.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    REPORT_FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for gamma, color, marker in (
        ("scale", "#C0392B", "o"),
        ("auto", "#2980B9", "s"),
    ):
        gamma_rows = [row for row in results if row["gamma"] == gamma]
        ax.errorbar(
            [row["C"] for row in gamma_rows],
            [row["F1_Macro_Mean"] for row in gamma_rows],
            yerr=[row["F1_Macro_Std"] for row in gamma_rows],
            marker=marker,
            color=color,
            capsize=3,
            label=f'gamma = "{gamma}"',
        )

    ax.set_xscale("log")
    ax.set_title("SVM RBF: F1-Macro theo C và gamma (Stratified 3-Fold CV)")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("F1-Macro")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    output = REPORT_FIGS / "04_svm_hyperparameter_tuning.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Best CV F1-Macro: {search.best_score_:.4f} ({search.best_params_})")
    print(f"Đã lưu kết quả thực nghiệm: {output}")


if __name__ == "__main__":
    run_hyperparameter_search()
