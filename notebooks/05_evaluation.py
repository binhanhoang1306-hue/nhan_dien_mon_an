# -*- coding: utf-8 -*-
"""Plot model and ablation results measured by the training pipeline."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORT_FIGS = REPORTS_DIR / "figures"


def load_results(filename, required_columns):
    results_path = REPORTS_DIR / filename
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Chưa có {results_path}. Hãy chạy `python train_production_model.py` trước."
        )

    with results_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(required_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{results_path} thiếu các cột: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{results_path} không có kết quả thực nghiệm.")
    return rows


def plot_model_comparison_chart():
    rows = load_results("model_comparison.csv", ("Model", "Accuracy", "F1_Macro"))
    models = [row["Model"] for row in rows]
    accuracies = [float(row["Accuracy"]) * 100 for row in rows]
    f1_scores = [float(row["F1_Macro"]) * 100 for row in rows]
    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_accuracy = ax.bar(
        x - width / 2, accuracies, width, label="Accuracy", color="#2980B9"
    )
    bars_f1 = ax.bar(
        x + width / 2, f1_scores, width, label="F1-Macro", color="#C0392B"
    )
    ax.set_ylabel("Điểm trên tập Test (%)")
    ax.set_title("So sánh mô hình trên cùng một tập Test")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    ax.bar_label(bars_accuracy, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(bars_f1, fmt="%.1f", padding=2, fontsize=8)
    fig.tight_layout()
    output = REPORT_FIGS / "05_models_performance_comparison.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Đã lưu biểu đồ kết quả thực tế: {output}")


def plot_ablation_study_chart():
    rows = load_results("ablation_results.csv", ("Stage", "F1_Macro"))
    stages = [row["Stage"] for row in rows]
    f1_scores = [float(row["F1_Macro"]) * 100 for row in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(stages))
    bars = ax.bar(x, f1_scores, color="#27AE60")
    ax.set_ylabel("F1-Macro trên tập Test (%)")
    ax.set_title("Ablation: đặc trưng và SoftVoting (cùng tập Test)")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=18, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, labels=[f"{score:.1f}%" for score in f1_scores], padding=2)
    fig.tight_layout()
    output = REPORT_FIGS / "05_ablation_study_progression.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Đã lưu biểu đồ ablation thực tế: {output}")


if __name__ == "__main__":
    REPORT_FIGS.mkdir(parents=True, exist_ok=True)
    plot_model_comparison_chart()
    plot_ablation_study_chart()
