"""Model evaluation metrics, plots, and comparison utilities.

Provides:
- Multi-class and per-class classification metrics
- Dropout recall and Macro-F1 computation
- Confusion matrix and ROC-AUC curve plotting
- Cross-validation results tabulation and visualization
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src.config import TARGET_NAMES, FIGURES_DIR


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute complete set of evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Recall per class: Dropout (0), Enrolled (1), Graduate (2)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "dropout_recall": round(float(recalls[0]), 4),
        "enrolled_recall": round(float(recalls[1]), 4) if len(recalls) > 1 else 0.0,
        "graduate_recall": round(float(recalls[2]), 4) if len(recalls) > 2 else 0.0,
        "dropout_precision": round(float(precisions[0]), 4),
        "enrolled_precision": round(float(precisions[1]), 4) if len(precisions) > 1 else 0.0,
        "graduate_precision": round(float(precisions[2]), 4) if len(precisions) > 2 else 0.0,
        "dropout_f1": round(float(f1s[0]), 4),
        "enrolled_f1": round(float(f1s[1]), 4) if len(f1s) > 1 else 0.0,
        "graduate_f1": round(float(f1s[2]), 4) if len(f1s) > 2 else 0.0,
    }

    if y_proba is not None and y_proba.shape[1] == 3:
        try:
            # One-vs-rest macro ROC-AUC
            y_bin = label_binarize(y_true, classes=[0, 1, 2])
            roc_auc = float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
            metrics["roc_auc_ovr"] = round(roc_auc, 4)
        except Exception:
            metrics["roc_auc_ovr"] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage: int,
    save_path: Optional[str] = None,
) -> str:
    """Plot and save confusion matrix heatmap."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = save_path or str(FIGURES_DIR / f"confusion_matrix_stage{stage}.png")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
        cbar=False,
    )
    plt.title(f"Stage {stage} Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    stage: int,
    save_path: Optional[str] = None,
) -> str:
    """Plot and save One-vs-Rest ROC curves per class."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = save_path or str(FIGURES_DIR / f"roc_curve_stage{stage}.png")

    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = 3

    plt.figure(figsize=(8, 6))
    colors = ["#e63946", "#f4a261", "#2a9d8f"]

    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{TARGET_NAMES[i]} (AUC = {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Stage {stage} One-vs-Rest ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    stage: int,
    save_path: Optional[str] = None,
) -> str:
    """Plot bar chart comparing model candidates in a stage."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = save_path or str(FIGURES_DIR / f"model_comparison_stage{stage}.png")

    plt.figure(figsize=(10, 5))
    metrics_to_plot = ["macro_f1", "dropout_recall", "accuracy"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    df_melt = pd.melt(
        comparison_df,
        id_vars=["model_name"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    sns.barplot(data=df_melt, x="model_name", y="Score", hue="Metric", palette="Set2")
    plt.title(f"Stage {stage} Model Candidate Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file

