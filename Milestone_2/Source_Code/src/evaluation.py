"""
Evaluation module for Milestone 2.

Provides metrics, confusion matrices, ROC/PR curves, and calibration plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    brier_score_loss,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Optional predicted probabilities.
        average: Averaging strategy for multi-class.

    Returns:
        Dictionary of metric values.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # Per-class metrics
    classes = np.unique(y_true)
    for cls in classes:
        y_true_cls = (np.array(y_true) == cls).astype(int)
        y_pred_cls = (np.array(y_pred) == cls).astype(int)

        metrics[f"precision_class_{cls}"] = precision_score(y_true_cls, y_pred_cls, zero_division=0)
        metrics[f"recall_class_{cls}"] = recall_score(y_true_cls, y_pred_cls, zero_division=0)
        metrics[f"f1_class_{cls}"] = f1_score(y_true_cls, y_pred_cls, zero_division=0)

    # ROC-AUC if probabilities available
    if y_proba is not None:
        try:
            if len(classes) == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics["roc_auc_ovr"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
        except Exception:
            pass

    return metrics


def compute_per_class_report(
    y_true,
    y_pred,
    target_names: Optional[List[str]] = None,
) -> str:
    """Generate classification report string.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        target_names: Optional class names.

    Returns:
        Classification report string.
    """
    return classification_report(y_true, y_pred, target_names=target_names, digits=3)


# ============================================================================
# REGRESSION METRICS
# ============================================================================

def compute_regression_metrics(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metric values.
    """
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
        "brier": float(np.mean((np.array(y_pred) - np.array(y_true)) ** 2)),
    }


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Plot and optionally save confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class names.
        title: Plot title.
        save_path: Optional path to save figure.
        figsize: Figure size.
    """
    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix: {save_path}")

    plt.close(fig)


# ============================================================================
# ROC AND PR CURVES
# ============================================================================

def plot_roc_curve(
    y_true,
    y_proba,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Dict[str, float]:
    """Plot ROC curve for binary or multiclass.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        title: Plot title.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Dictionary with AUC values.
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=figsize)
    aucs = {}

    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        aucs["auc"] = auc
    else:
        # Multiclass - one vs rest
        y_bin = label_binarize(y_true, classes=classes)

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, label=f"Class {cls} (AUC = {auc:.3f})")
            aucs[f"auc_class_{cls}"] = auc

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curve: {save_path}")

    plt.close(fig)
    return aucs


def plot_pr_curve(
    y_true,
    y_proba,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Dict[str, float]:
    """Plot Precision-Recall curve.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        title: Plot title.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Dictionary with average precision values.
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=figsize)
    aps = {}

    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        ap = average_precision_score(y_true, y_proba[:, 1])
        ax.plot(recall, precision, label=f"AP = {ap:.3f}")
        aps["average_precision"] = ap
    else:
        y_bin = label_binarize(y_true, classes=classes)

        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            ap = average_precision_score(y_bin[:, i], y_proba[:, i])
            ax.plot(recall, precision, label=f"Class {cls} (AP = {ap:.3f})")
            aps[f"ap_class_{cls}"] = ap

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved PR curve: {save_path}")

    plt.close(fig)
    return aps


# ============================================================================
# CALIBRATION
# ============================================================================

def plot_calibration_curve(
    y_true,
    y_proba,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Dict[str, float]:
    """Plot reliability diagram (calibration curve).

    Args:
        y_true: True labels (binary).
        y_proba: Predicted probabilities for positive class.
        n_bins: Number of bins for calibration.
        title: Plot title.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Dictionary with Brier score.
    """
    # Ensure binary
    y_true_bin = np.array(y_true)
    if y_true_bin.ndim > 1:
        y_true_bin = y_true_bin.argmax(axis=1)

    prob_true, prob_pred = calibration_curve(y_true_bin, y_proba, n_bins=n_bins)

    brier = brier_score_loss(y_true_bin, y_proba)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.plot(prob_pred, prob_true, "s-", label=f"Model (Brier={brier:.3f})")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(title)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.hist(y_proba, bins=n_bins, range=(0, 1), alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved calibration curve: {save_path}")

    plt.close(fig)
    return {"brier_score": brier}


def compute_brier_score(y_true, y_proba) -> float:
    """Compute Brier score for probability predictions.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.

    Returns:
        Brier score (lower is better).
    """
    return brier_score_loss(y_true, y_proba)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_evaluation_report(
    model_name: str,
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate comprehensive evaluation report.

    Args:
        model_name: Name of the model.
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Optional predicted probabilities.
        target_names: Optional class names.
        save_dir: Optional directory to save plots.

    Returns:
        Dictionary with all metrics and paths.
    """
    report = {
        "model_name": model_name,
        "metrics": compute_classification_metrics(y_true, y_pred, y_proba),
        "classification_report": compute_per_class_report(y_true, y_pred, target_names),
    }

    if save_dir is not None:
        save_dir = Path(save_dir)

        # Confusion matrix
        cm_path = save_dir / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, target_names, f"{model_name} - Confusion Matrix", cm_path)
        report["confusion_matrix_path"] = str(cm_path)

        if y_proba is not None:
            # ROC curve
            roc_path = save_dir / f"{model_name}_roc_curve.png"
            aucs = plot_roc_curve(y_true, y_proba, f"{model_name} - ROC Curve", roc_path)
            report["metrics"].update(aucs)

            # PR curve
            pr_path = save_dir / f"{model_name}_pr_curve.png"
            aps = plot_pr_curve(y_true, y_proba, f"{model_name} - PR Curve", pr_path)
            report["metrics"].update(aps)

    return report
