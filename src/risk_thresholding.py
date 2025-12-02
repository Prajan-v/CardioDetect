"""Utility functions for threshold tuning and operating modes
for the frozen risk prediction model in CardioDetect.

This module does NOT retrain any models. It only:
- Loads the existing frozen risk model pipeline from disk.
- Recreates the original Framingham train/val/test splits.
- Applies the same feature engineering used during training.
- Runs threshold sweeps on the validation set.
- Evaluates different operating modes on the test set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RISK_MODEL_PATH = PROJECT_ROOT / "models" / "risk_model_pipeline.pkl"
FRAMINGHAM_PATH = PROJECT_ROOT / "data" / "raw" / "framingham_raw.csv"

THRESH_SWEEP_CSV = PROJECT_ROOT / "reports" / "risk_threshold_sweep_validation.csv"
RISK_MODES_TEST_MD = PROJECT_ROOT / "reports" / "risk_modes_test_metrics.md"
HIGH_CONF_MD = PROJECT_ROOT / "reports" / "risk_high_confidence_mode.md"


# ---------------------------------------------------------------------------
# Data loading & feature engineering (copied from run_risk_pipeline.py)
# ---------------------------------------------------------------------------


def load_framingham_splits(
    filepath: Path | str = FRAMINGHAM_PATH,
    test_size: float = 0.30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load Framingham raw data and create deterministic train/val/test splits.

    Split logic matches scripts/run_risk_pipeline.py:
      - Drop rows with missing TenYearCHD
      - 70% train, 30% temp
      - Split temp into 50/50 val/test → 15% each
    """

    df = pd.read_csv(filepath)
    df = df.dropna(subset=["TenYearCHD"])

    X = df.drop("TenYearCHD", axis=1)
    y = df["TenYearCHD"]

    # 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as in run_risk_pipeline.py.

    - Pulse pressure & mean arterial pressure
    - BMI categories
    - Age groups
    - Risk flags (BP, cholesterol, glucose, BMI)
    - Metabolic syndrome count
    - Log transforms of key continuous variables
    """
    X = X.copy()

    # Pulse Pressure
    X["pulse_pressure"] = X["sysBP"] - X["diaBP"]

    # Mean Arterial Pressure
    X["map"] = X["diaBP"] + (X["pulse_pressure"] / 3.0)

    # BMI Categories
    X["bmi_cat"] = pd.cut(
        X["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Under", "Normal", "Over", "Obese"],
    )

    # Age Groups
    X["age_group"] = pd.cut(
        X["age"],
        bins=[20, 39, 49, 59, 69, 100],
        labels=["30s", "40s", "50s", "60s", "70+"],
    )

    # Risk Flags
    X["high_bp"] = ((X["sysBP"] >= 140) | (X["diaBP"] >= 90)).astype(int)
    X["high_chol"] = (X["totChol"] >= 240).astype(int)
    X["high_glucose"] = (X["glucose"] >= 126).astype(int)
    X["high_bmi"] = (X["BMI"] >= 30).astype(int)

    # Metabolic Syndrome Count
    X["metabolic_syndrome"] = (
        X["high_bp"] + X["high_chol"] + X["high_glucose"] + X["high_bmi"]
    )

    # Log transforms
    for col in ["totChol", "glucose", "sysBP", "BMI"]:
        if col in X.columns:
            X[f"log_{col}"] = np.log1p(X[col])

    return X


# ---------------------------------------------------------------------------
# Model loading and probability computation
# ---------------------------------------------------------------------------


def load_risk_model(model_path: Path | str = RISK_MODEL_PATH):
    """Load the frozen risk prediction pipeline from disk.

    The pipeline was trained in scripts/run_risk_pipeline.py and saved as
    models/risk_model_pipeline.pkl. It already contains preprocessing and the
    final classifier. This function only loads it; it never retrains.
    """

    model = joblib.load(model_path)
    return model


def get_val_test_probabilities(model, X_val: pd.DataFrame, X_test: pd.DataFrame,
                               y_val: pd.Series, y_test: pd.Series
                               ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Compute predicted probabilities for validation and test sets.

    I first apply the original feature engineering, then pass the engineered
    features into the frozen risk model pipeline.
    """

    X_val_fe = feature_engineering(X_val)
    X_test_fe = feature_engineering(X_test)

    val_proba = model.predict_proba(X_val_fe)[:, 1]
    test_proba = model.predict_proba(X_test_fe)[:, 1]

    return val_proba, test_proba, y_val, y_test


# ---------------------------------------------------------------------------
# Metrics and threshold sweeps
# ---------------------------------------------------------------------------


def compute_binary_metrics(y_true: pd.Series, y_proba: np.ndarray,
                           threshold: float) -> Dict[str, float]:
    """Compute standard binary classification metrics for a given threshold."""

    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def sweep_thresholds(y_true: pd.Series, y_proba: np.ndarray,
                     start: float = 0.05,
                     end: float = 0.95,
                     step: float = 0.01) -> pd.DataFrame:
    """Run a threshold sweep on the validation set.

    For each threshold in [start, end] with step size, compute accuracy,
    recall, precision, F1 and ROC-AUC and return a DataFrame.
    """

    thresholds = np.arange(start, end + 1e-8, step)
    rows: List[Dict[str, float]] = []

    for t in thresholds:
        rows.append(compute_binary_metrics(y_true, y_proba, float(t)))

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Operating mode selection
# ---------------------------------------------------------------------------


def choose_accuracy_mode(val_metrics: pd.DataFrame,
                         min_recall: float = 0.30) -> Dict[str, float]:
    """Choose Accuracy Mode threshold.

    - Maximize validation accuracy.
    - Subject to recall >= min_recall (to avoid collapsing to all negatives).
    """

    candidates = val_metrics[val_metrics["recall"] >= min_recall]
    if candidates.empty:
        # Fallback: best accuracy overall
        best = val_metrics.sort_values("accuracy", ascending=False).iloc[0]
    else:
        best = candidates.sort_values("accuracy", ascending=False).iloc[0]

    return best.to_dict()


def choose_balanced_mode(val_metrics: pd.DataFrame,
                         min_accuracy: float = 0.85) -> Dict[str, float]:
    """Choose Balanced Mode threshold.

    Heuristic:
    - Filter to thresholds with accuracy >= min_accuracy.
    - Among those, pick the one with highest F1.
    - If none meet the accuracy constraint, pick the global best F1.
    """

    candidates = val_metrics[val_metrics["accuracy"] >= min_accuracy]
    if candidates.empty:
        candidates = val_metrics

    best = candidates.sort_values([
        "f1",
        "recall",
        "accuracy",
    ], ascending=[False, False, False]).iloc[0]

    return best.to_dict()


def evaluate_mode_on_test(y_test: pd.Series, test_proba: np.ndarray,
                          threshold: float) -> Dict[str, float]:
    """Compute metrics for a given threshold on the test set."""

    return compute_binary_metrics(y_test, test_proba, threshold)


def high_confidence_slice(y_test: pd.Series, test_proba: np.ndarray,
                          low: float = 0.10,
                          high: float = 0.90) -> Dict[str, float]:
    """Evaluate a high-confidence slice on the test set.

    - High-confidence negatives: p < low
    - High-confidence positives: p > high
    - Coverage: fraction of test set falling into these bands
    - Accuracy/recall/precision computed on this subset only
    """

    mask = (test_proba < low) | (test_proba > high)
    if not mask.any():
        return {
            "coverage": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    y_true_sub = y_test[mask]
    y_proba_sub = test_proba[mask]
    # Use 0.5 threshold within the high-confidence region
    metrics = compute_binary_metrics(y_true_sub, y_proba_sub, threshold=0.5)
    coverage = float(mask.mean())

    metrics["coverage"] = coverage
    return metrics


# ---------------------------------------------------------------------------
# Report writing helpers
# ---------------------------------------------------------------------------


def write_risk_modes_test_markdown(
    acc_mode: Dict[str, float],
    bal_mode: Dict[str, float],
    acc_test: Dict[str, float],
    bal_test: Dict[str, float],
    path: Path = RISK_MODES_TEST_MD,
) -> None:
    """Write a small markdown report comparing Accuracy & Balanced modes.

    The file includes, for each mode:
      - Mode name
      - Threshold (chosen on validation)
      - Test set Accuracy, Recall, Precision, F1, ROC-AUC
    """

    lines = []
    lines.append("# Risk Model Operating Modes - Test Metrics\n")
    lines.append("\n")
    lines.append("I evaluated my frozen risk prediction model under two operating modes on the held-out test set. I did not retrain the model; I only changed the decision threshold on top of the same probability outputs.\n")
    lines.append("\n")
    lines.append("| Mode | Threshold | Accuracy | Recall | Precision | F1 | ROC-AUC |\n")
    lines.append("|------|-----------|----------|--------|-----------|----|---------|\n")

    def row(name: str, mode_info: Dict[str, float], test_info: Dict[str, float]) -> str:
        return (
            f"| {name} | "
            f"{mode_info['threshold']:.3f} | "
            f"{test_info['accuracy']:.4f} | "
            f"{test_info['recall']:.4f} | "
            f"{test_info['precision']:.4f} | "
            f"{test_info['f1']:.4f} | "
            f"{test_info['roc_auc']:.4f} |"
        )

    lines.append(row("Accuracy Mode", acc_mode, acc_test) + "\n")
    lines.append(row("Balanced Mode", bal_mode, bal_test) + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def write_high_confidence_markdown(
    high_conf: Dict[str, float],
    low: float = 0.10,
    high: float = 0.90,
    path: Path = HIGH_CONF_MD,
) -> None:
    """Write a short markdown explanation for the high-confidence mode.

    This is a first-person description that clarifies this mode is a
    high-confidence subset analysis, not the overall test accuracy.
    """

    coverage_pct = high_conf.get("coverage", 0.0) * 100.0

    lines = []
    lines.append("# High-Confidence Risk Mode (Subset Analysis)\n\n")
    lines.append(
        f"In this analysis, I kept the same frozen risk prediction model and only looked at a high-confidence subset of the test set. "
        f"I defined **high-confidence negatives** as patients with predicted risk lower than {low:.2f} and **high-confidence positives** as patients with predicted risk higher than {high:.2f}. "
        "I ignored all patients in the middle region for this specific slice.\n\n"
    )

    lines.append("## Metrics on the High-Confidence Subset\n\n")
    lines.append(f"- **Coverage:** About {coverage_pct:.1f}% of the test patients fall into this high-confidence band.\n")
    lines.append(f"- **Accuracy:** {high_conf.get('accuracy', 0.0):.4f}\n")
    lines.append(f"- **Recall:** {high_conf.get('recall', 0.0):.4f}\n")
    lines.append(f"- **Precision:** {high_conf.get('precision', 0.0):.4f}\n")
    lines.append(f"- **F1 Score:** {high_conf.get('f1', 0.0):.4f}\n\n")

    lines.append(
        "This mode is useful when I want very reliable predictions and I am willing to accept that some patients are left unclassified in this view. "
        "The accuracy in this subset can be higher than the overall test accuracy because I only keep patients where the model is very confident (either clearly low-risk or clearly high-risk). "
        "However, this comes with a trade-off in **coverage**: the high-confidence mode is not meant to replace the full model, only to provide an additional safety band where predictions are especially trustworthy.\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Convenience runner (for CLI use)
# ---------------------------------------------------------------------------


def run_threshold_analysis() -> None:
    """Run full threshold analysis and write standard report files.

    This function is safe: it only loads the frozen model and evaluates it.
    It never retrains or overwrites model artifacts.
    """

    print("=" * 80)
    print("RISK THRESHOLD ANALYSIS - USING FROZEN PIPELINE")
    print("=" * 80)

    # 1. Load data splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_framingham_splits()
    print(f"Train size: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 2. Load frozen risk model
    model = load_risk_model()
    print(f"Loaded risk model from: {RISK_MODEL_PATH}")

    # 3. Get probabilities
    val_proba, test_proba, y_val, y_test = get_val_test_probabilities(
        model, X_val, X_test, y_val, y_test
    )

    # 4. Threshold sweep on validation
    val_metrics = sweep_thresholds(y_val, val_proba, start=0.05, end=0.95, step=0.01)
    THRESH_SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    val_metrics.to_csv(THRESH_SWEEP_CSV, index=False)
    print(f"Saved validation threshold sweep to {THRESH_SWEEP_CSV}")

    # 5. Choose modes on validation
    acc_mode = choose_accuracy_mode(val_metrics, min_recall=0.30)
    bal_mode = choose_balanced_mode(val_metrics, min_accuracy=0.85)

    print("\nAccuracy Mode (validation):")
    print(acc_mode)

    print("\nBalanced Mode (validation):")
    print(bal_mode)

    # 6. Evaluate both modes on test
    acc_test = evaluate_mode_on_test(y_test, test_proba, threshold=acc_mode["threshold"])
    bal_test = evaluate_mode_on_test(y_test, test_proba, threshold=bal_mode["threshold"])

    print("\nAccuracy Mode (test):")
    print(acc_test)

    print("\nBalanced Mode (test):")
    print(bal_test)

    # 7. High-confidence slice
    high_conf = high_confidence_slice(y_test, test_proba, low=0.10, high=0.90)
    print("\nHigh-Confidence Slice (test):")
    print(high_conf)

    # 8. Write markdown reports
    write_risk_modes_test_markdown(acc_mode, bal_mode, acc_test, bal_test)
    write_high_confidence_markdown(high_conf, low=0.10, high=0.90)

    print(f"\nSaved test mode metrics to: {RISK_MODES_TEST_MD}")
    print(f"Saved high-confidence analysis to: {HIGH_CONF_MD}")


if __name__ == "__main__":
    run_threshold_analysis()
