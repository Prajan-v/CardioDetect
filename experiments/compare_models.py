from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlp_tuning import load_splits, encode_categorical_features, MLP_V2_PATH


DATA_SPLIT_DIR = Path(__file__).parent.parent / "data" / "split"
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_binary_model() -> Tuple[object, object, list | None]:
    """Load binary mlp_v2 model artifact."""
    artifact = joblib.load(MLP_V2_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact.get("feature_names")
    return model, scaler, feature_names


def load_3class_model() -> Tuple[object, object, list, list]:
    """Load 3-class mlp_v3_3class model artifact."""
    model_path = MODELS_DIR / "mlp_v3_3class.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"3-class model not found: {model_path}. Run train_3class_model.py first.")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    class_names = artifact.get("class_names", ["LOW", "MEDIUM", "HIGH"])
    return model, scaler, feature_names, class_names


def prepare_features_and_target_3class(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare X/y for 3-class model from *_3class.csv dataframe."""
    y = df["risk_class_3"].astype(int)

    drop_cols = [
        "risk_class_3",
        "risk_target",
        "target",
        "TenYearCHD",
        "data_source",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y


def encode_categorical_features_3class(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical features consistently across splits.

    This mirrors the logic used in train_3class_model.py.
    """
    combined = pd.concat(
        [X_train, X_val, X_test],
        keys=["train", "val", "test"],
        names=["split", None],
    )

    categorical_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        return X_train, X_val, X_test

    combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

    X_train_enc = combined_encoded.xs("train")
    X_val_enc = combined_encoded.xs("val")
    X_test_enc = combined_encoded.xs("test")

    return X_train_enc, X_val_enc, X_test_enc


def accuracy_by_class(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, float]:
    """Compute per-class accuracy (recall) for each class 0,1,2."""
    metrics: Dict[int, float] = {}
    for cls in [0, 1, 2]:
        mask = y_true == cls
        if mask.sum() == 0:
            metrics[cls] = float("nan")
        else:
            metrics[cls] = float((y_pred[mask] == y_true[mask]).mean())
    return metrics


def evaluate_binary_model() -> Tuple[float, Dict[int, float]]:
    """Evaluate binary mlp_v2 as a 3-class risk-band classifier."""
    print("Loading binary mlp_v2 and test data...")
    model, scaler, feature_names = load_binary_model()

    # Load original Framingham-style splits to build features exactly as in training
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    X_train_enc, X_val_enc, X_test_enc = encode_categorical_features(X_train, X_val, X_test)

    # Align columns with what the model expects, if provided
    if feature_names is not None:
        missing = set(feature_names) - set(X_test_enc.columns)
        if missing:
            raise ValueError(f"X_test is missing expected features for mlp_v2: {sorted(missing)}")
        X_test_enc = X_test_enc[feature_names]

    X_test_scaled = scaler.transform(X_test_enc)

    # Load 3-class labels
    test_3_path = DATA_SPLIT_DIR / "test_3class.csv"
    if not test_3_path.exists():
        raise FileNotFoundError("test_3class.csv not found. Run create_3class_labels.py first.")

    test_3 = pd.read_csv(test_3_path)
    y_true_3 = test_3["risk_class_3"].astype(int).values

    if len(y_true_3) != X_test_scaled.shape[0]:
        raise ValueError("Length mismatch between test features and 3-class labels.")

    # Predict event probabilities and convert to risk bands
    proba = model.predict_proba(X_test_scaled)[:, 1]
    # <0.10 -> LOW(0), 0.10-0.20 -> MEDIUM(1), >0.20 -> HIGH(2)
    y_pred_band = np.where(proba < 0.10, 0, np.where(proba <= 0.20, 1, 2))

    overall_acc = float((y_pred_band == y_true_3).mean())
    per_class_acc = accuracy_by_class(y_true_3, y_pred_band)

    print("\n=== Binary mlp_v2 as 3-class classifier ===")
    print(f"Overall 3-class accuracy: {overall_acc:.4f}")
    for cls, name in zip([0, 1, 2], ["LOW", "MEDIUM", "HIGH"]):
        print(f"  {name} (class {cls}) accuracy: {per_class_acc[cls]:.4f}")

    return overall_acc, per_class_acc


def evaluate_3class_model() -> Tuple[float, Dict[int, float]]:
    """Evaluate direct 3-class mlp_v3_3class model."""
    print("\nLoading 3-class mlp_v3_3class and 3-class data splits...")
    model, scaler, feature_names, class_names = load_3class_model()

    train_df = pd.read_csv(DATA_SPLIT_DIR / "train_3class.csv")
    val_df = pd.read_csv(DATA_SPLIT_DIR / "val_3class.csv")
    test_df = pd.read_csv(DATA_SPLIT_DIR / "test_3class.csv")

    X_train_raw, y_train = prepare_features_and_target_3class(train_df)
    X_val_raw, y_val = prepare_features_and_target_3class(val_df)
    X_test_raw, y_test = prepare_features_and_target_3class(test_df)

    X_train_enc, X_val_enc, X_test_enc = encode_categorical_features_3class(
        X_train_raw, X_val_raw, X_test_raw
    )

    # Align with training feature order
    missing = set(feature_names) - set(X_test_enc.columns)
    if missing:
        raise ValueError(f"X_test is missing expected features for mlp_v3: {sorted(missing)}")
    X_test_enc = X_test_enc[feature_names]

    X_test_scaled = scaler.transform(X_test_enc)
    y_pred = model.predict(X_test_scaled)

    overall_acc = float((y_pred == y_test.values).mean())
    per_class_acc = accuracy_by_class(y_test.values, y_pred)

    print("\n=== Direct 3-class mlp_v3_3class ===")
    print(f"Overall 3-class accuracy: {overall_acc:.4f}")
    for cls, name in zip([0, 1, 2], class_names):
        print(f"  {name} (class {cls}) accuracy: {per_class_acc[cls]:.4f}")

    print("\nConfusion Matrix (Test) - 3-class model")
    cm = confusion_matrix(y_test.values, y_pred, labels=[0, 1, 2])
    print("Rows = true, Cols = predicted (0=LOW,1=MEDIUM,2=HIGH)")
    print(cm)

    return overall_acc, per_class_acc


def print_comparison_table(
    overall_v2: float,
    per_class_v2: Dict[int, float],
    overall_v3: float,
    per_class_v3: Dict[int, float],
) -> None:
    print("\n\nSIDE-BY-SIDE COMPARISON")
    print("=" * 70)

    def winner(v2: float, v3: float) -> str:
        if np.isnan(v2) and np.isnan(v3):
            return "-"
        if v2 > v3:
            return "v2"
        if v3 > v2:
            return "v3"
        return "tie"

    header = f"{'Metric':<30} {'Binary mlp_v2':>15} {'3-class mlp_v3':>18} {'Winner':>10}"
    print(header)
    print("-" * len(header))

    # Overall
    print(f"{'Overall 3-class accuracy':<30} {overall_v2:15.4f} {overall_v3:18.4f} {winner(overall_v2, overall_v3):>10}")

    # Per class
    labels = [(0, "LOW"), (1, "MEDIUM"), (2, "HIGH")]
    for cls, name in labels:
        v2 = per_class_v2.get(cls, float("nan"))
        v3 = per_class_v3.get(cls, float("nan"))
        metric_name = f"{name} accuracy"
        print(f"{metric_name:<30} {v2:15.4f} {v3:18.4f} {winner(v2, v3):>10}")

    # Highlight MEDIUM improvement
    v2_med = per_class_v2.get(1, 0.0)
    v3_med = per_class_v3.get(1, 0.0)
    improvement = v3_med - v2_med

    print("\nKEY FINDING: MEDIUM RISK CLASSIFICATION")
    print(f"Binary model (v2) MEDIUM accuracy: {v2_med*100:.1f}%")
    print(f"3-Class model (v3) MEDIUM accuracy: {v3_med*100:.1f}%")
    print(f"Improvement: {improvement*100:.1f} percentage points")


def main() -> None:
    overall_v2, per_class_v2 = evaluate_binary_model()
    overall_v3, per_class_v3 = evaluate_3class_model()
    print_comparison_table(overall_v2, per_class_v2, overall_v3, per_class_v3)


if __name__ == "__main__":
    main()
