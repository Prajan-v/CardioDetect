from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


DATA_SPLIT_DIR = Path(__file__).parent.parent / "data" / "split"
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_3class_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = DATA_SPLIT_DIR / "train_3class.csv"
    val_path = DATA_SPLIT_DIR / "val_3class.csv"
    test_path = DATA_SPLIT_DIR / "test_3class.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("3-class split files not found. Run create_3class_labels.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
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
    """One-hot encode categorical features consistently across splits."""
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


def compute_class_weights(y: pd.Series) -> np.ndarray:
    classes = np.array(sorted(y.unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return weights


def train_model() -> None:
    print("Loading 3-class data splits...")
    train_df, val_df, test_df = load_3class_splits()

    X_train_raw, y_train = prepare_features_and_target(train_df)
    X_val_raw, y_val = prepare_features_and_target(val_df)
    X_test_raw, y_test = prepare_features_and_target(test_df)

    print("Encoding categorical features...")
    X_train_enc, X_val_enc, X_test_enc = encode_categorical_features_3class(
        X_train_raw, X_val_raw, X_test_raw
    )

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_val_scaled = scaler.transform(X_val_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    print("Computing class weights...")
    class_weights = compute_class_weights(y_train)
    classes = np.array(sorted(y_train.unique()))
    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}
    sample_weight = y_train.map(class_weight_dict).values

    print("Training 3-class MLP model (mlp_v3_3class)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        random_state=42,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
    )

    mlp.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    print("Evaluating on validation and test sets...")
    y_val_pred = mlp.predict(X_val_scaled)
    y_test_pred = mlp.predict(X_test_scaled)

    target_names = ["LOW", "MEDIUM", "HIGH"]

    print("\n=== Classification Report (Validation) ===")
    print(classification_report(y_val, y_val_pred, target_names=target_names, digits=4))

    print("\n=== Classification Report (Test) ===")
    print(classification_report(y_test, y_test_pred, target_names=target_names, digits=4))

    print("\n=== Confusion Matrix (Test) ===")
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2])
    print("Rows = true, Cols = predicted (0=LOW,1=MEDIUM,2=HIGH)")
    print(cm)

    print("\n=== Per-Class Accuracy (Test) ===")
    for cls, name in enumerate(target_names):
        mask = y_test == cls
        if mask.sum() == 0:
            acc = float("nan")
        else:
            acc = float((y_test_pred[mask] == y_test[mask]).mean())
        print(f"  {name} (class {cls}): {acc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mlp_v3_3class.pkl"

    artifact = {
        "model": mlp,
        "scaler": scaler,
        "feature_names": X_train_enc.columns.tolist(),
        "class_names": target_names,
    }

    joblib.dump(artifact, model_path)
    print(f"\nSaved 3-class model to: {model_path}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
