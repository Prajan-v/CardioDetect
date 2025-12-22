"""Train a guideline-based 10-year cardiovascular risk regressor.

This script:
- Loads the preprocessed CardioDetect train/val/test splits.
- Computes Framingham-like 10-year CVD risk (guideline_risk_10yr) for each row.
- Trains a HistGradientBoostingRegressor to approximate this continuous risk.
- Evaluates performance (continuous + binned LOW/MODERATE/HIGH).
- Saves the trained model artifact to models/risk_regressor_v2.pkl.

The artifact is intended for later integration into CardioDetect V3 as a
"regressor + bins" risk model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)

from src.guideline_risk import framingham_10yr_risk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "split"
MODELS_DIR = PROJECT_ROOT / "models"


def _compute_guideline_risk(df: pd.DataFrame) -> pd.Series:
    """Compute guideline-based 10-year CVD risk for each row.

    Uses the framingham_10yr_risk utility. Works on the *preprocessed* dataset
    (train/val/test CSVs) and is robust to missing HDL or medication columns.
    """

    def row_to_risk(row: pd.Series) -> float:
        return framingham_10yr_risk(
            age=row.get("age"),
            sex=row.get("sex"),
            total_cholesterol=row.get("total_cholesterol"),
            hdl_cholesterol=row.get("hdl_cholesterol", None),
            systolic_bp=row.get("systolic_bp"),
            on_treatment=bool(row.get("bp_meds", 0) or row.get("hypertension", 0)),
            smoking=row.get("smoking"),
            diabetes=row.get("diabetes"),
        )

    risks = df.apply(row_to_risk, axis=1)
    return risks.astype(float)


def _make_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features X and continuous target y.

    Expects df to already contain a 'guideline_risk_10yr' column.
    """
    y = df["guideline_risk_10yr"].astype(float)
    drop_cols = ["risk_target", "data_source", "guideline_risk_10yr"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


def _bin_risk(risk: np.ndarray, low_thr: float = 10.0, high_thr: float = 25.0) -> np.ndarray:
    """Bin continuous risk (in %) into LOW/MODERATE/HIGH indices 0/1/2."""
    risk = np.asarray(risk, dtype=float)
    labels = np.zeros_like(risk, dtype=int)
    labels[(risk >= low_thr) & (risk < high_thr)] = 1
    labels[risk >= high_thr] = 2
    return labels


def train_guideline_regressor() -> None:
    """Main training routine for guideline-based regressor."""
    print("=" * 80)
    print("TRAINING GUIDELINE-BASED 10-YEAR RISK REGRESSOR (v2)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load splits
    # ------------------------------------------------------------------
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected train/val/test CSVs in data/split. Run data_preprocessing first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")

    # ------------------------------------------------------------------
    # Compute guideline risk on each split
    # ------------------------------------------------------------------
    print("\nComputing guideline 10-year risk on train/val/test...")
    train_df["guideline_risk_10yr"] = _compute_guideline_risk(train_df)
    val_df["guideline_risk_10yr"] = _compute_guideline_risk(val_df)
    test_df["guideline_risk_10yr"] = _compute_guideline_risk(test_df)

    # Drop rows where risk is NaN (insufficient inputs)
    def _drop_nan(df: pd.DataFrame, name: str) -> pd.DataFrame:
        before = len(df)
        df = df[df["guideline_risk_10yr"].notna()].copy()
        after = len(df)
        print(f"  {name}: dropped {before - after} rows with NaN guideline risk (remaining {after})")
        return df

    train_df = _drop_nan(train_df, "train")
    val_df = _drop_nan(val_df, "val")
    test_df = _drop_nan(test_df, "test")

    print("\nGuideline risk summary (train):")
    print(train_df["guideline_risk_10yr"].describe())

    # ------------------------------------------------------------------
    # Build X/y
    # ------------------------------------------------------------------
    X_train, y_train = _make_xy(train_df)
    X_val, y_val = _make_xy(val_df)
    X_test, y_test = _make_xy(test_df)

    feature_names = X_train.columns.tolist()
    print(f"\nUsing {len(feature_names)} features for regression.")

    # ------------------------------------------------------------------
    # Sample weights to emphasize MODERATE/HIGH
    # ------------------------------------------------------------------
    risk_pct_train = (y_train.values * 100.0).astype(float)
    sample_weight = np.ones_like(risk_pct_train, dtype=float)
    sample_weight[(risk_pct_train >= 10.0) & (risk_pct_train < 25.0)] *= 1.5
    sample_weight[risk_pct_train >= 25.0] *= 2.0

    # ------------------------------------------------------------------
    # Model definition
    # ------------------------------------------------------------------
    reg = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=1.0,
        random_state=42,
    )

    print("\nFitting HistGradientBoostingRegressor...")
    reg.fit(X_train, y_train, sample_weight=sample_weight)

    # ------------------------------------------------------------------
    # Evaluation: continuous metrics
    # ------------------------------------------------------------------
    def _eval_split(name: str, X: pd.DataFrame, y_true: pd.Series) -> None:
        y_pred = reg.predict(X)
        mae = mean_absolute_error(y_true, y_pred)
        # Older sklearn in this environment does not support squared=False,
        # so compute RMSE manually from plain MSE.
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, y_pred)
        brier = np.mean((y_pred - y_true.values) ** 2)

        print(f"\n{name} continuous metrics:")
        print(f"  MAE:   {mae:.4f}")
        print(f"  RMSE:  {rmse:.4f}")
        print(f"  R^2:   {r2:.4f}")
        print(f"  Brier: {brier:.4f}")

        # Binned LOW/MOD/HIGH
        true_pct = y_true.values * 100.0
        pred_pct = y_pred * 100.0
        y_true_bins = _bin_risk(true_pct)
        y_pred_bins = _bin_risk(pred_pct)

        cm = confusion_matrix(y_true_bins, y_pred_bins, labels=[0, 1, 2])
        print(f"\n{name} binned LOW/MODERATE/HIGH confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("\n" + classification_report(y_true_bins, y_pred_bins, digits=3))

    _eval_split("VAL", X_val, y_val)
    _eval_split("TEST", X_test, y_test)

    # ------------------------------------------------------------------
    # Save artifact
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODELS_DIR / "risk_regressor_v2.pkl"

    artifact = {
        "model": reg,
        "feature_names": feature_names,
        "threshold_low": 10.0,
        "threshold_high": 25.0,
        "version": "guideline_regressor_v2",
    }

    joblib.dump(artifact, artifact_path)
    print(f"\nSaved guideline regressor artifact to: {artifact_path}")


if __name__ == "__main__":
    train_guideline_regressor()
