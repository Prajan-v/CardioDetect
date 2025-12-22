"""
Train guideline-based risk regressors.

Trains HistGradientBoostingRegressor, RandomForestRegressor, and MLPRegressor
to predict continuous 10-year CVD risk. Evaluates both continuous metrics
and binned (LOW/MODERATE/HIGH) classification metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.preprocessing import load_splits, get_feature_target_split, build_preprocessor_from_data
from src.targets import add_targets_to_df, to_three_class
from src.models_regressor import (
    build_hgb_regressor,
    build_rf_regressor,
    build_mlp_regressor,
    compute_sample_weights,
    tune_regressor,
    get_regressor_param_grids,
)
from src.evaluation import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_per_class_report,
    plot_confusion_matrix,
    plot_calibration_curve,
)
from src.utils_io import save_model_artifact, save_metrics, create_run_id


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "tune_models": False,
    "tune_n_iter": 10,
    "tune_cv": 3,
    "use_sample_weights": True,
    "low_threshold": 0.10,
    "high_threshold": 0.25,
}

TARGET_NAMES = ["LOW", "MODERATE", "HIGH"]


# ============================================================================
# MAIN TRAINING ROUTINE
# ============================================================================

def train_regressors():
    """Train all guideline risk regressors."""
    print("=" * 80)
    print("MILESTONE 2: TRAINING GUIDELINE RISK REGRESSORS")
    print("=" * 80)

    run_id = create_run_id("regressors")
    print(f"Run ID: {run_id}")

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    train_df, val_df, test_df = load_splits()

    # Add guideline risk
    train_df = add_targets_to_df(train_df)
    val_df = add_targets_to_df(val_df)
    test_df = add_targets_to_df(test_df)

    # Drop rows with NaN guideline risk
    train_df = train_df[train_df["guideline_risk_10yr"].notna()].copy()
    val_df = val_df[val_df["guideline_risk_10yr"].notna()].copy()
    test_df = test_df[test_df["guideline_risk_10yr"].notna()].copy()

    print(f"After filtering: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ------------------------------------------------------------------
    # Prepare features
    # ------------------------------------------------------------------
    print("\n[2/5] Preparing features...")

    X_train, _ = get_feature_target_split(train_df, target_col="guideline_risk_10yr")
    X_val, _ = get_feature_target_split(val_df, target_col="guideline_risk_10yr")
    X_test, _ = get_feature_target_split(test_df, target_col="guideline_risk_10yr")

    # Remove target columns from features
    drop_cols = ["guideline_risk_10yr", "risk_class_3", "risk_class_binary"]
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_val = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # Continuous target
    y_train = train_df["guideline_risk_10yr"].values
    y_val = val_df["guideline_risk_10yr"].values
    y_test = test_df["guideline_risk_10yr"].values

    # Build preprocessor
    preprocessor = build_preprocessor_from_data(X_train)

    # Sample weights
    sample_weight = None
    if CONFIG["use_sample_weights"]:
        sample_weight = compute_sample_weights(
            y_train,
            CONFIG["low_threshold"],
            CONFIG["high_threshold"],
        )
        print(f"Using sample weights (sum={sample_weight.sum():.1f})")

    print(f"Features: {X_train.shape[1]}")
    print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"Target mean: {y_train.mean():.3f}")

    # ------------------------------------------------------------------
    # Define models
    # ------------------------------------------------------------------
    print("\n[3/5] Building regressor models...")

    models = {
        "hgb_regressor": build_hgb_regressor(preprocessor),
        "rf_regressor": build_rf_regressor(preprocessor),
        "mlp_regressor": build_mlp_regressor(preprocessor),
    }

    # ------------------------------------------------------------------
    # Train and evaluate
    # ------------------------------------------------------------------
    print("\n[4/5] Training and evaluating models...")

    results = []
    reports_dir = Path(__file__).resolve().parents[1] / "reports" / "metrics"
    calibration_dir = Path(__file__).resolve().parents[1] / "reports" / "calibration"
    reports_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if CONFIG["tune_models"]:
            param_grid = get_regressor_param_grids().get(name, {})
            if param_grid:
                model, _, _ = tune_regressor(
                    model, X_train, y_train,
                    param_grid, sample_weight,
                    CONFIG["tune_n_iter"], CONFIG["tune_cv"]
                )
        else:
            fit_params = {}
            if sample_weight is not None:
                fit_params["reg__sample_weight"] = sample_weight
            model.fit(X_train, y_train, **fit_params)

        # Evaluate on validation - continuous metrics
        y_pred_val = model.predict(X_val)
        cont_metrics = compute_regression_metrics(y_val, y_pred_val)

        print(f"  Val MAE:  {cont_metrics['mae']:.4f}")
        print(f"  Val RMSE: {cont_metrics['rmse']:.4f}")
        print(f"  Val R^2:  {cont_metrics['r2']:.4f}")

        # Binned classification metrics
        y_val_bins = to_three_class(y_val, CONFIG["low_threshold"], CONFIG["high_threshold"])
        y_pred_bins = to_three_class(y_pred_val, CONFIG["low_threshold"], CONFIG["high_threshold"])

        class_metrics = compute_classification_metrics(y_val_bins, y_pred_bins)
        print(f"  Val F1 (macro, binned): {class_metrics['f1']:.4f}")
        print(compute_per_class_report(y_val_bins, y_pred_bins, TARGET_NAMES))

        # Combine metrics
        metrics = {
            "model_name": name,
            "target_type": "continuous",
            **cont_metrics,
            "binned_f1": class_metrics["f1"],
            "binned_accuracy": class_metrics["accuracy"],
        }

        # Save model and metrics
        artifact_meta = {
            "run_id": run_id,
            "feature_names": X_train.columns.tolist(),
            "threshold_low": CONFIG["low_threshold"] * 100,
            "threshold_high": CONFIG["high_threshold"] * 100,
            **metrics,
        }
        save_model_artifact(model, name, artifact_meta)
        save_metrics(metrics, name)

        # Plots
        plot_confusion_matrix(
            y_val_bins, y_pred_bins, TARGET_NAMES,
            f"{name} - Binned Confusion Matrix",
            reports_dir / f"{name}_confusion_matrix.png"
        )

        # Calibration curve (for each bin boundary)
        # LOW vs rest
        y_val_low = (y_val_bins == 0).astype(int)
        pred_low = 1.0 - y_pred_val / CONFIG["low_threshold"]  # rough inverse
        pred_low = np.clip(pred_low, 0, 1)
        plot_calibration_curve(
            y_val_low, pred_low, n_bins=10,
            title=f"{name} - Calibration (LOW class)",
            save_path=calibration_dir / f"{name}_calibration_low.png"
        )

        results.append(metrics)

    # ------------------------------------------------------------------
    # Evaluate best model on test set
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating best model on test set...")

    best_model_name = min(results, key=lambda x: x["mae"])["model_name"]
    best_model = models[best_model_name]

    y_pred_test = best_model.predict(X_test)
    test_metrics = compute_regression_metrics(y_test, y_pred_test)

    y_test_bins = to_three_class(y_test, CONFIG["low_threshold"], CONFIG["high_threshold"])
    y_pred_test_bins = to_three_class(y_pred_test, CONFIG["low_threshold"], CONFIG["high_threshold"])

    test_class_metrics = compute_classification_metrics(y_test_bins, y_pred_test_bins)

    print(f"\nBest model: {best_model_name}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R^2:  {test_metrics['r2']:.4f}")
    print(f"Test F1 (binned): {test_class_metrics['f1']:.4f}")
    print(compute_per_class_report(y_test_bins, y_pred_test_bins, TARGET_NAMES))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"Trained {len(results)} regressor models")
    print(f"Best model: {best_model_name}")
    print(f"Models saved to: milestone_2/models/")
    print(f"Metrics saved to: milestone_2/reports/metrics/")
    print("=" * 80)

    return results


if __name__ == "__main__":
    train_regressors()
