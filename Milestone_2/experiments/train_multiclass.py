"""
Train calibrated 3-class classifiers.

Trains HGB, RF, and MLP classifiers on 3-class risk labels (LOW/MODERATE/HIGH)
with optional probability calibration using isotonic regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.preprocessing import load_splits, get_feature_target_split, build_preprocessor_from_data
from src.targets import add_targets_to_df
from src.models_multiclass import (
    build_hgb_multiclass,
    build_rf_multiclass,
    build_mlp_multiclass,
    calibrate_classifier,
    tune_multiclass,
    get_multiclass_param_grids,
)
from src.evaluation import (
    compute_classification_metrics,
    compute_per_class_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
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
    "calibrate": True,
    "calibration_method": "isotonic",
    "calibration_cv": 3,
}

TARGET_NAMES = ["LOW", "MODERATE", "HIGH"]


# ============================================================================
# MAIN TRAINING ROUTINE
# ============================================================================

def train_multiclass():
    """Train calibrated 3-class classifiers."""
    print("=" * 80)
    print("MILESTONE 2: TRAINING CALIBRATED 3-CLASS CLASSIFIERS")
    print("=" * 80)

    run_id = create_run_id("multiclass")
    print(f"Run ID: {run_id}")

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    train_df, val_df, test_df = load_splits()

    # Add guideline risk and derived targets
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

    X_train, _ = get_feature_target_split(train_df, target_col="risk_class_3")
    X_val, _ = get_feature_target_split(val_df, target_col="risk_class_3")
    X_test, _ = get_feature_target_split(test_df, target_col="risk_class_3")

    # Remove target columns from features
    drop_cols = ["guideline_risk_10yr", "risk_class_3", "risk_class_binary"]
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_val = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # Targets
    y_train = train_df["risk_class_3"].values
    y_val = val_df["risk_class_3"].values
    y_test = test_df["risk_class_3"].values

    # Build preprocessor
    preprocessor = build_preprocessor_from_data(X_train)

    print(f"Features: {X_train.shape[1]}")
    print(f"Class distribution (train): {np.bincount(y_train)}")

    # ------------------------------------------------------------------
    # Define models
    # ------------------------------------------------------------------
    print("\n[3/5] Building multiclass models...")

    models = {
        "hgb_multiclass": build_hgb_multiclass(preprocessor),
        "rf_multiclass": build_rf_multiclass(preprocessor),
        "mlp_multiclass": build_mlp_multiclass(preprocessor),
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

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if CONFIG["tune_models"]:
            param_grid = get_multiclass_param_grids().get(name, {})
            if param_grid:
                model, _, _ = tune_multiclass(
                    model, X_train, y_train,
                    param_grid, CONFIG["tune_n_iter"], CONFIG["tune_cv"]
                )
        else:
            model.fit(X_train, y_train)

        # Optionally calibrate
        if CONFIG["calibrate"]:
            print(f"  Calibrating with {CONFIG['calibration_method']}...")
            model = calibrate_classifier(
                model, X_train, y_train,
                CONFIG["calibration_method"], CONFIG["calibration_cv"]
            )
            name = f"{name}_calibrated"

        trained_models[name] = model

        # Evaluate on validation
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        metrics = compute_classification_metrics(y_val, y_pred, y_proba)
        metrics["model_name"] = name
        metrics["target_type"] = "3class"
        metrics["calibrated"] = CONFIG["calibrate"]

        print(f"  Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Val F1 (macro): {metrics['f1']:.4f}")
        print(f"  Val Recall (class 2/HIGH): {metrics.get('recall_class_2', 'N/A')}")
        print(compute_per_class_report(y_val, y_pred, TARGET_NAMES))

        # Save model and metrics
        save_model_artifact(model, name, {"run_id": run_id, **metrics})
        save_metrics(metrics, name)

        # Plots
        plot_confusion_matrix(
            y_val, y_pred, TARGET_NAMES,
            f"{name} - Confusion Matrix",
            reports_dir / f"{name}_confusion_matrix.png"
        )

        plot_roc_curve(
            y_val, y_proba,
            f"{name} - ROC Curve",
            reports_dir / f"{name}_roc_curve.png"
        )

        plot_pr_curve(
            y_val, y_proba,
            f"{name} - PR Curve",
            reports_dir / f"{name}_pr_curve.png"
        )

        # Per-class calibration curves
        for cls_idx, cls_name in enumerate(TARGET_NAMES):
            y_val_cls = (y_val == cls_idx).astype(int)
            plot_calibration_curve(
                y_val_cls, y_proba[:, cls_idx], n_bins=10,
                title=f"{name} - Calibration ({cls_name})",
                save_path=calibration_dir / f"{name}_calibration_{cls_name.lower()}.png"
            )

        results.append(metrics)

    # ------------------------------------------------------------------
    # Evaluate best model on test set
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating best model on test set...")

    best_result = max(results, key=lambda x: x["f1"])
    best_model_name = best_result["model_name"]
    best_model = trained_models[best_model_name]

    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)

    test_metrics = compute_classification_metrics(y_test, y_pred_test, y_proba_test)

    print(f"\nBest model: {best_model_name}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1']:.4f}")
    print(compute_per_class_report(y_test, y_pred_test, TARGET_NAMES))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"Trained {len(results)} multiclass models")
    print(f"Best model: {best_model_name} (F1={best_result['f1']:.4f})")
    print(f"Models saved to: milestone_2/models/")
    print(f"Metrics saved to: milestone_2/reports/metrics/")
    print("=" * 80)

    return results, trained_models


if __name__ == "__main__":
    train_multiclass()
