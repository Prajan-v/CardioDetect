"""
Train baseline classification models.

Trains LogisticRegression, RandomForest, SVM, and MLP on both binary
and 3-class targets. Saves models and evaluation metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.preprocessing import load_splits, get_feature_target_split, build_preprocessor_from_data
from src.targets import add_targets_to_df
from src.models_baselines import (
    build_logistic_regression,
    build_random_forest,
    build_svm,
    build_mlp_classifier,
    tune_model,
    get_param_grids,
)
from src.evaluation import (
    compute_classification_metrics,
    compute_per_class_report,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.utils_io import save_model_artifact, save_metrics, create_run_id


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "tune_models": False,  # Set True to run hyperparameter tuning
    "tune_n_iter": 10,
    "tune_cv": 3,
    "random_state": 42,
}

TARGET_NAMES_3CLASS = ["LOW", "MODERATE", "HIGH"]
TARGET_NAMES_BINARY = ["LOW", "AT_RISK"]


# ============================================================================
# MAIN TRAINING ROUTINE
# ============================================================================

def train_baselines():
    """Train all baseline models on binary and 3-class targets."""
    print("=" * 80)
    print("MILESTONE 2: TRAINING BASELINE CLASSIFIERS")
    print("=" * 80)

    run_id = create_run_id("baselines")
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
    y_train_3class = train_df["risk_class_3"].values
    y_val_3class = val_df["risk_class_3"].values
    y_test_3class = test_df["risk_class_3"].values

    y_train_binary = train_df["risk_class_binary"].values
    y_val_binary = val_df["risk_class_binary"].values
    y_test_binary = test_df["risk_class_binary"].values

    # Build preprocessor
    preprocessor = build_preprocessor_from_data(X_train)

    print(f"Features: {X_train.shape[1]}")
    print(f"3-class distribution (train): {np.bincount(y_train_3class)}")
    print(f"Binary distribution (train): {np.bincount(y_train_binary)}")

    # ------------------------------------------------------------------
    # Define models
    # ------------------------------------------------------------------
    print("\n[3/5] Building baseline models...")

    models_3class = {
        "lr_3class": build_logistic_regression(preprocessor),
        "rf_3class": build_random_forest(preprocessor),
        "svm_3class": build_svm(preprocessor),
        "mlp_3class": build_mlp_classifier(preprocessor),
    }

    models_binary = {
        "lr_binary": build_logistic_regression(preprocessor),
        "rf_binary": build_random_forest(preprocessor),
        "svm_binary": build_svm(preprocessor),
        "mlp_binary": build_mlp_classifier(preprocessor),
    }

    # ------------------------------------------------------------------
    # Train and evaluate
    # ------------------------------------------------------------------
    print("\n[4/5] Training and evaluating models...")

    results = []
    reports_dir = Path(__file__).resolve().parents[1] / "reports" / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Train 3-class models
    print("\n--- 3-CLASS MODELS ---")
    for name, model in models_3class.items():
        print(f"\nTraining {name}...")

        if CONFIG["tune_models"]:
            base_name = name.replace("_3class", "")
            param_grid = get_param_grids().get(base_name.replace("_", "_"), {})
            if param_grid:
                model, _, _ = tune_model(
                    model, X_train, y_train_3class,
                    param_grid, CONFIG["tune_n_iter"], CONFIG["tune_cv"]
                )
        else:
            model.fit(X_train, y_train_3class)

        # Evaluate on validation
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

        metrics = compute_classification_metrics(y_val_3class, y_pred, y_proba)
        metrics["model_name"] = name
        metrics["target_type"] = "3class"

        print(f"  Val F1 (macro): {metrics['f1']:.4f}")
        print(compute_per_class_report(y_val_3class, y_pred, TARGET_NAMES_3CLASS))

        # Save model and metrics
        save_model_artifact(model, name, {"run_id": run_id, **metrics})
        save_metrics(metrics, name)

        # Plots
        plot_confusion_matrix(
            y_val_3class, y_pred, TARGET_NAMES_3CLASS,
            f"{name} - Confusion Matrix",
            reports_dir / f"{name}_confusion_matrix.png"
        )

        if y_proba is not None:
            plot_roc_curve(
                y_val_3class, y_proba,
                f"{name} - ROC Curve",
                reports_dir / f"{name}_roc_curve.png"
            )

        results.append(metrics)

    # Train binary models
    print("\n--- BINARY MODELS ---")
    for name, model in models_binary.items():
        print(f"\nTraining {name}...")

        if CONFIG["tune_models"]:
            base_name = name.replace("_binary", "")
            param_grid = get_param_grids().get(base_name.replace("_", "_"), {})
            if param_grid:
                model, _, _ = tune_model(
                    model, X_train, y_train_binary,
                    param_grid, CONFIG["tune_n_iter"], CONFIG["tune_cv"]
                )
        else:
            model.fit(X_train, y_train_binary)

        # Evaluate on validation
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

        metrics = compute_classification_metrics(y_val_binary, y_pred, y_proba)
        metrics["model_name"] = name
        metrics["target_type"] = "binary"

        print(f"  Val F1 (macro): {metrics['f1']:.4f}")
        print(compute_per_class_report(y_val_binary, y_pred, TARGET_NAMES_BINARY))

        # Save model and metrics
        save_model_artifact(model, name, {"run_id": run_id, **metrics})
        save_metrics(metrics, name)

        # Plots
        plot_confusion_matrix(
            y_val_binary, y_pred, TARGET_NAMES_BINARY,
            f"{name} - Confusion Matrix",
            reports_dir / f"{name}_confusion_matrix.png"
        )

        if y_proba is not None:
            plot_roc_curve(
                y_val_binary, y_proba,
                f"{name} - ROC Curve",
                reports_dir / f"{name}_roc_curve.png"
            )

        results.append(metrics)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n[5/5] Summary")
    print("=" * 80)
    print(f"Trained {len(results)} baseline models")
    print(f"Models saved to: milestone_2/models/")
    print(f"Metrics saved to: milestone_2/reports/metrics/")
    print("=" * 80)

    return results


if __name__ == "__main__":
    train_baselines()
