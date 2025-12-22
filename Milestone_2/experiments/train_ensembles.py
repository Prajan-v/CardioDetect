"""
Train ensemble models.

Builds VotingClassifier and StackingClassifier ensembles from
the best baseline and multiclass models.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Source_Code"))

import numpy as np

from src.preprocessing import load_splits, get_feature_target_split, build_preprocessor_from_data
from src.targets import add_targets_to_df
from src.models_baselines import (
    build_logistic_regression,
    build_random_forest,
    build_mlp_classifier,
)
from src.ensembles import (
    build_voting_ensemble,
    build_stacking_with_lr_meta,
    build_stacking_with_tree_meta,
    fit_voting_ensemble,
    fit_stacking_ensemble,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
    "ensemble_type": "both",  # 'voting', 'stacking', or 'both'
    "voting_method": "soft",
    "stacking_cv": 5,
}

TARGET_NAMES = ["Healthy", "Heart Disease"]


# ============================================================================
# MAIN TRAINING ROUTINE
# ============================================================================

def train_ensembles():
    """Train ensemble classifiers."""
    print("=" * 80)
    print("MILESTONE 2: TRAINING ENSEMBLE CLASSIFIERS")
    print("=" * 80)

    run_id = create_run_id("ensembles")
    print(f"Run ID: {run_id}")

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    data_path = Path("/Users/prajanv/CardioDetect/data/split/detection")
    train_df, val_df, test_df = load_splits(data_dir=data_path)

    # Add guideline risk and derived targets - SKIPPED FOR UCI (Missing smoking column)
    # train_df = add_targets_to_df(train_df)
    # val_df = add_targets_to_df(val_df)
    # test_df = add_targets_to_df(test_df)
    
    # Drop rows with NaN guideline risk - SKIPPED
    # train_df = train_df[train_df["guideline_risk_10yr"].notna()].copy()
    # val_df = val_df[val_df["guideline_risk_10yr"].notna()].copy()
    # test_df = test_df[test_df["guideline_risk_10yr"].notna()].copy()

    print(f"After filtering: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ------------------------------------------------------------------
    # Prepare features
    # ------------------------------------------------------------------
    print("\n[2/5] Preparing features...")

    X_train, _ = get_feature_target_split(train_df, target_col="target")
    X_val, _ = get_feature_target_split(val_df, target_col="target")
    X_test, _ = get_feature_target_split(test_df, target_col="target")

    # Remove target columns from features
    drop_cols = ["guideline_risk_10yr", "risk_class_3", "risk_class_binary"]
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_val = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # Targets
    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    # Build preprocessor
    preprocessor = build_preprocessor_from_data(X_train)

    print(f"Features: {X_train.shape[1]}")
    print(f"Class distribution (train): {np.bincount(y_train)}")

    # ------------------------------------------------------------------
    # Build base estimators
    # ------------------------------------------------------------------
    print("\n[3/5] Building base estimators...")

    base_estimators = [
        ("lr", ImbPipeline([("preprocessor", preprocessor), ("smote", SMOTE(random_state=42)), ("clf", build_logistic_regression(None))])),
        ("rf", ImbPipeline([("preprocessor", preprocessor), ("smote", SMOTE(random_state=42)), ("clf", build_random_forest(None))])),
        ("mlp", ImbPipeline([("preprocessor", preprocessor), ("smote", SMOTE(random_state=42)), ("clf", build_mlp_classifier(None))])),
        ("xgb", ImbPipeline([("preprocessor", preprocessor), ("smote", SMOTE(random_state=42)), ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric='logloss', use_label_encoder=False))])),
        ("lgbm", ImbPipeline([("preprocessor", preprocessor), ("smote", SMOTE(random_state=42)), ("clf", LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05))])),
    ]

    print(f"Base estimators: {[name for name, _ in base_estimators]}")

    # ------------------------------------------------------------------
    # Train ensembles
    # ------------------------------------------------------------------
    print("\n[4/5] Training ensemble models...")

    results = []
    trained_models = {}
    reports_dir = Path(__file__).resolve().parents[1] / "reports" / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Voting Ensemble
    if CONFIG["ensemble_type"] in ["voting", "both"]:
        print("\n--- VOTING ENSEMBLE ---")
        name = "voting_ensemble"

        voting = build_voting_ensemble(
            base_estimators,
            voting=CONFIG["voting_method"],
        )
        voting = fit_voting_ensemble(voting, X_train, y_train)

        trained_models[name] = voting

        # Evaluate
        y_pred = voting.predict(X_val)
        y_proba = voting.predict_proba(X_val) if CONFIG["voting_method"] == "soft" else None

        metrics = compute_classification_metrics(y_val, y_pred, y_proba)
        metrics["model_name"] = name
        metrics["target_type"] = "binary"
        metrics["ensemble_type"] = "voting"

        print(f"  Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Val F1 (macro): {metrics['f1']:.4f}")
        print(compute_per_class_report(y_val, y_pred, TARGET_NAMES))

        save_model_artifact(voting, name, {"run_id": run_id, **metrics})
        save_metrics(metrics, name)

        plot_confusion_matrix(
            y_val, y_pred, TARGET_NAMES,
            f"{name} - Confusion Matrix",
            reports_dir / f"{name}_confusion_matrix.png"
        )

        if y_proba is not None:
            plot_roc_curve(
                y_val, y_proba,
                f"{name} - ROC Curve",
                reports_dir / f"{name}_roc_curve.png"
            )

        results.append(metrics)

    # Stacking Ensemble (LR meta)
    if CONFIG["ensemble_type"] in ["stacking", "both"]:
        print("\n--- STACKING ENSEMBLE (LR Meta) ---")
        name = "stacking_lr_ensemble"

        stacking_lr = build_stacking_with_lr_meta(
            base_estimators,
            cv=CONFIG["stacking_cv"],
        )
        stacking_lr = fit_stacking_ensemble(stacking_lr, X_train, y_train)

        trained_models[name] = stacking_lr

        # Evaluate
        y_pred = stacking_lr.predict(X_val)
        y_proba = stacking_lr.predict_proba(X_val)

        metrics = compute_classification_metrics(y_val, y_pred, y_proba)
        metrics["model_name"] = name
        metrics["target_type"] = "binary"
        metrics["ensemble_type"] = "stacking"

        print(f"  Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Val F1 (macro): {metrics['f1']:.4f}")
        print(compute_per_class_report(y_val, y_pred, TARGET_NAMES))

        save_model_artifact(stacking_lr, name, {"run_id": run_id, **metrics})
        save_metrics(metrics, name)

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

        results.append(metrics)

    # Stacking Ensemble (Tree meta)
    if CONFIG["ensemble_type"] in ["stacking", "both"]:
        print("\n--- STACKING ENSEMBLE (Tree Meta) ---")
        name = "stacking_tree_ensemble"

        stacking_tree = build_stacking_with_tree_meta(
            base_estimators,
            max_depth=3,
            cv=CONFIG["stacking_cv"],
        )
        stacking_tree = fit_stacking_ensemble(stacking_tree, X_train, y_train)

        trained_models[name] = stacking_tree

        # Evaluate
        y_pred = stacking_tree.predict(X_val)
        y_proba = stacking_tree.predict_proba(X_val)

        metrics = compute_classification_metrics(y_val, y_pred, y_proba)
        metrics["model_name"] = name
        metrics["target_type"] = "binary"
        metrics["ensemble_type"] = "stacking"

        print(f"  Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Val F1 (macro): {metrics['f1']:.4f}")
        print(compute_per_class_report(y_val, y_pred, TARGET_NAMES))

        save_model_artifact(stacking_tree, name, {"run_id": run_id, **metrics})
        save_metrics(metrics, name)

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

        results.append(metrics)

    # ------------------------------------------------------------------
    # Evaluate best ensemble on test set
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating best ensemble on test set...")

    best_result = max(results, key=lambda x: x["f1"])
    best_model_name = best_result["model_name"]
    best_model = trained_models[best_model_name]

    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)

    test_metrics = compute_classification_metrics(y_test, y_pred_test, y_proba_test)

    print(f"\nBest ensemble: {best_model_name}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1']:.4f}")
    print(compute_per_class_report(y_test, y_pred_test, TARGET_NAMES))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"Trained {len(results)} ensemble models")
    print(f"Best ensemble: {best_model_name} (F1={best_result['f1']:.4f})")
    print(f"Models saved to: milestone_2/models/")
    print(f"Metrics saved to: milestone_2/reports/metrics/")
    print("=" * 80)

    return results


if __name__ == "__main__":
    train_ensembles()
