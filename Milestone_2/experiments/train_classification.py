"""
Train and compare all classification models for heart disease prediction.

Trains Logistic Regression, Random Forest, SVM, and MLP classifiers on the
3-class risk target (LOW/MODERATE/HIGH). Compares performance and saves the
best model based on validation F1-score (macro).

Following Milestone 2 requirements from PDF:
- Model Architecture Design and Comparison
- Model Training with Hyperparameter Tuning
- Comprehensive Evaluation (accuracy, precision, recall, F1-score)
- ROC-AUC Analysis
- Confusion Matrices
- Risk Categorization System
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import load_splits, get_feature_target_split, build_preprocessor_from_data
from src.targets import add_targets_to_df
from src.models_baselines import (
    build_logistic_regression,
    build_random_forest,
    build_svm,
    build_mlp_classifier,
    get_param_grids,
)
from src.utils_io import save_model_artifact, create_run_id

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "tune_models": True,  # Enable hyperparameter tuning
    "tune_cv": 3,         # Cross-validation folds for tuning
    "random_state": 42,
}

TARGET_NAMES = ["LOW", "MODERATE", "HIGH"]

# Tuning parameter grids (optimized for faster training)
PARAM_GRIDS = {
    "lr": {
        "clf__C": [0.1, 1.0],
    },
    "rf": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [10, None],
    },
    "svm": {
        "clf__C": [1.0],
        "clf__kernel": ["rbf"],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(100,), (100, 50)],
        "clf__max_iter": [500],
    },
}


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(y_true, y_proba, class_names, title, save_path):
    """Plot multi-class ROC curves."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'orange', 'green']
    
    all_auc = []
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        all_auc.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title}\nMean AUC = {np.mean(all_auc):.3f}')
    ax.legend(loc="lower right")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    return np.mean(all_auc)


def plot_model_comparison(results_df, save_path):
    """Plot model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df))
    width = 0.2
    
    metrics = ['val_accuracy', 'val_f1', 'val_recall', 'val_precision']
    labels = ['Accuracy', 'F1-Score', 'Recall', 'Precision']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        ax.bar(x + i*width, results_df[metric], width, label=label, color=color)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Classification Model Comparison (Validation Set)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['model_name'])
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# MAIN TRAINING ROUTINE
# ============================================================================

def train_and_compare_classifiers():
    """Train all classifiers, compare, and finalize the best one."""
    print("=" * 80)
    print("MILESTONE 2: CLASSIFICATION MODEL TRAINING & COMPARISON")
    print("Following PDF instructions: Train, Compare, and Finalize Best Model")
    print("=" * 80)
    
    run_id = create_run_id("classification")
    print(f"Run ID: {run_id}")
    
    # Create output directories
    base_dir = Path(__file__).resolve().parents[1]
    reports_dir = base_dir / "reports" / "classification"
    models_dir = base_dir / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    train_df, val_df, test_df = load_splits()
    
    # Add risk targets
    train_df = add_targets_to_df(train_df)
    val_df = add_targets_to_df(val_df)
    test_df = add_targets_to_df(test_df)
    
    # Filter out NaN guideline risk
    train_df = train_df[train_df["guideline_risk_10yr"].notna()].copy()
    val_df = val_df[val_df["guideline_risk_10yr"].notna()].copy()
    test_df = test_df[test_df["guideline_risk_10yr"].notna()].copy()
    
    print(f"Dataset sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # ------------------------------------------------------------------
    # 2. Prepare features and targets
    # ------------------------------------------------------------------
    print("\n[2/6] Preparing features...")
    
    X_train, _ = get_feature_target_split(train_df, target_col="risk_class_3")
    X_val, _ = get_feature_target_split(val_df, target_col="risk_class_3")
    X_test, _ = get_feature_target_split(test_df, target_col="risk_class_3")
    
    # Remove target columns AND leaky features
    # NOTE: 'risk_target' is excluded because it contains encoded target information (data leakage)
    drop_cols = ["guideline_risk_10yr", "risk_class_3", "risk_class_binary", "risk_target"]
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_val = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    
    print(f"⚠️ Removed 'risk_target' to prevent data leakage")
    
    y_train = train_df["risk_class_3"].values
    y_val = val_df["risk_class_3"].values
    y_test = test_df["risk_class_3"].values
    
    # Build preprocessor
    preprocessor = build_preprocessor_from_data(X_train)
    
    print(f"Features: {X_train.shape[1]}")
    print(f"Target distribution (train): {np.bincount(y_train)} (LOW, MODERATE, HIGH)")
    
    # ------------------------------------------------------------------
    # 3. Define and train all models
    # ------------------------------------------------------------------
    print("\n[3/6] Training and tuning all classifiers...")
    
    models = {
        "Logistic Regression": (build_logistic_regression(preprocessor), PARAM_GRIDS["lr"]),
        "Random Forest": (build_random_forest(preprocessor), PARAM_GRIDS["rf"]),
        "SVM": (build_svm(preprocessor), PARAM_GRIDS["svm"]),
        "MLP (Neural Network)": (build_mlp_classifier(preprocessor), PARAM_GRIDS["mlp"]),
    }
    
    results = []
    trained_models = {}
    
    for name, (model, param_grid) in models.items():
        print(f"\n--- Training {name} ---")
        
        # Hyperparameter tuning with GridSearchCV
        if CONFIG["tune_models"] and param_grid:
            print(f"  Tuning hyperparameters (GridSearchCV, cv={CONFIG['tune_cv']})...")
            grid_search = GridSearchCV(
                model, param_grid,
                cv=CONFIG["tune_cv"],
                scoring="f1_macro",
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Best CV F1: {grid_search.best_score_:.4f}")
        else:
            model.fit(X_train, y_train)
            best_model = model
        
        # Evaluate on validation set
        y_pred_val = best_model.predict(X_val)
        y_proba_val = best_model.predict_proba(X_val) if hasattr(best_model, "predict_proba") else None
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val, y_pred_val)
        val_precision = precision_score(y_val, y_pred_val, average='macro', zero_division=0)
        val_recall = recall_score(y_val, y_pred_val, average='macro', zero_division=0)
        val_f1 = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
        
        # ROC-AUC (if probabilities available)
        val_roc_auc = None
        if y_proba_val is not None:
            try:
                val_roc_auc = roc_auc_score(y_val, y_proba_val, multi_class='ovr', average='macro')
            except:
                pass
        
        print(f"  Validation Accuracy:  {val_accuracy:.4f}")
        print(f"  Validation Precision: {val_precision:.4f}")
        print(f"  Validation Recall:    {val_recall:.4f}")
        print(f"  Validation F1 (macro): {val_f1:.4f}")
        if val_roc_auc:
            print(f"  Validation ROC-AUC:   {val_roc_auc:.4f}")
        
        results.append({
            "model_name": name,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_roc_auc": val_roc_auc or 0,
        })
        
        trained_models[name] = {
            "model": best_model,
            "y_pred_val": y_pred_val,
            "y_proba_val": y_proba_val,
        }
    
    # ------------------------------------------------------------------
    # 4. Model Comparison and Selection
    # ------------------------------------------------------------------
    print("\n[4/6] Comparing models and selecting the best...")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("val_f1", ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by F1-score)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    
    # Select best model
    best_model_name = results_df.iloc[0]["model_name"]
    best_metrics = results_df.iloc[0]
    best_model_data = trained_models[best_model_name]
    
    print(f"\n*** BEST MODEL: {best_model_name} ***")
    print(f"    Validation Accuracy: {best_metrics['val_accuracy']:.4f}")
    print(f"    Validation F1-score: {best_metrics['val_f1']:.4f}")
    
    # Save comparison plot
    plot_model_comparison(results_df, reports_dir / "model_comparison.png")
    
    # ------------------------------------------------------------------
    # 5. Final Evaluation on Test Set
    # ------------------------------------------------------------------
    print("\n[5/6] Final evaluation on TEST set...")
    
    final_model = best_model_data["model"]
    y_pred_test = final_model.predict(X_test)
    y_proba_test = final_model.predict_proba(X_test) if hasattr(final_model, "predict_proba") else None
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
    
    print("\n" + "=" * 60)
    print(f"FINAL MODEL: {best_model_name}")
    print("=" * 60)
    print(f"TEST Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"TEST Precision: {test_precision:.4f}")
    print(f"TEST Recall:    {test_recall:.4f}")
    print(f"TEST F1 (macro): {test_f1:.4f}")
    
    # Check if meets >85% target
    if test_accuracy >= 0.85:
        print(f"\n✓ MEETS TARGET: Accuracy {test_accuracy*100:.2f}% >= 85%")
    else:
        print(f"\n⚠ Below target: Accuracy {test_accuracy*100:.2f}% < 85%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=TARGET_NAMES))
    
    # Generate and save confusion matrix
    plot_confusion_matrix(
        y_test, y_pred_test, TARGET_NAMES,
        f"{best_model_name} - Test Set Confusion Matrix",
        reports_dir / "final_confusion_matrix.png"
    )
    
    # Generate and save ROC curves
    if y_proba_test is not None:
        mean_auc = plot_roc_curves(
            y_test, y_proba_test, TARGET_NAMES,
            f"{best_model_name} - Test Set ROC Curves",
            reports_dir / "final_roc_curves.png"
        )
        print(f"TEST Mean ROC-AUC: {mean_auc:.4f}")
    
    # ------------------------------------------------------------------
    # 6. Save Final Model and Reports
    # ------------------------------------------------------------------
    print("\n[6/6] Saving final model and reports...")
    
    # Save model artifact
    final_metrics = {
        "run_id": run_id,
        "model_name": best_model_name,
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "target_names": TARGET_NAMES,
        "feature_names": X_train.columns.tolist(),
        "trained_at": datetime.now().isoformat(),
    }
    
    save_model_artifact(final_model, "final_classifier", final_metrics)
    
    # Save comparison results
    results_df.to_csv(reports_dir / "model_comparison.csv", index=False)
    
    # Save final metrics as JSON
    with open(reports_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n✓ Model saved: milestone_2/models/final_classifier.pkl")
    print(f"✓ Reports saved: milestone_2/reports/classification/")
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Trained {len(models)} classification models")
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test F1-Score: {test_f1:.4f}")
    print("=" * 80)
    
    return final_model, final_metrics


if __name__ == "__main__":
    train_and_compare_classifiers()
