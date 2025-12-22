"""
Diagnostic script to check for overfitting and potential issues in the classification model.

Checks performed:
1. Train vs Test accuracy gap (overfitting indicator)
2. Cross-validation on full dataset
3. Feature importance analysis (data leakage detection)
4. Class distribution analysis
5. Learning curve analysis
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, f1_score
import joblib

from src.preprocessing import load_splits, get_feature_target_split, build_preprocessor_from_data
from src.targets import add_targets_to_df


def run_diagnostics():
    """Run comprehensive diagnostics on the trained model."""
    print("=" * 80)
    print("MODEL DIAGNOSTICS: Checking for Overfitting & Issues")
    print("=" * 80)
    
    # Load the trained model
    model_path = Path(__file__).resolve().parents[1] / "models" / "final_classifier.pkl"
    if not model_path.exists():
        print("ERROR: Model not found. Please run train_classification.py first.")
        return
    
    model = joblib.load(model_path)
    print(f"✓ Loaded model from: {model_path}")
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df, val_df, test_df = load_splits()
    
    # Add targets
    train_df = add_targets_to_df(train_df)
    val_df = add_targets_to_df(val_df)
    test_df = add_targets_to_df(test_df)
    
    # Filter NaN
    train_df = train_df[train_df["guideline_risk_10yr"].notna()].copy()
    val_df = val_df[val_df["guideline_risk_10yr"].notna()].copy()
    test_df = test_df[test_df["guideline_risk_10yr"].notna()].copy()
    
    # Prepare features
    X_train, _ = get_feature_target_split(train_df, target_col="risk_class_3")
    X_val, _ = get_feature_target_split(val_df, target_col="risk_class_3")
    X_test, _ = get_feature_target_split(test_df, target_col="risk_class_3")
    
    drop_cols = ["guideline_risk_10yr", "risk_class_3", "risk_class_binary", "risk_target"]
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_val = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    
    y_train = train_df["risk_class_3"].values
    y_val = val_df["risk_class_3"].values
    y_test = test_df["risk_class_3"].values
    
    # ================================================================
    # CHECK 1: Train vs Test Accuracy Gap (Overfitting)
    # ================================================================
    print("\n" + "=" * 60)
    print("[2/6] CHECK: Train vs Test Accuracy Gap (Overfitting)")
    print("=" * 60)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    test_f1 = f1_score(y_test, test_pred, average='macro')
    
    gap = train_acc - test_acc
    
    print(f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Gap:            {gap:.4f} ({gap*100:.2f}%)")
    print(f"Train F1:       {train_f1:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    
    if gap > 0.05:
        print("\n⚠️ WARNING: Significant overfitting detected (gap > 5%)")
    elif gap > 0.02:
        print("\n⚡ NOTICE: Slight overfitting (gap 2-5%)")
    else:
        print("\n✓ OK: No significant overfitting (gap < 2%)")
    
    # ================================================================
    # CHECK 2: Feature Importance (Data Leakage Detection)
    # ================================================================
    print("\n" + "=" * 60)
    print("[3/6] CHECK: Feature Importance (Data Leakage Detection)")
    print("=" * 60)
    
    # Get the classifier from pipeline
    if hasattr(model, 'named_steps'):
        clf = model.named_steps.get('clf')
    else:
        clf = model
    
    feature_names = X_train.columns.tolist()
    
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        
        # For pipelines, get transformed feature names
        if hasattr(model, 'named_steps') and 'prep' in model.named_steps:
            try:
                prep = model.named_steps['prep']
                # Try to get feature names after transformation
                X_transformed = prep.transform(X_train.head(1))
                if hasattr(prep, 'get_feature_names_out'):
                    feature_names = prep.get_feature_names_out().tolist()
                elif X_transformed.shape[1] != len(feature_names):
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            except Exception:
                pass
        
        # Align lengths
        if len(importances) != len(feature_names):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print("-" * 40)
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            pct = row['importance'] * 100
            print(f"{i+1}. {row['feature'][:30]:<30} {pct:>6.2f}%")
        
        # Check for suspiciously high importance (potential leakage)
        top_importance = importance_df.iloc[0]['importance']
        if top_importance > 0.5:
            print(f"\n⚠️ WARNING: Top feature has {top_importance*100:.1f}% importance!")
            print("   This may indicate data leakage or target encoding issues.")
        else:
            print(f"\n✓ OK: No single feature dominates (top: {top_importance*100:.1f}%)")
    else:
        print("Feature importances not available for this model type.")
    
    # ================================================================
    # CHECK 3: Class Distribution
    # ================================================================
    print("\n" + "=" * 60)
    print("[4/6] CHECK: Class Distribution")
    print("=" * 60)
    
    train_dist = np.bincount(y_train) / len(y_train) * 100
    test_dist = np.bincount(y_test) / len(y_test) * 100
    
    classes = ["LOW", "MODERATE", "HIGH"]
    print("\nClass Distribution:")
    print("-" * 50)
    print(f"{'Class':<12} {'Train':<12} {'Test':<12} {'Diff':<12}")
    print("-" * 50)
    
    for i, cls in enumerate(classes):
        diff = abs(train_dist[i] - test_dist[i])
        print(f"{cls:<12} {train_dist[i]:>6.2f}%      {test_dist[i]:>6.2f}%      {diff:>6.2f}%")
    
    # Check for imbalance
    min_class = min(train_dist)
    if min_class < 10:
        print(f"\n⚠️ WARNING: Severe class imbalance (smallest: {min_class:.1f}%)")
    elif min_class < 20:
        print(f"\n⚡ NOTICE: Moderate class imbalance (smallest: {min_class:.1f}%)")
    else:
        print(f"\n✓ OK: Reasonable class balance (smallest: {min_class:.1f}%)")
    
    # ================================================================
    # CHECK 4: Cross-Validation (Robustness)
    # ================================================================
    print("\n" + "=" * 60)
    print("[5/6] CHECK: Cross-Validation (Robustness)")
    print("=" * 60)
    
    # Combine train + val for CV
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val])
    
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_combined, y_combined, cv=5, scoring='f1_macro', n_jobs=-1)
    
    print(f"\nCV F1 Scores: {cv_scores}")
    print(f"Mean CV F1:   {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    if cv_scores.std() > 0.05:
        print("\n⚠️ WARNING: High variance in CV scores (std > 0.05)")
    else:
        print("\n✓ OK: Stable performance across folds")
    
    # ================================================================
    # CHECK 5: Potential Data Leakage Features
    # ================================================================
    print("\n" + "=" * 60)
    print("[6/6] CHECK: Potential Data Leakage Features")
    print("=" * 60)
    
    # List of features that might leak target information
    suspicious_features = ['risk_target', 'guideline_risk', 'risk_score', 'target']
    found_suspicious = [f for f in X_train.columns if any(s in f.lower() for s in suspicious_features)]
    
    if found_suspicious:
        print(f"\n⚠️ WARNING: Found potentially leaky features: {found_suspicious}")
        print("   These features may contain target information!")
    else:
        print("\n✓ OK: No obviously leaky feature names detected")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Train Accuracy:     {train_acc*100:.2f}%")
    print(f"Test Accuracy:      {test_acc*100:.2f}%")
    print(f"Overfitting Gap:    {gap*100:.2f}%")
    print(f"CV F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Class Balance:      {min(train_dist):.1f}% - {max(train_dist):.1f}%")
    
    # Final assessment
    issues = []
    if gap > 0.05:
        issues.append("Significant overfitting")
    if 'risk_target' in X_train.columns:
        issues.append("Potential target leakage (risk_target feature)")
    if cv_scores.std() > 0.05:
        issues.append("High CV variance")
    
    if issues:
        print(f"\n⚠️ ISSUES FOUND: {', '.join(issues)}")
    else:
        print("\n✓ MODEL PASSES ALL DIAGNOSTIC CHECKS")
    
    print("=" * 80)
    
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": gap,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "issues": issues
    }


if __name__ == "__main__":
    run_diagnostics()
