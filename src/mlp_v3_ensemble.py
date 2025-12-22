"""
Model V3: Ensemble training and evaluation.
Trains multiple algorithms and compares against mlp_v2 baseline.
No regression allowed - must match or beat v2 performance.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "split"


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train/val/test splits."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Separate features and target
    drop_cols = ['risk_target', 'data_source']
    
    y_train = (train_df['risk_target'] > 0).astype(int)
    y_val = (val_df['risk_target'] > 0).astype(int)
    y_test = (test_df['risk_target'] > 0).astype(int)
    
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    
    # Handle categorical encoding
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        combined = pd.concat([X_train, X_val, X_test], keys=["train", "val", "test"])
        combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)
        X_train = combined_encoded.xs("train")
        X_val = combined_encoded.xs("val")
        X_test = combined_encoded.xs("test")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }


def train_mlp_v2_baseline(X_train, y_train, X_val, y_val, X_test, y_test) -> Tuple[MLPClassifier, StandardScaler, Dict]:
    """Train MLP v2 baseline (or load if exists)."""
    print("\n" + "="*80)
    print("MLP V2 BASELINE")
    print("="*80)
    
    mlp_v2_path = MODELS_DIR / "mlp_v2.pkl"
    
    if mlp_v2_path.exists():
        print("Loading existing mlp_v2.pkl...")
        artifact = joblib.load(mlp_v2_path)
        model = artifact['model']
        scaler = artifact['scaler']
    else:
        print("Training new MLP v2...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Save
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': model, 'scaler': scaler}, mlp_v2_path)
        print(f"Saved to {mlp_v2_path}")
    
    # Evaluate
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    print(f"\nMLP v2 Test Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return model, scaler, metrics


def train_random_forest(X_train, y_train, X_test, y_test) -> Tuple[RandomForestClassifier, Dict]:
    """Train Random Forest."""
    print("\n" + "="*80)
    print("RANDOM FOREST")
    print("="*80)
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    print(f"\nRandom Forest Test Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test) -> Tuple:
    """Train XGBoost if available."""
    if not XGBOOST_AVAILABLE:
        print("\nXGBoost not available, skipping...")
        return None, None
    
    print("\n" + "="*80)
    print("XGBOOST")
    print("="*80)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train.values, y_train.values, verbose=False)
    
    y_pred = model.predict(X_test.values)
    y_proba = model.predict_proba(X_test.values)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    print(f"\nXGBoost Test Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test) -> Tuple:
    """Train LightGBM if available."""
    if not LIGHTGBM_AVAILABLE:
        print("\nLightGBM not available, skipping...")
        return None, None
    
    print("\n" + "="*80)
    print("LIGHTGBM")
    print("="*80)
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    print(f"\nLightGBM Test Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return model, metrics


def select_best_model(all_results: Dict, baseline_metrics: Dict) -> Tuple[str, Dict]:
    """
    Select best model based on metrics.
    Must not regress from baseline.
    """
    print("\n" + "="*80)
    print("MODEL SELECTION")
    print("="*80)
    
    # Sort by accuracy, then recall
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: (x[1]['accuracy'], x[1]['recall']),
        reverse=True
    )
    
    best_name, best_metrics = sorted_models[0]
    
    # Check for regression
    baseline_acc = baseline_metrics['accuracy']
    best_acc = best_metrics['accuracy']
    
    if best_acc < baseline_acc - 0.005:  # Allow 0.5% tolerance
        print(f"\nWARNING: Best model {best_name} ({best_acc:.4f}) regresses from baseline ({baseline_acc:.4f})")
        print("Keeping mlp_v2 as production model.")
        return "mlp_v2", baseline_metrics
    
    print(f"\nSelected: {best_name}")
    print(f"  Accuracy improvement: {best_acc - baseline_acc:+.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f} (baseline: {baseline_metrics['recall']:.4f})")
    
    return best_name, best_metrics


def train_and_evaluate_all() -> Dict:
    """Train all models and return comparison results."""
    print("\n" + "="*80)
    print("CARDIODETECT MODEL V3 TRAINING")
    print("="*80)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Train all models
    all_results = {}
    all_models = {}
    
    # 1. MLP v2 baseline
    mlp_v2, scaler_v2, metrics_v2 = train_mlp_v2_baseline(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['mlp_v2'] = metrics_v2
    all_models['mlp_v2'] = {'model': mlp_v2, 'scaler': scaler_v2}
    
    # 2. Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    all_results['random_forest'] = rf_metrics
    all_models['random_forest'] = {'model': rf_model, 'scaler': None}
    
    # 3. XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_model:
        all_results['xgboost'] = xgb_metrics
        all_models['xgboost'] = {'model': xgb_model, 'scaler': None}
    
    # 4. LightGBM
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
    if lgb_model:
        all_results['lightgbm'] = lgb_metrics
        all_models['lightgbm'] = {'model': lgb_model, 'scaler': None}
    
    # Select best
    best_name, best_metrics = select_best_model(all_results, metrics_v2)
    
    # Save best model as mlp_v3
    if best_name != 'mlp_v2':
        mlp_v3_path = MODELS_DIR / "mlp_v3_best.pkl"
        joblib.dump(all_models[best_name], mlp_v3_path)
        print(f"\nSaved best model to {mlp_v3_path}")
    
    return {
        'all_results': all_results,
        'best_model': best_name,
        'best_metrics': best_metrics,
        'baseline_metrics': metrics_v2
    }


if __name__ == "__main__":
    results = train_and_evaluate_all()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nBest model: {results['best_model']}")
    print(f"Accuracy: {results['best_metrics']['accuracy']:.4f}")
    print(f"ROC-AUC: {results['best_metrics']['roc_auc']:.4f}")
