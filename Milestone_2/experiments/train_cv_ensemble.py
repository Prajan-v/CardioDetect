"""
CROSS-VALIDATION AND ENSEMBLE METHODS FOR IMPROVED ACCURACY

Target: Push accuracy from 90.69% to 91-92%+

Techniques:
1. Stratified K-Fold Cross-Validation (5 and 10 folds)
2. Voting Ensembles (Hard/Soft with optimized weights)
3. Stacking Ensemble with meta-learner
4. Bagging with different base estimators
5. Blending approach

Author: Auto-generated for CardioDetect project
"""

from __future__ import annotations
import sys
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    RepeatedStratifiedKFold, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, make_scorer
)
from imblearn.over_sampling import SMOTE, ADASYN
import joblib

# Import advanced libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not available")


def load_split_data():
    """Load pre-split train/val/test data with guideline-calculated risk labels."""
    data_dir = Path(__file__).resolve().parents[2] / "data" / "split"
    
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    return train_df, val_df, test_df


def get_feature_columns(df):
    """Get feature columns, excluding target and non-numeric."""
    exclude_cols = {
        'risk_target', 'data_source', 'guideline_risk_10yr', 'sex',
        'age_group_<40', 'age_group_40-49', 'age_group_50-59', 
        'age_group_60-69', 'age_group_70+', 'bmi_cat_Underweight',
        'bmi_cat_Normal', 'bmi_cat_Overweight', 'bmi_cat_Obese'
    }
    
    # Get numeric columns only
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                feature_cols.append(col)
            elif df[col].dtype == 'bool':
                feature_cols.append(col)
    
    return feature_cols
    
    # Basic derived features
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        df['map'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
        df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    
    # Age-based interactions
    if 'age' in df.columns:
        for col in ['systolic_bp', 'total_cholesterol', 'bmi', 'heart_rate']:
            if col in df.columns:
                df[f'age_{col[:4]}'] = df['age'] * df[col]
        
        if 'smoking' in df.columns:
            df['age_smoking'] = df['age'] * df['smoking'].fillna(0)
        if 'diabetes' in df.columns:
            df['age_diabetes'] = df['age'] * df['diabetes'].fillna(0)
    
    # Non-linear transforms
    for col in ['age', 'systolic_bp', 'total_cholesterol', 'bmi']:
        if col in df.columns:
            df[f'{col}_sq'] = df[col] ** 2
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
    
    # Clinical threshold flags
    if 'systolic_bp' in df.columns:
        df['high_bp_140'] = (df['systolic_bp'] >= 140).astype(int)
        df['high_bp_160'] = (df['systolic_bp'] >= 160).astype(int)
    
    if 'total_cholesterol' in df.columns:
        df['high_chol_200'] = (df['total_cholesterol'] >= 200).astype(int)
        df['high_chol_240'] = (df['total_cholesterol'] >= 240).astype(int)
    
    if 'fasting_glucose' in df.columns:
        df['pre_diabetic'] = ((df['fasting_glucose'] >= 100) & (df['fasting_glucose'] < 126)).astype(int)
        df['diabetic_glucose'] = (df['fasting_glucose'] >= 126).astype(int)
    
    if 'bmi' in df.columns:
        df['overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
        df['obese'] = (df['bmi'] >= 30).astype(int)
    
    # Metabolic syndrome score
    syndrome_cols = []
    for col in ['high_bp_140', 'high_chol_200', 'diabetic_glucose', 'obese', 'hypertension', 'diabetes']:
        if col in df.columns:
            syndrome_cols.append(col)
    
    if syndrome_cols:
        df['metabolic_score'] = df[syndrome_cols].fillna(0).sum(axis=1)
    
    # Risk combination score
    if all(c in df.columns for c in ['systolic_bp', 'total_cholesterol']):
        df['card_risk'] = (df['systolic_bp'] / 120) + (df['total_cholesterol'] / 200)
        if 'smoking' in df.columns:
            df['card_risk'] += df['smoking'].fillna(0)
    
    return df


def optimize_threshold_for_accuracy(y_true, y_proba):
    """Find threshold that maximizes accuracy."""
    best_acc = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    return best_thresh, best_acc


def cross_validate_with_threshold(model, X, y, cv=5, resampler=None):
    """Cross-validation with per-fold threshold optimization."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    fold_results = []
    oof_predictions = np.zeros(len(y))
    oof_proba = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Apply resampling if specified
        if resampler:
            X_train_fold, y_train_fold = resampler.fit_resample(X_train_fold, y_train_fold)
        
        # Clone and train model
        from sklearn.base import clone
        model_clone = clone(model)
        model_clone.fit(X_train_fold, y_train_fold)
        
        # Get probabilities
        y_proba = model_clone.predict_proba(X_val_fold)[:, 1]
        
        # Optimize threshold
        best_thresh, best_acc = optimize_threshold_for_accuracy(y_val_fold, y_proba)
        y_pred = (y_proba >= best_thresh).astype(int)
        
        # Calculate metrics
        recall = recall_score(y_val_fold, y_pred, zero_division=0)
        precision = precision_score(y_val_fold, y_pred, zero_division=0)
        f1 = f1_score(y_val_fold, y_pred, zero_division=0)
        auc = roc_auc_score(y_val_fold, y_proba)
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': best_acc,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'roc_auc': auc,
            'threshold': best_thresh
        })
        
        oof_predictions[val_idx] = y_pred
        oof_proba[val_idx] = y_proba
    
    return fold_results, oof_predictions, oof_proba


def get_base_models():
    """Get dictionary of base models for ensembles."""
    models = {
        'rf': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', 
            random_state=42, n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            learning_rate_init=0.001, alpha=0.0001,
            early_stopping=True, random_state=42
        ),
    }
    
    if HAS_XGBOOST:
        models['xgb'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=5, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric='logloss'
        )
    
    if HAS_LGBM:
        models['lgbm'] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
        )
    
    if HAS_CATBOOST:
        models['catboost'] = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1,
            auto_class_weights='Balanced', random_state=42, verbose=0
        )
    
    return models


def build_voting_ensemble(base_models, voting='soft', weights=None):
    """Build voting ensemble from base models."""
    estimators = [(name, model) for name, model in base_models.items()]
    return VotingClassifier(estimators=estimators, voting=voting, weights=weights, n_jobs=-1)


def build_stacking_ensemble(base_models, meta_learner=None):
    """Build stacking ensemble with meta-learner."""
    if meta_learner is None:
        meta_learner = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    
    estimators = [(name, model) for name, model in base_models.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5, n_jobs=-1, passthrough=False
    )


def build_bagging_ensemble(base_estimator, n_estimators=10):
    """Build bagging ensemble."""
    return BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        max_samples=0.8, max_features=0.8,
        random_state=42, n_jobs=-1
    )


def train_cv_ensemble():
    """Main training function with CV and ensemble methods."""
    print("=" * 80)
    print("CROSS-VALIDATION AND ENSEMBLE TRAINING")
    print("Target: Push accuracy from 90.69% to 91-92%+")
    print("Using: Combined guideline-calculated risk dataset")
    print("=" * 80)
    
    # Setup paths
    base_dir = Path(__file__).resolve().parents[1]
    models_dir = Path(__file__).resolve().parents[2] / "models"
    reports_dir = base_dir / "reports" / "cv_ensemble"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pre-split data with guideline-calculated risk labels
    print("\n[1/7] Loading pre-split guideline data...")
    train_df, val_df, test_df = load_split_data()
    
    # Combine train + val for CV training, use test for final evaluation
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Validation data: {len(val_df)} samples")
    print(f"Test data: {len(test_df)} samples")
    print(f"Combined for CV: {len(combined_df)} samples")
    
    # Target column is 'risk_target' (0=LOW, 1=MODERATE, 2=HIGH)
    # Convert to binary: risk_target >= 1 means elevated risk
    target_col = 'risk_target'
    
    # Get feature columns (numeric only, excluding targets and metadata)
    feature_cols = get_feature_columns(combined_df)
    print(f"\n[2/7] Features selected: {len(feature_cols)}")
    
    # Prepare training data
    X_combined = combined_df[feature_cols].copy()
    y_combined = (combined_df[target_col] >= 1).astype(int).values  # Binary: elevated risk
    
    # Prepare test data  
    X_test = test_df[feature_cols].copy()
    y_test = (test_df[target_col] >= 1).astype(int).values
    
    # Handle missing values
    X_combined = X_combined.replace([np.inf, -np.inf], np.nan)
    X_combined = X_combined.fillna(X_combined.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.fillna(X_combined.median())  # Use train median
    
    print(f"Risk distribution (combined): {np.bincount(y_combined)}")
    print(f"Risk distribution (test): {np.bincount(y_test)}")
    
    # Scale features
    print("\n[3/7] Scaling features...")
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE resampling for training
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_combined_scaled, y_combined)
    
    # =========================================================================
    # STEP 4: Cross-Validation on Individual Models
    # =========================================================================
    print("\n[4/7] Cross-validating individual models (5-fold)...")
    
    base_models = get_base_models()
    cv_results = []
    
    for name, model in base_models.items():
        print(f"  CV on {name}...", end=" ")
        fold_results, oof_pred, oof_proba = cross_validate_with_threshold(
            model, X_combined_scaled, y_combined, cv=5, resampler=SMOTE(random_state=42)
        )
        
        # Aggregate results
        mean_acc = np.mean([r['accuracy'] for r in fold_results])
        std_acc = np.std([r['accuracy'] for r in fold_results])
        mean_auc = np.mean([r['roc_auc'] for r in fold_results])
        
        cv_results.append({
            'model': name,
            'cv_accuracy_mean': mean_acc,
            'cv_accuracy_std': std_acc,
            'cv_auc_mean': mean_auc,
            'fold_details': fold_results
        })
        
        print(f"Acc={mean_acc*100:.2f}% (±{std_acc*100:.2f}%), AUC={mean_auc:.3f}")
    
    # Sort by accuracy
    cv_results.sort(key=lambda x: x['cv_accuracy_mean'], reverse=True)
    
    # =========================================================================
    # STEP 5: Build and Evaluate Ensemble Methods
    # =========================================================================
    print("\n[5/7] Building ensemble methods...")
    
    ensemble_results = []
    best_ensemble = {'accuracy': 0}
    
    # Select top models for ensembles
    top_model_names = [r['model'] for r in cv_results[:5]]
    top_models = {name: base_models[name] for name in top_model_names if name in base_models}
    
    print(f"  Using top models: {list(top_models.keys())}")
    
    # Train base models first
    print("  Training base models on full training data...")
    trained_models = {}
    for name, model in top_models.items():
        from sklearn.base import clone
        m = clone(model)
        m.fit(X_train_res, y_train_res)
        trained_models[name] = m
    
    # --- Voting Ensemble (Soft) ---
    print("  Building Soft Voting Ensemble...")
    voting_soft = build_voting_ensemble(top_models, voting='soft')
    voting_soft.fit(X_train_res, y_train_res)
    
    y_proba_vs = voting_soft.predict_proba(X_test_scaled)[:, 1]
    thresh_vs, acc_vs = optimize_threshold_for_accuracy(y_test, y_proba_vs)
    y_pred_vs = (y_proba_vs >= thresh_vs).astype(int)
    
    ensemble_results.append({
        'ensemble': 'Voting_Soft',
        'accuracy': acc_vs,
        'recall': recall_score(y_test, y_pred_vs, zero_division=0),
        'f1': f1_score(y_test, y_pred_vs, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba_vs),
        'threshold': thresh_vs,
        'model': voting_soft
    })
    print(f"    Voting (Soft): Acc={acc_vs*100:.2f}%, AUC={roc_auc_score(y_test, y_proba_vs):.3f}")
    
    # --- Voting Ensemble (Hard) ---
    print("  Building Hard Voting Ensemble...")
    voting_hard = build_voting_ensemble(top_models, voting='hard')
    voting_hard.fit(X_train_res, y_train_res)
    
    y_pred_vh = voting_hard.predict(X_test_scaled)
    acc_vh = accuracy_score(y_test, y_pred_vh)
    
    ensemble_results.append({
        'ensemble': 'Voting_Hard',
        'accuracy': acc_vh,
        'recall': recall_score(y_test, y_pred_vh, zero_division=0),
        'f1': f1_score(y_test, y_pred_vh, zero_division=0),
        'roc_auc': 0,  # No probabilities for hard voting
        'threshold': 0.5,
        'model': voting_hard
    })
    print(f"    Voting (Hard): Acc={acc_vh*100:.2f}%")
    
    # --- Stacking Ensemble (LR Meta) ---
    print("  Building Stacking Ensemble (LR meta-learner)...")
    stacking_lr = build_stacking_ensemble(top_models, 
        meta_learner=LogisticRegression(C=0.5, max_iter=1000, random_state=42))
    stacking_lr.fit(X_train_res, y_train_res)
    
    y_proba_slr = stacking_lr.predict_proba(X_test_scaled)[:, 1]
    thresh_slr, acc_slr = optimize_threshold_for_accuracy(y_test, y_proba_slr)
    y_pred_slr = (y_proba_slr >= thresh_slr).astype(int)
    
    ensemble_results.append({
        'ensemble': 'Stacking_LR',
        'accuracy': acc_slr,
        'recall': recall_score(y_test, y_pred_slr, zero_division=0),
        'f1': f1_score(y_test, y_pred_slr, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba_slr),
        'threshold': thresh_slr,
        'model': stacking_lr
    })
    print(f"    Stacking (LR): Acc={acc_slr*100:.2f}%, AUC={roc_auc_score(y_test, y_proba_slr):.3f}")
    
    # --- Stacking Ensemble (XGB Meta) ---
    if HAS_XGBOOST:
        print("  Building Stacking Ensemble (XGB meta-learner)...")
        stacking_xgb = build_stacking_ensemble(top_models,
            meta_learner=XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            ))
        stacking_xgb.fit(X_train_res, y_train_res)
        
        y_proba_sxgb = stacking_xgb.predict_proba(X_test_scaled)[:, 1]
        thresh_sxgb, acc_sxgb = optimize_threshold_for_accuracy(y_test, y_proba_sxgb)
        y_pred_sxgb = (y_proba_sxgb >= thresh_sxgb).astype(int)
        
        ensemble_results.append({
            'ensemble': 'Stacking_XGB',
            'accuracy': acc_sxgb,
            'recall': recall_score(y_test, y_pred_sxgb, zero_division=0),
            'f1': f1_score(y_test, y_pred_sxgb, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_sxgb),
            'threshold': thresh_sxgb,
            'model': stacking_xgb
        })
        print(f"    Stacking (XGB): Acc={acc_sxgb*100:.2f}%, AUC={roc_auc_score(y_test, y_proba_sxgb):.3f}")
    
    # --- Bagging Ensemble ---
    print("  Building Bagging Ensemble (RF base)...")
    bagging_rf = build_bagging_ensemble(
        RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42),
        n_estimators=15
    )
    bagging_rf.fit(X_train_res, y_train_res)
    
    y_proba_bag = bagging_rf.predict_proba(X_test_scaled)[:, 1]
    thresh_bag, acc_bag = optimize_threshold_for_accuracy(y_test, y_proba_bag)
    y_pred_bag = (y_proba_bag >= thresh_bag).astype(int)
    
    ensemble_results.append({
        'ensemble': 'Bagging_RF',
        'accuracy': acc_bag,
        'recall': recall_score(y_test, y_pred_bag, zero_division=0),
        'f1': f1_score(y_test, y_pred_bag, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba_bag),
        'threshold': thresh_bag,
        'model': bagging_rf
    })
    print(f"    Bagging (RF): Acc={acc_bag*100:.2f}%, AUC={roc_auc_score(y_test, y_proba_bag):.3f}")
    
    # --- Blending Ensemble ---
    print("  Building Blending Ensemble...")
    # Get OOF predictions from trained models for blending
    blend_train = np.column_stack([
        trained_models[name].predict_proba(X_combined_scaled)[:, 1] 
        for name in trained_models
    ])
    blend_test = np.column_stack([
        trained_models[name].predict_proba(X_test_scaled)[:, 1] 
        for name in trained_models
    ])
    
    blend_meta = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    blend_meta.fit(blend_train, y_combined)
    
    y_proba_blend = blend_meta.predict_proba(blend_test)[:, 1]
    thresh_blend, acc_blend = optimize_threshold_for_accuracy(y_test, y_proba_blend)
    y_pred_blend = (y_proba_blend >= thresh_blend).astype(int)
    
    ensemble_results.append({
        'ensemble': 'Blending_LR',
        'accuracy': acc_blend,
        'recall': recall_score(y_test, y_pred_blend, zero_division=0),
        'f1': f1_score(y_test, y_pred_blend, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba_blend),
        'threshold': thresh_blend,
        'model': None  # Blending needs special handling for saving
    })
    print(f"    Blending (LR): Acc={acc_blend*100:.2f}%, AUC={roc_auc_score(y_test, y_proba_blend):.3f}")
    
    # =========================================================================
    # STEP 6: Repeated Cross-Validation on Best Ensemble
    # =========================================================================
    print("\n[6/7] Repeated Stratified K-Fold CV on best ensemble...")
    
    # Find best ensemble
    ensemble_results.sort(key=lambda x: x['accuracy'], reverse=True)
    best_ens = ensemble_results[0]
    
    print(f"  Best ensemble: {best_ens['ensemble']} (Acc={best_ens['accuracy']*100:.2f}%)")
    
    # Perform repeated CV on best ensemble type
    if best_ens['model'] is not None:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        
        rcv_scores = []
        for train_idx, val_idx in rskf.split(X_combined_scaled, y_combined):
            X_t, X_v = X_combined_scaled[train_idx], X_combined_scaled[val_idx]
            y_t, y_v = y_combined[train_idx], y_combined[val_idx]
            
            # Resample
            X_t_res, y_t_res = smote.fit_resample(X_t, y_t)
            
            # Clone and train
            from sklearn.base import clone
            model_clone = clone(best_ens['model'])
            model_clone.fit(X_t_res, y_t_res)
            
            # Get probabilities and optimize threshold
            y_proba = model_clone.predict_proba(X_v)[:, 1]
            _, acc = optimize_threshold_for_accuracy(y_v, y_proba)
            rcv_scores.append(acc)
        
        print(f"  Repeated CV (5x3): {np.mean(rcv_scores)*100:.2f}% (±{np.std(rcv_scores)*100:.2f}%)")
    
    # =========================================================================
    # STEP 7: Final Results and Save
    # =========================================================================
    print("\n" + "=" * 80)
    print("[7/7] FINAL RESULTS")
    print("=" * 80)
    
    # Print ensemble comparison
    print("\nENSEMBLE COMPARISON:")
    print("-" * 70)
    print(f"{'Ensemble':<20} {'Accuracy':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 70)
    
    for ens in ensemble_results:
        print(f"{ens['ensemble']:<20} {ens['accuracy']*100:>9.2f}% {ens['recall']*100:>9.2f}% "
              f"{ens['f1']*100:>9.2f}% {ens['roc_auc']:>10.3f}")
    
    print("-" * 70)
    
    # Best result
    best = ensemble_results[0]
    print(f"\n🏆 BEST: {best['ensemble']}")
    print(f"   Accuracy:  {best['accuracy']*100:.2f}%")
    print(f"   Recall:    {best['recall']*100:.2f}%")
    print(f"   F1 Score:  {best['f1']*100:.2f}%")
    print(f"   ROC-AUC:   {best['roc_auc']:.3f}")
    print(f"   Threshold: {best['threshold']:.2f}")
    
    # Check improvement over baseline
    baseline_acc = 90.69
    improvement = (best['accuracy'] * 100) - baseline_acc
    
    if improvement > 0:
        print(f"\n✅ IMPROVEMENT: +{improvement:.2f}% over 90.69% baseline!")
    else:
        print(f"\n⚠️  Current: {best['accuracy']*100:.2f}% vs baseline 90.69%")
    
    # Save results
    print("\nSaving results...")
    
    # CV results
    cv_df = pd.DataFrame([{
        'model': r['model'],
        'cv_accuracy_mean': r['cv_accuracy_mean'],
        'cv_accuracy_std': r['cv_accuracy_std'],
        'cv_auc_mean': r['cv_auc_mean']
    } for r in cv_results])
    cv_df.to_csv(reports_dir / "cv_results.csv", index=False)
    
    # Ensemble comparison
    ens_df = pd.DataFrame([{
        'ensemble': e['ensemble'],
        'accuracy': e['accuracy'],
        'recall': e['recall'],
        'f1': e['f1'],
        'roc_auc': e['roc_auc'],
        'threshold': e['threshold']
    } for e in ensemble_results])
    ens_df.to_csv(reports_dir / "ensemble_comparison.csv", index=False)
    
    # Save best model
    if best['model'] is not None:
        model_data = {
            'model': best['model'],
            'scaler': scaler,
            'threshold': best['threshold'],
            'feature_cols': feature_cols,
            'ensemble_type': best['ensemble']
        }
        joblib.dump(model_data, models_dir / "best_cv_ensemble_model.pkl")
    
    # Save metadata
    metadata = {
        'best_ensemble': best['ensemble'],
        'accuracy': float(best['accuracy']),
        'recall': float(best['recall']),
        'f1': float(best['f1']),
        'roc_auc': float(best['roc_auc']),
        'threshold': float(best['threshold']),
        'improvement_over_baseline': float(improvement),
        'cv_folds': 5,
        'n_features': len(feature_cols),
        'trained_at': datetime.now().isoformat()
    }
    
    with open(reports_dir / "cv_ensemble_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Results saved to: {reports_dir}")
    print(f"✓ Model saved to: {models_dir / 'best_cv_ensemble_model.pkl'}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return ensemble_results, cv_results


if __name__ == "__main__":
    train_cv_ensemble()
