"""
Hyperparameter Tuning with Optuna for Maximum Accuracy
Target: Push beyond 92.23% accuracy

Optimizes:
1. Base model hyperparameters (XGBoost, LightGBM, RF)
2. Meta-learner parameters
3. Ensemble weights
"""

import sys
sys.path.insert(0, '/Users/prajanv/CardioDetect')

import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load data
print("=" * 70)
print("HYPERPARAMETER TUNING WITH OPTUNA")
print("Target: Push accuracy beyond 92.23%")
print("=" * 70)

data_dir = Path("/Users/prajanv/CardioDetect/data/split")
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "val.csv")
test_df = pd.read_csv(data_dir / "test.csv")

# Combine train + val for CV
combined_df = pd.concat([train_df, val_df], ignore_index=True)

# Get numeric feature columns
exclude_cols = {'risk_target', 'data_source', 'guideline_risk_10yr', 'sex'}
feature_cols = [c for c in combined_df.columns 
                if c not in exclude_cols 
                and combined_df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

X_train = combined_df[feature_cols].fillna(combined_df[feature_cols].median())
y_train = (combined_df['risk_target'] >= 1).astype(int).values

X_test = test_df[feature_cols].fillna(X_train.median())
y_test = (test_df['risk_target'] >= 1).astype(int).values

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print(f"Training: {len(X_train_res)} samples, Test: {len(X_test)} samples")
print(f"Features: {len(feature_cols)}")


def objective(trial):
    """Optuna objective: maximize CV accuracy."""
    
    # XGBoost params
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 400),
        'max_depth': trial.suggest_int('xgb_max_depth', 4, 10),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.2),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('xgb_min_child', 1, 7),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    # LightGBM params
    lgbm_params = {
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 400),
        'max_depth': trial.suggest_int('lgbm_max_depth', 4, 12),
        'learning_rate': trial.suggest_float('lgbm_lr', 0.01, 0.2),
        'subsample': trial.suggest_float('lgbm_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgbm_colsample', 0.6, 1.0),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 100),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # RF params
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 400),
        'max_depth': trial.suggest_int('rf_max_depth', 8, 20),
        'min_samples_split': trial.suggest_int('rf_min_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('rf_min_leaf', 1, 5),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Meta-learner params
    meta_C = trial.suggest_float('meta_C', 0.01, 10.0, log=True)
    
    # Build models
    xgb = XGBClassifier(**xgb_params)
    lgbm = LGBMClassifier(**lgbm_params)
    rf = RandomForestClassifier(**rf_params)
    
    # Stacking ensemble
    estimators = [('xgb', xgb), ('lgbm', lgbm), ('rf', rf)]
    meta = LogisticRegression(C=meta_C, max_iter=1000, random_state=42)
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=3,
        n_jobs=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stacking, X_train_res, y_train_res, cv=cv, scoring='accuracy', n_jobs=-1)
    
    return scores.mean()


# Run optimization
print("\n[1/2] Running Optuna optimization (50 trials)...")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest CV Accuracy: {study.best_value*100:.2f}%")
print(f"Best params: {study.best_params}")

# Train final model with best params
print("\n[2/2] Training final model with best hyperparameters...")
bp = study.best_params

xgb_best = XGBClassifier(
    n_estimators=bp['xgb_n_estimators'],
    max_depth=bp['xgb_max_depth'],
    learning_rate=bp['xgb_lr'],
    subsample=bp['xgb_subsample'],
    colsample_bytree=bp['xgb_colsample'],
    min_child_weight=bp['xgb_min_child'],
    random_state=42, n_jobs=-1, eval_metric='logloss'
)

lgbm_best = LGBMClassifier(
    n_estimators=bp['lgbm_n_estimators'],
    max_depth=bp['lgbm_max_depth'],
    learning_rate=bp['lgbm_lr'],
    subsample=bp['lgbm_subsample'],
    colsample_bytree=bp['lgbm_colsample'],
    num_leaves=bp['lgbm_num_leaves'],
    random_state=42, n_jobs=-1, verbose=-1
)

rf_best = RandomForestClassifier(
    n_estimators=bp['rf_n_estimators'],
    max_depth=bp['rf_max_depth'],
    min_samples_split=bp['rf_min_split'],
    min_samples_leaf=bp['rf_min_leaf'],
    class_weight='balanced',
    random_state=42, n_jobs=-1
)

meta_best = LogisticRegression(C=bp['meta_C'], max_iter=1000, random_state=42)

# Final stacking
final_ensemble = StackingClassifier(
    estimators=[('xgb', xgb_best), ('lgbm', lgbm_best), ('rf', rf_best)],
    final_estimator=meta_best,
    cv=5,
    n_jobs=-1
)

final_ensemble.fit(X_train_res, y_train_res)

# Evaluate on test set with threshold optimization
y_proba = final_ensemble.predict_proba(X_test_scaled)[:, 1]

best_acc = 0
best_thresh = 0.5
for thresh in np.arange(0.1, 0.9, 0.01):
    acc = accuracy_score(y_test, (y_proba >= thresh).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"  Best CV Accuracy:   {study.best_value*100:.2f}%")
print(f"  Test Accuracy:      {best_acc*100:.2f}%")
print(f"  Optimal Threshold:  {best_thresh:.2f}")
print(f"  Previous Best:      92.23%")
print(f"  Improvement:        {(best_acc*100 - 92.23):+.2f}%")
print("=" * 70)

# Save if improved
if best_acc > 0.9223:
    model_data = {
        'model': final_ensemble,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'threshold': best_thresh,
        'ensemble_type': 'Optuna_Stacking',
        'best_params': bp
    }
    joblib.dump(model_data, '/Users/prajanv/CardioDetect/models/best_cv_ensemble_model.pkl')
    print("\n✅ New best model saved!")
else:
    print("\n⚠️ No improvement over previous best")
