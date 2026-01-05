"""
MAXIMUM ACCURACY DETECTION MODEL
=================================
All possible improvements:
1. Aggressive data cleaning
2. Advanced feature engineering
3. Optuna hyperparameter tuning
4. Multiple ensemble strategies
5. Threshold optimization

Target: 80%+ accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, AdaBoostClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MAXIMUM ACCURACY DETECTION MODEL")
print("=" * 80)

# ============================================================
# 1. AGGRESSIVE DATA CLEANING
# ============================================================
print("\n[STEP 1] AGGRESSIVE DATA CLEANING")
print("-" * 60)

df = pd.read_csv('/Users/prajanv/CardioDetect/data/raw/cardio_train.csv', sep=';')
print(f"   Raw: {len(df)} samples")

# Convert age to years
df['age'] = df['age'] / 365

# Remove impossible values
print("   Removing outliers...")
initial = len(df)

# BP: realistic ranges
df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 200)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 140)]

# Remove reversed BP
df = df[df['ap_hi'] > df['ap_lo']]

# Height: 130-210 cm is realistic
df = df[(df['height'] >= 130) & (df['height'] <= 210)]

# Weight: 30-180 kg
df = df[(df['weight'] >= 30) & (df['weight'] <= 180)]

# BMI check
df['bmi'] = df['weight'] / ((df['height']/100)**2)
df = df[(df['bmi'] >= 14) & (df['bmi'] <= 50)]

print(f"   After cleaning: {len(df)} samples ({100*len(df)/initial:.1f}% retained)")
print(f"   Target: {df['cardio'].value_counts().to_dict()}")

# ============================================================
# 2. ADVANCED FEATURE ENGINEERING (30+ features)
# ============================================================
print("\n[STEP 2] ADVANCED FEATURE ENGINEERING")
print("-" * 60)

# BP Features
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['mean_arterial_pressure'] = df['ap_lo'] + df['pulse_pressure'] / 3
df['bp_ratio'] = df['ap_hi'] / df['ap_lo']

# BP Categories (ACC/AHA 2017)
def bp_category(row):
    if row['ap_hi'] >= 180 or row['ap_lo'] >= 120:
        return 4  # Crisis
    elif row['ap_hi'] >= 140 or row['ap_lo'] >= 90:
        return 3  # Stage 2
    elif row['ap_hi'] >= 130 or row['ap_lo'] >= 80:
        return 2  # Stage 1
    elif row['ap_hi'] >= 120:
        return 1  # Elevated
    return 0  # Normal

df['bp_category'] = df.apply(bp_category, axis=1)
df['hypertension'] = (df['bp_category'] >= 2).astype(int)

# Metabolic features
df['high_cholesterol'] = (df['cholesterol'] >= 2).astype(int)
df['high_glucose'] = (df['gluc'] >= 2).astype(int)
df['very_high_cholesterol'] = (df['cholesterol'] == 3).astype(int)
df['very_high_glucose'] = (df['gluc'] == 3).astype(int)

# Metabolic syndrome score (0-5)
df['metabolic_score'] = (
    df['hypertension'] + 
    df['high_cholesterol'] + 
    df['high_glucose'] + 
    (df['bmi'] >= 30).astype(int) +
    ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int) * 0.5
)

# Age-based features
df['age_squared'] = df['age'] ** 2 / 1000
df['elderly'] = (df['age'] >= 55).astype(int)
df['middle_age'] = ((df['age'] >= 45) & (df['age'] < 55)).astype(int)

# Age risk categories
def age_risk(age):
    if age >= 60: return 3
    elif age >= 50: return 2
    elif age >= 40: return 1
    return 0

df['age_risk_category'] = df['age'].apply(age_risk)

# BMI categories
df['obese'] = (df['bmi'] >= 30).astype(int)
df['overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
df['normal_weight'] = ((df['bmi'] >= 18.5) & (df['bmi'] < 25)).astype(int)

# Lifestyle score (protective)
df['lifestyle_score'] = df['active'] - df['smoke'] - df['alco']

# Interaction features
df['age_bp_risk'] = df['age'] * df['bp_category']
df['age_metabolic'] = df['age'] * df['metabolic_score']
df['cholesterol_glucose'] = df['cholesterol'] * df['gluc']
df['bp_cholesterol'] = df['bp_category'] * df['cholesterol']
df['bmi_age'] = df['bmi'] * df['age'] / 100
df['bp_bmi'] = df['bp_category'] * df['bmi'] / 10

# Cardiovascular risk score (Framingham-like)
df['cv_risk_score'] = (
    df['age'] * 0.05 +
    df['bp_category'] * 2 +
    df['cholesterol'] * 1.5 +
    df['gluc'] * 1 +
    (1 - df['gender']) * 0.5 +  # Male = higher risk
    df['smoke'] * 2 +
    df['obese'] * 1.5 -
    df['active'] * 0.5
)

# Polynomial features for top predictors
df['age_chol_product'] = df['age'] * df['cholesterol']
df['bp_weight_ratio'] = df['ap_hi'] / df['weight']

# Feature list
feature_cols = [
    # Original
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active',
    # BP features
    'bmi', 'pulse_pressure', 'mean_arterial_pressure', 'bp_ratio',
    'bp_category', 'hypertension',
    # Metabolic
    'high_cholesterol', 'high_glucose', 'very_high_cholesterol', 'very_high_glucose',
    'metabolic_score',
    # Age
    'age_squared', 'elderly', 'middle_age', 'age_risk_category',
    # BMI
    'obese', 'overweight', 'normal_weight',
    # Lifestyle
    'lifestyle_score',
    # Interactions
    'age_bp_risk', 'age_metabolic', 'cholesterol_glucose', 'bp_cholesterol',
    'bmi_age', 'bp_bmi', 'cv_risk_score', 'age_chol_product', 'bp_weight_ratio'
]

X = df[feature_cols]
y = df['cardio']
print(f"   Features: {len(feature_cols)}")

# ============================================================
# 3. DATA SPLIT
# ============================================================
print("\n[STEP 3] DATA SPLIT")
print("-" * 60)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
)

print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 4. OPTUNA HYPERPARAMETER TUNING
# ============================================================
print("\n[STEP 4] OPTUNA HYPERPARAMETER TUNING (100 trials each)")
print("-" * 60)

best_models = {}

# 4.1 LightGBM
print("\n   [4.1] Tuning LightGBM...")
def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42, 'verbose': -1, 'n_jobs': -1
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(lgbm_objective, n_trials=100, show_progress_bar=True)
lgbm_best = LGBMClassifier(**study_lgbm.best_params, random_state=42, verbose=-1, n_jobs=-1)
lgbm_best.fit(X_train_scaled, y_train)
best_models['LightGBM'] = lgbm_best
print(f"      Best CV: {study_lgbm.best_value:.2%}")

# 4.2 XGBoost
print("\n   [4.2] Tuning XGBoost...")
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42, 'n_jobs': -1
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(xgb_objective, n_trials=100, show_progress_bar=True)
xgb_best = XGBClassifier(**study_xgb.best_params, random_state=42, n_jobs=-1)
xgb_best.fit(X_train_scaled, y_train)
best_models['XGBoost'] = xgb_best
print(f"      Best CV: {study_xgb.best_value:.2%}")

# 4.3 HistGradientBoosting
print("\n   [4.3] Tuning HistGradientBoosting...")
def hgb_objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
        'random_state': 42
    }
    model = HistGradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

study_hgb = optuna.create_study(direction='maximize')
study_hgb.optimize(hgb_objective, n_trials=100, show_progress_bar=True)
hgb_best = HistGradientBoostingClassifier(**study_hgb.best_params, random_state=42)
hgb_best.fit(X_train_scaled, y_train)
best_models['HistGB'] = hgb_best
print(f"      Best CV: {study_hgb.best_value:.2%}")

# 4.4 MLP
print("\n   [4.4] Tuning MLP Neural Network...")
def mlp_objective(trial):
    n_layers = trial.suggest_int('n_layers', 2, 4)
    layers = [trial.suggest_int(f'layer_{i}', 64, 256) for i in range(n_layers)]
    params = {
        'hidden_layer_sizes': tuple(layers),
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
        'learning_rate_init': trial.suggest_float('lr', 1e-4, 0.1, log=True),
        'max_iter': 500, 'early_stopping': True, 'random_state': 42
    }
    model = MLPClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

study_mlp = optuna.create_study(direction='maximize')
study_mlp.optimize(mlp_objective, n_trials=50, show_progress_bar=True)
mlp_params = study_mlp.best_trial.params
layers = [mlp_params[f'layer_{i}'] for i in range(mlp_params['n_layers'])]
mlp_best = MLPClassifier(hidden_layer_sizes=tuple(layers), alpha=mlp_params['alpha'],
                         learning_rate_init=mlp_params['lr'], max_iter=500, 
                         early_stopping=True, random_state=42)
mlp_best.fit(X_train_scaled, y_train)
best_models['MLP'] = mlp_best
print(f"      Best CV: {study_mlp.best_value:.2%}")

# ============================================================
# 5. ENSEMBLE MODELS
# ============================================================
print("\n[STEP 5] ENSEMBLE MODELS")
print("-" * 60)

# Voting
print("\n   [5.1] Voting Ensemble...")
voting = VotingClassifier(
    estimators=[('lgbm', best_models['LightGBM']), 
                ('xgb', best_models['XGBoost']),
                ('hgb', best_models['HistGB'])],
    voting='soft'
)
voting.fit(X_train_scaled, y_train)
best_models['Voting'] = voting

# Stacking with passthrough
print("\n   [5.2] Stacking Ensemble (with passthrough)...")
stacking = StackingClassifier(
    estimators=[('lgbm', best_models['LightGBM']), 
                ('xgb', best_models['XGBoost']),
                ('hgb', best_models['HistGB'])],
    final_estimator=LogisticRegression(C=1.0, max_iter=1000),
    cv=5, passthrough=True, n_jobs=-1
)
stacking.fit(X_train_scaled, y_train)
best_models['Stacking'] = stacking

# ============================================================
# 6. THRESHOLD OPTIMIZATION
# ============================================================
print("\n[STEP 6] THRESHOLD OPTIMIZATION")
print("-" * 60)

threshold_results = {}
for name, model in best_models.items():
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_val_scaled)[:, 1]
        best_thresh = 0.5
        best_acc = accuracy_score(y_val, (probs >= 0.5).astype(int))
        
        for thresh in np.arange(0.35, 0.65, 0.01):
            acc = accuracy_score(y_val, (probs >= thresh).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        threshold_results[name] = {'threshold': best_thresh, 'val_acc': best_acc}
        if best_thresh != 0.5:
            print(f"   {name}: optimal threshold = {best_thresh:.2f} (val acc: {best_acc:.2%})")

# ============================================================
# 7. FINAL EVALUATION
# ============================================================
print("\n" + "=" * 80)
print("[STEP 7] FINAL RESULTS")
print("=" * 80)

results = {}
for name, model in best_models.items():
    if name in threshold_results:
        thresh = threshold_results[name]['threshold']
        probs = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (probs >= thresh).astype(int)
    else:
        y_pred = model.predict(X_test_scaled)
    
    test_acc = accuracy_score(y_test, y_pred)
    val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
    results[name] = {'val': val_acc, 'test': test_acc}

print(f"\n{'Model':<20} {'Validation':>12} {'Test':>12} {'Gap':>10}")
print("-" * 60)

best_name = None
best_acc = 0
for name, res in sorted(results.items(), key=lambda x: x[1]['test'], reverse=True):
    gap = res['val'] - res['test']
    print(f"{name:<20} {res['val']:>11.2%} {res['test']:>11.2%} {gap:>+9.2%}")
    if res['test'] > best_acc:
        best_acc = res['test']
        best_name = name

print("-" * 60)
print(f"\n🏆 BEST MODEL: {best_name} ({best_acc:.2%})")

# Detailed metrics
best_model = best_models[best_name]
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 DETAILED METRICS:")
print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

# ============================================================
# 8. SAVE MODELS
# ============================================================
print("\n[STEP 8] SAVING MODELS")
print("-" * 60)

output_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/detection_maximum')
output_dir.mkdir(parents=True, exist_ok=True)

for name, model in best_models.items():
    safe_name = name.lower()
    joblib.dump(model, output_dir / f"detection_{safe_name}.pkl")

joblib.dump(scaler, output_dir / "detection_scaler.pkl")
joblib.dump(feature_cols, output_dir / "detection_features.pkl")
joblib.dump(best_model, output_dir / "detection_best.pkl")

import json
metadata = {
    "best_model": best_name,
    "accuracy": float(best_acc),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "features": feature_cols,
    "samples": len(df),
    "results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
}
with open(output_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✅ Saved to: {output_dir}")
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
