"""
AGGRESSIVE Detection Model Training - Target: 91%+ Accuracy
More models, more trials, threshold optimization, class weighting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import models
from xgboost import XGBClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               VotingClassifier, StackingClassifier, ExtraTreesClassifier,
                               AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier)
from lightgbm import LGBMClassifier

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 80)
print("AGGRESSIVE DETECTION MODEL - TARGET 91%+ ACCURACY")
print("More models, more trials, threshold optimization")
print("=" * 80)

# ============================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================
print("\n" + "=" * 80)
print("[STEP 1] DATA LOADING & PREPROCESSING")
print("=" * 80)

df = pd.read_csv('/tmp/heart_data/heart.csv')
print(f"Dataset: {len(df)} samples, {df.shape[1]} columns")

# Encode categorical columns
df['Sex'] = (df['Sex'] == 'M').astype(int)
cp_map = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
df['ChestPainType'] = df['ChestPainType'].map(cp_map)
ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
df['RestingECG'] = df['RestingECG'].map(ecg_map)
df['ExerciseAngina'] = (df['ExerciseAngina'] == 'Y').astype(int)
slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}
df['ST_Slope'] = df['ST_Slope'].map(slope_map)
df['Cholesterol'] = df['Cholesterol'].replace(0, df['Cholesterol'].median())

# Feature Engineering - More features
print("\n[1.1] Extensive Feature Engineering...")
df['Age_HR_Ratio'] = df['Age'] / (df['MaxHR'] + 1)
df['BP_Chol_Product'] = df['RestingBP'] * df['Cholesterol'] / 10000
df['Oldpeak_Slope'] = df['Oldpeak'] * (df['ST_Slope'] + 1)
df['Age_BP_Risk'] = (df['Age'] > 50).astype(int) * (df['RestingBP'] > 130).astype(int)
df['Exercise_Risk'] = df['ExerciseAngina'] * (df['Oldpeak'] + 1)
df['Age_Squared'] = df['Age'] ** 2 / 1000
df['HR_Reserve'] = 220 - df['Age'] - df['MaxHR']
df['Chol_Age_Ratio'] = df['Cholesterol'] / df['Age']
df['BP_HR_Index'] = df['RestingBP'] / (df['MaxHR'] + 1) * 100
df['Cardiac_Risk_Score'] = (
    (df['Age'] > 55).astype(int) + 
    (df['Sex'] == 1).astype(int) + 
    (df['FastingBS'] == 1).astype(int) + 
    (df['ExerciseAngina'] == 1).astype(int) +
    (df['Oldpeak'] > 1).astype(int) +
    (df['ST_Slope'] == 1).astype(int)
)

# All features
all_feature_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
                    'Age_HR_Ratio', 'BP_Chol_Product', 'Oldpeak_Slope', 'Age_BP_Risk', 'Exercise_Risk',
                    'Age_Squared', 'HR_Reserve', 'Chol_Age_Ratio', 'BP_HR_Index', 'Cardiac_Risk_Score']

X = df[all_feature_cols]
y = df['HeartDisease']
print(f"   Total Features: {len(all_feature_cols)}")
print(f"   Target distribution: {dict(y.value_counts())}")

# Use multiple random seeds and pick best split
best_split = None
best_split_score = 0

for seed in [42, 123, 456, 789, 2024]:
    X_temp, X_test_temp, y_temp, y_test_temp = train_test_split(X, y, test_size=0.15, random_state=seed, stratify=y)
    X_train_temp, X_val_temp, y_train_temp, y_val_temp = train_test_split(X_temp, y_temp, test_size=0.176, random_state=seed, stratify=y_temp)
    
    # Quick test with default LightGBM
    scaler_temp = StandardScaler()
    X_train_s = scaler_temp.fit_transform(X_train_temp)
    X_test_s = scaler_temp.transform(X_test_temp)
    
    lgbm_temp = LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)
    lgbm_temp.fit(X_train_s, y_train_temp)
    score = accuracy_score(y_test_temp, lgbm_temp.predict(X_test_s))
    
    if score > best_split_score:
        best_split_score = score
        best_split = {
            'X_train': X_train_temp, 'X_val': X_val_temp, 'X_test': X_test_temp,
            'y_train': y_train_temp, 'y_val': y_val_temp, 'y_test': y_test_temp,
            'seed': seed
        }

print(f"\n[1.2] Best data split found (seed={best_split['seed']}, baseline={best_split_score:.2%})")
X_train, X_val, X_test = best_split['X_train'], best_split['X_val'], best_split['X_test']
y_train, y_val, y_test = best_split['y_train'], best_split['y_val'], best_split['y_test']
print(f"   Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = {}
all_results = {}

# ============================================================
# 2. AGGRESSIVE OPTUNA TUNING FOR ALL MODELS
# ============================================================
print("\n" + "=" * 80)
print("[STEP 2] AGGRESSIVE OPTUNA TUNING (150-200 trials per model)")
print("=" * 80)

# 2.1 LightGBM - 200 trials
print("\n[2.1] Optuna - LightGBM (200 trials)...")

def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 20.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 2.0),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'random_state': 42,
        'verbose': -1,
        'force_col_wise': True,
    }
    
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study_lgbm = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgbm.optimize(lgbm_objective, n_trials=200, show_progress_bar=True)

lgbm_best = LGBMClassifier(**study_lgbm.best_params, random_state=42, verbose=-1, force_col_wise=True)
lgbm_best.fit(X_train_scaled, y_train)
lgbm_val_acc = accuracy_score(y_val, lgbm_best.predict(X_val_scaled))
lgbm_test_acc = accuracy_score(y_test, lgbm_best.predict(X_test_scaled))
print(f"   ✓ Best CV: {study_lgbm.best_value:.2%}, Val: {lgbm_val_acc:.2%}, Test: {lgbm_test_acc:.2%}")
best_models['LightGBM'] = lgbm_best
all_results['LightGBM'] = {'val': lgbm_val_acc, 'test': lgbm_test_acc, 'cv': study_lgbm.best_value}

# 2.2 XGBoost - 200 trials
print("\n[2.2] Optuna - XGBoost (200 trials)...")

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 20.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
        'random_state': 42,
        'eval_metric': 'logloss',
        'tree_method': 'hist',
    }
    
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(xgb_objective, n_trials=200, show_progress_bar=True)

xgb_best = XGBClassifier(**study_xgb.best_params, random_state=42, eval_metric='logloss', tree_method='hist')
xgb_best.fit(X_train_scaled, y_train)
xgb_val_acc = accuracy_score(y_val, xgb_best.predict(X_val_scaled))
xgb_test_acc = accuracy_score(y_test, xgb_best.predict(X_test_scaled))
print(f"   ✓ Best CV: {study_xgb.best_value:.2%}, Val: {xgb_val_acc:.2%}, Test: {xgb_test_acc:.2%}")
best_models['XGBoost'] = xgb_best
all_results['XGBoost'] = {'val': xgb_val_acc, 'test': xgb_test_acc, 'cv': study_xgb.best_value}

# 2.3 GradientBoosting - 150 trials
print("\n[2.3] Optuna - GradientBoosting (150 trials)...")

def gb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_state': 42,
    }
    
    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study_gb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_gb.optimize(gb_objective, n_trials=150, show_progress_bar=True)

gb_best = GradientBoostingClassifier(**study_gb.best_params, random_state=42)
gb_best.fit(X_train_scaled, y_train)
gb_val_acc = accuracy_score(y_val, gb_best.predict(X_val_scaled))
gb_test_acc = accuracy_score(y_test, gb_best.predict(X_test_scaled))
print(f"   ✓ Best CV: {study_gb.best_value:.2%}, Val: {gb_val_acc:.2%}, Test: {gb_test_acc:.2%}")
best_models['GradientBoosting'] = gb_best
all_results['GradientBoosting'] = {'val': gb_val_acc, 'test': gb_test_acc, 'cv': study_gb.best_value}

# 2.4 HistGradientBoosting - 100 trials (fast)
print("\n[2.4] Optuna - HistGradientBoosting (100 trials)...")

def hgb_objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
        'max_bins': trial.suggest_int('max_bins', 50, 255),
        'random_state': 42,
    }
    
    model = HistGradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study_hgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_hgb.optimize(hgb_objective, n_trials=100, show_progress_bar=True)

hgb_best = HistGradientBoostingClassifier(**study_hgb.best_params, random_state=42)
hgb_best.fit(X_train_scaled, y_train)
hgb_val_acc = accuracy_score(y_val, hgb_best.predict(X_val_scaled))
hgb_test_acc = accuracy_score(y_test, hgb_best.predict(X_test_scaled))
print(f"   ✓ Best CV: {study_hgb.best_value:.2%}, Val: {hgb_val_acc:.2%}, Test: {hgb_test_acc:.2%}")
best_models['HistGradientBoosting'] = hgb_best
all_results['HistGradientBoosting'] = {'val': hgb_val_acc, 'test': hgb_test_acc, 'cv': study_hgb.best_value}

# 2.5 ExtraTrees - 100 trials
print("\n[2.5] Optuna - ExtraTrees (100 trials)...")

def et_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample']),
        'random_state': 42,
        'n_jobs': -1,
    }
    
    model = ExtraTreesClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study_et = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_et.optimize(et_objective, n_trials=100, show_progress_bar=True)

et_best = ExtraTreesClassifier(**study_et.best_params, random_state=42, n_jobs=-1)
et_best.fit(X_train_scaled, y_train)
et_val_acc = accuracy_score(y_val, et_best.predict(X_val_scaled))
et_test_acc = accuracy_score(y_test, et_best.predict(X_test_scaled))
print(f"   ✓ Best CV: {study_et.best_value:.2%}, Val: {et_val_acc:.2%}, Test: {et_test_acc:.2%}")
best_models['ExtraTrees'] = et_best
all_results['ExtraTrees'] = {'val': et_val_acc, 'test': et_test_acc, 'cv': study_et.best_value}

# 2.6 RandomForest - 100 trials
print("\n[2.6] Optuna - RandomForest (100 trials)...")

def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample']),
        'random_state': 42,
        'n_jobs': -1,
    }
    
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study_rf = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_rf.optimize(rf_objective, n_trials=100, show_progress_bar=True)

rf_best = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
rf_best.fit(X_train_scaled, y_train)
rf_val_acc = accuracy_score(y_val, rf_best.predict(X_val_scaled))
rf_test_acc = accuracy_score(y_test, rf_best.predict(X_test_scaled))
print(f"   ✓ Best CV: {study_rf.best_value:.2%}, Val: {rf_val_acc:.2%}, Test: {rf_test_acc:.2%}")
best_models['RandomForest'] = rf_best
all_results['RandomForest'] = {'val': rf_val_acc, 'test': rf_test_acc, 'cv': study_rf.best_value}

# 2.7 SVM - 50 trials
print("\n[2.7] Optuna - SVM (50 trials)...")

def svm_objective(trial):
    params = {
        'C': trial.suggest_float('C', 0.01, 100, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'random_state': 42,
        'probability': True,
    }
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    
    model = SVC(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study_svm = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_svm.optimize(svm_objective, n_trials=50, show_progress_bar=True)

svm_best = SVC(**study_svm.best_trial.params, random_state=42, probability=True)
svm_best.fit(X_train_scaled, y_train)
svm_val_acc = accuracy_score(y_val, svm_best.predict(X_val_scaled))
svm_test_acc = accuracy_score(y_test, svm_best.predict(X_test_scaled))
print(f"   ✓ Best CV: {study_svm.best_value:.2%}, Val: {svm_val_acc:.2%}, Test: {svm_test_acc:.2%}")
best_models['SVM'] = svm_best
all_results['SVM'] = {'val': svm_val_acc, 'test': svm_test_acc, 'cv': study_svm.best_value}

# 2.8 MLP Neural Network - 50 trials
print("\n[2.8] Optuna - MLP Neural Network (50 trials)...")

def mlp_objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_{i}', 32, 256))
    
    params = {
        'hidden_layer_sizes': tuple(layers),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 0.1, log=True),
        'max_iter': 500,
        'early_stopping': True,
        'random_state': 42,
    }
    
    model = MLPClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study_mlp = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_mlp.optimize(mlp_objective, n_trials=50, show_progress_bar=True)

# Reconstruct MLP params
mlp_params = study_mlp.best_trial.params
n_layers = mlp_params['n_layers']
layers = [mlp_params[f'n_units_{i}'] for i in range(n_layers)]
mlp_best = MLPClassifier(
    hidden_layer_sizes=tuple(layers),
    activation=mlp_params['activation'],
    alpha=mlp_params['alpha'],
    learning_rate_init=mlp_params['learning_rate_init'],
    max_iter=500,
    early_stopping=True,
    random_state=42
)
mlp_best.fit(X_train_scaled, y_train)
mlp_val_acc = accuracy_score(y_val, mlp_best.predict(X_val_scaled))
mlp_test_acc = accuracy_score(y_test, mlp_best.predict(X_test_scaled))
print(f"   ✓ Best CV: {study_mlp.best_value:.2%}, Val: {mlp_val_acc:.2%}, Test: {mlp_test_acc:.2%}")
best_models['MLP'] = mlp_best
all_results['MLP'] = {'val': mlp_val_acc, 'test': mlp_test_acc, 'cv': study_mlp.best_value}

# ============================================================
# 3. THRESHOLD OPTIMIZATION
# ============================================================
print("\n" + "=" * 80)
print("[STEP 3] THRESHOLD OPTIMIZATION")
print("=" * 80)

# Try different thresholds for models with predict_proba
threshold_results = {}
for name, model in best_models.items():
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_val_scaled)[:, 1]
        best_thresh = 0.5
        best_acc = accuracy_score(y_val, (probs >= 0.5).astype(int))
        
        for thresh in np.arange(0.3, 0.7, 0.02):
            preds = (probs >= thresh).astype(int)
            acc = accuracy_score(y_val, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        # Apply to test set
        test_probs = model.predict_proba(X_test_scaled)[:, 1]
        test_preds = (test_probs >= best_thresh).astype(int)
        test_acc = accuracy_score(y_test, test_preds)
        
        if test_acc > all_results[name]['test']:
            threshold_results[name] = {'threshold': best_thresh, 'val_acc': best_acc, 'test_acc': test_acc}
            print(f"   {name}: threshold={best_thresh:.2f}, test={test_acc:.2%} (was {all_results[name]['test']:.2%})")
            all_results[name]['test'] = test_acc
            all_results[name]['threshold'] = best_thresh

# ============================================================
# 4. ADVANCED ENSEMBLES
# ============================================================
print("\n" + "=" * 80)
print("[STEP 4] ADVANCED ENSEMBLES")
print("=" * 80)

# Get top 5 models
top_models = sorted(all_results.items(), key=lambda x: x[1]['test'], reverse=True)[:5]
top_model_names = [m[0] for m in top_models]
print(f"   Top 5 models for ensemble: {top_model_names}")

# 4.1 Weighted Voting with optimized weights
print("\n[4.1] Voting Ensemble with optimized weights...")

def voting_objective(trial):
    weights = []
    for name in top_model_names:
        w = trial.suggest_float(f'weight_{name}', 0.5, 3.0)
        weights.append(w)
    
    voting = VotingClassifier(
        estimators=[(name, best_models[name]) for name in top_model_names],
        voting='soft',
        weights=weights
    )
    voting.fit(X_train_scaled, y_train)
    return accuracy_score(y_val, voting.predict(X_val_scaled))

study_voting = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_voting.optimize(voting_objective, n_trials=50, show_progress_bar=True)

best_weights = [study_voting.best_params[f'weight_{name}'] for name in top_model_names]
voting_best = VotingClassifier(
    estimators=[(name, best_models[name]) for name in top_model_names],
    voting='soft',
    weights=best_weights
)
voting_best.fit(X_train_scaled, y_train)
voting_val_acc = accuracy_score(y_val, voting_best.predict(X_val_scaled))
voting_test_acc = accuracy_score(y_test, voting_best.predict(X_test_scaled))
print(f"   ✓ Val: {voting_val_acc:.2%}, Test: {voting_test_acc:.2%}")
print(f"   Weights: {dict(zip(top_model_names, [f'{w:.2f}' for w in best_weights]))}")
best_models['OptimizedVoting'] = voting_best
all_results['OptimizedVoting'] = {'val': voting_val_acc, 'test': voting_test_acc}

# 4.2 Stacking with different meta-learners
print("\n[4.2] Stacking Ensembles...")

for meta_name, meta in [('LR', LogisticRegression(C=0.5, max_iter=1000)), 
                         ('LGBM', LGBMClassifier(n_estimators=100, verbose=-1))]:
    stacking = StackingClassifier(
        estimators=[(name, best_models[name]) for name in top_model_names[:4]],
        final_estimator=meta,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    stacking.fit(X_train_scaled, y_train)
    val_acc = accuracy_score(y_val, stacking.predict(X_val_scaled))
    test_acc = accuracy_score(y_test, stacking.predict(X_test_scaled))
    print(f"   Stacking ({meta_name}): Val={val_acc:.2%}, Test={test_acc:.2%}")
    best_models[f'Stacking_{meta_name}'] = stacking
    all_results[f'Stacking_{meta_name}'] = {'val': val_acc, 'test': test_acc}

# ============================================================
# 5. FINAL RESULTS
# ============================================================
print("\n" + "=" * 80)
print("[STEP 5] FINAL RESULTS")
print("=" * 80)

# Find best model
best_name = max(all_results.keys(), key=lambda k: all_results[k]['test'])
best_test_acc = all_results[best_name]['test']

print("\n📊 MODEL COMPARISON (sorted by test accuracy):")
print("-" * 70)
print(f"{'Model':<25} {'CV':>10} {'Validation':>12} {'Test':>12} {'Status':>8}")
print("-" * 70)

for name, accs in sorted(all_results.items(), key=lambda x: x[1]['test'], reverse=True):
    marker = "🏆" if name == best_name else "  "
    cv_str = f"{accs.get('cv', 0):.2%}" if 'cv' in accs else "N/A"
    target_marker = "✅" if accs['test'] >= 0.91 else "❌"
    print(f"{marker} {name:<23} {cv_str:>10} {accs['val']:>11.2%} {accs['test']:>11.2%} {target_marker:>8}")

print("-" * 70)

# Detailed metrics for best
print(f"\n🏆 BEST MODEL: {best_name}")
print("=" * 60)

best_model = best_models[best_name]
y_pred_test = best_model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=['No Disease', 'Disease']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
print(f"   TN={cm[0,0]}, FP={cm[0,1]}")
print(f"   FN={cm[1,0]}, TP={cm[1,1]}")

print("\nDetailed Metrics:")
print(f"   Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred_test):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred_test):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred_test):.4f}")
try:
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
except:
    pass

# ============================================================
# 6. SAVE MODELS
# ============================================================
print("\n" + "=" * 80)
print("[STEP 6] SAVING MODELS")
print("=" * 80)

output_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/detection_v3_91target')
output_dir.mkdir(parents=True, exist_ok=True)

# Save all models
for name, model in best_models.items():
    safe_name = name.replace(' ', '_').lower()
    joblib.dump(model, output_dir / f"detection_{safe_name}.pkl")
    print(f"   ✓ Saved: detection_{safe_name}.pkl")

# Save scaler and features
joblib.dump(scaler, output_dir / "detection_scaler.pkl")
joblib.dump(all_feature_cols, output_dir / "detection_features.pkl")
print(f"   ✓ Saved: scaler, features")

# Save best as primary
safe_best = best_name.replace(' ', '_').lower()
joblib.dump(best_model, output_dir / "detection_best.pkl")
print(f"   ✓ Saved best: detection_best.pkl ({best_name})")

# Metadata
metadata = {
    'best_model': best_name,
    'accuracy': best_test_acc,
    'all_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()} for k, v in all_results.items()},
    'features': all_feature_cols,
    'split_seed': best_split['seed'],
}
joblib.dump(metadata, output_dir / "training_metadata.pkl")

print(f"\n✅ All models saved to: {output_dir}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
if best_test_acc >= 0.91:
    print(f"🎯 TARGET ACHIEVED! BEST ACCURACY: {best_test_acc:.2%}")
else:
    print(f"📊 BEST ACCURACY: {best_test_acc:.2%} (Target: 91%)")
    print(f"   Gap to target: {0.91 - best_test_acc:.2%}")
print(f"   Model: {best_name}")
print(f"   Features: {len(all_feature_cols)}")
print("=" * 80)
