"""
Advanced Detection Model Training with Feature Selection & Hyperparameter Optimization
Target: >90% accuracy
Methods: LASSO, RFE, Correlation Analysis, GridSearchCV, RandomizedSearchCV, Optuna
Regularization: L1/L2 penalties, early stopping, feature pruning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
from sklearn.linear_model import LassoCV, LogisticRegression
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
# CatBoost removed - build issues on Python 3.14

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 80)
print("ADVANCED DETECTION MODEL - FEATURE SELECTION + HYPERPARAMETER OPTIMIZATION")
print("Target: >90% Accuracy | Dataset: Heart Failure Prediction (918 samples)")
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

# Feature Engineering - Add interaction features
print("\n[1.1] Feature Engineering...")
df['Age_HR_Ratio'] = df['Age'] / (df['MaxHR'] + 1)
df['BP_Chol_Product'] = df['RestingBP'] * df['Cholesterol'] / 10000
df['Oldpeak_Slope'] = df['Oldpeak'] * (df['ST_Slope'] + 1)
df['Age_BP_Risk'] = (df['Age'] > 50).astype(int) * (df['RestingBP'] > 130).astype(int)
df['Exercise_Risk'] = df['ExerciseAngina'] * (df['Oldpeak'] + 1)

# All features
all_feature_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
                    'Age_HR_Ratio', 'BP_Chol_Product', 'Oldpeak_Slope', 'Age_BP_Risk', 'Exercise_Risk']

X = df[all_feature_cols]
y = df['HeartDisease']
print(f"   Total Features: {len(all_feature_cols)}")
print(f"   Target distribution: {dict(y.value_counts())}")

# Split data (70/15/15)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)  # 0.176 of 85% = 15%
print(f"   Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 2. FEATURE SELECTION
# ============================================================
print("\n" + "=" * 80)
print("[STEP 2] FEATURE SELECTION")
print("=" * 80)

# 2.1 Correlation Analysis
print("\n[2.1] Correlation Analysis...")
correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
print("   Top 10 correlated features:")
for feat, corr in correlations.head(10).items():
    print(f"      {feat}: {corr:.4f}")

# Remove highly correlated features (multicollinearity)
corr_matrix = X_train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.85)]
print(f"   Highly correlated features to consider removing: {high_corr_features}")

# 2.2 LASSO Feature Selection
print("\n[2.2] LASSO Feature Selection (L1 Regularization)...")
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
lasso_importance = np.abs(lasso.coef_)
lasso_features = [f for f, imp in zip(all_feature_cols, lasso_importance) if imp > 0.01]
print(f"   Selected {len(lasso_features)} features: {lasso_features}")
print(f"   Best alpha: {lasso.alpha_:.4f}")

# 2.3 RFE (Recursive Feature Elimination)
print("\n[2.3] RFE with Random Forest...")
rf_for_rfe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfe = RFE(rf_for_rfe, n_features_to_select=10, step=1)
rfe.fit(X_train_scaled, y_train)
rfe_features = [f for f, selected in zip(all_feature_cols, rfe.support_) if selected]
print(f"   Selected {len(rfe_features)} features: {rfe_features}")
rfe_acc = accuracy_score(y_val, rfe.predict(X_val_scaled))
print(f"   RFE Validation Accuracy: {rfe_acc:.2%}")

# 2.4 Mutual Information
print("\n[2.4] Mutual Information...")
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
mi_ranking = pd.Series(mi_scores, index=all_feature_cols).sort_values(ascending=False)
print("   Top features by MI:")
for feat, score in mi_ranking.head(8).items():
    print(f"      {feat}: {score:.4f}")

# Combine feature selection results
print("\n[2.5] Combined Feature Selection...")
# Use features selected by at least 2 methods
feature_votes = {}
for feat in all_feature_cols:
    votes = 0
    if correlations.get(feat, 0) > 0.1:
        votes += 1
    if feat in lasso_features:
        votes += 1
    if feat in rfe_features:
        votes += 1
    if mi_ranking.get(feat, 0) > 0.05:
        votes += 1
    feature_votes[feat] = votes

# Select features with at least 2 votes
selected_features = [f for f, v in feature_votes.items() if v >= 2]
print(f"   Features selected by consensus: {selected_features}")

# Use selected features
X_train_sel = X_train[selected_features].values
X_val_sel = X_val[selected_features].values
X_test_sel = X_test[selected_features].values

# Re-scale with selected features
scaler_final = StandardScaler()
X_train_sel_scaled = scaler_final.fit_transform(X_train_sel)
X_val_sel_scaled = scaler_final.transform(X_val_sel)
X_test_sel_scaled = scaler_final.transform(X_test_sel)

# Store results
feature_selection_results = {
    'correlation': list(correlations.head(10).index),
    'lasso': lasso_features,
    'rfe': rfe_features,
    'selected': selected_features
}

# ============================================================
# 3. HYPERPARAMETER TUNING
# ============================================================
print("\n" + "=" * 80)
print("[STEP 3] HYPERPARAMETER TUNING")
print("=" * 80)

best_models = {}
all_results = {}

# 3.1 GridSearchCV - Random Forest with Regularization
print("\n[3.1] GridSearchCV - Random Forest (with regularization)...")
rf_params = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [5, 8, 10, 12, 15],
    'min_samples_split': [5, 10, 15, 20],
    'min_samples_leaf': [2, 4, 6, 8],
    'max_features': ['sqrt', 'log2', 0.5],
    'max_samples': [0.7, 0.8, 0.9],  # Bootstrap sample ratio (regularization)
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
grid_rf = GridSearchCV(rf, rf_params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
print("   Searching... (this may take a few minutes)")
grid_rf.fit(X_train_sel_scaled, y_train)

rf_best = grid_rf.best_estimator_
rf_val_acc = accuracy_score(y_val, rf_best.predict(X_val_sel_scaled))
rf_test_acc = accuracy_score(y_test, rf_best.predict(X_test_sel_scaled))
print(f"   ✓ Best CV Score: {grid_rf.best_score_:.2%}")
print(f"   ✓ Validation Accuracy: {rf_val_acc:.2%}")
print(f"   ✓ Test Accuracy: {rf_test_acc:.2%}")
print(f"   ✓ Best params: {grid_rf.best_params_}")
best_models['RandomForest'] = rf_best
all_results['RandomForest (GridSearch)'] = {'val': rf_val_acc, 'test': rf_test_acc}

# 3.2 RandomizedSearchCV - XGBoost with L1/L2 Regularization
print("\n[3.2] RandomizedSearchCV - XGBoost (with L1/L2 regularization)...")
xgb_params = {
    'n_estimators': [200, 300, 400, 500, 600],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7, 10],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 2.0],  # L1 regularization
    'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],  # L2 regularization
    'scale_pos_weight': [1.0, 1.2, 1.5],
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', tree_method='hist')
random_xgb = RandomizedSearchCV(xgb, xgb_params, n_iter=150, cv=cv, scoring='accuracy', 
                                 n_jobs=-1, random_state=42, verbose=0)
print("   Searching 150 random combinations...")
random_xgb.fit(X_train_sel_scaled, y_train)

xgb_best = random_xgb.best_estimator_
xgb_val_acc = accuracy_score(y_val, xgb_best.predict(X_val_sel_scaled))
xgb_test_acc = accuracy_score(y_test, xgb_best.predict(X_test_sel_scaled))
print(f"   ✓ Best CV Score: {random_xgb.best_score_:.2%}")
print(f"   ✓ Validation Accuracy: {xgb_val_acc:.2%}")
print(f"   ✓ Test Accuracy: {xgb_test_acc:.2%}")
print(f"   ✓ reg_alpha (L1): {random_xgb.best_params_.get('reg_alpha', 0)}")
print(f"   ✓ reg_lambda (L2): {random_xgb.best_params_.get('reg_lambda', 1)}")
best_models['XGBoost'] = xgb_best
all_results['XGBoost (RandomSearch)'] = {'val': xgb_val_acc, 'test': xgb_test_acc}

# 3.3 Optuna - LightGBM with Regularization
print("\n[3.3] Optuna - LightGBM (100 trials)...")

def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 15, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),  # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),  # L2
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),  # Regularization
        'random_state': 42,
        'verbose': -1,
        'force_col_wise': True,
    }
    
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train_sel_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study_lgbm = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
print("   Running 100 Optuna trials...")
study_lgbm.optimize(lgbm_objective, n_trials=100, show_progress_bar=True)

lgbm_best = LGBMClassifier(**study_lgbm.best_params, random_state=42, verbose=-1, force_col_wise=True)
lgbm_best.fit(X_train_sel_scaled, y_train)
lgbm_val_acc = accuracy_score(y_val, lgbm_best.predict(X_val_sel_scaled))
lgbm_test_acc = accuracy_score(y_test, lgbm_best.predict(X_test_sel_scaled))
print(f"   ✓ Best CV Score: {study_lgbm.best_value:.2%}")
print(f"   ✓ Validation Accuracy: {lgbm_val_acc:.2%}")
print(f"   ✓ Test Accuracy: {lgbm_test_acc:.2%}")
print(f"   ✓ reg_alpha (L1): {study_lgbm.best_params.get('reg_alpha', 0):.4f}")
print(f"   ✓ reg_lambda (L2): {study_lgbm.best_params.get('reg_lambda', 0):.4f}")
best_models['LightGBM'] = lgbm_best
all_results['LightGBM (Optuna)'] = {'val': lgbm_val_acc, 'test': lgbm_test_acc}

# 3.4 Optuna - Gradient Boosting
print("\n[3.4] Optuna - GradientBoosting (50 trials)...")

def gb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': 42,
    }
    
    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_train_sel_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study_gb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
print("   Running 50 Optuna trials...")
study_gb.optimize(gb_objective, n_trials=50, show_progress_bar=True)

gb_best = GradientBoostingClassifier(**study_gb.best_params, random_state=42)
gb_best.fit(X_train_sel_scaled, y_train)
gb_val_acc = accuracy_score(y_val, gb_best.predict(X_val_sel_scaled))
gb_test_acc = accuracy_score(y_test, gb_best.predict(X_test_sel_scaled))
print(f"   ✓ Best CV Score: {study_gb.best_value:.2%}")
print(f"   ✓ Validation Accuracy: {gb_val_acc:.2%}")
print(f"   ✓ Test Accuracy: {gb_test_acc:.2%}")
best_models['GradientBoosting'] = gb_best
all_results['GradientBoosting (Optuna)'] = {'val': gb_val_acc, 'test': gb_test_acc}

# ============================================================
# 4. ENSEMBLE METHODS
# ============================================================
print("\n" + "=" * 80)
print("[STEP 4] ENSEMBLE METHODS")
print("=" * 80)

# 4.1 Soft Voting Ensemble
print("\n[4.1] Soft Voting Ensemble...")
voting = VotingClassifier(
    estimators=[
        ('xgb', xgb_best),
        ('lgbm', lgbm_best),
        ('rf', rf_best),
        ('gb', gb_best),
    ],
    voting='soft',
    weights=[2, 2, 1, 1]  # Weight best performers higher (XGB, LGBM, RF, GB)
)
voting.fit(X_train_sel_scaled, y_train)
voting_val_acc = accuracy_score(y_val, voting.predict(X_val_sel_scaled))
voting_test_acc = accuracy_score(y_test, voting.predict(X_test_sel_scaled))
print(f"   ✓ Validation Accuracy: {voting_val_acc:.2%}")
print(f"   ✓ Test Accuracy: {voting_test_acc:.2%}")
best_models['Voting'] = voting
all_results['Voting Ensemble'] = {'val': voting_val_acc, 'test': voting_test_acc}

# 4.2 Stacking Ensemble with Regularized Meta-Learner
print("\n[4.2] Stacking Ensemble (Regularized Meta-Learner)...")
stacking = StackingClassifier(
    estimators=[
        ('xgb', xgb_best),
        ('lgbm', lgbm_best),
        ('rf', rf_best),
        ('gb', gb_best),
    ],
    final_estimator=LogisticRegression(C=0.5, max_iter=1000, random_state=42, penalty='l2'),  # L2 regularization
    cv=5,
    n_jobs=-1,
    passthrough=False
)
stacking.fit(X_train_sel_scaled, y_train)
stacking_val_acc = accuracy_score(y_val, stacking.predict(X_val_sel_scaled))
stacking_test_acc = accuracy_score(y_test, stacking.predict(X_test_sel_scaled))
print(f"   ✓ Validation Accuracy: {stacking_val_acc:.2%}")
print(f"   ✓ Test Accuracy: {stacking_test_acc:.2%}")
best_models['Stacking'] = stacking
all_results['Stacking Ensemble'] = {'val': stacking_val_acc, 'test': stacking_test_acc}

# ============================================================
# 5. FINAL RESULTS & METRICS
# ============================================================
print("\n" + "=" * 80)
print("[STEP 5] FINAL RESULTS")
print("=" * 80)

# Find best model
best_name = max(all_results.keys(), key=lambda k: all_results[k]['test'])
best_test_acc = all_results[best_name]['test']

print("\n📊 MODEL COMPARISON (sorted by test accuracy):")
print("-" * 60)
print(f"{'Model':<30} {'Validation':>12} {'Test':>12} {'Status':>10}")
print("-" * 60)

for name, accs in sorted(all_results.items(), key=lambda x: x[1]['test'], reverse=True):
    marker = "🏆" if name == best_name else "  "
    target_marker = "✅" if accs['test'] >= 0.90 else "❌"
    print(f"{marker} {name:<28} {accs['val']:>11.2%} {accs['test']:>11.2%} {target_marker:>10}")

print("-" * 60)

# Detailed metrics for best model
print(f"\n🏆 BEST MODEL: {best_name}")
print("=" * 60)

best_model = best_models[best_name.split(' (')[0].split()[0] if '(' in best_name else best_name]
y_pred_test = best_model.predict(X_test_sel_scaled)

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
    y_prob = best_model.predict_proba(X_test_sel_scaled)[:, 1]
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
except:
    pass

# ============================================================
# 6. SAVE MODELS
# ============================================================
print("\n" + "=" * 80)
print("[STEP 6] SAVING MODELS")
print("=" * 80)

output_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/detection_advanced')
output_dir.mkdir(parents=True, exist_ok=True)

# Save all models
for name, model in best_models.items():
    joblib.dump(model, output_dir / f"detection_{name.lower()}_advanced.pkl")
    print(f"   ✓ Saved: detection_{name.lower()}_advanced.pkl")

# Save scaler and features
joblib.dump(scaler_final, output_dir / "detection_scaler_advanced.pkl")
joblib.dump(selected_features, output_dir / "detection_features_advanced.pkl")
joblib.dump(feature_selection_results, output_dir / "feature_selection_results.pkl")
print(f"   ✓ Saved: scaler, features, feature selection results")

# Save best as primary
joblib.dump(best_model, output_dir / "detection_best_advanced.pkl")
print(f"   ✓ Saved best model: detection_best_advanced.pkl")

# Save metadata
metadata = {
    'best_model': best_name,
    'accuracy': best_test_acc,
    'selected_features': selected_features,
    'all_results': all_results,
    'dataset': '/tmp/heart_data/heart.csv',
    'samples': len(df),
    'train_size': len(X_train),
    'val_size': len(X_val),
    'test_size': len(X_test),
}
joblib.dump(metadata, output_dir / "training_metadata.pkl")

print(f"\n✅ All models saved to: {output_dir}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
if best_test_acc >= 0.90:
    print(f"🎯 TARGET ACHIEVED! BEST ACCURACY: {best_test_acc:.2%}")
else:
    print(f"📊 BEST ACCURACY: {best_test_acc:.2%} (Target: 90%)")
    print(f"   Gap to target: {0.90 - best_test_acc:.2%}")
print(f"   Model: {best_name}")
print(f"   Features: {len(selected_features)}")
print("=" * 80)
