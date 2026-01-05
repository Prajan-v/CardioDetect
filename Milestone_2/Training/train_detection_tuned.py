"""
Advanced Detection Model Training with Hyperparameter Optimization
Target: >90% accuracy
Methods: GridSearchCV, RandomizedSearchCV, Optuna
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# Optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except:
    HAS_OPTUNA = False
    print("⚠ Optuna not found, will use sklearn tuning only")

print("="*70)
print("ADVANCED DETECTION MODEL TRAINING - TARGET >90% ACCURACY")
print("="*70)

# Load and preprocess data
df = pd.read_csv('/tmp/heart_data/heart.csv')
print(f"\n📊 Dataset: {len(df)} samples")

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

# Features and target
feature_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
X = df[feature_cols]
y = df['HeartDisease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_overall = {'accuracy': 0, 'model': None, 'name': None}

# ============================================================
# 1. GridSearchCV for RandomForest
# ============================================================
print("\n" + "="*70)
print("[1/4] GridSearchCV - RandomForest")
print("="*70)

rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_rf = GridSearchCV(rf, rf_params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
print("   Searching... (this may take a minute)")
grid_rf.fit(X_train_scaled, y_train)

rf_best = grid_rf.best_estimator_
rf_acc = accuracy_score(y_test, rf_best.predict(X_test_scaled))
print(f"   ✓ Best params: {grid_rf.best_params_}")
print(f"   ✓ CV Score: {grid_rf.best_score_:.2%}")
print(f"   ✓ Test Accuracy: {rf_acc:.2%}")

if rf_acc > best_overall['accuracy']:
    best_overall = {'accuracy': rf_acc, 'model': rf_best, 'name': 'RandomForest (GridSearch)'}

# ============================================================
# 2. RandomizedSearchCV for XGBoost
# ============================================================
print("\n" + "="*70)
print("[2/4] RandomizedSearchCV - XGBoost")
print("="*70)

xgb_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2, 3]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
random_xgb = RandomizedSearchCV(xgb, xgb_params, n_iter=100, cv=cv, scoring='accuracy', 
                                 n_jobs=-1, random_state=42, verbose=0)
print("   Searching 100 random combinations...")
random_xgb.fit(X_train_scaled, y_train)

xgb_best = random_xgb.best_estimator_
xgb_acc = accuracy_score(y_test, xgb_best.predict(X_test_scaled))
print(f"   ✓ Best params: {random_xgb.best_params_}")
print(f"   ✓ CV Score: {random_xgb.best_score_:.2%}")
print(f"   ✓ Test Accuracy: {xgb_acc:.2%}")

if xgb_acc > best_overall['accuracy']:
    best_overall = {'accuracy': xgb_acc, 'model': xgb_best, 'name': 'XGBoost (RandomSearch)'}

# ============================================================
# 3. Optuna for LightGBM
# ============================================================
print("\n" + "="*70)
print("[3/4] Optuna - LightGBM (100 trials)")
print("="*70)

if HAS_OPTUNA:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    print("   Running 100 Optuna trials...")
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    lgbm_best = LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
    lgbm_best.fit(X_train_scaled, y_train)
    lgbm_acc = accuracy_score(y_test, lgbm_best.predict(X_test_scaled))
    print(f"   ✓ Best params: {study.best_params}")
    print(f"   ✓ CV Score: {study.best_value:.2%}")
    print(f"   ✓ Test Accuracy: {lgbm_acc:.2%}")
    
    if lgbm_acc > best_overall['accuracy']:
        best_overall = {'accuracy': lgbm_acc, 'model': lgbm_best, 'name': 'LightGBM (Optuna)'}
else:
    # Fallback without Optuna
    lgbm_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 100]
    }
    lgbm = LGBMClassifier(random_state=42, verbose=-1)
    grid_lgbm = GridSearchCV(lgbm, lgbm_params, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_lgbm.fit(X_train_scaled, y_train)
    lgbm_best = grid_lgbm.best_estimator_
    lgbm_acc = accuracy_score(y_test, lgbm_best.predict(X_test_scaled))
    print(f"   ✓ Test Accuracy: {lgbm_acc:.2%}")

# ============================================================
# 4. Stacking Ensemble
# ============================================================
print("\n" + "="*70)
print("[4/4] Stacking Ensemble")
print("="*70)

estimators = [
    ('rf', rf_best),
    ('xgb', xgb_best),
    ('lgbm', lgbm_best)
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    n_jobs=-1
)

print("   Training stacking ensemble...")
stacking.fit(X_train_scaled, y_train)
stacking_acc = accuracy_score(y_test, stacking.predict(X_test_scaled))
print(f"   ✓ Stacking Accuracy: {stacking_acc:.2%}")

if stacking_acc > best_overall['accuracy']:
    best_overall = {'accuracy': stacking_acc, 'model': stacking, 'name': 'Stacking Ensemble'}

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

results = {
    'RandomForest (GridSearch)': rf_acc,
    'XGBoost (RandomSearch)': xgb_acc,
    'LightGBM (Optuna)': lgbm_acc,
    'Stacking Ensemble': stacking_acc
}

for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    marker = "🏆" if acc == best_overall['accuracy'] else "  "
    target_marker = "✅" if acc >= 0.90 else "❌"
    print(f"   {marker} {name}: {acc:.2%} {target_marker}")

# Save best model
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

output_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/Final_models/detection')
output_dir.mkdir(parents=True, exist_ok=True)

# Save all tuned models
joblib.dump(rf_best, output_dir / "detection_rf_tuned.pkl")
joblib.dump(xgb_best, output_dir / "detection_xgb_tuned.pkl")
joblib.dump(lgbm_best, output_dir / "detection_lgbm_tuned.pkl")
joblib.dump(stacking, output_dir / "detection_stacking.pkl")
joblib.dump(scaler, output_dir / "detection_scaler.pkl")
joblib.dump(feature_cols, output_dir / "detection_features.pkl")

# Save best as primary
joblib.dump(best_overall['model'], output_dir / "detection_best.pkl")

print(f"   ✓ All models saved to: {output_dir}")

# Classification report for best
print("\n" + "="*70)
print(f"BEST MODEL: {best_overall['name']}")
print("="*70)
y_pred = best_overall['model'].predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

print("\n" + "="*70)
if best_overall['accuracy'] >= 0.90:
    print(f"🎯 TARGET ACHIEVED! BEST ACCURACY: {best_overall['accuracy']:.2%}")
else:
    print(f"📊 BEST ACCURACY: {best_overall['accuracy']:.2%} (Target: 90%)")
print("="*70)
