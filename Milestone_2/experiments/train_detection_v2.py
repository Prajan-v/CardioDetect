"""
Train New Detection Model on Larger Dataset
Dataset: Heart Failure Prediction (918 samples, 5 combined datasets)
Source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from lightgbm import LGBMClassifier

print("="*60)
print("DETECTION MODEL TRAINING - LARGER DATASET")
print("="*60)

# Load data
df = pd.read_csv('/tmp/heart_data/heart.csv')
print(f"\n📊 Dataset: {len(df)} samples (3x larger than original UCI)")

# Encode categorical columns
print("\n[1/5] Preprocessing data...")
le = LabelEncoder()

# Sex: M=1, F=0
df['Sex'] = (df['Sex'] == 'M').astype(int)

# ChestPainType: ASY=0, ATA=1, NAP=2, TA=3
cp_map = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
df['ChestPainType'] = df['ChestPainType'].map(cp_map)

# RestingECG: Normal=0, ST=1, LVH=2
ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
df['RestingECG'] = df['RestingECG'].map(ecg_map)

# ExerciseAngina: Y=1, N=0
df['ExerciseAngina'] = (df['ExerciseAngina'] == 'Y').astype(int)

# ST_Slope: Up=0, Flat=1, Down=2
slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}
df['ST_Slope'] = df['ST_Slope'].map(slope_map)

# Handle zero cholesterol (missing values)
df['Cholesterol'] = df['Cholesterol'].replace(0, df['Cholesterol'].median())

# Features and target
feature_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
X = df[feature_cols]
y = df['HeartDisease']

print(f"   Features: {len(feature_cols)}")
print(f"   Target distribution: {dict(y.value_counts())}")

# Split data
print("\n[2/5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n[3/5] Training models...")

models = {
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    results[name] = {'model': model, 'accuracy': acc, 'cv_mean': cv_scores.mean()}
    print(f"   {name}: {acc:.2%} (CV: {cv_scores.mean():.2%})")

# Create ensemble
print("\n[4/5] Creating ensemble...")
best_models = [(name, results[name]['model']) for name in ['XGBoost', 'LightGBM', 'RandomForest']]
ensemble = VotingClassifier(estimators=best_models, voting='soft')
ensemble.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble.predict(X_test_scaled)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
print(f"   Ensemble (Voting): {ensemble_acc:.2%}")

# Best model
best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
best_acc = results[best_name]['accuracy']

# Use ensemble if better
if ensemble_acc > best_acc:
    best_model = ensemble
    best_name = "Ensemble"
    best_acc = ensemble_acc

print(f"\n[5/5] Saving best model...")
print(f"   Best: {best_name} ({best_acc:.2%})")

# Save models
output_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/detection_v2')
output_dir.mkdir(parents=True, exist_ok=True)

# Save each model
for name, data in results.items():
    joblib.dump(data['model'], output_dir / f"detection_{name.lower()}_v2.pkl")

# Save ensemble and scaler
joblib.dump(ensemble, output_dir / "detection_ensemble_v2.pkl")
joblib.dump(scaler, output_dir / "detection_scaler_v2.pkl")
joblib.dump(feature_cols, output_dir / "detection_features_v2.pkl")

# Save best model as primary
joblib.dump(best_model, output_dir / "detection_best_v2.pkl")

print(f"\n✅ Models saved to: {output_dir}")

# Classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred_ensemble, target_names=['No Disease', 'Disease']))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_ensemble)
print(f"   TN={cm[0,0]}, FP={cm[0,1]}")
print(f"   FN={cm[1,0]}, TP={cm[1,1]}")

print("\n" + "="*60)
print(f"🎯 NEW DETECTION MODEL ACCURACY: {best_acc:.2%}")
print(f"📈 Dataset size: 918 samples (vs 303 original)")
print("="*60)
