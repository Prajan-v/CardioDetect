"""
Detection Model Analysis: Calibration, OCR, and Data Quality Check
"""

import sys
sys.path.insert(0, '/Users/prajanv/CardioDetect/Milestone_2')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DETECTION MODEL ANALYSIS: CALIBRATION & DATA QUALITY")
print("=" * 80)

# ============================================================
# 1. LOAD AND ANALYZE DATA
# ============================================================
print("\n" + "=" * 80)
print("[1] DATA QUALITY ANALYSIS")
print("=" * 80)

df = pd.read_csv('/tmp/heart_data/heart.csv')
print(f"\nDataset: {len(df)} samples, {df.shape[1]} columns")

# Check for missing values
print("\n[1.1] Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✓ No missing values")
else:
    for col, count in missing[missing > 0].items():
        print(f"   ⚠ {col}: {count} missing ({count/len(df)*100:.1f}%)")

# Check for zero values (potential missing encoded as 0)
print("\n[1.2] Suspicious Zero Values:")
for col in ['Cholesterol', 'RestingBP', 'MaxHR']:
    if col in df.columns:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            print(f"   ⚠ {col}: {zero_count} zeros ({zero_count/len(df)*100:.1f}%)")
            print(f"      -> These may be missing values encoded as 0!")

# Class distribution
print("\n[1.3] Class Distribution:")
class_dist = df['HeartDisease'].value_counts()
print(f"   No Disease (0): {class_dist[0]} ({class_dist[0]/len(df)*100:.1f}%)")
print(f"   Disease (1):    {class_dist[1]} ({class_dist[1]/len(df)*100:.1f}%)")
imbalance_ratio = max(class_dist) / min(class_dist)
print(f"   Imbalance Ratio: {imbalance_ratio:.2f}:1")
if imbalance_ratio < 1.5:
    print("   ✓ Classes are reasonably balanced")
else:
    print("   ⚠ Moderate class imbalance detected")

# Feature statistics
print("\n[1.4] Feature Statistics:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
stats = df[numeric_cols].describe().T
print(stats[['mean', 'std', 'min', 'max']])

# Check for outliers (values > 3 std from mean)
print("\n[1.5] Potential Outliers (>3 std):")
for col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
    if col in df.columns:
        mean, std = df[col].mean(), df[col].std()
        outliers = ((df[col] < mean - 3*std) | (df[col] > mean + 3*std)).sum()
        if outliers > 0:
            print(f"   {col}: {outliers} outliers")

# Check for data leakage - correlation with target
print("\n[1.6] Feature Correlation with Target:")
# Encode categorical columns first
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_encoded[col] = pd.Categorical(df_encoded[col]).codes

correlations = df_encoded.corr()['HeartDisease'].drop('HeartDisease').abs().sort_values(ascending=False)
print("   Top correlations:")
for feat, corr in correlations.head(8).items():
    flag = "⚠ VERY HIGH" if corr > 0.7 else ("HIGH" if corr > 0.5 else "")
    print(f"      {feat}: {corr:.4f} {flag}")

if correlations.max() > 0.8:
    print("\n   ⚠ WARNING: Very high correlation detected - potential data leakage!")

# ============================================================
# 2. CALIBRATION ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("[2] CALIBRATION ANALYSIS")
print("=" * 80)

# Prepare data (same as training)
df_work = df.copy()
df_work['Sex'] = (df_work['Sex'] == 'M').astype(int)
cp_map = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
df_work['ChestPainType'] = df_work['ChestPainType'].map(cp_map)
ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
df_work['RestingECG'] = df_work['RestingECG'].map(ecg_map)
df_work['ExerciseAngina'] = (df_work['ExerciseAngina'] == 'Y').astype(int)
slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}
df_work['ST_Slope'] = df_work['ST_Slope'].map(slope_map)
df_work['Cholesterol'] = df_work['Cholesterol'].replace(0, df_work['Cholesterol'].median())

feature_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
X = df_work[feature_cols]
y = df_work['HeartDisease']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load the new model (we'll use LightGBM from the ensemble for calibration)
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train base model
print("\n[2.1] Training base model for calibration comparison...")
base_model = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
base_model.fit(X_train_scaled, y_train)

# Get uncalibrated probabilities
y_prob_uncal = base_model.predict_proba(X_test_scaled)[:, 1]

# Create calibrated model (Platt scaling = sigmoid)
print("[2.2] Applying calibration methods...")
calibrated_sigmoid = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
calibrated_sigmoid.fit(X_train_scaled, y_train)
y_prob_sigmoid = calibrated_sigmoid.predict_proba(X_test_scaled)[:, 1]

# Isotonic regression
calibrated_isotonic = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated_isotonic.fit(X_train_scaled, y_train)
y_prob_isotonic = calibrated_isotonic.predict_proba(X_test_scaled)[:, 1]

# Compute metrics
print("\n[2.3] Calibration Metrics:")
print("-" * 60)
print(f"{'Method':<20} {'Brier Score':>15} {'Log Loss':>15} {'Accuracy':>12}")
print("-" * 60)

for name, probs in [('Uncalibrated', y_prob_uncal), 
                     ('Platt (Sigmoid)', y_prob_sigmoid),
                     ('Isotonic', y_prob_isotonic)]:
    brier = brier_score_loss(y_test, probs)
    logloss = log_loss(y_test, probs)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"{name:<20} {brier:>15.4f} {logloss:>15.4f} {acc:>11.2%}")

print("-" * 60)
print("Lower Brier Score = Better calibration")

# Analyze calibration bins
print("\n[2.4] Calibration by Probability Bin:")
print("-" * 60)
print(f"{'Bin':<15} {'Predicted':>12} {'Actual':>12} {'Count':>10} {'Gap':>10}")
print("-" * 60)

for name, probs in [('Uncalibrated', y_prob_uncal), ('Platt', y_prob_sigmoid)]:
    print(f"\n{name}:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins)-1):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() > 0:
            pred_mean = probs[mask].mean()
            actual_mean = y_test.values[mask].mean()
            gap = abs(pred_mean - actual_mean)
            flag = "⚠" if gap > 0.1 else "✓"
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}      {pred_mean:>10.2%}   {actual_mean:>10.2%}   {mask.sum():>8}   {gap:>8.2%} {flag}")

# ============================================================
# 3. DATA QUALITY ISSUES SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("[3] ISSUES & RECOMMENDATIONS")
print("=" * 80)

issues_found = []

# Check cholesterol zeros
chol_zeros = (df['Cholesterol'] == 0).sum()
if chol_zeros > 0:
    issues_found.append(f"Cholesterol has {chol_zeros} zeros (likely missing values)")

# Check high correlations
if correlations.max() > 0.6:
    top_corr_feat = correlations.idxmax()
    issues_found.append(f"Feature '{top_corr_feat}' has high correlation ({correlations.max():.2f}) with target")

# Check imbalance
if imbalance_ratio > 1.3:
    issues_found.append(f"Class imbalance: {imbalance_ratio:.1f}:1 ratio")

print("\n📋 ISSUES FOUND:")
if issues_found:
    for i, issue in enumerate(issues_found, 1):
        print(f"   {i}. {issue}")
else:
    print("   ✓ No major issues detected")

print("\n💡 RECOMMENDATIONS:")
print("   1. Use Platt (Sigmoid) calibration for better probability estimates")
print("   2. Handle Cholesterol=0 values (impute with median or drop)")
print("   3. ST_Slope is highly predictive - ensure OCR extracts this field")
print("   4. ExerciseAngina is critical - improve OCR for this field")

# Save calibrated model
print("\n" + "=" * 80)
print("[4] SAVING CALIBRATED MODEL")
print("=" * 80)

output_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/detection_calibrated')
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(calibrated_sigmoid, output_dir / 'detection_lgbm_calibrated.pkl')
joblib.dump(scaler, output_dir / 'detection_scaler.pkl')
joblib.dump(feature_cols, output_dir / 'detection_features.pkl')

print(f"   ✓ Saved calibrated model to: {output_dir}")
print("=" * 80)
