"""
Test New Detection Model (Stacking Ensemble) with Medical Report
"""

import sys
sys.path.insert(0, '/Users/prajanv/CardioDetect/Milestone_2')

import joblib
import numpy as np
from pathlib import Path

# Import OCR pipeline
from pipeline.integrated_pipeline import EnhancedMedicalOCR

print("=" * 70)
print("TESTING NEW STACKING ENSEMBLE MODEL (91.30% Accuracy)")
print("=" * 70)

# Load new model
model_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/detection_v3_91target')
model = joblib.load(model_dir / 'detection_best.pkl')
scaler = joblib.load(model_dir / 'detection_scaler.pkl')
features = joblib.load(model_dir / 'detection_features.pkl')

print(f"\n✓ Loaded model: Stacking Ensemble (LR meta-learner)")
print(f"✓ Features required: {len(features)}")
print(f"  {features}")

# Test with medical report
report_path = '/Users/prajanv/CardioDetect/Milestone_2/Medical_report/Synthetic_report/SYN-016.png'
print(f"\n📄 Testing with: {report_path}")

# Run OCR
ocr = EnhancedMedicalOCR(verbose=False)
result = ocr.extract_from_file(report_path)

print("\n📊 EXTRACTED VALUES:")
print("-" * 40)
for key, value in sorted(result['fields'].items()):
    if value is not None:
        print(f"   {key}: {value}")

# Prepare features for model
# Map OCR fields to model features
fields = result['fields']

# Build feature vector
# Original features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
# Engineered: Age_HR_Ratio, BP_Chol_Product, Oldpeak_Slope, Age_BP_Risk, Exercise_Risk, Age_Squared, HR_Reserve, Chol_Age_Ratio, BP_HR_Index, Cardiac_Risk_Score

# Get values with defaults
age = fields.get('age', 50)
sex = 1 if fields.get('sex', 'male').lower() in ['male', 'm'] else 0
chest_pain = 0  # ASY by default
resting_bp = fields.get('systolic_bp', 130)
cholesterol = fields.get('total_cholesterol', 200)
fasting_bs = 1 if fields.get('diabetes', False) else 0
resting_ecg = 0  # Normal
max_hr = fields.get('heart_rate', 150)
exercise_angina = 0  # Default no
oldpeak = 0.0  # Default
st_slope = 1  # Flat

# Engineered features
age_hr_ratio = age / (max_hr + 1)
bp_chol_product = resting_bp * cholesterol / 10000
oldpeak_slope = oldpeak * (st_slope + 1)
age_bp_risk = 1 if age > 50 and resting_bp > 130 else 0
exercise_risk = exercise_angina * (oldpeak + 1)
age_squared = age ** 2 / 1000
hr_reserve = 220 - age - max_hr
chol_age_ratio = cholesterol / age
bp_hr_index = resting_bp / (max_hr + 1) * 100
cardiac_risk_score = (
    (1 if age > 55 else 0) + 
    sex + 
    fasting_bs + 
    exercise_angina +
    (1 if oldpeak > 1 else 0) +
    (1 if st_slope == 1 else 0)
)

# Create feature vector in correct order
feature_values = [
    age, sex, chest_pain, resting_bp, cholesterol,
    fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope,
    age_hr_ratio, bp_chol_product, oldpeak_slope, age_bp_risk, exercise_risk,
    age_squared, hr_reserve, chol_age_ratio, bp_hr_index, cardiac_risk_score
]

print("\n🔧 FEATURE VALUES:")
print("-" * 40)
for name, val in zip(features, feature_values):
    print(f"   {name}: {val}")

# Scale and predict
X = np.array([feature_values])
X_scaled = scaler.transform(X)

prediction = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0]

print("\n" + "=" * 70)
print("🔬 DETECTION RESULT")
print("=" * 70)
print(f"\n   Prediction: {'❤️ HEART DISEASE DETECTED' if prediction == 1 else '✅ NO HEART DISEASE'}")
print(f"\n   Confidence:")
print(f"      No Disease:  {proba[0]:.1%}")
print(f"      Disease:     {proba[1]:.1%}")

# Risk assessment
if prediction == 1:
    if proba[1] >= 0.8:
        risk = "HIGH RISK ⚠️"
    elif proba[1] >= 0.6:
        risk = "MODERATE RISK ⚡"
    else:
        risk = "BORDERLINE RISK ⚠️"
else:
    if proba[0] >= 0.8:
        risk = "LOW RISK ✅"
    else:
        risk = "BORDERLINE - MONITOR 👁️"

print(f"\n   Overall Assessment: {risk}")
print("=" * 70)
