"""
Test OLD Detection Model (XGBoost) with Medical Report - For Comparison
"""

import sys
sys.path.insert(0, '/Users/prajanv/CardioDetect/Milestone_2')

import joblib
import numpy as np
from pathlib import Path

# Import OCR pipeline
from pipeline.integrated_pipeline import EnhancedMedicalOCR

print("=" * 70)
print("TESTING OLD DETECTION MODEL (XGBoost - 91.45% Accuracy)")
print("=" * 70)

# Load OLD model
model_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/Final_models/detection')
model = joblib.load(model_dir / 'detection_best.pkl')
scaler = joblib.load(model_dir / 'detection_scaler.pkl')
features = joblib.load(model_dir / 'detection_features.pkl')

print(f"\n✓ Loaded model: XGBoost (old model)")
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

# Prepare features for OLD model (13 features)
# Features: ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
fields = result['fields']

# Get values with defaults
age = fields.get('age', 50)
sex = 1 if fields.get('sex', 'male').lower() in ['male', 'm'] else 0
chest_pain = fields.get('cp', 0)  # ASY by default
resting_bp = fields.get('systolic_bp', 130)
cholesterol = fields.get('total_cholesterol', 200)
fasting_bs = 1 if fields.get('diabetes', False) else 0
resting_ecg = fields.get('restecg', 0)  # Normal
max_hr = fields.get('thalach', fields.get('heart_rate', 150))
exercise_angina = fields.get('exang', 0)
oldpeak = fields.get('oldpeak', 0.0)
st_slope = fields.get('slope', 1)  # Flat

# Create feature vector in correct order (11 features for old model)
feature_values = [
    age, sex, chest_pain, resting_bp, cholesterol,
    fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
]

print("\n🔧 FEATURE VALUES (Old Model - 11 features):")
print("-" * 40)
for name, val in zip(features, feature_values):
    print(f"   {name}: {val}")

# Scale and predict
X = np.array([feature_values])
X_scaled = scaler.transform(X)

prediction = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0]

print("\n" + "=" * 70)
print("🔬 DETECTION RESULT (OLD MODEL)")
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
