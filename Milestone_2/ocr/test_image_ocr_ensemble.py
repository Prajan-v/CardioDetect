"""
Test medical report image with OCR and ensemble prediction
"""
import sys
sys.path.insert(0, '/Users/prajanv/CardioDetect')

import cv2
import numpy as np
import pytesseract
from PIL import Image
import joblib
import pandas as pd
import re

# Load the image
image_path = "/Users/prajanv/CardioDetect/Medical_report/Synthetic_report/ChatGPT Image Dec 4, 2025, 03_58_47 PM.png"

print("=" * 80)
print("CardioDetect: Image OCR → Ensemble Risk Prediction")
print("=" * 80)

# Step 1: OCR the image
print("\n[1/3] Running OCR on medical report image...")
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Run Tesseract
text = pytesseract.image_to_string(gray)
print(f"\n--- Extracted Text ---")
print(text)
print("--- End Text ---")

# Step 2: Parse medical fields
print("\n[2/3] Parsing medical fields...")

fields = {}

# Age
age_match = re.search(r'age[:\s]*(\d+)', text, re.IGNORECASE)
if age_match:
    fields['age'] = int(age_match.group(1))

# Sex
sex_match = re.search(r'sex[:\s]*(male|female|m|f)', text, re.IGNORECASE)
if sex_match:
    sex_str = sex_match.group(1).lower()
    fields['sex'] = 1 if sex_str in ['male', 'm'] else 0
    fields['sex_label'] = 'Male' if fields['sex'] == 1 else 'Female'

# Blood Pressure (find systolic from "162/98" pattern)
bp_match = re.search(r'(?:blood pressure|bp)[:\s]*(\d+)/(\d+)', text, re.IGNORECASE)
if bp_match:
    fields['systolic_bp'] = int(bp_match.group(1))
    fields['diastolic_bp'] = int(bp_match.group(2))

# Heart Rate
hr_match = re.search(r'(?:heart rate|hr)[:\s]*(\d+)', text, re.IGNORECASE)
if hr_match:
    fields['heart_rate'] = int(hr_match.group(1))

# Total Cholesterol
chol_match = re.search(r'(?:total cholesterol)[:\s]*(\d+)', text, re.IGNORECASE)
if chol_match:
    fields['total_cholesterol'] = int(chol_match.group(1))

# HDL
hdl_match = re.search(r'hdl[^\d]*(\d+)', text, re.IGNORECASE)
if hdl_match:
    fields['hdl_cholesterol'] = int(hdl_match.group(1))

# LDL
ldl_match = re.search(r'ldl[^\d]*(\d+)', text, re.IGNORECASE)
if ldl_match:
    fields['ldl_cholesterol'] = int(ldl_match.group(1))

# Glucose
glucose_match = re.search(r'glucose[^\d]*(\d+)', text, re.IGNORECASE)
if glucose_match:
    fields['fasting_glucose'] = int(glucose_match.group(1))

# Diabetes indicator
if 'diabetes' in text.lower() or 'diabetic' in text.lower():
    fields['diabetes'] = 1

# Smoking
smoke_match = re.search(r'smok\w*[:\s]*(yes|no|non-?smoker|smoker)', text, re.IGNORECASE)
if smoke_match:
    smoke_str = smoke_match.group(1).lower()
    fields['smoking'] = 0 if 'no' in smoke_str or 'non' in smoke_str else 1

# Hypertension indicator
if 'hypertension' in text.lower():
    fields['hypertension'] = 1

print(f"Extracted fields: {fields}")

# Step 3: Load ensemble model and predict
print("\n[3/3] Running ensemble prediction...")

model_path = "/Users/prajanv/CardioDetect/models/best_cv_ensemble_model.pkl"
try:
    model_data = joblib.load(model_path)
    
    # Get the model components
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    threshold = model_data['threshold']
    ensemble_type = model_data['ensemble_type']
    
    print(f"Loaded: {ensemble_type} (threshold={threshold:.2f})")
    print(f"Features required: {len(feature_cols)}")
    
    # Build feature vector with defaults
    # Load training data to get median values for missing features
    train_df = pd.read_csv("/Users/prajanv/CardioDetect/data/split/train.csv")
    
    # Create feature row with median defaults (only for numeric columns in feature_cols)
    feature_row = {}
    for col in feature_cols:
        if col in train_df.columns:
            col_data = pd.to_numeric(train_df[col], errors='coerce')
            feature_row[col] = col_data.median()
        else:
            feature_row[col] = 0
    
    # Override with extracted values if available
    field_mapping = {
        'age': 'age',
        'sex': 'sex',
        'systolic_bp': 'systolic_bp',
        'diastolic_bp': 'diastolic_bp',
        'total_cholesterol': 'total_cholesterol',
        'hdl_cholesterol': 'hdl_cholesterol',
        'fasting_glucose': 'fasting_glucose',
        'smoking': 'smoking',
        'diabetes': 'diabetes',
        'heart_rate': 'heart_rate',
        'hypertension': 'hypertension'
    }
    
    for field_name, col_name in field_mapping.items():
        if field_name in fields and col_name in feature_cols:
            feature_row[col_name] = fields[field_name]
            print(f"  Using extracted: {col_name} = {fields[field_name]}")
    
    # Create DataFrame and scale
    X = pd.DataFrame([feature_row])[feature_cols]
    X_scaled = scaler.transform(X)
    
    # Predict
    proba = model.predict_proba(X_scaled)[0, 1]
    prediction = 1 if proba >= threshold else 0
    
    # Risk level
    if proba < 0.30:
        risk_level = "LOW"
    elif proba < 0.60:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"  Extracted Fields:")
    for k, v in fields.items():
        if k != 'sex_label':
            print(f"    • {k}: {v}")
    print(f"\n  Using: {ensemble_type}")
    print(f"  Probability: {proba:.4f} ({proba*100:.2f}%)")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Risk Level: {risk_level}")
    print(f"  Prediction: {'ELEVATED RISK' if prediction == 1 else 'LOW RISK'}")
    print("=" * 80)

except FileNotFoundError as e:
    print(f"Error: {e}")
