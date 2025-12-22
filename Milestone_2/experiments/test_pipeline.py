"""
End-to-End Pipeline Test

Tests the complete flow:
1. PDF Input → OCR Extraction
2. Feature Engineering
3. Classification Model (MLP)
4. Risk Category Output (LOW/MODERATE/HIGH)
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

# Import OCR module
from src.production_ocr import ProductionOCR


def load_classification_model():
    """Load the trained classification model."""
    model_path = PROJECT_ROOT / "models" / "final_classifier.pkl"
    meta_path = PROJECT_ROOT / "models" / "final_classifier_meta.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    return model, meta


def create_features_from_ocr(ocr_fields):
    """Create model features from OCR extracted fields."""
    
    # Default values for missing fields
    defaults = {
        "sex": 1,  # Male as default
        "age": 50,
        "smoking": 0,
        "bp_meds": 0,
        "hypertension": 0,
        "diabetes": 0,
        "total_cholesterol": 200,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "bmi": 25,
        "heart_rate": 75,
        "fasting_glucose": 100,
    }
    
    # Map OCR field names to model field names
    field_mapping = {
        "cholesterol": "total_cholesterol",
        "glucose": "fasting_glucose",
    }
    
    # Start with defaults
    patient = defaults.copy()
    
    # Update with OCR values
    for key, value in ocr_fields.items():
        mapped_key = field_mapping.get(key, key)
        if mapped_key in patient and value is not None:
            patient[mapped_key] = float(value)
    
    # Create DataFrame
    df = pd.DataFrame([patient])
    
    # Add derived features
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3)
    df["hypertension_flag"] = (df["systolic_bp"] >= 140).astype(int)
    df["high_cholesterol_flag"] = (df["total_cholesterol"] >= 240).astype(int)
    df["high_glucose_flag"] = (df["fasting_glucose"] >= 126).astype(int)
    df["obesity_flag"] = (df["bmi"] >= 30).astype(int)
    df["metabolic_syndrome_score"] = (
        df["hypertension_flag"] + df["high_cholesterol_flag"] + 
        df["high_glucose_flag"] + df["obesity_flag"] + df["diabetes"]
    )
    
    # Age groups
    for ag in ["<40", "40-49", "50-59", "60-69", "70+"]:
        df[f"age_group_{ag}"] = 0
    age = patient["age"]
    if age < 40:
        df["age_group_<40"] = 1
    elif age < 50:
        df["age_group_40-49"] = 1
    elif age < 60:
        df["age_group_50-59"] = 1
    elif age < 70:
        df["age_group_60-69"] = 1
    else:
        df["age_group_70+"] = 1
    
    # BMI categories
    for cat in ["Underweight", "Normal", "Overweight", "Obese"]:
        df[f"bmi_cat_{cat}"] = 0
    bmi = patient["bmi"]
    if bmi < 18.5:
        df["bmi_cat_Underweight"] = 1
    elif bmi < 25:
        df["bmi_cat_Normal"] = 1
    elif bmi < 30:
        df["bmi_cat_Overweight"] = 1
    else:
        df["bmi_cat_Obese"] = 1
    
    # Log transforms
    df["log_total_cholesterol"] = np.log1p(df["total_cholesterol"])
    df["log_fasting_glucose"] = np.log1p(df["fasting_glucose"])
    df["log_bmi"] = np.log1p(df["bmi"])
    
    # Interactions
    df["age_sbp_interaction"] = df["age"] * df["systolic_bp"]
    df["bmi_glucose_interaction"] = df["bmi"] * df["fasting_glucose"]
    df["age_smoking_interaction"] = df["age"] * df["smoking"]
    
    return df, patient


def run_pipeline(pdf_path):
    """Run the complete pipeline on a PDF."""
    print("=" * 70)
    print("CARDIODETECT - COMPLETE PIPELINE TEST")
    print("=" * 70)
    
    pdf_path = Path(pdf_path)
    print(f"\n📄 Input: {pdf_path.name}")
    
    # Step 1: OCR
    print("\n[Step 1/4] Running OCR extraction...")
    ocr = ProductionOCR()
    ocr_result = ocr.extract(str(pdf_path))
    
    if not ocr_result.success:
        print(f"❌ OCR failed: {ocr_result.error}")
        return None
    
    print(f"✓ OCR successful (confidence: {ocr_result.overall_confidence*100:.1f}%)")
    print(f"\nExtracted fields:")
    for field, value in ocr_result.fields.items():
        conf = ocr_result.field_confidences.get(field, 0) * 100
        print(f"  • {field}: {value} ({conf:.0f}% confidence)")
    
    # Step 2: Feature Engineering
    print("\n[Step 2/4] Creating model features...")
    features_df, patient_data = create_features_from_ocr(ocr_result.fields)
    
    print(f"✓ Created {features_df.shape[1]} features")
    print(f"\nPatient profile:")
    print(f"  • Age: {patient_data['age']} years")
    print(f"  • Sex: {'Male' if patient_data['sex']==1 else 'Female'}")
    print(f"  • BP: {patient_data['systolic_bp']}/{patient_data['diastolic_bp']} mmHg")
    print(f"  • Total Cholesterol: {patient_data['total_cholesterol']} mg/dL")
    print(f"  • Fasting Glucose: {patient_data['fasting_glucose']} mg/dL")
    
    # Step 3: Load Model
    print("\n[Step 3/4] Loading classification model...")
    model, meta = load_classification_model()
    print(f"✓ Model loaded: {meta.get('model_name', 'Unknown')}")
    print(f"  Trained at: {meta.get('trained_at', 'Unknown')}")
    print(f"  Accuracy: {meta.get('test_accuracy', 0)*100:.1f}%")
    
    # Ensure features match model expectations
    expected_features = meta.get('feature_names', [])
    for col in expected_features:
        if col not in features_df.columns:
            features_df[col] = 0
    features_df = features_df[expected_features]
    
    # Step 4: Prediction
    print("\n[Step 4/4] Running prediction...")
    
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    
    risk_labels = ["LOW", "MODERATE", "HIGH"]
    risk_level = risk_labels[prediction]
    confidence = probabilities[prediction] * 100
    
    # Risk descriptions
    risk_descriptions = {
        "LOW": "Low cardiovascular risk. Continue healthy lifestyle. Regular checkups recommended.",
        "MODERATE": "Moderate cardiovascular risk. Consider lifestyle modifications. Consult healthcare provider for risk management.",
        "HIGH": "High cardiovascular risk. Immediate medical attention recommended. Discuss treatment options with healthcare provider."
    }
    
    # Results
    print("\n" + "=" * 70)
    print("🏥 RISK ASSESSMENT RESULT")
    print("=" * 70)
    
    if risk_level == "LOW":
        emoji = "🟢"
    elif risk_level == "MODERATE":
        emoji = "🟡"
    else:
        emoji = "🔴"
    
    print(f"\n{emoji} Risk Category: {risk_level}")
    print(f"   Confidence: {confidence:.1f}%")
    print(f"\n   Probabilities:")
    print(f"     • LOW:      {probabilities[0]*100:5.1f}%")
    print(f"     • MODERATE: {probabilities[1]*100:5.1f}%")
    print(f"     • HIGH:     {probabilities[2]*100:5.1f}%")
    
    print(f"\n📋 Recommendation:")
    print(f"   {risk_descriptions[risk_level]}")
    
    # Key risk factors
    print(f"\n⚠️  Key Risk Factors Detected:")
    if patient_data['age'] >= 60:
        print(f"   • Age ({patient_data['age']} years) - elevated risk due to age")
    if patient_data['systolic_bp'] >= 140:
        print(f"   • High blood pressure ({patient_data['systolic_bp']} mmHg)")
    if patient_data['total_cholesterol'] >= 240:
        print(f"   • High cholesterol ({patient_data['total_cholesterol']} mg/dL)")
    if patient_data['smoking']:
        print(f"   • Smoking")
    if patient_data['diabetes']:
        print(f"   • Diabetes")
    
    # Create result dictionary
    result = {
        "success": True,
        "document": pdf_path.name,
        "risk_category": risk_level,
        "confidence": round(confidence, 2),
        "probabilities": {
            "LOW": round(probabilities[0] * 100, 2),
            "MODERATE": round(probabilities[1] * 100, 2),
            "HIGH": round(probabilities[2] * 100, 2)
        },
        "patient_data": patient_data,
        "ocr_confidence": round(ocr_result.overall_confidence * 100, 2),
        "recommendation": risk_descriptions[risk_level],
        "model_info": {
            "name": meta.get('model_name', 'unknown'),
            "accuracy": meta.get('test_accuracy', 0),
        }
    }
    
    # Save result
    result_path = PROJECT_ROOT / "reports" / "classification" / f"pipeline_result_{pdf_path.stem}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Result saved: {result_path}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    # Test with the specific PDF
    pdf_path = "/Users/prajanv/CardioDetect/Medical_report/Synthetic_report/ChatGPT Image Dec 4, 2025, 03_58_47 PM-compressed.pdf"
    
    if not Path(pdf_path).exists():
        print(f"ERROR: PDF not found: {pdf_path}")
    else:
        result = run_pipeline(pdf_path)
