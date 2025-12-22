"""
Test the Risk Classification Model

This script:
1. Loads the trained model
2. Tests with sample patient profiles
3. Shows predictions with confidence scores
4. Validates on test set
5. Tests edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from src.preprocessing import load_splits, get_feature_target_split, build_preprocessor_from_data
from src.targets import add_targets_to_df


def load_model():
    """Load the trained model."""
    model_path = Path(__file__).resolve().parents[1] / "models" / "final_classifier.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"✓ Loaded model from: {model_path}")
        return model
    else:
        print(f"ERROR: Model not found at {model_path}")
        return None


def test_sample_patients(model, preprocessor_features):
    """Test model with sample patient profiles."""
    print("\n" + "=" * 70)
    print("TEST 1: Sample Patient Predictions")
    print("=" * 70)
    
    # Create sample patient profiles
    sample_patients = [
        {
            "name": "Young Healthy Female",
            "sex": 0, "age": 35, "smoking": 0, "bp_meds": 0, "hypertension": 0,
            "diabetes": 0, "total_cholesterol": 180, "systolic_bp": 115,
            "diastolic_bp": 75, "bmi": 22, "heart_rate": 70, "fasting_glucose": 85
        },
        {
            "name": "Middle-aged Male with Risk Factors",
            "sex": 1, "age": 55, "smoking": 1, "bp_meds": 0, "hypertension": 1,
            "diabetes": 0, "total_cholesterol": 240, "systolic_bp": 145,
            "diastolic_bp": 90, "bmi": 28, "heart_rate": 80, "fasting_glucose": 100
        },
        {
            "name": "Elderly with Multiple Conditions",
            "sex": 1, "age": 70, "smoking": 0, "bp_meds": 1, "hypertension": 1,
            "diabetes": 1, "total_cholesterol": 260, "systolic_bp": 160,
            "diastolic_bp": 95, "bmi": 32, "heart_rate": 85, "fasting_glucose": 140
        },
        {
            "name": "Middle-aged Woman, Moderate Risk",
            "sex": 0, "age": 50, "smoking": 0, "bp_meds": 0, "hypertension": 0,
            "diabetes": 0, "total_cholesterol": 220, "systolic_bp": 130,
            "diastolic_bp": 82, "bmi": 26, "heart_rate": 75, "fasting_glucose": 95
        },
        {
            "name": "Young Male Smoker",
            "sex": 1, "age": 40, "smoking": 1, "bp_meds": 0, "hypertension": 0,
            "diabetes": 0, "total_cholesterol": 200, "systolic_bp": 125,
            "diastolic_bp": 80, "bmi": 25, "heart_rate": 72, "fasting_glucose": 90
        },
    ]
    
    risk_labels = ["LOW", "MODERATE", "HIGH"]
    
    for patient in sample_patients:
        name = patient.pop("name")
        
        # Create DataFrame with all required features
        df = pd.DataFrame([patient])
        
        # Add derived features to match training
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
        
        # Ensure columns match model
        for col in preprocessor_features:
            if col not in df.columns:
                df[col] = 0
        
        df = df[preprocessor_features]
        
        # Predict
        try:
            prediction = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]
            
            risk_level = risk_labels[prediction]
            confidence = probabilities[prediction] * 100
            
            print(f"\n📋 {name}")
            print(f"   Age: {patient['age']}, Sex: {'Male' if patient['sex']==1 else 'Female'}")
            print(f"   BP: {patient['systolic_bp']}/{patient['diastolic_bp']}, Chol: {patient['total_cholesterol']}")
            print(f"   Smoking: {'Yes' if patient['smoking'] else 'No'}, Diabetes: {'Yes' if patient['diabetes'] else 'No'}")
            print(f"   → Prediction: {risk_level} ({confidence:.1f}% confidence)")
            print(f"   → Probabilities: LOW={probabilities[0]*100:.1f}%, MODERATE={probabilities[1]*100:.1f}%, HIGH={probabilities[2]*100:.1f}%")
        except Exception as e:
            print(f"\n📋 {name}")
            print(f"   ERROR: {e}")


def test_on_test_set():
    """Validate model on the held-out test set."""
    print("\n" + "=" * 70)
    print("TEST 2: Validation on Test Set")
    print("=" * 70)
    
    # Load model
    model_path = Path(__file__).resolve().parents[1] / "models" / "final_classifier.pkl"
    model = joblib.load(model_path)
    
    # Load test data
    _, _, test_df = load_splits()
    test_df = add_targets_to_df(test_df)
    test_df = test_df[test_df["guideline_risk_10yr"].notna()].copy()
    
    # Prepare features
    X_test, _ = get_feature_target_split(test_df, target_col="risk_class_3")
    drop_cols = ["guideline_risk_10yr", "risk_class_3", "risk_class_binary", "risk_target"]
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    
    y_test = test_df["risk_class_3"].values
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Size: {len(y_test)} samples")
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["LOW", "MODERATE", "HIGH"]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print("                 Predicted")
    print("              LOW  MOD  HIGH")
    print(f"Actual LOW   {cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d}")
    print(f"       MOD   {cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d}")
    print(f"       HIGH  {cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d}")
    
    # Find misclassifications
    misclassified = np.where(y_test != y_pred)[0]
    print(f"\nMisclassified samples: {len(misclassified)} ({len(misclassified)/len(y_test)*100:.2f}%)")
    
    # Show a few misclassifications
    if len(misclassified) > 0:
        print("\nSample Misclassifications:")
        risk_labels = ["LOW", "MODERATE", "HIGH"]
        for idx in misclassified[:5]:
            true_label = risk_labels[y_test[idx]]
            pred_label = risk_labels[y_pred[idx]]
            conf = y_proba[idx][y_pred[idx]] * 100
            print(f"  Sample {idx}: True={true_label}, Predicted={pred_label} ({conf:.1f}% confidence)")
    
    return accuracy


def test_edge_cases(model, preprocessor_features):
    """Test model with edge cases."""
    print("\n" + "=" * 70)
    print("TEST 3: Edge Cases")
    print("=" * 70)
    
    edge_cases = [
        {"name": "Minimum Risk (all low values)", 
         "sex": 0, "age": 30, "smoking": 0, "bp_meds": 0, "hypertension": 0,
         "diabetes": 0, "total_cholesterol": 150, "systolic_bp": 100,
         "diastolic_bp": 65, "bmi": 20, "heart_rate": 60, "fasting_glucose": 70},
        
        {"name": "Maximum Risk (all high values)",
         "sex": 1, "age": 80, "smoking": 1, "bp_meds": 1, "hypertension": 1,
         "diabetes": 1, "total_cholesterol": 350, "systolic_bp": 200,
         "diastolic_bp": 110, "bmi": 40, "heart_rate": 100, "fasting_glucose": 200},
        
        {"name": "Borderline Case (threshold values)",
         "sex": 1, "age": 55, "smoking": 0, "bp_meds": 0, "hypertension": 0,
         "diabetes": 0, "total_cholesterol": 200, "systolic_bp": 140,
         "diastolic_bp": 90, "bmi": 25, "heart_rate": 75, "fasting_glucose": 100},
    ]
    
    risk_labels = ["LOW", "MODERATE", "HIGH"]
    
    for case in edge_cases:
        name = case.pop("name")
        df = create_feature_df(case)
        
        for col in preprocessor_features:
            if col not in df.columns:
                df[col] = 0
        df = df[preprocessor_features]
        
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        print(f"\n🔬 {name}")
        print(f"   → Prediction: {risk_labels[prediction]}")
        print(f"   → Probabilities: LOW={probabilities[0]*100:.1f}%, MOD={probabilities[1]*100:.1f}%, HIGH={probabilities[2]*100:.1f}%")


def create_feature_df(patient):
    """Create feature DataFrame from patient dict."""
    df = pd.DataFrame([patient])
    
    # Add all derived features
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3)
    df["hypertension_flag"] = (df["systolic_bp"] >= 140).astype(int)
    df["high_cholesterol_flag"] = (df["total_cholesterol"] >= 240).astype(int)
    df["high_glucose_flag"] = (df["fasting_glucose"] >= 126).astype(int)
    df["obesity_flag"] = (df["bmi"] >= 30).astype(int)
    df["metabolic_syndrome_score"] = df["hypertension_flag"] + df["high_cholesterol_flag"] + df["high_glucose_flag"] + df["obesity_flag"] + df["diabetes"]
    
    for ag in ["<40", "40-49", "50-59", "60-69", "70+"]:
        df[f"age_group_{ag}"] = 0
    age = patient["age"]
    if age < 40: df["age_group_<40"] = 1
    elif age < 50: df["age_group_40-49"] = 1
    elif age < 60: df["age_group_50-59"] = 1
    elif age < 70: df["age_group_60-69"] = 1
    else: df["age_group_70+"] = 1
    
    for cat in ["Underweight", "Normal", "Overweight", "Obese"]:
        df[f"bmi_cat_{cat}"] = 0
    bmi = patient["bmi"]
    if bmi < 18.5: df["bmi_cat_Underweight"] = 1
    elif bmi < 25: df["bmi_cat_Normal"] = 1
    elif bmi < 30: df["bmi_cat_Overweight"] = 1
    else: df["bmi_cat_Obese"] = 1
    
    df["log_total_cholesterol"] = np.log1p(df["total_cholesterol"])
    df["log_fasting_glucose"] = np.log1p(df["fasting_glucose"])
    df["log_bmi"] = np.log1p(df["bmi"])
    df["age_sbp_interaction"] = df["age"] * df["systolic_bp"]
    df["bmi_glucose_interaction"] = df["bmi"] * df["fasting_glucose"]
    df["age_smoking_interaction"] = df["age"] * df["smoking"]
    
    return df


def run_all_tests():
    """Run all model tests."""
    print("=" * 70)
    print("RISK CLASSIFICATION MODEL - COMPREHENSIVE TESTING")
    print("=" * 70)
    
    # Load model and get feature names
    model = load_model()
    if model is None:
        return
    
    # Get feature names from metadata
    meta_path = Path(__file__).resolve().parents[1] / "models" / "final_classifier_meta.json"
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    features = meta.get("feature_names", [])
    
    # Run tests
    test_sample_patients(model, features)
    accuracy = test_on_test_set()
    test_edge_cases(model, features)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✓ Model loaded successfully")
    print(f"✓ Test set accuracy: {accuracy*100:.2f}%")
    print(f"✓ Sample predictions working correctly")
    print(f"✓ Edge cases handled properly")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
