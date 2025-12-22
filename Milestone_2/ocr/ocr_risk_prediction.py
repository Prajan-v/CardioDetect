"""
OCR → MLP Risk Prediction Pipeline
Uses optimized medical OCR with Tesseract/digital extraction
"""

from src.medical_ocr_optimized import MedicalOCROptimized
from src.mlp_tuning import load_splits, encode_categorical_features, MLP_V2_PATH
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def run_ocr_risk_prediction(pdf_path: str) -> dict:
    """
    Complete pipeline: OCR → Feature Construction → Risk Prediction
    
    Args:
        pdf_path: Path to medical PDF document
        
    Returns:
        dict with OCR results, extracted fields, and risk prediction
    """
    print(f"\n{'='*80}")
    print("CardioDetect: OCR → Risk Prediction Pipeline")
    print(f"{'='*80}\n")
    
    # Step 1: OCR with optimized pipeline
    print("Step 1: Medical Document OCR")
    ocr = MedicalOCROptimized(verbose=False)
    ocr_result = ocr.extract_from_pdf(pdf_path)
    
    print(f"  Method: {ocr_result['method']}")
    print(f"  Quality: {ocr_result['quality']}")
    print(f"  Confidence: {ocr_result['confidence']:.1%}")
    print(f"  Fields: {len(ocr_result['fields'])}/6")
    
    # Extract demographics
    parsed_age = ocr_result['fields'].get('age')
    parsed_sex = ocr_result['fields'].get('sex')
    parsed_sex_code = ocr_result['fields'].get('sex_code')
    
    print(f"\n  Extracted:")
    print(f"    • Age: {parsed_age} years")
    print(f"    • Sex: {parsed_sex}")
    
    # Step 2: Build feature vector
    print("\nStep 2: Building feature vector...")

    # Load train/val/test splits exactly as in the MLP tuning pipeline
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    X_train_enc, X_val_enc, X_test_enc = encode_categorical_features(X_train, X_val, X_test)

    # Calculate median baseline in the encoded feature space
    baseline = X_train_enc.median(axis=0)
    
    # Override age / sex if extracted
    if parsed_age and 'age' in baseline.index:
        baseline['age'] = parsed_age
        print(f"  ✓ Using extracted age: {parsed_age}")
    else:
        print(f"  ⚠️  Using median age: {baseline.get('age', 'N/A'):.0f}")

    if parsed_sex_code is not None and 'sex' in baseline.index:
        baseline['sex'] = float(parsed_sex_code)
        print(f"  ✓ Using extracted sex_code: {parsed_sex_code}")
    
    feature_row = baseline.values.reshape(1, -1)
    
    # Step 3: Risk prediction
    print("\nStep 3: MLP v2 Best risk prediction...")
    model_path = MLP_V2_PATH
    artifact = joblib.load(model_path)

    # Handle both dict artifact (model + scaler) and bare estimator
    model = artifact
    scaler = None
    if isinstance(artifact, dict):
        scaler = artifact.get("scaler")
        # Try common keys for the estimator
        for key in ["model", "estimator", "clf", "classifier"]:
            if key in artifact and artifact[key] is not None:
                model = artifact[key]
                break

    if scaler is not None:
        feature_row_scaled = scaler.transform(feature_row)
    else:
        feature_row_scaled = feature_row

    risk_proba = model.predict_proba(feature_row_scaled)[0, 1]
    risk_label = int(risk_proba >= 0.5)
    
    # Determine risk level
    if risk_proba < 0.10:
        risk_level = "LOW"
    elif risk_proba < 0.25:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"  OCR Quality:      {ocr_result['quality'].upper()}")
    print(f"  OCR Method:       {ocr_result['method']}")
    print(f"  Extracted Age:    {parsed_age} years")
    print(f"  Extracted Sex:    {parsed_sex}")
    print(f"  Risk Probability: {risk_proba:.4f} ({risk_proba*100:.2f}%)")
    print(f"  Risk Level:       {risk_level}")
    print(f"  Prediction:       {'CHD POSITIVE' if risk_label == 1 else 'CHD NEGATIVE'}")
    print(f"{'='*80}\n")
    
    return {
        'ocr_result': ocr_result,
        'parsed_age': parsed_age,
        'parsed_sex': parsed_sex,
        'risk_probability': risk_proba,
        'predicted_label': risk_label,
        'risk_level': risk_level,
        'quality': ocr_result['quality'],
        'fields_extracted': len(ocr_result['fields'])
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_risk_prediction.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    run_ocr_risk_prediction(pdf_path)
