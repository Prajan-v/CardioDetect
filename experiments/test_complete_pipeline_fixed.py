from __future__ import annotations

"""Complete OCR → Feature → Model pipeline with full feature vector for mlp_v2.

This script:
- Runs REAL OCR via Tesseract on synthetic PNG images
- Builds a complete feature vector aligned with the mlp_v2 training schema
- Applies the saved StandardScaler from mlp_v2.pkl
- Gets risk predictions and classifies failures as OCR vs MODEL vs OCR_LUCKY
"""

import sys
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

# Ensure project root (and src/) is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlp_tuning import load_splits, encode_categorical_features, MLP_V2_PATH


def extract_with_tesseract(image_path: Path) -> str:
    """Real OCR using Tesseract on PNG images."""
    img = Image.open(image_path).convert("RGB")
    return pytesseract.image_to_string(img, config="--psm 6")


def extract_numeric(text: str, pattern: str, datatype=float) -> Optional[float]:
    """Extract numeric from OCR text with simple cleanup."""
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        value = match.group(1).replace(",", "").replace("O", "0")
        return datatype(value)
    except Exception:
        return None


def build_complete_features(
    extracted: Dict[str, Any],
    baseline_enc: pd.Series,
) -> pd.Series:
    """Build a complete feature vector for mlp_v2 from OCR-extracted values.

    - Starts from the median encoded feature vector (baseline_enc)
    - Overrides with OCR values where appropriate
    - Recomputes engineered features only if their columns exist
    """

    features = baseline_enc.copy()

    # Helper for setting plain numeric features when columns are present
    def _set_if_present(col: str, value_key: str) -> None:
        if value_key in extracted and extracted[value_key] is not None and col in features.index:
            features[col] = float(extracted[value_key])

    # Map core fields to training schema
    _set_if_present("age", "age")
    _set_if_present("systolic_bp", "systolic_bp")
    _set_if_present("diastolic_bp", "diastolic_bp")
    _set_if_present("bmi", "bmi")

    # Total cholesterol / fasting glucose naming in training data
    _set_if_present("total_cholesterol", "cholesterol")
    _set_if_present("fasting_glucose", "glucose")

    # Smoking & diabetes usually numeric in training data
    if "smoking" in features.index and extracted.get("smoking") is not None:
        # 1 for Yes, 0 for No; scale factor (e.g. 10 vs 0) is less critical with scaling
        features["smoking"] = 10.0 if extracted["smoking"] else 0.0
    if "diabetes" in features.index and extracted.get("diabetes") is not None:
        features["diabetes"] = 1.0 if extracted["diabetes"] else 0.0

    # Heart rate and bp_meds not in reports: keep baseline medians

    # Engineered features: pulse pressure & MAP
    sbp = extracted.get("systolic_bp")
    dbp = extracted.get("diastolic_bp")
    chol = extracted.get("cholesterol")
    glu = extracted.get("glucose")
    bmi = extracted.get("bmi")
    age = extracted.get("age")

    if sbp is not None and dbp is not None:
        if "pulse_pressure" in features.index:
            features["pulse_pressure"] = float(sbp) - float(dbp)
        if "mean_arterial_pressure" in features.index:
            features["mean_arterial_pressure"] = (float(sbp) + 2 * float(dbp)) / 3.0

    # Risk flags
    if "hypertension_flag" in features.index and sbp is not None:
        features["hypertension_flag"] = 1.0 if float(sbp) >= 140.0 else 0.0
    if "high_cholesterol_flag" in features.index and chol is not None:
        features["high_cholesterol_flag"] = 1.0 if float(chol) >= 240.0 else 0.0
    if "high_glucose_flag" in features.index and glu is not None:
        features["high_glucose_flag"] = 1.0 if float(glu) >= 126.0 else 0.0
    if "obesity_flag" in features.index and bmi is not None:
        features["obesity_flag"] = 1.0 if float(bmi) >= 30.0 else 0.0

    # Log transforms - adapt to actual column names
    if "log_total_cholesterol" in features.index and chol is not None:
        features["log_total_cholesterol"] = np.log(float(chol) + 1.0)
    if "log_fasting_glucose" in features.index and glu is not None:
        features["log_fasting_glucose"] = np.log(float(glu) + 1.0)
    if "log_bmi" in features.index and bmi is not None:
        features["log_bmi"] = np.log(float(bmi) + 1.0)

    # Interaction terms
    if "age_sbp_interaction" in features.index and age is not None and sbp is not None:
        features["age_sbp_interaction"] = float(age) * float(sbp)
    if "bmi_glucose_interaction" in features.index and bmi is not None and glu is not None:
        features["bmi_glucose_interaction"] = float(bmi) * float(glu)
    if "age_smoking_interaction" in features.index and age is not None:
        smoking_num = features.get("smoking", 0.0)
        features["age_smoking_interaction"] = float(age) * float(smoking_num)

    # One-hot encoded sex handling (if present)
    sex_val = extracted.get("sex")  # 1 = Male, 0 = Female
    if sex_val is not None:
        # Clear existing sex_* dummies
        sex_columns = [c for c in features.index if c.lower().startswith("sex_")]
        for col in sex_columns:
            features[col] = 0.0
        if sex_columns:
            target_token = "male" if sex_val == 1 else "female"
            for col in sex_columns:
                if target_token in col.lower():
                    features[col] = 1.0

    return features


def main() -> None:
    experiments_dir = PROJECT_ROOT / "experiments"
    images_dir = experiments_dir / "synthetic_images"
    csv_path = experiments_dir / "synthetic_ground_truth.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Ground truth CSV not found at {csv_path} - run generate_synthetic_patients.py first."
        )
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Synthetic images folder not found at {images_dir} - run generate_synthetic_images.py first."
        )

    ground_truth = pd.read_csv(csv_path)

    # Load training data and build baseline encoded feature vector
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    X_train_enc, X_val_enc, X_test_enc = encode_categorical_features(X_train, X_val, X_test)
    baseline_enc = X_train_enc.median()

    # Load frozen mlp_v2 artifact (model + scaler)
    artifact = joblib.load(MLP_V2_PATH)
    mlp_model = artifact["model"]
    mlp_scaler = artifact["scaler"]

    results: List[Dict[str, Any]] = []

    for _, patient in ground_truth.iterrows():
        patient_id = patient["id"]
        name = patient["name"]
        expected_level = patient["expected_risk_level"]

        print("\n" + "=" * 70)
        print(f"Testing: {patient_id} - {name}")
        print(f"Expected: {expected_level}")
        print("=" * 70)

        image_path = images_dir / f"{patient_id}.png"
        if not image_path.exists():
            print(f"  ✗ Image not found: {image_path}")
            continue

        # Step 1: OCR
        text = extract_with_tesseract(image_path)

        extracted: Dict[str, Any] = {
            "age": extract_numeric(text, r"Age[:\s]+(\d+)", int),
            "sex": 1 if re.search(r"Sex[:\s]*Male", text, re.IGNORECASE) else 0,
            "systolic_bp": extract_numeric(text, r"Systolic\s+BP[:\s]+(\d+)", float),
            "diastolic_bp": extract_numeric(text, r"Diastolic\s+BP[:\s]+(\d+)", float),
            "bmi": extract_numeric(text, r"BMI[:\s]+([\d.]+)", float),
            "cholesterol": extract_numeric(text, r"Total\s+Cholesterol[:\s]+(\d+)", float),
            "glucose": extract_numeric(text, r"Fasting\s+Glucose[:\s]+(\d+)", float),
            "smoking": bool(re.search(r"Smoking[:\s]+Yes", text, re.IGNORECASE)),
            "diabetes": bool(re.search(r"Diabetes[:\s]+Yes", text, re.IGNORECASE)),
        }

        # Step 2: OCR accuracy check on key numeric fields
        key_fields = ["age", "systolic_bp", "cholesterol", "glucose"]
        ocr_correct = 0
        for field in key_fields:
            ocr_val = extracted.get(field)
            ground_val = patient[field]
            if ocr_val is None:
                continue
            try:
                g = float(ground_val)
                o = float(ocr_val)
            except Exception:
                continue
            denom = abs(g) + 0.001
            rel_err = abs(g - o) / denom
            if rel_err <= 0.05:
                ocr_correct += 1
        ocr_accuracy = (ocr_correct / len(key_fields)) * 100.0 if key_fields else 0.0
        ocr_pass = ocr_accuracy >= 75.0

        print(f"OCR Accuracy: {ocr_accuracy:.1f}% ({ocr_correct}/{len(key_fields)})")

        # Step 3: Build complete feature vector
        features = build_complete_features(extracted, baseline_enc)

        # Debug: Show key feature values
        print("\nKey Features:")
        for f in ["age", "systolic_bp", "total_cholesterol", "fasting_glucose", "bmi"]:
            if f in features.index:
                print(f"  {f}: {features[f]:.2f}")

        # Step 4: Predict with mlp_v2
        feature_array = features.values.reshape(1, -1)
        feature_scaled = mlp_scaler.transform(feature_array)
        proba = float(mlp_model.predict_proba(feature_scaled)[0, 1])
        risk_percent = proba * 100.0

        if risk_percent < 10.0:
            predicted_level = "LOW"
        elif risk_percent < 20.0:
            predicted_level = "MEDIUM"
        else:
            predicted_level = "HIGH"

        prediction_correct = predicted_level == expected_level

        print(f"\nPrediction: {predicted_level} ({risk_percent:.1f}%)")
        print(f"Expected:   {expected_level}")
        print(f"Match:      {'✓' if prediction_correct else '✗'}")

        # Step 5: Classify failure type
        if ocr_pass and prediction_correct:
            status = "✅ PASS"
            failure_type: Optional[str] = None
        elif not ocr_pass and not prediction_correct:
            status = "❌ OCR FAILURE"
            failure_type = "OCR"
        elif ocr_pass and not prediction_correct:
            status = "❌ MODEL FAILURE"
            failure_type = "MODEL"
        else:
            status = "⚠️ LUCKY"
            failure_type = "OCR_LUCKY"

        print(f"Status: {status}")

        results.append(
            {
                "patient_id": patient_id,
                "name": name,
                "expected": expected_level,
                "predicted": predicted_level,
                "probability": risk_percent,
                "ocr_accuracy": ocr_accuracy,
                "ocr_pass": ocr_pass,
                "prediction_correct": prediction_correct,
                "status": status,
                "failure_type": failure_type,
            }
        )

    # Save results
    results_df = pd.DataFrame(results)
    out_path = experiments_dir / "complete_pipeline_results_fixed.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    total = len(results_df)
    if total == 0:
        print("No test cases processed.")
    else:
        passes = int((results_df["status"] == "✅ PASS").sum())
        ocr_fails = int((results_df["failure_type"] == "OCR").sum())
        model_fails = int((results_df["failure_type"] == "MODEL").sum())

        print(f"Total Tests: {total}")
        print(f"✅ PASS: {passes}/{total}")
        print(f"❌ OCR Failures: {ocr_fails}")
        print(f"❌ Model Failures: {model_fails}")
        print(f"\nAvg OCR Accuracy: {results_df['ocr_accuracy'].mean():.1f}%")
        print(
            f"Prediction Accuracy: "
            f"{int(results_df['prediction_correct'].sum())}/{total}"
        )

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
