from __future__ import annotations

import sys
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root (and src/) is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlp_tuning import load_splits, encode_categorical_features, MLP_V2_PATH
from src.medical_ocr_optimized import MedicalOCROptimized


def extract_numeric(text: str, pattern: str, datatype=float) -> Optional[float]:
    """Extract a numeric value from OCR text using a regex pattern.

    Returns None if parsing fails.
    """

    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        raw = match.group(1).replace(",", "")
        return datatype(raw)
    except Exception:
        return None


def run_ocr_on_image(image_path: Path, ocr: MedicalOCROptimized) -> Dict[str, Any]:
    """Run the optimized OCR pipeline on a PNG image.

    Reuses the same preprocessing and Tesseract configuration as the
    PDF-based pipeline, but starts directly from an image file.
    """

    img = Image.open(image_path).convert("RGB")
    binary = ocr._preprocess(img, dpi=300)  # type: ignore[attr-defined]
    ocr_res = ocr._run_tesseract(binary)  # type: ignore[attr-defined]
    return ocr_res


def build_feature_row(
    baseline_enc: pd.Series,
    extracted: Dict[str, Optional[float]],
    patient_row: pd.Series,
) -> pd.Series:
    """Build a single encoded feature row from OCR-extracted values.

    Uses the median encoded feature vector as a baseline and overrides
    core numeric risk features when available.
    """

    row = baseline_enc.copy()

    def _set_if_present(name: str, value: Optional[float]) -> None:
        if value is not None and name in row.index:
            row[name] = float(value)

    # Core numeric features (match training column names)
    _set_if_present("age", extracted.get("age"))
    _set_if_present("systolic_bp", extracted.get("systolic_bp"))
    _set_if_present("diastolic_bp", extracted.get("diastolic_bp"))
    _set_if_present("total_cholesterol", extracted.get("cholesterol"))
    _set_if_present("fasting_glucose", extracted.get("glucose"))
    _set_if_present("bmi", extracted.get("bmi"))

    # Smoking and diabetes from structured ground truth (not OCR)
    smoking_str = str(patient_row.get("smoking", "No")).strip().lower()
    diabetes_str = str(patient_row.get("diabetes", "No")).strip().lower()

    if "smoking" in row.index:
        row["smoking"] = 10.0 if smoking_str == "yes" else 0.0
    if "diabetes" in row.index:
        row["diabetes"] = 1.0 if diabetes_str == "yes" else 0.0

    return row


def main() -> None:
    experiments_dir = PROJECT_ROOT / "experiments"
    images_dir = experiments_dir / "synthetic_images"
    csv_path = experiments_dir / "synthetic_ground_truth.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found at {csv_path} - run generate_synthetic_patients.py first.")
    if not images_dir.exists():
        raise FileNotFoundError(f"Synthetic images folder not found at {images_dir} - run generate_synthetic_images.py first.")

    ground_truth = pd.read_csv(csv_path)

    # Load data and model artifacts for feature shape and scaling
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    X_train_enc, X_val_enc, X_test_enc = encode_categorical_features(X_train, X_val, X_test)

    baseline_enc = X_train_enc.median()

    artifact = joblib.load(MLP_V2_PATH)
    mlp_model = artifact["model"]
    mlp_scaler = artifact["scaler"]

    ocr = MedicalOCROptimized(verbose=False)

    results: List[Dict[str, Any]] = []

    for _, patient in ground_truth.iterrows():
        patient_id = patient["id"]
        name = patient["name"]
        expected_level = patient["expected_risk_level"]

        print("\n" + "=" * 70)
        print(f"Testing: {patient_id} - {name}")
        print(f"Expected Risk: {expected_level}")
        print("=" * 70)

        image_path = images_dir / f"{patient_id}.png"
        if not image_path.exists():
            print(f"  ✗ Image not found: {image_path}")
            continue

        # === STEP 1: OCR EXTRACTION ===
        try:
            ocr_res = run_ocr_on_image(image_path, ocr)
            text = ocr_res.get("text", "")
        except Exception as e:  # pragma: no cover - defensive
            print(f"  ✗ OCR error: {e}")
            text = ""

        extracted: Dict[str, Optional[float]] = {
            "age": extract_numeric(text, r"Age[:\s]+(\d+)", int),
            "systolic_bp": extract_numeric(text, r"Systolic BP[:\s]+(\d+)", float),
            "diastolic_bp": extract_numeric(text, r"Diastolic BP[:\s]+(\d+)", float),
            "cholesterol": extract_numeric(text, r"Total Cholesterol[:\s]+(\d+)", float),
            "glucose": extract_numeric(text, r"Fasting Glucose[:\s]+(\d+)", float),
            "hemoglobin": extract_numeric(text, r"Hemoglobin[:\s]+([\d.]+)", float),
        }

        # Also pass through BMI from ground truth so model sees consistent value
        bmi_val = float(patient["bmi"]) if not pd.isna(patient["bmi"]) else None
        extracted["bmi"] = bmi_val

        # === STEP 2: OCR ACCURACY CHECK ===
        ocr_errors: List[str] = []
        ocr_correct = 0
        ocr_total = 0

        for field in ["age", "systolic_bp", "cholesterol", "glucose"]:
            ocr_total += 1
            ground_val = float(patient[field]) if field in patient and not pd.isna(patient[field]) else None
            ocr_val = extracted.get(field)

            if ground_val is None:
                continue

            if ocr_val is None:
                ocr_errors.append(f"{field}: NOT EXTRACTED")
                print(f"  ✗ OCR: {field} - NOT EXTRACTED (expected {ground_val})")
            else:
                # Allow 5% relative error
                if ground_val == 0:
                    rel_err = abs(ocr_val - ground_val)
                else:
                    rel_err = abs(ocr_val - ground_val) / abs(ground_val)
                if rel_err > 0.05:
                    ocr_errors.append(f"{field}: {ocr_val} vs {ground_val}")
                    print(f"  ✗ OCR: {field} = {ocr_val} (expected {ground_val})")
                else:
                    ocr_correct += 1
                    print(f"  ✓ OCR: {field} = {ocr_val}")

        ocr_accuracy = (ocr_correct / ocr_total) if ocr_total > 0 else 0.0
        ocr_pass = ocr_accuracy >= 0.75  # 75% threshold

        # === STEP 3: BUILD FEATURE VECTOR ===
        feature_row = build_feature_row(baseline_enc, extracted, patient)
        feature_array = feature_row.values.reshape(1, -1)
        feature_scaled = mlp_scaler.transform(feature_array)

        # === STEP 4: MODEL PREDICTION ===
        proba = mlp_model.predict_proba(feature_scaled)[0, 1]
        risk_percent = float(proba * 100.0)

        if risk_percent < 10.0:
            predicted_level = "LOW"
        elif risk_percent < 20.0:
            predicted_level = "MEDIUM"
        else:
            predicted_level = "HIGH"

        prediction_correct = (predicted_level == expected_level)

        print(f"\n  Model Prediction: {predicted_level} ({risk_percent:.1f}%)")
        print(f"  Expected: {expected_level}")
        print(f"  Match: {'✓ YES' if prediction_correct else '✗ NO'}")

        # === STEP 5: CLASSIFY FAILURE TYPE ===
        if ocr_pass and prediction_correct:
            status = "✅ PASS - System Works"
            failure_type = None
        elif not ocr_pass and not prediction_correct:
            status = "❌ OCR FAILURE (likely caused wrong prediction)"
            failure_type = "OCR"
        elif ocr_pass and not prediction_correct:
            status = "❌ MODEL FAILURE (OCR correct but prediction wrong)"
            failure_type = "MODEL"
        else:  # not ocr_pass but prediction_correct
            status = "⚠️ LUCKY PASS (OCR wrong but prediction still correct)"
            failure_type = "OCR_LUCKY"

        print(f"\n  Status: {status}")

        results.append(
            {
                "patient_id": patient_id,
                "name": name,
                "expected_risk": expected_level,
                "predicted_risk": predicted_level,
                "risk_probability": risk_percent,
                "ocr_accuracy": ocr_accuracy * 100.0,
                "ocr_pass": ocr_pass,
                "prediction_correct": prediction_correct,
                "status": status,
                "failure_type": failure_type,
                "ocr_errors": "; ".join(ocr_errors) if ocr_errors else None,
            }
        )

    # === FINAL SUMMARY ===
    results_df = pd.DataFrame(results)
    out_path = experiments_dir / "synthetic_pipeline_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total = len(results_df)
    passes = int(results_df["status"].str.contains("PASS").sum())
    ocr_fails = int((results_df["failure_type"] == "OCR").sum())
    model_fails = int((results_df["failure_type"] == "MODEL").sum())

    print(f"\nTotal Tests: {total}")
    print(f"✅ PASS: {passes}/{total} ({(passes / total * 100.0) if total else 0.0:.1f}%)")
    print(f"❌ OCR Failures: {ocr_fails}/{total}")
    print(f"❌ Model Failures: {model_fails}/{total}")

    if not results_df.empty:
        print(f"\nAverage OCR Accuracy: {results_df['ocr_accuracy'].mean():.1f}%")
        print(
            f"Model Prediction Accuracy: "
            f"{int(results_df['prediction_correct'].sum())}/{total}"
        )

    if ocr_fails > 0:
        print("\n⚠️ OCR NEEDS IMPROVEMENT")
        print("Failed cases:")
        print(results_df[results_df["failure_type"] == "OCR"][
            ["patient_id", "name", "ocr_errors"]
        ])

    if model_fails > 0:
        print("\n⚠️ MODEL NEEDS IMPROVEMENT")
        print("Failed cases:")
        print(results_df[results_df["failure_type"] == "MODEL"][
            ["patient_id", "name", "expected_risk", "predicted_risk"]
        ])

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
