from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.production_ocr import ProductionOCR  # type: ignore
from src.production_model import ProductionModel  # type: ignore


MODELS_DIR = PROJECT_ROOT / "models"


def ocr_to_patient_data(fields: Dict[str, Any], ocr_conf: float) -> Dict[str, Any]:
    """Map OCR extracted_fields dict -> patient_data dict."""
    patient_data: Dict[str, Any] = {}

    field_mapping = {
        "age": "age",
        "sex": "sex",
        "systolic_bp": "systolic_bp",
        "diastolic_bp": "diastolic_bp",
        "bmi": "bmi",
        "total_cholesterol": "total_cholesterol",
        "hdl": "hdl",
        "ldl": "ldl",
        "triglycerides": "triglycerides",
        "fasting_glucose": "fasting_glucose",
        "hemoglobin": "hemoglobin",
        "wbc": "wbc",
        "rbc": "rbc",
        "platelet": "platelet",
        "smoking": "smoking",
        "diabetes": "diabetes",
        "heart_rate": "heart_rate",
    }

    for ocr_field, patient_field in field_mapping.items():
        if ocr_field in fields:
            value = fields[ocr_field]["value"] if isinstance(fields[ocr_field], dict) else fields[ocr_field]
            patient_data[patient_field] = value

    patient_data["ocr_confidence"] = ocr_conf
    return patient_data


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python experiments/test_report_v3.py <document_path>")
        sys.exit(1)

    doc_path = Path(sys.argv[1])
    if not doc_path.exists():
        print(f"File not found: {doc_path}")
        sys.exit(1)

    print(f"\nTesting document: {doc_path}")

    # 1) Run OCR
    ocr = ProductionOCR(verbose=False)
    ocr_result = ocr.extract(doc_path)

    if not ocr_result.success:
        print("OCR failed; cannot proceed.")
        sys.exit(1)

    # extracted_fields from ProductionOCR is already a mapping name -> OCRField
    # For this script we mimic ProductionPipeline's JSON structure
    extracted_fields = {
        name: {"value": f.value, "confidence": f.confidence}
        for name, f in ocr_result.fields.items()
    }

    patient_data = ocr_to_patient_data(extracted_fields, ocr_result.overall_confidence)

    # 2) Use existing ProductionModel (v2) to build feature vector and predict
    prod_model_v2 = ProductionModel(enable_shap=False, verbose=False)

    features_v2 = prod_model_v2.build_feature_vector(patient_data)
    pred_v2 = prod_model_v2.predict(patient_data)

    print("\n=== Binary v2 model ===")
    print(f"Risk level: {pred_v2.risk_level}")
    print(f"Risk probability: {pred_v2.risk_probability:.2f}%")

    # 3) Load tuned 3-class v3 model and evaluate on same features
    tuned_path = MODELS_DIR / "mlp_v3_tuned.pkl"
    if not tuned_path.exists():
        print(f"\nTuned 3-class model not found at {tuned_path}, aborting v3 test.")
        sys.exit(0)

    artifact = joblib.load(tuned_path)
    model_v3 = artifact["model"]
    scaler_v3 = artifact["scaler"]
    feature_names_v3 = artifact["feature_names"]
    class_names_v3 = artifact.get("class_names", ["LOW", "MEDIUM", "HIGH"])

    # Align features: v2 build_feature_vector returns encoded features; reorder for v3
    missing = set(feature_names_v3) - set(features_v2.index)
    if missing:
        print("\nERROR: Feature vector for v2 is missing features required by v3:")
        print(sorted(missing))
        sys.exit(1)

    x_v3 = features_v2.reindex(feature_names_v3).astype(float).values.reshape(1, -1)
    x_v3_scaled = scaler_v3.transform(x_v3)
    proba_v3 = model_v3.predict_proba(x_v3_scaled)[0]
    pred_idx = int(np.argmax(proba_v3))
    pred_class = class_names_v3[pred_idx]

    print("\n=== Tuned 3-class v3 model ===")
    print(f"Predicted class: {pred_class} (class {pred_idx})")
    print("Class probabilities:")
    for i, (name, p) in enumerate(zip(class_names_v3, proba_v3)):
        print(f"  {i} - {name}: {p*100:.2f}%")


if __name__ == "__main__":
    main()
