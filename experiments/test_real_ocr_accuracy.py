from __future__ import annotations

"""Test REAL OCR accuracy using Tesseract on synthetic PNG images.

This script uses pytesseract directly on PNG images in
`experiments/synthetic_images/` and compares extracted values against
`experiments/synthetic_ground_truth.csv` with a 5% numeric tolerance.
"""

import sys
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import pytesseract
from PIL import Image

# Ensure project root (and src/) is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_with_tesseract(image_path: Path) -> str:
    """Use Tesseract OCR to read a PNG image and return raw text."""
    img = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(img, config="--psm 6")
    return text


def extract_numeric(text: str, pattern: str, datatype=float) -> Optional[float]:
    """Extract numeric value from OCR text with simple error correction.

    Replaces common misreads like O→0, I→1 and strips commas.
    Returns None if parsing fails.
    """
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        value = (
            match.group(1)
            .replace(",", "")
            .replace("O", "0")
            .replace("I", "1")
        )
        return datatype(value)
    except Exception:
        return None


def extract_yesno(text: str, pattern: str) -> Optional[str]:
    """Extract 'Yes' or 'No' from text as a string value."""
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).strip().lower()
    return "Yes" if "yes" in token else "No"


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

    results: List[Dict[str, Any]] = []

    key_fields = [
        "age",
        "systolic_bp",
        "cholesterol",
        "glucose",
        "bmi",
        "smoking",
        "diabetes",
    ]

    for _, patient in ground_truth.iterrows():
        patient_id = patient["id"]
        image_path = images_dir / f"{patient_id}.png"

        print(f"\nTesting: {patient_id}")

        if not image_path.exists():
            print(f"  ✗ Image not found at {image_path}")
            results.append(
                {
                    "patient_id": patient_id,
                    "ocr_accuracy": 0.0,
                    "fields_correct": 0,
                    "fields_total": len(key_fields),
                    "errors": "IMAGE_NOT_FOUND",
                }
            )
            continue

        # REAL OCR extraction
        text = extract_with_tesseract(image_path)

        # Extract all fields with regex patterns
        extracted: Dict[str, Any] = {
            "age": extract_numeric(text, r"Age[:\s]+(\d+)", int),
            "sex": 1 if re.search(r"Sex[:\s]*Male", text, re.IGNORECASE) else 0,
            "systolic_bp": extract_numeric(text, r"Systolic\s+BP[:\s]+(\d+)", float),
            "diastolic_bp": extract_numeric(text, r"Diastolic\s+BP[:\s]+(\d+)", float),
            "bmi": extract_numeric(text, r"BMI[:\s]+([\d.]+)", float),
            "cholesterol": extract_numeric(text, r"Total\s+Cholesterol[:\s]+(\d+)", float),
            "hdl": extract_numeric(text, r"HDL\s+Cholesterol[:\s]+(\d+)", float),
            "ldl": extract_numeric(text, r"LDL\s+Cholesterol[:\s]+(\d+)", float),
            "triglycerides": extract_numeric(text, r"Triglycerides[:\s]+(\d+)", float),
            "glucose": extract_numeric(text, r"Fasting\s+Glucose[:\s]+(\d+)", float),
            "hemoglobin": extract_numeric(text, r"Hemoglobin[:\s]+([\d.]+)", float),
            "wbc": extract_numeric(text, r"WBC\s+Count[:\s]+(\d+)", float),
            "rbc": extract_numeric(text, r"RBC\s+Count[:\s]+([\d.]+)", float),
            "platelet": extract_numeric(text, r"Platelet\s+Count[:\s]+(\d+)", float),
            "smoking": extract_yesno(text, r"Smoking[:\s]+(Yes|No)"),
            "diabetes": extract_yesno(text, r"Diabetes[:\s]+(Yes|No)"),
        }

        correct = 0
        total = len(key_fields)
        errors: List[str] = []

        for field in key_fields:
            ground_val = patient[field]
            ocr_val = extracted.get(field)

            if ocr_val is None:
                errors.append(f"{field}: NOT EXTRACTED")
                print(f"  ✗ {field}: NOT FOUND")
                continue

            # Numeric comparison with 5% tolerance
            if isinstance(ground_val, (int, float)):
                try:
                    ground_f = float(ground_val)
                    ocr_f = float(ocr_val)
                except Exception:
                    errors.append(f"{field}: {ocr_val} vs {ground_val}")
                    print(f"  ✗ {field}: {ocr_val} (expected {ground_val})")
                    continue

                denom = abs(ground_f) + 0.001
                rel_err = abs(ground_f - ocr_f) / denom
                if rel_err <= 0.05:
                    correct += 1
                    print(f"  ✓ {field}: {ocr_f}")
                else:
                    errors.append(f"{field}: {ocr_f} vs {ground_f}")
                    print(f"  ✗ {field}: {ocr_f} (expected {ground_f})")
            else:
                # Categorical exact match (e.g., 'Yes'/'No')
                if str(ocr_val) == str(ground_val):
                    correct += 1
                    print(f"  ✓ {field}: {ocr_val}")
                else:
                    errors.append(f"{field}: {ocr_val} vs {ground_val}")
                    print(f"  ✗ {field}: {ocr_val} (expected {ground_val})")

        accuracy = (correct / total) * 100.0 if total > 0 else 0.0
        print(f"  OCR Accuracy: {accuracy:.1f}% ({correct}/{total})")

        results.append(
            {
                "patient_id": patient_id,
                "ocr_accuracy": accuracy,
                "fields_correct": correct,
                "fields_total": total,
                "errors": "; ".join(errors) if errors else None,
            }
        )

    df = pd.DataFrame(results)
    out_path = experiments_dir / "real_ocr_accuracy_results.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print("REAL OCR SUMMARY")
    print(f"{'=' * 60}")
    if not df.empty:
        print(f"Average OCR Accuracy: {df['ocr_accuracy'].mean():.1f}%")
        print(f"Perfect (100%): {(df['ocr_accuracy'] == 100).sum()}/{len(df)}")
        print(f"Good (≥80%): {(df['ocr_accuracy'] >= 80).sum()}/{len(df)}")
        print(f"Failed (<80%): {(df['ocr_accuracy'] < 80).sum()}/{len(df)}")
    else:
        print("No results recorded.")
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
