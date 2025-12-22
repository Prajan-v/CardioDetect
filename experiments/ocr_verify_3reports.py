from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pytesseract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR_DEFAULT = PROJECT_ROOT / "data" / "real_reports" / "kaggle_240"

# Filenames to process (5 specific lbmaske PNG files)
IMAGE_FILES: List[str] = [
    "AHD-0425-PA-0007561_JITENDRA TRIVEDI DS_28-04-2025_1019-21_AM.pdf_page_9.png",
    "AHD-0425-PA-0007719_E-REPORTS_250427_2032@E.pdf_page_4.png",
    "AHD-0425-PA-0007719_E-REPORTS_250427_2032@E.pdf_page_7.png",
    "AHD-0425-PA-0008061_E-mahendrasinghdischargecard_250427_1114@E.pdf_page_13.png",
    "AHD-0425-PA-0008061_E-mahendrasinghdischargecard_250427_1114@E.pdf_page_27.png",
]


def preprocess_enhanced(gray: np.ndarray) -> np.ndarray:
    """Denoise + CLAHE + Otsu thresholding."""
    den = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(den)
    _, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def extract_first_number_after(label: str, text: str) -> str | None:
    """Find the first number that appears on the same line as the label."""
    pattern = re.compile(rf"{re.escape(label)}[^\n\r]*", re.IGNORECASE)
    for match in pattern.finditer(text):
        segment = match.group(0)
        num_match = re.search(r"(\d+(?:\.\d+)?)", segment)
        if num_match:
            return num_match.group(1)
    return None


def extract_values(image_idx: int, text: str) -> Dict[str, str | None]:
    """Extract a set of common lab values using simple label-based regex."""
    labels = [
        "Hemoglobin",
        "RBC",
        "WBC",
        "Glucose",
        "RBS",
        "Prothrombin",
        "INR",
    ]

    results: Dict[str, str | None] = {}
    for label in labels:
        results[label] = extract_first_number_after(label, text)
    return results


def run_ocr_on_image(img_path: Path, idx: int) -> None:
    if not img_path.exists():
        print(f"--- IMAGE: {img_path.name} ---")
        print(f"[ERROR] File not found: {img_path}")
        return

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"--- IMAGE: {img_path.name} ---")
        print(f"[ERROR] Failed to read image: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pre = preprocess_enhanced(gray)

    config = "--psm 6 --oem 3"
    text = pytesseract.image_to_string(pre, config=config)

    print(f"--- IMAGE: {img_path.name} ---")
    print("[RAW OCR OUTPUT START]")
    print(text)
    print("[RAW OCR OUTPUT END]")

    values = extract_values(idx, text)
    if values:
        print("\nEXTRACTED VALUES:")
        for field, val in values.items():
            print(f"Field: {field} | Value: {val}")
    print("\n" + "=" * 80 + "\n")


def main() -> None:
    # Optional CLI argument: custom image directory
    if len(sys.argv) > 1:
        image_dir = Path(sys.argv[1])
    else:
        image_dir = IMAGE_DIR_DEFAULT

    print("Using image directory:", image_dir)
    for idx, fname in enumerate(IMAGE_FILES):
        img_path = image_dir / fname
        run_ocr_on_image(img_path, idx)


if __name__ == "__main__":
    main()
