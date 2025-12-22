from __future__ import annotations

import sys

import cv2
import pytesseract
from PIL import Image  # noqa: F401 (imported to satisfy requirements)
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR_DEFAULT = PROJECT_ROOT / "data" / "real_reports" / "kaggle_240"
OUTPUT_CSV = PROJECT_ROOT / "ocr_test_results.csv"


# Target filenames (30 reports)
TARGET_FILENAMES: List[str] = [
    "BLR-0425-PA-0036693_ARVIND-REDDY-REPALA-0036693_28-04-2025_1120-45_AM-E.pdf_page_24.jpg",
    "BLR-0425-PA-0037318_SASHANK-P-K-0037318-2-OF-2_28-04-2025_1007-19_AM-E.pdf_page_31.jpg",
    "BLR-0425-PA-0038965_BIPUL-CHAKRABORTY-0038965-2-OF-2_28-04-2025_1014-26_AM.pdf_page_7.jpg",
    "AHD-0425-PA-0008061_E-mahendrasinghdischargecard_250427_1114-E.pdf_page_27.jpg",
    "AHD-0425-PA-0007719_E-REPORTS_250427_2032-E.pdf_page_7.jpg",
    "BLR-0425-PA-0037318_SASHANK-P-K-0037318-2-OF-2_28-04-2025_1007-19_AM-E.pdf_page_33.jpg",
    "AHD-0425-PA-0008061_E-mahendrasinghdischargecard_250427_1114-E.pdf_page_13.jpg",
    "BLR-0425-PA-0038965_BIPUL-CHAKRABORTY-0038965-2-OF-2_28-04-2025_1014-26_AM.pdf_page_16.jpg",
    "BLR-0425-PA-0039192_E-PareshwarFinalBill_250427_1337-E.pdf_page_88.jpg",
    "BLR-0425-PA-0039930_E-BABY_OF_SIREESHA_BADVELU_2_250427_1048-E.pdf_page_8.jpg",
    "BLR-0425-PA-0039883_ALL-CLAIMS-DOCM-BHUVANESHWARI-VIDAL_0001_27-04-2025_1131-10_AM-E.pdf_page_41.jpg",
    "BLR-0425-PA-0040749_F-RAJU_GODUGULA_1_250422_1924-G.pdf_page_24.jpg",
    "BLR-0425-PA-0040652_LAB-MERG_27-04-2025_1239-18_PM-E.pdf_page_7.jpg",
    "BLR-0425-PA-0040301_Q-Report3_250422_1436-F.pdf_page_2.jpg",
    "BLR-0425-PA-0040749_G-RAJU_27-04-2025_1103-55_PM.pdf_page_35.jpg",
    "BLR-0425-PA-0039320_501848074-Final-bill-and-DS-26042025_27-04-2025_1054-20_AM.pdf_page_10.jpg",
    "BLR-0425-PA-0039192_E-PareshwarFinalBill_250427_1337-E.pdf_page_99.jpg",
    "BLR-0425-PA-0039930_E-BABY_OF_SIREESHA_BADVELU_2_250427_1048-E.pdf_page_11.jpg",
    "BLR-0425-PA-0040652_LAB-MERG_27-04-2025_1239-18_PM-E.pdf_page_2.jpg",
    "BLR-0425-PA-0040749_G-RAJU_27-04-2025_1103-55_PM.pdf_page_42.jpg",
    "BLR-0425-PA-0038965_BIPUL-CHAKRABORTY-0038965-2-OF-2_28-04-2025_1014-26_AM.pdf_page_8.jpg",
    "BLR-0425-PA-0040326_Lab-Report-Sunita_27-04-2025_0131-15_PM-G.pdf_page_11.jpg",
    "AHD-0425-PA-0007719_E-REPORTS_250427_2032-E.pdf_page_4.jpg",
    "BLR-0425-PA-0039883_ALL-CLAIMS-DOCM-BHUVANESHWARI-VIDAL_0001_27-04-2025_1131-10_AM-E.pdf_page_38.jpg",
    "BLR-0425-PA-0037318_SASHANK-P-K-0037318-2-OF-2_28-04-2025_1007-19_AM-E.pdf_page_20.jpg",
    "AHD-0425-PA-0007561_JITENDRA-TRIVEDI-DS_28-04-2025_1019-21_AM.pdf_page_9.jpg",
    "BLR-0425-PA-0037318_SASHANK-P-K-0037318-2-OF-2_28-04-2025_1007-19_AM-E.pdf_page_29.jpg",
    "BLR-0425-PA-0039192_05c45741fa5d4b5180df06f200423a00__2_files_merged__26-04-2025_0430-01_PM-E.pdf_page_104.jpg",
    "BLR-0425-PA-0039192_a044492e47444b7ca0f3e3bbe049ce2c_compressed_27-04-2025_0228-52_PM-E.pdf_page_76.jpg",
    "BLR-0425-PA-0039192_E-ParmeshwarRunningBill_250426_1612-E.pdf_page_93.jpg",
]


# Ground-truth metadata for quality / blur (known samples)
GROUND_TRUTH_META: Dict[str, Dict[str, str]] = {
    # Sample 1
    "BLR-0425-PA-0036693_ARVIND-REDDY-REPALA-0036693_28-04-2025_1120-45_AM-E.pdf_page_24.jpg": {
        "quality": "GOOD",
        "blur_sections": "Patient name",
    },
    # Sample 2
    "BLR-0425-PA-0037318_SASHANK-P-K-0037318-2-OF-2_28-04-2025_1007-19_AM-E.pdf_page_31.jpg": {
        "quality": "MODERATE",
        "blur_sections": "Patient name, dates",
    },
    # Sample 3
    "BLR-0425-PA-0038965_BIPUL-CHAKRABORTY-0038965-2-OF-2_28-04-2025_1014-26_AM.pdf_page_7.jpg": {
        "quality": "EXCELLENT",
        "blur_sections": "None",
    },
}


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def preprocess_basic(gray: np.ndarray) -> np.ndarray:
    """Basic thresholding."""
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return th


def preprocess_otsu(gray: np.ndarray) -> np.ndarray:
    """OTSU thresholding."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_enhanced(gray: np.ndarray) -> np.ndarray:
    """Denoise + CLAHE + OTSU thresholding (expected best for most images)."""
    den = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(den)
    _, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_aggressive(gray: np.ndarray) -> np.ndarray:
    """Median blur + CLAHE + dilation (for poor-quality images)."""
    med = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(med)
    _, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    dil = cv2.dilate(th, kernel, iterations=1)
    return dil


PREPROCESSORS = {
    "basic": preprocess_basic,
    "otsu": preprocess_otsu,
    "enhanced": preprocess_enhanced,
    "aggressive": preprocess_aggressive,
}


def run_ocr(image: np.ndarray, psm: int) -> Tuple[str, float, int]:
    """Run Tesseract OCR and return (text, avg_confidence, word_count)."""
    config = f"--psm {psm} --oem 3"

    text = pytesseract.image_to_string(image, config=config)

    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    confs: List[float] = []
    word_count = 0

    for conf, token in zip(data.get("conf", []), data.get("text", [])):
        if token and str(token).strip():
            word_count += 1
        try:
            c = float(conf)
        except (ValueError, TypeError):
            continue
        if c >= 0:
            confs.append(c)

    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return text, avg_conf, word_count


def main() -> None:
    print("=" * 100)
    print("OCR PIPELINE TEST - 30 Medical Reports")
    print("=" * 100)

    # Optional CLI argument: custom image directory (e.g. Medical_report/Sample_reports)
    if len(sys.argv) > 1:
        image_dir = Path(sys.argv[1])
    else:
        image_dir = IMAGE_DIR_DEFAULT

    print(f"Image directory: {image_dir}")

    image_paths: List[Path] = []
    for name in TARGET_FILENAMES:
        path = image_dir / name
        if not path.exists():
            print(f"[WARN] Missing image: {path}")
            continue
        image_paths.append(path)

    # If none of the target filenames are present, fall back to all JPG/JPEG/PNG files
    if not image_paths:
        print(f"No target filenames found in {image_dir}, falling back to all *.jpg/jpeg/png")
        fallback_paths = sorted(
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
            + list(image_dir.glob("*.png"))
        )
        if not fallback_paths:
            print(f"No images found in {image_dir}")
            return
        image_paths = fallback_paths

    rows: List[Dict[str, object]] = []

    total = len(image_paths)
    method_names = list(PREPROCESSORS.keys())
    psms = [3, 6, 11]

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{total}] Processing: {img_path.name}")

        img = load_image(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        best_conf = -1.0
        best_text = ""
        best_method = ""
        best_psm: int | None = None
        best_word_count = 0

        for method_name in method_names:
            pre_fn = PREPROCESSORS[method_name]
            pre_img = pre_fn(gray)

            for psm in psms:
                try:
                    text, conf, wc = run_ocr(pre_img, psm)
                except Exception as e:  # noqa: BLE001
                    print(f"  {method_name:10s} PSM{psm}: ERROR ({e})")
                    continue

                print(f"  {method_name:10s} PSM{psm}: conf={conf:.1f}% words={wc}")

                if conf > best_conf:
                    best_conf = conf
                    best_text = text
                    best_method = method_name
                    best_psm = psm
                    best_word_count = wc

        if best_psm is None:
            print("  No successful OCR result; skipping")
            continue

        best_method_label = f"{best_method}+PSM{best_psm}"
        print(f"  BEST: {best_method} PSM{best_psm} @ {best_conf:.1f}%")

        meta = GROUND_TRUTH_META.get(img_path.name, {})
        quality = meta.get("quality", "")
        blur = meta.get("blur_sections", "")

        match_status = "✓ GOOD" if best_conf > 70.0 else ""

        rows.append(
            {
                "filename": img_path.name,
                "best_method": best_method_label,
                "best_psm": best_psm,
                "confidence_score": best_conf,
                "word_count": best_word_count,
                "ground_truth_quality": quality,
                "blur_sections": blur,
                "text_sample": best_text[:200].replace("\n", " "),
                "match_status": match_status,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 100)
    print("OCR TEST SUMMARY")
    print("=" * 100)

    print(f"\nTotal Reports Tested: {len(df)}")

    if not df.empty:
        avg_conf = float(df["confidence_score"].mean())
        good_mask = df["confidence_score"] > 70.0
        num_good = int(good_mask.sum())
        pct_good = (num_good / len(df)) * 100.0 if len(df) > 0 else 0.0

        print(f"Average Confidence: {avg_conf:.1f}%")
        print(f"Reports with >70% Confidence: {num_good}/{len(df)} ({pct_good:.1f}%)")

        print("\nTop 5 images by confidence:")
        top5 = df.sort_values("confidence_score", ascending=False).head(5)
        for _, row in top5.iterrows():
            print(
                f"  {row['filename']}: {row['confidence_score']:.1f}% "
                f"({row['best_method']})"
            )

        print("\nBottom 5 images by confidence:")
        bottom5 = df.sort_values("confidence_score", ascending=True).head(5)
        for _, row in bottom5.iterrows():
            print(
                f"  {row['filename']}: {row['confidence_score']:.1f}% "
                f"({row['best_method']})"
            )

        print("\nTop Methods Used:")
        method_counts = df["best_method"].value_counts()
        for method, count in method_counts.items():
            print(f"  {method}: {count}")

    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
