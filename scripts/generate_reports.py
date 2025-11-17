# Author: Prajan V (Infosys Springboard 6.0)
# Date: 2025-11-17

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.ocr.pipeline import extract_structured

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reports")


def _paths() -> Tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    output = root / "output"
    reports = root / "reports"
    figures = reports / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return root, output, reports, figures


def _load_or_build_ultimate(output: Path) -> pd.DataFrame:
    f = output / "ultimate_heart_disease_dataset.csv"
    if not f.exists():
        from scripts.build_ultimate_timeline_dataset import main as build_main
        logger.info("Ultimate dataset not found. Building now.")
        build_main()
    return pd.read_csv(f)


def _summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    total = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_rate = float(df[numeric_cols].isna().mean().mean()) if numeric_cols else 0.0
    by_source = (
        df.groupby(["data_source"]).size().rename("count").reset_index().to_dict(orient="records")
        if "data_source" in df.columns
        else []
    )
    by_year = (
        df.groupby(["collection_year"]).size().rename("count").reset_index().to_dict(orient="records")
        if "collection_year" in df.columns
        else []
    )
    corr = None
    cols = [c for c in ["age", "trestbps", "chol", "bmi", "thalach", "oldpeak", "target"] if c in df.columns]
    if len(cols) >= 2:
        corr = df[cols].apply(pd.to_numeric, errors="coerce").corr().round(3).to_dict()
    return {
        "total_rows": total,
        "missing_rate_numeric": missing_rate,
        "by_source": by_source,
        "by_year": by_year,
        "corr": corr,
    }


def _ensure_tesseract_env():
    if os.environ.get("TESSERACT_CMD"):
        return
    default = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    if Path(default).exists():
        os.environ["TESSERACT_CMD"] = default


def _ensure_sample_docs(reports_dir: Path):
    png = reports_dir / "figures" / "sample_report.png"
    pdf = reports_dir / "figures" / "sample_report.pdf"
    if png.exists() and pdf.exists():
        return png, pdf
    text = (
        "Patient Report\n"
        "BP 120/80 mmHg\n"
        "Cholesterol 200 mg/dL\n"
        "HDL 60 mg/dL\n"
        "LDL 120 mg/dL\n"
        "Triglycerides 150 mg/dL\n"
        "Glucose 100 mg/dL"
    )
    img = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    draw.multiline_text((50, 50), text, fill=(0, 0, 0), font=font, spacing=10)
    img.save(png)
    img.save(pdf, "PDF", resolution=300.0)
    return png, pdf


def _ocr_ground_truth() -> Dict[str, float]:
    return {
        "bp_sys": 120.0,
        "bp_dia": 80.0,
        "chol": 200.0,
        "hdl": 60.0,
        "ldl": 120.0,
        "triglycerides": 150.0,
        "glucose": 100.0,
    }


def _compute_ocr_metrics(pred: Dict[str, float], gt: Dict[str, float]) -> Tuple[pd.DataFrame, float]:
    rows = []
    correct = 0
    total = 0
    for k, v in gt.items():
        pv = pred.get(k)
        ok = False
        if pv is not None:
            try:
                ok = abs(float(pv) - float(v)) <= 1e-3
            except Exception:
                ok = False
        rows.append({"field": k, "pred": pv, "gt": v, "correct": int(ok)})
        total += 1
        correct += int(ok)
    df = pd.DataFrame(rows)
    acc = correct / total if total else 0.0
    return df, acc


def write_data_analysis_report(reports_dir: Path, summary: Dict[str, object]):
    path = reports_dir / "data_analysis_report.md"
    by_source = summary.get("by_source", [])
    by_year = summary.get("by_year", [])
    corr = summary.get("corr") or {}

    lines = []
    lines.append("# Data Analysis Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(
        f"This integrated dataset contains {summary['total_rows']} patients spanning multiple sources and years. "
        f"Numeric missingness averages {summary['missing_rate_numeric']:.2%}. "
        "Engineered risk factors and standardized preprocessing prepare the data for robust modeling."
    )
    lines.append("")
    lines.append("## Composition by Source")
    for r in by_source:
        lines.append(f"- {r['data_source']}: {r['count']}")
    lines.append("")
    lines.append("## Composition by Year")
    for r in by_year:
        lines.append(f"- {r['collection_year']}: {r['count']}")
    lines.append("")
    lines.append("## Key Correlations (numeric vs target)")
    if corr:
        t_corr = corr.get("target", {})
        for k, v in t_corr.items():
            if k == "target":
                continue
            lines.append(f"- {k} vs target: {v:.2f}")
    else:
        lines.append("- Correlation matrix unavailable.")
    lines.append("")
    lines.append("## Data Quality")
    lines.append("- Median imputation for numeric features; most-frequent for categoricals")
    lines.append("- Standardization and one-hot encoding within a reproducible pipeline")
    lines.append("- Outlier handling considered during modeling stage")
    lines.append("")
    lines.append("## Recommendations for Milestone 2")
    lines.append("- Train baselines (Logistic, RF, XGBoost) with calibration")
    lines.append("- Evaluate with ROC-AUC and decision curves; define risk thresholds")
    lines.append("- Perform feature selection (RFE/L1) and sensitivity analysis")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def write_ocr_evaluation_report(reports_dir: Path, pred_png: Dict[str, float], pred_pdf: Dict[str, float], metrics_png: pd.DataFrame, acc_png: float, metrics_pdf: pd.DataFrame, acc_pdf: float):
    path = reports_dir / "ocr_evaluation_report.md"
    gt = _ocr_ground_truth()

    lines = []
    lines.append("# OCR Evaluation Report")
    lines.append("")
    lines.append("## System Architecture")
    lines.append("Pipeline uses PyMuPDF for PDF rasterization, OpenCV for preprocessing, and Tesseract for OCR.")
    lines.append("")
    lines.append("## Testing Methodology")
    lines.append("Evaluated on programmatically generated documents (PNG and PDF) with known ground truth values.")
    lines.append("This provides a deterministic baseline for field-level accuracy.")
    lines.append("")
    lines.append("## Ground Truth")
    for k, v in gt.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Results (PNG)")
    lines.append(f"Overall accuracy: {acc_png:.1%}")
    lines.append(metrics_png.to_markdown(index=False))
    lines.append("")
    lines.append("## Results (PDF)")
    lines.append(f"Overall accuracy: {acc_pdf:.1%}")
    lines.append(metrics_pdf.to_markdown(index=False))
    lines.append("")
    lines.append("## Error Analysis and Confidence")
    lines.append("Errors typically arise from font rendering and thresholding in synthetic samples; real-world scans may introduce skew and noise.")
    lines.append("Confidence can be approximated via Tesseract confidences or ensemble OCR; not implemented in this milestone.")
    lines.append("")
    lines.append("## Limitations and Mitigation")
    lines.append("- Synthetic tests underrepresent real-world variability; add more diverse scanned samples.")
    lines.append("- Layout-aware parsing (tables) and spell-check normalization improve robustness.")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def write_single_report(root: Path, reports_dir: Path, summary: Dict[str, object], acc_png: float, acc_pdf: float):
    out_dir = root / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"

    lines = []
    lines.append("# Heart Disease Risk Detection — Milestone 1 Summary")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"Total rows: {summary['total_rows']}")
    lines.append(f"Numeric missingness: {summary['missing_rate_numeric']:.2%}")
    lines.append("Sources:")
    for r in summary.get("by_source", []):
        lines.append(f"- {r['data_source']}: {r['count']}")
    lines.append("Years:")
    for r in summary.get("by_year", []):
        lines.append(f"- {r['collection_year']}: {r['count']}")
    lines.append("")
    lines.append("## OCR Snapshot Accuracy")
    lines.append(f"PNG accuracy: {acc_png:.1%}")
    lines.append(f"PDF accuracy: {acc_pdf:.1%}")
    lines.append("")
    lines.append("## Figures")
    lines.append("See reports/figures for distribution, correlation, demographics, and risk factor plots.")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def main():
    root, output, reports_dir, figures_dir = _paths()

    df = _load_or_build_ultimate(output)
    summary = _summarize_dataset(df)

    # Ensure advanced figures are present
    try:
        from scripts.generate_advanced_eda import main as eda_main
        eda_main()
    except Exception as e:
        logger.warning("Failed to regenerate figures: %s", e)

    # OCR evaluation
    _ensure_tesseract_env()
    png, pdf = _ensure_sample_docs(reports_dir)
    pred_png = extract_structured(png)
    pred_pdf = extract_structured(pdf)
    gt = _ocr_ground_truth()
    metrics_png, acc_png = _compute_ocr_metrics(pred_png, gt)
    metrics_pdf, acc_pdf = _compute_ocr_metrics(pred_pdf, gt)

    # Persist raw OCR outputs for transparency
    Path(reports_dir / "ocr_output_png.json").write_text(json.dumps(pred_png, indent=2))
    Path(reports_dir / "ocr_output_pdf.json").write_text(json.dumps(pred_pdf, indent=2))

    write_data_analysis_report(reports_dir, summary)
    write_ocr_evaluation_report(reports_dir, pred_png, pred_pdf, metrics_png, acc_png, metrics_pdf, acc_pdf)
    write_single_report(root, reports_dir, summary, acc_png, acc_pdf)


if __name__ == "__main__":
    main()
