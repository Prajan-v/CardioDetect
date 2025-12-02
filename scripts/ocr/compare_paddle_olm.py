"""Compare PaddleOCR vs olmOCR on the CBC sample PDF.

I implemented this script to:
- Run both engines (where available) on the same CBC PDF
- Compute character-, word-, and field-level accuracy vs a fixed
  ground-truth transcription
- Report timing and basic metadata from each engine
- Produce a detailed console report AND a markdown report for docs

Usage
-----
    python scripts/compare_paddle_olm.py

This script is designed to **fail gracefully**:
- If PaddleOCR is not installed, its block is reported as FAILED
- If olmOCR is not installed or cannot run on this hardware, its block
  is reported as FAILED with a clear error
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple
from difflib import SequenceMatcher
import re
import time
from datetime import datetime

# Ensure local src/ package is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paddle_ocr import PaddleOCREngine  # type: ignore
from src.olm_ocr import olmOCREngine  # type: ignore


_GROUND_TRUTH = """Patient Name: Jeevan
Age: 21 years
Sex: Male
Date: 15/06/2021

HAEMATOLOGY - COMPLETE BLOOD COUNT

Test Name            Result    Unit        Reference Range
Haemoglobin          15.5      g/dL        13.0 - 17.0
RBC Count            5.2       mill/cumm   4.5 - 5.5
PCV                  46.8      %           40 - 50
MCV                  90.0      fL          83 - 101
MCH                  29.8      pg          27 - 32
MCHC                 33.1      g/dL        31.5 - 34.5
RDW                  13.2      %           11.6 - 14.0
Total WBC Count      8500      cumm        4000 - 11000
Neutrophils          65        %           40 - 80
Lymphocytes          30        %           20 - 40
Monocytes            4         %           2 - 10
Eosinophils          1         %           1 - 6
Basophils            0         %           0 - 2
Platelet Count       250000    cumm        150000 - 410000"""


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\-\:\(\)/]", "", text)
    return text.lower().strip()


def _field_patterns() -> Dict[str, str]:
    return {
        "age": r"age[:\s]+(\d+)",
        "sex": r"sex[:\s]+(male|female)",
        "hemoglobin": r"haemoglobin[:\s]+(\d+\.?\d*)",
        "rbc": r"rbc count[:\s]+(\d+\.?\d*)",
        "wbc": r"wbc count[:\s]+(\d+)",
        "platelet": r"platelet count[:\s]+(\d+)",
    }


def _compute_accuracy_metrics(ocr_text: str) -> Dict[str, Any]:
    gt_norm = _normalize_text(_GROUND_TRUTH)
    ocr_norm = _normalize_text(ocr_text)

    # Character accuracy
    char_matcher = SequenceMatcher(None, gt_norm, ocr_norm)
    char_acc = char_matcher.ratio()

    # Word accuracy
    gt_words = set(gt_norm.split())
    ocr_words = set(ocr_norm.split())
    common = gt_words.intersection(ocr_words)
    word_acc = len(common) / len(gt_words) if gt_words else 0.0

    # Field-level accuracy
    patterns = _field_patterns()
    field_matches = 0
    total_fields = len(patterns)

    def _extract(text: str, pattern: str) -> str | None:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    field_details: Dict[str, Tuple[str | None, str | None, bool]] = {}
    for name, pat in patterns.items():
        gt_val = _extract(_GROUND_TRUTH, pat)
        ocr_val = _extract(ocr_text, pat)
        ok = gt_val is not None and gt_val == ocr_val
        if ok:
            field_matches += 1
        field_details[name] = (gt_val, ocr_val, ok)

    field_acc = field_matches / total_fields if total_fields else 0.0

    return {
        "char_accuracy": char_acc,
        "word_accuracy": word_acc,
        "field_accuracy": field_acc,
        "fields_matched": field_matches,
        "total_fields": total_fields,
        "words_matched": len(common),
        "total_words": len(gt_words),
        "field_details": field_details,
    }


def _run_paddle(pdf_path: Path) -> Dict[str, Any]:
    engine = PaddleOCREngine()
    start = time.perf_counter()
    out = engine.extract_from_pdf(str(pdf_path))
    total_time = time.perf_counter() - start

    text = str(out.get("text", ""))
    metrics = _compute_accuracy_metrics(text) if text.strip() else {}

    meta = out.get("metadata", {}) or {}

    return {
        "name": "PaddleOCR",
        "status": "ok" if "error" not in out else "error",
        "error": out.get("error"),
        "text": text,
        "total_time": total_time,
        "model_load_time": float(meta.get("model_load_time_sec", 0.0)),
        "inference_time": float(meta.get("inference_time_sec", total_time)),
        "processing_time": float(meta.get("processing_time_sec", total_time)),
        "lines_detected": int(meta.get("lines_detected", text.count("\n") + 1 if text else 0)),
        "avg_confidence": float(out.get("confidence", 0.0)),
        "chars": len(text),
        **metrics,
    }


def _run_olm(pdf_path: Path) -> Dict[str, Any]:
    engine = olmOCREngine()
    start = time.perf_counter()
    out = engine.extract_from_pdf(str(pdf_path))
    total_time = time.perf_counter() - start

    text = str(out.get("text", ""))
    metrics = _compute_accuracy_metrics(text) if text.strip() else {}
    meta = out.get("metadata", {}) or {}

    return {
        "name": "olmOCR",
        "status": "ok" if "error" not in out else "error",
        "error": out.get("error"),
        "text": text,
        "total_time": total_time,
        "model_load_time": float(meta.get("model_load_time_sec", 0.0)),
        "inference_time": float(meta.get("inference_time_sec", total_time)),
        "processing_time": float(meta.get("processing_time_sec", total_time)),
        "lines_detected": int(meta.get("lines_detected", text.count("\n") + 1 if text else 0)),
        "avg_confidence": float(out.get("confidence", 0.0)),
        "chars": len(text),
        **metrics,
    }


def _format_engine_block(result: Dict[str, Any]) -> str:
    lines: list[str] = []
    border = "┌" + "─" * 73 + "┐"
    lines.append(border)
    title = f"│ {result['name'].upper():<71} │"
    lines.append(title)
    lines.append("├" + "─" * 73 + "┤")

    if result["status"] != "ok":
        err = result.get("error", "Unknown error")
        lines.append(f"│ STATUS: FAILED - {err:<54} │")
        lines.append("└" + "─" * 73 + "┘")
        return "\n".join(lines)

    lines.append(
        f"│ Model Loading: {result['model_load_time']:.2f} seconds".ljust(73 + 2) + "│"
    )
    lines.append(
        f"│ Inference Time: {result['inference_time']:.2f} seconds".ljust(73 + 2) + "│"
    )
    lines.append(
        f"│ Total Processing: {result['processing_time']:.2f} seconds".ljust(73 + 2) + "│"
    )
    lines.append("│" + " " * 73 + "│")

    lines.append(
        f"│ Characters Extracted: {result['chars']} characters".ljust(73 + 2) + "│"
    )
    lines.append(
        f"│ Lines Detected: {result['lines_detected']} lines/blocks".ljust(73 + 2)
        + "│"
    )
    lines.append(
        f"│ Average Confidence: {result['avg_confidence']*100:5.1f}%".ljust(73 + 2)
        + "│"
    )
    lines.append("│" + " " * 73 + "│")

    lines.append(
        f"│ FIELD EXTRACTION: {result['fields_matched']}/{result['total_fields']} "
        f"fields ({result['field_accuracy']*100:4.1f}%)".ljust(73 + 2)
        + "│"
    )

    field_details = result.get("field_details", {})
    for key in ("age", "sex", "hemoglobin", "wbc", "rbc", "platelet"):
        detail = field_details.get(key)
        if not detail:
            tick = "✗"
            val = "[N/A]"
        else:
            gt, ocr, ok = detail
            tick = "✓" if ok else "✗"
            val = str(ocr)
        lines.append(f"│ {tick} {key}: {val:<60}│")

    lines.append("└" + "─" * 73 + "┘")
    return "\n".join(lines)


def _write_markdown_report(paddle: Dict[str, Any], olm: Dict[str, Any], out: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _engine_section(name: str, r: Dict[str, Any]) -> list[str]:
        lines: list[str] = []
        lines.append(f"## {name}")
        lines.append("")
        if r["status"] != "ok":
            lines.append(f"Status: **FAILED** — {r.get('error', 'Unknown error')}")
            lines.append("")
            return lines

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Model Loading | {r['model_load_time']:.2f} s |")
        lines.append(f"| Inference Time | {r['inference_time']:.2f} s |")
        lines.append(f"| Total Processing | {r['processing_time']:.2f} s |")
        lines.append(f"| Characters Extracted | {r['chars']} |")
        lines.append(f"| Lines Detected | {r['lines_detected']} |")
        lines.append(f"| Average Confidence | {r['avg_confidence']*100:.2f}% |")
        lines.append(
            f"| Field Extraction | {r['fields_matched']}/{r['total_fields']} "
            f"({r['field_accuracy']*100:.2f}%) |")
        lines.append("")
        return lines

    lines: list[str] = []
    lines.append("# OCR Engine Comparison Report: PaddleOCR vs olmOCR")
    lines.append("")
    lines.append(f"Date: {ts}")
    lines.append("")
    lines.extend(_engine_section("PaddleOCR", paddle))
    lines.extend(_engine_section("olmOCR", olm))

    # Simple recommendation based on field accuracy then char accuracy
    winner = None
    loser = None
    if paddle["status"] == "ok" and olm["status"] == "ok":
        score_p = (paddle["field_accuracy"], paddle["char_accuracy"])
        score_o = (olm["field_accuracy"], olm["char_accuracy"])
        if score_p > score_o:
            winner, loser = ("PaddleOCR", paddle), ("olmOCR", olm)
        elif score_o > score_p:
            winner, loser = ("olmOCR", olm), ("PaddleOCR", paddle)
    elif paddle["status"] == "ok" and olm["status"] != "ok":
        winner, loser = ("PaddleOCR", paddle), ("olmOCR", olm)
    elif olm["status"] == "ok" and paddle["status"] != "ok":
        winner, loser = ("olmOCR", olm), ("PaddleOCR", paddle)

    lines.append("---")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")

    if winner is None:
        lines.append("No engine completed successfully. See errors above.")
    else:
        w_name, w = winner
        lines.append(f"Primary engine: **{w_name}**")
        if loser is not None:
            l_name, l = loser
            lines.append("")
            lines.append("Rationale:")
            if w["status"] == "ok" and l["status"] != "ok":
                lines.append(
                    f"- {w_name} completed successfully, while {l_name} "
                    "failed on this environment."
                )
            else:
                lines.append(
                    f"- Higher field extraction accuracy and/or character "
                    f"accuracy compared to {l_name}."
                )
        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare PaddleOCR vs olmOCR on CBC PDF.",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf",
        help="Path to the CBC PDF document",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return

    print("=" * 80)
    print("OCR ENGINE COMPARISON REPORT")
    print("PaddleOCR vs olmOCR on Medical Document")
    print(f"Test Document: {pdf_path.name}")
    print("Hardware: MacBook Air M3, 24GB unified memory (reported)")
    print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    print()

    paddle_res = _run_paddle(pdf_path)
    olm_res = _run_olm(pdf_path)

    print("-" * 80)
    print(_format_engine_block(paddle_res))
    print()
    print(_format_engine_block(olm_res))
    print("-" * 80)

    # Simple textual recommendation in the console
    print("RECOMMENDATION")
    print("=" * 80)

    if paddle_res["status"] != "ok" and olm_res["status"] != "ok":
        print("No engine completed successfully. See errors above.")
    elif paddle_res["status"] == "ok" and olm_res["status"] != "ok":
        print("Winner: PaddleOCR (olmOCR failed on this environment)")
        print("Reason: PaddleOCR produced usable text and fields, whereas "
              "olmOCR reported an error.")
    elif olm_res["status"] == "ok" and paddle_res["status"] != "ok":
        print("Winner: olmOCR (PaddleOCR failed on this environment)")
        print("Reason: olmOCR produced usable text and fields, whereas "
              "PaddleOCR reported an error.")
    else:
        score_p = (paddle_res["field_accuracy"], paddle_res["char_accuracy"])
        score_o = (olm_res["field_accuracy"], olm_res["char_accuracy"])
        if score_p > score_o:
            print("Winner: PaddleOCR")
        elif score_o > score_p:
            print("Winner: olmOCR")
        else:
            print("Result: Tie based on field and character accuracy.")

    out_md = Path("docs") / "PADDLE_vs_OLM_COMPARISON.md"
    _write_markdown_report(paddle_res, olm_res, out_md)
    print()
    print(f"Saved markdown report to {out_md}")


if __name__ == "__main__":
    main()
