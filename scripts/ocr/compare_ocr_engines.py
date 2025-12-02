"""Compare OCR engines (Tesseract vs DeepSeek) on the CBC sample PDF.

I designed this script to:
- Run both OCR engines (where available) on the same CBC PDF
- Compute character-, word-, and field-level accuracy vs a fixed ground truth
- Measure wall-clock processing time
- Print a side-by-side comparison report
- Save a markdown report to ``docs/OCR_COMPARISON.md``

Usage
-----
    python scripts/compare_ocr_engines.py

You may optionally pass a custom PDF path and DeepSeek binary path:

    python scripts/compare_ocr_engines.py \
        --pdf CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf \
        --deepseek-binary ocr_models/deepseek-ocr
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure I can import src modules when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_manager import OCRManager, _GROUND_TRUTH  # type: ignore[attr-defined]
from src.deepseek_ocr_wrapper import DeepSeekOCREngine


def _format_engine_block(name: str, metrics: Dict[str, Any]) -> str:
    label = metrics.get("engine_label", name.upper())
    lines: list[str] = []
    border = "┌" + "─" * 73 + "┐"
    lines.append(border)
    title_line = f"│ {label.upper():<71} │"
    lines.append(title_line)
    lines.append("├" + "─" * 73 + "┤")

    status = metrics.get("status", "unknown")
    if status != "ok":
        err = metrics.get("error", "Unknown error")
        lines.append(f"│ STATUS: FAILED - {err:<54} │")
        lines.append("└" + "─" * 73 + "┘")
        return "\n".join(lines)

    elapsed = metrics.get("elapsed_seconds", 0.0)
    char_acc = metrics.get("char_accuracy", 0.0)
    word_acc = metrics.get("word_accuracy", 0.0)
    field_acc = metrics.get("field_accuracy", 0.0)
    fields_matched = metrics.get("fields_matched", 0)
    total_fields = metrics.get("total_fields", 0)
    total_chars = metrics.get("total_chars", "?")

    lines.append(f"│ Processing Time: {elapsed:.2f} seconds{' ' * 40}│")
    if isinstance(total_chars, int):
        lines.append(f"│ Characters Extracted: {total_chars:<5d}{' ' * 43}│")

    lines.append(
        f"│ Character Accuracy: {char_acc*100:5.1f}%   "
        f"Word Accuracy: {word_acc*100:5.1f}%{' ' * 8}│"
    )
    lines.append(
        f"│ Field Extraction: {fields_matched}/{total_fields} "
        f"({field_acc*100:5.1f}%){' ' * 24}│"
    )

    fields = metrics.get("structured_fields", {}) or {}
    for key in ("age", "sex", "hemoglobin", "wbc", "rbc", "platelet"):
        val = fields.get(key)
        tick = "✓" if val is not None else "✗"
        lines.append(f"│ {tick} {key}: {str(val):<60}│")

    lines.append("└" + "─" * 73 + "┘")
    return "\n".join(lines)


def _write_markdown_report(comparison: Dict[str, Any], out_path: Path) -> None:
    engines = comparison.get("engines", {})
    winner = comparison.get("recommended_engine")

    lines: list[str] = []
    lines.append("# OCR Engine Comparison Report")
    lines.append("")
    lines.append("Ground truth is the manually transcribed CBC report used in the")
    lines.append("`complete_accuracy_test.py` script.")
    lines.append("")

    for name in ("tesseract", "deepseek"):
        m = engines.get(name)
        if not m:
            continue
        label = m.get("engine_label", name)
        lines.append(f"## {label}")
        lines.append("")
        status = m.get("status", "unknown")
        if status != "ok":
            lines.append(f"Status: **FAILED** — {m.get('error', 'Unknown error')}")
            lines.append("")
            continue

        elapsed = m.get("elapsed_seconds", 0.0)
        char_acc = m.get("char_accuracy", 0.0)
        word_acc = m.get("word_accuracy", 0.0)
        field_acc = m.get("field_accuracy", 0.0)
        fields_matched = m.get("fields_matched", 0)
        total_fields = m.get("total_fields", 0)

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Processing Time | {elapsed:.2f} s |")
        lines.append(f"| Character Accuracy | {char_acc*100:.2f}% |")
        lines.append(f"| Word Accuracy | {word_acc*100:.2f}% |")
        lines.append(
            f"| Field Extraction | {fields_matched}/{total_fields} "
            f"({field_acc*100:.2f}%) |"
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    if winner is None:
        lines.append("**Recommendation:** No OCR engine succeeded; see error details above.")
    else:
        label = engines[winner].get("engine_label", winner)
        lines.append(f"**Recommendation:** Use **{label}** as the primary OCR engine.")
        lines.append(
            "Tesseract remains a fallback when the DeepSeek binary is missing "
            "or fails."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare OCR engines (Tesseract vs DeepSeek) on CBC PDF.",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf",
        help="Path to the CBC PDF document",
    )
    parser.add_argument(
        "--deepseek-binary",
        type=str,
        default=None,
        help="Optional custom path to deepseek-ocr binary",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    # If a custom binary path is provided, I patch the DeepSeek engine's
    # default path before running the comparison, so OCRManager will pick
    # it up.
    if args.deepseek_binary is not None:
        engine = DeepSeekOCREngine(binary_path=args.deepseek_binary)
        if not engine.is_available():
            print(
                f"[WARN] Provided DeepSeek binary '{args.deepseek_binary}' "
                "does not seem executable. Comparison will still run, but "
                "DeepSeek will likely report an error."
            )

    print("=" * 80)
    print("OCR ENGINE COMPARISON REPORT")
    print(f"Test Document: {pdf_path}")
    print("=" * 80)
    print()

    comparison = OCRManager.compare_engines(pdf_path)
    engines = comparison.get("engines", {})

    for name in ("tesseract", "deepseek"):
        m = engines.get(name)
        if not m:
            continue
        print(_format_engine_block(name, m))
        print()

    winner = comparison.get("recommended_engine")
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if winner is None:
        print("Winner: None (no engine succeeded)")
        print("Reason: All OCR engines reported errors; see blocks above.")
    else:
        m = engines[winner]
        label = m.get("engine_label", winner)
        print(f"Winner: {label}")
        print(
            "Reason: highest field accuracy and character accuracy among "
            "engines that completed successfully."
        )

    docs_path = Path("docs") / "OCR_COMPARISON.md"
    _write_markdown_report(comparison, docs_path)
    print()
    print(f"Saved markdown report to {docs_path}")


if __name__ == "__main__":
    main()
