"""Tesseract-based OCR utilities for CardioDetect.

This module provides a thin wrapper around Tesseract OCR for extracting
text and key CBC-style laboratory fields from PDF reports.

I designed it to:
- Work with multi-page PDFs (processing each page as an image)
- Return both raw text and a small structured dict of key fields
- Fail gracefully if system dependencies are missing

External dependencies (NOT installed by this module):
- poppler (for `pdftoppm` used by pdf2image)
- tesseract (OCR engine)
- Python packages: pdf2image, pillow, pytesseract

Example Homebrew + pip setup on macOS:

    brew install poppler tesseract
    pip install pdf2image pillow pytesseract

If any of these are missing, `extract_text_tesseract` will return an
object with an `error` key explaining what went wrong instead of raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import re
import shutil
import time


# Optional imports for OCR backends. I intentionally guard them so that
# importing this module never crashes even if OCR deps are absent.
try:  # pragma: no cover - import is environment-dependent
    from pdf2image import convert_from_path
    import pytesseract
    from pytesseract import Output as TesseractOutput  # type: ignore[attr-defined]

    _PDF2IMAGE_AVAILABLE = True
    _PYTESSERACT_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PDF2IMAGE_AVAILABLE = False
    _PYTESSERACT_AVAILABLE = False


def _have_system_binaries() -> bool:
    """Return True if required CLI tools are visible on PATH.

    I check for both `tesseract` and `pdftoppm` because pdf2image relies on
    the latter (from poppler) for PDF→image conversion.
    """

    has_tesseract = shutil.which("tesseract") is not None
    has_pdftoppm = shutil.which("pdftoppm") is not None
    return bool(has_tesseract and has_pdftoppm)


def _extract_structured_fields(text: str) -> Dict[str, str | None]:
    """Extract key CBC fields from OCR text using simple regexes.

    I keep this intentionally simple and robust for the sample CBC report.
    All matches are returned as raw strings (no unit conversion).
    """

    def _search(pattern: str) -> str | None:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    fields: Dict[str, str | None] = {
        "age": _search(r"age[:\s]+(\d{1,3})"),
        "sex": _search(r"sex[:\s]+(male|female)"),
        "hemoglobin": _search(r"haemoglobin\s+(\d+\.?\d*)"),
        "rbc": _search(r"rbc\s+count\s+(\d+\.?\d*)"),
        "wbc": _search(r"(?:total\s+)?wbc\s+count\s+(\d+)"),
        "platelet": _search(r"platelet\s+count\s+(\d+)")
    }
    return fields


def extract_text_tesseract(pdf_path: str | Path) -> Dict[str, Any]:
    """Run Tesseract OCR on a PDF and return text + parsed fields.

    Parameters
    ----------
    pdf_path:
        Path to the input PDF file.

    Returns
    -------
    dict
        A dictionary with at least the following keys:

        - ``text``: full OCR text (possibly empty on failure)
        - ``structured_fields``: dict with keys ``age``, ``sex``,
          ``hemoglobin``, ``rbc``, ``wbc``, ``platelet`` (values may be None)
        - ``confidence``: mean Tesseract confidence in [0, 1]
        - ``model``: fixed string ``"Tesseract"``

        On errors, an additional ``error`` key is added with a human
        readable explanation, and ``confidence`` will be 0.0.
    """

    result: Dict[str, Any] = {
        "text": "",
        "structured_fields": {},
        "confidence": 0.0,
        "model": "Tesseract",
    }

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        msg = f"PDF not found: {pdf_path}"
        print(f"[ERROR] {msg}")
        result["error"] = msg
        return result

    if not (_PDF2IMAGE_AVAILABLE and _PYTESSERACT_AVAILABLE):
        msg = (
            "Tesseract OCR dependencies are missing. Install pdf2image, "
            "pytesseract, and pillow, and ensure poppler/tesseract are "
            "available on your system."
        )
        print(f"[ERROR] {msg}")
        result["error"] = msg
        return result

    if not _have_system_binaries():
        msg = (
            "Required system binaries 'tesseract' and/or 'pdftoppm' are not "
            "on PATH. On macOS you can run 'brew install poppler tesseract'."
        )
        print(f"[ERROR] {msg}")
        result["error"] = msg
        return result

    start = time.perf_counter()

    try:
        pages = convert_from_path(str(pdf_path), dpi=300)
    except Exception as exc:  # noqa: BLE001
        msg = f"pdf2image.convert_from_path failed: {exc}"
        print(f"[ERROR] {msg}")
        result["error"] = msg
        return result

    texts: list[str] = []
    confidences: list[int] = []

    for idx, img in enumerate(pages, start=1):
        try:
            page_text = pytesseract.image_to_string(img, lang="eng")
            texts.append(page_text)

            data = pytesseract.image_to_data(
                img,
                lang="eng",
                output_type=TesseractOutput.DICT,
            )
            conf_vals = [
                int(c)
                for c in data.get("conf", [])
                if isinstance(c, (str, int)) and str(c).isdigit()
            ]
            confidences.extend(conf_vals)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Tesseract failed on page {idx}: {exc}")
            continue

    full_text = "\n".join(texts)
    result["text"] = full_text

    if confidences:
        # Tesseract confidences are 0–100; I normalize to 0–1.
        mean_conf = sum(confidences) / float(len(confidences))
        result["confidence"] = mean_conf / 100.0

    result["structured_fields"] = _extract_structured_fields(full_text)
    result["elapsed_seconds"] = time.perf_counter() - start

    return result


if __name__ == "__main__":  # pragma: no cover - manual smoke test helper
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Tesseract OCR on a PDF and print parsed fields.",
    )
    parser.add_argument("pdf_path", type=str, help="Path to CBC PDF report")
    args = parser.parse_args()

    out = extract_text_tesseract(args.pdf_path)
    print("=" * 80)
    print("TESSERACT OCR RESULT")
    print("=" * 80)
    if "error" in out:
        print("[ERROR]", out["error"])
    else:
        print(f"Model: {out['model']}")
        print(f"Confidence: {out['confidence']:.3f}")
        print("Structured fields:")
        for k, v in out["structured_fields"].items():
            print(f"  - {k}: {v}")
        print("\nPreview:\n")
        print(out["text"][:800])
        if len(out["text"]) > 800:
            print("... (truncated)")
