"""DeepSeek-OCR (Rust binary) wrapper for CardioDetect.

This module wraps the `deepseek-ocr` macOS ARM64 binary from the
`deepseek-ocr.rs` project, so I can call it from Python as an OCR engine.

The expected setup (performed *outside* this module) is:

    mkdir -p ocr_models
    cd ocr_models
    curl -L \
      https://github.com/TimmyOVO/deepseek-ocr.rs/releases/latest/download/deepseek-ocr-macos-arm64 \
      -o deepseek-ocr
    chmod +x deepseek-ocr

I assume the binary lives at ``ocr_models/deepseek-ocr`` relative to the
project root unless a custom path is provided.

The CLI usage is based on the upstream README; the wrapper calls the
binary roughly as:

    deepseek-ocr \
      --prompt "<image>\n<|grounding|>Extract all text from this lab report." \
      --image /path/to/page.png \
      --device metal --dtype f16 --max-new-tokens 512

Because this is an external project, the exact flags may evolve. I
therefore:
- Capture stdout/stderr
- Surface any non-zero exit codes as errors in the returned dict
- Fail gracefully if the binary is missing or crashes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os
import re
import subprocess
import tempfile
import time

try:  # pragma: no cover - depends on local environment
    import fitz  # type: ignore

    _PYMUPDF_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PYMUPDF_AVAILABLE = False


def _extract_structured_fields(text: str) -> Dict[str, str | None]:
    """Extract key CBC-style fields from DeepSeek text output.

    I intentionally mirror the patterns used in the Tesseract wrapper so
    comparison between engines is fair.
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


class DeepSeekOCREngine:
    """Thin wrapper around the DeepSeek-OCR Rust binary.

    I focus on a simple "extract text from first page" workflow, which is
    enough to compare OCR quality on the sample CBC report. Multi-page
    support can be added later if needed.
    """

    def __init__(self, binary_path: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        default_path = project_root / "ocr_models" / "deepseek-ocr"
        self.binary_path = Path(binary_path) if binary_path is not None else default_path

    def is_available(self) -> bool:
        """Return True if the DeepSeek binary exists and is executable."""

        return self.binary_path.exists() and os.access(self.binary_path, os.X_OK)

    def extract_from_pdf(self, pdf_path: str | Path) -> Dict[str, Any]:
        """Run DeepSeek-OCR on the first page of a PDF.

        Parameters
        ----------
        pdf_path:
            Path to the PDF file to process.

        Returns
        -------
        dict
            A dictionary with at least the following keys:

            - ``text``: OCR output text (possibly empty on failure)
            - ``structured_fields``: dict of parsed CBC-style fields
            - ``confidence``: placeholder float (0.0 – DeepSeek does not
              expose token-level confidences via CLI)
            - ``model``: fixed string ``"DeepSeek-OCR"``

            On failure, an additional ``error`` key is present.
        """

        result: Dict[str, Any] = {
            "text": "",
            "structured_fields": {},
            "confidence": 0.0,
            "model": "DeepSeek-OCR",
        }

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            msg = f"PDF not found: {pdf_path}"
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        if not self.is_available():
            msg = (
                "DeepSeek-OCR binary not found or not executable at "
                f"'{self.binary_path}'. Please download it as described in "
                "the project README."
            )
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        if not _PYMUPDF_AVAILABLE:
            msg = (
                "PyMuPDF (fitz) is required to render PDF pages to images "
                "for DeepSeek-OCR. Install it with 'pip install pymupdf'."
            )
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        start = time.perf_counter()

        # Render first page to a temporary PNG image
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir_path = Path(tmpdir)
            image_path = tmp_dir_path / "page1.png"

            try:
                doc = fitz.open(str(pdf_path))  # type: ignore[arg-type]
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=300)
                pix.save(str(image_path))
                doc.close()
            except Exception as exc:  # noqa: BLE001
                msg = f"Failed to render PDF page to image: {exc}"
                print(f"[ERROR] {msg}")
                result["error"] = msg
                return result

            prompt = (
                "<image>\n<|grounding|>Extract all text from this lab report "
                "in plain, line-oriented format."
            )

            cmd = [
                str(self.binary_path),
                "--prompt",
                prompt,
                "--image",
                str(image_path),
                "--device",
                "metal",
                "--dtype",
                "f16",
                "--max-new-tokens",
                "512",
            ]

            try:
                completed = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"Failed to invoke DeepSeek-OCR binary: {exc}"
                print(f"[ERROR] {msg}")
                result["error"] = msg
                return result

        if completed.returncode != 0:
            msg = (
                "DeepSeek-OCR process exited with non-zero status "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        text_output = completed.stdout.strip()
        result["text"] = text_output
        result["structured_fields"] = _extract_structured_fields(text_output)
        result["elapsed_seconds"] = time.perf_counter() - start

        # I leave confidence at 0.0 for now; the CLI does not expose
        # token-wise confidences. The comparison script will compute
        # empirical accuracies against ground truth instead.
        return result


if __name__ == "__main__":  # pragma: no cover - manual smoke test helper
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR on a PDF and print parsed fields.",
    )
    parser.add_argument("pdf_path", type=str, help="Path to CBC PDF report")
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help="Optional custom path to deepseek-ocr binary",
    )
    args = parser.parse_args()

    engine = DeepSeekOCREngine(binary_path=args.binary)
    out = engine.extract_from_pdf(args.pdf_path)

    print("=" * 80)
    print("DEEPSEEK-OCR RESULT")
    print("=" * 80)
    if "error" in out:
        print("[ERROR]", out["error"])
    else:
        print(f"Model: {out['model']}")
        print("Structured fields:")
        for k, v in out["structured_fields"].items():
            print(f"  - {k}: {v}")
        print("\nPreview:\n")
        print(out["text"][:800])
        if len(out["text"]) > 800:
            print("... (truncated)")
