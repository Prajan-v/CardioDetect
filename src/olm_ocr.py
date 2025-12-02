"""olmOCR-based engine wrapper for CardioDetect.

I implemented this module as a *best-effort* integration with the
`olmocr` toolkit from AllenAI. The official pipeline is designed for
NVIDIA GPUs and is invoked via a CLI like::

    python -m olmocr.pipeline ./workspace --markdown --pdfs my.pdf

On your MacBook Air M3 (Apple Silicon, no NVIDIA GPU), this pipeline is
very likely **not runnable** locally. My priorities here are:

- Provide a clean `olmOCREngine` class with `extract_from_pdf()` matching
  the same schema as the PaddleOCR engine
- Attempt to call the official CLI when `olmocr` is installed
- Fail gracefully with a clear error message when the environment does
  not satisfy olmOCR's requirements (e.g., no GPU)

Because I cannot rely on GPU availability or heavy dependencies inside
this project, I avoid importing large frameworks at module import time.
Instead, I rely on checking for the `olmocr` Python package with
`importlib` and then invoking the CLI via :mod:`subprocess`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import importlib.util
import os
import re
import subprocess
import tempfile
import time


def _olmocr_available() -> bool:
    """Return True if the `olmocr` package looks importable.

    I intentionally use :func:`importlib.util.find_spec` instead of
    importing the package directly to avoid triggering heavyweight
    initialization on import.
    """

    return importlib.util.find_spec("olmocr") is not None


def _parse_cbc_fields(text: str) -> Dict[str, Any]:
    """Parse CBC-style medical fields from OCR text.

    I reuse the same regex patterns as other engines so that comparisons
    remain fair.
    """

    def _search(pattern: str, cast):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except Exception:  # noqa: BLE001
            return None

    fields: Dict[str, Any] = {
        "age": _search(r"age[:\s]+(\d{1,3})", int),
        "sex": _search(r"sex[:\s]+(male|female)", str),
        "hemoglobin": _search(r"haemoglobin\s+(\d+\.?\d*)", float),
        "wbc": _search(r"(?:total\s+)?wbc\s+count\s+(\d+)", int),
        "rbc": _search(r"rbc\s+count\s+(\d+\.?\d*)", float),
        "platelet": _search(r"platelet\s+count\s+(\d+)", int),
    }
    return fields


def _markdown_to_text(markdown_text: str) -> str:
    """Convert basic Markdown to plain text.

    I intentionally keep this simple: strip leading heading markers and
    table pipes, and collapse multiple spaces.
    """

    lines = []
    for raw in markdown_text.splitlines():
        line = raw.lstrip()
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if line.startswith("|") and line.endswith("|"):
            # Very naive table strip: drop leading/trailing pipes
            line = line.strip("|")
        lines.append(line)
    text = "\n".join(lines)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class olmOCREngine:
    """Wrapper around the `olmocr` CLI pipeline.

    On a machine with a supported NVIDIA GPU and the `olmocr[gpu]`
    extras installed, this class can:

    - Run ``python -m olmocr.pipeline`` on a PDF
    - Read the produced Markdown output
    - Convert it to plain text and parse CBC fields

    On your current M3 Mac (no NVIDIA GPU), I expect the CLI invocation
    to fail; in that case I surface the error in the returned dict.
    """

    def __init__(self, python_executable: str = "python") -> None:
        self.python_executable = python_executable

    def extract_from_pdf(self, pdf_path: str | Path) -> Dict[str, Any]:
        """Run olmOCR pipeline on a PDF and return structured results.

        Returns a dict with the same schema as the PaddleOCR engine. On
        failure, includes an ``error`` key with a human-readable
        message. I also include basic timing metadata, but I cannot
        cleanly separate model loading from inference when calling the
        CLI, so both are effectively merged into ``processing_time_sec``.
        """

        pdf_path = Path(pdf_path)
        result: Dict[str, Any] = {
            "text": "",
            "structured_fields": {},
            "confidence": 0.0,
            "model": "olmOCR",
            "metadata": {
                "processing_time_sec": 0.0,
                "model_load_time_sec": 0.0,
                "inference_time_sec": 0.0,
                "lines_detected": 0,
                "device": "Unknown (olmOCR pipeline)",
                "model_version": "unknown",
            },
        }

        if not pdf_path.exists():
            msg = f"PDF not found: {pdf_path}"
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        if not _olmocr_available():
            msg = (
                "olmocr is not installed in this environment. It is "
                "designed for NVIDIA GPUs and a dedicated conda env. "
                "See https://github.com/allenai/olmocr for installation "
                "instructions."
            )
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        start_total = time.perf_counter()

        # Use a temporary workspace, mirroring the README examples.
        with tempfile.TemporaryDirectory(prefix="olmocr_ws_") as tmpdir:
            workspace = Path(tmpdir)
            cmd = [
                self.python_executable,
                "-m",
                "olmocr.pipeline",
                str(workspace),
                "--markdown",
                "--pdfs",
                str(pdf_path),
            ]

            try:
                completed = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"Failed to invoke olmocr pipeline: {exc}"
                print(f"[ERROR] {msg}")
                result["error"] = msg
                return result

            if completed.returncode != 0:
                msg = (
                    "olmocr pipeline exited with non-zero status "
                    f"{completed.returncode}: {completed.stderr.strip()}"
                )
                print(f"[ERROR] {msg}")
                result["error"] = msg
                return result

            # Locate Markdown output (if any)
            md_root = workspace / "markdown"
            md_path: Optional[Path] = None
            if md_root.exists():
                for p in md_root.glob("*.md"):
                    md_path = p
                    break

            if md_path is None or not md_path.exists():
                msg = (
                    "olmocr pipeline reported success, but no markdown "
                    "output was found in the workspace."
                )
                print(f"[ERROR] {msg}")
                result["error"] = msg
                return result

            try:
                markdown_text = md_path.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                msg = f"Failed to read olmocr markdown output: {exc}"
                print(f"[ERROR] {msg}")
                result["error"] = msg
                return result

        total_time = time.perf_counter() - start_total

        plain_text = _markdown_to_text(markdown_text)
        result["text"] = plain_text
        result["structured_fields"] = _parse_cbc_fields(plain_text)

        # The olmocr pipeline does not expose per-token confidences via
        # this CLI path, so I leave `confidence` at 0.0 and let higher-
        # level evaluation scripts compute empirical accuracies instead.

        meta = result["metadata"]
        meta["processing_time_sec"] = float(total_time)
        meta["inference_time_sec"] = float(total_time)
        meta["lines_detected"] = int(plain_text.count("\n") + 1 if plain_text else 0)

        return result


if __name__ == "__main__":  # pragma: no cover - manual smoke test helper
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Run olmOCR pipeline on a CBC PDF and print parsed fields. "
            "This will only work in an environment where olmocr is "
            "properly installed and a compatible GPU is available."
        ),
    )
    parser.add_argument("pdf_path", type=str, help="Path to CBC PDF report")
    args = parser.parse_args()

    engine = olmOCREngine()
    out = engine.extract_from_pdf(args.pdf_path)

    print("=" * 80)
    print("olmOCR RESULT")
    print("=" * 80)
    print(json.dumps(out, indent=2, default=str))
