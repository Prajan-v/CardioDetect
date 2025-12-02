"""PaddleOCR-based engine for CardioDetect.

I implemented this engine as a thin wrapper around the `PaddleOCR` class
from the `paddleocr` package. My design goals were:

- Keep the public API simple: a single `extract_from_pdf` method
- Return **both** raw text and a small set of structured CBC fields
- Report timing and basic metadata for comparison experiments
- Fail gracefully when `paddleocr`/`paddlepaddle`/PDF tooling are missing

External setup (run outside Python):

    pip install paddleocr paddlepaddle pillow numpy

Optional but recommended (already present in this project):

    pip install pdf2image
    brew install poppler

If any of these dependencies are missing at runtime, this module will
return a dict with an ``error`` key instead of raising an exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import time

# Optional imports: I never want importing this module to crash.
try:  # pragma: no cover - depends on local environment
    from paddleocr import PaddleOCR  # type: ignore

    _PADDLE_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PADDLE_AVAILABLE = False

try:  # pragma: no cover - depends on local environment
    from pdf2image import convert_from_path

    _PDF2IMAGE_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PDF2IMAGE_AVAILABLE = False


@dataclass
class _PaddleLine:
    """Internal helper representing one OCR line.

    I keep this light-weight because I only need text and a confidence.
    """

    text: str
    confidence: float


def _parse_cbc_fields(text: str) -> Dict[str, Any]:
    """Parse CBC-style medical fields from OCR text.

    I deliberately mirror the regexes used elsewhere in the project so
    comparisons across engines are fair.
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


class PaddleOCREngine:
    """PaddleOCR wrapper focused on single-page CBC PDFs.

    I lazily initialize `PaddleOCR` on first use because model loading
    can take a few seconds and may download weights.
    """

    def __init__(self, use_angle_cls: bool = True, lang: str = "en") -> None:
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self._ocr: Optional[PaddleOCR] = None
        self._model_load_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load PaddleOCR model on first use.

        I keep track of how long the first initialization takes so the
        comparison script can reason about startup overhead.
        """

        if self._ocr is not None or not _PADDLE_AVAILABLE:
            return

        start = time.perf_counter()
        # Default args: English-only model, CPU inference
        self._ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang)
        self._model_load_time = time.perf_counter() - start

    def _pdf_to_images(self, pdf_path: Path) -> List["Image.Image"]:
        """Convert PDF pages to 300 DPI images.

        I prefer `pdf2image` because it provides consistent rasterization
        across platforms.
        """

        if not _PDF2IMAGE_AVAILABLE:
            raise RuntimeError(
                "pdf2image is not available. Install it with 'pip install pdf2image'.",
            )

        images = convert_from_path(str(pdf_path), dpi=300)
        return images

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_pdf(self, pdf_path: str | Path) -> Dict[str, Any]:
        """Run PaddleOCR on a PDF and return structured results.

        Returns a dict with the schema described in the user spec. On
        any failure, an ``error`` key is included and numeric metrics are
        set to safe defaults.
        """

        pdf_path = Path(pdf_path)
        result: Dict[str, Any] = {
            "text": "",
            "structured_fields": {},
            "confidence": 0.0,
            "model": "PaddleOCR",
            "metadata": {
                "processing_time_sec": 0.0,
                "model_load_time_sec": 0.0,
                "inference_time_sec": 0.0,
                "lines_detected": 0,
                "device": "CPU (M3 optimized)",
                "model_version": "unknown",
            },
        }

        if not pdf_path.exists():
            msg = f"PDF not found: {pdf_path}"
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        if not _PADDLE_AVAILABLE:
            msg = (
                "paddleocr is not installed. Run 'pip install paddleocr "
                "paddlepaddle pillow numpy' to enable this engine."
            )
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        start_total = time.perf_counter()

        # 1) Ensure model is loaded
        try:
            self._ensure_model()
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to initialize PaddleOCR: {exc}"
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        # 2) Convert PDF to images
        try:
            images = self._pdf_to_images(pdf_path)
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to convert PDF to images: {exc}"
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        if not images or self._ocr is None:
            msg = "No images produced from PDF or OCR model not initialized."
            print(f"[ERROR] {msg}")
            result["error"] = msg
            return result

        # 3) Run OCR
        start_infer = time.perf_counter()
        lines: List[_PaddleLine] = []

        for idx, img in enumerate(images, start=1):
            try:
                ocr_result = self._ocr.ocr(img, cls=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] PaddleOCR failed on page {idx}: {exc}")
                continue

            # `ocr_result` is a nested list; I flatten it to lines.
            for block in ocr_result or []:
                for line in block:
                    # line: [box, (text, score)]
                    try:
                        text, score = line[1]
                        lines.append(_PaddleLine(text=text, confidence=float(score)))
                    except Exception:  # noqa: BLE001
                        continue

        inference_time = time.perf_counter() - start_infer
        total_time = time.perf_counter() - start_total

        # 4) Aggregate
        full_text = "\n".join(l.text for l in lines)
        result["text"] = full_text

        if lines:
            avg_conf = sum(l.confidence for l in lines) / float(len(lines))
            result["confidence"] = float(avg_conf)

        result["structured_fields"] = _parse_cbc_fields(full_text)

        meta = result["metadata"]
        meta["processing_time_sec"] = float(total_time)
        meta["model_load_time_sec"] = float(self._model_load_time)
        meta["inference_time_sec"] = float(inference_time)
        meta["lines_detected"] = int(len(lines))
        # `model_version` is not trivial to retrieve; I leave it as "unknown".

        return result


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on a CBC PDF and print parsed fields.",
    )
    parser.add_argument("pdf_path", type=str, help="Path to CBC PDF report")
    args = parser.parse_args()

    engine = PaddleOCREngine()
    out = engine.extract_from_pdf(args.pdf_path)

    print("=" * 80)
    print("PADDLEOCR RESULT")
    print("=" * 80)
    print(json.dumps(out, indent=2, default=str))
