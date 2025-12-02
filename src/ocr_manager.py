"""Unified OCR manager for CardioDetect.

This module provides a single interface over multiple OCR backends:

- Tesseract OCR via :mod:`src.tesseract_ocr`
- DeepSeek-OCR Rust binary via :mod:`src.deepseek_ocr_wrapper`
- Optional PyMuPDF fallback when all else fails

I use the same ground-truth string and regex patterns as in the
`complete_accuracy_test.py` script so that accuracy metrics are
comparable across engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Literal
import re
import time

from src.tesseract_ocr import extract_text_tesseract
from src.deepseek_ocr_wrapper import DeepSeekOCREngine
from src.paddle_ocr import PaddleOCREngine
from src.olm_ocr import olmOCREngine

EngineName = Literal["tesseract", "deepseek", "paddle", "olm", "pymupdf"]


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
    """Normalize text for char/word-level comparison."""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\-\:\(\)/]", "", text)
    return text.lower().strip()


def _field_patterns() -> Dict[str, str]:
    """Return regex patterns for the key CBC-style fields."""

    return {
        "age": r"age[:\s]+(\d+)",
        "sex": r"sex[:\s]+(male|female)",
        "hemoglobin": r"haemoglobin[:\s]+(\d+\.?\d*)",
        "rbc": r"rbc count[:\s]+(\d+\.?\d*)",
        "wbc": r"wbc count[:\s]+(\d+)",
        "platelet": r"platelet count[:\s]+(\d+)",
    }


def _compute_metrics(ocr_text: str) -> Dict[str, float | int]:
    """Compute character-, word-, and field-level accuracy metrics."""

    gt_normalized = _normalize_text(_GROUND_TRUTH)
    ocr_normalized = _normalize_text(ocr_text)

    # Character-level
    char_matcher = SequenceMatcher(None, gt_normalized, ocr_normalized)
    char_accuracy = char_matcher.ratio()

    # Word-level
    gt_words = set(gt_normalized.split())
    ocr_words = set(ocr_normalized.split())
    common_words = gt_words.intersection(ocr_words)
    word_accuracy = len(common_words) / len(gt_words) if gt_words else 0.0

    # Field-level
    patterns = _field_patterns()
    field_matches = 0
    field_total = len(patterns)

    def _extract_field(text: str, pattern: str) -> str | None:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    for field_name, pattern in patterns.items():
        gt_value = _extract_field(_GROUND_TRUTH, pattern)
        ocr_value = _extract_field(ocr_text, pattern)
        if gt_value is not None and gt_value == ocr_value:
            field_matches += 1

    field_accuracy = field_matches / field_total if field_total else 0.0

    return {
        "char_accuracy": float(char_accuracy),
        "word_accuracy": float(word_accuracy),
        "field_accuracy": float(field_accuracy),
        "fields_matched": int(field_matches),
        "total_fields": int(field_total),
        "words_matched": int(len(common_words)),
        "total_words": int(len(gt_words)),
    }


@dataclass
class OCREngineResult:
    """Container for a single OCR engine run."""

    engine: EngineName
    text: str
    structured_fields: Dict[str, Any]
    elapsed_seconds: float
    confidence: float
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - simple adapter
        data: Dict[str, Any] = {
            "engine": self.engine,
            "text": self.text,
            "structured_fields": self.structured_fields,
            "elapsed_seconds": self.elapsed_seconds,
            "confidence": self.confidence,
        }
        if self.error is not None:
            data["error"] = self.error
        return data


class OCRManager:
    """Unified interface for multiple OCR engines.

    Parameters
    ----------
    engine:
        One of ``"tesseract"``, ``"deepseek"``, or ``"auto"``. In
        ``"auto"`` mode I try DeepSeek first (to get best accuracy when
        available), then fall back to Tesseract, and finally to a simple
        PyMuPDF text extractor if nothing else works.
    """

    def __init__(self, engine: str = "auto") -> None:
        self.engine = engine

    # ------------------------------------------------------------------
    # Low-level engine runners
    # ------------------------------------------------------------------

    def _run_tesseract(self, pdf_path: Path) -> OCREngineResult:
        start = time.perf_counter()
        res = extract_text_tesseract(pdf_path)
        elapsed = res.get("elapsed_seconds", time.perf_counter() - start)
        return OCREngineResult(
            engine="tesseract",
            text=str(res.get("text", "")),
            structured_fields=dict(res.get("structured_fields", {})),
            confidence=float(res.get("confidence", 0.0)),
            elapsed_seconds=float(elapsed),
            error=str(res["error"]) if "error" in res else None,
        )

    def _run_paddle(self, pdf_path: Path) -> OCREngineResult:
        start = time.perf_counter()
        engine = PaddleOCREngine()
        res = engine.extract_from_pdf(str(pdf_path))
        meta = res.get("metadata", {}) or {}
        elapsed = meta.get("processing_time_sec", time.perf_counter() - start)
        return OCREngineResult(
            engine="paddle",
            text=str(res.get("text", "")),
            structured_fields=dict(res.get("structured_fields", {})),
            confidence=float(res.get("confidence", 0.0)),
            elapsed_seconds=float(elapsed),
            error=str(res["error"]) if "error" in res else None,
        )

    def _run_olm(self, pdf_path: Path) -> OCREngineResult:
        start = time.perf_counter()
        engine = olmOCREngine()
        res = engine.extract_from_pdf(str(pdf_path))
        meta = res.get("metadata", {}) or {}
        elapsed = meta.get("processing_time_sec", time.perf_counter() - start)
        return OCREngineResult(
            engine="olm",
            text=str(res.get("text", "")),
            structured_fields=dict(res.get("structured_fields", {})),
            confidence=float(res.get("confidence", 0.0)),
            elapsed_seconds=float(elapsed),
            error=str(res["error"]) if "error" in res else None,
        )

    def _run_deepseek(self, pdf_path: Path) -> OCREngineResult:
        engine = DeepSeekOCREngine()
        start = time.perf_counter()
        res = engine.extract_from_pdf(pdf_path)
        elapsed = res.get("elapsed_seconds", time.perf_counter() - start)
        return OCREngineResult(
            engine="deepseek",
            text=str(res.get("text", "")),
            structured_fields=dict(res.get("structured_fields", {})),
            confidence=float(res.get("confidence", 0.0)),
            elapsed_seconds=float(elapsed),
            error=str(res["error"]) if "error" in res else None,
        )

    def _run_pymupdf(self, pdf_path: Path) -> OCREngineResult:
        try:
            import fitz  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return OCREngineResult(
                engine="pymupdf",
                text="",
                structured_fields={},
                elapsed_seconds=0.0,
                confidence=0.0,
                error=(
                    "PyMuPDF (fitz) is not installed; please run "
                    "'pip install pymupdf' if you want the fallback "
                    "extractor. Original error: " + str(exc)
                ),
            )

        start = time.perf_counter()
        try:
            doc = fitz.open(str(pdf_path))  # type: ignore[arg-type]
            texts: list[str] = []
            for page in doc:
                texts.append(page.get_text("text"))
            doc.close()
            text = "\n".join(texts)
            elapsed = time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001
            return OCREngineResult(
                engine="pymupdf",
                text="",
                structured_fields={},
                elapsed_seconds=time.perf_counter() - start,
                confidence=0.0,
                error=f"PyMuPDF text extraction failed: {exc}",
            )

        # For fairness, I do not attempt to compute a real confidence here.
        return OCREngineResult(
            engine="pymupdf",
            text=text,
            structured_fields={},
            elapsed_seconds=elapsed,
            confidence=0.0,
            error=None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, pdf_path: str | Path) -> Dict[str, Any]:
        """Extract text using the configured engine.

        In ``engine='auto'`` mode I try DeepSeek → Tesseract → PyMuPDF.
        The first engine that succeeds (no ``error``) wins.
        """

        pdf_path = Path(pdf_path)
        last_error: str | None = None

        def _ok(res: OCREngineResult) -> bool:
            return res.error is None and bool(res.text.strip())

        if self.engine == "tesseract":
            res = self._run_tesseract(pdf_path)
            return res.to_dict()

        if self.engine == "deepseek":
            res = self._run_deepseek(pdf_path)
            return res.to_dict()

        if self.engine == "paddle":
            res = self._run_paddle(pdf_path)
            return res.to_dict()

        if self.engine == "olm":
            res = self._run_olm(pdf_path)
            return res.to_dict()

        # auto: prefer highest quality engines first, then fall back
        # to lighter-weight or classical options. I currently assume
        # the following priority:
        #   olmOCR (research-grade VLM) → PaddleOCR → DeepSeek →
        #   Tesseract → PyMuPDF text extraction.
        for runner in (
            self._run_olm,
            self._run_paddle,
            self._run_deepseek,
            self._run_tesseract,
            self._run_pymupdf,
        ):
            res = runner(pdf_path)
            if _ok(res):
                return res.to_dict()
            last_error = res.error

        # If nothing worked, return a unified error structure
        return {
            "engine": self.engine,
            "text": "",
            "structured_fields": {},
            "elapsed_seconds": 0.0,
            "confidence": 0.0,
            "error": last_error or "All OCR engines failed for this PDF.",
        }

    # ------------------------------------------------------------------
    # Comparison helper
    # ------------------------------------------------------------------

    @staticmethod
    def compare_engines(pdf_path: str | Path) -> Dict[str, Any]:
        """Run both engines on a PDF and compute accuracy metrics.

        This function does **not** perform any printing. It returns a
        structured dictionary that higher-level scripts (like
        ``scripts/compare_ocr_engines.py`` or
        ``scripts/complete_accuracy_test.py``) can use to render
        comparison tables.
        """

        pdf_path = Path(pdf_path)
        mgr_t = OCRManager(engine="tesseract")
        mgr_d = OCRManager(engine="deepseek")

        engines: Dict[str, Dict[str, Any]] = {}

        for name, mgr in ("tesseract", mgr_t), ("deepseek", mgr_d):
            raw = mgr.extract(pdf_path)
            metrics: Dict[str, Any] = {
                "status": "error" if "error" in raw else "ok",
                "error": raw.get("error"),
                "engine_label": "Tesseract" if name == "tesseract" else "DeepSeek-OCR",
            }

            text = str(raw.get("text", ""))
            if text.strip():
                metrics.update(_compute_metrics(text))
                metrics["total_chars"] = len(text)
            else:
                # Fill with zeros if text is empty
                metrics.update(
                    {
                        "char_accuracy": 0.0,
                        "word_accuracy": 0.0,
                        "field_accuracy": 0.0,
                        "fields_matched": 0,
                        "total_fields": len(_field_patterns()),
                        "words_matched": 0,
                        "total_words": len(_normalize_text(_GROUND_TRUTH).split()),
                    }
                )

            metrics["elapsed_seconds"] = float(raw.get("elapsed_seconds", 0.0))
            metrics["confidence"] = float(raw.get("confidence", 0.0))
            metrics["structured_fields"] = dict(raw.get("structured_fields", {}))

            engines[name] = metrics

        # Choose recommended engine: highest field_accuracy, then char_accuracy
        best_engine: str | None = None
        best_score: float = -1.0
        for name, m in engines.items():
            if m["status"] != "ok":
                continue
            score = float(m["field_accuracy"]) * 100.0 + float(m["char_accuracy"])
            if score > best_score:
                best_score = score
                best_engine = name

        return {
            "engines": engines,
            "recommended_engine": best_engine,
        }
