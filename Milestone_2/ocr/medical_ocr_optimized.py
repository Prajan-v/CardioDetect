import cv2
import numpy as np
import pytesseract
from PIL import Image  # noqa: F401
from pdf2image import convert_from_path
from pathlib import Path
import re
from typing import Dict, Any, Optional
import os


class MedicalOCROptimized:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _try_digital_extraction(self, pdf_path: Path) -> Optional[str]:
        try:
            import fitz  # type: ignore
        except Exception:
            return None

        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            return None

        texts = []
        for page in doc:
            try:
                texts.append(page.get_text("text"))
            except Exception:
                continue
        doc.close()
        full_text = "".join(texts)
        if len(full_text) > 200:
            return full_text
        return None

    def _preprocess(self, pil_image: Image.Image, dpi: int) -> np.ndarray:
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        os.makedirs("temp", exist_ok=True)
        out_path = Path("temp") / f"preprocessed_{dpi}.png"
        cv2.imwrite(str(out_path), binary)
        return binary

    def _run_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        config = r"--oem 3 --psm 6"
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(image, config=config)
        confs = []
        for c in data.get("conf", []):
            try:
                v = float(c)
            except Exception:
                continue
            if v >= 0:
                confs.append(v)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return {"text": text, "avg_conf": avg_conf / 100.0}

    def _parse_fields(self, text: str) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}

        def _search(patterns, cast, validate=None):
            for pat in patterns:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if not m:
                    continue
                try:
                    val = cast(m.group(1))
                except Exception:
                    continue
                if validate is not None and not validate(val):
                    continue
                return val
            return None

        age = _search([
            r"age[:\s]+(\d+)",
            r"(\d+)\s*years?\s*old",
        ], int, lambda v: 0 < v < 120)
        if age is not None:
            fields["age"] = int(age)

        sex_raw = _search([
            r"sex[:\s]+(male|female|m|f)",
        ], str)
        if sex_raw is not None:
            token = str(sex_raw).strip().lower()
            if token in {"male", "m"}:
                fields["sex"] = "Male"
                fields["sex_code"] = 1
            elif token in {"female", "f"}:
                fields["sex"] = "Female"
                fields["sex_code"] = 0

        hb = _search([
            r"h[ae]emoglobin[:\s]+(\d+\.?\d*)",
            r"h[ae]emoglobin\s+(\d+\.?\d*)",
        ], float, lambda v: 5.0 <= v <= 20.0)
        if hb is not None:
            fields["hemoglobin"] = float(hb)

        wbc = _search([
            r"(?:total\s+)?wbc[:\s]+(\d+)",
            r"(?:total\s+)?wbc\s+count\s+(\d+)",
        ], int, lambda v: 3000 <= v <= 15000)
        if wbc is not None:
            fields["wbc"] = int(wbc)

        rbc = _search([
            r"rbc[:\s]+(\d+\.?\d*)",
            r"rbc\s+count\s+(\d+\.?\d*)",
        ], float, lambda v: 3.0 <= v <= 8.0)
        if rbc is not None:
            fields["rbc"] = float(rbc)

        platelet = _search([
            r"platelet[:\s]+(\d+)",
            r"platelet\s+count\s+(\d+)",
        ], int, lambda v: 100000 <= v <= 500000)
        if platelet is not None:
            fields["platelet"] = int(platelet)

        return fields

    def extract_from_pdf(self, pdf_path: str | Path, dpi: int = 300, retry_dpi: int = 400, conf_threshold: float = 70.0) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        self._log(f"Medical OCR - {pdf_path.name}")

        text = ""
        method = "ocr"
        dpi_used: Optional[int] = None
        confidence = 0.0

        digital_text = self._try_digital_extraction(pdf_path)
        if digital_text:
            text = digital_text
            method = "digital_extraction"
            dpi_used = None
            confidence = 1.0
            self._log("✓ Digital PDF detected, using text extraction (skipping OCR)")
        else:
            self._log(f"⏳ Converting PDF to {dpi} DPI image...")
            images = convert_from_path(str(pdf_path), dpi=dpi)
            if not images:
                return {
                    "text": "",
                    "fields": {},
                    "confidence": 0.0,
                    "quality": "low",
                    "method": "ocr",
                    "dpi_used": None,
                }
            img = images[0]
            width, height = img.size
            self._log(f"✓ Image: {width}x{height} pixels @ {dpi} DPI")
            self._log("⏳ Preprocessing (median blur + CLAHE + Otsu)...")
            binary = self._preprocess(img, dpi)
            self._log(f"✓ Saved: temp/preprocessed_{dpi}.png")
            self._log("⏳ Running Tesseract OCR...")
            ocr_res = self._run_tesseract(binary)
            text = ocr_res["text"]
            confidence = float(ocr_res["avg_conf"])
            self._log(f"✓ Extracted {len(text)} chars, confidence: {confidence * 100:.1f}%")
            dpi_used = dpi

            if confidence < (conf_threshold / 100.0) and retry_dpi > dpi:
                self._log(f"⚠️ Low confidence ({confidence * 100:.1f}%), retrying at {retry_dpi} DPI...")
                images_retry = convert_from_path(str(pdf_path), dpi=retry_dpi, first_page=1, last_page=1)
                if images_retry:
                    img_r = images_retry[0]
                    binary_r = self._preprocess(img_r, retry_dpi)
                    ocr_res_r = self._run_tesseract(binary_r)
                    text_r = ocr_res_r["text"]
                    conf_r = float(ocr_res_r["avg_conf"])
                    if conf_r > confidence:
                        text = text_r
                        confidence = conf_r
                        dpi_used = retry_dpi
                    self._log(f"✓ Improved: {confidence * 100:.1f}% confidence at {dpi_used} DPI")

        self._log("⏳ Parsing medical fields...")
        fields = self._parse_fields(text)
        self._log(f"✓ Extracted {len(fields)} fields")

        if len(fields) >= 5 and confidence >= 0.85:
            quality = "high"
        elif len(fields) >= 3 and confidence >= 0.70:
            quality = "medium"
        else:
            quality = "low"

        dpi_display = dpi_used if dpi_used is not None else "N/A"
        self._log(
            f"Quality: {quality.upper()}, Confidence: {confidence * 100:.1f}%, "
            f"Fields: {len(fields)}/6, DPI: {dpi_display}"
        )

        return {
            "text": text,
            "fields": fields,
            "confidence": confidence,
            "quality": quality,
            "method": method,
            "dpi_used": dpi_used,
        }


def test_optimized_ocr(pdf_path: str) -> Dict[str, Any]:
    ocr = MedicalOCROptimized(verbose=True)
    result = ocr.extract_from_pdf(pdf_path)

    print("\nFields Extracted:")
    for k, v in result["fields"].items():
        print(f"  ✓ {k}: {v}")

    print(f"\nMethod: {result['method']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Rating: {int(len(result['fields']) / 6 * 100)}/100")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf = sys.argv[1]
    else:
        pdf = "CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf"
    test_optimized_ocr(pdf)
