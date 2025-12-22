"""Production-Grade OCR Pipeline for Medical Documents.

This module provides robust OCR capabilities for:
- Scanned PDFs (150-300 DPI)
- Mobile phone photos (skewed, blurred, poor lighting)
- Faded/old medical records
- Multi-column layouts

Features:
- Multi-engine OCR (Tesseract + EasyOCR fallback)
- Advanced preprocessing (deskew, denoise, contrast enhancement)
- Field extraction with confidence scores
- Graceful error handling for all document types
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

warnings.filterwarnings("ignore")

# OCR engines - import with fallbacks
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


@dataclass
class ExtractedField:
    """A single extracted field with metadata."""
    name: str
    value: Any
    confidence: float  # 0-100
    raw_text: str = ""
    extraction_method: str = "regex"


@dataclass
class OCRResult:
    """Complete OCR extraction result."""
    success: bool
    fields: Dict[str, ExtractedField] = field(default_factory=dict)
    raw_text: str = ""
    overall_confidence: float = 0.0
    document_quality: str = "unknown"  # high, medium, low
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    preprocessing_applied: List[str] = field(default_factory=list)
    ocr_engine_used: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "fields": {
                name: {
                    "value": f.value,
                    "confidence": f.confidence,
                    "raw_text": f.raw_text,
                }
                for name, f in self.fields.items()
            },
            "overall_confidence": self.overall_confidence,
            "document_quality": self.document_quality,
            "warnings": self.warnings,
            "errors": self.errors,
            "ocr_engine_used": self.ocr_engine_used,
        }

    def get_value(self, field_name: str, default: Any = None) -> Any:
        """Get field value or default."""
        if field_name in self.fields:
            return self.fields[field_name].value
        return default

    def get_confidence(self, field_name: str) -> float:
        """Get field confidence or 0."""
        if field_name in self.fields:
            return self.fields[field_name].confidence
        return 0.0


class ProductionOCR:
    """Production-grade OCR for medical documents."""

    # Field extraction patterns with multiple variations
    FIELD_PATTERNS = {
        "age": [
            r"Age[:\s]*(\d+)\s*(?:years?|yrs?)?",
            r"(\d+)\s*(?:years?|yrs?)\s*old",
            r"DOB.*?Age[:\s]*(\d+)",
        ],
        "sex": [
            r"Sex[:\s]*(Male|Female|M|F)\b",
            r"Gender[:\s]*(Male|Female|M|F)\b",
        ],
        "systolic_bp": [
            r"Systolic\s*(?:BP|Blood\s*Pressure)?[:\s]*(\d+)\s*(?:mmHg)?",
            r"BP[:\s]*(\d+)\s*/\s*\d+",
            r"Blood\s*Pressure[:\s]*(\d+)\s*/",
        ],
        "diastolic_bp": [
            r"Diastolic\s*(?:BP|Blood\s*Pressure)?[:\s]*(\d+)\s*(?:mmHg)?",
            r"BP[:\s]*\d+\s*/\s*(\d+)",
            r"Blood\s*Pressure[:\s]*\d+\s*/\s*(\d+)",
        ],
        "bmi": [
            r"BMI[:\s]*([\d.]+)\s*(?:kg/m|kg\/m)?",
            r"Body\s*Mass\s*Index[:\s]*([\d.]+)",
        ],
        "total_cholesterol": [
            r"Total\s*Cholesterol[:\s]*(\d+)\s*(?:mg/dL)?",
            r"Cholesterol\s*(?:\(Total\))?[:\s]*(\d+)",
            r"TC[:\s]*(\d+)\s*(?:mg/dL)?",
        ],
        "hdl": [
            r"HDL\s*(?:Cholesterol)?[:\s]*(\d+)\s*(?:mg/dL)?",
            r"HDL-C[:\s]*(\d+)",
        ],
        "ldl": [
            r"LDL\s*(?:Cholesterol)?[:\s]*(\d+)\s*(?:mg/dL)?",
            r"LDL-C[:\s]*(\d+)",
        ],
        "triglycerides": [
            r"Triglycerides?[:\s]*(\d+)\s*(?:mg/dL)?",
            r"TG[:\s]*(\d+)\s*(?:mg/dL)?",
        ],
        "fasting_glucose": [
            r"(?:Fasting\s*)?Glucose[:\s]*(\d+)\s*(?:mg/dL)?",
            r"FBS[:\s]*(\d+)",
            r"Blood\s*Sugar[:\s]*(\d+)",
        ],
        "hemoglobin": [
            r"H[ae]moglobin[:\s]*([\d.]+)\s*(?:g/dL)?",
            r"Hb[:\s]*([\d.]+)\s*(?:g/dL)?",
            r"HGB[:\s]*([\d.]+)",
        ],
        "wbc": [
            r"WBC\s*(?:Count)?[:\s]*(\d+)\s*(?:cells?/[µu]L)?",
            r"White\s*Blood\s*Cell[:\s]*(\d+)",
            r"Leukocytes?[:\s]*(\d+)",
        ],
        "rbc": [
            r"RBC\s*(?:Count)?[:\s]*([\d.]+)\s*(?:mill?/[µu]L)?",
            r"Red\s*Blood\s*Cell[:\s]*([\d.]+)",
            r"Erythrocytes?[:\s]*([\d.]+)",
        ],
        "platelet": [
            r"Platelet\s*(?:Count)?[:\s]*(\d+)\s*(?:cells?/[µu]L)?",
            r"PLT[:\s]*(\d+)",
            r"Thrombocytes?[:\s]*(\d+)",
        ],
        "smoking": [
            r"Smoking[:\s]*(Yes|No|Y|N)\b",
            r"Smoker[:\s]*(Yes|No|Y|N)\b",
            r"Tobacco[:\s]*(Yes|No|Y|N)\b",
        ],
        "diabetes": [
            r"Diabetes[:\s]*(Yes|No|Y|N)\b",
            r"Diabetic[:\s]*(Yes|No|Y|N)\b",
            r"DM[:\s]*(Yes|No|Y|N|Positive|Negative)\b",
        ],
        "heart_rate": [
            r"Heart\s*Rate[:\s]*(\d+)\s*(?:bpm)?",
            r"Pulse[:\s]*(\d+)\s*(?:bpm)?",
            r"HR[:\s]*(\d+)",
        ],
    }

    # Valid ranges for sanity checking
    VALID_RANGES = {
        "age": (1, 120),
        "systolic_bp": (70, 250),
        "diastolic_bp": (40, 150),
        "bmi": (10, 60),
        "total_cholesterol": (80, 400),
        "hdl": (15, 100),
        "ldl": (30, 300),
        "triglycerides": (30, 600),
        "fasting_glucose": (40, 400),
        "hemoglobin": (5, 20),
        "wbc": (2000, 20000),
        "rbc": (2, 8),
        "platelet": (50000, 600000),
        "heart_rate": (40, 200),
    }

    def __init__(
        self,
        use_easyocr: bool = True,
        verbose: bool = False,
        confidence_threshold: float = 60.0,
    ):
        """Initialize the production OCR pipeline.

        Args:
            use_easyocr: Whether to use EasyOCR as fallback
            verbose: Print debug information
            confidence_threshold: Minimum confidence for extracted fields
        """
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE

        # Initialize EasyOCR reader (lazy load)
        self._easyocr_reader = None

    def _log(self, msg: str) -> None:
        """Print if verbose mode."""
        if self.verbose:
            print(f"[ProductionOCR] {msg}")

    def _get_easyocr_reader(self):
        """Lazy load EasyOCR reader."""
        if self._easyocr_reader is None and EASYOCR_AVAILABLE:
            self._easyocr_reader = easyocr.Reader(["en"], gpu=False)
        return self._easyocr_reader

    # -------------------------------------------------------------------------
    # Image Preprocessing
    # -------------------------------------------------------------------------

    def _load_image(self, path: Union[str, Path]) -> Tuple[np.ndarray, List[str]]:
        """Load image from file path (supports PDF, PNG, JPG, etc.).

        Returns:
            (image_array, preprocessing_steps_applied)
        """
        path = Path(path)
        steps = []

        if path.suffix.lower() == ".pdf":
            if not PDF2IMAGE_AVAILABLE:
                raise ImportError("pdf2image required for PDF processing")
            self._log(f"Converting PDF to image: {path.name}")
            images = convert_from_path(str(path), dpi=300)
            if not images:
                raise ValueError("Failed to convert PDF to image")
            pil_img = images[0]
            steps.append("pdf_to_image")
        else:
            pil_img = Image.open(path).convert("RGB")

        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img, steps

    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct image skew.

        Returns:
            (deskewed_image, skew_angle)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=10
        )

        if lines is None:
            return image, 0.0

        # Calculate angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:  # Filter near-horizontal lines
                angles.append(angle)

        if not angles:
            return image, 0.0

        median_angle = np.median(angles)

        if abs(median_angle) < 0.5:  # Skip if nearly straight
            return image, median_angle

        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated, median_angle

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = image

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(l_channel)

        if len(image.shape) == 3:
            lab[:, :, 0] = enhanced
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising."""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary using adaptive thresholding."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Otsu's thresholding
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary

    def _preprocess_image(
        self,
        image: np.ndarray,
        quality_level: str = "auto",
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply preprocessing pipeline based on document quality.

        Args:
            image: Input BGR image
            quality_level: 'auto', 'clean', 'scanned', 'photo'

        Returns:
            (preprocessed_image, steps_applied)
        """
        steps = []

        # Auto-detect quality if needed
        if quality_level == "auto":
            quality_level = self._detect_document_quality(image)
            steps.append(f"quality_detected:{quality_level}")

        # Deskew for scanned/photo documents
        if quality_level in ("scanned", "photo"):
            image, angle = self._deskew(image)
            if abs(angle) > 0.5:
                steps.append(f"deskew:{angle:.1f}deg")

        # Contrast enhancement for all non-clean
        if quality_level != "clean":
            image = self._enhance_contrast(image)
            steps.append("clahe")

        # Denoising for scanned/photo
        if quality_level in ("scanned", "photo"):
            image = self._denoise(image)
            steps.append("denoise")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        steps.append("grayscale")

        # Binarization for OCR
        binary = self._binarize(gray)
        steps.append("binarize")

        return binary, steps

    def _detect_document_quality(self, image: np.ndarray) -> str:
        """Detect document quality: clean, scanned, or photo."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Calculate sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Calculate noise level
        noise = np.std(gray)

        # Calculate contrast
        contrast = gray.max() - gray.min()

        self._log(f"Quality metrics: sharpness={laplacian_var:.1f}, noise={noise:.1f}, contrast={contrast}")

        if laplacian_var > 500 and contrast > 200:
            return "clean"
        elif laplacian_var > 100:
            return "scanned"
        else:
            return "photo"

    # -------------------------------------------------------------------------
    # OCR Engines
    # -------------------------------------------------------------------------

    def _run_tesseract(
        self,
        image: np.ndarray,
        config: str = "--psm 6 --oem 3",
    ) -> Tuple[str, float]:
        """Run Tesseract OCR.

        Returns:
            (extracted_text, average_confidence)
        """
        if not TESSERACT_AVAILABLE:
            return "", 0.0

        # Get text
        text = pytesseract.image_to_string(image, config=config)

        # Get confidence
        data = pytesseract.image_to_data(
            image, config=config, output_type=pytesseract.Output.DICT
        )
        confidences = [
            float(c) for c in data.get("conf", [])
            if str(c).lstrip("-").isdigit() and float(c) >= 0
        ]

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return text, avg_conf

    def _run_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Run EasyOCR as fallback.

        Returns:
            (extracted_text, average_confidence)
        """
        if not self.use_easyocr:
            return "", 0.0

        reader = self._get_easyocr_reader()
        if reader is None:
            return "", 0.0

        results = reader.readtext(image)

        texts = []
        confidences = []

        for bbox, text, conf in results:
            texts.append(text)
            confidences.append(conf * 100)  # Convert to percentage

        full_text = " ".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return full_text, avg_conf

    # -------------------------------------------------------------------------
    # Field Extraction
    # -------------------------------------------------------------------------

    def _extract_numeric(
        self,
        text: str,
        patterns: List[str],
        valid_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Optional[float], float, str]:
        """Extract numeric value using multiple patterns.

        Returns:
            (value, confidence, raw_match)
        """
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    raw = match.group(1)
                    # Fix common OCR errors
                    cleaned = raw.replace(",", "").replace("O", "0").replace("l", "1")
                    value = float(cleaned)

                    # Validate range
                    if valid_range:
                        if value < valid_range[0] or value > valid_range[1]:
                            continue  # Try next pattern

                    # Confidence based on pattern specificity
                    confidence = 85.0 if len(patterns) == 1 else 75.0

                    return value, confidence, match.group(0)
                except (ValueError, AttributeError):
                    continue

        return None, 0.0, ""

    def _extract_categorical(
        self,
        text: str,
        patterns: List[str],
        mapping: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], float, str]:
        """Extract categorical value using patterns.

        Returns:
            (value, confidence, raw_match)
        """
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()

                # Apply mapping if provided
                if mapping:
                    normalized = raw.lower()
                    for key, val in mapping.items():
                        if normalized.startswith(key.lower()):
                            return val, 85.0, match.group(0)
                else:
                    return raw, 85.0, match.group(0)

        return None, 0.0, ""

    def _extract_all_fields(self, text: str) -> Dict[str, ExtractedField]:
        """Extract all medical fields from OCR text."""
        fields = {}

        # Numeric fields
        numeric_fields = [
            "age", "systolic_bp", "diastolic_bp", "bmi",
            "total_cholesterol", "hdl", "ldl", "triglycerides",
            "fasting_glucose", "hemoglobin", "wbc", "rbc",
            "platelet", "heart_rate",
        ]

        for field_name in numeric_fields:
            patterns = self.FIELD_PATTERNS.get(field_name, [])
            valid_range = self.VALID_RANGES.get(field_name)

            value, conf, raw = self._extract_numeric(text, patterns, valid_range)

            if value is not None:
                fields[field_name] = ExtractedField(
                    name=field_name,
                    value=value,
                    confidence=conf,
                    raw_text=raw,
                    extraction_method="regex",
                )

        # Sex field
        sex_mapping = {"m": 1, "male": 1, "f": 0, "female": 0}
        sex_val, sex_conf, sex_raw = self._extract_categorical(
            text, self.FIELD_PATTERNS["sex"], sex_mapping
        )
        if sex_val is not None:
            fields["sex"] = ExtractedField(
                name="sex",
                value=sex_val,
                confidence=sex_conf,
                raw_text=sex_raw,
            )

        # Yes/No fields
        yesno_mapping = {"y": 1, "yes": 1, "positive": 1, "n": 0, "no": 0, "negative": 0}

        for field_name in ["smoking", "diabetes"]:
            patterns = self.FIELD_PATTERNS.get(field_name, [])
            val, conf, raw = self._extract_categorical(text, patterns, yesno_mapping)
            if val is not None:
                fields[field_name] = ExtractedField(
                    name=field_name,
                    value=val,
                    confidence=conf,
                    raw_text=raw,
                )

        return fields

    # -------------------------------------------------------------------------
    # Main Extraction Method
    # -------------------------------------------------------------------------

    def extract(
        self,
        document_path: Union[str, Path],
        quality_hint: str = "auto",
    ) -> OCRResult:
        """Extract medical fields from a document.

        Args:
            document_path: Path to PDF, PNG, JPG, etc.
            quality_hint: 'auto', 'clean', 'scanned', 'photo'

        Returns:
            OCRResult with extracted fields and metadata
        """
        result = OCRResult(success=False)

        try:
            path = Path(document_path)
            if not path.exists():
                result.errors.append(f"File not found: {path}")
                return result

            self._log(f"Processing: {path.name}")

            # Load image
            image, load_steps = self._load_image(path)
            result.preprocessing_applied.extend(load_steps)

            # Detect quality
            doc_quality = self._detect_document_quality(image)
            result.document_quality = doc_quality
            self._log(f"Document quality: {doc_quality}")

            # Preprocess
            preprocessed, prep_steps = self._preprocess_image(image, quality_hint)
            result.preprocessing_applied.extend(prep_steps)

            # Run Tesseract (primary)
            self._log("Running Tesseract OCR...")
            text, confidence = self._run_tesseract(preprocessed)
            result.ocr_engine_used = "tesseract"

            # Fallback to EasyOCR if low confidence
            if confidence < 50 and self.use_easyocr:
                self._log("Low confidence, trying EasyOCR...")
                easy_text, easy_conf = self._run_easyocr(image)  # Use original
                if easy_conf > confidence:
                    text = easy_text
                    confidence = easy_conf
                    result.ocr_engine_used = "easyocr"
                    result.warnings.append("Fallback to EasyOCR due to low Tesseract confidence")

            result.raw_text = text
            result.overall_confidence = confidence

            # Extract fields
            fields = self._extract_all_fields(text)
            result.fields = fields

            # Calculate success
            key_fields = ["age", "systolic_bp", "total_cholesterol", "fasting_glucose"]
            extracted_key = sum(1 for f in key_fields if f in fields)
            result.success = extracted_key >= 2

            # Add warnings for missing critical fields
            for field in key_fields:
                if field not in fields:
                    result.warnings.append(f"Missing critical field: {field}")

            self._log(f"Extracted {len(fields)} fields, confidence={confidence:.1f}%")

        except Exception as e:
            result.errors.append(str(e))
            self._log(f"Error: {e}")

        return result

    def extract_from_image(
        self,
        image: np.ndarray,
        quality_hint: str = "auto",
    ) -> OCRResult:
        """Extract from numpy array directly."""
        result = OCRResult(success=False)

        try:
            # Detect quality
            doc_quality = self._detect_document_quality(image)
            result.document_quality = doc_quality

            # Preprocess
            preprocessed, prep_steps = self._preprocess_image(image, quality_hint)
            result.preprocessing_applied = prep_steps

            # Run Tesseract
            text, confidence = self._run_tesseract(preprocessed)
            result.ocr_engine_used = "tesseract"
            result.raw_text = text
            result.overall_confidence = confidence

            # Extract fields
            result.fields = self._extract_all_fields(text)

            # Calculate success
            key_fields = ["age", "systolic_bp", "total_cholesterol", "fasting_glucose"]
            extracted_key = sum(1 for f in key_fields if f in result.fields)
            result.success = extracted_key >= 2

        except Exception as e:
            result.errors.append(str(e))

        return result


# Convenience function
def extract_medical_fields(
    document_path: Union[str, Path],
    verbose: bool = False,
) -> OCRResult:
    """Quick extraction function.

    Args:
        document_path: Path to medical document
        verbose: Print debug info

    Returns:
        OCRResult with extracted fields
    """
    ocr = ProductionOCR(verbose=verbose)
    return ocr.extract(document_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result = extract_medical_fields(sys.argv[1], verbose=True)
        print("\n=== EXTRACTED FIELDS ===")
        for name, field in result.fields.items():
            print(f"  {name}: {field.value} (conf={field.confidence:.1f}%)")
        print(f"\nOverall confidence: {result.overall_confidence:.1f}%")
        print(f"Document quality: {result.document_quality}")
    else:
        print("Usage: python production_ocr.py <document_path>")
