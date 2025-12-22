"""
Universal Medical OCR Engine V3
Handles JPG/PNG/PDF (digital + scanned) with 95%+ field extraction target.
Uses Tesseract + PyTesseract + OpenCV only (no cloud APIs).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Optional imports with fallbacks
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


@dataclass
class OCRResult:
    """Complete OCR extraction result."""
    success: bool
    engine: str = "tesseract"
    file_type: str = "unknown"
    text: str = ""
    structured_fields: Dict[str, Any] = field(default_factory=dict)
    field_confidences: Dict[str, float] = field(default_factory=dict)
    avg_ocr_confidence: float = 0.0
    pages: int = 1
    processing_time_sec: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class UniversalMedicalOCREngine:
    """
    Universal OCR engine for medical documents.
    Handles all formats: JPG, PNG, PDF (digital + scanned).
    """
    
    # Comprehensive field patterns with multiple variants
    FIELD_PATTERNS = {
        "hemoglobin": [
            r"(?:Hemoglobin|Haemoglobin|HB|Hb|HGB|HG)[:\s=]*(\d+\.?\d*)",
            r"(?:Hb|HB)\s*\([^)]*\)[:\s=]*(\d+\.?\d*)",
            r"(?:Hemoglobin|Haemoglobin)[:\s=]*(\d+\.?\d*)\s*(?:g|gm|g/dL|gm/dL|g%)",
        ],
        "rbc": [
            r"(?:RBC|Red\s+(?:Blood\s+)?Cells?|R\.B\.C\.?|Total\s+R\.B\.C\.)[:\s=]*(\d+\.?\d*)",
            r"RBC\s+Count[:\s=]*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*(?:million|mill|×10⁶|×106|million/cumm|mill/cumm)",
        ],
        "wbc": [
            r"(?:WBC|White\s+(?:Blood\s+)?Cells?|W\.B\.C\.?|Total\s+(?:W\.B\.C\.|Leukocytes?))[:\s=]*(\d+\.?\d*)",
            r"(?:Total\s+)?WBC\s+Count[:\s=]*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*(?:cells/cumm|cells/μL|thousand/cumm|×10³)",
        ],
        "platelets": [
            r"(?:Platelet|PLT|PLAT|Thrombocyte)[s]?\s*(?:Count)?[:\s=]*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*(?:×10³|×10⁹/L|thousand/cumm|cells/cumm)",
        ],
        "systolic_bp": [
            r"(?:Systolic|SBP|BP)[:\s/=]*(\d{2,3})",
            r"(\d{2,3})\s*/\s*\d{2,3}\s*(?:mmHg|mm\s*Hg)?",
        ],
        "diastolic_bp": [
            r"(?:Diastolic|DBP)[:\s/=]*(\d{2,3})",
            r"\d{2,3}\s*/\s*(\d{2,3})\s*(?:mmHg|mm\s*Hg)?",
        ],
        "cholesterol": [
            r"(?:Total\s+)?Cholesterol[:\s=]*(\d+\.?\d*)",
            r"TC[:\s=]*(\d+\.?\d*)",
            r"(?:Chol|Cholesterol)[:\s=]*(\d+\.?\d*)\s*(?:mg/dL|mg/dl)",
        ],
        "glucose": [
            r"(?:(?:Random|Fasting|Blood)?\s*)?Glucose[:\s=]*(\d+\.?\d*)",
            r"(?:RBS|FBS)[:\s=]*(\d+\.?\d*)",
            r"(?:Glucose|Glu)[:\s=]*(\d+\.?\d*)\s*(?:mg/dL|mg/dl)",
        ],
        "age": [
            r"Age[:\s=]*(\d{1,3})\s*(?:years?|yrs?|Y)?",
            r"(\d{1,3})\s*(?:years?|yrs?)\s*old",
        ],
        "sex": [
            r"Sex[:\s=]*(Male|Female|M|F)\b",
            r"Gender[:\s=]*(Male|Female|M|F)\b",
        ],
        "smoking": [
            r"Smoking[:\s]*?(Yes|No|Current|Never|Former|Ex-?smoker|Smoker)",
            r"Smoker[:\s]*?(Yes|No)",
        ],
        "diabetes": [
            r"Diabetes[:\s]*?(Yes|No|Type\s*1|Type\s*2|T1DM|T2DM)",
            r"Known\s+Diabetic[:\s]*?(Yes|No)",
        ],
    }
    
    # Biological plausibility ranges
    VALID_RANGES = {
        "hemoglobin": (5.0, 20.0),
        "rbc": (2.0, 8.0),
        "wbc": (2.0, 20.0),
        "platelets": (50.0, 600.0),
        "systolic_bp": (70, 250),
        "diastolic_bp": (40, 150),
        "cholesterol": (80, 500),
        "glucose": (40, 600),
        "age": (1, 120),
    }
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[UniversalOCR] {msg}")
    
    # -------------------------------------------------------------------------
    # File Type Detection
    # -------------------------------------------------------------------------
    
    def detect_file_type(self, path: Path) -> str:
        """Detect file type: jpg/png/pdf-digital/pdf-scanned."""
        suffix = path.suffix.lower()
        
        if suffix in ['.jpg', '.jpeg']:
            return 'jpg'
        elif suffix == '.png':
            return 'png'
        elif suffix == '.pdf':
            # Try to detect if PDF has selectable text (digital) or is scanned
            if PDFPLUMBER_AVAILABLE:
                try:
                    with pdfplumber.open(path) as pdf:
                        if len(pdf.pages) > 0:
                            text = pdf.pages[0].extract_text()
                            if text and len(text.strip()) > 100:
                                return 'pdf-digital'
                except Exception:
                    pass
            return 'pdf-scanned'
        else:
            return 'unknown'
    
    # -------------------------------------------------------------------------
    # Image Loading
    # -------------------------------------------------------------------------
    
    def load_images_from_path(self, path: Path) -> List[Image.Image]:
        """Load images from file path (handles PDF conversion)."""
        suffix = path.suffix.lower()
        
        if suffix in ['.jpg', '.jpeg', '.png']:
            return [Image.open(path).convert('RGB')]
        
        elif suffix == '.pdf':
            if not PDF2IMAGE_AVAILABLE:
                raise ImportError("pdf2image required for PDF processing. Install: pip install pdf2image")
            
            try:
                images = convert_from_path(str(path), dpi=300)
                return images
            except Exception as e:
                raise RuntimeError(f"Failed to convert PDF to images: {e}")
        
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    # -------------------------------------------------------------------------
    # OpenCV Preprocessing Pipeline
    # -------------------------------------------------------------------------
    
    def preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Advanced preprocessing pipeline:
        1. Grayscale conversion
        2. Denoising
        3. Deskew/rotate
        4. Upscaling (if needed)
        5. CLAHE contrast enhancement
        6. Morphological operations
        7. Adaptive thresholding
        """
        # Convert PIL to OpenCV
        img = np.array(pil_image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 1. Denoise
        img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. Deskew
        img = self._deskew_image(img)
        
        # 3. Upscale if image is small
        h, w = img.shape
        if h < 1000 or w < 1000:
            scale = max(1000 / h, 1000 / w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 4. CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # 5. Morphological closing to connect broken text
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 6. Adaptive thresholding
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return img
    
    def simple_preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
        img = np.array(pil_image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape
        if h < 1000 or w < 1000:
            scale = max(1000 / h, 1000 / w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return img
    
    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Auto-correct document angle using minAreaRect."""
        # Find coordinates of non-zero pixels
        coords = np.column_stack(np.where(img > 0))
        
        if len(coords) == 0:
            return img
        
        # Fit minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Normalize angle
        if angle < -45:
            angle = 90 + angle
        
        # Only rotate if significant
        if abs(angle) > 3:
            h, w = img.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
        
        return img
    
    # -------------------------------------------------------------------------
    # Tesseract OCR
    # -------------------------------------------------------------------------
    
    def run_tesseract(self, image: np.ndarray, psm: int = 4) -> Tuple[str, float]:
        """
        Run Tesseract with optimized config.
        PSM 4: Single column of text (best for most medical reports).
        """
        config = f"--oem 3 --psm {psm} -l eng"
        
        try:
            # Get text
            text = pytesseract.image_to_string(image, config=config)
            
            # Get confidence
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            confidences = [
                float(c) for c in data.get('conf', [])
                if str(c).lstrip('-').replace('.', '').isdigit() and float(c) >= 0
            ]
            
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            avg_conf = avg_conf / 100.0  # Normalize to 0-1
            
            return text, avg_conf
        
        except Exception as e:
            self._log(f"Tesseract error: {e}")
            return "", 0.0
    
    # -------------------------------------------------------------------------
    # Field Extraction
    # -------------------------------------------------------------------------
    
    def extract_fields(self, text: str) -> Dict[str, Any]:
        """Extract structured fields using comprehensive regex patterns."""
        fields = {}
        
        for field_name, patterns in self.FIELD_PATTERNS.items():
            if not isinstance(patterns, list):
                patterns = [patterns]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    raw_value = match.group(1)
                    
                    # Categorical fields: sex, smoking, diabetes
                    if field_name == "sex":
                        sex_val = raw_value.strip().upper()
                        if sex_val in ['M', 'MALE']:
                            fields[field_name] = 1  # Male
                            break
                        elif sex_val in ['F', 'FEMALE']:
                            fields[field_name] = 0  # Female
                            break
                    elif field_name == "smoking":
                        token = raw_value.strip().lower()
                        if token in [
                            'yes', 'y', 'current', 'smoker',
                            'active smoker', 'active-smoker'
                        ]:
                            fields[field_name] = 1.0
                            break
                        elif token in [
                            'no', 'n', 'never', 'non-smoker', 'nonsmoker', 'non smoker',
                            'former', 'ex-smoker', 'ex smoker'
                        ]:
                            fields[field_name] = 0.0
                            break
                    elif field_name == "diabetes":
                        token = raw_value.strip().lower()
                        if token in [
                            'yes', 'y', 'type 1', 'type1', 'type 2', 'type2',
                            't1dm', 't2dm', 'dm', 'diabetic'
                        ]:
                            fields[field_name] = 1.0
                            break
                        elif token in [
                            'no', 'n', 'non-diabetic', 'nondiabetic', 'non diabetic'
                        ]:
                            fields[field_name] = 0.0
                            break
                    else:
                        # Numeric field
                        cleaned = self._clean_numeric(raw_value)
                        try:
                            value = float(cleaned)
                            if self._validate_field(field_name, value):
                                fields[field_name] = value
                                break
                        except ValueError:
                            continue
        
        return fields
    
    def _clean_numeric(self, text: str) -> str:
        """Fix common OCR mistakes in numbers."""
        # O (letter O) → 0 (digit zero)
        text = text.replace("O", "0").replace("o", "0")
        # l (lowercase L) or I (uppercase i) → 1
        text = text.replace("l", "1").replace("I", "1")
        # S → 5 (when isolated)
        text = re.sub(r"(?<!\w)S(?!\w)", "5", text)
        # B → 8 (when before digit)
        text = re.sub(r"(?<!\w)B(?=\d)", "8", text)
        # Multiple dots/commas → single decimal point
        text = re.sub(r"[.,]+", ".", text)
        # Remove spaces
        text = text.replace(" ", "")
        
        return text
    
    def _validate_field(self, field_name: str, value: float) -> bool:
        """Check if value is within biological plausibility range."""
        if field_name not in self.VALID_RANGES:
            return True
        
        min_val, max_val = self.VALID_RANGES[field_name]
        return min_val <= value <= max_val
    
    # -------------------------------------------------------------------------
    # Confidence Computation
    # -------------------------------------------------------------------------
    
    def compute_field_confidences(
        self, text: str, fields: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute per-field confidence based on pattern match quality."""
        confidences = {}
        
        for field_name, value in fields.items():
            # Base confidence from pattern match
            conf = 0.8  # Default for successful extraction
            
            # Boost if value is in middle of valid range
            if field_name in self.VALID_RANGES:
                min_val, max_val = self.VALID_RANGES[field_name]
                mid = (min_val + max_val) / 2
                range_size = max_val - min_val
                
                if isinstance(value, (int, float)):
                    distance_from_mid = abs(value - mid) / range_size
                    if distance_from_mid < 0.3:
                        conf = 0.9
                    elif distance_from_mid > 0.7:
                        conf = 0.7
            
            confidences[field_name] = conf
        
        return confidences
    
    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------
    
    def extract(self, path: str | Path) -> OCRResult:
        """
        Main entry point: extract structured fields from medical document.
        
        Args:
            path: Path to JPG/PNG/PDF file
            
        Returns:
            OCRResult with all extracted data and metadata
        """
        start_time = time.time()
        path = Path(path)
        
        result = OCRResult(success=False)
        
        if not path.exists():
            result.errors.append(f"File not found: {path}")
            return result
        
        try:
            # 1. Detect file type
            file_type = self.detect_file_type(path)
            result.file_type = file_type
            self._log(f"Detected file type: {file_type}")
            
            # 2. Handle digital PDFs with direct text extraction
            if file_type == 'pdf-digital' and PDFPLUMBER_AVAILABLE:
                try:
                    with pdfplumber.open(path) as pdf:
                        text_parts = []
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)
                        
                        if text_parts:
                            result.text = "\n".join(text_parts)
                            result.pages = len(pdf.pages)
                            result.avg_ocr_confidence = 1.0  # Digital text is perfect
                            self._log("Extracted text from digital PDF")
                except Exception as e:
                    self._log(f"Digital PDF extraction failed: {e}, falling back to OCR")
                    file_type = 'pdf-scanned'
            
            # 3. OCR path for images and scanned PDFs
            if not result.text:
                images = self.load_images_from_path(path)
                result.pages = len(images)
                self._log(f"Loaded {len(images)} page(s)")
                
                all_text = []
                all_confidences = []
                
                for idx, img in enumerate(images):
                    self._log(f"Processing page {idx + 1}/{len(images)}")
                    
                    # Preprocess
                    preprocessed = self.preprocess_image(img)
                    
                    # Run Tesseract
                    page_text, page_conf = self.run_tesseract(preprocessed, psm=4)
                    
                    if (not page_text.strip()) or page_conf < 0.05:
                        self._log("Low or empty OCR output, retrying with simple preprocessing and PSM 6")
                        simple = self.simple_preprocess_image(img)
                        alt_text, alt_conf = self.run_tesseract(simple, psm=6)
                        if alt_text.strip() and (alt_conf > page_conf or not page_text.strip()):
                            page_text, page_conf = alt_text, alt_conf
                        if not page_text.strip():
                            self._log("Still empty OCR text, running Tesseract on raw image")
                            raw = np.array(img.convert("RGB"))
                            try:
                                raw_text = pytesseract.image_to_string(raw, config="--oem 3 --psm 6 -l eng")
                                if raw_text.strip():
                                    page_text = raw_text
                                    data = pytesseract.image_to_data(
                                        raw,
                                        config="--oem 3 --psm 6 -l eng",
                                        output_type=pytesseract.Output.DICT,
                                    )
                                    confs = [
                                        float(c)
                                        for c in data.get("conf", [])
                                        if str(c).lstrip("-").replace(".", "").isdigit() and float(c) >= 0
                                    ]
                                    if confs:
                                        page_conf = (sum(confs) / len(confs)) / 100.0
                            except Exception as e:
                                self._log(f"Fallback Tesseract error: {e}")
                    
                    all_text.append(page_text)
                    all_confidences.append(page_conf)
                
                result.text = "\n".join(all_text)
                result.avg_ocr_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            # 4. Extract structured fields
            result.structured_fields = self.extract_fields(result.text)
            result.field_confidences = self.compute_field_confidences(result.text, result.structured_fields)
            
            # 5. Success if we got at least some fields
            result.success = len(result.structured_fields) > 0
            
            if not result.success:
                result.warnings.append("No structured fields extracted")
            
        except Exception as e:
            result.errors.append(f"Extraction failed: {str(e)}")
            self._log(f"Error: {e}")
        
        result.processing_time_sec = time.time() - start_time
        return result


# Convenience function
def extract_medical_fields(document_path: str | Path, verbose: bool = False) -> OCRResult:
    """Quick extraction function."""
    engine = UniversalMedicalOCREngine(verbose=verbose)
    return engine.extract(document_path)
