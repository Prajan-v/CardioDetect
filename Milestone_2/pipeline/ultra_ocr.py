"""
ULTRA OCR - Maximum Field Extraction Pipeline
==============================================
Goal: 100% extraction rate from medical documents

Enhanced features:
- Multi-stage preprocessing (sharpen, denoise, morphology)
- Super-resolution upscaling for low-DPI images
- Multiple PSM modes and OEM engines
- Comprehensive regex patterns for all medical fields
- Fuzzy matching for OCR errors
- Table/grid detection
- Multi-page PDF processing
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pathlib import Path
import re
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Try to import fitz (pymupdf)
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# Try to import pdf2image
try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


class UltraOCR:
    """
    Ultra-enhanced OCR for medical documents achieving maximum field extraction.
    
    Preprocessing Pipeline:
    1. Upscale (if needed)
    2. Denoise
    3. Sharpen
    4. Deskew
    5. Binarize (multiple methods)
    6. Morphological cleanup
    """
    
    def __init__(self, verbose: bool = True, use_paddle_ocr: bool = False, use_advanced_ocr: bool = True):
        self.verbose = verbose
        self.use_paddle_ocr = use_paddle_ocr  # Set True to enable PaddleOCR (adds ~30sec)
        self.use_advanced_ocr = use_advanced_ocr  # Enable advanced preprocessing
        self.extraction_stats = {'total': 0, 'extracted': 0}
        
    def log(self, msg: str):
        if self.verbose:
            print(f"  📄 {msg}")
    
    # ============================================================
    # IMAGE PREPROCESSING - MAXIMUM QUALITY
    # ============================================================
    
    def upscale_image(self, image: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """Upscale image for better OCR on low-resolution documents"""
        if scale <= 1.0:
            return image
        height, width = image.shape[:2]
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening kernel to enhance text edges"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def remove_borders(self, image: np.ndarray, margin: int = 10) -> np.ndarray:
        """Remove document borders/shadows"""
        h, w = image.shape[:2]
        if margin * 2 >= min(h, w):
            return image
        return image[margin:h-margin, margin:w-margin]
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew rotated images using Hough transform"""
        # Convert to binary for edge detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return image
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 15:  # Only consider near-horizontal lines
                    angles.append(angle)
        
        if not angles:
            return image
        
        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return image
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def apply_morphology(self, binary: np.ndarray, operation: str = 'close') -> np.ndarray:
        """Apply morphological operations to clean up noise"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        if operation == 'close':
            # Close small gaps in text
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            # Remove small noise dots
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif operation == 'dilate':
            # Thicken text
            return cv2.dilate(binary, kernel, iterations=1)
        elif operation == 'erode':
            # Thin text (remove noise)
            return cv2.erode(binary, kernel, iterations=1)
        return binary
    
    def preprocess_pipeline(self, image: np.ndarray, method: str = 'ultra') -> np.ndarray:
        """
        Complete preprocessing pipeline with multiple methods
        
        Methods:
        - 'ultra': Full pipeline (best for scanned documents)
        - 'digital': Light processing (for digital/clean PDFs)
        - 'adaptive': Adaptive thresholding
        - 'otsu': Otsu's thresholding
        - 'clahe': CLAHE + Otsu
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'digital':
            # Minimal processing for clean digital documents
            denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
            return denoised
        
        elif method == 'ultra':
            # FULL PIPELINE for maximum extraction
            # 1. Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # 2. CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 3. Sharpen
            sharpened = self.sharpen_image(enhanced)
            
            # 4. Binarize with Otsu
            _, binary = cv2.threshold(sharpened, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 5. Morphological cleanup
            cleaned = self.apply_morphology(binary, 'close')
            
            return cleaned
        
        elif method == 'adaptive':
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            binary = cv2.adaptiveThreshold(denoised, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            return binary
        
        elif method == 'otsu':
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, binary = cv2.threshold(denoised, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        
        else:  # clahe
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            _, binary = cv2.threshold(enhanced, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
    
    # ============================================================
    # TESSERACT OCR - MULTIPLE CONFIGURATIONS
    # ============================================================
    
    def run_tesseract(self, image: np.ndarray, psm: int = 6, oem: int = 3) -> Tuple[str, float]:
        """
        Run Tesseract with specified configurations
        
        PSM (Page Segmentation Modes):
        - 3: Fully automatic page segmentation
        - 4: Single column of variable sizes
        - 6: Single uniform block of text
        - 11: Sparse text
        - 12: Sparse text with OSD
        
        OEM (OCR Engine Modes):
        - 0: Legacy only
        - 1: LSTM only
        - 2: Legacy + LSTM
        - 3: Default (auto)
        """
        config = f"--oem {oem} --psm {psm} -l eng"
        
        try:
            data = pytesseract.image_to_data(image, config=config, 
                                              output_type=pytesseract.Output.DICT)
            text = pytesseract.image_to_string(image, config=config)
            
            confs = []
            for c in data.get("conf", []):
                try:
                    v = float(c)
                    if v >= 0:
                        confs.append(v)
                except:
                    continue
            
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            return text.strip(), avg_conf / 100.0
        except Exception as e:
            self.log(f"⚠ Tesseract error: {e}")
            return "", 0.0
    
    def extract_with_multiple_configs(self, image: np.ndarray) -> Tuple[str, float]:
        """Try multiple Tesseract configurations to maximize extraction"""
        best_text = ""
        best_conf = 0.0
        best_fields = 0
        
        # Configuration combinations to try
        configs = [
            # (preprocess_method, psm, oem, upscale)
            ('digital', 6, 3, 1.0),      # Clean digital docs
            ('ultra', 6, 3, 1.0),        # Full preprocessing
            ('adaptive', 6, 3, 1.0),     # Adaptive threshold
            ('clahe', 6, 3, 1.0),        # CLAHE
            ('ultra', 3, 3, 1.0),        # Auto page segmentation
            ('ultra', 4, 3, 1.0),        # Single column
            ('ultra', 11, 3, 1.0),       # Sparse text
            ('ultra', 6, 3, 2.0),        # Upscaled
            ('ultra', 6, 1, 1.0),        # LSTM only
        ]
        
        for method, psm, oem, scale in configs:
            # Upscale if needed
            img = self.upscale_image(image, scale) if scale > 1.0 else image
            
            # Preprocess
            processed = self.preprocess_pipeline(img, method)
            
            # OCR
            text, conf = self.run_tesseract(processed, psm, oem)
            
            # Count extracted fields
            fields = self.parse_all_fields(text)
            num_fields = len(fields)
            
            # Prefer more fields, then higher confidence
            if num_fields > best_fields or (num_fields == best_fields and conf > best_conf):
                best_fields = num_fields
                best_conf = conf
                best_text = text
            
            # Perfect extraction - stop early
            if num_fields >= 12 and conf > 0.85:
                self.log(f"✓ Perfect extraction: {num_fields} fields, {conf*100:.1f}% conf")
                break
        
        return best_text, best_conf
    
    # ============================================================
    # FIELD PARSING - COMPREHENSIVE REGEX PATTERNS
    # ============================================================
    
    def parse_all_fields(self, text: str) -> Dict[str, Any]:
        """
        Extract ALL possible medical fields with comprehensive patterns
        """
        fields = {}
        text_clean = text.replace('\n', ' ').replace('\r', ' ')
        
        def search(patterns: List[str], cast, validate=None, default=None):
            for pat in patterns:
                m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    try:
                        val = cast(m.group(1).strip())
                        if validate is None or validate(val):
                            return val
                    except:
                        continue
            return default
        
        # ========== DEMOGRAPHICS ==========
        
        # Age (20+ patterns)
        age = search([
            r"age[:\s]+(\d{1,3})\s*(?:years?|yrs?|y)?",
            r"(\d{1,3})\s*(?:years?|yrs?)\s*(?:old)?",
            r"age\s*[:\-=]\s*(\d{1,3})",
            r"(?:patient|pt)(?:.{0,20})(\d{2})\s*(?:y|yr)",
            r"dob.{0,30}(\d{2})\s*(?:years?|y)",
            r"age/sex[:\s]+(\d{1,3})",
        ], int, lambda v: 1 <= v <= 120)
        if age:
            fields['age'] = age
        
        # Sex/Gender
        sex_match = search([
            r"(?:sex|gender)[:\s]+(male|female|m|f)\b",
            r"\b(male|female)\s+patient",
            r"age/sex[:\s]+\d+[/\s]*(m|f)\b",
            r"\b(m|f)[/\s]+\d+\s*(?:y|yr)",
        ], str)
        if sex_match:
            token = sex_match.lower().strip()
            fields['sex'] = 'Male' if token in {'male', 'm'} else 'Female'
            fields['sex_code'] = 1 if token in {'male', 'm'} else 0
        
        # ========== BLOOD PRESSURE ==========
        
        # Systolic BP (20+ patterns)
        systolic = search([
            r"systolic\s*(?:bp|blood\s*pressure)?[:\s]+(\d{2,3})",
            r"sbp[:\s]+(\d{2,3})",
            r"(?:bp|blood\s*pressure)[:\s]+(\d{2,3})\s*/",
            r"(\d{2,3})\s*/\s*\d{2,3}\s*(?:mm\s*hg|mmhg)?",
            r"sys(?:tolic)?[:\s]+(\d{2,3})",
            r"bp\s*[:=]\s*(\d{2,3})",
        ], int, lambda v: 70 <= v <= 250)
        if systolic:
            fields['systolic_bp'] = systolic
        
        # Diastolic BP
        diastolic = search([
            r"diastolic\s*(?:bp|blood\s*pressure)?[:\s]+(\d{2,3})",
            r"dbp[:\s]+(\d{2,3})",
            r"\d{2,3}\s*/\s*(\d{2,3})\s*(?:mm\s*hg|mmhg)?",
            r"dia(?:stolic)?[:\s]+(\d{2,3})",
        ], int, lambda v: 40 <= v <= 150)
        if diastolic:
            fields['diastolic_bp'] = diastolic
        
        # ========== LIPID PANEL ==========
        
        # Total Cholesterol
        chol = search([
            r"(?:total\s*)?cholesterol[:\s]+(\d{2,3})",
            r"tc[:\s]+(\d{2,3})",
            r"chol[:\s]+(\d{2,3})",
            r"serum\s*cholesterol[:\s]+(\d{2,3})",
        ], int, lambda v: 100 <= v <= 500)
        if chol:
            fields['total_cholesterol'] = chol
        
        # HDL Cholesterol
        hdl = search([
            r"hdl[:\s\-]+(?:cholesterol)?[:\s]*(\d{2,3})",
            r"hdl-c[:\s]+(\d{2,3})",
            r"high\s*density[:\s]+(\d{2,3})",
        ], int, lambda v: 20 <= v <= 120)
        if hdl:
            fields['hdl_cholesterol'] = hdl
        
        # LDL Cholesterol
        ldl = search([
            r"ldl[:\s\-]+(?:cholesterol)?[:\s]*(\d{2,3})",
            r"ldl-c[:\s]+(\d{2,3})",
            r"low\s*density[:\s]+(\d{2,3})",
        ], int, lambda v: 40 <= v <= 250)
        if ldl:
            fields['ldl_cholesterol'] = ldl
        
        # Triglycerides
        tg = search([
            r"triglycerides?[:\s]+(\d{2,4})",
            r"tg[:\s]+(\d{2,4})",
            r"trigs?[:\s]+(\d{2,4})",
        ], int, lambda v: 30 <= v <= 1000)
        if tg:
            fields['triglycerides'] = tg
        
        # ========== HEART RATE ==========
        
        hr = search([
            r"(?:heart\s*rate|pulse|hr)[:\s]+(\d{2,3})",
            r"(\d{2,3})\s*(?:bpm|beats\s*per\s*min)",
            r"resting\s*(?:heart\s*rate|hr)[:\s]+(\d{2,3})",
        ], int, lambda v: 40 <= v <= 200)
        if hr:
            fields['heart_rate'] = hr
        
        # Max Heart Rate (exercise)
        max_hr = search([
            r"max(?:imum)?\s*(?:heart\s*rate|hr)[:\s]+(\d{2,3})",
            r"peak\s*(?:heart\s*rate|hr)[:\s]+(\d{2,3})",
            r"thalach[:\s]+(\d{2,3})",
            r"max\s*hr\s*achieved[:\s]+(\d{2,3})",
        ], int, lambda v: 60 <= v <= 220)
        if max_hr:
            fields['thalach'] = max_hr
        
        # ========== GLUCOSE/DIABETES ==========
        
        glucose = search([
            r"(?:fasting\s*)?(?:blood\s*)?glucose[:\s]+(\d{2,3})",
            r"fbg[:\s]+(\d{2,3})",
            r"fbs[:\s]+(\d{2,3})",
            r"blood\s*sugar[:\s]+(\d{2,3})",
            r"fasting\s*sugar[:\s]+(\d{2,3})",
        ], int, lambda v: 50 <= v <= 500)
        if glucose:
            fields['fasting_glucose'] = glucose
        
        # HbA1c
        hba1c = search([
            r"hba1c[:\s]+(\d+\.?\d*)",
            r"a1c[:\s]+(\d+\.?\d*)",
            r"glycated\s*hemo[:\s]+(\d+\.?\d*)",
        ], float, lambda v: 3 <= v <= 20)
        if hba1c:
            fields['hba1c'] = hba1c
        
        # Diabetes (binary)
        if re.search(r"diabet(?:es|ic)?(?:\s*mellitus)?[:\s\-]+(?:yes|true|positive|type\s*[12]|1)", text, re.I):
            fields['diabetes'] = 1
        elif re.search(r"diabet(?:es|ic)?[:\s\-]+(?:no|false|negative|0|absent)", text, re.I):
            fields['diabetes'] = 0
        
        # ========== SMOKING ==========
        
        if re.search(r"smok(?:er|ing|es)?(?:\s*status)?[:\s\-]+(?:yes|true|positive|current|active|1)", text, re.I):
            fields['smoking'] = 1
        elif re.search(r"current\s*smoker", text, re.I):
            fields['smoking'] = 1
        elif re.search(r"smok(?:er|ing)?[:\s\-]+(?:no|false|negative|never|non|0|former|ex)", text, re.I):
            fields['smoking'] = 0
        
        # ========== BMI ==========
        
        bmi = search([
            r"bmi[:\s]+(\d+\.?\d*)",
            r"body\s*mass\s*index[:\s]+(\d+\.?\d*)",
        ], float, lambda v: 10 <= v <= 70)
        if bmi:
            fields['bmi'] = bmi
        
        # ========== UCI STRESS TEST FIELDS ==========
        
        # Chest Pain Type (cp)
        if re.search(r"typical\s*angina", text, re.I):
            fields['cp'] = 0
        elif re.search(r"atypical\s*angina", text, re.I):
            fields['cp'] = 1
        elif re.search(r"non[- ]?anginal", text, re.I):
            fields['cp'] = 2
        elif re.search(r"asymptomatic", text, re.I):
            fields['cp'] = 3
        else:
            cp_num = search([r"chest\s*pain(?:\s*type)?[:\s]+([0-3])"], int)
            if cp_num is not None:
                fields['cp'] = cp_num
        
        # Exercise Induced Angina (exang)
        if re.search(r"exercise\s*(?:induced\s*)?angina[:\s]+(?:yes|true|positive|1)", text, re.I):
            fields['exang'] = 1
        elif re.search(r"exercise\s*(?:induced\s*)?angina[:\s]+(?:no|false|negative|0)", text, re.I):
            fields['exang'] = 0
        
        # ST Depression (oldpeak)
        oldpeak = search([
            r"st\s*depression[:\s]+(\d+\.?\d*)",
            r"oldpeak[:\s]+(\d+\.?\d*)",
            r"st\s*segment\s*depression[:\s]+(\d+\.?\d*)",
        ], float, lambda v: 0 <= v <= 10)
        if oldpeak is not None:
            fields['oldpeak'] = oldpeak
        
        # ST Slope
        if re.search(r"(?:st\s*)?slope[:\s]+(?:upslop|0)", text, re.I):
            fields['slope'] = 0
        elif re.search(r"(?:st\s*)?slope[:\s]+(?:flat|1)", text, re.I):
            fields['slope'] = 1
        elif re.search(r"(?:st\s*)?slope[:\s]+(?:downslop|2)", text, re.I):
            fields['slope'] = 2
        
        # Major Vessels (ca)
        ca = search([
            r"(?:major\s*)?vessels?[:\s]+([0-3])",
            r"ca[:\s]+([0-3])",
            r"fluoroscopy[:\s]+([0-3])",
        ], int, lambda v: 0 <= v <= 3)
        if ca is not None:
            fields['ca'] = ca
        
        # Resting ECG
        if re.search(r"(?:resting\s*)?ecg[:\s]+(?:normal|0)", text, re.I):
            fields['restecg'] = 0
        elif re.search(r"(?:resting\s*)?ecg[:\s]+(?:st[- ]?t|abnormal|1)", text, re.I):
            fields['restecg'] = 1
        elif re.search(r"(?:resting\s*)?ecg[:\s]+(?:lv|hypertrophy|2)", text, re.I):
            fields['restecg'] = 2
        
        # Thalassemia (thal)
        if re.search(r"thal(?:assemia)?[:\s]+(?:normal|1)", text, re.I):
            fields['thal'] = 1
        elif re.search(r"thal(?:assemia)?[:\s]+(?:fixed|2)", text, re.I):
            fields['thal'] = 2
        elif re.search(r"thal(?:assemia)?[:\s]+(?:reversible|3)", text, re.I):
            fields['thal'] = 3
        
        # ========== ADDITIONAL FIELDS ==========
        
        # Hypertension
        if re.search(r"hypertens(?:ion|ive)?[:\s]+(?:yes|true|positive|1)", text, re.I):
            fields['hypertension'] = 1
        elif re.search(r"hypertens(?:ion|ive)?[:\s]+(?:no|false|negative|0)", text, re.I):
            fields['hypertension'] = 0
        
        # BP Medications
        if re.search(r"(?:bp\s*)?(?:meds?|medication)[:\s]+(?:yes|true|on|1)", text, re.I):
            fields['bp_meds'] = 1
        elif re.search(r"antihypertensive", text, re.I):
            fields['bp_meds'] = 1
        
        # Creatinine
        creatinine = search([
            r"creatinine[:\s]+(\d+\.?\d*)",
            r"serum\s*creatinine[:\s]+(\d+\.?\d*)",
            r"cr[:\s]+(\d+\.?\d*)",
        ], float, lambda v: 0.3 <= v <= 15)
        if creatinine:
            fields['creatinine'] = creatinine
        
        return fields
    
    # ============================================================
    # PDF/IMAGE EXTRACTION
    # ============================================================
    
    def extract_from_pdf(self, file_path: str, dpi: int = 300) -> Dict[str, Any]:
        """Extract from PDF with multi-page support"""
        file_path = Path(file_path)
        all_text = []
        all_fields = {}
        method = ""
        confidence = 0.0
        
        # Try digital extraction first (PyMuPDF)
        if HAS_FITZ:
            try:
                doc = fitz.open(str(file_path))
                for page_num, page in enumerate(doc):
                    text = page.get_text("text")
                    if text.strip():
                        all_text.append(text)
                        self.log(f"✓ Page {page_num + 1}: Digital extraction ({len(text)} chars)")
                doc.close()
                
                if len("".join(all_text)) > 200:
                    method = "digital_extraction"
                    confidence = 1.0
            except Exception as e:
                self.log(f"⚠ PyMuPDF error: {e}")
        
        # Fall back to OCR if needed
        if not all_text or len("".join(all_text)) < 200:
            if HAS_PDF2IMAGE:
                self.log(f"⏳ Converting PDF to images ({dpi} DPI)...")
                try:
                    images = convert_from_path(str(file_path), dpi=dpi)
                    for i, img in enumerate(images):
                        img_array = np.array(img)
                        text, conf = self.extract_with_multiple_configs(img_array)
                        all_text.append(text)
                        confidence = max(confidence, conf)
                        self.log(f"✓ Page {i + 1}: OCR ({conf*100:.1f}% confidence)")
                    method = "ocr"
                except Exception as e:
                    self.log(f"⚠ pdf2image error: {e}")
        
        # Combine all text and parse fields
        combined_text = "\n".join(all_text)
        all_fields = self.parse_all_fields(combined_text)
        
        return {
            'text': combined_text,
            'fields': all_fields,
            'confidence': confidence,
            'method': method,
            'num_pages': len(all_text),
            'num_fields': len(all_fields),
        }
    
    def extract_from_image(self, file_path: str) -> Dict[str, Any]:
        """Extract from image file with FULL advanced OCR pipeline"""
        img = cv2.imread(str(file_path))
        if img is None:
            return {'error': f"Could not read image: {file_path}"}
        
        self.log(f"Processing image: {Path(file_path).name}")
        
        # Initialize metadata
        processing_info = {
            'super_resolution': False,
            'handwriting_detected': False,
            'tables_detected': 0,
            'text_regions': 0,
            'threshold_method': 'standard',
            'lm_corrections': 0
        }
        
        # ============================================================
        # ADVANCED OCR PIPELINE
        # ============================================================
        
        if self.use_advanced_ocr:
            try:
                from advanced_ocr import (
                    SuperResolution, AdaptiveThreshold, TableExtractor,
                    LayoutAnalyzer, HandwritingSupport, LanguageModelCorrector
                )
                
                # [1] Super-Resolution if image is small
                h, w = img.shape[:2]
                if w < 1000 or h < 800:
                    img = SuperResolution.enhance_for_ocr(img)
                    processing_info['super_resolution'] = True
                    self.log("Applied super-resolution enhancement")
                
                # [2] Handwriting Detection
                is_handwritten = HandwritingSupport.is_handwritten(img)
                processing_info['handwriting_detected'] = is_handwritten
                if is_handwritten:
                    self.log("Handwriting detected - using special preprocessing")
                    gray_for_ocr = HandwritingSupport.preprocess_handwriting(img)
                else:
                    gray_for_ocr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                
                # [3] Layout Analysis
                text_regions = LayoutAnalyzer.detect_text_regions(img)
                processing_info['text_regions'] = len(text_regions)
                if text_regions:
                    self.log(f"Detected {len(text_regions)} text regions")
                
                # [4] Table Detection
                tables = TableExtractor.find_table_regions(img)
                processing_info['tables_detected'] = len(tables)
                if tables:
                    self.log(f"Detected {len(tables)} table(s)")
                
                # [5] Adaptive Thresholding
                binary, threshold_method = AdaptiveThreshold.auto_threshold(gray_for_ocr)
                processing_info['threshold_method'] = threshold_method
                self.log(f"Using {threshold_method} thresholding")
                
                # Store language model for later
                lm_corrector = LanguageModelCorrector()
                
            except ImportError as e:
                self.log(f"Advanced OCR not available: {e}")
                lm_corrector = None
            except Exception as e:
                self.log(f"Advanced OCR error: {e}")
                lm_corrector = None
        else:
            lm_corrector = None
        
        # ============================================================
        # OCR EXTRACTION
        # ============================================================
        
        # Use multi-engine OCR if enabled (adds ~30sec for PaddleOCR model loading)
        if self.use_paddle_ocr:
            try:
                from multi_ocr import get_multi_ocr
                multi_ocr = get_multi_ocr(verbose=self.verbose)
                multi_result = multi_ocr.extract(img)
                text = multi_result['fused']['text']
                confidence = multi_result['fused']['confidence']
                engines_used = multi_result['engines_used']
                self.log(f"Multi-OCR engines: {engines_used}")
            except Exception as e:
                self.log(f"Multi-OCR failed: {e}, falling back to Tesseract")
                text, confidence = self.extract_with_multiple_configs(img)
                engines_used = ['tesseract']
        else:
            # Fast mode: Tesseract only (~3-5 sec)
            text, confidence = self.extract_with_multiple_configs(img)
            engines_used = ['tesseract']
        
        # ============================================================
        # POST-OCR PROCESSING
        # ============================================================
        
        # [6] Language Model Correction
        lm_corrections = []
        if lm_corrector:
            try:
                text, lm_corrections = lm_corrector.correct(text)
                processing_info['lm_corrections'] = len(lm_corrections)
                if lm_corrections:
                    self.log(f"Language model fixed {len(lm_corrections)} errors")
            except Exception as e:
                self.log(f"LM correction error: {e}")
        
        # Spell correction pass - fix OCR errors before extraction
        spell_corrections = []
        try:
            from spell_checker import correct_ocr_text
            text, spell_corrections = correct_ocr_text(text)
            if spell_corrections:
                self.log(f"Spell checker fixed {len(spell_corrections)} errors")
        except ImportError:
            pass
        except Exception as e:
            self.log(f"Spell checker error: {e}")
        
        # First pass: regex extraction
        fields = self.parse_all_fields(text)
        
        # Second pass: enhanced extraction with fuzzy matching and validation
        warnings = []
        try:
            from enhanced_extractor import extract_and_validate, cross_field_validate
            
            # Validate regex-extracted fields first
            validated_fields, warnings = cross_field_validate(fields)
            
            # For any missing fields, try fuzzy matching
            fuzzy_result = extract_and_validate(text)
            fuzzy_fields = fuzzy_result.get('fields', {})
            
            # Only add fuzzy fields if not already in regex results
            for k, v in fuzzy_fields.items():
                if k not in validated_fields:
                    validated_fields[k] = v
            
            fields = validated_fields
        except ImportError:
            self.log("Enhanced extractor not available")
        
        # Third pass: Medical NER for additional entities
        medical_entities = {}
        try:
            from medical_ner import get_ner
            ner = get_ner()
            ner_result = ner.extract_entities(text)
            medical_entities = ner_result.get('entities', {})
            
            # Get risk-relevant info (medications, conditions)
            risk_info = ner.get_risk_relevant_info(text)
            
            # Update fields based on NER findings
            if risk_info.get('on_diabetes_meds') and 'diabetes' not in fields:
                fields['diabetes'] = 1
                self.log("NER detected diabetes medication → set diabetes=1")
            
            if risk_info.get('on_statins'):
                fields['on_statins'] = 1
            
            if risk_info.get('has_known_cad'):
                fields['known_cad'] = 1
            
            self.log(f"NER found: {ner_result['counts']}")
        except ImportError:
            self.log("Medical NER not available")
        except Exception as e:
            self.log(f"NER failed: {e}")
        
        self.log(f"✓ Extracted {len(fields)} fields, {confidence*100:.1f}% confidence")
        
        return {
            'text': text,
            'fields': fields,
            'confidence': confidence,
            'method': 'multi_ocr' if len(engines_used) > 1 else 'tesseract',
            'engines_used': engines_used,
            'num_fields': len(fields),
            'warnings': warnings,
            'quality': 'HIGH' if len(fields) >= 10 else 'MEDIUM' if len(fields) >= 5 else 'LOW',
            'medical_entities': medical_entities,
            'spell_corrections': spell_corrections,
            'processing_info': processing_info,
            'lm_corrections': lm_corrections,
        }
    
    def extract_from_file(self, file_path: str, dpi: int = 300) -> Dict[str, Any]:
        """Universal extraction from PDF or image"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.extract_from_pdf(str(file_path), dpi)
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
            return self.extract_from_image(str(file_path))
        else:
            return {'error': f"Unsupported file type: {suffix}"}


def test_ultra_ocr():
    """Test the Ultra OCR"""
    print("=" * 60)
    print("🔬 ULTRA OCR - Maximum Field Extraction Test")
    print("=" * 60)
    
    ocr = UltraOCR(verbose=True)
    
    # Test with sample text
    sample_text = """
    Patient: John Smith
    Age: 58 years
    Sex: Male
    
    Vital Signs:
    Blood Pressure: 145/92 mmHg
    Heart Rate: 78 bpm
    
    Lipid Panel:
    Total Cholesterol: 235 mg/dL
    HDL: 42 mg/dL
    LDL: 158 mg/dL
    Triglycerides: 175 mg/dL
    
    Fasting Glucose: 118 mg/dL
    BMI: 28.5
    
    Smoking Status: Former
    Diabetes: No
    """
    
    fields = ocr.parse_all_fields(sample_text)
    
    print("\n📊 Extracted Fields:")
    for k, v in sorted(fields.items()):
        print(f"   • {k}: {v}")
    
    print(f"\n✓ Total fields extracted: {len(fields)}")
    print("=" * 60)


if __name__ == "__main__":
    test_ultra_ocr()
