"""
INTEGRATED DUAL-MODEL PIPELINE
==============================
This is the CORE ML pipeline that powers CardioDetect.

┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE OVERVIEW                            │
├─────────────────────────────────────────────────────────────────┤
│  1. OCR: Extract text from medical documents (PDF/Images)       │
│  2. Field Extraction: Parse medical values using regex          │
│  3. Detection Model: Does patient currently have heart disease? │
│     - Uses: Voting Ensemble (XGBoost + RandomForest + LightGBM) │
│     - Accuracy: 91.45%                                          │
│  4. Prediction Model: 10-year cardiovascular risk               │
│     - Uses: XGBoost                                              │
│     - Accuracy: 91.63%                                          │
│  5. Clinical Advisor: ACC/AHA guideline recommendations         │
└─────────────────────────────────────────────────────────────────┘

Key Components:
- EnhancedMedicalOCR: Handles document text extraction
- DualModelPipeline: Orchestrates the entire prediction flow
- ClinicalAdvisor: Generates clinical recommendations

Technologies Used:
- Tesseract OCR (via pytesseract)
- OpenCV for image preprocessing
- PyMuPDF for PDF text extraction
- scikit-learn, XGBoost, LightGBM for ML models
- SHAP for model explainability
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pathlib import Path
import re
from typing import Dict, Any, Optional, Tuple
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import Clinical Advisor
try:
    from clinical_advisor import ClinicalAdvisor, format_recommendations_text
    HAS_CLINICAL_ADVISOR = True
except ImportError:
    HAS_CLINICAL_ADVISOR = False

# Import PDF Report Generator
try:
    from report_generator import generate_clinical_report
    HAS_PDF_GENERATOR = True
except ImportError:
    HAS_PDF_GENERATOR = False

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

# Try to import Ensemble OCR (multi-engine for higher accuracy)
try:
    from .ensemble_ocr import EnsembleOCR
    HAS_ENSEMBLE = True
except ImportError:
    try:
        from ensemble_ocr import EnsembleOCR
        HAS_ENSEMBLE = True
    except ImportError:
        HAS_ENSEMBLE = False

# ============================================================================
# CLASS: EnhancedMedicalOCR
# ============================================================================
# This class handles all document processing and text extraction.
# It uses Tesseract OCR with multiple preprocessing techniques to
# maximize extraction accuracy from medical documents.
# ============================================================================
class EnhancedMedicalOCR:
    """
    Enhanced OCR for medical documents with improved preprocessing.
    
    OCR FLOW:
    1. Check file type (PDF or Image)
    2. For PDFs: Try direct text extraction first (PyMuPDF)
    3. If scanned: Convert to image and run Tesseract
    4. Apply preprocessing if needed (CLAHE, Otsu, Adaptive)
    5. Parse extracted text for medical values using regex
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        DESKEW: Fix rotated/tilted scanned documents.
        
        Why needed: Scanned documents are often slightly rotated,
        which significantly reduces OCR accuracy.
        
        How it works:
        1. Find all non-zero pixels (text)
        2. Calculate minimum bounding rectangle
        3. Get rotation angle and rotate image to straighten
        """
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 100:
            return image  # Not enough content to deskew
        
        # Calculate rotation angle from bounding rectangle
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Skip if already straight (within 0.5 degrees)
        if abs(angle) < 0.5:
            return image
        
        # Rotate image to correct the skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def preprocess_image(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """
        Enhanced preprocessing with multiple methods
        
        Methods:
        - 'adaptive': Adaptive thresholding (good for varying lighting)
        - 'otsu': Otsu's thresholding (good for bimodal images)
        - 'clahe': CLAHE + Otsu (good for low contrast)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        if method == 'adaptive':
            # Adaptive thresholding - good for varying lighting
            binary = cv2.adaptiveThreshold(denoised, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
        elif method == 'otsu':
            # Otsu's thresholding
            _, binary = cv2.threshold(denoised, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # clahe
            # CLAHE + Otsu
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            _, binary = cv2.threshold(enhanced, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Deskew
        binary = self.deskew(binary)
        
        return binary
    
    def run_tesseract(self, image: np.ndarray, psm: int = 6) -> Tuple[str, float]:
        """
        Run Tesseract with specified Page Segmentation Mode
        
        PSM modes:
        - 3: Fully automatic page segmentation
        - 4: Assume single column of variable sizes
        - 6: Assume uniform block of text (default)
        - 11: Sparse text with minimal structure
        """
        config = f"--oem 3 --psm {psm}"
        
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
            return text, avg_conf / 100.0
        except Exception as e:
            self.log(f"⚠ Tesseract error: {e}")
            return "", 0.0
    
    def parse_cardiovascular_fields(self, text: str) -> Dict[str, Any]:
        """Extract cardiovascular-specific fields from text"""
        fields = {}
        text_lower = text.lower()
        
        def search(patterns, cast, validate=None):
            for pat in patterns:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m:
                    try:
                        val = cast(m.group(1))
                        if validate is None or validate(val):
                            return val
                    except:
                        continue
            return None
        
        # Age - enhanced patterns
        age = search([
            r"age[:\s]+(\d+)\s*(?:years?|yrs?)?",
            r"(\d+)\s*(?:years?|yrs?)\s*(?:old)?",
            r"(?:patient|pt)[:\s]+.*?(\d{2,3})\s*(?:y|yr|year)",
            r"age\s*[:\-]?\s*(\d+)",
        ], int, lambda v: 18 < v < 120)
        if age:
            fields['age'] = age
        
        # Sex
        sex = search([
            r"(?:sex|gender)[:\s]+(male|female|m|f)",
            r"\b(male|female)\b",
        ], str)
        if sex:
            token = sex.strip().lower()
            fields['sex'] = 'Male' if token in {'male', 'm'} else 'Female'
            fields['sex_code'] = 1 if token in {'male', 'm'} else 0
        
        # Systolic Blood Pressure - enhanced patterns
        systolic = search([
            r"systolic\s*(?:bp|blood\s*pressure)?[:\s]+(\d{2,3})",
            r"systolic\s+bp\s*[:\s]+(\d{2,3})",
            r"sbp[:\s]+(\d{2,3})",
            r"(?:bp|blood\s*pressure)[:\s]+(\d{2,3})\s*/",
        ], int, lambda v: 70 <= v <= 250)
        if systolic:
            fields['systolic_bp'] = systolic
        
        # Diastolic Blood Pressure - enhanced patterns
        diastolic = search([
            r"diastolic\s*(?:bp|blood\s*pressure)?[:\s]+(\d{2,3})",
            r"diastolic\s+bp\s*[:\s]+(\d{2,3})",
            r"dbp[:\s]+(\d{2,3})",
            r"/\s*(\d{2,3})\s*(?:mmhg|mm\s*hg)?",
        ], int, lambda v: 40 <= v <= 150)
        if diastolic:
            fields['diastolic_bp'] = diastolic
        
        # BP as pair (fallback)
        if 'systolic_bp' not in fields or 'diastolic_bp' not in fields:
            bp_match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg|mm\s*hg)?", text, re.IGNORECASE)
            if bp_match:
                try:
                    sys_bp = int(bp_match.group(1))
                    dia_bp = int(bp_match.group(2))
                    if 70 <= sys_bp <= 250 and 40 <= dia_bp <= 150:
                        if 'systolic_bp' not in fields:
                            fields['systolic_bp'] = sys_bp
                        if 'diastolic_bp' not in fields:
                            fields['diastolic_bp'] = dia_bp
                except:
                    pass
        
        # Cholesterol
        chol = search([
            r"(?:total\s*)?cholesterol[:\s]+(\d{2,3})",
            r"chol[:\s]+(\d{2,3})",
            r"tc[:\s]+(\d{2,3})",
        ], int, lambda v: 100 <= v <= 400)
        if chol:
            fields['total_cholesterol'] = chol
        
        # Heart Rate
        hr = search([
            r"(?:heart\s*rate|pulse|hr)[:\s]+(\d{2,3})",
            r"(\d{2,3})\s*(?:bpm|beats)",
        ], int, lambda v: 40 <= v <= 200)
        if hr:
            fields['heart_rate'] = hr
        
        # Glucose
        glucose = search([
            r"(?:fasting\s*)?(?:blood\s*)?glucose[:\s]+(\d{2,3})",
            r"fbg[:\s]+(\d{2,3})",
            r"fbs[:\s]+(\d{2,3})",
        ], int, lambda v: 50 <= v <= 400)
        if glucose:
            fields['fasting_glucose'] = glucose
        
        # BMI
        bmi = search([
            r"bmi[:\s]+(\d+\.?\d*)",
            r"body\s*mass\s*index[:\s]+(\d+\.?\d*)",
        ], float, lambda v: 15 <= v <= 60)
        if bmi:
            fields['bmi'] = bmi
        
        # Smoking - enhanced patterns
        if re.search(r"smok(?:er|ing|es)?(?:\s*status)?[:\s\-]+(?:yes|true|positive|current|1)", text, re.IGNORECASE):
            fields['smoking'] = 1
        elif re.search(r"current\s*smoker", text, re.IGNORECASE):
            fields['smoking'] = 1
        elif re.search(r"smok(?:er|ing|es)?(?:\s*status)?[:\s\-]+(?:no|false|negative|never|non|0)", text, re.IGNORECASE):
            fields['smoking'] = 0
        
        # Diabetes - enhanced patterns
        if re.search(r"diabet(?:es|ic)?(?:\s*mellitus)?[:\s\-]+(?:yes|true|positive|type|1)", text, re.IGNORECASE):
            fields['diabetes'] = 1
        elif re.search(r"diabet(?:es|ic)?(?:\s*mellitus)?[:\s\-]+(?:no|false|negative|0)", text, re.IGNORECASE):
            fields['diabetes'] = 0
        
        # Hypertension
        if re.search(r"hypertens(?:ion|ive)?[:\s]+(?:yes|true|positive)", text, re.IGNORECASE):
            fields['hypertension'] = 1
        elif re.search(r"hypertens(?:ion|ive)?[:\s]+(?:no|false|negative)", text, re.IGNORECASE):
            fields['hypertension'] = 0
        
        # ============================================================
        # UCI STRESS TEST FIELDS (Required for Detection Model)
        # ============================================================
        
        # Chest Pain Type (cp): 0-3
        # 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic
        cp_match = re.search(r"chest\s*pain(?:\s*type)?[:\s]+(\d)", text, re.IGNORECASE)
        if cp_match:
            fields['cp'] = int(cp_match.group(1))
        elif re.search(r"typical\s*angina", text, re.IGNORECASE):
            fields['cp'] = 0
        elif re.search(r"atypical\s*angina", text, re.IGNORECASE):
            fields['cp'] = 1
        elif re.search(r"non[- ]?anginal", text, re.IGNORECASE):
            fields['cp'] = 2
        elif re.search(r"asymptomatic", text, re.IGNORECASE):
            fields['cp'] = 3
        
        # Maximum Heart Rate Achieved (thalach) - during exercise
        thalach = search([
            r"max(?:imum)?\s*(?:heart\s*rate|hr)[:\s]+(\d{2,3})",
            r"peak\s*(?:heart\s*rate|hr)[:\s]+(\d{2,3})",
            r"thalach[:\s]+(\d{2,3})",
            r"exercise\s*(?:heart\s*rate|hr)[:\s]+(\d{2,3})",
            r"max(?:imum)?\s*hr\s*achieved[:\s]+(\d{2,3})",
        ], int, lambda v: 60 <= v <= 220)
        if thalach:
            fields['thalach'] = thalach
        
        # Exercise Induced Angina (exang): 0 or 1
        if re.search(r"exercise\s*(?:induced\s*)?angina[:\s]+(?:yes|true|positive|1)", text, re.IGNORECASE):
            fields['exang'] = 1
        elif re.search(r"exercise\s*(?:induced\s*)?angina[:\s]+(?:no|false|negative|0)", text, re.IGNORECASE):
            fields['exang'] = 0
        
        # ST Depression (oldpeak) - induced by exercise relative to rest
        oldpeak = search([
            r"st\s*depression[:\s]+(\d+\.?\d*)",
            r"oldpeak[:\s]+(\d+\.?\d*)",
            r"st\s*segment\s*depression[:\s]+(\d+\.?\d*)",
        ], float, lambda v: 0 <= v <= 10)
        if oldpeak is not None:
            fields['oldpeak'] = oldpeak
        
        # ST Slope (slope): 0 = upsloping, 1 = flat, 2 = downsloping
        if re.search(r"(?:st\s*)?slope[:\s]+(?:upslop|0)", text, re.IGNORECASE):
            fields['slope'] = 0
        elif re.search(r"(?:st\s*)?slope[:\s]+(?:flat|1)", text, re.IGNORECASE):
            fields['slope'] = 1
        elif re.search(r"(?:st\s*)?slope[:\s]+(?:downslop|2)", text, re.IGNORECASE):
            fields['slope'] = 2
        
        # Number of Major Vessels (ca): 0-3
        ca = search([
            r"(?:major\s*)?vessels?[:\s]+(\d)",
            r"ca[:\s]+(\d)",
            r"fluoroscopy[:\s]+(\d)",
        ], int, lambda v: 0 <= v <= 3)
        if ca is not None:
            fields['ca'] = ca
        
        # Resting ECG (restecg): 0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy
        if re.search(r"(?:resting\s*)?ecg[:\s]+(?:normal|0)", text, re.IGNORECASE):
            fields['restecg'] = 0
        elif re.search(r"(?:resting\s*)?ecg[:\s]+(?:st[- ]?t|abnormal|1)", text, re.IGNORECASE):
            fields['restecg'] = 1
        elif re.search(r"(?:resting\s*)?ecg[:\s]+(?:lv|hypertrophy|2)", text, re.IGNORECASE):
            fields['restecg'] = 2
        
        # Thalassemia (thal): 1 = normal, 2 = fixed defect, 3 = reversible defect
        if re.search(r"thal(?:assemia)?[:\s]+(?:normal|1)", text, re.IGNORECASE):
            fields['thal'] = 1
        elif re.search(r"thal(?:assemia)?[:\s]+(?:fixed|2)", text, re.IGNORECASE):
            fields['thal'] = 2
        elif re.search(r"thal(?:assemia)?[:\s]+(?:reversible|3)", text, re.IGNORECASE):
            fields['thal'] = 3
        
        return fields
    
    def extract_from_file(self, file_path: str, dpi: int = 300) -> Dict[str, Any]:
        """
        MAIN OCR ENTRY POINT
        ====================
        Extract text and medical fields from PDF or image files.
        
        FLOW:
        1. Detect file type (PDF vs Image)
        2. For PDF: Try PyMuPDF direct extraction first (faster, 100% accurate)
        3. If scanned PDF or Image: Use Tesseract OCR
        4. Parse extracted text for medical values
        5. Assess extraction quality
        
        Args:
            file_path: Path to medical document
            dpi: Resolution for PDF conversion (higher = better quality but slower)
            
        Returns:
            Dict with: text, fields, confidence, quality, method
        """
        file_path = Path(file_path)
        self.log(f"Processing: {file_path.name}")
        
        text = ""
        method = ""
        confidence = 0.0
        
        # =====================================
        # STEP 1: Detect file type
        # =====================================
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            # =====================================
            # STEP 2A: Try DIGITAL PDF extraction
            # =====================================
            # Digital PDFs have embedded text - extract directly (100% accurate)
            if HAS_FITZ:
                try:
                    doc = fitz.open(str(file_path))
                    texts = [page.get_text("text") for page in doc]
                    doc.close()
                    text = "\n".join(texts)
                    if len(text) > 200:  # Has meaningful text content
                        method = "digital_extraction"
                        confidence = 1.0  # Direct extraction = perfect confidence
                        self.log("✓ Digital PDF - text extracted directly")
                except Exception as e:
                    self.log(f"⚠ PyMuPDF error: {e}")
            
            # =====================================
            # STEP 2B: Fall back to OCR for scanned PDFs
            # =====================================
            # Scanned PDFs are images - need OCR
            if not text and HAS_PDF2IMAGE:
                self.log(f"⏳ Converting PDF to image ({dpi} DPI)...")
                try:
                    # Convert PDF page to high-resolution image
                    images = convert_from_path(str(file_path), dpi=dpi, first_page=1, last_page=1)
                    if images:
                        img_array = np.array(images[0])
                        text, confidence = self._process_image_ocr(img_array)
                        method = "ocr"
                except Exception as e:
                    self.log(f"⚠ pdf2image error: {e}")
        
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            # =====================================
            # STEP 2C: Direct image OCR
            # =====================================
            img = cv2.imread(str(file_path))
            if img is not None:
                text, confidence = self._process_image_ocr(img)
                method = "ocr"
        
        # Parse fields
        fields = self.parse_cardiovascular_fields(text)
        
        # Quality assessment
        quality = self._assess_quality(fields, confidence)
        
        self.log(f"✓ Extracted {len(fields)} fields, confidence: {confidence*100:.1f}%")
        
        return {
            'text': text,
            'fields': fields,
            'confidence': confidence,
            'quality': quality,
            'method': method,
        }
    
    def _process_image_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Process image through OCR - try raw first, then preprocessing"""
        best_text = ""
        best_conf = 0.0
        
        # FIRST: Try raw image (works best for clean/digital documents)
        for psm in [6, 3]:
            text, conf = self.run_tesseract(image, psm)
            if conf > best_conf:
                best_conf = conf
                best_text = text
            if conf > 0.85:  # Good enough
                return best_text, best_conf
        
        # FALLBACK: Try preprocessing only if raw didn't work well
        if best_conf < 0.70:
            for preprocess_method in ['clahe', 'otsu', 'adaptive']:
                processed = self.preprocess_image(image, preprocess_method)
                
                for psm in [6, 3]:
                    text, conf = self.run_tesseract(processed, psm)
                    if conf > best_conf:
                        best_conf = conf
                        best_text = text
                    if conf > 0.80:
                        return best_text, best_conf
        
        return best_text, best_conf
    
    def _assess_quality(self, fields: Dict, confidence: float) -> str:
        """Assess extraction quality"""
        if len(fields) >= 6 and confidence >= 0.85:
            return "high"
        elif len(fields) >= 4 and confidence >= 0.70:
            return "medium"
        else:
            return "low"


# ============================================================================
# CLASS: DualModelPipeline
# ============================================================================
# This is the MAIN ORCHESTRATOR that ties everything together.
# It loads the trained ML models and coordinates the prediction flow.
# ============================================================================
class DualModelPipeline:
    """
    MAIN PREDICTION PIPELINE
    ========================
    Integrates OCR + Detection Model + Prediction Model + Clinical Advisor
    
    MODELS USED:
    1. Detection Model: Voting Ensemble (XGBoost + RandomForest + LightGBM)
       - Purpose: Does patient currently have heart disease?
       - Accuracy: 91.45%
       
    2. Prediction Model: XGBoost
       - Purpose: 10-year cardiovascular risk prediction
       - Accuracy: 91.63%
    
    COMPLETE FLOW:
    process_document(file) → OCR → Detection → Prediction → Clinical Risk
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.ocr = EnhancedMedicalOCR(verbose=verbose)  # Initialize OCR engine
        
        # Load trained ML models from .pkl files
        self._load_models()
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def _load_models(self):
        """
        LOAD TRAINED MODELS
        ===================
        Loads pre-trained models from Milestone_2/models/Final_models/
        
        Models loaded:
        - detection_voting_optimized.pkl: Voting Ensemble (XGB + RF + LightGBM)
        - detection_scaler_v3.pkl: StandardScaler for feature normalization
        - prediction_xgb.pkl: XGBoost for 10-year risk prediction
        """
        # Path resolution: pipeline/ -> models/Final_models/
        pipeline_dir = Path(__file__).parent  # Milestone_2/pipeline
        models_dir = pipeline_dir.parent / "models" / "Final_models"  # Milestone_2/models/Final_models
        
        # =============================================
        # DETECTION MODEL: Heart Disease Detection
        # =============================================
        # Uses Voting Ensemble: XGBoost + RandomForest + LightGBM
        # Soft voting combines probability estimates from all 3 models
        self.detection_models = {}
        self.detection_scaler = None
        self.detection_config = None
        self.primary_detection_model = None
        
        detection_dir = models_dir / "detection"
        if detection_dir.exists():
            # Load the main voting ensemble model (91.30% accuracy)
            optimized_path = detection_dir / "detection_voting_optimized.pkl"
            if optimized_path.exists():
                # This is a VotingClassifier with estimators: [('xgb', XGBClassifier), 
                # ('rf', RandomForestClassifier), ('lgbm', LGBMClassifier)]
                self.primary_detection_model = joblib.load(optimized_path)
                self.detection_models['voting_optimized'] = self.primary_detection_model
            
            # Load v3 scaler (for 18 engineered features)
            scaler_v3_path = detection_dir / "detection_scaler_v3.pkl"
            if scaler_v3_path.exists():
                self.detection_scaler = joblib.load(scaler_v3_path)
            else:
                scaler_path = detection_dir / "detection_scaler.pkl"
                if scaler_path.exists():
                    self.detection_scaler = joblib.load(scaler_path)
            
            # Load config with threshold and features
            config_path = detection_dir / "detection_config_v3.pkl"
            if config_path.exists():
                self.detection_config = joblib.load(config_path)
            
            # Load other models as fallback
            for model_file in detection_dir.glob("detection_*.pkl"):
                if 'scaler' not in model_file.name and 'config' not in model_file.name and 'features' not in model_file.name:
                    name = model_file.stem.replace('detection_', '')
                    if name not in self.detection_models:
                        self.detection_models[name] = joblib.load(model_file)
        
        # Prediction model (91.63% XGBoost)
        self.prediction_model = None
        self.prediction_scaler = None
        self.prediction_features = None
        
        prediction_dir = models_dir / "prediction"
        pred_path = prediction_dir / "prediction_xgb.pkl"
        if pred_path.exists():
            model_data = joblib.load(pred_path)
            # Handle both dict format and direct model objects
            if hasattr(model_data, 'predict'):
                self.prediction_model = model_data
            else:
                self.prediction_model = model_data.get('model')
                self.prediction_scaler = model_data.get('scaler')
                self.prediction_features = model_data.get('feature_cols', [])
                self.prediction_threshold = model_data.get('threshold', 0.5)
        
        self.log(f"✓ Loaded {len(self.detection_models)} detection models (primary: voting_optimized 91.30%)")
        self.log(f"✓ Loaded prediction model: {self.prediction_model is not None}")
    
    def categorize_risk(self, probability: float) -> Tuple[str, str, str]:
        """Categorize risk level with color coding (clinically calibrated thresholds)"""
        if probability < 0.15:
            return "🟢 LOW", "Routine monitoring recommended", "low"
        elif probability < 0.40:
            return "🟡 MODERATE", "Lifestyle modifications advised", "moderate"
        else:
            return "🔴 HIGH", "Medical consultation recommended", "high"
    
    def check_missing_values(self, fields: Dict[str, Any]) -> Tuple[list, list]:
        """Check for missing critical and important fields"""
        critical_fields = ['age', 'systolic_bp']
        important_fields = ['sex_code', 'total_cholesterol', 'smoking', 'diabetes', 'bmi', 'fasting_glucose']
        
        missing_critical = [f for f in critical_fields if f not in fields or fields.get(f) is None]
        missing_important = [f for f in important_fields if f not in fields or fields.get(f) is None]
        
        return missing_critical, missing_important
    
    def calculate_clinical_risk(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate clinical risk score based on standard cardiovascular risk factors.
        Uses dynamic max_score based on extracted fields only.
        Includes age-based minimum risk thresholds.
        """
        score = 0
        max_score = 0
        risk_factors = []
        
        # Get age (critical factor)
        age = features.get('age', 55)
        
        # ========== AGE SCORING (0-30 points) ==========
        max_score += 30
        if age >= 75:
            score += 30
            risk_factors.append(f"Age {age} (≥75 = VERY HIGH)")
        elif age >= 65:
            score += 22
            risk_factors.append(f"Age {age} (65-74 = HIGH)")
        elif age >= 55:
            score += 12
            risk_factors.append(f"Age {age} (55-64 = MODERATE)")
        elif age >= 45:
            score += 6
            risk_factors.append(f"Age {age} (45-54 = MILD)")
        
        # ========== BLOOD PRESSURE (0-20 points) ==========
        if 'systolic_bp' in features:
            sbp = features['systolic_bp']
            dbp = features.get('diastolic_bp', 80)
            max_score += 20
            if sbp >= 160 or dbp >= 100:
                score += 20
                risk_factors.append(f"BP {sbp}/{dbp} (Stage 2 HTN)")
            elif sbp >= 140 or dbp >= 90:
                score += 15
                risk_factors.append(f"BP {sbp}/{dbp} (Stage 1 HTN)")
            elif sbp >= 130 or dbp >= 85:  # Changed from 80 to 85
                score += 6
                risk_factors.append(f"BP {sbp}/{dbp} (Elevated)")
            # Normal BP (< 130/85) adds 0 points
        
        # ========== CHOLESTEROL (0-15 points) ==========
        if 'total_cholesterol' in features:
            chol = features['total_cholesterol']
            max_score += 15
            if chol >= 280:
                score += 15
                risk_factors.append(f"Cholesterol {chol} (Very High)")
            elif chol >= 240:
                score += 12
                risk_factors.append(f"Cholesterol {chol} (High)")
            elif chol >= 200:
                score += 5
                risk_factors.append(f"Cholesterol {chol} (Borderline)")
            # Normal cholesterol (< 200) adds 0 points
        
        # ========== DIABETES/GLUCOSE (0-15 points) ==========
        if 'diabetes' in features or 'fasting_glucose' in features:
            diabetes = features.get('diabetes', 0)
            glucose = features.get('fasting_glucose', 90)
            max_score += 15
            if diabetes == 1 or glucose >= 126:
                score += 15
                risk_factors.append("Diabetes (Major risk)")
            elif glucose >= 110:
                score += 5
                risk_factors.append(f"Pre-diabetic glucose {glucose}")
            # Normal glucose (< 110) adds 0 points
        
        # ========== SMOKING (0-15 points) ==========
        if 'smoking' in features:
            max_score += 15
            if features['smoking'] == 1:
                score += 15
                risk_factors.append("Current smoker")
            # Non-smoker adds 0 points
        
        # ========== BMI/OBESITY (0-10 points) ==========
        if 'bmi' in features:
            bmi = features['bmi']
            max_score += 10
            if bmi >= 35:
                score += 10
                risk_factors.append(f"BMI {bmi:.1f} (Severe obesity)")
            elif bmi >= 30:
                score += 7
                risk_factors.append(f"BMI {bmi:.1f} (Obese)")
            elif bmi >= 27:  # Changed from 25 to 27
                score += 3
                risk_factors.append(f"BMI {bmi:.1f} (Overweight)")
            # Normal BMI (< 27) adds 0 points
        
        # ========== SEX (0-5 points) ==========
        if 'sex_code' in features:
            max_score += 5
            if features['sex_code'] == 1:
                score += 5
                risk_factors.append("Male sex")
        
        # Ensure max_score is at least 100
        if max_score == 0:
            max_score = 100
        
        # Calculate percentage
        pct = (score / max_score) * 100
        
        # ========== RISK LEVEL DETERMINATION ==========
        # Count major risk factors for multi-factor escalation
        major_factors = 0
        if 'smoking' in features and features['smoking'] == 1:
            major_factors += 1
        if 'diabetes' in features and features['diabetes'] == 1:
            major_factors += 1
        if 'systolic_bp' in features and features['systolic_bp'] >= 140:
            major_factors += 1
        if 'total_cholesterol' in features and features['total_cholesterol'] >= 240:
            major_factors += 1
        
        # Multi-factor escalation: ALL 4 major factors = HIGH regardless of age
        if major_factors == 4 and pct >= 50:
            level = "🔴 HIGH"
            level_code = "high"
        # Apply age-based thresholds with higher bar for HIGH classification
        elif age >= 75:
            # Age 75+ always at least MODERATE, HIGH only at very high scores
            if pct > 75:
                level = "🔴 HIGH"
                level_code = "high"
            else:
                level = "🟡 MODERATE"
                level_code = "moderate"
        elif age >= 65:
            # Age 65-74: elevated baseline, HIGH at 65%+
            if pct > 65:
                level = "🔴 HIGH"
                level_code = "high"
            elif pct >= 20:
                level = "🟡 MODERATE"
                level_code = "moderate"
            else:
                level = "🟢 LOW"
                level_code = "low"
        else:
            # Under 65: HIGH only at >60%, MODERATE at 20%+
            if pct > 60:
                level = "🔴 HIGH"
                level_code = "high"
            elif pct >= 20:
                level = "🟡 MODERATE"
                level_code = "moderate"
            else:
                level = "🟢 LOW"
                level_code = "low"
        
        # Set recommendation
        if level_code == "high":
            recommendation = "⚠️ URGENT: Medical consultation strongly recommended"
        elif level_code == "moderate":
            recommendation = "Lifestyle modifications and regular monitoring advised"
        else:
            recommendation = "Maintain healthy lifestyle, routine check-ups"
        
        return {
            'score': score,
            'max_score': max_score,
            'probability': score / max_score,
            'percentage': pct,
            'risk_level': level,
            'level_code': level_code,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
        }
    
    def predict_from_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction using extracted features"""
        results = {
            'detection': None,
            'prediction': None,
        }
        
        # Get base features with defaults
        age = features.get('age', 55)
        sex = features.get('sex_code', 1)
        systolic_bp = features.get('systolic_bp', 130)
        diastolic_bp = features.get('diastolic_bp', 85)
        total_cholesterol = features.get('total_cholesterol', 220)
        heart_rate = features.get('heart_rate', 75)
        fasting_glucose = features.get('fasting_glucose', 100)
        bmi = features.get('bmi', 26)
        smoking = features.get('smoking', 0)
        diabetes = features.get('diabetes', 0)
        hypertension = features.get('hypertension', 0)
        bp_meds = features.get('bp_meds', 0)
        
        # Detection prediction (if we have required features)
        # NOTE: Detection model was trained on UCI Heart Disease dataset which includes
        # clinical stress test results (chest pain type, max HR during exercise, 
        # exercise-induced angina, ST depression). OCR-extracted data typically only
        # has resting vitals and cannot reliably be used with this model.
        
        # Check if this is OCR-extracted data (missing UCI-specific features)
        has_uci_features = ('cp' in features and 'thalach' in features and 
                           'exang' in features and 'oldpeak' in features)
        
        if self.detection_models and self.detection_scaler and has_uci_features:
            # Patient has actual UCI-style clinical test data
            try:
                det_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                               'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                               'ca', 'thal', 'age_hr']
                
                age_hr = features['age'] * features['thalach']  # age × heart rate interaction
                
                det_values = [
                    features['age'], features.get('sex_code', 1), features['cp'], 
                    features.get('systolic_bp', 130), features.get('total_cholesterol', 200),
                    1 if features.get('fasting_glucose', 100) > 120 else 0,
                    features.get('restecg', 0), features['thalach'],
                    features['exang'], features['oldpeak'],
                    features.get('slope', 0), features.get('ca', 0), features.get('thal', 0),
                    age_hr
                ]
                
                X = np.array([det_values])
                X_scaled = self.detection_scaler.transform(X)
                
                probs = [model.predict_proba(X_scaled)[0, 1] for model in self.detection_models.values()]
                det_prob = np.mean(probs)
                
                results['detection'] = {
                    'probability': det_prob,
                    'prediction': 'Disease Detected' if det_prob >= 0.5 else 'No Disease Detected',
                    'confidence': abs(det_prob - 0.5) * 2
                }
            except Exception as e:
                self.log(f"⚠ Detection error: {e}")
        
        if self.prediction_model and self.prediction_scaler:
            try:
                # New model uses 8 features: age, sex, systolic_bp, cholesterol, max_hr, age_sq, elderly, high_bp
                age_sq = age ** 2
                elderly = 1 if age >= 55 else 0
                high_bp_flag = 1 if systolic_bp >= 140 else 0
                
                # Build feature vector (8 features matching new model)
                pred_values = [
                    age, sex, systolic_bp, total_cholesterol, heart_rate,
                    age_sq, elderly, high_bp_flag
                ]
                
                X = np.array([pred_values])
                X_scaled = self.prediction_scaler.transform(X)
                
                pred_prob = self.prediction_model.predict_proba(X_scaled)[0, 1]
                
                # Apply threshold
                risk_positive = pred_prob >= self.prediction_threshold
                
                level, recommendation, level_code = self.categorize_risk(pred_prob)
                
                results['prediction'] = {
                    'probability': pred_prob,
                    'risk_level': level,
                    'level_code': level_code,
                    'recommendation': recommendation,
                    'high_risk': risk_positive,
                }
            except Exception as e:
                self.log(f"⚠ Prediction error: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Complete pipeline: OCR → Detection → Prediction"""
        self.log("\n" + "=" * 70)
        self.log("CARDIODETECT: DUAL-MODEL PIPELINE")
        self.log("Detection (91.45%) + Prediction (91.63%)")
        self.log("=" * 70)
        
        # Step 1: OCR
        self.log("\n[1/4] DOCUMENT OCR")
        self.log("-" * 40)
        ocr_result = self.ocr.extract_from_file(file_path)
        
        self.log(f"  Method: {ocr_result['method']}")
        self.log(f"  Quality: {ocr_result['quality'].upper()}")
        self.log(f"  Fields: {len(ocr_result['fields'])}")
        
        # Step 2: Check Missing Values
        self.log("\n[2/4] DATA VALIDATION")
        self.log("-" * 40)
        missing_critical, missing_important = self.check_missing_values(ocr_result['fields'])
        
        if missing_critical:
            self.log(f"  ⚠️  CRITICAL MISSING: {', '.join(missing_critical)}")
            self.log(f"     These fields are essential for accurate prediction!")
        
        if missing_important:
            self.log(f"  ⚡ IMPORTANT MISSING: {', '.join(missing_important)}")
            self.log(f"     Using default values - results may be less accurate")
        
        if not missing_critical and not missing_important:
            self.log("  ✅ All important fields extracted")
        
        # Step 3: Model Predictions
        self.log("\n[3/4] MODEL PREDICTIONS")
        self.log("-" * 40)
        
        predictions = self.predict_from_features(ocr_result['fields'])
        
        # Step 4: Clinical Risk Assessment
        self.log("\n[4/4] CLINICAL RISK ASSESSMENT")
        self.log("-" * 40)
        clinical_risk = self.calculate_clinical_risk(ocr_result['fields'])
        
        # Results
        self.log("\n" + "=" * 70)
        self.log("                    📊 CARDIODETECT RESULTS")
        self.log("=" * 70)
        
        # Show extracted fields
        self.log("\n📋 EXTRACTED PATIENT DATA:")
        self.log("   " + "-" * 40)
        for k, v in ocr_result['fields'].items():
            self.log(f"   • {k}: {v}")
        
        # Show missing value warnings
        if missing_critical or missing_important:
            self.log("\n⚠️  MISSING DATA WARNING:")
            if missing_critical:
                self.log(f"   CRITICAL: {', '.join(missing_critical)}")
            if missing_important:
                self.log(f"   Important: {', '.join(missing_important)}")
        
        # ========== SECTION 1: CURRENT STATUS DETECTION ==========
        self.log("\n" + "─" * 70)
        self.log("🔍 CURRENT HEART DISEASE STATUS")
        self.log("   \"Does the patient currently have heart disease?\"")
        self.log("─" * 70)
        
        if predictions['detection']:
            det = predictions['detection']
            if det['probability'] >= 0.5:
                self.log(f"\n   ⚠️  DISEASE DETECTED")
                self.log(f"   Confidence: {det['probability']*100:.1f}%")
                self.log(f"   → Recommend immediate cardiology evaluation")
            else:
                self.log(f"\n   ✅ NO DISEASE DETECTED")
                self.log(f"   Normal probability: {(1-det['probability'])*100:.1f}%")
                self.log(f"   → Patient unlikely to have current heart disease")
        else:
            self.log("\n   ℹ️  Detection model not applicable")
            self.log("   → This model requires stress test data (chest pain type,")
            self.log("      max heart rate during exercise, ST depression, etc.)")
            self.log("   → OCR-extracted reports typically contain only resting vitals")
            self.log("   → See 10-year risk prediction below for risk assessment")
        
        # ========== SECTION 2: 10-YEAR RISK PREDICTION ==========
        self.log("\n" + "─" * 70)
        self.log("📈 10-YEAR CARDIOVASCULAR RISK PREDICTION")
        self.log("   \"What is the patient's risk of developing CHD in 10 years?\"")
        self.log("─" * 70)
        
        # Clinical Risk Assessment (primary)
        self.log(f"\n   📊 CLINICAL ASSESSMENT (Guideline-Based)")
        self.log(f"   Score: {clinical_risk['score']}/{clinical_risk['max_score']} points")
        self.log(f"   Risk Level: {clinical_risk['risk_level']} ({clinical_risk['percentage']:.0f}%)")
        self.log(f"   → {clinical_risk['recommendation']}")
        
        if clinical_risk['risk_factors']:
            self.log(f"\n   Contributing Risk Factors:")
            for factor in clinical_risk['risk_factors']:
                self.log(f"      • {factor}")
        
        
        # ML Model prediction removed - Clinical Assessment is more accurate
        
        # ========== SECTION 3: RECOMMENDATIONS ==========
        self.log("\n" + "─" * 70)
        self.log("💡 RECOMMENDATIONS")
        self.log("─" * 70)
        
        if clinical_risk['level_code'] == 'high':
            self.log("\n   🔴 HIGH RISK PATIENT")
            self.log("   → Schedule immediate cardiology consultation")
            self.log("   → Begin aggressive risk factor management")
            self.log("   → Consider statin therapy and BP medications")
            self.log("   → Lifestyle intervention: diet, exercise, smoking cessation")
        elif clinical_risk['level_code'] == 'moderate':
            self.log("\n   🟡 MODERATE RISK PATIENT")
            self.log("   → Schedule follow-up with primary care physician")
            self.log("   → Lifestyle modifications recommended")
            self.log("   → Monitor blood pressure and cholesterol regularly")
            self.log("   → Consider further testing if symptoms develop")
        else:
            self.log("\n   🟢 LOW RISK PATIENT")
            self.log("   → Continue healthy lifestyle")
            self.log("   → Annual wellness check recommended")
            self.log("   → Maintain regular exercise and balanced diet")
        
        self.log("\n" + "=" * 70)
        self.log("   CardioDetect v2.0 | Detection (91.45%) | Prediction (91.63%)")
        self.log("=" * 70)
        
        # ========== SECTION 4: CLINICAL ADVISOR (Guideline-Based) ==========
        clinical_recommendations = None
        if HAS_CLINICAL_ADVISOR:
            self.log("\n" + "─" * 70)
            self.log("📋 CLINICAL ADVISOR (ACC/AHA & WHO Guidelines)")
            self.log("─" * 70)
            advisor = ClinicalAdvisor()
            clinical_recommendations = advisor.generate_recommendations(ocr_result['fields'])
            self.log(format_recommendations_text(clinical_recommendations))
        
        return {
            'ocr': ocr_result,
            'detection': predictions['detection'],
            'prediction': predictions['prediction'],
            'clinical_risk': clinical_risk,
            'clinical_recommendations': clinical_recommendations,
            'missing_critical': missing_critical,
            'missing_important': missing_important,
        }
    
    def export_pdf(self, result: Dict[str, Any], output_path: str = None) -> str:
        """
        Export the pipeline results as a PDF clinical report.
        
        Args:
            result: Output from process_document()
            output_path: Path for the PDF output (optional)
            
        Returns:
            Path to the generated PDF
        """
        if not HAS_PDF_GENERATOR:
            raise ImportError("PDF generator not available. Install reportlab.")
        
        # Extract patient data from OCR fields
        fields = result.get('ocr', {}).get('fields', {})
        
        patient_data = {
            'age': fields.get('age'),
            'sex': fields.get('sex', 0),
            'systolic_bp': fields.get('systolic_bp'),
            'diastolic_bp': fields.get('diastolic_bp'),
            'total_cholesterol': fields.get('total_cholesterol'),
            'hdl_cholesterol': fields.get('hdl'),
            'heart_rate': fields.get('heart_rate'),
            'diabetes': fields.get('diabetes', 0),
            'smoking': fields.get('smoker', 0),
            'bmi': fields.get('bmi'),
            'on_bp_meds': fields.get('on_treatment', False)
        }
        
        # Extract risk results
        clinical_risk = result.get('clinical_risk', {})
        risk_results = {
            'probability': clinical_risk.get('percentage', 0) / 100,
            'risk_level': clinical_risk.get('risk_level', 'Unknown'),
            'recommendation': clinical_risk.get('recommendation', ''),
            'high_risk': clinical_risk.get('level_code') == 'high'
        }
        
        # Clinical recommendations
        clinical_recs = result.get('clinical_recommendations')
        
        # Default output path
        if not output_path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(__file__).parent.parent / 'reports'
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f"clinical_report_{timestamp}.pdf")
        
        # Generate PDF
        pdf_path = generate_clinical_report(
            patient_data,
            risk_results,
            clinical_recs,
            output_path
        )
        
        self.log(f"\n📄 PDF Report exported: {pdf_path}")
        return pdf_path


def main():
    """Test the pipeline"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python integrated_pipeline.py <path_to_document> [--pdf]")
        print("\nOptions:")
        print("  --pdf    Export results to PDF report")
        print("\nSupported formats: PDF, PNG, JPG, JPEG, TIFF")
        sys.exit(1)
    
    file_path = sys.argv[1]
    export_pdf = '--pdf' in sys.argv
    
    pipeline = DualModelPipeline(verbose=True)
    result = pipeline.process_document(file_path)
    
    if export_pdf:
        try:
            pdf_path = pipeline.export_pdf(result)
            print(f"\n📄 PDF Report saved: {pdf_path}")
        except Exception as e:
            print(f"\n⚠️ PDF export failed: {e}")
    
    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()

