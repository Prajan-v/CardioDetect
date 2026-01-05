"""
UNIFIED MEDICAL REPORT PIPELINE
================================
Extracts data from medical reports and runs BOTH:
- Detection Model (Current heart disease status)
- Prediction Model (10-year CHD risk)

Features:
- Enhanced OCR with patterns for ST_Slope, ExerciseAngina
- Smart imputation for missing values (Cholesterol=0 handling)
- Calibrated models for accurate probability estimates
- Combined clinical report output
"""

import sys
sys.path.insert(0, '/Users/prajanv/CardioDetect/Milestone_2')

import re
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import base OCR
try:
    from pipeline.integrated_pipeline import EnhancedMedicalOCR
except ImportError:
    EnhancedMedicalOCR = None

# Clinical guidelines integration
from pipeline.clinical_advisor import ClinicalAdvisor, format_recommendations_text



class UnifiedMedicalPipeline:
    """
    Unified pipeline for medical reports containing both detection and prediction data.
    """
    
    # ===== IMPUTATION DEFAULTS (based on dataset analysis) =====
    DEFAULTS = {
        # Detection model features
        'age': 55,
        'sex': 1,  # Male (conservative)
        'chest_pain_type': 0,  # ASY
        'resting_bp': 132,
        'cholesterol': 240,  # Median of non-zero values
        'fasting_bs': 0,
        'resting_ecg': 0,
        'max_hr': 140,
        'exercise_angina': 0,
        'oldpeak': 0.9,
        'st_slope': 1,  # Flat (most predictive of disease)
        
        # Prediction model features
        'hdl': 50,
        'smoking': 0,
        'bp_meds': 0,
    }
    
    # ===== ENHANCED OCR PATTERNS =====
    ST_SLOPE_PATTERNS = [
        r'(?i)st[_\s-]*slope[:\s]+(\w+)',
        r'(?i)st[_\s-]*segment[:\s]+(\w+)',
        r'(?i)slope[:\s]*(up|flat|down|ascending|descending|normal)',
        r'(?i)(upsloping|flat|downsloping)\s*st',
        r'(?i)st\s*(elevation|depression|normal)',
    ]
    
    EXERCISE_ANGINA_PATTERNS = [
        r'(?i)exercise[_\s-]*(?:induced[_\s-]*)?angina[:\s]*(yes|no|y|n|positive|negative|1|0)',
        r'(?i)exang[:\s]*(yes|no|y|n|1|0)',
        r'(?i)exertional[_\s-]*(?:chest[_\s-]*)?pain[:\s]*(yes|no|present|absent)',
        r'(?i)stress[_\s-]*test[:\s]*(positive|negative|normal|abnormal)',
        r'(?i)angina[_\s-]*on[_\s-]*exertion[:\s]*(yes|no)',
    ]
    
    CHOLESTEROL_PATTERNS = [
        r'(?i)total[_\s-]*cholesterol[:\s]*(\d+(?:\.\d+)?)',
        r'(?i)cholesterol[:\s]*(\d+(?:\.\d+)?)\s*(?:mg|mmol)?',
        r'(?i)tc[:\s]*(\d+(?:\.\d+)?)',
    ]
    
    HDL_PATTERNS = [
        r'(?i)hdl[_\s-]*(?:cholesterol)?[:\s]*(\d+(?:\.\d+)?)',
        r'(?i)high[_\s-]*density[_\s-]*lipoprotein[:\s]*(\d+(?:\.\d+)?)',
        r'(?i)hdl-c[:\s]*(\d+(?:\.\d+)?)',
    ]
    
    def __init__(self, verbose: bool = True):
        self.clinical_advisor = ClinicalAdvisor()
        self.verbose = verbose
        self.ocr = EnhancedMedicalOCR(verbose=False) if EnhancedMedicalOCR else None
        
        # Load models
        self._load_models()
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def _load_models(self):
        """Load detection and prediction models"""
        
        # Detection model (calibrated)
        det_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/archive/detection_calibrated')
        if det_dir.exists():
            self.detection_model = joblib.load(det_dir / 'detection_lgbm_calibrated.pkl')
            self.detection_scaler = joblib.load(det_dir / 'detection_scaler.pkl')
            self.detection_features = joblib.load(det_dir / 'detection_features.pkl')
            self.log("✓ Loaded: Calibrated Detection Model (LightGBM)")
        else:
            # Fallback to original
            det_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/Final_models/detection')
            self.detection_model = joblib.load(det_dir / 'detection_best.pkl')
            self.detection_scaler = joblib.load(det_dir / 'detection_scaler.pkl')
            self.detection_features = joblib.load(det_dir / 'detection_features.pkl')
            self.log("✓ Loaded: Original Detection Model (XGBoost)")
        
        # Prediction model
        pred_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/Final_models/prediction')
        if pred_dir.exists():
            try:
                pred_data = joblib.load(pred_dir / 'prediction_xgb.pkl')
                # Handle dictionary format
                if isinstance(pred_data, dict):
                    self.prediction_model = pred_data.get('model')
                    self.prediction_scaler = pred_data.get('scaler')
                    self.prediction_threshold = pred_data.get('threshold', 0.5)
                    self.prediction_features = pred_data.get('feature_cols', [])
                else:
                    self.prediction_model = pred_data
                    self.prediction_scaler = None
                    self.prediction_threshold = 0.5
                    self.prediction_features = []
                self.log("✓ Loaded: Prediction Model (XGBoost)")
            except Exception as e:
                self.prediction_model = None
                self.log(f"⚠ Prediction model error: {e}")
        else:
            self.prediction_model = None
    
    def _extract_st_slope(self, text: str) -> Optional[int]:
        """Extract ST_Slope from text: Up=0, Flat=1, Down=2"""
        for pattern in self.ST_SLOPE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                value = match.group(1).lower()
                if value in ['up', 'upsloping', 'ascending', 'normal']:
                    return 0
                elif value in ['flat']:
                    return 1
                elif value in ['down', 'downsloping', 'descending', 'depression', 'abnormal']:
                    return 2
        return None
    
    def _extract_exercise_angina(self, text: str) -> Optional[int]:
        """Extract ExerciseAngina from text: No=0, Yes=1"""
        for pattern in self.EXERCISE_ANGINA_PATTERNS:
            match = re.search(pattern, text)
            if match:
                value = match.group(1).lower()
                if value in ['yes', 'y', 'positive', '1', 'present', 'abnormal']:
                    return 1
                elif value in ['no', 'n', 'negative', '0', 'absent', 'normal']:
                    return 0
        return None
    
    def _extract_cholesterol(self, text: str) -> Optional[float]:
        """Extract cholesterol, handling zero values"""
        for pattern in self.CHOLESTEROL_PATTERNS:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                # Treat values < 50 as invalid/missing
                if value >= 50:
                    return value
        return None
    
    def _extract_hdl(self, text: str) -> Optional[float]:
        """Extract HDL cholesterol"""
        for pattern in self.HDL_PATTERNS:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                if 10 <= value <= 150:  # Valid HDL range
                    return value
        return None
    
    def _impute_cholesterol(self, age: int, sex: int) -> float:
        """Impute cholesterol based on age and sex"""
        base = 200
        age_adj = (age - 50) * 0.5
        sex_adj = 10 if sex == 1 else 0  # Males have slightly higher
        return base + age_adj + sex_adj
    
    def _impute_maxhr(self, age: int) -> int:
        """Impute max heart rate from age (220 - age rule)"""
        return max(100, min(200, 220 - age))
    
    def _impute_hdl(self, sex: int) -> float:
        """Impute HDL based on sex"""
        return 45 if sex == 1 else 55  # Females have higher HDL
    
    def extract_from_report(self, file_path: str) -> Dict[str, Any]:
        """
        Extract all fields from medical report using enhanced OCR.
        Returns raw fields with additional extracted values.
        """
        self.log(f"\n📄 Processing: {file_path}")
        
        # Run base OCR
        if self.ocr:
            result = self.ocr.extract_from_file(file_path)
            fields = result['fields']
            raw_text = result.get('raw_text', '')
        else:
            # Fallback: just load image and run basic extraction
            fields = {}
            raw_text = ''
        
        # Enhanced extraction for critical fields
        if raw_text:
            # ST_Slope
            if fields.get('slope') is None:
                st_slope = self._extract_st_slope(raw_text)
                if st_slope is not None:
                    fields['slope'] = st_slope
                    self.log("   ✓ Enhanced: Extracted ST_Slope from text")
            
            # ExerciseAngina
            if fields.get('exang') is None:
                exang = self._extract_exercise_angina(raw_text)
                if exang is not None:
                    fields['exang'] = exang
                    self.log("   ✓ Enhanced: Extracted ExerciseAngina from text")
            
            # Cholesterol (with zero handling)
            if fields.get('total_cholesterol') is None or fields.get('total_cholesterol', 0) < 50:
                chol = self._extract_cholesterol(raw_text)
                if chol is not None:
                    fields['total_cholesterol'] = chol
                    self.log("   ✓ Enhanced: Extracted Cholesterol from text")
            
            # HDL
            if fields.get('hdl') is None:
                hdl = self._extract_hdl(raw_text)
                if hdl is not None:
                    fields['hdl'] = hdl
                    self.log("   ✓ Enhanced: Extracted HDL from text")
        
        return {
            'fields': fields,
            'raw_text': raw_text,
        }
    
    def prepare_detection_features(self, fields: Dict) -> Tuple[np.ndarray, list]:
        """
        Map OCR fields to detection model features with smart imputation.
        Returns feature array and list of imputed fields.
        """
        imputed = []
        
        # Age
        age = fields.get('age', self.DEFAULTS['age'])
        if fields.get('age') is None:
            imputed.append('Age')
        
        # Sex
        sex_val = fields.get('sex', '')
        if isinstance(sex_val, str) and sex_val.lower() in ['male', 'm']:
            sex = 1
        elif isinstance(sex_val, str) and sex_val.lower() in ['female', 'f']:
            sex = 0
        elif fields.get('sex_code') is not None:
            sex = fields['sex_code']
        else:
            sex = self.DEFAULTS['sex']
            imputed.append('Sex')
        
        # ChestPainType
        chest_pain = fields.get('cp', self.DEFAULTS['chest_pain_type'])
        if fields.get('cp') is None:
            imputed.append('ChestPainType')
        
        # RestingBP
        resting_bp = fields.get('systolic_bp', self.DEFAULTS['resting_bp'])
        if fields.get('systolic_bp') is None:
            imputed.append('RestingBP')
        
        # Cholesterol (with zero handling!)
        chol = fields.get('total_cholesterol', 0)
        if chol is None or chol < 50:
            chol = self._impute_cholesterol(age, sex)
            imputed.append('Cholesterol (imputed from age/sex)')
        
        # FastingBS
        if fields.get('diabetes') is not None:
            fasting_bs = 1 if fields['diabetes'] else 0
        elif fields.get('fasting_glucose'):
            fasting_bs = 1 if fields['fasting_glucose'] > 126 else 0
        else:
            fasting_bs = self.DEFAULTS['fasting_bs']
            imputed.append('FastingBS')
        
        # RestingECG
        resting_ecg = fields.get('restecg', self.DEFAULTS['resting_ecg'])
        if fields.get('restecg') is None:
            imputed.append('RestingECG')
        
        # MaxHR
        max_hr = fields.get('thalach')
        if max_hr is None:
            max_hr = self._impute_maxhr(age)
            imputed.append('MaxHR (estimated from age)')
        
        # ExerciseAngina
        exercise_angina = fields.get('exang', self.DEFAULTS['exercise_angina'])
        if fields.get('exang') is None:
            imputed.append('ExerciseAngina')
        
        # Oldpeak
        oldpeak = fields.get('oldpeak', self.DEFAULTS['oldpeak'])
        if fields.get('oldpeak') is None:
            imputed.append('Oldpeak')
        
        # ST_Slope
        st_slope = fields.get('slope', self.DEFAULTS['st_slope'])
        if fields.get('slope') is None:
            imputed.append('ST_Slope')
        
        features = np.array([[age, sex, chest_pain, resting_bp, chol,
                             fasting_bs, resting_ecg, max_hr, exercise_angina, 
                             oldpeak, st_slope]])
        
        return features, imputed
    
    def prepare_prediction_features(self, fields: Dict) -> Tuple[np.ndarray, list]:
        """
        Map OCR fields to prediction model features with smart imputation.
        Returns feature array and list of imputed fields.
        """
        imputed = []
        
        # Age
        age = fields.get('age', self.DEFAULTS['age'])
        if fields.get('age') is None:
            imputed.append('age')
        
        # Sex (male = 1)
        sex_val = fields.get('sex', '')
        if isinstance(sex_val, str) and sex_val.lower() in ['male', 'm']:
            male = 1
        elif isinstance(sex_val, str) and sex_val.lower() in ['female', 'f']:
            male = 0
        elif fields.get('sex_code') is not None:
            male = fields['sex_code']
        else:
            male = 1
            imputed.append('male')
        
        # Smoking
        smoking = 1 if fields.get('smoking') else 0
        if fields.get('smoking') is None:
            imputed.append('currentSmoker')
        
        cigs_per_day = 0  # Default
        
        # BP Meds
        bp_meds = 1 if fields.get('hypertension') else 0
        if fields.get('hypertension') is None:
            imputed.append('BPMeds')
        
        # Diabetes
        diabetes = 1 if fields.get('diabetes') else 0
        if fields.get('diabetes') is None:
            imputed.append('diabetes')
        
        # Total Cholesterol
        tot_chol = fields.get('total_cholesterol', 0)
        if tot_chol is None or tot_chol < 50:
            tot_chol = self._impute_cholesterol(age, male)
            imputed.append('totChol')
        
        # Systolic BP
        sys_bp = fields.get('systolic_bp', self.DEFAULTS['resting_bp'])
        if fields.get('systolic_bp') is None:
            imputed.append('sysBP')
        
        # Diastolic BP
        dia_bp = fields.get('diastolic_bp', 80)
        if fields.get('diastolic_bp') is None:
            imputed.append('diaBP')
        
        # BMI
        bmi = fields.get('bmi', 25)
        if fields.get('bmi') is None:
            imputed.append('BMI')
        
        # Heart Rate
        heart_rate = fields.get('heart_rate', 75)
        if fields.get('heart_rate') is None:
            imputed.append('heartRate')
        
        # Glucose
        glucose = fields.get('fasting_glucose', 90)
        if fields.get('fasting_glucose') is None:
            imputed.append('glucose')
        
        # Feature order: male, age, currentSmoker, cigsPerDay, BPMeds, diabetes, 
        #                totChol, sysBP, diaBP, BMI, heartRate, glucose
        features = np.array([[male, age, smoking, cigs_per_day, bp_meds, diabetes,
                             tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose]])
        
        return features, imputed
    
    def run_detection(self, features: np.ndarray) -> Dict:
        """Run detection model"""
        X_scaled = self.detection_scaler.transform(features)
        prediction = self.detection_model.predict(X_scaled)[0]
        proba = self.detection_model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'disease_probability': float(proba[1]),
            'no_disease_probability': float(proba[0]),
        }
    
    def run_prediction(self, features: np.ndarray) -> Dict:
        """Run prediction model"""
        if self.prediction_model is None:
            return {'error': 'Prediction model not available'}
        
        try:
            # Scale if scaler exists, otherwise use raw features
            if self.prediction_scaler is not None:
                X = self.prediction_scaler.transform(features)
            else:
                X = features
            
            prediction = self.prediction_model.predict(X)[0]
            proba = self.prediction_model.predict_proba(X)[0]
            
            # Categorize risk
            risk_prob = proba[1] * 100
            if risk_prob < 10:
                category = 'LOW'
            elif risk_prob < 20:
                category = 'MODERATE'
            else:
                category = 'HIGH'
            
            return {
                'prediction': int(prediction),
                'risk_probability': float(proba[1]),
                'risk_percentage': risk_prob,
                'risk_category': category,
            }
        except Exception as e:
            return {'error': str(e)}
    
    def process_report(self, file_path: str) -> Dict:
        """
        Main entry point: Process medical report through entire pipeline.
        
        Returns combined detection + prediction results.
        """
        self.log("=" * 70)
        self.log("UNIFIED MEDICAL REPORT PIPELINE")
        self.log("=" * 70)
        
        # Step 1: Extract fields
        extraction = self.extract_from_report(file_path)
        fields = extraction['fields']
        
        self.log("\n📊 Extracted Fields:")
        for k, v in sorted(fields.items()):
            if v is not None:
                self.log(f"   {k}: {v}")
        
        # Step 2: Prepare features for both models
        det_features, det_imputed = self.prepare_detection_features(fields)
        pred_features, pred_imputed = self.prepare_prediction_features(fields)
        
        # Step 3: Run models
        det_result = self.run_detection(det_features)
        pred_result = self.run_prediction(pred_features)
        
        # Step 4: Assess data quality
        total_imputed = len(set(det_imputed + pred_imputed))
        if total_imputed <= 3:
            confidence = 'HIGH'
        elif total_imputed <= 6:
            confidence = 'MODERATE'
        else:
            confidence = 'LOW'
        
        # Compile results
        result = {
            'detection': {
                'status': 'DISEASE DETECTED' if det_result['prediction'] == 1 else 'NO DISEASE',
                'probability': det_result['disease_probability'],
                'imputed_fields': det_imputed,
            },
            'prediction': {
                'risk_category': pred_result.get('risk_category', 'UNKNOWN'),
                'risk_percentage': pred_result.get('risk_percentage', 0),
                'imputed_fields': pred_imputed,
            } if 'error' not in pred_result else {'error': pred_result['error']},
            'data_quality': {
                'confidence': confidence,
                'total_imputed': total_imputed,
            },
            'raw_fields': fields,
        }
        # Generate clinical recommendations using ClinicalAdvisor
        advisor_features = {
            'age': fields.get('age', self.DEFAULTS['age']),
            'sex_code': fields.get('sex', self.DEFAULTS['sex']),
            'systolic_bp': fields.get('systolic_bp', self.DEFAULTS['resting_bp']),
            'diastolic_bp': fields.get('diastolic_bp', 80),
            'total_cholesterol': fields.get('total_cholesterol', self.DEFAULTS['cholesterol']),
            'hdl': fields.get('hdl', self.DEFAULTS['hdl']),
            'smoking': fields.get('smoking', self.DEFAULTS['smoking']),
            'on_bp_meds': fields.get('on_bp_meds', self.DEFAULTS['bp_meds']),
            'on_statin': fields.get('on_statin', False),
            'allergies': fields.get('allergies', []),
        }
        clinical_recs = self.clinical_advisor.generate_recommendations(advisor_features)
        result['clinical_recommendations'] = clinical_recs
        result['clinical_recommendations_text'] = format_recommendations_text(clinical_recs)
        
        # Print results
        self.log("\n" + "=" * 70)
        self.log("🔬 RESULTS")
        self.log("=" * 70)
        
        self.log("\n📍 DETECTION (Current Heart Disease Status):")
        self.log(f"   Status: {'❤️ ' + result['detection']['status'] if det_result['prediction'] == 1 else '✅ ' + result['detection']['status']}")
        self.log(f"   Probability: {det_result['disease_probability']:.1%}")
        self.log(f"   Imputed: {len(det_imputed)} fields")
        
        self.log("\n📊 PREDICTION (10-Year CHD Risk):")
        if 'error' not in pred_result:
            risk_emoji = '🔴' if pred_result['risk_category'] == 'HIGH' else ('🟡' if pred_result['risk_category'] == 'MODERATE' else '🟢')
            self.log(f"   Category: {risk_emoji} {pred_result['risk_category']}")
            self.log(f"   Risk: {pred_result['risk_percentage']:.1f}%")
            self.log(f"   Imputed: {len(pred_imputed)} fields")
        else:
            self.log(f"   Error: {pred_result['error']}")
        
        self.log(f"\n📋 Data Quality: {confidence}")
        if confidence == 'LOW':
            self.log("   ⚠️  Many fields imputed - results may be less reliable")
        
        if result.get('clinical_recommendations_text'):
            self.log("\n🩺 CLINICAL RECOMMENDATIONS:")
            self.log(result['clinical_recommendations_text'])
        
        self.log("=" * 70)
        
        return result


def main():
    """Test the unified pipeline"""
    pipeline = UnifiedMedicalPipeline(verbose=True)
    
    # Test with both reports
    test_files = [
        '/Users/prajanv/CardioDetect/Milestone_2/Medical_report/Synthetic_report/SYN-012.png',
        '/Users/prajanv/CardioDetect/Milestone_2/Medical_report/Synthetic_report/SYN-016.png',
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            result = pipeline.process_report(test_file)
            print("\n")


if __name__ == "__main__":
    main()
