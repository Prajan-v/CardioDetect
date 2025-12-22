"""
ML Service - Integration with Milestone 2 models.
Provides prediction and OCR functionality for the Django backend.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add Milestone 2 paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MILESTONE_2 = PROJECT_ROOT / 'Milestone_2'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MILESTONE_2))
sys.path.insert(0, str(MILESTONE_2 / 'pipeline'))


class MLService:
    """Machine Learning service for CardioDetect - Singleton Pattern."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.pipeline = None
        self.loaded = False
        self._initialized = True
        self.enhanced_predictor = None
        self.clinical_advisor = None
        self.format_recommendations = None
        # Lazy load on first use - avoid conflicts with Django reloader
    
    def _ensure_loaded(self):
        """Lazy load the pipeline on first use."""
        if self.loaded:
            return True
        return self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the integrated pipeline from Milestone 2."""
        try:
            from integrated_pipeline import DualModelPipeline
            self.pipeline = DualModelPipeline(verbose=False)
            self.loaded = True
            print("✓ ML Pipeline loaded successfully")
            
            # Try to load enhanced predictor for SHAP explanations
            try:
                from enhanced_predictor import create_enhanced_predictor
                self.enhanced_predictor = create_enhanced_predictor()
                if self.enhanced_predictor:
                    print("✓ Enhanced predictor with SHAP loaded")
            except Exception as e:
                print(f"⚠ Enhanced predictor not available: {e}")
                self.enhanced_predictor = None
            
            # Try to load Clinical Advisor for ACC/AHA guidelines
            try:
                from clinical_advisor import ClinicalAdvisor, format_recommendations_text
                self.clinical_advisor = ClinicalAdvisor()
                self.format_recommendations = format_recommendations_text
                print("✓ Clinical Advisor (ACC/AHA guidelines) loaded")
            except Exception as e:
                print(f"⚠ Clinical Advisor not available: {e}")
                self.clinical_advisor = None
                self.format_recommendations = None
            
            return True
        except ImportError as e:
            print(f"⚠ Could not load ML pipeline: {e}")
            self.loaded = False
            return False
        except Exception as e:
            print(f"⚠ Error loading ML pipeline: {e}")
            self.loaded = False
            return False
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run prediction on input features.
        
        Args:
            features: Dictionary of patient features
            
        Returns:
            Dictionary with prediction results
        """
        # Lazy load pipeline on first use
        self._ensure_loaded()
        
        if not self.loaded:
            return self._fallback_prediction(features)
        
        try:
            # Map feature names to pipeline expectations
            mapped_features = self._map_features(features)
            
            # Run pipeline prediction
            result = self.pipeline.predict_from_features(mapped_features)
            
            # Use clinical assessment for risk (more accurate for OCR-extracted data)
            # The ML model overweights age; clinical assessment uses guideline-based scoring
            clinical = result.get('clinical_risk', result.get('clinical_assessment', {}))
            prediction_data = result.get('prediction', {})
            
            # Prefer clinical assessment; fallback to ML prediction
            risk_percentage = clinical.get('percentage', prediction_data.get('probability', 0) * 100)
            risk_category = clinical.get('level_code', 'moderate').upper()
            
            response = {
                'risk_score': clinical.get('score'),
                'risk_percentage': risk_percentage,
                'risk_category': risk_category.replace('🟢 ', '').replace('🟡 ', '').replace('🔴 ', ''),
                'confidence': prediction_data.get('confidence'),
                'detection_result': result.get('detection', {}).get('prediction') == 'Disease Detected' if result.get('detection') else None,
                'detection_probability': result.get('detection', {}).get('probability'),
                'clinical_score': clinical.get('score'),
                'clinical_max_score': clinical.get('max_score'),
                'recommendations': clinical.get('recommendation', self._get_recommendations(result)),
                'risk_factors': clinical.get('risk_factors', []),
                'model_used': 'clinical_assessment',
                'explanations': [],  # SHAP explanations
            }
            
            # Add SHAP explanations if enhanced predictor available
            if hasattr(self, 'enhanced_predictor') and self.enhanced_predictor:
                try:
                    shap_result = self.enhanced_predictor.predict_with_explanation(mapped_features)
                    response['explanations'] = shap_result.get('explanations', [])
                    response['shap_base_value'] = shap_result.get('base_value')
                    response['model_probability'] = shap_result.get('probability')
                except Exception as e:
                    print(f"SHAP explanation error: {e}")
            
            # Add Clinical Advisor recommendations (ACC/AHA guidelines)
            if hasattr(self, 'clinical_advisor') and self.clinical_advisor:
                try:
                    clinical_recs = self.clinical_advisor.generate_recommendations(mapped_features)
                    response['clinical_recommendations'] = {
                        'recommendations': clinical_recs.get('recommendations', []),
                        'urgency': clinical_recs.get('urgency', 'Routine'),
                        'summary': clinical_recs.get('summary', ''),
                    }
                    # Also add formatted text version
                    if self.format_recommendations:
                        response['clinical_recommendations_text'] = self.format_recommendations(clinical_recs)
                except Exception as e:
                    print(f"Clinical Advisor error: {e}")
            
            return response
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction(features)
    
    def _map_features(self, features: Dict) -> Dict:
        """Map input features to pipeline format."""
        return {
            'age': features.get('age'),
            'sex': features.get('sex'),
            'sex_code': features.get('sex', 1),
            'systolic_bp': features.get('systolic_bp'),
            'diastolic_bp': features.get('diastolic_bp'),
            'total_cholesterol': features.get('cholesterol'),
            'heart_rate': features.get('heart_rate'),
            'fasting_glucose': features.get('glucose'),
            'bmi': features.get('bmi'),
            'smoking': 1 if features.get('smoking') else 0,
            'diabetes': 1 if features.get('diabetes') else 0,
            'hypertension': 1 if features.get('systolic_bp', 0) >= 140 else 0,
            'bp_meds': 1 if features.get('bp_medication') else 0,
            # Stress test fields
            'cp': features.get('chest_pain_type'),
            'thalach': features.get('max_heart_rate'),
            'exang': 1 if features.get('exercise_angina') else 0,
            'oldpeak': features.get('st_depression'),
            'slope': features.get('st_slope'),
            'ca': features.get('major_vessels'),
            'thal': features.get('thalassemia'),
            'restecg': features.get('resting_ecg'),
        }
    
    def _fallback_prediction(self, features: Dict) -> Dict:
        """Fallback prediction using clinical rules."""
        score = 0
        risk_factors = []
        
        age = features.get('age', 50)
        if age >= 75:
            score += 20
            risk_factors.append(f"Age {age} (Very High)")
        elif age >= 65:
            score += 15
            risk_factors.append(f"Age {age} (High)")
        elif age >= 55:
            score += 10
            risk_factors.append(f"Age {age} (Moderate)")
        
        sbp = features.get('systolic_bp', 120)
        if sbp >= 180:
            score += 20
            risk_factors.append(f"BP {sbp} (Stage 2 HTN)")
        elif sbp >= 160:
            score += 15
            risk_factors.append(f"BP {sbp} (Stage 2 HTN)")
        elif sbp >= 140:
            score += 10
            risk_factors.append(f"BP {sbp} (Stage 1 HTN)")
        
        chol = features.get('cholesterol', 200)
        if chol >= 280:
            score += 15
            risk_factors.append(f"Cholesterol {chol} (Very High)")
        elif chol >= 240:
            score += 10
            risk_factors.append(f"Cholesterol {chol} (High)")
        
        if features.get('smoking'):
            score += 15
            risk_factors.append("Current Smoker")
        
        if features.get('diabetes'):
            score += 15
            risk_factors.append("Diabetes")
        
        if features.get('sex') == 1:
            score += 5
            risk_factors.append("Male Sex")
        
        bmi = features.get('bmi', 25)
        if bmi >= 30:
            score += 10
            risk_factors.append(f"BMI {bmi} (Obese)")
        elif bmi >= 25:
            score += 5
            risk_factors.append(f"BMI {bmi} (Overweight)")
        
        # Calculate category - lowered thresholds for better sensitivity
        if score >= 45:
            category = 'HIGH'
        elif score >= 20:
            category = 'MODERATE'
        else:
            category = 'LOW'
        
        risk_percentage = min(score, 100)
        
        return {
            'risk_score': score / 100,
            'risk_percentage': risk_percentage,
            'risk_category': category,
            'confidence': 0.75,
            'detection_result': None,
            'detection_probability': None,
            'clinical_score': score,
            'clinical_max_score': 100,
            'recommendations': self._get_recommendations_by_category(category),
            'risk_factors': risk_factors,
            'model_used': 'clinical'
        }
    
    def _get_recommendations(self, result: Dict) -> str:
        """Generate recommendations based on results."""
        category = result.get('prediction', {}).get('risk_level', '').upper()
        if 'HIGH' in category:
            return self._get_recommendations_by_category('HIGH')
        elif 'MODERATE' in category:
            return self._get_recommendations_by_category('MODERATE')
        return self._get_recommendations_by_category('LOW')
    
    def _get_recommendations_by_category(self, category: str) -> str:
        """Get recommendations by category."""
        recs = {
            'HIGH': """🔴 HIGH RISK - Immediate Action Required:
• Schedule cardiology consultation within 2 weeks
• Begin aggressive risk factor management
• Consider statin therapy if not already prescribed
• Blood pressure medication review
• Lifestyle modifications: diet, exercise, smoking cessation
• Regular monitoring every 3 months""",
            
            'MODERATE': """🟡 MODERATE RISK - Lifestyle Modification Recommended:
• Schedule follow-up with primary care physician
• Implement heart-healthy diet (Mediterranean, DASH)
• Regular aerobic exercise (150 min/week)
• Monitor blood pressure and cholesterol
• Consider lifestyle modifications before medication
• Re-evaluate in 6 months""",
            
            'LOW': """🟢 LOW RISK - Maintain Healthy Lifestyle:
• Continue current healthy lifestyle
• Annual cardiovascular screening
• Regular exercise and balanced diet
• Avoid tobacco and limit alcohol
• Stress management techniques
• Re-evaluate annually"""
        }
        return recs.get(category, recs['MODERATE'])
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document using OCR for maximum extraction.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with OCR results and extracted fields
        """
        # Lazy load pipeline on first use
        self._ensure_loaded()
        print(f"[DEBUG] process_document: loaded={self.loaded}, pipeline={self.pipeline is not None}, file={file_path}")
        
        # Always use DualModelPipeline for consistent results
        if self.loaded and self.pipeline:
            try:
                print(f"[DEBUG] Using DualModelPipeline for extraction...")
                result = self.pipeline.process_document(file_path)
                print(f"[DEBUG] Pipeline result keys: {result.keys() if result else 'None'}")
                # DualModelPipeline returns result['ocr']['fields'], not result['extracted_data']['fields']
                ocr_data = result.get('ocr', {})
                print(f"[DEBUG] OCR data keys: {ocr_data.keys() if ocr_data else 'None'}")
                fields = ocr_data.get('fields', {})
                print(f"[DEBUG] Extracted fields count: {len(fields)}")
                return {
                    'text': ocr_data.get('text', ''),
                    'confidence': ocr_data.get('confidence', 0),
                    'method': ocr_data.get('method', 'ocr'),
                    'fields': fields,
                    'quality': ocr_data.get('quality', 'HIGH' if len(fields) >= 10 else 'MEDIUM' if len(fields) >= 5 else 'LOW'),
                    'num_fields': len(fields)
                }
            except Exception as e:
                print(f"[DEBUG] Pipeline extraction error: {e}")
                import traceback
                traceback.print_exc()
                return {'error': str(e), 'fields': {}, 'confidence': 0, 'quality': 'FAILED'}
        
        # Fallback to UltraOCR if pipeline not available
        try:
            from ultra_ocr import UltraOCR
            ocr = UltraOCR(verbose=False)
            result = ocr.extract_from_file(file_path)
            
            return {
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0),
                'method': result.get('method', 'ocr'),
                'fields': result.get('fields', {}),
                'quality': 'HIGH' if result.get('num_fields', 0) >= 10 else 'MEDIUM' if result.get('num_fields', 0) >= 5 else 'LOW',
                'num_fields': result.get('num_fields', 0)
            }
        except Exception as e:
            return {
                'error': str(e),
                'text': '',
                'confidence': 0,
                'method': 'failed',
                'fields': {},
                'quality': 'FAILED'
            }


class UnitConverter:
    """Unit conversion utility."""
    
    CONVERSIONS = {
        # Cholesterol
        ('mg/dL', 'mmol/L'): lambda x: x / 38.67,
        ('mmol/L', 'mg/dL'): lambda x: x * 38.67,
        
        # Glucose
        ('mg/dL_glucose', 'mmol/L'): lambda x: x / 18.02,
        ('mmol/L', 'mg/dL_glucose'): lambda x: x * 18.02,
        
        # Weight
        ('kg', 'lbs'): lambda x: x * 2.205,
        ('lbs', 'kg'): lambda x: x / 2.205,
        
        # Height
        ('cm', 'in'): lambda x: x / 2.54,
        ('in', 'cm'): lambda x: x * 2.54,
        ('cm', 'ft'): lambda x: x / 30.48,
        ('ft', 'cm'): lambda x: x * 30.48,
        
        # Temperature
        ('C', 'F'): lambda x: (x * 9/5) + 32,
        ('F', 'C'): lambda x: (x - 32) * 5/9,
    }
    
    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value between units."""
        if from_unit == to_unit:
            return value
        
        key = (from_unit, to_unit)
        if key in self.CONVERSIONS:
            return self.CONVERSIONS[key](value)
        
        raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")
    
    def get_supported_conversions(self) -> list:
        """Get list of supported conversions."""
        return list(self.CONVERSIONS.keys())



# Global instances - load pipeline immediately for Django views
ml_service = MLService()
ml_service._load_pipeline()  # Load eagerly for module import
unit_converter = UnitConverter()

