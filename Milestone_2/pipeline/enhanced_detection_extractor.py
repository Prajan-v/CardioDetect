"""
Enhanced OCR Field Extractor with Clinical Defaults
Improves detection model accuracy by better handling missing fields
"""

import sys
sys.path.insert(0, '/Users/prajanv/CardioDetect/Milestone_2')

import joblib
import numpy as np
from pathlib import Path
from pipeline.integrated_pipeline import EnhancedMedicalOCR

class EnhancedDetectionExtractor:
    """
    Enhanced OCR extractor that:
    1. Extracts all available fields
    2. Maps to model features with clinical defaults
    3. Uses age/sex-based imputation for missing values
    """
    
    # Clinical defaults based on population statistics
    CLINICAL_DEFAULTS = {
        'age': 55,  # Mean age in dataset
        'sex': 1,  # Male (higher risk, conservative default)
        'chest_pain_type': 0,  # ASY (Asymptomatic) - most common in heart disease
        'resting_bp': 132,  # Mean from dataset
        'cholesterol': 240,  # Using median of non-zero values
        'fasting_bs': 0,  # No diabetes by default
        'resting_ecg': 0,  # Normal
        'max_hr': 137,  # Mean from dataset
        'exercise_angina': 0,  # No by default
        'oldpeak': 0.9,  # Mean from dataset
        'st_slope': 1,  # Flat (most predictive of disease)
    }
    
    # Age-based MaxHR estimates (220 - age rule)
    AGE_MAXHR_MAP = {
        (20, 30): 175, (30, 40): 165, (40, 50): 155,
        (50, 60): 145, (60, 70): 130, (70, 80): 115, (80, 100): 100
    }
    
    # Sex-based cholesterol estimates (mg/dL)
    SEX_CHOLESTEROL_MAP = {
        'male': 230,  # Higher for males
        'female': 220,  # Slightly lower for females
    }
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.ocr = EnhancedMedicalOCR(verbose=False)
        
    def log(self, msg):
        if self.verbose:
            print(msg)
    
    def get_age_based_maxhr(self, age):
        """Estimate max heart rate based on age"""
        for (low, high), hr in self.AGE_MAXHR_MAP.items():
            if low <= age < high:
                return hr
        return 140  # Default
    
    def extract_and_map(self, file_path):
        """Extract OCR fields and map to model features with intelligent defaults"""
        
        self.log(f"\n📄 Processing: {file_path}")
        
        # Run OCR
        result = self.ocr.extract_from_file(file_path)
        fields = result['fields']
        
        self.log("\n📊 Raw OCR Extraction:")
        for k, v in sorted(fields.items()):
            if v is not None:
                self.log(f"   {k}: {v}")
        
        # Build feature dictionary with intelligent defaults
        features = {}
        missing_fields = []
        
        # Age
        if fields.get('age'):
            features['Age'] = fields['age']
        else:
            features['Age'] = self.CLINICAL_DEFAULTS['age']
            missing_fields.append('age')
        
        # Sex
        sex_val = fields.get('sex', '')
        if isinstance(sex_val, str) and sex_val.lower() in ['male', 'm']:
            features['Sex'] = 1
        elif isinstance(sex_val, str) and sex_val.lower() in ['female', 'f']:
            features['Sex'] = 0
        elif fields.get('sex_code') is not None:
            features['Sex'] = fields['sex_code']
        else:
            features['Sex'] = self.CLINICAL_DEFAULTS['sex']
            missing_fields.append('sex')
        
        # Chest Pain Type (Critical for detection!)
        # Map: ASY=0, ATA=1, NAP=2, TA=3
        cp = fields.get('cp')
        if cp is not None:
            cp_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}  # 4 sometimes used for ASY
            features['ChestPainType'] = cp_map.get(cp, 0)
        else:
            # If age > 60 or has other risk factors, assume ASY (more conservative)
            features['ChestPainType'] = 0  # ASY - common in actual heart disease
            missing_fields.append('chest_pain_type')
        
        # Resting BP
        if fields.get('systolic_bp'):
            features['RestingBP'] = fields['systolic_bp']
        else:
            features['RestingBP'] = self.CLINICAL_DEFAULTS['resting_bp']
            missing_fields.append('systolic_bp')
        
        # Cholesterol - CRITICAL: Handle zeros!
        chol = fields.get('total_cholesterol', 0)
        if chol and chol > 50:  # Valid cholesterol value
            features['Cholesterol'] = chol
        else:
            # Use sex and age-adjusted estimate
            sex_str = 'male' if features['Sex'] == 1 else 'female'
            base_chol = self.SEX_CHOLESTEROL_MAP[sex_str]
            age_adjustment = (features['Age'] - 50) * 0.5  # Cholesterol increases with age
            features['Cholesterol'] = base_chol + age_adjustment
            missing_fields.append('cholesterol')
        
        # Fasting Blood Sugar / Diabetes
        if fields.get('diabetes') is not None:
            features['FastingBS'] = 1 if fields['diabetes'] else 0
        elif fields.get('fasting_glucose'):
            # Fasting glucose > 126 indicates diabetes
            features['FastingBS'] = 1 if fields['fasting_glucose'] > 126 else 0
        else:
            features['FastingBS'] = self.CLINICAL_DEFAULTS['fasting_bs']
            missing_fields.append('fasting_bs')
        
        # Resting ECG
        if fields.get('restecg') is not None:
            features['RestingECG'] = fields['restecg']
        else:
            features['RestingECG'] = self.CLINICAL_DEFAULTS['resting_ecg']
            missing_fields.append('resting_ecg')
        
        # Max Heart Rate - Use thalach if available, else estimate from age
        if fields.get('thalach'):
            features['MaxHR'] = fields['thalach']
        elif fields.get('heart_rate'):
            # Resting heart rate is NOT max HR, need to estimate
            features['MaxHR'] = self.get_age_based_maxhr(features['Age'])
            missing_fields.append('max_hr (estimated from age)')
        else:
            features['MaxHR'] = self.get_age_based_maxhr(features['Age'])
            missing_fields.append('max_hr')
        
        # Exercise Angina (Critical feature!)
        if fields.get('exang') is not None:
            features['ExerciseAngina'] = fields['exang']
        else:
            # Conservative: assume no unless evidence
            features['ExerciseAngina'] = self.CLINICAL_DEFAULTS['exercise_angina']
            missing_fields.append('exercise_angina')
        
        # Oldpeak (ST depression)
        if fields.get('oldpeak') is not None:
            features['Oldpeak'] = float(fields['oldpeak'])
        else:
            features['Oldpeak'] = self.CLINICAL_DEFAULTS['oldpeak']
            missing_fields.append('oldpeak')
        
        # ST Slope (CRITICAL - highest correlation with heart disease!)
        # Up=0, Flat=1, Down=2
        if fields.get('slope') is not None:
            features['ST_Slope'] = fields['slope']
        else:
            # Conservative: assume Flat (associated with disease)
            features['ST_Slope'] = 1
            missing_fields.append('st_slope')
        
        self.log(f"\n⚠️  Missing Fields ({len(missing_fields)}):")
        for f in missing_fields:
            self.log(f"   - {f}")
        
        self.log("\n🔧 Mapped Features:")
        for k, v in features.items():
            self.log(f"   {k}: {v}")
        
        return {
            'features': features,
            'fields': fields,
            'missing_count': len(missing_fields),
            'missing_fields': missing_fields,
            'confidence': result.get('confidence', 0.5),
        }


def test_enhanced_extractor(file_path):
    """Test the enhanced extractor and compare with old model"""
    
    print("=" * 70)
    print("ENHANCED OCR EXTRACTOR TEST")
    print("=" * 70)
    
    extractor = EnhancedDetectionExtractor(verbose=True)
    result = extractor.extract_and_map(file_path)
    
    features = result['features']
    feature_order = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                     'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    
    # Load calibrated model
    model_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/detection_calibrated')
    if model_dir.exists():
        model = joblib.load(model_dir / 'detection_lgbm_calibrated.pkl')
        scaler = joblib.load(model_dir / 'detection_scaler.pkl')
        model_name = "Calibrated LightGBM"
    else:
        # Fallback to original
        model_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/models/Final_models/detection')
        model = joblib.load(model_dir / 'detection_best.pkl')
        scaler = joblib.load(model_dir / 'detection_scaler.pkl')
        model_name = "Original XGBoost"
    
    # Prepare feature vector
    X = np.array([[features[f] for f in feature_order]])
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    
    print("\n" + "=" * 70)
    print(f"🔬 DETECTION RESULT ({model_name})")
    print("=" * 70)
    print(f"\n   Prediction: {'❤️ HEART DISEASE DETECTED' if prediction == 1 else '✅ NO HEART DISEASE'}")
    print(f"\n   Confidence:")
    print(f"      No Disease:  {proba[0]:.1%}")
    print(f"      Disease:     {proba[1]:.1%}")
    print(f"\n   Data Quality: {result['missing_count']} fields imputed")
    
    if result['missing_count'] > 5:
        print("   ⚠️  Low confidence due to many missing fields")
    
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    # Test with the medical report
    test_file = '/Users/prajanv/CardioDetect/Milestone_2/Medical_report/Synthetic_report/SYN-012.png'
    test_enhanced_extractor(test_file)
