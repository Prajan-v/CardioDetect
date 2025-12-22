"""
Enhanced OCR Extractor with:
1. RapidFuzz fuzzy key matching
2. Pydantic field validation
3. Cross-field consistency checks
4. Confidence thresholding
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, validator, ValidationError
from rapidfuzz import fuzz, process


# =============================================================================
# Key Aliases for Fuzzy Matching
# =============================================================================
KEY_ALIASES = {
    'age': ['age', 'patient age', 'years', 'yrs', 'age years', 'patient\'s age'],
    'sex': ['sex', 'gender', 'male/female', 'm/f'],
    'systolic_bp': ['systolic', 'systolic bp', 'systolic blood pressure', 'sbp', 'sys bp', 'sys'],
    'diastolic_bp': ['diastolic', 'diastolic bp', 'diastolic blood pressure', 'dbp', 'dia bp', 'dia'],
    'total_cholesterol': ['total cholesterol', 'cholesterol', 'tc', 'total chol', 'chol', 'serum cholesterol'],
    'hdl_cholesterol': ['hdl', 'hdl cholesterol', 'hdl-c', 'high density', 'good cholesterol'],
    'ldl_cholesterol': ['ldl', 'ldl cholesterol', 'ldl-c', 'low density', 'bad cholesterol'],
    'triglycerides': ['triglycerides', 'trig', 'tg', 'trigly'],
    'fasting_glucose': ['fasting glucose', 'fbs', 'fasting blood sugar', 'glucose', 'blood sugar', 'fasting bg'],
    'bmi': ['bmi', 'body mass index', 'body mass'],
    'heart_rate': ['heart rate', 'hr', 'pulse', 'pulse rate', 'bpm'],
    'smoking': ['smoking', 'smoker', 'tobacco', 'cigarette', 'current smoker', 'smokes'],
    'diabetes': ['diabetes', 'diabetic', 'dm', 'type 2 diabetes', 'type 1 diabetes'],
    'hypertension': ['hypertension', 'htn', 'high blood pressure', 'high bp'],
    'bp_meds': ['bp medication', 'antihypertensive', 'bp meds', 'blood pressure medication'],
    # UCI stress test fields
    'cp': ['chest pain', 'chest pain type', 'cp', 'angina type'],
    'thalach': ['max heart rate', 'maximum heart rate', 'thalach', 'max hr', 'peak hr'],
    'exang': ['exercise angina', 'exang', 'exercise induced angina', 'angina on exertion'],
    'oldpeak': ['st depression', 'oldpeak', 'st segment', 'exercise st'],
    'slope': ['st slope', 'slope', 'st segment slope'],
    'ca': ['major vessels', 'ca', 'number of vessels', 'vessels colored'],
    'thal': ['thalassemia', 'thal', 'thallium', 'thallium test'],
    'restecg': ['resting ecg', 'restecg', 'ecg', 'ekg', 'resting electrocardiogram'],
}


# =============================================================================
# Pydantic Validation Models
# =============================================================================
class MedicalFieldsModel(BaseModel):
    """Pydantic model for medical field validation."""
    age: Optional[int] = None
    sex: Optional[str] = None
    sex_code: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    total_cholesterol: Optional[int] = None
    hdl_cholesterol: Optional[int] = None
    ldl_cholesterol: Optional[int] = None
    triglycerides: Optional[int] = None
    fasting_glucose: Optional[int] = None
    bmi: Optional[float] = None
    heart_rate: Optional[int] = None
    smoking: Optional[int] = None
    diabetes: Optional[int] = None
    hypertension: Optional[int] = None
    bp_meds: Optional[int] = None
    cp: Optional[int] = None
    thalach: Optional[int] = None
    exang: Optional[int] = None
    oldpeak: Optional[float] = None
    slope: Optional[int] = None
    ca: Optional[int] = None
    thal: Optional[int] = None
    restecg: Optional[int] = None
    
    @validator('age')
    def validate_age(cls, v):
        if v is not None and not (0 < v <= 120):
            raise ValueError(f'Age {v} out of valid range (1-120)')
        return v
    
    @validator('systolic_bp')
    def validate_systolic(cls, v):
        if v is not None and not (50 <= v <= 300):
            raise ValueError(f'Systolic BP {v} out of valid range (50-300)')
        return v
    
    @validator('diastolic_bp')
    def validate_diastolic(cls, v):
        if v is not None and not (30 <= v <= 200):
            raise ValueError(f'Diastolic BP {v} out of valid range (30-200)')
        return v
    
    @validator('total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol')
    def validate_cholesterol(cls, v):
        if v is not None and not (20 <= v <= 600):
            raise ValueError(f'Cholesterol {v} out of valid range (20-600)')
        return v
    
    @validator('triglycerides')
    def validate_triglycerides(cls, v):
        if v is not None and not (20 <= v <= 1000):
            raise ValueError(f'Triglycerides {v} out of valid range (20-1000)')
        return v
    
    @validator('fasting_glucose')
    def validate_glucose(cls, v):
        if v is not None and not (30 <= v <= 500):
            raise ValueError(f'Glucose {v} out of valid range (30-500)')
        return v
    
    @validator('bmi')
    def validate_bmi(cls, v):
        if v is not None and not (10 <= v <= 70):
            raise ValueError(f'BMI {v} out of valid range (10-70)')
        return v
    
    @validator('heart_rate', 'thalach')
    def validate_heart_rate(cls, v):
        if v is not None and not (30 <= v <= 250):
            raise ValueError(f'Heart rate {v} out of valid range (30-250)')
        return v
    
    class Config:
        extra = 'allow'  # Allow extra fields


def cross_field_validate(fields: Dict) -> Tuple[Dict, List[str]]:
    """
    Cross-field consistency checks.
    Returns validated fields and list of warnings.
    """
    warnings = []
    validated = fields.copy()
    
    # Systolic should be >= Diastolic
    sys = validated.get('systolic_bp')
    dia = validated.get('diastolic_bp')
    if sys and dia and sys < dia:
        warnings.append(f"Systolic ({sys}) < Diastolic ({dia}) - values may be swapped")
        validated['systolic_bp'], validated['diastolic_bp'] = dia, sys
    
    # LDL + HDL should be <= Total Cholesterol (approximately)
    tc = validated.get('total_cholesterol')
    hdl = validated.get('hdl_cholesterol')
    ldl = validated.get('ldl_cholesterol')
    if tc and hdl and ldl and (hdl + ldl) > tc * 1.2:
        warnings.append(f"HDL ({hdl}) + LDL ({ldl}) > Total Cholesterol ({tc}) - check values")
    
    # Age sanity check with BP
    age = validated.get('age')
    if age and age < 20 and sys and sys > 160:
        warnings.append(f"Age {age} with very high BP {sys} - unusual combination")
    
    return validated, warnings


# =============================================================================
# Fuzzy Key Matching
# =============================================================================
def fuzzy_match_key(text: str, threshold: int = 75) -> Optional[str]:
    """
    Match text to a standard field key using fuzzy matching.
    Returns the matched key or None if no match found.
    """
    text_clean = text.lower().strip()
    
    # Build a flat list of (alias, key) pairs
    alias_to_key = {}
    for key, aliases in KEY_ALIASES.items():
        for alias in aliases:
            alias_to_key[alias] = key
    
    # Find best match
    result = process.extractOne(
        text_clean, 
        list(alias_to_key.keys()), 
        scorer=fuzz.ratio
    )
    
    if result and result[1] >= threshold:
        return alias_to_key[result[0]]
    
    return None


def extract_value_near_key(text: str, key_position: int, max_window: int = 50) -> Optional[str]:
    """
    Extract a numeric value near the found key position.
    Looks for values within a window after the key.
    """
    window = text[key_position:key_position + max_window]
    
    # Look for number patterns
    number_patterns = [
        r'[\d]+\.[\d]+',  # Decimal
        r'[\d]+',         # Integer
    ]
    
    for pattern in number_patterns:
        match = re.search(pattern, window)
        if match:
            return match.group()
    
    return None


# =============================================================================
# Enhanced Field Parser
# =============================================================================
def parse_fields_fuzzy(text: str) -> Dict[str, Any]:
    """
    Parse medical fields using fuzzy key matching.
    More robust than pure regex approach.
    """
    fields = {}
    text_lower = text.lower()
    
    # Split into lines and process each
    lines = text.split('\n')
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 3:
            continue
        
        # Try to find key-value patterns
        # Pattern: "Key: Value" or "Key = Value" or "Key Value"
        kv_patterns = [
            r'^([^:=]+)[:\s=]+(.+)$',
            r'^(.+?)\s+(\d+\.?\d*)\s*(?:mg|mmHg|bpm|%|years?)?$',
        ]
        
        for pattern in kv_patterns:
            match = re.match(pattern, line_clean, re.IGNORECASE)
            if match:
                potential_key = match.group(1).strip()
                potential_value = match.group(2).strip()
                
                # Try fuzzy matching on the key
                matched_key = fuzzy_match_key(potential_key)
                if matched_key and matched_key not in fields:
                    # Extract numeric value
                    num_match = re.search(r'(\d+\.?\d*)', potential_value)
                    if num_match:
                        value = num_match.group(1)
                        try:
                            if '.' in value:
                                fields[matched_key] = float(value)
                            else:
                                fields[matched_key] = int(value)
                        except ValueError:
                            pass
                break
    
    # Also run the original regex patterns for backup
    fields = _supplement_with_regex(text, fields)
    
    return fields


def _supplement_with_regex(text: str, existing_fields: Dict) -> Dict:
    """Supplement fuzzy results with traditional regex patterns."""
    patterns = {
        'age': [r'age[:\s]+(\d{1,3})', r'(\d{1,3})\s*(?:years?|yrs?)\s*(?:old)?'],
        'systolic_bp': [r'(?:systolic|sbp|sys)[:\s]+(\d{2,3})', r'(?:bp|blood pressure)[:\s]+(\d{2,3})/'],
        'diastolic_bp': [r'(?:diastolic|dbp|dia)[:\s]+(\d{2,3})', r'/(\d{2,3})\s*(?:mmHg)?'],
        'total_cholesterol': [r'(?:total\s*)?cholesterol[:\s]+(\d{2,3})', r'(?:tc|chol)[:\s]+(\d{2,3})'],
        'hdl_cholesterol': [r'hdl[:\s]+(\d{2,3})'],
        'ldl_cholesterol': [r'ldl[:\s]+(\d{2,3})'],
        'triglycerides': [r'triglycerides?[:\s]+(\d{2,4})'],
        'fasting_glucose': [r'(?:fasting\s*)?(?:glucose|blood\s*sugar|fbs)[:\s]+(\d{2,3})'],
        'bmi': [r'bmi[:\s]+(\d{1,2}\.?\d*)'],
        'heart_rate': [r'(?:heart\s*rate|hr|pulse)[:\s]+(\d{2,3})'],
        'smoking': [r'smoking[:\s]*(yes|no|true|false|1|0)', r'(?:current\s*)?smoker[:\s]*(yes|no)'],
        'diabetes': [r'diabetes[:\s]*(yes|no|true|false|1|0)', r'diabetic[:\s]*(yes|no)'],
    }
    
    text_lower = text.lower()
    
    for field, pats in patterns.items():
        if field in existing_fields:
            continue  # Already found via fuzzy match
        
        for pat in pats:
            match = re.search(pat, text_lower)
            if match:
                value = match.group(1)
                try:
                    if field in ['smoking', 'diabetes']:
                        existing_fields[field] = 1 if value.lower() in ['yes', 'true', '1'] else 0
                    elif '.' in value:
                        existing_fields[field] = float(value)
                    else:
                        existing_fields[field] = int(value)
                    break
                except ValueError:
                    pass
    
    return existing_fields


# =============================================================================
# Main Extractor Function
# =============================================================================
def extract_and_validate(text: str) -> Dict[str, Any]:
    """
    Main extraction function with fuzzy matching and validation.
    
    Returns:
        Dict with 'fields', 'warnings', 'validation_errors', 'confidence'
    """
    # Step 1: Extract fields using fuzzy matching
    raw_fields = parse_fields_fuzzy(text)
    
    # Step 2: Cross-field validation
    validated_fields, xfield_warnings = cross_field_validate(raw_fields)
    
    # Step 3: Pydantic validation
    validation_errors = []
    try:
        model = MedicalFieldsModel(**validated_fields)
        validated_fields = model.dict(exclude_none=True)
    except ValidationError as e:
        for error in e.errors():
            field = error['loc'][0]
            msg = error['msg']
            validation_errors.append(f"{field}: {msg}")
            # Remove invalid field
            if field in validated_fields:
                del validated_fields[field]
    
    # Step 4: Calculate confidence based on fields found and validation
    total_possible = 15  # Key medical fields
    found = len(validated_fields)
    error_penalty = len(validation_errors) * 0.05
    confidence = max(0, min(1, (found / total_possible) - error_penalty))
    
    return {
        'fields': validated_fields,
        'num_fields': len(validated_fields),
        'warnings': xfield_warnings,
        'validation_errors': validation_errors,
        'confidence': confidence,
        'quality': 'HIGH' if confidence >= 0.7 else 'MEDIUM' if confidence >= 0.4 else 'LOW'
    }


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    test_text = """
    Patient Information
    Age: 55 years
    Sex: Male
    
    Vital Signs:
    Blood Pressure: 145/92 mmHg
    Heart Rate: 78 bpm
    
    Laboratory Results:
    Total Cholesterol: 245 mg/dL
    HDL: 42 mg/dL
    LDL: 165 mg/dL
    Triglycerides: 180 mg/dL
    Fasting Glucose: 118 mg/dL
    
    Risk Factors:
    Smoking: Yes
    Diabetes: No
    BMI: 28.5
    """
    
    result = extract_and_validate(test_text)
    print("Extracted Fields:")
    for k, v in result['fields'].items():
        print(f"  {k}: {v}")
    print(f"\nConfidence: {result['confidence']*100:.1f}%")
    print(f"Quality: {result['quality']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")
    if result['validation_errors']:
        print(f"Validation Errors: {result['validation_errors']}")
