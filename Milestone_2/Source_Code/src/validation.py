"""
Input Validation Module for CardioDetect

Validates input data for missing values, out-of-range values, and data quality.
Provides confidence scores and warnings before predictions.

Usage:
    from milestone_2.src.validation import InputValidator
    
    validator = InputValidator()
    result = validator.validate(patient_data)
    
    if result.is_valid:
        prediction = model.predict(patient_data)
    else:
        print(result.warnings)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    missing_critical: List[str]
    missing_optional: List[str]
    out_of_range: List[str]
    warnings: List[str]
    errors: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "missing_critical": self.missing_critical,
            "missing_optional": self.missing_optional,
            "out_of_range": self.out_of_range,
            "warnings": self.warnings,
            "errors": self.errors,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.is_valid and self.confidence >= 0.8:
            return f"✅ Data valid (confidence: {self.confidence:.0%})"
        elif self.is_valid:
            warnings_text = "; ".join(self.warnings[:2])
            return f"⚠️ Data valid with warnings (confidence: {self.confidence:.0%}): {warnings_text}"
        else:
            errors_text = "; ".join(self.errors[:2])
            return f"❌ Data invalid: {errors_text}"


class InputValidator:
    """
    Validates input data for CardioDetect risk predictions.
    
    Checks for:
    - Missing critical features (Age, SBP, Total Cholesterol, Smoking, Diabetes)
    - Missing optional features (HDL, BMI, etc.)
    - Out-of-range values (physiologically impossible)
    - Data type issues
    """
    
    # Critical features for Framingham risk calculation
    CRITICAL_FEATURES = [
        'age',
        'systolic_bp', 
        'total_cholesterol',
        'smoking',
        'diabetes',
    ]
    
    # Important but not critical
    IMPORTANT_FEATURES = [
        'hdl_cholesterol',
        'sex',
        'bp_meds',
        'diastolic_bp',
    ]
    
    # Optional enrichment features
    OPTIONAL_FEATURES = [
        'bmi',
        'heart_rate',
        'fasting_glucose',
        'ldl_cholesterol',
    ]
    
    # Physiological ranges for validation
    VALID_RANGES = {
        'age': (18, 120),
        'systolic_bp': (60, 250),
        'diastolic_bp': (30, 150),
        'total_cholesterol': (80, 500),
        'hdl_cholesterol': (10, 150),
        'ldl_cholesterol': (20, 400),
        'fasting_glucose': (40, 500),
        'bmi': (10, 70),
        'heart_rate': (30, 220),
    }
    
    def __init__(
        self,
        max_missing_critical: int = 2,
        warn_on_imputation: bool = True,
    ):
        """
        Initialize validator.
        
        Args:
            max_missing_critical: Max allowed missing critical features
            warn_on_imputation: Whether to warn when imputation will be used
        """
        self.max_missing_critical = max_missing_critical
        self.warn_on_imputation = warn_on_imputation
    
    def validate(
        self, 
        data: pd.DataFrame,
        raise_on_invalid: bool = False
    ) -> ValidationResult:
        """
        Validate input data.
        
        Args:
            data: Input DataFrame (single row or batch)
            raise_on_invalid: If True, raise exception for invalid data
            
        Returns:
            ValidationResult with detailed findings
        """
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        
        missing_critical = []
        missing_optional = []
        out_of_range = []
        warnings = []
        errors = []
        
        # Check for missing critical features
        for feature in self.CRITICAL_FEATURES:
            if feature not in data.columns:
                missing_critical.append(feature)
            elif data[feature].isna().any():
                missing_critical.append(feature)
        
        # Check for missing important features
        for feature in self.IMPORTANT_FEATURES:
            if feature not in data.columns:
                missing_optional.append(feature)
            elif data[feature].isna().any():
                missing_optional.append(feature)
        
        # Check for missing optional features
        for feature in self.OPTIONAL_FEATURES:
            if feature not in data.columns or data[feature].isna().any():
                missing_optional.append(feature)
        
        # Check for out-of-range values
        for feature, (min_val, max_val) in self.VALID_RANGES.items():
            if feature in data.columns:
                values = data[feature].dropna()
                if len(values) > 0:
                    if (values < min_val).any() or (values > max_val).any():
                        out_of_range.append(f"{feature} (valid: {min_val}-{max_val})")
        
        # Generate warnings
        if missing_critical:
            warnings.append(
                f"Missing critical features will be imputed: {', '.join(missing_critical)}"
            )
        
        if missing_optional:
            if self.warn_on_imputation:
                warnings.append(
                    f"Missing optional features: {', '.join(missing_optional[:3])}"
                    + (f" (+{len(missing_optional)-3} more)" if len(missing_optional) > 3 else "")
                )
        
        if out_of_range:
            warnings.append(
                f"Out-of-range values detected: {', '.join(out_of_range)}"
            )
        
        # Generate errors
        if len(missing_critical) > self.max_missing_critical:
            errors.append(
                f"Too many critical features missing ({len(missing_critical)} > {self.max_missing_critical}): "
                f"{', '.join(missing_critical)}"
            )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            missing_critical, missing_optional, out_of_range
        )
        
        # Determine validity
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            missing_critical=missing_critical,
            missing_optional=missing_optional,
            out_of_range=out_of_range,
            warnings=warnings,
            errors=errors,
        )
        
        if raise_on_invalid and not is_valid:
            raise ValueError(f"Invalid input data: {'; '.join(errors)}")
        
        return result
    
    def _calculate_confidence(
        self,
        missing_critical: List[str],
        missing_optional: List[str],
        out_of_range: List[str],
    ) -> float:
        """Calculate confidence score based on data quality."""
        confidence = 1.0
        
        # Deduct for missing critical features (heavy penalty)
        confidence -= len(missing_critical) * 0.15
        
        # Deduct for missing important features (moderate penalty)
        important_missing = [f for f in missing_optional if f in self.IMPORTANT_FEATURES]
        confidence -= len(important_missing) * 0.08
        
        # Deduct for missing optional features (light penalty)
        optional_only = [f for f in missing_optional if f not in self.IMPORTANT_FEATURES]
        confidence -= len(optional_only) * 0.02
        
        # Deduct for out-of-range values
        confidence -= len(out_of_range) * 0.10
        
        return max(0.0, min(1.0, confidence))
    
    def validate_single(
        self, 
        patient_dict: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate a single patient record from dictionary.
        
        Args:
            patient_dict: Dictionary with feature names as keys
            
        Returns:
            ValidationResult
        """
        df = pd.DataFrame([patient_dict])
        return self.validate(df)
    
    def get_completeness_report(
        self, 
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate detailed completeness report for a dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with completeness statistics
        """
        report = {
            "total_samples": len(data),
            "features_checked": {},
            "overall_completeness": 0.0,
        }
        
        all_features = (
            self.CRITICAL_FEATURES + 
            self.IMPORTANT_FEATURES + 
            self.OPTIONAL_FEATURES
        )
        
        total_present = 0
        total_possible = 0
        
        for feature in all_features:
            if feature in data.columns:
                present = data[feature].notna().sum()
                total = len(data)
                pct = present / total * 100
                
                report["features_checked"][feature] = {
                    "present": int(present),
                    "missing": int(total - present),
                    "completeness_pct": round(pct, 1),
                    "category": (
                        "critical" if feature in self.CRITICAL_FEATURES
                        else "important" if feature in self.IMPORTANT_FEATURES
                        else "optional"
                    )
                }
                
                total_present += present
                total_possible += total
            else:
                report["features_checked"][feature] = {
                    "present": 0,
                    "missing": len(data),
                    "completeness_pct": 0.0,
                    "category": (
                        "critical" if feature in self.CRITICAL_FEATURES
                        else "important" if feature in self.IMPORTANT_FEATURES
                        else "optional"
                    ),
                    "note": "Feature column not found"
                }
                total_possible += len(data)
        
        report["overall_completeness"] = (
            total_present / total_possible * 100 if total_possible > 0 else 0
        )
        
        return report


def validate_and_predict(
    model,
    data: pd.DataFrame,
    validator: Optional[InputValidator] = None,
    min_confidence: float = 0.5,
) -> Tuple[np.ndarray, ValidationResult]:
    """
    Validate input and make predictions with confidence check.
    
    Args:
        model: Trained sklearn model/pipeline
        data: Input DataFrame
        validator: InputValidator instance (creates default if None)
        min_confidence: Minimum confidence required for prediction
        
    Returns:
        Tuple of (predictions, validation_result)
        
    Raises:
        ValueError: If confidence below minimum
    """
    if validator is None:
        validator = InputValidator()
    
    result = validator.validate(data)
    
    if result.confidence < min_confidence:
        raise ValueError(
            f"Data quality too low for reliable prediction. "
            f"Confidence: {result.confidence:.0%}, Required: {min_confidence:.0%}. "
            f"Missing critical features: {', '.join(result.missing_critical)}"
        )
    
    predictions = model.predict(data)
    
    return predictions, result


# Convenience function for quick validation
def quick_validate(data: pd.DataFrame) -> None:
    """
    Quick validation with printed output.
    
    Args:
        data: Input DataFrame to validate
    """
    validator = InputValidator()
    result = validator.validate(data)
    
    print("=" * 50)
    print("INPUT VALIDATION REPORT")
    print("=" * 50)
    print(f"Status: {result.summary()}")
    print(f"Confidence: {result.confidence:.0%}")
    print()
    
    if result.missing_critical:
        print(f"⚠️  Missing Critical: {', '.join(result.missing_critical)}")
    if result.missing_optional:
        print(f"ℹ️  Missing Optional: {', '.join(result.missing_optional[:5])}")
    if result.out_of_range:
        print(f"⚠️  Out of Range: {', '.join(result.out_of_range)}")
    
    if result.warnings:
        print()
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")
    
    if result.errors:
        print()
        print("Errors:")
        for e in result.errors:
            print(f"  ❌ {e}")
    
    print("=" * 50)
