"""Production Pipeline: End-to-End OCR → Risk Prediction.

This module provides the complete integration:
- Document input (PDF, PNG, JPG, photo)
- Advanced OCR extraction with confidence
- Complete feature engineering
- Risk prediction with uncertainty and explanation
- Single API endpoint returning JSON

Usage:
    from src.production_pipeline import predict_risk
    result = predict_risk("/path/to/document.pdf")
    print(result)  # JSON with risk, confidence, explanation
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import our production modules
from src.production_ocr import ProductionOCR, OCRResult
from src.production_model import ProductionModel, RiskPrediction


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    # Status
    success: bool
    error: Optional[str] = None
    
    # Risk Prediction
    risk_level: str = "UNKNOWN"
    risk_probability: float = 0.0
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Explanation
    explanation: str = ""
    top_risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Flags
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    is_ood: bool = False
    
    # OCR Details
    ocr_success: bool = False
    ocr_confidence: float = 0.0
    ocr_quality: str = "unknown"
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    
    # Metadata
    processing_time_ms: float = 0.0
    timestamp: str = ""
    document_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "error": self.error,
            "risk": {
                "level": self.risk_level,
                "probability": round(self.risk_probability, 2),
                "confidence": round(self.confidence, 2),
                "probabilities": self.probabilities,
            },
            "explanation": self.explanation,
            "top_risk_factors": self.top_risk_factors,
            "flags": {
                "needs_review": self.needs_review,
                "review_reasons": self.review_reasons,
                "is_out_of_distribution": self.is_ood,
            },
            "ocr": {
                "success": self.ocr_success,
                "confidence": round(self.ocr_confidence, 2),
                "quality": self.ocr_quality,
                "extracted_fields": self.extracted_fields,
                "missing_fields": self.missing_fields,
            },
            "metadata": {
                "processing_time_ms": round(self.processing_time_ms, 2),
                "timestamp": self.timestamp,
                "document_path": self.document_path,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class ProductionPipeline:
    """Complete end-to-end pipeline for medical document risk prediction."""

    # Required fields for prediction
    REQUIRED_FIELDS = ["age", "systolic_bp", "total_cholesterol"]
    RECOMMENDED_FIELDS = ["fasting_glucose", "bmi", "sex"]
    OPTIONAL_FIELDS = ["diastolic_bp", "hdl", "ldl", "smoking", "diabetes", "heart_rate"]

    def __init__(
        self,
        verbose: bool = False,
        enable_shap: bool = False,
        use_easyocr: bool = False,  # Disabled by default for speed
    ):
        """Initialize the production pipeline.

        Args:
            verbose: Print debug information
            enable_shap: Enable SHAP explanations (slower but more accurate)
            use_easyocr: Use EasyOCR as fallback (requires easyocr package)
        """
        self.verbose = verbose

        # Initialize components
        logger.info("Initializing Production Pipeline...")

        self.ocr = ProductionOCR(
            use_easyocr=use_easyocr,
            verbose=verbose,
        )

        self.model = ProductionModel(
            enable_shap=enable_shap,
            verbose=verbose,
        )

        logger.info("Pipeline ready")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Pipeline] {msg}")

    def _ocr_to_patient_data(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Convert OCR result to patient data dictionary."""
        patient_data = {}

        # Map OCR fields to patient data
        field_mapping = {
            "age": "age",
            "sex": "sex",
            "systolic_bp": "systolic_bp",
            "diastolic_bp": "diastolic_bp",
            "bmi": "bmi",
            "total_cholesterol": "total_cholesterol",
            "hdl": "hdl",
            "ldl": "ldl",
            "triglycerides": "triglycerides",
            "fasting_glucose": "fasting_glucose",
            "hemoglobin": "hemoglobin",
            "wbc": "wbc",
            "rbc": "rbc",
            "platelet": "platelet",
            "smoking": "smoking",
            "diabetes": "diabetes",
            "heart_rate": "heart_rate",
        }

        for ocr_field, patient_field in field_mapping.items():
            if ocr_field in ocr_result.fields:
                patient_data[patient_field] = ocr_result.fields[ocr_field].value

        # Add OCR confidence for uncertainty calculation
        patient_data["ocr_confidence"] = ocr_result.overall_confidence

        return patient_data

    def _check_missing_fields(
        self,
        patient_data: Dict[str, Any],
    ) -> List[str]:
        """Check for missing required and recommended fields."""
        missing = []

        for field in self.REQUIRED_FIELDS:
            if field not in patient_data or patient_data[field] is None:
                missing.append(f"REQUIRED: {field}")

        for field in self.RECOMMENDED_FIELDS:
            if field not in patient_data or patient_data[field] is None:
                missing.append(f"recommended: {field}")

        return missing

    def predict_from_document(
        self,
        document_path: Union[str, Path],
        quality_hint: str = "auto",
    ) -> PipelineResult:
        """Run complete pipeline on a document.

        Args:
            document_path: Path to medical document (PDF, PNG, JPG)
            quality_hint: 'auto', 'clean', 'scanned', 'photo'

        Returns:
            PipelineResult with complete analysis
        """
        import time
        start_time = time.time()

        result = PipelineResult(
            success=False,
            timestamp=datetime.now().isoformat(),
            document_path=str(document_path),
        )

        try:
            path = Path(document_path)
            if not path.exists():
                result.error = f"File not found: {path}"
                return result

            self._log(f"Processing: {path.name}")

            # Step 1: OCR Extraction
            self._log("Running OCR...")
            ocr_result = self.ocr.extract(path, quality_hint=quality_hint)

            result.ocr_success = ocr_result.success
            result.ocr_confidence = ocr_result.overall_confidence
            result.ocr_quality = ocr_result.document_quality

            # Store extracted fields
            result.extracted_fields = {
                name: {
                    "value": f.value,
                    "confidence": f.confidence,
                }
                for name, f in ocr_result.fields.items()
            }

            if not ocr_result.success:
                result.error = "OCR extraction failed"
                result.needs_review = True
                result.review_reasons.append("OCR failed to extract key fields")
                result.processing_time_ms = (time.time() - start_time) * 1000
                return result

            # Step 2: Convert to patient data
            patient_data = self._ocr_to_patient_data(ocr_result)
            self._log(f"Extracted {len(patient_data)} fields")

            # Check missing fields
            result.missing_fields = self._check_missing_fields(patient_data)
            missing_required = [m.replace("REQUIRED: ", "") for m in result.missing_fields if "REQUIRED" in m]
            
            if missing_required:
                result.error = f"Necessary data missing: {', '.join(missing_required)}"
                result.needs_review = True
                result.review_reasons.append(f"Missing necessary data: {', '.join(missing_required)}")

            # Step 3: Risk Prediction
            self._log("Running risk prediction...")
            prediction = self.model.predict(patient_data)

            # Populate result
            result.success = True
            result.risk_level = prediction.risk_level
            result.risk_probability = prediction.risk_probability
            result.confidence = prediction.confidence
            result.probabilities = prediction.probabilities
            result.explanation = prediction.explanation
            result.top_risk_factors = prediction.top_risk_factors
            result.needs_review = prediction.needs_review
            result.review_reasons = prediction.review_reasons.copy()
            result.is_ood = prediction.is_ood

            # Add OCR-related review reasons
            if ocr_result.overall_confidence < 70:
                result.needs_review = True
                result.review_reasons.append(f"Low OCR confidence: {ocr_result.overall_confidence:.1f}%")

            if result.missing_fields:
                result.needs_review = True

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.needs_review = True
            result.review_reasons.append(f"Pipeline error: {e}")
            logger.exception("Pipeline error")

        result.processing_time_ms = (time.time() - start_time) * 1000
        self._log(f"Completed in {result.processing_time_ms:.1f}ms")

        return result

    def predict_from_data(
        self,
        patient_data: Dict[str, Any],
    ) -> PipelineResult:
        """Run prediction from pre-extracted patient data.

        Args:
            patient_data: Dictionary with patient fields

        Returns:
            PipelineResult with prediction
        """
        import time
        start_time = time.time()

        result = PipelineResult(
            success=False,
            timestamp=datetime.now().isoformat(),
            document_path="direct_input",
        )

        try:
            # Skip OCR, go directly to prediction
            result.ocr_success = True
            result.ocr_confidence = 100.0
            result.extracted_fields = patient_data

            # Check missing fields
            result.missing_fields = self._check_missing_fields(patient_data)

            # Run prediction
            prediction = self.model.predict(patient_data)

            result.success = True
            result.risk_level = prediction.risk_level
            result.risk_probability = prediction.risk_probability
            result.confidence = prediction.confidence
            result.probabilities = prediction.probabilities
            result.explanation = prediction.explanation
            result.top_risk_factors = prediction.top_risk_factors
            result.needs_review = prediction.needs_review
            result.review_reasons = prediction.review_reasons.copy()
            result.is_ood = prediction.is_ood

        except Exception as e:
            result.error = str(e)
            result.needs_review = True
            result.review_reasons.append(f"Prediction error: {e}")

        result.processing_time_ms = (time.time() - start_time) * 1000

        return result


# -----------------------------------------------------------------------------
# Convenience API
# -----------------------------------------------------------------------------

# Global pipeline instance (lazy initialized)
_pipeline: Optional[ProductionPipeline] = None


def get_pipeline(
    verbose: bool = False,
    enable_shap: bool = False,
) -> ProductionPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ProductionPipeline(
            verbose=verbose,
            enable_shap=enable_shap,
        )
    return _pipeline


def predict_risk(
    document_path: Union[str, Path],
    verbose: bool = False,
    return_json: bool = False,
) -> Union[PipelineResult, Dict[str, Any], str]:
    """Main API: Predict cardiovascular risk from a medical document.

    Args:
        document_path: Path to medical document (PDF, PNG, JPG, photo)
        verbose: Print debug information
        return_json: Return JSON string instead of PipelineResult

    Returns:
        PipelineResult, dict, or JSON string with:
        - risk_level: LOW, MEDIUM, HIGH
        - risk_probability: 0-100%
        - confidence: 0-100%
        - explanation: Human-readable explanation
        - top_risk_factors: List of contributing factors
        - needs_review: Whether manual review is recommended
        - review_reasons: Why review is needed
        - ocr details: Extraction quality and confidence
    """
    pipeline = get_pipeline(verbose=verbose)
    result = pipeline.predict_from_document(document_path)

    if return_json:
        return result.to_json()
    return result


def predict_risk_from_data(
    patient_data: Dict[str, Any],
    verbose: bool = False,
) -> PipelineResult:
    """Predict risk from pre-extracted patient data.

    Args:
        patient_data: Dictionary with patient fields:
            - age (required)
            - systolic_bp (required)
            - total_cholesterol (recommended)
            - fasting_glucose (recommended)
            - bmi (recommended)
            - sex (recommended)
            - diastolic_bp, hdl, ldl, smoking, diabetes, heart_rate (optional)

    Returns:
        PipelineResult with complete analysis
    """
    pipeline = get_pipeline(verbose=verbose)
    return pipeline.predict_from_data(patient_data)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python production_pipeline.py <document_path>")
        print("\nExample:")
        print("  python production_pipeline.py report.pdf")
        print("  python production_pipeline.py scan.png")
        print("  python production_pipeline.py photo.jpg")
        sys.exit(1)

    document = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print(f"\n{'='*60}")
    print("CARDIODETECT PRODUCTION PIPELINE")
    print(f"{'='*60}")
    print(f"Document: {document}")
    print(f"{'='*60}\n")

    result = predict_risk(document, verbose=verbose)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    if result.success:
        print(f"\n🎯 RISK LEVEL: {result.risk_level}")
        print(f"   Probability: {result.risk_probability:.1f}%")
        print(f"   Confidence: {result.confidence:.1f}%")
        if result.probabilities:
            probs = [f"{k}={v*100:.1f}%" for k, v in result.probabilities.items()]
            print(f"   Distribution: {', '.join(probs)}")
        print(f"\n📝 Explanation: {result.explanation}")

        if result.top_risk_factors:
            print("\n📊 Top Risk Factors:")
            for factor in result.top_risk_factors[:3]:
                print(f"   - {factor['feature']}: {factor['direction']} risk")

        if result.needs_review:
            print(f"\n⚠️  NEEDS REVIEW:")
            for reason in result.review_reasons:
                print(f"   - {reason}")

        print(f"\n📄 OCR Quality: {result.ocr_quality} ({result.ocr_confidence:.1f}% confidence)")
        print(f"⏱️  Processing time: {result.processing_time_ms:.1f}ms")
    else:
        print(f"\n❌ ERROR: {result.error}")
        
        # Check for specific "Necessary data missing" error for "pop message" effect
        if result.error and "Necessary data missing" in result.error:
             print(f"\n⚠️  ATTENTION NEEDS REQUIRED DATA")
             print(f"   {result.error}")
        
        if result.review_reasons:
            print("\nReasons:")
            for reason in result.review_reasons:
                print(f"   - {reason}")

    print(f"\n{'='*60}")
    print("JSON OUTPUT:")
    print(f"{'='*60}")
    print(result.to_json())
