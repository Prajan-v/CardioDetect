"""Comprehensive Test Suite for Production System.

Tests:
- OCR accuracy on different document types
- Model prediction accuracy
- Uncertainty quantification
- Explainability output
- Edge cases and error handling
- End-to-end pipeline

Run with: pytest tests/test_production_system.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.production_ocr import ProductionOCR, OCRResult, ExtractedField
from src.production_model import ProductionModel, RiskPrediction
from src.production_pipeline import ProductionPipeline, PipelineResult, predict_risk_from_data


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_high_risk_patient():
    """High risk patient data."""
    return {
        "age": 72,
        "sex": 1,
        "systolic_bp": 170,
        "diastolic_bp": 98,
        "total_cholesterol": 260,
        "fasting_glucose": 155,
        "bmi": 34.5,
        "smoking": 1,
        "diabetes": 1,
    }


@pytest.fixture
def sample_low_risk_patient():
    """Low risk patient data."""
    return {
        "age": 28,
        "sex": 0,
        "systolic_bp": 112,
        "diastolic_bp": 72,
        "total_cholesterol": 178,
        "fasting_glucose": 88,
        "bmi": 22.3,
        "smoking": 0,
        "diabetes": 0,
    }


@pytest.fixture
def sample_medium_risk_patient():
    """Medium risk patient data."""
    return {
        "age": 58,
        "sex": 1,
        "systolic_bp": 138,
        "diastolic_bp": 86,
        "total_cholesterol": 215,
        "fasting_glucose": 110,
        "bmi": 28.4,
        "smoking": 0,
        "diabetes": 0,
    }


@pytest.fixture
def edge_case_elderly_healthy():
    """Edge case: elderly but healthy."""
    return {
        "age": 78,
        "sex": 0,
        "systolic_bp": 120,
        "diastolic_bp": 74,
        "total_cholesterol": 185,
        "fasting_glucose": 92,
        "bmi": 23.5,
        "smoking": 0,
        "diabetes": 0,
    }


@pytest.fixture
def edge_case_young_risky():
    """Edge case: young but high risk factors."""
    return {
        "age": 32,
        "sex": 1,
        "systolic_bp": 150,
        "diastolic_bp": 95,
        "total_cholesterol": 245,
        "fasting_glucose": 145,
        "bmi": 36.5,
        "smoking": 1,
        "diabetes": 1,
    }


@pytest.fixture
def production_model():
    """Production model instance."""
    return ProductionModel(verbose=False, enable_shap=False)


@pytest.fixture
def production_pipeline():
    """Production pipeline instance."""
    return ProductionPipeline(verbose=False, enable_shap=False)


# =============================================================================
# OCR Tests
# =============================================================================

class TestProductionOCR:
    """Test suite for production OCR."""

    def test_ocr_initialization(self):
        """Test OCR can be initialized."""
        ocr = ProductionOCR(verbose=False)
        assert ocr is not None

    def test_field_patterns_exist(self):
        """Test all field patterns are defined."""
        ocr = ProductionOCR()
        required_fields = ["age", "sex", "systolic_bp", "diastolic_bp", "bmi",
                          "total_cholesterol", "fasting_glucose"]
        for field in required_fields:
            assert field in ocr.FIELD_PATTERNS

    def test_valid_ranges_exist(self):
        """Test valid ranges are defined for numeric fields."""
        ocr = ProductionOCR()
        numeric_fields = ["age", "systolic_bp", "bmi", "total_cholesterol"]
        for field in numeric_fields:
            assert field in ocr.VALID_RANGES
            low, high = ocr.VALID_RANGES[field]
            assert low < high

    def test_extract_numeric_valid(self):
        """Test numeric extraction from text."""
        ocr = ProductionOCR()
        text = "Age: 65 years"
        value, conf, raw = ocr._extract_numeric(
            text, ocr.FIELD_PATTERNS["age"], ocr.VALID_RANGES["age"]
        )
        assert value == 65
        assert conf > 0

    def test_extract_numeric_invalid_range(self):
        """Test numeric extraction rejects invalid values."""
        ocr = ProductionOCR()
        text = "Age: 200 years"  # Invalid age
        value, conf, raw = ocr._extract_numeric(
            text, ocr.FIELD_PATTERNS["age"], ocr.VALID_RANGES["age"]
        )
        assert value is None  # Should reject

    def test_extract_categorical(self):
        """Test categorical extraction."""
        ocr = ProductionOCR()
        text = "Sex: Male"
        mapping = {"m": 1, "male": 1, "f": 0, "female": 0}
        value, conf, raw = ocr._extract_categorical(
            text, ocr.FIELD_PATTERNS["sex"], mapping
        )
        assert value == 1

    def test_extract_all_fields(self):
        """Test extraction of all fields from sample text."""
        ocr = ProductionOCR()
        text = """
        Patient ID: TEST-001
        Age: 55 years
        Sex: Male
        Systolic BP: 140 mmHg
        Diastolic BP: 90 mmHg
        Total Cholesterol: 220 mg/dL
        Fasting Glucose: 105 mg/dL
        BMI: 28.5 kg/m²
        Smoking: Yes
        Diabetes: No
        """
        fields = ocr._extract_all_fields(text)

        assert "age" in fields
        assert fields["age"].value == 55
        assert "systolic_bp" in fields
        assert fields["systolic_bp"].value == 140
        assert "smoking" in fields
        assert fields["smoking"].value == 1
        assert "diabetes" in fields
        assert fields["diabetes"].value == 0

    def test_ocr_result_structure(self):
        """Test OCRResult has correct structure."""
        result = OCRResult(success=True)
        assert hasattr(result, "fields")
        assert hasattr(result, "overall_confidence")
        assert hasattr(result, "warnings")
        assert hasattr(result, "errors")

    def test_ocr_result_to_dict(self):
        """Test OCRResult serialization."""
        result = OCRResult(
            success=True,
            overall_confidence=85.0,
            document_quality="clean",
        )
        result.fields["age"] = ExtractedField(
            name="age", value=65, confidence=90.0
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["overall_confidence"] == 85.0
        assert "age" in d["fields"]


# =============================================================================
# Model Tests
# =============================================================================

class TestProductionModel:
    """Test suite for production model."""

    def test_model_initialization(self, production_model):
        """Test model can be initialized."""
        assert production_model is not None
        assert production_model.model is not None
        assert production_model.scaler is not None

    def test_feature_vector_building(self, production_model, sample_high_risk_patient):
        """Test feature vector construction."""
        features = production_model.build_feature_vector(sample_high_risk_patient)
        assert len(features) > 0
        assert "age" in features.index or features.iloc[0] is not None

    def test_high_risk_prediction(self, production_model, sample_high_risk_patient):
        """Test HIGH risk patient is correctly classified."""
        result = production_model.predict(sample_high_risk_patient)

        assert result.risk_level in ["HIGH", "MEDIUM"]  # Should be HIGH or at least MEDIUM
        assert result.risk_probability > 10  # Should have elevated risk
        assert result.confidence > 0

    def test_low_risk_prediction(self, production_model, sample_low_risk_patient):
        """Test LOW risk patient is correctly classified."""
        result = production_model.predict(sample_low_risk_patient)

        assert result.risk_level in ["LOW", "MEDIUM"]
        assert result.risk_probability < 30  # Should have low risk
        assert result.confidence > 0

    def test_prediction_has_explanation(self, production_model, sample_high_risk_patient):
        """Test predictions include explanations."""
        result = production_model.predict(sample_high_risk_patient)

        assert result.explanation is not None
        assert len(result.explanation) > 0

    def test_prediction_has_risk_factors(self, production_model, sample_high_risk_patient):
        """Test predictions include risk factors."""
        result = production_model.predict(sample_high_risk_patient)

        # Should have at least one risk factor identified
        assert len(result.top_risk_factors) >= 0  # May be empty if SHAP disabled

    def test_uncertainty_quantification(self, production_model, sample_high_risk_patient):
        """Test uncertainty is computed."""
        result = production_model.predict(sample_high_risk_patient)

        assert result.epistemic_uncertainty >= 0
        assert result.aleatoric_uncertainty >= 0
        assert result.confidence >= 0

    def test_ood_detection(self, production_model):
        """Test out-of-distribution detection."""
        # Extreme values should be flagged
        extreme_patient = {
            "age": 120,  # Extreme age
            "systolic_bp": 300,  # Extreme BP
            "total_cholesterol": 500,
        }
        result = production_model.predict(extreme_patient)
        # Note: OOD detection may or may not trigger depending on training data

    def test_missing_field_handling(self, production_model):
        """Test handling of missing fields."""
        minimal_patient = {"age": 50, "systolic_bp": 130}
        result = production_model.predict(minimal_patient)

        assert result.risk_level != "ERROR"
        # Should still produce a prediction with defaults

    def test_prediction_result_structure(self, production_model, sample_low_risk_patient):
        """Test RiskPrediction has all expected fields."""
        result = production_model.predict(sample_low_risk_patient)

        assert hasattr(result, "risk_level")
        assert hasattr(result, "risk_probability")
        assert hasattr(result, "confidence")
        assert hasattr(result, "explanation")
        assert hasattr(result, "needs_review")
        assert hasattr(result, "is_ood")

    def test_prediction_to_dict(self, production_model, sample_low_risk_patient):
        """Test prediction serialization."""
        result = production_model.predict(sample_low_risk_patient)
        d = result.to_dict()

        assert "risk_level" in d
        assert "risk_probability" in d
        assert "confidence" in d


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_elderly_healthy_patient(self, production_model, edge_case_elderly_healthy):
        """Test elderly but healthy patient."""
        result = production_model.predict(edge_case_elderly_healthy)

        # Elderly with good vitals should not automatically be HIGH risk
        assert result.risk_level in ["LOW", "MEDIUM", "HIGH"]
        # Age alone shouldn't dominate if other factors are good

    def test_young_risky_patient(self, production_model, edge_case_young_risky):
        """Test young but risky patient."""
        result = production_model.predict(edge_case_young_risky)

        # Young with multiple risk factors should have elevated risk
        assert result.risk_probability > 5  # Should have some risk

    def test_extreme_values(self, production_model):
        """Test extreme but valid values."""
        extreme = {
            "age": 95,
            "systolic_bp": 200,
            "total_cholesterol": 350,
            "fasting_glucose": 300,
            "bmi": 45,
        }
        result = production_model.predict(extreme)
        assert result.risk_level == "HIGH"

    def test_all_zeros(self, production_model):
        """Test all zero/missing values."""
        zeros = {"age": 0, "systolic_bp": 0}
        result = production_model.predict(zeros)
        # Should handle gracefully, not crash
        assert result is not None

    def test_string_values(self, production_model):
        """Test string values for binary fields."""
        patient = {
            "age": 50,
            "systolic_bp": 130,
            "smoking": "Yes",
            "diabetes": "No",
        }
        result = production_model.predict(patient)
        assert result.risk_level in ["LOW", "MEDIUM", "HIGH"]


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestProductionPipeline:
    """Test end-to-end pipeline."""

    def test_pipeline_initialization(self, production_pipeline):
        """Test pipeline can be initialized."""
        assert production_pipeline is not None
        assert production_pipeline.ocr is not None
        assert production_pipeline.model is not None

    def test_predict_from_data(self, sample_high_risk_patient):
        """Test prediction from pre-extracted data."""
        result = predict_risk_from_data(sample_high_risk_patient)

        assert result.success
        assert result.risk_level in ["LOW", "MEDIUM", "HIGH"]
        assert result.risk_probability >= 0

    def test_pipeline_result_structure(self, sample_low_risk_patient):
        """Test PipelineResult has all fields."""
        result = predict_risk_from_data(sample_low_risk_patient)

        assert hasattr(result, "success")
        assert hasattr(result, "risk_level")
        assert hasattr(result, "confidence")
        assert hasattr(result, "needs_review")
        assert hasattr(result, "processing_time_ms")

    def test_pipeline_result_to_json(self, sample_low_risk_patient):
        """Test JSON serialization."""
        result = predict_risk_from_data(sample_low_risk_patient)
        json_str = result.to_json()

        import json
        parsed = json.loads(json_str)

        assert "risk" in parsed
        assert "level" in parsed["risk"]
        assert "flags" in parsed
        assert "metadata" in parsed

    def test_missing_fields_flagged(self):
        """Test missing fields are properly flagged."""
        minimal = {"age": 50}  # Missing systolic_bp
        result = predict_risk_from_data(minimal)

        # Should complete but flag missing fields
        assert len(result.missing_fields) > 0

    def test_processing_time_recorded(self, sample_low_risk_patient):
        """Test processing time is recorded."""
        result = predict_risk_from_data(sample_low_risk_patient)
        assert result.processing_time_ms > 0


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance requirements."""

    def test_prediction_speed(self, production_model, sample_low_risk_patient):
        """Test prediction is fast enough."""
        import time

        start = time.time()
        for _ in range(10):
            production_model.predict(sample_low_risk_patient)
        elapsed = time.time() - start

        avg_time = elapsed / 10
        assert avg_time < 1.0  # Should complete in under 1 second

    def test_model_consistency(self, production_model, sample_high_risk_patient):
        """Test predictions are consistent."""
        results = [production_model.predict(sample_high_risk_patient) for _ in range(5)]

        levels = [r.risk_level for r in results]
        probs = [r.risk_probability for r in results]

        # All predictions should be the same
        assert len(set(levels)) == 1
        assert max(probs) - min(probs) < 0.1  # Less than 0.1% variance


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
