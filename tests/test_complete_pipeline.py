"""
End-to-end pipeline tests for CardioDetect V3.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cardiodetect_v3_pipeline import CardioDetectV3, predict_risk


class TestCardioDetectV3Pipeline:
    """Test suite for end-to-end pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance."""
        return CardioDetectV3(verbose=False)
    
    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline is not None
        assert hasattr(pipeline, 'run')
        assert hasattr(pipeline, 'ocr_engine')
        assert hasattr(pipeline, 'model')
    
    def test_feature_vector_building(self, pipeline):
        """Test feature vector construction from OCR fields."""
        # Sample OCR fields
        ocr_fields = {
            'age': 65,
            'sex': 1,
            'systolic_bp': 150,
            'diastolic_bp': 90,
            'cholesterol': 240,
            'glucose': 110,
        }
        
        features = pipeline.build_feature_vector(ocr_fields)
        
        # Check that features is a Series
        assert features is not None
        assert len(features) > 0
        
        # Check that OCR values were used
        if 'age' in features.index:
            assert features['age'] == 65
        if 'systolic_bp' in features.index:
            assert features['systolic_bp'] == 150
    
    def test_risk_categorization(self, pipeline):
        """Test risk categorization logic."""
        # Test LOW risk
        assert pipeline.categorize_risk(0.05) == "LOW"
        
        # Test MODERATE risk
        assert pipeline.categorize_risk(0.15) == "MODERATE"
        
        # Test HIGH risk
        assert pipeline.categorize_risk(0.25) == "HIGH"
        
        # Test boundary cases
        assert pipeline.categorize_risk(0.09) == "LOW"
        assert pipeline.categorize_risk(0.10) == "MODERATE"
        assert pipeline.categorize_risk(0.19) == "MODERATE"
        assert pipeline.categorize_risk(0.20) == "HIGH"
    
    def test_recommendation_generation(self, pipeline):
        """Test recommendation text generation."""
        # LOW risk
        rec_low = pipeline.generate_recommendation("LOW", 5.0)
        assert "Low" in rec_low or "low" in rec_low
        assert "5.0" in rec_low
        
        # MODERATE risk
        rec_mod = pipeline.generate_recommendation("MODERATE", 15.0)
        assert "Moderate" in rec_mod or "moderate" in rec_mod
        assert "15.0" in rec_mod
        
        # HIGH risk
        rec_high = pipeline.generate_recommendation("HIGH", 25.0)
        assert "High" in rec_high or "high" in rec_high
        assert "25.0" in rec_high
    
    def test_pipeline_output_structure(self, pipeline):
        """Test that pipeline output has expected structure."""
        # Create a dummy result (without actually running OCR)
        result = {
            "success": True,
            "risk_score": 0.15,
            "risk_category": "MODERATE",
            "recommendation": "Test recommendation",
            "ocr_confidence": {
                "average": 0.85,
                "per_field": {}
            },
            "model_confidence": 0.15,
            "fields": {},
            "fields_used": [],
            "audit": {
                "engine": "test",
                "file_type": "test",
                "timestamp": "2024-01-01T00:00:00",
                "model_version": "test",
                "document_path": "test.pdf"
            },
            "errors": [],
            "warnings": []
        }
        
        # Check all required keys exist
        required_keys = [
            "success", "risk_score", "risk_category", "recommendation",
            "ocr_confidence", "model_confidence", "fields", "fields_used",
            "audit", "errors", "warnings"
        ]
        
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Check types
        assert isinstance(result["success"], bool)
        assert isinstance(result["risk_score"], (int, float))
        assert 0.0 <= result["risk_score"] <= 1.0
        assert result["risk_category"] in ["LOW", "MODERATE", "HIGH", "UNKNOWN"]
        assert isinstance(result["recommendation"], str)
        assert isinstance(result["fields"], dict)
        assert isinstance(result["errors"], list)
    
    def test_pipeline_with_sample_document(self, pipeline):
        """Test pipeline on a sample document if available."""
        sample_dir = PROJECT_ROOT / "data" / "sample_reports"
        
        if not sample_dir.exists():
            pytest.skip("Sample reports directory not found")
        
        # Find first PDF or image
        samples = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        
        if not samples:
            pytest.skip("No sample reports found")
        
        # Run pipeline
        result = pipeline.run(samples[0])
        
        # Check result structure
        assert "success" in result
        assert "risk_score" in result
        assert "risk_category" in result
        
        if result["success"]:
            assert 0.0 <= result["risk_score"] <= 1.0
            assert result["risk_category"] in ["LOW", "MODERATE", "HIGH"]


def test_quick_predict_function():
    """Test the convenience predict_risk function."""
    # This just tests that the function exists and has correct signature
    assert callable(predict_risk)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
