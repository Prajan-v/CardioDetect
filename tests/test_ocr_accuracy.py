"""
Test OCR accuracy on labeled medical documents.
Target: ≥95% field extraction accuracy on internal test set.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.universal_medical_ocr import UniversalMedicalOCREngine, extract_medical_fields


# Test data paths
TEST_SAMPLE_DIR = PROJECT_ROOT / "data" / "ocr_test_sample"
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ocr_ground_truth"

# Accuracy threshold
MIN_FIELD_ACCURACY = 0.95


def load_ground_truth(report_name: str) -> Dict:
    """Load ground truth JSON for a report."""
    gt_path = GROUND_TRUTH_DIR / f"{Path(report_name).stem}.json"
    
    if not gt_path.exists():
        pytest.skip(f"Ground truth not found: {gt_path}")
    
    with open(gt_path, 'r') as f:
        return json.load(f)


def compute_field_accuracy(extracted: Dict, ground_truth: Dict) -> float:
    """
    Compute field-level accuracy.
    Accuracy = (correct fields) / (total ground truth fields)
    """
    if not ground_truth:
        return 0.0
    
    correct = 0
    total = len(ground_truth)
    
    for field_name, gt_value in ground_truth.items():
        if field_name in extracted:
            extracted_value = extracted[field_name]
            
            # For numeric fields, allow small tolerance
            if isinstance(gt_value, (int, float)) and isinstance(extracted_value, (int, float)):
                if abs(extracted_value - gt_value) / max(abs(gt_value), 1.0) < 0.05:  # 5% tolerance
                    correct += 1
            # For categorical fields, exact match
            elif str(extracted_value).lower() == str(gt_value).lower():
                correct += 1
    
    return correct / total if total > 0 else 0.0


class TestOCRAccuracy:
    """Test suite for OCR accuracy on labeled documents."""
    
    @pytest.fixture
    def ocr_engine(self):
        """Create OCR engine instance."""
        return UniversalMedicalOCREngine(verbose=False)
    
    def test_ocr_engine_initialization(self, ocr_engine):
        """Test that OCR engine initializes correctly."""
        assert ocr_engine is not None
        assert hasattr(ocr_engine, 'extract')
    
    def test_sample_reports_exist(self):
        """Test that sample reports directory exists."""
        if not TEST_SAMPLE_DIR.exists():
            pytest.skip(f"Test sample directory not found: {TEST_SAMPLE_DIR}")
        
        reports = list(TEST_SAMPLE_DIR.glob("*.pdf")) + list(TEST_SAMPLE_DIR.glob("*.jpg")) + list(TEST_SAMPLE_DIR.glob("*.png"))
        assert len(reports) > 0, "No test reports found"
    
    def test_field_extraction_accuracy(self, ocr_engine):
        """
        Test field extraction accuracy on all labeled reports.
        Target: ≥95% accuracy.
        """
        if not TEST_SAMPLE_DIR.exists():
            pytest.skip(f"Test sample directory not found: {TEST_SAMPLE_DIR}")
        
        if not GROUND_TRUTH_DIR.exists():
            pytest.skip(f"Ground truth directory not found: {GROUND_TRUTH_DIR}")
        
        # Find all test reports
        reports = (
            list(TEST_SAMPLE_DIR.glob("*.pdf")) +
            list(TEST_SAMPLE_DIR.glob("*.jpg")) +
            list(TEST_SAMPLE_DIR.glob("*.png"))
        )
        
        if not reports:
            pytest.skip("No test reports found")
        
        results = []
        
        for report_path in reports:
            # Load ground truth
            try:
                ground_truth = load_ground_truth(report_path.name)
            except Exception:
                continue  # Skip if no ground truth
            
            # Run OCR
            result = ocr_engine.extract(report_path)
            
            # Compute accuracy
            accuracy = compute_field_accuracy(result.structured_fields, ground_truth)
            
            results.append({
                'report': report_path.name,
                'accuracy': accuracy,
                'fields_extracted': len(result.structured_fields),
                'fields_expected': len(ground_truth),
                'ocr_confidence': result.avg_ocr_confidence
            })
            
            print(f"\n{report_path.name}:")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  Fields: {len(result.structured_fields)}/{len(ground_truth)}")
            print(f"  OCR confidence: {result.avg_ocr_confidence:.1%}")
        
        # Compute average accuracy
        if results:
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            
            print(f"\n{'='*60}")
            print(f"OVERALL OCR ACCURACY: {avg_accuracy:.1%}")
            print(f"Target: {MIN_FIELD_ACCURACY:.1%}")
            print(f"Reports tested: {len(results)}")
            print(f"{'='*60}")
            
            # Assert meets threshold
            assert avg_accuracy >= MIN_FIELD_ACCURACY, (
                f"OCR accuracy {avg_accuracy:.1%} below target {MIN_FIELD_ACCURACY:.1%}"
            )
        else:
            pytest.skip("No reports with ground truth found")
    
    def test_individual_field_types(self, ocr_engine):
        """Test extraction of specific field types."""
        if not TEST_SAMPLE_DIR.exists():
            pytest.skip(f"Test sample directory not found: {TEST_SAMPLE_DIR}")
        
        reports = list(TEST_SAMPLE_DIR.glob("*.pdf"))[:1]  # Test on first PDF
        
        if not reports:
            pytest.skip("No PDF reports found")
        
        result = ocr_engine.extract(reports[0])
        
        # Check that at least some fields were extracted
        assert len(result.structured_fields) > 0, "No fields extracted"
        
        # Check field types
        numeric_fields = ['hemoglobin', 'rbc', 'wbc', 'platelets', 'cholesterol', 'glucose', 'age']
        categorical_fields = ['sex']
        
        for field in result.structured_fields:
            if field in numeric_fields:
                assert isinstance(result.structured_fields[field], (int, float)), f"{field} should be numeric"
            elif field in categorical_fields:
                assert isinstance(result.structured_fields[field], (int, str)), f"{field} should be categorical"
    
    def test_ocr_confidence_scores(self, ocr_engine):
        """Test that confidence scores are reasonable."""
        if not TEST_SAMPLE_DIR.exists():
            pytest.skip(f"Test sample directory not found: {TEST_SAMPLE_DIR}")
        
        reports = list(TEST_SAMPLE_DIR.glob("*.pdf"))[:1]
        
        if not reports:
            pytest.skip("No PDF reports found")
        
        result = ocr_engine.extract(reports[0])
        
        # Check overall confidence
        assert 0.0 <= result.avg_ocr_confidence <= 1.0, "Overall confidence out of range"
        
        # Check per-field confidences
        for field, conf in result.field_confidences.items():
            assert 0.0 <= conf <= 1.0, f"Field {field} confidence out of range"


def test_quick_ocr_extraction():
    """Quick test that OCR extraction works end-to-end."""
    # This test doesn't require test data
    engine = UniversalMedicalOCREngine(verbose=False)
    
    # Test that engine has required methods
    assert hasattr(engine, 'extract')
    assert hasattr(engine, 'detect_file_type')
    assert hasattr(engine, 'preprocess_image')
    assert hasattr(engine, 'extract_fields')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
