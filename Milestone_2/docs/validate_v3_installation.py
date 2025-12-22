#!/usr/bin/env python3
"""
CardioDetect V3 Installation Validator
Checks that all dependencies and components are properly installed.

Usage:
    python validate_v3_installation.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)


def check_python_packages():
    """Check Python package dependencies."""
    print_header("CHECKING PYTHON PACKAGES")
    
    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'pytesseract': 'pytesseract',
        'pdf2image': 'pdf2image',
        'pdfplumber': 'pdfplumber',
        'joblib': 'joblib',
    }
    
    optional_packages = {
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
    }
    
    all_ok = True
    
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name:20s} - installed")
        except ImportError:
            print(f"✗ {package_name:20s} - MISSING (required)")
            all_ok = False
    
    print("\nOptional packages:")
    for module_name, package_name in optional_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name:20s} - installed")
        except ImportError:
            print(f"○ {package_name:20s} - not installed (optional)")
    
    return all_ok


def check_system_dependencies():
    """Check system-level dependencies."""
    print_header("CHECKING SYSTEM DEPENDENCIES")
    
    import shutil
    
    deps = {
        'tesseract': 'Tesseract OCR',
        'pdftoppm': 'Poppler (for PDF conversion)',
    }
    
    all_ok = True
    
    for binary, description in deps.items():
        if shutil.which(binary):
            print(f"✓ {description:30s} - found")
        else:
            print(f"✗ {description:30s} - MISSING")
            all_ok = False
            
            if binary == 'tesseract':
                print("  Install: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)")
            elif binary == 'pdftoppm':
                print("  Install: brew install poppler (macOS) or apt-get install poppler-utils (Ubuntu)")
    
    return all_ok


def check_cardiodetect_modules():
    """Check CardioDetect V3 modules."""
    print_header("CHECKING CARDIODETECT V3 MODULES")
    
    modules = [
        ('src.universal_medical_ocr', 'Universal OCR V3'),
        ('src.mlp_v3_ensemble', 'Model V3 Ensemble'),
        ('src.cardiodetect_v3_pipeline', 'End-to-End Pipeline'),
        ('src.production_model', 'Production Model'),
        ('src.production_ocr', 'Production OCR'),
    ]
    
    all_ok = True
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {description:30s} - loaded")
        except Exception as e:
            print(f"✗ {description:30s} - ERROR: {e}")
            all_ok = False
    
    return all_ok


def check_data_directories():
    """Check required data directories."""
    print_header("CHECKING DATA DIRECTORIES")
    
    dirs = {
        'data/split': 'Training data splits',
        'models': 'Model artifacts',
        'Medical_report': 'Sample reports',
        'results': 'Output directory',
    }
    
    for dir_path, description in dirs.items():
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"✓ {description:30s} - exists ({full_path})")
        else:
            print(f"○ {description:30s} - not found (will be created)")
            full_path.mkdir(parents=True, exist_ok=True)
    
    return True


def check_models():
    """Check model files."""
    print_header("CHECKING MODEL FILES")
    
    models = {
        'models/mlp_v2.pkl': 'MLP v2 baseline',
        'models/mlp_v3_tuned.pkl': 'MLP v3 tuned (3-class)',
    }
    
    for model_path, description in models.items():
        full_path = PROJECT_ROOT / model_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✓ {description:30s} - found ({size_mb:.1f} MB)")
        else:
            print(f"○ {description:30s} - not found (will be created on first train)")
    
    return True


def test_ocr_quick():
    """Quick OCR functionality test."""
    print_header("TESTING OCR FUNCTIONALITY")
    
    try:
        from src.universal_medical_ocr import UniversalMedicalOCREngine
        
        engine = UniversalMedicalOCREngine(verbose=False)
        print("✓ OCR engine initialized")
        
        # Test that methods exist
        assert hasattr(engine, 'extract')
        assert hasattr(engine, 'detect_file_type')
        assert hasattr(engine, 'preprocess_image')
        assert hasattr(engine, 'extract_fields')
        print("✓ All OCR methods available")
        
        return True
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        return False


def test_pipeline_quick():
    """Quick pipeline functionality test."""
    print_header("TESTING PIPELINE FUNCTIONALITY")
    
    try:
        from src.cardiodetect_v3_pipeline import CardioDetectV3
        
        # Try to initialize (may fail if model doesn't exist yet)
        try:
            pipeline = CardioDetectV3(verbose=False)
            print("✓ Pipeline initialized")
            
            # Test methods exist
            assert hasattr(pipeline, 'run')
            assert hasattr(pipeline, 'build_feature_vector')
            assert hasattr(pipeline, 'categorize_risk')
            print("✓ All pipeline methods available")
            
            return True
        except FileNotFoundError as e:
            print(f"○ Pipeline initialization skipped (model not found)")
            print(f"  Run 'python src/mlp_v3_ensemble.py' to train models first")
            return True
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "="*80)
    print("CARDIODETECT V3 INSTALLATION VALIDATOR".center(80))
    print("="*80)
    
    results = []
    
    # Run all checks
    results.append(("Python Packages", check_python_packages()))
    results.append(("System Dependencies", check_system_dependencies()))
    results.append(("CardioDetect Modules", check_cardiodetect_modules()))
    results.append(("Data Directories", check_data_directories()))
    results.append(("Model Files", check_models()))
    results.append(("OCR Functionality", test_ocr_quick()))
    results.append(("Pipeline Functionality", test_pipeline_quick()))
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for check_name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status:10s} - {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! CardioDetect V3 is ready to use.")
        print("\nQuick start:")
        print("  python run_cardiodetect_v3.py path/to/report.pdf")
        print("  jupyter notebook notebooks/05_cardiodetect_v3_complete.ipynb")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please install missing dependencies.")
        print("\nSee README_V3.md for installation instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
