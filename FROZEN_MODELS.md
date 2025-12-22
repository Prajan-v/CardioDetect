# CardioDetect - Finalized ML Models Registry
# =============================================
# FROZEN: December 17, 2024
# DO NOT MODIFY THESE MODELS - They are production-frozen

## Production Pipeline Location
PIPELINE_ROOT = "Milestone_2/pipeline/"
MODELS_ROOT = "Milestone_2/models/"

## Core Pipeline Modules (Milestone_2/pipeline/)
# ----------------------------------------------
INTEGRATED_PIPELINE = "integrated_pipeline.py"  # DualModelPipeline - main entry
DETECTION_PIPELINE = "detection_pipeline.py"    # Heart disease detection
PREDICTION_PIPELINE = "prediction_pipeline.py"  # 10-year CVD risk prediction
CLINICAL_ADVISOR = "clinical_advisor.py"        # ACC/AHA guideline recommendations
ENHANCED_PREDICTOR = "enhanced_predictor.py"    # SHAP explanations
ULTRA_OCR = "ultra_ocr.py"                      # OCR extraction engine

## Finalized Production Models
# ----------------------------

### Detection Model (Heart Disease Detection)
# Accuracy: 91.30% on UCI Heart Disease Dataset
# Primary model: Voting Ensemble (XGBoost + RandomForest + LightGBM)
DETECTION_MODELS = {
    "primary": "Milestone_2/models/Final_models/detection/detection_voting_optimized.pkl",
    "best": "Milestone_2/models/Final_models/detection/detection_best.pkl",
    "stacking": "Milestone_2/models/Final_models/detection/detection_stacking.pkl",
    "scaler": "Milestone_2/models/Final_models/detection/detection_scaler_v3.pkl",
    "features": "Milestone_2/models/Final_models/detection/detection_features_v2.pkl",
    "config": "Milestone_2/models/Final_models/detection/detection_config_v3.pkl",
}

### Prediction Model (10-Year CVD Risk)
# Accuracy: 91.63% on Framingham Heart Study Dataset
# Algorithm: XGBoost with optimized threshold
PREDICTION_MODELS = {
    "primary": "Milestone_2/models/Final_models/prediction/prediction_xgb.pkl",
    "metadata": "Milestone_2/models/Final_models/prediction/model_meta.json",
}

### Archived Models (for reference only)
# These are in Milestone_2/models/archive/ - NOT used in production
ARCHIVED_MODELS = {
    "classification": "Milestone_2/models/archive/classification/",
    "detection_backups": "Milestone_2/models/archive/detection_backups/",
    "experiments": "Milestone_2/models/archive/",
}

## Model Performance Summary
# --------------------------
PERFORMANCE = {
    "detection": {
        "accuracy": 0.9145,
        "dataset": "UCI Heart Disease",
        "samples": 303,
        "algorithm": "Stacking Ensemble (LightGBM, XGBoost, GradientBoosting)",
    },
    "prediction": {
        "accuracy": 0.9163,
        "dataset": "Framingham Heart Study (5K subset)",
        "samples": 5000,
        "algorithm": "Optimized Gradient Boosting with threshold tuning",
    },
}

## Integration Points (Milestone_3)
# ----------------------------------
# The models are loaded via:
#   Milestone_3/services/ml_service.py -> MLService class
#   
# Which imports from:
#   Milestone_2/pipeline/integrated_pipeline.py -> DualModelPipeline
#
# Usage:
#   from services.ml_service import ml_service
#   result = ml_service.predict(features)
#   ocr_result = ml_service.process_document(file_path)

## CRITICAL: DO NOT MODIFY
# -------------------------
# These models have been validated and frozen for production.
# Any model retraining should create NEW versioned files.
# The accuracy metrics above are verified and documented.
