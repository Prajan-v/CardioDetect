# 🫀 CardioDetect - Milestone 2

## Machine Learning Implementation for Cardiovascular Disease Detection & Prediction

---

## 📊 Model Performance Summary

| Model | Accuracy | ROC-AUC | Purpose |
|-------|----------|---------|---------|
| **Detection** | 91.45% | 0.931 | Current heart disease status |
| **Prediction** | 94.01% | 0.980 | 10-year CHD risk |

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Quick Start](#4-quick-start)
5. [Data Pipeline](#5-data-pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Zoo](#7-model-zoo)
8. [Risk Categorization](#8-risk-categorization)
9. [Model Evaluation](#9-model-evaluation)
10. [Interpretability & SHAP](#10-interpretability--shap)
11. [Input Validation](#11-input-validation)
12. [Thresholds & Guidelines](#12-thresholds--guidelines)
13. [V3 Implementation](#13-v3-implementation)
14. [Demo & Usage](#14-demo--usage)
15. [Viva Q&A Preparation](#15-viva-qa-preparation)

---

# 1. PROJECT OVERVIEW

## What is CardioDetect?

CardioDetect is an **end-to-end cardiovascular disease (CVD) risk prediction system** that:
- Extracts patient data from medical reports via OCR
- Predicts 10-year CVD risk using Framingham-based ML models
- Categorizes patients into LOW, MODERATE, or HIGH risk
- Provides SHAP-based explanations for predictions
- Validates input data quality before making predictions

## Key Achievement

- **Detection Model**: 91.45% accuracy for current heart disease status
- **Prediction Model**: 94.01% accuracy for 10-year CHD risk prediction

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| OCR | Tesseract / OpenCV |
| Explainability | SHAP |

---

# 2. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                     CARDIODETECT PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐
│ Medical  │───▶│   OCR    │───▶│ Preprocessing │───▶│   Model   │
│  Report  │    │ Engine   │    │   Pipeline    │    │ Inference │
└──────────┘    └──────────┘    └──────────────┘    └───────────┘
   (PDF/IMG)    (Text Extract)  (Impute+Scale)      
                                                           │
                                                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        RISK OUTPUT                               │
├────────────────┬─────────────────┬───────────────────────────────┤
│ 🟢 LOW (<10%)  │ 🟡 MODERATE     │ 🔴 HIGH (≥25%)                │
│                │   (10-25%)      │                               │
└────────────────┴─────────────────┴───────────────────────────────┘
```

### Component Details

| Component | Description |
|-----------|-------------|
| **OCR Extraction** | Tesseract with adaptive preprocessing for PDF/images |
| **Preprocessing** | SimpleImputer + StandardScaler for numeric; OneHotEncoder for categorical |
| **Model Inference** | Ensemble of XGBoost, LightGBM, Random Forest models |
| **Risk Categorization** | Continuous risk score binned to LOW/MODERATE/HIGH |

---

# 3. PROJECT STRUCTURE

```
Milestone_2/
│
├── models/                            # Trained Models
│   ├── detection/                     # 91.45% accuracy
│   │   ├── detection_xgb.pkl
│   │   ├── detection_lgbm.pkl
│   │   ├── detection_rf.pkl
│   │   ├── detection_et.pkl
│   │   └── detection_scaler.pkl
│   └── prediction/                    # 94.01% accuracy
│       ├── best_cv_ensemble_model.pkl
│       ├── prediction_xgb.pkl
│       ├── prediction_lgbm.pkl
│       ├── prediction_rf.pkl
│       ├── prediction_gb.pkl
│       └── prediction_scaler.pkl
│
├── pipeline/                          # Inference Pipelines
│   ├── integrated_pipeline.py         # Full OCR → Detection → Prediction
│   ├── detection_pipeline.py          # Detection only
│   └── prediction_pipeline.py         # Prediction only
│
├── ocr/                               # OCR Components
│   ├── Final_ocr/
│   │   └── production_ocr.py          # Final production OCR
│   ├── tesseract_ocr.py               # Tesseract wrapper
│   ├── robust_ocr_pipeline.py         # Multi-method OCR
│   └── medical_ocr_optimized.py       # Medical document OCR
│
├── notebooks/                         # Jupyter Notebooks
│   ├── CardioDetect_Complete_Demo.ipynb
│   ├── CardioDetect_Detection.ipynb
│   └── CardioDetect_Prediction.ipynb
│
├── experiments/                       # Training Scripts
│   ├── train_cv_ensemble.py           # Cross-validation ensemble
│   ├── tune_ensemble.py               # Optuna hyperparameter tuning
│   ├── train_classification.py        # Classification training
│   └── train_regressors.py            # Regression training
│
├── Source_Code/                       # Reusable Modules
│   └── src/
│       ├── preprocessing.py           # Data preprocessing
│       ├── models_*.py                # Model implementations
│       ├── ensembles.py               # Ensemble methods
│       ├── evaluation.py              # Metrics
│       └── risk_scoring.py            # Risk calculation
│
├── docs/                              # Documentation
├── reports/                           # Generated Reports
├── results/                           # Saved Results
│
└── requirements.txt                   # Dependencies
```

---

# 4. QUICK START

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Integrated Pipeline
```python
from pipeline.integrated_pipeline import DualModelPipeline

pipeline = DualModelPipeline()
result = pipeline.process_document('path/to/medical_report.png')

print(result['clinical_risk']['risk_level'])  # 🟢 LOW / 🟡 MODERATE / 🔴 HIGH
```

### 3. Run Individual Pipelines
```python
# Detection only
from pipeline.detection_pipeline import DetectionPipeline
detector = DetectionPipeline()
result = detector.predict({'age': 63, 'sex': 1, 'cp': 3, ...})

# Prediction only
from pipeline.prediction_pipeline import PredictionPipeline
predictor = PredictionPipeline()
result = predictor.predict({'age': 55, 'systolic_bp': 140, ...})
```

---

# 5. DATA PIPELINE

## Data Sources

| Source | Samples | Purpose |
|--------|---------|---------|
| UCI Heart Disease | 303 | Detection model |
| Framingham Heart Study | ~11,500 | Prediction model |
| NHANES 2013-2014 | ~3,800 | Feature enrichment |
| Kaggle datasets | ~600 | Supplementary |

## Data Splits

```
Total: ~15,900 samples (Prediction)
    │
    ├── Train (70%): ~11,000 samples
    ├── Validation (15%): ~2,400 samples
    └── Test (15%): ~2,400 samples
```

## Preprocessing Pipeline

```python
numeric_pipeline = [
    SimpleImputer(strategy='median'),  # Handle missing
    StandardScaler()                    # Normalize
]

categorical_pipeline = [
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
]
```

---

# 6. FEATURE ENGINEERING

## Core Features (34 total)

### Demographic
- `age` - Patient age in years
- `sex` - Male (1) / Female (0)

### Cardiovascular
- `systolic_bp` - Systolic blood pressure (mmHg)
- `diastolic_bp` - Diastolic blood pressure (mmHg)
- `heart_rate` - Beats per minute

### Lipid Panel
- `total_cholesterol` - Total cholesterol (mg/dL)
- `hdl_cholesterol` - HDL "good" cholesterol (mg/dL)
- `ldl_cholesterol` - LDL "bad" cholesterol (mg/dL)
- `triglycerides` - Triglycerides (mg/dL)

### Metabolic
- `fasting_glucose` - Fasting blood sugar (mg/dL)
- `bmi` - Body Mass Index (kg/m²)

### Risk Factors
- `smoking` - Current smoker (1/0)
- `diabetes` - Has diabetes (1/0)
- `hypertension` - Has hypertension (1/0)
- `bp_meds` - On BP medication (1/0)

## Derived Features

```python
# Pulse Pressure (arterial stiffness indicator)
pulse_pressure = systolic_bp - diastolic_bp

# Mean Arterial Pressure (average BP)
map = diastolic_bp + (systolic_bp - diastolic_bp) / 3

# Cholesterol Ratio (atherogenic index)
cholesterol_ratio = total_cholesterol / hdl_cholesterol
```

## Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Age | 0.28 |
| 2 | Systolic BP | 0.18 |
| 3 | Total Cholesterol | 0.12 |
| 4 | HDL Cholesterol | 0.10 |
| 5 | Smoking | 0.08 |
| 6 | Diabetes | 0.07 |
| 7 | Sex | 0.06 |
| 8 | BP Meds | 0.05 |
| 9 | BMI | 0.03 |
| 10 | Others | 0.03 |

---

# 7. MODEL ZOO

## Detection Model (91.45%)
- **Purpose:** Detect current heart disease
- **Dataset:** UCI Heart Disease (303 records)
- **Features:** 14 clinical features
- **Models:** XGBoost, LightGBM, Random Forest, Extra Trees

## Prediction Model (94.01%)
- **Purpose:** Predict 10-year CHD risk
- **Dataset:** Framingham + Kaggle (~5,600 records, cleaned)
- **Features:** 35 features (11 core + 24 engineered)
- **Models:** XGBoost, LightGBM, RF, GB ensemble

## Cross-Validation Results (5-Fold)

| Model | Mean Accuracy | Std |
|-------|--------------|-----|
| XGBoost | 93.5% | ±1.2% |
| LightGBM | 93.8% | ±1.0% |
| Ensemble | 94.0% | ±0.8% |

## Model Files

```
models/
├── detection/
│   ├── detection_xgb.pkl     # XGBoost detector
│   ├── detection_lgbm.pkl    # LightGBM detector
│   └── detection_scaler.pkl  # Preprocessing scaler
│
└── prediction/
    ├── prediction_xgb.pkl    # XGBoost predictor
    ├── prediction_lgbm.pkl   # LightGBM predictor
    └── prediction_scaler.pkl # Preprocessing scaler
```

---

# 8. RISK CATEGORIZATION

## Thresholds (Based on ACC/AHA Guidelines)

| Category | 10-Year Risk | Clinical Action |
|----------|--------------|-----------------|
| 🟢 LOW | <10% | Lifestyle maintenance, routine follow-up |
| 🟡 MODERATE | 10-25% | Risk factor modification, regular monitoring |
| 🔴 HIGH | ≥25% | Intensive intervention, specialist referral |

## Threshold Justification

- **10%**: Below this, benefits of statin therapy are marginal
- **25%**: Above this, strong evidence for pharmacological intervention
- Based on NNT (Number Needed to Treat) analysis

## Hard Override Rules

For clinical safety, certain combinations **force HIGH risk**:
```python
if age >= 65 and systolic_bp >= 160 and smoking and diabetes:
    category = "HIGH"  # Override model prediction
```

---

# 9. MODEL EVALUATION

## Metrics Used

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / Actual positives
- **F1 Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Area under ROC curve
- **Macro F1**: Average F1 across all classes

### Regression Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **Brier Score**: Calibration quality

## Key Results

| Metric | Validation | Test |
|--------|------------|------|
| MAE | 0.0064 | 0.0067 |
| RMSE | 0.0121 | 0.0125 |
| R² | 0.9905 | 0.9898 |
| Binned Accuracy | 96.5% | 96.1% |
| Binned Macro F1 | 0.9649 | 0.9608 |

## Confusion Matrix Interpretation

```
              Predicted
            LOW  MOD  HIGH
Actual LOW  [98%  2%   0% ]
      MOD  [ 3% 94%   3% ]
      HIGH [ 0%  4%  96% ]
```

Most errors are between adjacent categories, not extreme misclassifications.

---

# 10. INTERPRETABILITY & SHAP

## What is SHAP?

**SHAP (SHapley Additive exPlanations)** assigns each feature a contribution to the prediction:

```
Base risk (population average) = 9%
+ Age (60)              = +4.5%
+ Systolic BP (140)     = +2.8%
+ On BP Therapy         = +1.5%
+ TC/HDL Ratio (4.4)    = +1.2%
- HDL (50)              = -0.5%
- Non-Smoker            = -1.0%
─────────────────────────────────
Final Risk              = 18.5%
```

## Why SHAP for Medical AI?

1. **Clinical Trust**: Doctors verify reasoning matches medical knowledge
2. **Actionable Insights**: Shows which risk factors to target
3. **Error Detection**: Unusual attributions may indicate OCR errors
4. **Patient Communication**: Explain risk factors in simple terms
5. **Regulatory Compliance**: FDA/GDPR require explainability

## Example Explanations

**HIGH Risk Patient:**
> "Risk is HIGH (28.5%) mainly because: Age 72, systolic BP 175, and current smoking status strongly increase estimated 10-year CVD risk."

**LOW Risk Patient:**
> "Risk is LOW (6.2%) because: Age 42, normal blood pressure (118/76), and high HDL cholesterol (65 mg/dL) are protective factors."

---

# 11. INPUT VALIDATION

## Critical Features

These 5 features are **essential** for accurate predictions:

| Feature | Why Critical |
|---------|--------------|
| Age | Strongest predictor |
| Systolic BP | Key modifiable risk |
| Total Cholesterol | Lipid risk marker |
| Smoking | Major risk factor |
| Diabetes | Comorbidity |

## Confidence Scoring

| Missing Type | Penalty |
|--------------|---------|
| Critical feature | -15% each |
| Important feature (HDL, Sex) | -8% each |
| Optional feature (BMI) | -2% each |
| Out-of-range value | -10% each |

## Physiological Ranges

```python
VALID_RANGES = {
    'age': (18, 120),
    'systolic_bp': (60, 250),
    'diastolic_bp': (30, 150),
    'total_cholesterol': (80, 500),
    'hdl_cholesterol': (10, 150),
    'bmi': (10, 70),
    'heart_rate': (30, 220),
}
```

---

# 12. THRESHOLDS & GUIDELINES

## Framingham-Based Guidelines

The thresholds align with the Framingham Heart Study risk categorization:
- **<10%**: Low 10-year risk
- **10-20%**: Intermediate risk
- **>20%**: High risk

## Guideline Risk Function

CardioDetect uses a Framingham-like 10-year general CVD risk function:

```python
def compute_guideline_risk(
    age, sex, total_cholesterol, hdl_cholesterol,
    systolic_bp, on_treatment, smoking, diabetes
) -> float:
    """Returns 10-year CVD risk as probability [0, 1]."""
```

## Framingham Coefficients

**Male coefficients:**
- β_age = 3.06117
- β_TC = 1.12370
- β_HDL = -0.93263
- β_SBP (treated) = 1.99881
- β_smoking = 0.65451
- β_diabetes = 0.57367

**Female coefficients:**
- β_age = 2.32888
- β_TC = 1.20904
- β_HDL = -0.70833
- β_SBP (treated) = 2.82263
- β_smoking = 0.52873
- β_diabetes = 0.69154

---

# 13. V3 IMPLEMENTATION

## What's New in V3

### Universal OCR V3 Engine
- Multi-format: JPG, PNG, PDF (digital + scanned)
- 95%+ field extraction target
- 7-step OpenCV preprocessing
- Per-field confidence scoring

### Model V3 Ensemble Framework
- MLP v2 baseline (93.59% accuracy)
- Random Forest, XGBoost, LightGBM
- Auto-selection with no-regression policy

### End-to-End Pipeline
- Document → Risk JSON in <10s
- Complete integration with audit trail
- Batch processing support

## V3 Quick Start

```bash
# Install
pip install -r requirements_v3.txt
brew install tesseract poppler

# Validate
python validate_v3_installation.py

# Run
python run_cardiodetect_v3.py Medical_report/sample.pdf
```

## V3 Performance Targets

| Component | Metric | Target |
|-----------|--------|--------|
| OCR V3 | Field Accuracy | ≥95% |
| OCR V3 | Processing Time | <5s |
| Model V3 | Accuracy | ≥93.59% |
| Pipeline | End-to-End | <10s |

---

# 14. DEMO & USAGE

## Quick Demo

```python
import numpy as np

# Patient data
patient = {
    'age': 60, 'sex': 1, 'systolic_bp': 140, 'diastolic_bp': 90,
    'total_cholesterol': 220, 'hdl_cholesterol': 50,
    'smoking': 0, 'diabetes': 0, 'bp_meds': 1
}

# Framingham Risk Calculation
log_age = np.log(patient['age'])
log_tc = np.log(patient['total_cholesterol'])
log_hdl = np.log(patient['hdl_cholesterol'])
log_sbp = np.log(patient['systolic_bp'])

# Male coefficient (treated BP)
L = (3.06117 * log_age + 1.12370 * log_tc - 0.93263 * log_hdl + 
     1.99881 * log_sbp + 0.65451 * patient['smoking'] + 
     0.57367 * patient['diabetes'] - 23.9802)

risk = 1 - 0.88936 ** np.exp(L)
risk = max(0, min(1, risk))

# Categorize
if risk < 0.10: cat = "🟢 LOW"
elif risk < 0.25: cat = "🟡 MODERATE"
else: cat = "🔴 HIGH"

print(f"10-Year CVD Risk: {risk:.1%}")
print(f"Category: {cat}")
```

## Data Paths

```python
# Detection
detection_data = '../data/split/detection/'

# Prediction
prediction_data = '../data/split/prediction/'

# Models
detection_models = '../models/detection/'
prediction_models = '../models/prediction/'
```

---

# 15. VIVA Q&A PREPARATION

## Key Questions & Answers

### Q: What is CardioDetect?
**A:** An end-to-end ML system that predicts 10-year cardiovascular disease risk from patient data, categorizes risk levels, and provides explainable AI insights for clinical decision support.

### Q: How accurate is your model?
**A:** Detection: 91.45% accuracy. Prediction: 94.01% accuracy with MAE of 0.0064 on continuous risk prediction and R² of 0.99.

### Q: Why Random Forest over Deep Learning?
**A:** 
1. Tabular data with ~34 features doesn't benefit from DL
2. RF provides built-in feature importance
3. Easier to interpret for regulatory compliance
4. Comparable or better accuracy with less training time

### Q: How do you handle missing values?
**A:** We use sklearn's SimpleImputer:
- Numeric features → median imputation
- Categorical features → mode imputation
The validation module tracks imputation and reduces confidence score accordingly.

### Q: What are the risk thresholds?
**A:** Based on ACC/AHA guidelines:
- LOW: <10% (lifestyle advice)
- MODERATE: 10-25% (enhanced monitoring)
- HIGH: ≥25% (aggressive intervention)

### Q: Why SHAP for explainability?
**A:** SHAP provides:
1. Per-patient feature attributions
2. Additive contributions (they sum to the prediction)
3. Theoretically grounded (Shapley values from game theory)
4. Regulatory compliance for medical AI

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Detection Accuracy | 91.45% |
| Prediction Accuracy | 94.01% |
| Total features | 34 |
| Final model | XGBoost + LightGBM Ensemble |
| R² | 0.990 |
| MAE | 0.0064 (0.64%) |
| LOW threshold | <10% |
| HIGH threshold | ≥25% |

---

## 🎯 QUICK REFERENCE CARD

```
╔══════════════════════════════════════════════════════════════╗
║                  CARDIODETECT QUICK REFERENCE                ║
╠══════════════════════════════════════════════════════════════╣
║ Pipeline: OCR → Preprocess → ML Model → Risk Category        ║
║                                                              ║
║ Risk Thresholds:                                             ║
║   🟢 LOW: <10%  │  🟡 MODERATE: 10-25%  │  🔴 HIGH: ≥25%     ║
║                                                              ║
║ Models:                                                      ║
║   - Detection: 91.45% (XGBoost + LightGBM)                  ║
║   - Prediction: 94.01% (Ensemble)                           ║
║                                                              ║
║ Key Features: Age, SBP, Total Cholesterol, HDL, Smoking      ║
║                                                              ║
║ Explainability: SHAP (per-patient feature contributions)     ║
║ Validation: Confidence scoring, missing value detection      ║
╚══════════════════════════════════════════════════════════════╝
```

---

# 16. DATA INVENTORY

## Final Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Samples** | 16,123 |
| **Total Features** | 34 |
| **Target Column** | `risk_target` |
| **Positive Cases** | 3,986 (24.7%) |
| **Negative Cases** | 12,137 (75.3%) |

### Split Distribution

| Split | Samples | Positive % |
|-------|---------|-----------|
| **Train** | 11,286 (70%) | 24.7% |
| **Validation** | 2,418 (15%) | 24.7% |
| **Test** | 2,419 (15%) | 24.7% |

## Raw Data Sources

### Framingham Heart Study Data
| File | Rows | Target Column | Status |
|------|------|---------------|--------|
| `framingham_mahatir.csv` | 70,000 | `Heart_Risk` | ✓ Loaded |
| `framingham_raw.csv` | 4,240 | `TenYearCHD` | ✓ Loaded |
| `framingham_alt.csv` | 4,238 | `TenYearCHD` | ✓ Loaded |

### UCI/Kaggle Heart Disease Data
| File | Rows | Target Column | Status |
|------|------|---------------|--------|
| `kaggle_heart_1190.csv` | 918 | `HeartDisease` | ✓ Loaded |
| `kaggle_combined_1190.csv` | 1,190 | `target` | ✓ Loaded |
| `new_data.csv` | 1,025 | `target` | ✓ Loaded |

## Data Cleaning Steps

1. **Impossible Value Handling**: Cholesterol: NaN if = 0 or > 400; BP bounds enforced
2. **Missing Value Imputation**: Numeric → median; Categorical → mode
3. **Target Standardization**: All targets mapped to binary `risk_target` (0/1)

---

# 17. PRODUCTION INSTALLATION

## System Requirements

- **Python:** 3.10+
- **OS:** macOS, Linux, or Windows
- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** 500MB for models and dependencies

## Quick Install

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Tesseract OCR
# macOS:
brew install tesseract poppler

# Ubuntu:
sudo apt-get install tesseract-ocr poppler-utils

# 4. Verify installation
python -c "from src.production_pipeline import predict_risk_from_data; print('✓ Ready')"
```

## Core Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
opencv-python-headless>=4.8.0
pytesseract>=0.3.10
pdf2image>=1.16.0
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `TesseractNotFoundError` | Install Tesseract: `brew install tesseract` |
| `PDFInfoNotInstalledError` | Install poppler: `brew install poppler` |
| `Model not found` | Ensure `models/mlp_v2.pkl` exists |

---

# 18. EXPERIMENT LOG

## Key Experiments Summary

| Exp | Model | Accuracy | Recall | Notes |
|-----|-------|----------|--------|-------|
| 1 | Logistic Regression | 83.90% | 36.60% | Too conservative |
| 2 | Random Forest (default) | 83.12% | 35.20% | Needs tuning |
| 3 | Random Forest (tuned) | 84.38% | 38.68% | Best overall |
| 4 | XGBoost | 82.48% | 42.58% | Best recall |
| 5 | Threshold Tuning | 85.00% | 42.10% | Improved balance |
| 6 | Neural Network | 80.43% | 41.80% | Needs more data |
| 7 | SMOTE | 76.45% | 64.07% | High recall, low precision |
| 10 | Sweet Spot (t=0.25) | **76.84%** | **62.77%** | Final choice |

## Key Insights

1. **Data Quality > Model Complexity**: Splitting data by quality was more effective than complex models
2. **Ensemble Methods Work**: Random Forest and stacking outperformed single models
3. **Recall vs Accuracy Trade-off**: For medical apps, optimize for recall (catching disease cases)
4. **Neural Networks Need More Data**: ~5000 samples insufficient for deep learning

## Hybrid Architecture (Final Solution)

- **High Quality Model**: 91.25% accuracy (patients with complete data)
- **Low Quality Model**: 83.72% accuracy (patients with missing features)
- **Router**: Automatically selects appropriate model based on data availability

---

*CardioDetect v2.0 | Detection: 91.45% | Prediction: 94.01%*  
*Last updated: December 2025*
