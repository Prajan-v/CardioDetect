# 📊 CardioDetect Data Folder

## 📂 Folder Structure

```
data/
│
├── raw/                                      # Original Source Datasets
│   ├── detection_raw/
│   │   └── heart.csv                         # UCI Heart Disease (303 records)
│   └── prediction_raw/
│       └── framingham_combined.csv           # Framingham + Kaggle (16K records)
│
├── processed_data/                           # Cleaned & Preprocessed
│   ├── detection_processed/
│   │   └── combined_data.csv                 # Merged UCI datasets
│   └── prediction_processed/
│       └── combined_processed.csv            # Processed with features
│
├── final_dataset/                            # Final Training-Ready Data
│   ├── detection_data/
│   │   └── uci_all.csv                       # Final detection dataset
│   └── prediction_data/
│       ├── final_risk_dataset.csv            # Complete with risk scores
│       └── sample_demo_100.csv               # Demo sample
│
└── split/                                    # Train/Val/Test Splits (70/15/15)
    ├── detection/
    │   ├── train.csv                         # 212 records
    │   ├── val.csv                           # 45 records
    │   └── test.csv                          # 46 records
    └── prediction/
        ├── train.csv                         # ~11,000 records
        ├── val.csv                           # ~2,500 records
        └── test.csv                          # ~2,500 records
```

---

## 🔍 Detection Model (91.45% Accuracy)

**Purpose:** Detect current heart disease  
**Data Path:** `data/split/detection/`

| Dataset | Records | Features |
|---------|---------|----------|
| Train | 212 | 14 |
| Validation | 45 | 14 |
| Test | 46 | 14 |

### Features:
```
age, sex, cp, trestbps, chol, fbs, restecg, 
thalach, exang, oldpeak, slope, ca, thal
```

### Target: `target` (0=No Disease, 1=Disease)

---

## 📈 Prediction Model (94.01% Accuracy)

**Purpose:** Predict 10-year CHD risk  
**Data Path:** `data/split/prediction/`

| Dataset | Records | Features |
|---------|---------|----------|
| Train | ~11,000 | 36 |
| Validation | ~2,500 | 36 |
| Test | ~2,500 | 36 |

### Core Features:
```
age, sex, smoking, bp_meds, hypertension, diabetes,
total_cholesterol, systolic_bp, diastolic_bp, 
bmi, heart_rate, fasting_glucose
```

### Engineered Features:
```
pulse_pressure, mean_arterial_pressure,
hypertension_flag, high_cholesterol_flag, 
metabolic_syndrome_score, log transforms,
age_sbp_interaction, bmi_glucose_interaction
```

### Target: `risk_target` (0=LOW, 1=MODERATE, 2=HIGH)

---

## 📍 Code Paths

```python
# Detection Model:
detection_train = 'data/split/detection/train.csv'
detection_val = 'data/split/detection/val.csv'
detection_test = 'data/split/detection/test.csv'

# Prediction Model:
prediction_train = 'data/split/prediction/train.csv'
prediction_val = 'data/split/prediction/val.csv'
prediction_test = 'data/split/prediction/test.csv'
```

---

*CardioDetect v2.0 | Milestone 2*
