# CardioDetect - Data Inventory & Processing Summary

**Generated:** November 29, 2025  
**Pipeline:** `src/data_preprocessing.py`

---

## 📊 Final Dataset Statistics

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

---

## 📁 Raw Data Sources

### Framingham Heart Study Data
| File | Rows | Columns | Target Column | Status |
|------|------|---------|---------------|--------|
| `framingham_mahatir.csv` | 70,000 | 19 | `Heart_Risk` | ✓ Loaded |
| `framingham_raw.csv` | 4,240 | 16 | `TenYearCHD` | ✓ Loaded |
| `framingham_alt.csv` | 4,238 | 16 | `TenYearCHD` | ✓ Loaded |
| `framingham_noey.csv` | 4,240 | 16 | `TenYearCHD` | ✓ Loaded |
| `framingham_christofel.csv` | — | 1 | None | ✗ No target |

**Total Framingham rows:** ~82,718

### NHANES 2013-2014 Data
| File | Rows | Purpose |
|------|------|---------|
| `demographic.csv` | 9,813 | Age, sex, demographics |
| `examination.csv` | 9,813 | BP, BMI, physical exam |
| `labs.csv` | 9,813 | Cholesterol, glucose, biomarkers |
| `questionnaire.csv` | 9,813 | Smoking, health history |
| `medications.csv` | 1,299 | Medication data |
| `diet.csv` | 4,583 | Dietary intake |

**Status:** Merged but no direct CHD outcome available (skipped for now)

### Custom / UCI Heart Disease Data
| File | Rows | Columns | Target Column | Status |
|------|------|---------|---------------|--------|
| `kaggle_heart_1190.csv` | 918 | 12 | `HeartDisease` | ✓ Loaded |
| `kaggle_combined_1190.csv` | 1,190 | 12 | `target` | ✓ Loaded |
| `new_data.csv` | 1,025 | 14 | `target` | ✓ Loaded |
| `new_heart.csv` | 918 | 12 | `HeartDisease` | ✓ Loaded |
| `redwan_heart.csv` | 920 | 16 | `num` | ✓ Loaded |
| `utkarsh_heart.csv` | 270 | 14 | `heart disease` | ✓ Loaded |
| `uci_va.csv` | 200 | 14 | — | ✗ No target |
| `uci_switzerland.csv` | 123 | 14 | — | ✗ No target |

**Total Custom rows:** ~5,564

---

## 🔧 Feature Engineering Pipeline

### Base Features (12 mapped from raw data)
1. **age** - Patient age in years
2. **sex** - Sex (0=Female, 1=Male)
3. **systolic_bp** - Systolic blood pressure (mmHg)
4. **diastolic_bp** - Diastolic blood pressure (mmHg)
5. **total_cholesterol** - Total cholesterol (mg/dL)
6. **fasting_glucose** - Fasting blood glucose (mg/dL)
7. **bmi** - Body Mass Index (kg/m²)
8. **smoking** - Current smoker indicator
9. **diabetes** - Diabetes diagnosis
10. **bp_meds** - Blood pressure medication
11. **hypertension** - Hypertension diagnosis
12. **heart_rate** - Resting heart rate (bpm)

### Engineered Features (22 created)

#### Cardiovascular Metrics
- **pulse_pressure** = systolic_bp - diastolic_bp
- **mean_arterial_pressure** = diastolic_bp + (pulse_pressure / 3)

#### Risk Flags (Binary)
- **hypertension_flag** - BP ≥ 140/90 mmHg
- **high_cholesterol_flag** - Total cholesterol ≥ 240 mg/dL
- **high_glucose_flag** - Fasting glucose ≥ 126 mg/dL
- **obesity_flag** - BMI ≥ 30
- **metabolic_syndrome_score** - Sum of all 4 risk flags

#### Age Groups (One-Hot Encoded)
- **age_group_<40, 40-49, 50-59, 60-69, 70+** - 5 binary features

#### BMI Categories (One-Hot Encoded)
- **bmi_cat_Underweight, Normal, Overweight, Obese** - 4 binary features

#### Log Transforms (Skew Reduction)
- **log_total_cholesterol**
- **log_fasting_glucose**
- **log_bmi**

#### Interaction Terms
- **age_sbp_interaction** - Age × Systolic BP
- **bmi_glucose_interaction** - BMI × Glucose
- **age_smoking_interaction** - Age × Smoking

---

## 🧹 Data Cleaning Steps

### 1. Impossible Value Handling
- Cholesterol: Set to NaN if = 0 or > 400
- Systolic BP: Set to NaN if = 0, < 70, or > 250
- Diastolic BP: Set to NaN if = 0, < 40, or > 150
- Glucose: Set to NaN if = 0 or > 400
- BMI: Set to NaN if < 10 or > 60
- Age: Removed if < 18 or > 120

### 2. Missing Value Imputation
- **Numeric features:** Median imputation
- **Categorical features:** Mode imputation

### 3. Target Standardization
All target columns mapped to binary `risk_target`:
- **0** = No disease / Low risk
- **1** = Disease present / High risk (10-year CHD)

Source target columns:
- `TenYearCHD` (Framingham) → `risk_target`
- `Heart_Risk` (Framingham Mahatir) → `risk_target`
- `HeartDisease` (Kaggle) → `risk_target`
- `target` (UCI/Custom) → `risk_target`
- `num` (UCI format, 0-4 scale) → binarized to 0 vs 1+

### 4. Data Loss Analysis
- **Original unified rows:** 87,959
- **After preprocessing:** 16,123
- **Rows removed:** 71,836 (81.7%)

**Reasons for data loss:**
1. Multi-class targets converted to binary (e.g., utkarsh_heart.csv had class "2")
2. Rows with missing target values
3. Incomplete feature coverage across diverse datasets (framingham_mahatir.csv had different schema)

---

## 💾 Output Files

### Processed Datasets
| File | Location | Rows | Purpose |
|------|----------|------|---------|
| Merged Dataset | `./data/processed/merged_risk_dataset.csv` | 16,123 | Full dataset with all features |
| Final Dataset | `./data/final/final_risk_dataset.csv` | 16,123 | Ready for modeling (same as merged) |
| Train Split | `./data/split/train.csv` | 11,286 | Training set (70%) |
| Validation Split | `./data/split/val.csv` | 2,418 | Validation set (15%) |
| Test Split | `./data/split/test.csv` | 2,419 | Test set (15%) |

---

## ⚠️ Known Limitations

1. **NHANES data not yet integrated** - NHANES 2013-2014 lacks a direct 10-year CHD outcome. Future work could derive a risk score from questionnaire data.

2. **Schema heterogeneity** - `framingham_mahatir.csv` (70,000 rows) has different column names and only contributed target labels, not features. Future improvement: enhance FEATURE_MAPPING to cover its schema.

3. **Data loss** - 81.7% of rows were removed due to:
   - Multi-class targets in some UCI datasets
   - Missing critical features across sources
   - Different feature sets preventing full alignment

4. **No synthetic data** - As requested, SMOTE and other synthetic oversampling methods were NOT used. All 16,123 samples are real patient records.

---

## 🎯 Next Steps for Milestone 2

✅ **Complete:**
- [x] Unified dataset creation
- [x] Feature engineering (34 total features)
- [x] Stratified train/val/test splits
- [x] Data preprocessing pipeline

🔄 **In Progress:**
- [ ] Model training (8 algorithms)
- [ ] Risk banding (Low/Medium/High)
- [ ] OCR validation
- [ ] Jupyter notebooks (5 total)
- [ ] Milestone 2 PDF report

---

**Generated by:** `src/data_preprocessing.py`  
**Contact:** CardioDetect Project Team
