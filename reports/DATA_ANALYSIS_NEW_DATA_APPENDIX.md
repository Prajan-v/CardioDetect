# New Data Appendix: CardioDetect Risk Dataset (v2)

## 1. Why a New Dataset Was Introduced

The original data analysis in `DATA_ANALYSIS_REPORT.pdf` was based on legacy public datasets (UCI Heart Disease, early Framingham subsets, and small merged cohorts). That work was valuable, but it had several limitations:

1. **Limited Sample Size**  
   - Original combined cohorts: **≈5,000–7,000** patients.  
   - Many models (especially deep or highly regularized ones) were data‑hungry and could not fully realize their potential.

2. **Heterogeneous Schemas**  
   - Different studies used different feature names and encodings.  
   - Substantial effort was spent on manual mapping, and some information was lost or inconsistently encoded.

3. **Mixed Objectives**  
   - Earlier work focused primarily on **diagnostic classification** (current heart disease yes/no).  
   - Long‑term **10‑year risk prediction** requires a different set of features, assumptions, and evaluation metrics.

4. **Ceiling on Risk Prediction Performance**  
   - The Framingham‑only risk models plateaued around **85–86% accuracy**.  
   - To push beyond this without sacrificing recall would require richer features and a larger, more representative cohort.

To address these issues, we constructed a **new, unified risk dataset** specifically for the CardioDetect 10‑year cardiovascular risk prediction arm.

---

## 2. Overview of the New CardioDetect Risk Dataset

### 2.1 High‑Level Summary

| Attribute | Value |
|----------|-------|
| Total Patients | **16,123** |
| Total Features | **34** (demographic, clinical, derived) |
| Target | `risk_target` (10‑year CHD event: 0/1) |
| Class Balance | ~76% negative / 24% positive |
| Splits | 70% train / 15% val / 15% test (stratified) |

### 2.2 Data Sources

The dataset integrates multiple established sources into a single, clean table:

| Source | Approx. Patients | Description |
|--------|------------------|-------------|
| Framingham Heart Study | ~4,000 | Longitudinal CHD cohort |
| NHANES | ~10,000 | National health & nutrition survey |
| Custom Clinical Records | ~2,000 | Supplementary anonymized records |

All sources were harmonized into a consistent schema and stored as:

- Final dataset: `data/final/final_risk_dataset.csv`  
- Split datasets: `data/split/train.csv`, `data/split/val.csv`, `data/split/test.csv`

### 2.3 Feature Categories

The 34 features are grouped into the following categories:

- **Demographics (2)**  
  `age`, `sex`

- **Clinical Measurements (6)**  
  `systolic_bp`, `diastolic_bp`, `bmi`, `heart_rate`, `total_cholesterol`, `fasting_glucose`

- **Risk Factors (5)**  
  `smoking`, `bp_meds`, `hypertension`, `diabetes`, `data_source`

- **Derived Measurements & Scores (21)**  
  - Hemodynamic: `pulse_pressure`, `mean_arterial_pressure`  
  - Clinical flags: `hypertension_flag`, `high_cholesterol_flag`, `high_glucose_flag`, `obesity_flag`  
  - Categorical encodings: `age_group_*`, `bmi_cat_*`  
  - Interaction terms: `age_sbp_interaction`, `bmi_glucose_interaction`, `age_smoking_interaction`  
  - Composite: `metabolic_syndrome_score`

This feature engineering is implemented and documented in the main CardioDetect codebase and the `DATA_DICTIONARY.pdf` deliverable.

---

## 3. Improvements Over the Old Data

### 3.1 Scale and Statistical Power

- Old risk datasets: **≈5–7k** patients.  
- New CardioDetect risk dataset: **16,123** patients.  

This increase in sample size improves:

- Stability of estimates (narrower confidence intervals).  
- Robustness of train/validation/test splits.  
- Reliability of rare‑event modeling in the positive (CHD) class.

### 3.2 Feature Richness and Clinical Structure

Compared with the older data used in the original `DATA_ANALYSIS_REPORT.pdf`:

- The new dataset encodes **clinical flags** and **interaction terms** explicitly, rather than relying only on raw vitals.  
- Risk‑relevant constructs (e.g., `metabolic_syndrome_score`, `pulse_pressure`, `mean_arterial_pressure`) are included as first‑class features.  
- Age and BMI are represented both continuously and via clinically meaningful bins (`age_group_*`, `bmi_cat_*`).

This brings the dataset closer to how clinicians reason about cardiovascular risk, and it aligns with common risk calculators (Framingham, ASCVD) while adding derived features those calculators do not expose directly.

### 3.3 Clean, Reproducible Splits

The new data pipeline:

- Uses a **fixed random seed** and **stratified splitting** to keep class balance consistent across train/val/test.  
- Stores the exact splits in `data/split/` for reproducibility.  
- Ensures that all model training and tuning (e.g., in `mlp_tuning.py`) uses the same split definitions.

This is a significant improvement over earlier ad‑hoc splits in the legacy experiments.

---

## 4. New Model Performance on the CardioDetect Risk Dataset

On the new 16,123‑patient dataset, the final chosen model is a **Multi‑Layer Perceptron (MLP)** trained and tuned in `src/mlp_tuning.py`.

### 4.1 Test Set Metrics (n ≈ 2,418)

| Metric | Value |
|--------|-------|
| Accuracy | **93.59%** |
| Precision | 83.15% |
| Recall (Sensitivity) | 91.90% |
| F1‑Score | 0.8731 |
| ROC‑AUC | **0.9673** |

**Confusion Matrix (Test Set):**

- True Negatives (TN): 1,731  
- False Positives (FP): 108  
- False Negatives (FN): 47  
- True Positives (TP): 533

These results indicate a **high‑performing risk model** that prioritizes sensitivity (recall) while maintaining strong overall accuracy.

---

## 5. How to Interpret This Appendix

This appendix is intended to be read **after** the original `DATA_ANALYSIS_REPORT.pdf`:

1. The **original report** documents the historical data analysis and initial modeling path.  
2. This **appendix** explains why a new dataset was introduced, how it was constructed, and what performance gains it enabled.  
3. Together, they provide a complete story:
   - From small, heterogeneous diagnostic datasets  
   - To a large, purpose‑built risk dataset powering the current CardioDetect risk prediction system.

For detailed field‑level definitions, refer to the separate `DATA_DICTIONARY.pdf` deliverable.
