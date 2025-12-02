# Data Analysis Report (Updated)

> [!NOTE]
> This document serves as an update to the original Data Analysis Report. It includes a reference to the original findings and appends a new section detailing the CardioDetect Risk Dataset (v2) and the latest model performance.

## Part 1: Original Report

The original data analysis and initial modeling findings are documented in the following PDF:

[DATA_ANALYSIS_REPORT.pdf](file:///Users/prajanv/CardioDetect/pdf/DATA_ANALYSIS_REPORT(!).pdf)

---

## Part 2: New Data Appendix

# Detailed Rationale for Switching to the New Risk Dataset

The original data analysis presented in `DATA_ANALYSIS_REPORT.pdf` relied on legacy public datasets such as the UCI Heart Disease repository, early Framingham subsets, and other small, merged cohorts. While these datasets served as an initial testbed for our modeling pipeline, they ultimately proved insufficient for the ambitious goals of the CardioDetect 10-year risk prediction system.

We transitioned to this new, purpose-built **CardioDetect Risk Dataset (v2)** for four critical reasons:

### 1. Fundamental Shift from Diagnosis to Prognosis
The legacy datasets were primarily designed for **diagnostic classification**—determining whether a patient *currently* has heart disease (presence/absence). However, our core objective is **prognostic modeling**: predicting the *probability* of a cardiovascular event over the next **10 years**.
- **Old Data**: "Is the patient sick right now?" (Binary classification of current state).
- **New Data**: "What is the likelihood of an event by 2035?" (Long-term risk estimation).
This requires a longitudinal study design with clear "time-to-event" or "10-year outcome" labels, which the diagnostic datasets lacked.

### 2. Overcoming Sample Size Limitations
Modern machine learning models, particularly deep neural networks like the MLP we are deploying, require substantial data to generalize well and avoid overfitting.
- The original combined cohorts totaled only **≈5,000–7,000** patients.
- The new dataset contains **16,123** patients.
This **>2x increase** in sample size allows us to:
- Capture subtle non-linear interactions between risk factors.
- Achieve stable convergence during training.
- reliably model rare events in the positive class (CHD events).

### 3. Data Harmonization and Quality
The previous analysis suffered from **heterogeneous schemas**. Merging disparate studies (e.g., Cleveland, Hungarian, Long Beach) resulted in:
- Inconsistent feature definitions (e.g., different units for cholesterol, different categorizations for chest pain).
- High rates of missing values for critical biomarkers.
- Loss of information due to aggressive simplification during merging.

The new dataset is the result of a rigorous **harmonization process** integrating the Framingham Heart Study, NHANES, and custom clinical records into a single, unified schema. Every feature has been standardized, validated, and cleaned to ensure high data quality.

### 4. Breaking the Performance Ceiling
Models trained on the legacy data plateaued at approximately **85–86% accuracy**. To push performance beyond this threshold—specifically to achieve our target of >90% accuracy with high recall—we needed richer features.
The new dataset includes **34 features**, compared to the ~14 standard features in the old datasets. Crucially, it adds:
- **Derived Clinical Scores**: Pulse pressure, mean arterial pressure, metabolic syndrome scores.
- **Interaction Terms**: Explicit features capturing the interplay between age, blood pressure, and smoking.
- **Clinical Flags**: Pre-computed risk flags (e.g., hypertension, obesity) that align with clinical guidelines.

In summary, the switch to the CardioDetect Risk Dataset (v2) was not just an update; it was a necessary evolution to build a clinically relevant, high-performance risk prediction tool.

---

## 2. Overview of the New CardioDetect Risk Dataset

### 2.1 High‑Level Summary

| Attribute       | Value         |
|----------------|---------------|
| Total Patients | **16,123**    |
| Total Features | **34**        |
| Target         | `risk_target` (10‑year CHD event: 0/1) |
| Class Balance  | ~76% negative / 24% positive |
| Splits         | 70% train / 15% val / 15% test (stratified) |

### 2.2 Data Sources

The dataset integrates multiple established sources into a single, clean table:

| Source                  | Approx. Patients | Description                    |
|-------------------------|------------------|--------------------------------|
| Framingham Heart Study  | ~4,000           | Longitudinal CHD cohort       |
| NHANES                  | ~10,000          | National health & nutrition survey |
| Custom Clinical Records | ~2,000           | Supplementary anonymized records |

All sources were harmonized into a consistent schema and stored as:

- Final dataset: [data/final/final_risk_dataset.csv](file:///Users/prajanv/CardioDetect/data/final/final_risk_dataset.csv)  
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

Feature engineering is implemented in the main CardioDetect codebase and documented in `DATA_DICTIONARY.pdf`.

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

This brings the dataset closer to how clinicians reason about cardiovascular risk and aligns with common risk calculators (Framingham, ASCVD), while adding derived features those calculators do not expose directly.

### 3.3 Clean, Reproducible Splits

The new data pipeline:

- Uses a **fixed random seed** and **stratified splitting** to keep class balance consistent across train/val/test.  
- Stores the exact splits in `data/split/` for reproducibility.  
- Ensures that all model training and tuning (e.g., in [mlp_tuning.py](file:///Users/prajanv/CardioDetect/src/mlp_tuning.py)) uses the same split definitions.

This is a significant improvement over earlier ad‑hoc splits in the legacy experiments.

---

## 4. New Model Performance on the CardioDetect Risk Dataset

On the new 16,123‑patient dataset, the final chosen model is a **Multi‑Layer Perceptron (MLP)** trained and tuned in [src/mlp_tuning.py](file:///Users/prajanv/CardioDetect/src/mlp_tuning.py).

### 4.1 Test Set Metrics (n ≈ 2,418)

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | **93.59%** |
| Precision     | 83.15%    |
| Recall        | 91.90%    |
| F1‑Score      | 0.8731    |
| ROC‑AUC       | **0.9673** |

**Confusion Matrix (Test Set):**

- True Negatives (TN): 1,731  
- False Positives (FP): 108  
- False Negatives (FN): 47  
- True Positives (TP): 533  

#### Visualizations

![Confusion Matrix](file:///Users/prajanv/CardioDetect/reports/figures/confusion_matrix_v2.png)

![ROC Curve](file:///Users/prajanv/CardioDetect/reports/figures/roc_curve_v2.png)

These results indicate a **high‑performing risk model** that prioritizes sensitivity (recall) while maintaining strong overall accuracy.

For detailed field‑level definitions, refer to the separate `DATA_DICTIONARY.pdf` deliverable.
