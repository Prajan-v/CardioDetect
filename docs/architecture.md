# CardioDetect — Architecture Deep Dive

> Full technical architecture for the system. See the [main README](../README.md) for a high-level overview.

---

## Hybrid OCR Pipeline

```mermaid
flowchart TD
    DOC[Input Document] --> TYPE{File Type?}
    TYPE -- PDF --> PDF_TEXT[Direct PDF Text Extraction]
    TYPE -- Image --> PREPROCESS[Image Preprocessing\nDeskew · Denoise · Binarize]
    TYPE -- Scanned PDF --> CONVERT[PDF → Image via pdf2image]

    PDF_TEXT --> PARSE[Regex Field Parser]
    PREPROCESS --> TESS[Tesseract OCR — Primary]
    CONVERT --> TESS

    TESS -- Low Confidence --> EASY[EasyOCR — Fallback]
    TESS -- High Confidence --> PARSE
    EASY --> PARSE

    PARSE --> FIELDS["Structured Fields\nAge · BP · Cholesterol · Glucose\nBMI · Smoking · Diabetes · CBC"]
    FIELDS --> CONF[Per-Field Confidence Scoring]
    CONF --> VALID[Missing Field Warnings]
    VALID --> OUT[OCRResult + Audit Metadata]
```

**Supported fields:**

| Category | Fields |
|----------|--------|
| Demographics | Age, Sex |
| Vitals | Systolic BP, Diastolic BP, Heart Rate |
| Lipid Panel | Total Cholesterol, HDL, LDL, Triglycerides |
| Metabolic | Fasting Glucose, BMI |
| CBC | Hemoglobin, WBC, RBC, Platelets |
| Lifestyle | Smoking Status, Diabetes Status |

---

## Data Pipeline

```mermaid
graph TD
    subgraph Sources
        F[Framingham\n5 dataset variants]
        N[NHANES 2013-14\nDemographic · Examination · Labs]
        U[UCI Heart Disease\nVA · Switzerland · Kaggle]
    end

    subgraph Processing
        P1[Auto Target Column Detection]
        P2[Cross-Dataset Feature Mapping\n40+ column alias mappings]
        P3[Physiological Range Validation]
        P4[Median Imputation]
        P5[Feature Engineering]
    end

    subgraph Splits
        S1[Train — 70%]
        S2[Validation — 15%]
        S3[Test — 15%]
    end

    F & N & U --> P1 --> P2 --> P3 --> P4 --> P5
    P5 --> S1 & S2 & S3
```

---

## Feature Engineering (34+ Features)

From raw clinical values, the pipeline derives the following:

| Category | Features |
|----------|---------|
| **Base Clinical** | Age, Sex, Systolic/Diastolic BP, Cholesterol (Total/HDL/LDL), Triglycerides, Fasting Glucose, BMI, Heart Rate, Smoking, Diabetes |
| **Derived Cardiovascular** | Pulse Pressure, Mean Arterial Pressure, LDL/HDL Ratio, Total/HDL Ratio |
| **Risk Flags** | Hypertension Flag, High Cholesterol Flag, High Glucose Flag, Obesity Flag, Metabolic Syndrome Score |
| **Age Groups (one-hot)** | <40, 40–49, 50–59, 60–69, 70+ |
| **Interaction Terms** | Age × Systolic BP, BMI × Glucose, Age × Smoking |
| **Log Transforms** | log(Total Cholesterol), log(Fasting Glucose), log(BMI) |

---

## Clinical Safety Override Rules

| Rule | Trigger | Effect |
|------|---------|--------|
| High-Risk Profile Guard | Age ≥ 65 + SBP ≥ 160 + Smoking + Diabetes | Force category → HIGH |
| Elderly Low-Risk Guard | Age ≥ 75 with very low numeric score | Upgrade → MODERATE |
| Missing Essentials Warning | Any critical field not extracted by OCR | Append clinical warning to output |

---

## Development Timeline

| Milestone | Deliverable |
|-----------|-------------|
| **M1** | EDA, multi-source data collection, preprocessing pipeline |
| **M2** | 38+ model variants trained, Voting Ensemble + XGBoost Regressor finalized, OCR engine built, V3 end-to-end pipeline |
| **M3** | Django REST backend + Next.js 14 frontend, user auth, prediction UI |
| **M4** | TechRxiv preprint publication, final comprehensive report |
