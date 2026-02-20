<div align="center">

<h1>CardioDetect</h1>

<p><strong>Clinical Decision Support System for Cardiovascular Risk Assessment</strong></p>

<p>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" alt="License"></a>
  <a href="https://doi.org/10.36227/techrxiv.177154153.36052407/v1"><img src="https://img.shields.io/badge/Preprint-TechRxiv-0066cc?style=flat-square&logo=ieee&logoColor=white" alt="TechRxiv"></a>
  <a href="https://orcid.org/0009-0008-3295-2950"><img src="https://img.shields.io/badge/ORCID-0009--0008--3295--2950-A6CE39?style=flat-square&logo=orcid&logoColor=white" alt="ORCID"></a>
  <img src="https://img.shields.io/badge/build-passing-brightgreen?style=flat-square" alt="Build Status">
  <img src="https://img.shields.io/badge/coverage-84%25-brightgreen?style=flat-square" alt="Coverage">
</p>

CardioDetect is a research-driven, production-oriented clinical decision support system. It maps raw medical documents (PDFs, images) to structured cardiovascular risk assessments using a hybrid OCR pipeline and a dual-engine machine learning architecture.

> **Disclaimer:** Not a substitute for professional medical diagnosis. Intended for research and clinical decision-support prototyping only.

---

## Performance

| Engine | Task | Result |
|--------|------|--------|
| Detection Engine (Voting Ensemble) | Binary disease detection | **91.30% accuracy** |
| Prediction Engine (XGBoost Regressor) | 10-year CHD risk stratification | **91.63% category agreement** |
| Universal OCR Engine | Clinical field extraction | PDF · Image · Text |
| Feature pipeline | Engineered clinical vector | 34+ features |

*The Detection Engine was trained on the UCI Heart Disease dataset, while the Prediction Engine was trained on an extended 16,000+ row dataset harmonizing the Framingham Heart Study with Kaggle cardiovascular cohorts. (NHANES was evaluated for feature enrichment but excluded from supervised training due to lacking a 10-year CHD target).*

## Model Validation & Reproducibility

### Evaluation Methodology
Models were evaluated using a strict **70/15/15 stratified split** to preserve minority-class target distributions. 
- **Harmonization:** Cross-dataset standardization aligned 40+ clinical aliases across multiple epidemiological sources.
- **Grounding:** 10-year CHD risk agreement is measured directly against **2013 ACC/AHA Pooled Cohort Equations** and **D'Agostino et al. (2008)** clinical thresholds.
- **Explainability:** Feature attributions trace risk classifications back to primary clinical inputs (e.g., elevated SBP, lipid profile abnormalities).

<img src="docs/assets/analytics_1.png" width="800" alt="CardioDetect Explainability Dashboard" />

### Reproducing the Pipeline
Training pipelines and hyperparameter search spaces (Optuna) are preserved in the respective Milestone directories.

To reproduce the optimal model weights:
1. Fetch the raw datasets (see [Dataset Credits](#dataset-credits-citations)).
2. Execute the inference/harmonisation notebooks (`Milestone_1/`).
3. Run the ensemble training processes (`Milestone_2/`) to regenerate `.pkl` artifacts.

---

## Architecture

```mermaid
graph TB
    A([Medical Document]) --> B[Hybrid OCR Engine<br/>Tesseract · OpenCV · EasyOCR]
    B --> C[Feature Engineering<br/>34-feature clinical vector]
    C --> D1[Detection Engine<br/>Voting Ensemble · 91.30%]
    C --> D2[Prediction Engine<br/>XGBoost Regressor · 91.63%]
    D1 & D2 --> E[Clinical Safety Layer<br/>Override rules · Explainability]
    E --> F([Risk Report — LOW · MODERATE · HIGH])
```

### Dual-Engine Design

```mermaid
graph LR
    subgraph Detection["Binary Detection"]
        direction TB
        A1[XGBoost] & A2[LightGBM] & A3[MLP] & A4[Random Forest] --> V[Soft Voting Ensemble<br/>91.30% Accuracy]
    end
    subgraph Prediction["Continuous Prediction"]
        direction TB
        B1[XGBoost Regressor] --> R[10-Year CHD Risk %<br/>91.63% Category Agreement]
    end
    V & R --> OUT([LOW · MODERATE · HIGH])
```

The dual-engine design separates binary disease detection from long-term risk prediction, enabling modular optimisation and clearer clinical interpretation. The outputs are fused through a clinical safety layer before the final classification is emitted.

---

## Quickstart

**Requirements:** Python 3.10+, Tesseract OCR

```bash
# 1. Clone and install
git clone https://github.com/Prajan-v/CardioDetect.git
cd CardioDetect
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract        # macOS — apt install tesseract-ocr on Ubuntu

# 2. Run a prediction
python -c "
from src.cardiodetect_v3_pipeline import CardioDetectV3
result = CardioDetectV3().run('lab_report.pdf')
print(result['risk_category'], f\"{result['risk_score']*100:.1f}%\")
"
```

**Web application** (Django + Next.js):

```bash
cd Milestone_3 && python manage.py runserver      # API  → localhost:8000
cd Milestone_3/frontend && npm i && npm run dev   # UI   → localhost:3000
```

> **Reproducibility:** To reproduce training results, run the notebooks located in `Milestone_1/` and `Milestone_2/`.

<div align="center">
  <img src="docs/assets/ui_1.png" width="800" alt="CardioDetect Risk Assessment Dashboard" style="margin-bottom: 20px;" />
  <img src="docs/assets/ui_2.png" width="800" alt="Patient Clinical History" style="margin-bottom: 20px;" />
  <img src="docs/assets/ui_3.png" width="800" alt="Detailed Laboratory Report Output" />
</div>

---

## Output

```json
{
  "risk_category": "MODERATE",
  "risk_score": 0.327,
  "recommendation": "Moderate cardiovascular risk (32.7%). Consult a healthcare provider for risk factor management.",
  "explanations": {
    "top_reasons": [
      "Systolic blood pressure 148 mmHg is elevated.",
      "Total cholesterol 225 mg/dL is borderline high.",
      "Age 58 years contributes to elevated cardiovascular risk."
    ]
  },
  "ocr_confidence": { "average": 0.89 },
  "audit": { "engine": "tesseract_ocr", "model_version": "risk_regressor_v2" }
}
```

---

## Project Layout

```
CardioDetect/
├── Milestone_1/        EDA and data preprocessing
├── Milestone_2/        ML model development and OCR pipeline
├── Milestone_3/        Django REST backend + Next.js frontend
├── Milestone_4/        Research paper and final report
├── src/                Core library — pipeline, models, preprocessing
├── tests/              pytest test suite
├── docs/               Architecture, API reference, project structure
└── results/            Model evaluation outputs
```

Extended documentation → [`/docs`](docs/)

---

## Dataset Credits & Citations

This pipeline harmonizes three publicly available epidemiological datasets. If adapting this work, please respect the usage agreements of the original authors:

1. **Framingham Heart Study:** Provided via the NHLBI Biologic Specimen and Data Repository.
2. **NHANES 2013–2014:** National Health and Nutrition Examination Survey, curated by the Centers for Disease Control and Prevention (CDC).
3. **UCI Heart Disease Repository:** Includes cohorts from Cleveland Clinic, Hungarian Institute of Cardiology, V.A. Medical Center (Long Beach), and University Hospital (Zurich). [Link](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## Limitations

- Trained on publicly available epidemiological datasets; not validated in clinical deployment
- OCR field extraction quality depends on document scan resolution
- Model demographics reflect Framingham/NHANES/UCI populations and may not generalise universally

---

## Publication

CardioDetect is formally published as a preprint on **TechRxiv**, IEEE's preprint platform for technology research.

**Prajan Narayanan V** — *CardioDetect: An Integrated Clinical Decision Support System for Cardiovascular Risk Assessment Using Dual-Engine ML and Hybrid OCR Pipeline* — TechRxiv, 2026.

DOI: [10.36227/techrxiv.177154153.36052407/v1](https://doi.org/10.36227/techrxiv.177154153.36052407/v1)

---

## License

Released under the [MIT License](LICENSE).
