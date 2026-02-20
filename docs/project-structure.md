# CardioDetect — Project Structure

> Full directory breakdown. See the [main README](../README.md) for a quick overview.

---

```
CardioDetect/
│
├── Milestone_1/                        # EDA & Data Preprocessing
│   ├── data/                           # Raw datasets (Framingham, NHANES, UCI)
│   └── README.md
│
├── Milestone_2/                        # ML Model Development
│   ├── Source_Code/                    # Model training scripts
│   ├── Training/                       # Training runs & logs
│   ├── experiments/                    # Hyperparameter search results
│   ├── models/
│   │   ├── Final_models/               # ✅ Production models only
│   │   │   ├── detection_voting_optimized.pkl   # Primary — 91.30%
│   │   │   ├── detection_stacking.pkl           # Backup ensemble
│   │   │   ├── detection_best.pkl               # Best single model
│   │   │   ├── prediction_xgb.pkl               # 10-yr CHD — 91.63%
│   │   │   └── risk_regressor_v2.pkl            # Guideline-aligned model
│   │   └── archive/                    # ⚠️ Experimental — do NOT use in production
│   ├── ocr/                            # OCR engine source
│   ├── pipeline/                       # Integration pipeline
│   ├── clinical_guidelines/            # WHO/ACC/AHA reference data
│   └── reports/                        # Performance charts & evaluations
│
├── Milestone_3/                        # Full-Stack Web Application
│   ├── cardiodetect/                   # Django project (settings, URLs, WSGI)
│   ├── accounts/                       # User auth & profile management
│   ├── predictions/                    # Core prediction Django app
│   ├── services/                       # ML service integration layer
│   ├── templates/                      # Django HTML templates (20 pages)
│   ├── static/                         # CSS, JS, assets
│   ├── frontend/                       # Next.js 14 React app
│   ├── media/                          # Uploaded medical documents
│   └── manage.py
│
├── Milestone_4/                        # Research & Publication
│   ├── IEEE_Paper/                     # TechRxiv-formatted research paper
│   └── Final Report/                   # Comprehensive project report
│
├── src/                                # Core Python Library
│   ├── cardiodetect_v3_pipeline.py     # End-to-end V3 pipeline
│   ├── data_preprocessing.py           # Multi-source data pipeline
│   ├── models.py                       # ML model definitions
│   ├── production_pipeline.py          # Production-hardened pipeline wrapper
│   ├── production_model.py             # Production model interface
│   ├── risk_thresholding.py            # Risk categorization logic
│   ├── mlp_v3_ensemble.py              # MLP ensemble architecture
│   ├── guideline_risk.py               # Clinical guideline scorer
│   └── train_guideline_regressor_v2.py # Guideline regressor training
│
├── tests/                              # pytest test suite
├── results/                            # Model evaluation outputs
├── scripts/                            # Utility & automation scripts
├── docs/                               # Extended documentation
│   ├── architecture.md                 # OCR pipeline, data pipeline, diagrams
│   ├── project-structure.md            # This file
│   └── api-reference.md                # REST API endpoints
├── requirements.txt
└── start.sh                            # One-command startup
```
