# CardioDetect

End-to-end cardiovascular risk and diagnostic system with:

- A **risk prediction arm** (10-year CHD risk on tabular data)
- An **OCR arm** that extracts values from lab PDFs and feeds the risk model
- A full set of notebooks and reports for analysis and publication

This README summarizes the **current, cleaned project structure** so you can quickly find data, code, notebooks, and reports.

---

## Project Structure

```text
CardioDetect/
  data/
    raw/               # Original / unmodified data (CSVs, images, PDFs)
    processed/         # Cleaned / merged intermediate datasets
    final/             # Modeling-ready tables (e.g. final_risk_dataset.csv)
    split/             # Train / validation / test CSV splits
    external/          # Metadata and info about external datasets
    merged/            # (Optional) combined tables if you create them
    synthetic_reports/ # (Optional) synthetic report inputs

  src/
    data_preprocessing.py   # Data cleaning / feature engineering utilities
    mlp_tuning.py           # Training & tuning of the MLP risk model
    models.py               # Model definitions / wrappers
    risk_thresholding.py    # Operating modes and threshold logic
    medical_ocr_optimized.py# Optimized OCR pipeline (digital + Tesseract)
    tesseract_ocr.py        # Tesseract-specific wrapper
    paddle_ocr.py           # PaddleOCR wrapper (where supported)
    olm_ocr.py              # olmOCR wrapper (GPU-only)
    ocr_manager.py          # Orchestrates OCR engines / fallback logic
    ocr_risk_prediction.py  # End-to-end OCR → risk pipeline entry point
    deepseek_ocr_wrapper.py # DeepSeek OCR integration (experimental)

  notebooks/
    00_complete_project_walkthrough.ipynb  # Full project narrative & demo
    01_data_overview.ipynb  # Data exploration & feature summary
    02_model_training.ipynb # Model comparison (LogReg, RF, GBM, MLP, etc.)
    03_ocr_pipeline.ipynb   # OCR + MLP risk prediction demo
    archive/                # Legacy / exploratory notebooks

  scripts/
    risk/                   # Risk-model CLI scripts
      run_risk_pipeline.py
      run_risk_accuracy.py
      run_extreme_optimization.py
      train_final_model.py

    ocr/                    # OCR engine tests and comparisons
      compare_ocr_engines.py
      compare_paddle_olm.py
      complete_accuracy_test.py

    reports/                # Report & visualization generators
      create_visual_journey*.py
      generate_report_plots.py
      generate_merged_report.py
      generate_v2_visuals.py
      generate_visualizations.py
      make_charts_pil.py

    notebooks/              # Scripts that create/update notebooks
      create_complete_demo.py
      create_risk_notebook.py
      update_risk_notebook.py
      fix_model_training_notebook.py

    dev/                    # Scratch / dev-only helpers
      test_viz.py

  reports/
    final/                  # Publication-quality PDFs and key PDFs
      CardioDetect_Milestone1_Report.pdf
      MILESTONE_2_REPORT.pdf
      TECHNICAL_SUMMARY.pdf
      OCR_IMPLEMENTATION_DETAILS.pdf
      DATA_DICTIONARY.pdf
      DATA_ANALYSIS_REPORT(!).pdf
      CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf

    # Experiment logs, tuning results, and status summaries
    DATA_ANALYSIS_REPORT_FINAL.pdf      # Updated/alternate data analysis
    DATA_ANALYSIS_REPORT_UPDATED.md     # Markdown source for updated report
    DATA_ANALYSIS_NEW_DATA_APPENDIX.md  # (If present) new risk-data appendix
    RISK_OPERATING_MODES_SUMMARY.md
    risk_model_status.md
    risk_modes_test_metrics.md
    mlp_baseline_metrics.md
    mlp_best_summary.md
    mlp_candidates_vs_baseline.md
    mlp_tuning_log.csv
    risk_threshold_sweep_validation.csv
    extreme_optimization_results.csv
    project_journey.pdf
    VISUAL_JOURNEY*.pdf
    figures/ images/ visualizations/ results/  # Generated assets

  docs/
    PROJECT_MASTER_DOC.md        # Complete project history & decisions
    PROJECT_SUMMARY.md           # High-level narrative of both arms
    DATA_INVENTORY.md            # Inventory of all datasets
    OCR_COMPARISON.md            # Tesseract vs others on CardioDetect data
    PADDLE_vs_OLM_COMPARISON.md  # Detailed comparison (where supported)

  models/
    # Serialized models (e.g. mlp_v2_best.pkl) live here

  output/
    # Legacy location for some CSVs; new work should prefer data/processed

  tests/
    # (Optional) test suite for src/ modules

  requirements.txt    # Python dependencies
  LICENSE
```

---

## Typical Workflows

### 1. Run OCR → Risk Prediction Pipeline

From the project root:

```bash
source .venv/bin/activate
python -m src.ocr_risk_prediction CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf
```

This will:

- Attempt fast digital extraction from the PDF
- Fall back to Tesseract OCR with OpenCV preprocessing if needed
- Parse structured CBC fields (age, sex, hemoglobin, WBC, RBC, platelets)
- Build a feature vector consistent with `mlp_tuning.py`
- Load the trained MLP model and output risk probability + risk level

### 2. Explore Data and Models in Notebooks

```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

Suggested order:

1. `00_complete_project_walkthrough.ipynb` – Full project narrative from start to finish
2. `01_data_overview.ipynb` – Understand the final risk dataset and features
3. `02_model_training.ipynb` – Compare baseline models and confirm MLP wins
4. `03_ocr_pipeline.ipynb` – Run the full OCR + risk pipeline interactively

### 3. Generate Reports (Optional)

Use the scripts under `scripts/reports/` to regenerate visual journeys and report figures, for example:

```bash
source .venv/bin/activate
python scripts/reports/create_visual_journey_v3.py
```

Outputs will appear under `reports/` and `reports/final/` depending on the script.
