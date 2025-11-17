# Data and Transformations

## Datasets
- Source: UCI Heart Disease (Cleveland subset)
- Records processed: 303
- Target mapping: original `num` > 0 -> `target` = 1, else 0
- Class balance: see docs/target_distribution.csv

## Feature Space
- Numeric base: age, trestbps, chol, thalach, oldpeak, ca
- Categorical base: sex, cp, fbs, restecg, exang, slope, thal
- Engineered:
  - age_decade = floor(age / 10)
  - is_hypertensive = 1 if trestbps >= 140 else 0
  - is_hyperlipidemic = 1 if chol >= 240 else 0
  - st_depr_high = 1 if oldpeak >= 2.0 else 0
- Final one-hot feature names: docs/feature_names.txt

## Preprocessing Pipeline
- Numeric: SimpleImputer(strategy=median) -> StandardScaler()
- Categorical: SimpleImputer(strategy=most_frequent) -> OneHotEncoder(handle_unknown=ignore)
- Implemented via ColumnTransformer + Pipeline
- Saved transformer: models/preprocessor.joblib

## Splitting
- Stratified train/val/test = 70/15/15 (random_state=42)
- Files: data/processed/train.csv, val.csv, test.csv

## OCR Integration
- Ingestion: PDFs rendered with PyMuPDF; images read via OpenCV
- Image preprocessing: grayscale -> Gaussian blur -> adaptive threshold
- OCR: pytesseract; parsing via regex with unit normalization
- Extracted fields: bp_sys, bp_dia, chol, hdl, ldl, triglycerides, glucose
- Row append: OCR dict -> row -> feature engineering -> preprocessor.transform -> append to processed.csv -> processed_with_ocr.csv
- Missing fields allowed; imputed by the transformer

## Artifacts
- Clean dataset: data/processed/processed.csv
- With OCR row appended: data/processed/processed_with_ocr.csv
- EDA: docs/eda_distributions.png, docs/eda_correlation.png, docs/eda_counts.png, docs/eda_summary.csv, docs/target_distribution.csv
- Sample docs: docs/sample_report.png, docs/sample_report.pdf
- OCR outputs: docs/ocr_output.json, docs/ocr_output_pdf.json
