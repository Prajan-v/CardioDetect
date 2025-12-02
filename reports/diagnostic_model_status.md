# Diagnostic Model Status (Frozen)

I treat my diagnostic arm as completely finished and locked. This arm predicts **current heart disease / diagnostic status**.

## Model Artifact

- **Model file:** `models/diagnostic_lightgbm.pkl`
- **Preprocessor:** `models/diagnostic_preprocessor.pkl`
- **Training script:** `scripts/train_final_model.py`
- **Training data:** `data/processed/diagnostic_2019.csv`

## Data Split

In my final diagnostic training script, I:
- Loaded `diagnostic_2019.csv`.
- Dropped the `data_source` column if present.
- Encoded any object columns with a label encoder.
- Split the data into **85% train** and **15% test** using `train_test_split` with `random_state=42` and stratification on the target.

## Final Test Metrics (Frozen)

From the final training run (recorded in `models/model_metadata.json`):

- **Accuracy:** 0.9241  (≈ 92.4%)
- **ROC-AUC:** 0.9431
- **Recall:** 0.8990
- **Precision:** 0.8990
- **Training samples:** 1,413
- **Test samples:** 303
- **Total dataset:** 2,019
- **Features:** 14

## Lock Status

For this Milestone, I treat this diagnostic model as **fully frozen**:

- I do **not** retrain it.
- I do **not** change its hyperparameters.
- I do **not** change its probability threshold.

All further work in this project only **reads** this diagnostic model; I treat it as the final diagnostic arm with ≈92.4% accuracy and strong recall.
