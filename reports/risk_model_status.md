# Risk Model Status (Frozen)

I treat a single frozen risk prediction pipeline as my **official risk prediction arm** for 10‑year CHD risk. All later work only changes thresholds and operating modes on top of this fixed model.

## Model Artifact

- **Model file:** `models/risk_model_pipeline.pkl`
- **Type:** Sklearn `Pipeline` wrapping preprocessing + a gradient boosting model (LightGBM)
- **Training script:** `scripts/run_risk_pipeline.py`
- **Source data:** `data/raw/framingham_raw.csv`

## Data Split Used

In the risk pipeline script, I:

1. Loaded the original Framingham data from `data/raw/framingham_raw.csv`.
2. Dropped rows with missing `TenYearCHD`.
3. Split the data into:
   - **70% Train**
   - **15% Validation**
   - **15% Test**
   using stratified `train_test_split` with `random_state=42`.
4. Applied a consistent feature‑engineering step (pulse pressure, MAP, BMI categories, age bands, risk flags, log transforms).
5. Trained an optimized LightGBM model with these features and saved the full pipeline as `models/risk_model_pipeline.pkl`.

## Baseline Test Metrics (Single Frozen Model)

Using the frozen pipeline **without retraining**, and recomputing probabilities on the Framingham test split, I can summarize a simple baseline at the standard threshold **0.50**:

- **Accuracy:** 0.8522
- **Recall:** 0.0521
- **Precision:** 0.6250
- **F1:** 0.0962
- **ROC‑AUC:** 0.6724

These numbers confirm that the underlying probability model has useful discriminative power (ROC‑AUC ≈ 0.67), but the raw 0.50 threshold is not a good operating point if I care about recall. That is why I define separate operating modes (Accuracy Mode and Balanced Mode) by sweeping thresholds on the validation set while keeping this **same frozen pipeline**.

## Lock Status

For Milestone 2 threshold and mode tuning I:

- Keep `models/risk_model_pipeline.pkl` **read‑only and frozen**.
- Do **not** retrain or replace the model.
- Only recompute probabilities and vary the decision threshold.

All thresholds and operating modes described in the risk threshold tuning notebook sit **on top of this single saved pipeline**.
