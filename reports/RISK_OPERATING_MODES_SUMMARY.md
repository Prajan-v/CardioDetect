# Risk Operating Modes – Summary

In this section I describe how I turned my frozen risk prediction model into several practical operating modes. I did not retrain the model at all. I only changed the **probability threshold** used to convert risk scores into binary decisions.

---

## 1. Fixed Diagnostic Arm (Context)

I treat my **diagnostic arm** (current disease status) as fully complete and frozen:

- Final diagnostic model: a LightGBM classifier saved as `models/diagnostic_lightgbm.pkl` with its preprocessor in `models/diagnostic_preprocessor.pkl`.
- Trained on `data/processed/diagnostic_2019.csv` with an 85/15 train–test split.
- Final test performance is about **92.4% accuracy**, with high ROC‑AUC and strong recall.

I do **not** modify this diagnostic model, its data, or its thresholds in this milestone.

---

## 2. Frozen Risk Prediction Arm

For **10‑year CHD risk prediction**, I keep a single frozen model:

- Frozen risk pipeline: `models/risk_model_pipeline.pkl`.
- This pipeline was trained on Framingham data (`data/raw/framingham_raw.csv`) with a 70/15/15 train–validation–test split and a consistent feature‑engineering step.
- It is stored as a full sklearn `Pipeline` (preprocessing + LightGBM classifier).

All of my work in this threshold tuning step uses **only this saved pipeline**. I reload it from disk, compute predicted probabilities on the existing validation and test splits, and then search for better thresholds.

---

## 3. Threshold Sweep on the Validation Set

On the validation split, I:

1. Loaded the frozen pipeline from `models/risk_model_pipeline.pkl`.
2. Recreated the original Framingham train/val/test splits.
3. Applied the same feature engineering used during training.
4. Computed predicted probabilities for the positive class (10‑year CHD event).
5. Swept thresholds from **0.05** to **0.95** in steps of **0.01**.

For each threshold I recorded:

- Accuracy
- Recall (sensitivity) for the positive class
- Precision
- F1 score
- ROC‑AUC (unchanged across thresholds)

I saved the full sweep as:

- `reports/risk_threshold_sweep_validation.csv`

This table lets me see exactly how accuracy and recall trade off as I move the threshold.

---

## 4. Mode A – Accuracy Mode

**Goal:** Maximize accuracy, but avoid a degenerate model that predicts almost all patients as low risk.

Selection rule on the **validation set**:

- Search all thresholds between 0.05 and 0.95.
- Keep only thresholds where **recall ≥ 0.30**.
- Among those, choose the threshold with the **highest accuracy**.

The chosen validation operating point was:

- **Accuracy Mode threshold:** ≈ **0.29**
- Validation performance at this threshold (from the sweep):
  - Accuracy ≈ **0.8491**
  - Recall ≈ **0.42**
  - Precision ≈ **0.51**
  - F1 ≈ **0.46**

When I apply this same threshold once to the **held‑out test set**, I obtain:

- **Accuracy Mode – Test Metrics** (from `reports/risk_modes_test_metrics.md`):
  - Threshold: **0.290**
  - Accuracy: **0.8176**
  - Recall: **0.2708**
  - Precision: **0.3611**
  - F1: **0.3095**
  - ROC‑AUC: **0.6724**

This mode leans toward overall accuracy and keeps recall above zero, but it does not achieve very high sensitivity. It behaves more like a conservative risk flagger.

---

## 5. Mode B – Balanced Mode

**Goal:** Find a more balanced point where accuracy stays solid while recall does not collapse.

Selection rule on the **validation set**:

- Filter thresholds to those with **accuracy ≥ 0.85**.
- Among those, choose the threshold with the **highest F1 score** (and, secondarily, higher recall and accuracy).

The chosen validation operating point was:

- **Balanced Mode threshold:** ≈ **0.40**
- Validation performance:
  - Accuracy ≈ **0.8506**
  - Recall ≈ **0.19**
  - Precision ≈ **0.53**
  - F1 ≈ **0.27**

On the **held‑out test set**, using the same threshold exactly once, I obtain:

- **Balanced Mode – Test Metrics**:
  - Threshold: **0.400**
  - Accuracy: **0.8491**
  - Recall: **0.1250**
  - Precision: **0.5000**
  - F1: **0.2000**
  - ROC‑AUC: **0.6724**

In this particular dataset, the Balanced Mode still ends up relatively conservative in terms of recall, but it provides a clearer trade‑off: it keeps accuracy high while allowing some true positives to be detected.

---

## 6. Optional High‑Confidence Mode (Subset Only)

I also defined a **high‑confidence mode** that does not change the underlying model, but looks only at patients where the risk score is very extreme:

- **High‑confidence negatives:** predicted probability `< 0.10`
- **High‑confidence positives:** predicted probability `> 0.90`
- All patients in the middle band `[0.10, 0.90]` are ignored in this view.

On this high‑confidence subset of the **test** set I measured:

- Coverage: about **42.8%** of test patients fall into this band.
- Accuracy: **0.9301**
- Recall: **0.0000**
- Precision: **0.0000**
- F1: **0.0000**

This confirms that when the model is extremely confident it is usually correct on negatives, but it almost never emits very high probabilities for positives on this particular dataset with the current calibration. As a result, the high‑confidence band has very good accuracy overall, but it does not help much with detecting positive cases.

I treat this as a **diagnostic slice**, not as a deployment mode: it shows how the model behaves at the extremes, and it explains why the accuracy in this narrow subset can be higher than the overall test accuracy while recall and coverage are poor.

---

## 7. Final Position

For Milestone 2 I:

- Keep my **diagnostic LightGBM model** fully frozen as the final diagnostic arm (≈92.4% accuracy).
- Keep a single **frozen risk prediction pipeline** (`models/risk_model_pipeline.pkl`) as the official risk arm model.
- Define multiple **operating modes** (Accuracy Mode, Balanced Mode, and an optional High‑Confidence analysis) by changing only the **decision threshold** on top of the same probability outputs.

This setup keeps the modelling work honest: I am not chasing metrics by retraining over and over. Instead, I fix the models and carefully document how different thresholds and modes behave on the existing validation and test splits.
