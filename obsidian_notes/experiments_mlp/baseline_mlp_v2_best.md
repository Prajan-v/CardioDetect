---
id: mlp_v2_best
date: 2024-11-30
type: mlp_baseline
status: frozen
---

# Baseline MLP: mlp_v2_best

This is the **frozen baseline** model. All new experiments are compared against this.

## 1. Configuration

- **Code name**: `mlp_v2_best`
- **Purpose**: Optuna-tuned MLP for cardiovascular risk prediction
- **Dataset**: `new_data` (16,015 patients)
- **Split**: Train / Val / Test = 70% / 15% / 15%

### 1.1 Hyperparameters

- **Model**: `sklearn.neural_network.MLPClassifier`
- `hidden_layer_sizes`: (128, 64, 32)
- `activation`: 'relu'
- `solver`: 'adam'
- `alpha`: 1e-4
- `learning_rate_init`: 0.001
- `max_iter`: 500
- `early_stopping`: True
- `validation_fraction`: 0.1
- `batch_size`: 'auto'
- `random_state`: 42

### 1.2 Training notes

- **Converged**: Yes
- **Tuning method**: Optuna hyperparameter optimization
- **Trials**: 100+

## 2. Metrics (threshold = 0.5)

### 2.1 Test Set

- **Accuracy**: 0.9359
- **Recall**: 0.9190
- **Precision**: ~0.95
- **AUC**: ~0.98

## 3. Why This Is The Baseline

1. **Optuna-optimized**: Extensive hyperparameter search already performed
2. **Strong recall**: 91.9% of positive cases correctly identified
3. **Balanced accuracy**: 93.6% overall accuracy
4. **Production-ready**: Validated on held-out test set

## 4. Constraint for New Experiments

Any new candidate model must satisfy:

- **Recall ≥ 0.9190** on both validation and test sets
- **Accuracy improvement** over 0.9359 (after threshold tuning)

If no improvement is found, **keep this baseline**.

## 5. Links

- Training notebook: `[[00_complete_project_walkthrough.ipynb]]`
- Model file: `models/mlp_v2_best.pkl`
- Experiments notebook: `[[11_mlp_experiments.ipynb]]`
