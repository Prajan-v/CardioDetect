# MLP Baseline Metrics

These are the test-set metrics for the **locked baseline MLP** model.

## Test Set Performance (threshold = 0.5)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9082 |
| Precision | 0.7869 |
| Recall | 0.8466 |
| F1 Score | 0.8156 |
| ROC-AUC | 0.9588 |

## Confusion Matrix

```
              Predicted
              Neg    Pos
Actual Neg [[ 1706   133]
       Pos  [   89   491]]
```

## Model Details

- **Architecture:** (128, 64, 32) hidden layers
- **Activation:** ReLU
- **Optimizer:** Adam (lr=0.001)
- **Early stopping:** Yes (validation_fraction=0.1)
- **Lock status:** This model is frozen and will not be modified.
