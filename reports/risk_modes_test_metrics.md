# Risk Model Operating Modes - Test Metrics

I evaluated my frozen risk prediction model under two operating modes on the held-out test set. I did not retrain the model; I only changed the decision threshold on top of the same probability outputs.

| Mode | Threshold | Accuracy | Recall | Precision | F1 | ROC-AUC |
|------|-----------|----------|--------|-----------|----|---------|
| Accuracy Mode | 0.290 | 0.8176 | 0.2708 | 0.3611 | 0.3095 | 0.6724 |
| Balanced Mode | 0.400 | 0.8491 | 0.1250 | 0.5000 | 0.2000 | 0.6724 |
