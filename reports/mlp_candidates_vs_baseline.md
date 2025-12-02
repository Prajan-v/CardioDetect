# MLP Candidates vs Baseline Comparison

This table compares the locked baseline MLP against tuned candidates.

## Test Set Metrics (threshold = 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Baseline_MLP** | 0.9082 | 0.7869 | 0.8466 | 0.8156 | 0.9588 |
| Candidate_1 | 0.9301 | 0.8246 | 0.9000 | 0.8607 | 0.9725 |
| Candidate_2 | 0.9339 | 0.8201 | 0.9276 | 0.8706 | 0.9699 |
| Candidate_3 | 0.9359 | 0.8315 | 0.9190 | 0.8731 | 0.9673 |

## Selection Result

**Candidate_3** is selected as the new best model.

- Accuracy improvement: +0.0277
- Recall change: +0.0724
