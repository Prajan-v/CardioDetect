# CardioDetect ML Models

## Production Models (`Final_models/`)

These are the **only** models used by the production pipeline.

### Detection (Heart Disease Status)
| File | Purpose | Accuracy |
|------|---------|----------|
| `detection_voting_optimized.pkl` | **Primary** - Voting ensemble  | **91.30%** |
| `detection_best.pkl` | XGBoost best single model | - |
| `detection_stacking.pkl` | Stacking ensemble (backup) | - |
| `detection_scaler_v3.pkl` | Feature scaler | - |
| `detection_features_v2.pkl` | Feature list | - |
| `detection_config_v3.pkl` | Configuration with threshold | - |

### Prediction (10-Year CHD Risk)
| File | Purpose | Accuracy |
|------|---------|----------|
| `prediction_xgb.pkl` | XGBoost regressor for 10-year risk | **91.63%** |
| `model_meta.json` | Model metadata | - |

---

## Archive (`archive/`)

Contains **experimental** and **deprecated** models:
- `classification/` - 38 classifier variants
- `alternatives/` - Alternative ensemble approaches
- `detection_backups/` - Old detection model versions
- `archive_detection/` - Previous detection experiments
- `regression/` - Regression models
- `experiments/` - Training experiments
- Various performance charts (.png)

> ⚠️ Do NOT use archive models in production.

---

## Last Updated
2025-12-19
