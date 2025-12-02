# Global Model & Ensemble Search Summary

Baseline (frozen) **mlp_v2_best** – Test Acc=0.9359, Test Recall=0.9190 (at threshold 0.5)

## Models Evaluated (tuned thresholds)

| Name | Kind | Val Acc | Val Rec | Test Acc | Test Rec | Test AUC | Meets Constraint? | Beats Baseline? |
|------|------|---------|---------|----------|----------|----------|-------------------|-----------------|
| mlp_v2_best | baseline_mlp | 0.9529 | 0.9810 | 0.9310 | 0.9345 | 0.9673 | ✅ | ❌ |
| LogisticRegression_balanced | single | 0.5678 | 0.9241 | 0.5639 | 0.9017 | 0.8024 | ❌ | ❌ |
| RandomForest_default_tuned | single | 0.8131 | 0.9241 | 0.8222 | 0.9414 | 0.9042 | ✅ | ❌ |
| GradientBoosting_default_tuned | single | 0.8462 | 0.9207 | 0.8483 | 0.9276 | 0.9299 | ✅ | ❌ |
| XGBoost_default_tuned | single | 0.8730 | 0.9224 | 0.8768 | 0.9259 | 0.9356 | ✅ | ❌ |
| LightGBM_default_tuned | single | 0.9069 | 0.9241 | 0.9029 | 0.8983 | 0.9452 | ❌ | ❌ |
| SVM_RBF_balanced | single | 0.8044 | 0.6000 | 0.8107 | 0.6155 | 0.8348 | ❌ | ❌ |
| Ensemble_RF_XGB_LGBM_MLP_soft | ensemble | 0.9115 | 0.9466 | 0.9074 | 0.9276 | 0.9547 | ✅ | ❌ |

## Recommendation

No model or ensemble strictly outperformed `mlp_v2_best` under the recall constraint (Recall ≥ 0.9190 on val & test).

**Verdict:** Keep `mlp_v2_best` as the production model.
