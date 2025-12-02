# MLP Experiments - Obsidian Workspace

This folder contains experiment notes for CardioDetect MLP model tuning.

## Structure

```
experiments_mlp/
├── README.md                    # This file
├── baseline_mlp_v2_best.md      # Frozen baseline model
├── mlp_exp_YYYYMMDD_HHMM_*.md   # Individual experiment notes
└── summaries/
    ├── mlp_leaderboard.md       # Ranked comparison of all experiments
    ├── threshold_tuning_notes.md # Notes on threshold optimization
    └── all_experiments.csv      # Full results for analysis
```

## Workflow

1. **Run experiments** via `notebooks/11_mlp_experiments.ipynb`
2. **Review leaderboard** in `summaries/mlp_leaderboard.md`
3. **Inspect individual experiments** by opening the corresponding `.md` file
4. **If a new leader emerges**, update the production model in `00_complete_project_walkthrough.ipynb`

## Baseline

- **Model**: `mlp_v2_best`
- **Test Accuracy**: 0.9359
- **Test Recall**: 0.9190
- **Threshold**: 0.5 (default)

## Constraint

All candidate models must satisfy:
- **Validation Recall ≥ 0.9190**
- **Test Recall ≥ 0.9190**

Only then is accuracy improvement considered valid.
