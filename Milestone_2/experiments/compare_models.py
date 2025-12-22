"""
Compare all trained models.

Loads metrics from all saved model runs and generates a comparison table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils_io import REPORTS_DIR, save_comparison_table


# ============================================================================
# MAIN COMPARISON ROUTINE
# ============================================================================

def compare_models():
    """Load all metrics and generate comparison table."""
    print("=" * 80)
    print("MILESTONE 2: MODEL COMPARISON")
    print("=" * 80)

    metrics_dir = REPORTS_DIR / "metrics"

    if not metrics_dir.exists():
        print("No metrics found. Run training scripts first.")
        return

    # Load all metrics files
    metrics_files = list(metrics_dir.glob("*_metrics.json"))
    print(f"\nFound {len(metrics_files)} metric files")

    results = []
    for mf in sorted(metrics_files):
        with open(mf, "r") as f:
            data = json.load(f)
            results.append(data)

    if not results:
        print("No results to compare.")
        return

    # Build comparison dataframe
    df = pd.DataFrame(results)

    # Select key columns for display
    key_cols = [
        "model_name",
        "target_type",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "roc_auc_ovr",
    ]

    # Filter to existing columns
    display_cols = [c for c in key_cols if c in df.columns]
    df_display = df[display_cols].copy()

    # Sort by F1 score
    if "f1" in df_display.columns:
        df_display = df_display.sort_values("f1", ascending=False)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    print(df_display.to_string(index=False))

    # Save comparison
    comparisons_dir = REPORTS_DIR / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    save_comparison_table(results, "all_models_comparison.json", comparisons_dir)

    # Save as CSV
    csv_path = comparisons_dir / "all_models_comparison.csv"
    df_display.to_csv(csv_path, index=False)
    print(f"\nSaved comparison CSV: {csv_path}")

    # ------------------------------------------------------------------
    # Summary by model type
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY BY TARGET TYPE")
    print("=" * 80)

    if "target_type" in df.columns and "f1" in df.columns:
        for target_type in df["target_type"].unique():
            subset = df[df["target_type"] == target_type]

            # Skip target types with no valid F1 (e.g., continuous regressors)
            if "f1" not in subset.columns or not subset["f1"].notna().any():
                print(f"\n{target_type.upper()}:")
                print("  No F1 metric available for this target type (skipping).")
                continue

            valid_subset = subset[subset["f1"].notna()]
            best_row = valid_subset.loc[valid_subset["f1"].idxmax()]
            print(f"\n{target_type.upper()}:")
            print(f"  Best model: {best_row.get('model_name', 'N/A')}")
            print(f"  F1 (macro): {best_row.get('f1', 0):.4f}")
            print(f"  Accuracy: {best_row.get('accuracy', 0):.4f}")

    # ------------------------------------------------------------------
    # Best overall model
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BEST OVERALL MODEL")
    print("=" * 80)

    if "f1" in df.columns:
        df_with_f1 = df[df["f1"].notna()]
        if not df_with_f1.empty:
            best_overall = df_with_f1.loc[df_with_f1["f1"].idxmax()]
            print(f"Model: {best_overall.get('model_name', 'N/A')}")
            print(f"Target Type: {best_overall.get('target_type', 'N/A')}")
            print(f"F1 (macro): {best_overall.get('f1', 0):.4f}")
            print(f"Accuracy: {best_overall.get('accuracy', 0):.4f}")
            print(f"Precision: {best_overall.get('precision', 0):.4f}")
            print(f"Recall: {best_overall.get('recall', 0):.4f}")
        else:
            print("No valid F1 metrics available across models.")

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print(f"Results saved to: {comparisons_dir}")
    print("=" * 80)

    return df


if __name__ == "__main__":
    compare_models()
