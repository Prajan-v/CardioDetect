from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


DATA_SPLIT_DIR = Path(__file__).parent.parent / "data" / "split"


def compute_risk_score(row: pd.Series) -> int:
    """Compute heuristic risk score (0-10) for a patient.

    Rules:
      - Age >65: +2
      - Systolic BP >160: +2, >140: +1
      - Cholesterol >240: +2, >200: +1
      - Glucose >126 or diabetes: +2
      - BMI >30: +1
      - Smoking: +1
    """
    score = 0

    age = row.get("age", np.nan)
    try:
        if float(age) > 65:
            score += 2
    except Exception:
        pass

    sbp = row.get("systolic_bp", np.nan)
    try:
        sbp = float(sbp)
        if sbp > 160:
            score += 2
        elif sbp > 140:
            score += 1
    except Exception:
        pass

    # Total cholesterol may be stored as total_cholesterol or cholesterol
    chol = row.get("total_cholesterol", np.nan)
    if pd.isna(chol):
        chol = row.get("cholesterol", np.nan)
    try:
        chol = float(chol)
        if chol > 240:
            score += 2
        elif chol > 200:
            score += 1
    except Exception:
        pass

    glu = row.get("fasting_glucose", np.nan)
    diabetes = row.get("diabetes", 0)
    try:
        glu_val = float(glu)
    except Exception:
        glu_val = np.nan
    try:
        diab_val = float(diabetes)
    except Exception:
        diab_val = 0.0

    if (not np.isnan(glu_val) and glu_val > 126) or diab_val > 0:
        score += 2

    bmi = row.get("bmi", np.nan)
    try:
        if float(bmi) > 30:
            score += 1
    except Exception:
        pass

    smoking = row.get("smoking", 0)
    try:
        if float(smoking) > 0:
            score += 1
    except Exception:
        pass

    return int(score)


def assign_risk_class(score: int) -> int:
    """Map risk score to 3-class label.

    - score <= 3  -> 0 (LOW)
    - 4 <= score <= 6 -> 1 (MEDIUM)
    - score >= 7 -> 2 (HIGH)
    """
    if score <= 3:
        return 0
    if score <= 6:
        return 1
    return 2


def get_event_indicator_columns(df: pd.DataFrame) -> List[str]:
    """Return possible event indicator columns present in the dataframe."""
    candidates = ["risk_target", "target", "TenYearCHD"]
    return [c for c in candidates if c in df.columns]


def upgrade_high_if_event(df: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Upgrade any patient with an actual event to HIGH (2).

    If any of the known event columns is >0, force risk_class_3 = 2.
    """
    event_cols = get_event_indicator_columns(df)
    if not event_cols:
        return labels

    labels = labels.copy()
    # Event if any of the candidate columns is >0
    event_mask = np.zeros(len(df), dtype=bool)
    for col in event_cols:
        try:
            event_mask |= df[col].astype(float).values > 0
        except Exception:
            continue

    labels[event_mask] = 2
    return labels


def process_split(name: str) -> None:
    in_path = DATA_SPLIT_DIR / f"{name}.csv"
    out_path = DATA_SPLIT_DIR / f"{name}_3class.csv"

    if not in_path.exists():
        print(f"[WARN] Missing split file: {in_path}, skipping")
        return

    print(f"\n=== Processing {name} split ===")
    df = pd.read_csv(in_path)

    # Compute heuristic score
    scores = df.apply(compute_risk_score, axis=1)
    # Base risk class from score
    labels = scores.apply(assign_risk_class)
    # Upgrade to HIGH if actual event occurred
    labels = upgrade_high_if_event(df, labels)

    df["risk_class_3"] = labels.astype(int)

    # Save
    df.to_csv(out_path, index=False)

    # Print distribution
    counts = df["risk_class_3"].value_counts().sort_index()
    ratios = df["risk_class_3"].value_counts(normalize=True).sort_index()

    print(f"Saved: {out_path}")
    print("Class distribution (0=LOW, 1=MEDIUM, 2=HIGH):")
    for cls in [0, 1, 2]:
        count = int(counts.get(cls, 0))
        ratio = float(ratios.get(cls, 0.0))
        print(f"  Class {cls}: {count:5d} ({ratio*100:5.1f}%)")


def main() -> None:
    DATA_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        process_split(split)


if __name__ == "__main__":
    main()
