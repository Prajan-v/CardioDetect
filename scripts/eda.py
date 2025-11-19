from __future__ import annotations

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load_dataset() -> pd.DataFrame:
    f = OUT / "ultimate_heart_disease_dataset.csv"
    if f.exists():
        return pd.read_csv(f)
    # Fallback to processed combined if ultimate absent
    f2 = ROOT / "data" / "processed" / "combined_processed.csv"
    if f2.exists():
        return pd.read_csv(f2)
    # Fallback to cleveland processed
    f3 = ROOT / "data" / "processed" / "processed.csv"
    if f3.exists():
        return pd.read_csv(f3)
    raise FileNotFoundError("No dataset found. Run scripts/build.py or scripts/build_combined_dataset.py first.")


def _style():
    sns.set_theme(style="whitegrid")


def distributions(df: pd.DataFrame):
    cols = [c for c in ["age", "trestbps", "chol", "thalach", "oldpeak"] if c in df.columns]
    if not cols:
        return
    plt.figure(figsize=(12, 8))
    for i, c in enumerate(cols, start=1):
        ax = plt.subplot(2, 3, i)
        sns.histplot(pd.to_numeric(df[c], errors="coerce").dropna(), kde=False, ax=ax)
        ax.set_title(c)
    plt.tight_layout()
    plt.savefig(FIG / "feature_distributions.png", dpi=300)
    plt.close()


def correlation(df: pd.DataFrame):
    cols = [c for c in ["age", "trestbps", "chol", "thalach", "oldpeak"] if c in df.columns]
    if "target" in df.columns:
        cols.append("target")
    if len(cols) < 2:
        return
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIG / "correlation_heatmap.png", dpi=300)
    plt.close()


def demographics(df: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    if "target" in df.columns:
        sns.countplot(x=pd.to_numeric(df["target"], errors="coerce"), ax=ax1)
        ax1.set_title("Outcome distribution")
    ax2 = plt.subplot(2, 2, 2)
    if "sex" in df.columns:
        sns.countplot(x=pd.to_numeric(df["sex"], errors="coerce"), ax=ax2)
        ax2.set_title("Sex (1=male)")
    ax3 = plt.subplot(2, 2, 3)
    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        bins = [0, 30, 40, 50, 60, 70, 120]
        labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
        grp = pd.cut(age, bins=bins, labels=labels, include_lowest=True)
        sns.countplot(x=grp, order=labels, ax=ax3)
        ax3.set_title("Age groups")
    plt.tight_layout()
    plt.savefig(FIG / "patient_demographics.png", dpi=300)
    plt.close()


def risk_factors(df: pd.DataFrame):
    metrics = [c for c in ["trestbps", "chol", "thalach"] if c in df.columns]
    if not metrics:
        return
    plt.figure(figsize=(12, 8))
    for i, c in enumerate(metrics, start=1):
        ax = plt.subplot(2, 2, i)
        tmp = df[[c, "target"]].copy() if "target" in df.columns else df[[c]].copy()
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        if "target" in tmp.columns:
            tmp["target"] = pd.to_numeric(tmp["target"], errors="coerce")
            tmp = tmp.dropna(subset=[c, "target"])
            if tmp.empty or tmp["target"].nunique() == 0:
                ax.set_axis_off(); continue
            sns.boxplot(x="target", y=c, data=tmp, ax=ax)
            ax.set_title(f"{c} by outcome")
        else:
            sns.histplot(tmp[c].dropna(), ax=ax)
            ax.set_title(c)
    plt.tight_layout()
    plt.savefig(FIG / "risk_factors_by_outcome.png", dpi=300)
    plt.close()


def main():
    _style()
    df = _load_dataset()
    distributions(df)
    correlation(df)
    demographics(df)
    risk_factors(df)
    print(f"EDA figures saved to {FIG}")


if __name__ == "__main__":
    main()
