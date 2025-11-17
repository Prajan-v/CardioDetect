# Author: Prajan V (Infosys Springboard 6.0)
# Date: 2025-11-17

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced_eda")


def _paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    output = root / "output"
    reports = root / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    return root, output, figures


def _load_or_build_ultimate(output: Path) -> pd.DataFrame:
    f = output / "ultimate_heart_disease_dataset.csv"
    if not f.exists():
        from scripts.build_ultimate_timeline_dataset import main as build_main
        logger.info("Ultimate dataset not found. Building now.")
        build_main()
    return pd.read_csv(f)


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _style():
    sns.set_theme(style="whitegrid")
    sns.set_palette(["#2c7fb8", "#f03b20", "#7b3294", "#009392", "#de8f05"])  # chose palette for clarity


def plot_feature_distributions(df: pd.DataFrame, outdir: Path):
    cols = ["age", "trestbps", "chol", "bmi", "thalach", "oldpeak"]
    plt.figure(figsize=(12, 8))
    for i, c in enumerate(cols, start=1):
        ax = plt.subplot(2, 3, i)
        sns.histplot(_safe_num(df.get(c)).dropna(), kde=False, ax=ax)
        ax.set_title(c)
        ax.set_xlabel(c)
    plt.tight_layout()
    p = outdir / "feature_distributions.png"
    plt.savefig(p, dpi=300)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, outdir: Path):
    num_cols = [c for c in ["age", "trestbps", "chol", "bmi", "thalach", "oldpeak"] if c in df.columns]
    corr = df[num_cols + ["target"]].apply(pd.to_numeric, errors="coerce").corr()
    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    ax.set_title("Correlation analysis (note: proxies for BP/chol in BRFSS)")
    plt.tight_layout()
    p = outdir / "correlation_heatmap.png"
    plt.savefig(p, dpi=300)
    plt.close()


def plot_patient_demographics(df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    sns.countplot(x=df["target"].astype("Int64"), ax=ax1)
    ax1.set_title("Outcome distribution")
    ax1.set_xlabel("target")

    ax2 = plt.subplot(2, 2, 2)
    sns.countplot(x=df["sex"].astype("Int64"), ax=ax2)
    ax2.set_title("Sex distribution (1=male)")
    ax2.set_xlabel("sex")

    ax3 = plt.subplot(2, 2, 3)
    age = _safe_num(df["age"]) if "age" in df.columns else pd.Series(dtype=float)
    bins = [0, 30, 40, 50, 60, 70, 120]
    labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
    grp = pd.cut(age, bins=bins, labels=labels, include_lowest=True)
    sns.countplot(x=grp, order=labels, ax=ax3)
    ax3.set_title("Age groups")
    ax3.set_xlabel("age group")

    ax4 = plt.subplot(2, 2, 4)
    if "data_source" in df.columns:
        sns.countplot(x=df["data_source"], ax=ax4)
        ax4.set_title("Data sources")
        ax4.set_xlabel("")
    plt.tight_layout()
    p = outdir / "patient_demographics.png"
    plt.savefig(p, dpi=300)
    plt.close()


def plot_risk_factors_by_outcome(df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(12, 8))
    metrics = [c for c in ["trestbps", "chol", "bmi", "thalach"] if c in df.columns]
    for i, c in enumerate(metrics, start=1):
        ax = plt.subplot(2, 2, i)
        tmp = df[[c, "target"]].copy()
        tmp[c] = _safe_num(tmp[c])
        tmp["target"] = pd.to_numeric(tmp["target"], errors="coerce")
        tmp = tmp.dropna(subset=[c, "target"])  # ensure valid pairs
        if tmp.empty or tmp["target"].nunique() == 0:
            ax.set_title(f"{c} by outcome (insufficient data)")
            ax.set_axis_off()
            continue
        sns.boxplot(x="target", y=c, data=tmp, ax=ax)
        ax.set_title(f"{c} by outcome")
        ax.set_xlabel("target")
        ax.set_ylabel(c)
    plt.tight_layout()
    p = outdir / "risk_factors_by_outcome.png"
    plt.savefig(p, dpi=300)
    plt.close()


def main():
    _style()
    root, output, figures = _paths()
    df = _load_or_build_ultimate(output)

    plot_feature_distributions(df, figures)
    plot_correlation_heatmap(df, figures)
    plot_patient_demographics(df, figures)
    plot_risk_factors_by_outcome(df, figures)

    logger.info("Advanced EDA figures saved to %s", figures)


if __name__ == "__main__":
    main()
