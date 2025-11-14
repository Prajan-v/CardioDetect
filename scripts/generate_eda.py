from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import ensure_dirs, RAW_DIR, DOCS_DIR
from src.data.cleveland import load_dataframe


def main():
    ensure_dirs()
    raw = RAW_DIR / "processed.cleveland.data"
    if not raw.exists():
        from src.data.download import download_cleveland
        raw = download_cleveland()
    df = load_dataframe(raw)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for i, c in enumerate(["age", "trestbps", "chol", "thalach", "oldpeak" ], start=1):
        plt.subplot(2, 3, i)
        sns.histplot(df[c].dropna(), kde=False)
        plt.title(c)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "eda_distributions.png", dpi=150)
    plt.close()
    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "eda_correlation.png", dpi=150)
    plt.close()
    plt.figure(figsize=(10, 6))
    cats = ["cp", "fbs", "restecg", "exang", "slope", "thal"]
    n = len(cats)
    for i, c in enumerate(cats, start=1):
        plt.subplot(2, 3, i)
        sns.countplot(x=df[c])
        plt.title(c)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "eda_counts.png", dpi=150)
    plt.close()
    desc = df.describe(include='all')
    desc.to_csv(DOCS_DIR / "eda_summary.csv")
    dist = df['target'].value_counts(normalize=False).rename('count').to_frame()
    dist['ratio'] = dist['count'] / dist['count'].sum()
    dist.to_csv(DOCS_DIR / "target_distribution.csv")


if __name__ == "__main__":
    main()
