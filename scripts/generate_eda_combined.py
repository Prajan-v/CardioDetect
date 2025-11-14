from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import ensure_dirs, RAW_DIR, DOCS_DIR
from src.data.uci import download_uci_subsets, load_uci_combined
from src.data.framingham import load_dataframe as load_framingham


def try_load_framingham() -> pd.DataFrame:
    for p in [RAW_DIR / "framingham.csv", RAW_DIR / "Framingham.csv"]:
        if p.exists():
            return load_framingham(p)
    return pd.DataFrame()


def main():
    ensure_dirs()
    download_uci_subsets()
    df_uci = load_uci_combined(None)
    df_fram = try_load_framingham()
    if not df_fram.empty:
        df = pd.concat([df_uci, df_fram], axis=0, ignore_index=True)
    else:
        df = df_uci
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for i, c in enumerate(["age", "trestbps", "chol", "thalach", "oldpeak" ], start=1):
        plt.subplot(2, 3, i)
        sns.histplot(df[c].dropna(), kde=False)
        plt.title(c)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "eda_combined_distributions.png", dpi=150)
    plt.close()
    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "eda_combined_correlation.png", dpi=150)
    plt.close()
    plt.figure(figsize=(10, 6))
    cats = ["cp", "fbs", "restecg", "exang", "slope", "thal"]
    for i, c in enumerate(cats, start=1):
        plt.subplot(2, 3, i)
        sns.countplot(x=df[c])
        plt.title(c)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "eda_combined_counts.png", dpi=150)
    plt.close()
    df.describe(include='all').to_csv(DOCS_DIR / "eda_combined_summary.csv")
    dist = df['target'].value_counts(normalize=False).rename('count').to_frame()
    dist['ratio'] = dist['count'] / dist['count'].sum()
    dist.to_csv(DOCS_DIR / "combined_target_distribution.csv")


if __name__ == "__main__":
    main()
