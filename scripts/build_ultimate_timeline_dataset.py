# Author: Prajan V (Infosys Springboard 6.0)
# Date: 2025-11-17

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.uci import download_uci_subsets, load_uci_combined
from src.data.brfss_loader import load_all_brfss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ultimate_builder")


def _paths() -> Tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    output = root / "output"
    reports = root / "reports"
    brfss_dir = data / "brfss"
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return data, output, reports, brfss_dir


essential_cols = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "bmi",
    "is_hypertensive",
    "is_hyperlipidemic",
    "is_overweight",
    "is_obese",
    "target",
    "data_source",
    "collection_year",
]


def _annotate_uci(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    if "bmi" not in X.columns:
        X["bmi"] = np.nan
    X["is_hypertensive"] = (pd.to_numeric(X.get("trestbps"), errors="coerce") >= 140).astype("Int64")
    X["is_hyperlipidemic"] = (pd.to_numeric(X.get("chol"), errors="coerce") >= 240).astype("Int64")
    X["is_overweight"] = (pd.to_numeric(X.get("bmi"), errors="coerce") >= 25).astype("Int64")
    X["is_obese"] = (pd.to_numeric(X.get("bmi"), errors="coerce") >= 30).astype("Int64")
    X["data_source"] = "UCI"
    X["collection_year"] = 1988
    return X


def _harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in essential_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[essential_cols]


def main():
    data, output, reports, brfss_dir = _paths()

    logger.info("Loading UCI subsets")
    uci_paths = download_uci_subsets()
    df_uci = load_uci_combined(uci_paths)
    df_uci = _annotate_uci(df_uci)
    df_uci = _harmonize_columns(df_uci)

    logger.info("Loading BRFSS years from %s", brfss_dir)
    df_brfss = load_all_brfss(brfss_dir, sample_per_year=3000)
    if df_brfss is not None and len(df_brfss) > 0:
        df_brfss = _harmonize_columns(df_brfss)
        df_all = pd.concat([df_uci, df_brfss], axis=0, ignore_index=True)
    else:
        logger.warning("BRFSS files not found; continuing with UCI only")
        df_all = df_uci.copy()

    out_path = output / "ultimate_heart_disease_dataset.csv"
    df_all.to_csv(out_path, index=False)

    # Summary tables
    summary = (
        df_all.groupby(["data_source", "collection_year"]).size().rename("count").reset_index()
    )
    summary.to_csv(output / "ultimate_summary.csv", index=False)

    logger.info("Saved ultimate dataset: %s (rows=%d)", out_path, len(df_all))


if __name__ == "__main__":
    main()
