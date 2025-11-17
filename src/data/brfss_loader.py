# Author: Prajan V (Infosys Springboard 6.0)
# Date: 2025-11-17

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _age_code_to_midpoint(age_code: pd.Series) -> pd.Series:
    """
    Map BRFSS _AGEG5YR groups to approximate midpoints in years.

    1: 18-24 -> 21
    2: 25-29 -> 27
    3: 30-34 -> 32
    4: 35-39 -> 37
    5: 40-44 -> 42
    6: 45-49 -> 47
    7: 50-54 -> 52
    8: 55-59 -> 57
    9: 60-64 -> 62
    10: 65-69 -> 67
    11: 70-74 -> 72
    12: 75-79 -> 77
    13: 80+   -> 82
    """
    mapping = {
        1: 21,
        2: 27,
        3: 32,
        4: 37,
        5: 42,
        6: 47,
        7: 52,
        8: 57,
        9: 62,
        10: 67,
        11: 72,
        12: 77,
        13: 82,
    }
    s = pd.to_numeric(age_code, errors="coerce").map(mapping)
    return s.astype("float64")


def _as_series(val, index) -> pd.Series:
    if isinstance(val, pd.Series):
        return pd.to_numeric(val, errors="coerce")
    # fallback to NaNs if column missing or scalar provided
    return pd.Series(np.nan, index=index, dtype="float64")


def _safe_read_csv(path: Path, usecols: List[str]) -> pd.DataFrame:
    """Load large BRFSS file with only the needed columns."""
    try:
        df = pd.read_csv(
            path,
            usecols=[c for c in usecols if c is not None],
            dtype="Int64",
            low_memory=False,
        )
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", path)
        raise
    except ValueError:
        head = pd.read_csv(path, nrows=1, low_memory=False)
        present = [c for c in usecols if c in head.columns]
        df = pd.read_csv(path, usecols=present, dtype="Int64", low_memory=False)
        logger.warning("Columns missing in %s; using subset: %s", path.name, present)
        return df


def _safe_read_xpt(path: Path, usecols: List[str]) -> pd.DataFrame:
    try:
        df = pd.read_sas(path, format="xport")
        # Normalize column names to uppercase to match BRFSS docs, then select
        df.columns = [str(c) for c in df.columns]
        # Some XPT readers yield bytes columns; ensure str
        upper = {c: c.upper() for c in df.columns}
        df = df.rename(columns=upper)
        present = [c for c in usecols if c in df.columns]
        return df[present].copy()
    except FileNotFoundError:
        logger.error("XPT file not found: %s", path)
        raise


def _resolve_brfss_file(data_dir: Path, year: int) -> Optional[Path]:
    csv = data_dir / f"{year}.csv"
    if csv.exists():
        return csv
    raw = data_dir / "raw"
    xpt = raw / f"LLCP{year}.XPT"
    if xpt.exists():
        return xpt
    # common alternative lowercase
    xpt2 = raw / f"llcp{year}.xpt"
    if xpt2.exists():
        return xpt2
    return None


def _map_brfss_to_schema(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Vectorized mapping from BRFSS columns to UCI-like schema with lineage."""
    out = pd.DataFrame(index=df.index)

    age_code = _as_series(df.get("_AGEG5YR"), df.index)
    out["age"] = _age_code_to_midpoint(age_code)

    sex_s = _as_series(df.get("SEX"), df.index)
    out["sex"] = (sex_s == 1).astype("Int64")

    hyp = _as_series(df.get("_RFHYPE5"), df.index)
    out["trestbps"] = hyp.map({1: 145, 2: 120}).astype("float64")

    chol_flag = _as_series(df.get("_RFCHOL"), df.index)
    out["chol"] = chol_flag.map({1: 240, 2: 190}).astype("float64")

    diab = _as_series(df.get("DIABETE3"), df.index)
    out["fbs"] = (diab == 1).astype("Int64")

    infarct = _as_series(df.get("CVDINFR4"), df.index)
    chd = _as_series(df.get("CVDCRHD4"), df.index)
    out["target"] = ((infarct == 1) | (chd == 1)).astype("Int64")

    bmi10 = _as_series(df.get("_BMI5"), df.index)
    out["bmi"] = (bmi10 / 10.0).astype("float64")

    for col in ["cp", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]:
        out[col] = pd.Series(np.nan, index=df.index, dtype="float64")

    out["is_hypertensive"] = (out["trestbps"] >= 140).astype("Int64")
    out["is_hyperlipidemic"] = (out["chol"] >= 240).astype("Int64")
    out["is_overweight"] = (out["bmi"] >= 25).astype("Int64")
    out["is_obese"] = (out["bmi"] >= 30).astype("Int64")

    out["data_source"] = "BRFSS"
    out["collection_year"] = int(year)

    return out


def load_brfss_year(filepath: Path | str, year: int, sample_n: Optional[int] = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Load a single BRFSS year and map to harmonized schema.
    """
    usecols = [
        "_AGEG5YR",
        "SEX",
        "_RFHYPE5",
        "_RFCHOL",
        "DIABETE3",
        "CVDINFR4",
        "CVDCRHD4",
        "_BMI5",
    ]
    path = Path(filepath)
    logger.info("Loading BRFSS %s from %s", year, path)
    if path.suffix.lower() == ".xpt":
        df_raw = _safe_read_xpt(path, usecols=usecols)
    else:
        df_raw = _safe_read_csv(path, usecols=usecols)

    df_h = _map_brfss_to_schema(df_raw, year)

    if sample_n is not None and len(df_h) > sample_n:
        df_h = df_h.sample(n=sample_n, random_state=random_state)

    logger.info("BRFSS %s loaded: %d rows", year, len(df_h))
    return df_h.reset_index(drop=True)


def load_all_brfss(data_dir: Path | str, sample_per_year: int = 3000, years: Iterable[int] = (2011, 2012, 2013, 2014, 2015), random_state: int = 42) -> pd.DataFrame:
    """
    Load and combine multiple BRFSS years.
    """
    base = Path(data_dir)
    frames: List[pd.DataFrame] = []
    for y in years:
        f = _resolve_brfss_file(base, y)
        if f is None:
            logger.warning("Skipping BRFSS %s, file not found in %s", y, base)
            continue
        frames.append(load_brfss_year(f, year=y, sample_n=sample_per_year, random_state=random_state))

    if not frames:
        logger.error("No BRFSS files found in %s", base)
        return pd.DataFrame()

    out = pd.concat(frames, axis=0, ignore_index=True)
    logger.info("Combined BRFSS rows: %d", len(out))
    return out
