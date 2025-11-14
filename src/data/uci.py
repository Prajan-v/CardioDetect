from pathlib import Path
import requests
import pandas as pd
import numpy as np
from src.config import RAW_DIR
from .schema import TARGET_COLUMN

UCI_BASE_HTTPS = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
UCI_BASE_HTTP = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
UCI_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
COLUMNS = [
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
    "num",
]


def _download_file(name: str, overwrite=False, timeout=30) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fname = UCI_FILES[name]
    out = RAW_DIR / fname
    if out.exists() and not overwrite:
        return out
    for base in [UCI_BASE_HTTPS, UCI_BASE_HTTP]:
        url = base + fname
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            out.write_bytes(r.content)
            return out
        except Exception:
            continue
    raise RuntimeError(f"Failed to download {name}")


def download_uci_subsets(overwrite=False):
    paths = {}
    for k in UCI_FILES:
        try:
            paths[k] = _download_file(k, overwrite=overwrite)
        except Exception:
            pass
    return paths


def _parse_uci(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df.replace("?", np.nan)
    for c in ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal", "sex", "cp", "fbs", "restecg", "exang", "slope", "num"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET_COLUMN] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])    
    return df


def load_uci_combined(paths: dict | None = None) -> pd.DataFrame:
    if paths is None:
        paths = {k: RAW_DIR / v for k, v in UCI_FILES.items()}
    dfs = []
    for k, p in paths.items():
        if Path(p).exists():
            dfs.append(_parse_uci(Path(p)))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=0, ignore_index=True)
