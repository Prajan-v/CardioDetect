from pathlib import Path
import pandas as pd
import numpy as np
from .schema import TARGET_COLUMN

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
]


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    out["sex"] = pd.to_numeric(df.get("male"), errors="coerce")
    out["cp"] = np.nan
    out["trestbps"] = pd.to_numeric(df.get("sysBP"), errors="coerce")
    out["chol"] = pd.to_numeric(df.get("totChol"), errors="coerce")
    glu = pd.to_numeric(df.get("glucose"), errors="coerce")
    out["fbs"] = (glu > 120).astype(float)
    out["restecg"] = np.nan
    out["thalach"] = pd.to_numeric(df.get("heartRate"), errors="coerce")
    out["exang"] = np.nan
    out["oldpeak"] = np.nan
    out["slope"] = np.nan
    out["ca"] = np.nan
    out["thal"] = np.nan
    out[TARGET_COLUMN] = pd.to_numeric(df.get("TenYearCHD"), errors="coerce").fillna(0).astype(int)
    return out
