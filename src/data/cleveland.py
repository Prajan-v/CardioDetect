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
    "num"
]


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df.replace("?", np.nan)
    for c in ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal", "sex", "cp", "fbs", "restecg", "exang", "slope", "num"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET_COLUMN] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df
