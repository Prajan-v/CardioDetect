import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for c in ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]:
        X[c] = pd.to_numeric(X.get(c), errors="coerce")
    X["age_decade"] = (X["age"] // 10).astype(float)
    X["is_hypertensive"] = (X["trestbps"] >= 140).astype(int)
    X["is_hyperlipidemic"] = (X["chol"] >= 240).astype(int)
    X["st_depr_high"] = (X["oldpeak"] >= 2.0).astype(int)
    return X
