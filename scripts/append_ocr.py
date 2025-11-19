import argparse
import os
from pathlib import Path
import pandas as pd
import joblib
from src.config import PROCESSED_DIR, MODELS_DIR
from src.ocr.pipeline import extract_structured
from src.data.features import add_features


def to_input_row(d):
    row = {
        "age": d.get("age"),
        "trestbps": d.get("bp_sys"),
        "chol": d.get("chol"),
        "thalach": d.get("heart_rate"),
        "oldpeak": None,
        "ca": None,
        "sex": None,
        "cp": None,
        "fbs": None,
        "restecg": None,
        "exang": None,
        "slope": None,
        "thal": None,
    }
    return pd.DataFrame([row])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("document", type=str)
    p.add_argument("--output", type=str, default=str(PROCESSED_DIR / "processed_with_ocr.csv"))
    p.add_argument("--base", type=str, default=str(PROCESSED_DIR / "processed.csv"))
    p.add_argument("--preprocessor", type=str, default=str(MODELS_DIR / "preprocessor.joblib"))
    args = p.parse_args()
    d = extract_structured(Path(args.document))
    X = to_input_row(d)
    X = add_features(X)
    pre = joblib.load(args.preprocessor)
    Xp = pre.transform(X)
    cols = pre.get_feature_names_out()
    row_df = pd.DataFrame(Xp.toarray() if hasattr(Xp, "toarray") else Xp, columns=cols)
    df = pd.read_csv(args.base)
    df2 = pd.concat([df, row_df], axis=0, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
