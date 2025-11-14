from pathlib import Path
import pandas as pd
from src.config import ensure_dirs, RAW_DIR, PROCESSED_DIR, MODELS_DIR, DOCS_DIR
from src.data.download import download_cleveland
from src.data.cleveland import load_dataframe
from src.data.schema import TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.data.preprocess import build_preprocessor
from src.data.split import stratified_splits
from src.data.features import add_features
import joblib


def main():
    ensure_dirs()
    raw_path = download_cleveland()
    df = load_dataframe(raw_path)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    X = add_features(X)
    pre = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X_proc = pre.fit_transform(X)
    cols = pre.get_feature_names_out()
    proc_df = pd.DataFrame(X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc, columns=cols)
    proc_df[TARGET_COLUMN] = y.values
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    proc_df.to_csv(PROCESSED_DIR / "processed.csv", index=False)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_splits(proc_df.drop(columns=[TARGET_COLUMN]), proc_df[TARGET_COLUMN])
    pd.concat([X_train, y_train], axis=1).to_csv(PROCESSED_DIR / "train.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(PROCESSED_DIR / "val.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROCESSED_DIR / "test.csv", index=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    Path(DOCS_DIR / "feature_names.txt").write_text("\n".join(cols))


if __name__ == "__main__":
    main()
