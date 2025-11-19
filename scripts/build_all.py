from pathlib import Path
import pandas as pd
import joblib
from src.config import ensure_dirs, RAW_DIR, PROCESSED_DIR, MODELS_DIR, DOCS_DIR
from src.data.uci import download_uci_subsets, load_uci_combined
from src.data.framingham import load_dataframe as load_framingham
from src.data.schema import TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.data.features import add_features
from src.data.preprocess import build_preprocessor
from src.data.split import stratified_splits


def try_load_framingham() -> pd.DataFrame:
    candidates = [RAW_DIR / "framingham.csv", RAW_DIR / "Framingham.csv"]
    for p in candidates:
        if p.exists():
            return load_framingham(p)
    return pd.DataFrame()


def main():
    ensure_dirs()
    uci_paths = download_uci_subsets()
    df_uci = load_uci_combined(uci_paths)
    df_fram = try_load_framingham()
    if not df_fram.empty:
        df = pd.concat([df_uci, df_fram], axis=0, ignore_index=True)
    else:
        df = df_uci.copy()
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    X = add_features(X)
    pre = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    Xp = pre.fit_transform(X)
    cols = pre.get_feature_names_out()
    proc_df = pd.DataFrame(Xp.toarray() if hasattr(Xp, "toarray") else Xp, columns=cols)
    proc_df[TARGET_COLUMN] = y.values
    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_df.to_csv(out_dir / "combined_processed.csv", index=False)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_splits(proc_df.drop(columns=[TARGET_COLUMN]), proc_df[TARGET_COLUMN])
    pd.concat([X_train, y_train], axis=1).to_csv(out_dir / "combined_train.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(out_dir / "combined_val.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(out_dir / "combined_test.csv", index=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, MODELS_DIR / "preprocessor_combined.joblib")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    Path(DOCS_DIR / "combined_feature_names.txt").write_text("\n".join(cols))


if __name__ == "__main__":
    main()
