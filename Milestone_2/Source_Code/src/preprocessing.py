"""
Preprocessing module for Milestone 2.

Handles data loading, feature engineering, and sklearn transformers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LassoCV


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARDIO_DATA_DIR = PROJECT_ROOT / "data" / "split"

# Default feature lists (aligned with CardioDetect preprocessing)
NUMERIC_FEATURES = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "total_cholesterol",
    "hdl_cholesterol",
    "fasting_glucose",
    "bmi",
    "heart_rate",
    "pulse_pressure",
    "mean_arterial_pressure",
]

CATEGORICAL_FEATURES = [
    "sex",
    "smoking",
    "diabetes",
    "hypertension",
    "bp_meds",
]

BINARY_FLAGS = [
    "hypertension_flag",
    "high_cholesterol_flag",
    "high_glucose_flag",
    "obesity_flag",
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_splits(
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits from CSV files.

    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv.
                  Defaults to CardioDetect's data/split folder.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if data_dir is None:
        data_dir = CARDIO_DATA_DIR

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    print(f"Loaded: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")
    return train_df, val_df, test_df


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "risk_target",
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features X and target y.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        drop_cols: Additional columns to drop from features.

    Returns:
        Tuple of (X, y).
    """
    if drop_cols is None:
        drop_cols = ["data_source"]

    all_drop = [target_col] + [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=[c for c in all_drop if c in df.columns])
    y = df[target_col] if target_col in df.columns else pd.Series()

    return X, y


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical features in DataFrame.

    Args:
        X: Feature DataFrame.

    Returns:
        Tuple of (numeric_cols, categorical_cols).
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    impute_strategy: str = "median",
) -> ColumnTransformer:
    """Build a ColumnTransformer for preprocessing.

    Args:
        numeric_cols: List of numeric feature names.
        categorical_cols: List of categorical feature names.
        impute_strategy: Imputation strategy for numeric features.

    Returns:
        Configured ColumnTransformer.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="passthrough",
    )

    return preprocessor


def build_preprocessor_from_data(
    X: pd.DataFrame,
    impute_strategy: str = "median",
) -> ColumnTransformer:
    """Build preprocessor by auto-detecting feature types.

    Args:
        X: Feature DataFrame.
        impute_strategy: Imputation strategy for numeric features.

    Returns:
        Configured ColumnTransformer.
    """
    numeric_cols, categorical_cols = identify_feature_types(X)
    return build_preprocessor(numeric_cols, categorical_cols, impute_strategy)


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features_l1(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    threshold: str = "median",
) -> Tuple[np.ndarray, List[int]]:
    """Select features using L1 regularization (Lasso).

    Args:
        X: Feature DataFrame.
        y: Target Series.
        preprocessor: Fitted ColumnTransformer.
        threshold: Threshold for SelectFromModel.

    Returns:
        Tuple of (selected feature mask, selected indices).
    """
    X_transformed = preprocessor.fit_transform(X)

    lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
    lasso.fit(X_transformed, y)

    selector = SelectFromModel(lasso, prefit=True, threshold=threshold)
    mask = selector.get_support()
    indices = np.where(mask)[0].tolist()

    print(f"L1 feature selection: {sum(mask)} / {len(mask)} features selected")
    return mask, indices


def select_features_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    estimator,
    n_features_to_select: int = 15,
) -> Tuple[np.ndarray, List[int]]:
    """Select features using Recursive Feature Elimination.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        preprocessor: Fitted ColumnTransformer.
        estimator: Estimator with feature_importances_ or coef_.
        n_features_to_select: Number of features to select.

    Returns:
        Tuple of (selected feature mask, selected indices).
    """
    X_transformed = preprocessor.fit_transform(X)

    rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X_transformed, y)

    mask = rfe.support_
    indices = np.where(mask)[0].tolist()

    print(f"RFE feature selection: {sum(mask)} / {len(mask)} features selected")
    return mask, indices


# ============================================================================
# UTILITIES
# ============================================================================

def get_feature_names_after_transform(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> List[str]:
    """Get feature names after ColumnTransformer transformation.

    Args:
        preprocessor: Fitted ColumnTransformer.
        X: Original feature DataFrame.

    Returns:
        List of feature names after transformation.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn versions
        numeric_cols, categorical_cols = identify_feature_types(X)
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = []
        if hasattr(cat_encoder, "get_feature_names_out"):
            cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
        elif hasattr(cat_encoder, "get_feature_names"):
            cat_names = list(cat_encoder.get_feature_names(categorical_cols))
        return numeric_cols + cat_names
