"""
Regression models for guideline-based continuous risk prediction.

Provides HistGradientBoostingRegressor, RandomForestRegressor, and MLPRegressor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


# ============================================================================
# HYPERPARAMETER GRIDS
# ============================================================================

HGBR_PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_iter": [200, 400, 600],
    "l2_regularization": [0.0, 0.5, 1.0],
    "min_samples_leaf": [10, 20, 50],
}

RFR_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

MLPR_PARAM_GRID = {
    "hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [500, 1000],
    "early_stopping": [True],
}


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_hgb_regressor(
    preprocessor=None,
    **kwargs,
) -> Pipeline:
    """Build HistGradientBoostingRegressor pipeline.

    Args:
        preprocessor: Optional sklearn preprocessor/transformer.
        **kwargs: Additional kwargs for HistGradientBoostingRegressor.

    Returns:
        sklearn Pipeline.
    """
    default_params = {
        "max_depth": 3,
        "learning_rate": 0.05,
        "max_iter": 400,
        "l2_regularization": 1.0,
        "random_state": 42,
    }
    default_params.update(kwargs)

    reg = HistGradientBoostingRegressor(**default_params)

    if preprocessor is not None:
        return Pipeline([("prep", preprocessor), ("reg", reg)])
    return Pipeline([("reg", reg)])


def build_rf_regressor(
    preprocessor=None,
    **kwargs,
) -> Pipeline:
    """Build RandomForestRegressor pipeline.

    Args:
        preprocessor: Optional sklearn preprocessor/transformer.
        **kwargs: Additional kwargs for RandomForestRegressor.

    Returns:
        sklearn Pipeline.
    """
    default_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    default_params.update(kwargs)

    reg = RandomForestRegressor(**default_params)

    if preprocessor is not None:
        return Pipeline([("prep", preprocessor), ("reg", reg)])
    return Pipeline([("reg", reg)])


def build_mlp_regressor(
    preprocessor=None,
    **kwargs,
) -> Pipeline:
    """Build MLPRegressor pipeline.

    Args:
        preprocessor: Optional sklearn preprocessor/transformer.
        **kwargs: Additional kwargs for MLPRegressor.

    Returns:
        sklearn Pipeline.
    """
    default_params = {
        "hidden_layer_sizes": (100, 50),
        "alpha": 0.001,
        "learning_rate": "adaptive",
        "max_iter": 1000,
        "early_stopping": True,
        "random_state": 42,
    }
    default_params.update(kwargs)

    reg = MLPRegressor(**default_params)

    if preprocessor is not None:
        return Pipeline([("prep", preprocessor), ("reg", reg)])
    return Pipeline([("reg", reg)])


# ============================================================================
# SAMPLE WEIGHTING
# ============================================================================

def compute_sample_weights(
    y: np.ndarray,
    low_threshold: float = 0.10,
    high_threshold: float = 0.25,
    low_weight: float = 1.0,
    mod_weight: float = 1.5,
    high_weight: float = 2.0,
) -> np.ndarray:
    """Compute sample weights to emphasize moderate/high risk cases.

    Args:
        y: Array of continuous risk values (0–1).
        low_threshold: Upper bound for LOW class.
        high_threshold: Lower bound for HIGH class.
        low_weight: Weight for LOW cases.
        mod_weight: Weight for MODERATE cases.
        high_weight: Weight for HIGH cases.

    Returns:
        Array of sample weights.
    """
    y = np.asarray(y, dtype=float)
    weights = np.full_like(y, low_weight)

    mod_mask = (y >= low_threshold) & (y < high_threshold)
    high_mask = y >= high_threshold

    weights[mod_mask] = mod_weight
    weights[high_mask] = high_weight

    return weights


# ============================================================================
# MODEL TUNING
# ============================================================================

def tune_regressor(
    model: Pipeline,
    X_train,
    y_train,
    param_grid: Dict[str, List[Any]],
    sample_weight: Optional[np.ndarray] = None,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "neg_mean_absolute_error",
) -> tuple:
    """Tune regressor using RandomizedSearchCV.

    Args:
        model: sklearn Pipeline to tune.
        X_train: Training features.
        y_train: Training targets.
        param_grid: Parameter grid for tuning.
        sample_weight: Optional sample weights.
        n_iter: Number of random iterations.
        cv: Cross-validation folds.
        scoring: Scoring metric.

    Returns:
        Tuple of (best_model, best_params, cv_results).
    """
    # Prefix param names with 'reg__' for pipeline
    prefixed_grid = {f"reg__{k}": v for k, v in param_grid.items()}

    search = RandomizedSearchCV(
        model,
        prefixed_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    fit_params = {}
    if sample_weight is not None:
        fit_params["reg__sample_weight"] = sample_weight

    search.fit(X_train, y_train, **fit_params)

    print(f"Best score: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.cv_results_


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_all_regressors(preprocessor=None) -> Dict[str, Pipeline]:
    """Get all regressor models.

    Args:
        preprocessor: Optional sklearn preprocessor.

    Returns:
        Dictionary of model name to Pipeline.
    """
    return {
        "hgb_regressor": build_hgb_regressor(preprocessor),
        "rf_regressor": build_rf_regressor(preprocessor),
        "mlp_regressor": build_mlp_regressor(preprocessor),
    }


def get_regressor_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Get all parameter grids for regressor tuning.

    Returns:
        Dictionary of model name to parameter grid.
    """
    return {
        "hgb_regressor": HGBR_PARAM_GRID,
        "rf_regressor": RFR_PARAM_GRID,
        "mlp_regressor": MLPR_PARAM_GRID,
    }
