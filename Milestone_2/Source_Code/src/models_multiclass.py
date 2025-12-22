"""
Multi-class classification models for 3-class risk prediction (LOW/MODERATE/HIGH).

Provides HGB, RF, and MLP classifiers with calibration wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


# ============================================================================
# HYPERPARAMETER GRIDS
# ============================================================================

HGB_PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_iter": [200, 400],
    "l2_regularization": [0.0, 0.5, 1.0],
    "min_samples_leaf": [10, 20],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", "balanced_subsample", None],
}

MLP_PARAM_GRID = {
    "hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [500, 1000],
    "early_stopping": [True],
}

# Default class weights for 3-class (emphasize MODERATE and HIGH)
DEFAULT_CLASS_WEIGHT = {0: 1.0, 1: 1.5, 2: 2.0}


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_hgb_multiclass(
    preprocessor=None,
    class_weight: Optional[Dict[int, float]] = None,
    **kwargs,
) -> Pipeline:
    """Build HistGradientBoostingClassifier for multiclass.

    Args:
        preprocessor: Optional sklearn preprocessor/transformer.
        class_weight: Class weight dictionary.
        **kwargs: Additional kwargs for HistGradientBoostingClassifier.

    Returns:
        sklearn Pipeline.
    """
    if class_weight is None:
        class_weight = DEFAULT_CLASS_WEIGHT

    default_params = {
        "max_depth": 3,
        "learning_rate": 0.05,
        "max_iter": 400,
        "l2_regularization": 1.0,
        "class_weight": class_weight,
        "random_state": 42,
    }
    default_params.update(kwargs)

    clf = HistGradientBoostingClassifier(**default_params)

    if preprocessor is not None:
        return Pipeline([("prep", preprocessor), ("clf", clf)])
    return Pipeline([("clf", clf)])


def build_rf_multiclass(
    preprocessor=None,
    class_weight: str = "balanced",
    **kwargs,
) -> Pipeline:
    """Build RandomForestClassifier for multiclass.

    Args:
        preprocessor: Optional sklearn preprocessor/transformer.
        class_weight: Class weight strategy.
        **kwargs: Additional kwargs for RandomForestClassifier.

    Returns:
        sklearn Pipeline.
    """
    default_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": class_weight,
        "random_state": 42,
        "n_jobs": -1,
    }
    default_params.update(kwargs)

    clf = RandomForestClassifier(**default_params)

    if preprocessor is not None:
        return Pipeline([("prep", preprocessor), ("clf", clf)])
    return Pipeline([("clf", clf)])


def build_mlp_multiclass(
    preprocessor=None,
    **kwargs,
) -> Pipeline:
    """Build MLPClassifier for multiclass.

    Args:
        preprocessor: Optional sklearn preprocessor/transformer.
        **kwargs: Additional kwargs for MLPClassifier.

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

    clf = MLPClassifier(**default_params)

    if preprocessor is not None:
        return Pipeline([("prep", preprocessor), ("clf", clf)])
    return Pipeline([("clf", clf)])


# ============================================================================
# CALIBRATION
# ============================================================================

def calibrate_classifier(
    classifier: Pipeline,
    X_train,
    y_train,
    method: str = "isotonic",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Wrap classifier with probability calibration.

    Args:
        classifier: Trained sklearn classifier/pipeline.
        X_train: Training features.
        y_train: Training labels.
        method: Calibration method ('isotonic' or 'sigmoid').
        cv: Cross-validation folds for calibration.

    Returns:
        CalibratedClassifierCV wrapper.
    """
    calibrated = CalibratedClassifierCV(
        classifier,
        method=method,
        cv=cv,
    )
    calibrated.fit(X_train, y_train)

    print(f"Calibrated classifier with method='{method}', cv={cv}")
    return calibrated


def build_calibrated_multiclass(
    base_model_fn,
    preprocessor=None,
    X_train=None,
    y_train=None,
    method: str = "isotonic",
    cv: int = 3,
    **model_kwargs,
) -> CalibratedClassifierCV:
    """Build and calibrate a multiclass classifier.

    Args:
        base_model_fn: Function to build base model (e.g., build_hgb_multiclass).
        preprocessor: Optional sklearn preprocessor.
        X_train: Training features.
        y_train: Training labels.
        method: Calibration method.
        cv: Cross-validation folds.
        **model_kwargs: Additional kwargs for base model.

    Returns:
        Calibrated classifier.
    """
    base_model = base_model_fn(preprocessor=preprocessor, **model_kwargs)

    if X_train is not None and y_train is not None:
        base_model.fit(X_train, y_train)
        return calibrate_classifier(base_model, X_train, y_train, method, cv)

    return base_model


# ============================================================================
# MODEL TUNING
# ============================================================================

def tune_multiclass(
    model: Pipeline,
    X_train,
    y_train,
    param_grid: Dict[str, List[Any]],
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "f1_macro",
) -> tuple:
    """Tune multiclass classifier using RandomizedSearchCV.

    Args:
        model: sklearn Pipeline to tune.
        X_train: Training features.
        y_train: Training labels.
        param_grid: Parameter grid for tuning.
        n_iter: Number of random iterations.
        cv: Cross-validation folds.
        scoring: Scoring metric.

    Returns:
        Tuple of (best_model, best_params, cv_results).
    """
    # Prefix param names with 'clf__' for pipeline
    prefixed_grid = {f"clf__{k}": v for k, v in param_grid.items()}

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

    search.fit(X_train, y_train)

    print(f"Best score: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.cv_results_


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_all_multiclass_models(preprocessor=None) -> Dict[str, Pipeline]:
    """Get all multiclass classifier models.

    Args:
        preprocessor: Optional sklearn preprocessor.

    Returns:
        Dictionary of model name to Pipeline.
    """
    return {
        "hgb_multiclass": build_hgb_multiclass(preprocessor),
        "rf_multiclass": build_rf_multiclass(preprocessor),
        "mlp_multiclass": build_mlp_multiclass(preprocessor),
    }


def get_multiclass_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Get all parameter grids for multiclass tuning.

    Returns:
        Dictionary of model name to parameter grid.
    """
    return {
        "hgb_multiclass": HGB_PARAM_GRID,
        "rf_multiclass": RF_PARAM_GRID,
        "mlp_multiclass": MLP_PARAM_GRID,
    }
