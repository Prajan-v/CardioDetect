"""
Baseline classification models for Milestone 2.

Provides LogisticRegression, RandomForest, SVM, and MLP baselines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


# ============================================================================
# HYPERPARAMETER GRIDS
# ============================================================================

LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "max_iter": [500, 1000],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", None],
}

SVM_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"],
    "probability": [True],
}

MLP_PARAM_GRID = {
    "hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [500, 1000],
    "early_stopping": [True],
}


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_logistic_regression(
    preprocessor=None,
    **kwargs,
) -> Pipeline:
    """Build LogisticRegression pipeline.

    Args:
        preprocessor: Optional sklearn preprocessor/transformer.
        **kwargs: Additional kwargs for LogisticRegression.

    Returns:
        sklearn Pipeline.
    """
    default_params = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    }
    default_params.update(kwargs)

    clf = LogisticRegression(**default_params)

    if preprocessor is not None:
        return Pipeline([("prep", preprocessor), ("clf", clf)])
    return Pipeline([("clf", clf)])


def build_random_forest(
    preprocessor=None,
    **kwargs,
) -> Pipeline:
    """Build RandomForestClassifier pipeline.

    Args:
        preprocessor: Optional sklearn preprocessor/transformer.
        **kwargs: Additional kwargs for RandomForestClassifier.

    Returns:
        sklearn Pipeline.
    """
    default_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    default_params.update(kwargs)

    clf = RandomForestClassifier(**default_params)

    if preprocessor is not None:
        return Pipeline([("prep", preprocessor), ("clf", clf)])
    return Pipeline([("clf", clf)])


def build_svm(
    preprocessor=None,
    **kwargs,
) -> Pipeline:
    """Build SVC pipeline with probability estimates.

    Args:
        preprocessor: Optional sklearn preprocessor/transformer.
        **kwargs: Additional kwargs for SVC.

    Returns:
        sklearn Pipeline.
    """
    default_params = {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": 42,
    }
    default_params.update(kwargs)

    clf = SVC(**default_params)

    if preprocessor is not None:
        return Pipeline([("prep", preprocessor), ("clf", clf)])
    return Pipeline([("clf", clf)])


def build_mlp_classifier(
    preprocessor=None,
    **kwargs,
) -> Pipeline:
    """Build MLPClassifier pipeline.

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
# MODEL TUNING
# ============================================================================

def tune_model(
    model: Pipeline,
    X_train,
    y_train,
    param_grid: Dict[str, List[Any]],
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "f1_macro",
) -> tuple:
    """Tune model using RandomizedSearchCV.

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

def get_all_baselines(preprocessor=None) -> Dict[str, Pipeline]:
    """Get all baseline models.

    Args:
        preprocessor: Optional sklearn preprocessor.

    Returns:
        Dictionary of model name to Pipeline.
    """
    return {
        "logistic_regression": build_logistic_regression(preprocessor),
        "random_forest": build_random_forest(preprocessor),
        "svm": build_svm(preprocessor),
        "mlp": build_mlp_classifier(preprocessor),
    }


def get_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Get all parameter grids for tuning.

    Returns:
        Dictionary of model name to parameter grid.
    """
    return {
        "logistic_regression": LR_PARAM_GRID,
        "random_forest": RF_PARAM_GRID,
        "svm": SVM_PARAM_GRID,
        "mlp": MLP_PARAM_GRID,
    }
