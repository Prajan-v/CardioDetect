"""
Ensemble models for Milestone 2.

Provides VotingClassifier and StackingClassifier combinations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


# ============================================================================
# VOTING ENSEMBLE
# ============================================================================

def build_voting_ensemble(
    estimators: List[Tuple[str, Pipeline]],
    voting: str = "soft",
    weights: Optional[List[float]] = None,
) -> VotingClassifier:
    """Build a voting ensemble from multiple classifiers.

    Args:
        estimators: List of (name, pipeline) tuples.
        voting: 'hard' or 'soft' voting.
        weights: Optional weights for each estimator.

    Returns:
        VotingClassifier.
    """
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1,
    )

    print(f"Built VotingClassifier with {len(estimators)} estimators, voting={voting}")
    return ensemble


def build_voting_from_trained(
    trained_models: Dict[str, Pipeline],
    model_names: Optional[List[str]] = None,
    voting: str = "soft",
    weights: Optional[List[float]] = None,
) -> VotingClassifier:
    """Build voting ensemble from already-trained models.

    Args:
        trained_models: Dictionary of model name to trained Pipeline.
        model_names: Subset of model names to include. If None, use all.
        voting: 'hard' or 'soft' voting.
        weights: Optional weights for each estimator.

    Returns:
        VotingClassifier (requires refitting to aggregate predictions).
    """
    if model_names is None:
        model_names = list(trained_models.keys())

    estimators = [(name, trained_models[name]) for name in model_names]

    return build_voting_ensemble(estimators, voting, weights)


# ============================================================================
# STACKING ENSEMBLE
# ============================================================================

def build_stacking_ensemble(
    estimators: List[Tuple[str, Pipeline]],
    final_estimator: Optional[Any] = None,
    cv: int = 5,
    passthrough: bool = False,
) -> StackingClassifier:
    """Build a stacking ensemble from multiple classifiers.

    Args:
        estimators: List of (name, pipeline) tuples for base learners.
        final_estimator: Meta-learner. Defaults to LogisticRegression.
        cv: Cross-validation folds for base learner predictions.
        passthrough: Whether to pass original features to meta-learner.

    Returns:
        StackingClassifier.
    """
    if final_estimator is None:
        final_estimator = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        passthrough=passthrough,
        n_jobs=-1,
    )

    print(f"Built StackingClassifier with {len(estimators)} base learners")
    return ensemble


def build_stacking_with_tree_meta(
    estimators: List[Tuple[str, Pipeline]],
    max_depth: int = 3,
    cv: int = 5,
) -> StackingClassifier:
    """Build stacking ensemble with shallow decision tree as meta-learner.

    Args:
        estimators: List of (name, pipeline) tuples for base learners.
        max_depth: Max depth of decision tree meta-learner.
        cv: Cross-validation folds.

    Returns:
        StackingClassifier.
    """
    meta = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
    )

    return build_stacking_ensemble(estimators, meta, cv)


def build_stacking_with_lr_meta(
    estimators: List[Tuple[str, Pipeline]],
    C: float = 1.0,
    cv: int = 5,
) -> StackingClassifier:
    """Build stacking ensemble with logistic regression as meta-learner.

    Args:
        estimators: List of (name, pipeline) tuples for base learners.
        C: Regularization strength for logistic regression.
        cv: Cross-validation folds.

    Returns:
        StackingClassifier.
    """
    meta = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=42,
    )

    return build_stacking_ensemble(estimators, meta, cv)


# ============================================================================
# ENSEMBLE UTILITIES
# ============================================================================

def select_top_models(
    model_scores: Dict[str, float],
    n_top: int = 3,
) -> List[str]:
    """Select top N models by score.

    Args:
        model_scores: Dictionary of model name to score (higher is better).
        n_top: Number of top models to select.

    Returns:
        List of top model names.
    """
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_models[:n_top]]


def create_ensemble_from_results(
    trained_models: Dict[str, Pipeline],
    model_scores: Dict[str, float],
    ensemble_type: str = "voting",
    n_top: int = 3,
    **kwargs,
) -> Any:
    """Create ensemble from top performing models.

    Args:
        trained_models: Dictionary of model name to trained Pipeline.
        model_scores: Dictionary of model name to validation score.
        ensemble_type: 'voting' or 'stacking'.
        n_top: Number of top models to include.
        **kwargs: Additional kwargs for ensemble builder.

    Returns:
        Ensemble classifier.
    """
    top_names = select_top_models(model_scores, n_top)
    print(f"Selected top {n_top} models: {top_names}")

    estimators = [(name, trained_models[name]) for name in top_names]

    if ensemble_type == "voting":
        return build_voting_ensemble(estimators, **kwargs)
    elif ensemble_type == "stacking":
        return build_stacking_ensemble(estimators, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def fit_voting_ensemble(
    ensemble: VotingClassifier,
    X_train,
    y_train,
) -> VotingClassifier:
    """Fit voting ensemble on training data.

    Args:
        ensemble: VotingClassifier to fit.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted VotingClassifier.
    """
    print("Fitting VotingClassifier...")
    ensemble.fit(X_train, y_train)
    print("VotingClassifier fitted.")
    return ensemble


def fit_stacking_ensemble(
    ensemble: StackingClassifier,
    X_train,
    y_train,
) -> StackingClassifier:
    """Fit stacking ensemble on training data.

    Args:
        ensemble: StackingClassifier to fit.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted StackingClassifier.
    """
    print("Fitting StackingClassifier...")
    ensemble.fit(X_train, y_train)
    print("StackingClassifier fitted.")
    return ensemble
