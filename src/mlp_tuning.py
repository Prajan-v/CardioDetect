"""MLP Tuning Module for CardioDetect Risk Prediction

This module provides utilities for:
- Phase 1: Locking and evaluating the baseline MLP
- Phase 2: Hyperparameter search with Optuna (train on TRAIN, tune on VAL)
- Phase 3: Candidate selection and TEST evaluation
- Phase 4: Saving improved models (without touching baseline)

IMPORTANT: This module NEVER overwrites mlp_baseline_locked.pkl once created.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings('ignore')

# Try importing optuna for hyperparameter tuning
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with `pip install optuna`")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_SPLIT_DIR = PROJECT_ROOT / "data" / "split"

# Model paths
BASELINE_MLP_PATH = MODELS_DIR / "mlp_baseline_locked.pkl"
MLP_V2_PATH = MODELS_DIR / "mlp_v2_best.pkl"

# Report paths
BASELINE_METRICS_MD = REPORTS_DIR / "mlp_baseline_metrics.md"
TUNING_LOG_CSV = REPORTS_DIR / "mlp_tuning_log.csv"
CANDIDATES_VS_BASELINE_MD = REPORTS_DIR / "mlp_candidates_vs_baseline.md"
MLP_BEST_SUMMARY_MD = REPORTS_DIR / "mlp_best_summary.md"


# ---------------------------------------------------------------------------
# Data Loading (reusing logic from models.py)
# ---------------------------------------------------------------------------


def load_splits(
    train_path: Path = DATA_SPLIT_DIR / "train.csv",
    val_path: Path = DATA_SPLIT_DIR / "val.csv",
    test_path: Path = DATA_SPLIT_DIR / "test.csv",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train/val/test CSVs, split into X/y, drop non-feature columns.
    
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    drop_cols = ['risk_target', 'data_source']
    
    # Binarize target: 0 = no event, 1 = any event
    y_train = (train_df['risk_target'] > 0).astype(int)
    y_val = (val_df['risk_target'] > 0).astype(int)
    y_test = (test_df['risk_target'] > 0).astype(int)
    
    X_train = train_df.drop(columns=drop_cols)
    X_val = val_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-hot encode any non-numeric features."""
    combined = pd.concat(
        [X_train, X_val, X_test],
        keys=["train", "val", "test"],
        names=["split", None]
    )
    
    categorical_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if not categorical_cols:
        return X_train, X_val, X_test
    
    combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)
    
    X_train_enc = combined_encoded.xs("train")
    X_val_enc = combined_encoded.xs("val")
    X_test_enc = combined_encoded.xs("test")
    
    return X_train_enc, X_val_enc, X_test_enc


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def get_confusion_matrix_str(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return confusion matrix as a formatted string."""
    cm = confusion_matrix(y_true, y_pred)
    return f"[[{cm[0,0]:5d} {cm[0,1]:5d}]\n [{cm[1,0]:5d} {cm[1,1]:5d}]]"


# ---------------------------------------------------------------------------
# PHASE 1: Baseline Lock and Verification
# ---------------------------------------------------------------------------


def train_baseline_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[MLPClassifier, StandardScaler]:
    """Train the baseline MLP with the same architecture as in models.py.
    
    Architecture: (128, 64, 32) hidden layers, relu, adam, lr=0.001
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def save_baseline_mlp(model: MLPClassifier, scaler: StandardScaler) -> None:
    """Save the baseline MLP and scaler as a locked artifact."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "architecture": (128, 64, 32),
        "created_at": datetime.now().isoformat(),
        "description": "Baseline MLP - DO NOT OVERWRITE",
    }
    
    joblib.dump(artifact, BASELINE_MLP_PATH)
    print(f"Saved baseline MLP to: {BASELINE_MLP_PATH}")


def load_baseline_mlp() -> Tuple[MLPClassifier, StandardScaler]:
    """Load the locked baseline MLP from disk."""
    if not BASELINE_MLP_PATH.exists():
        raise FileNotFoundError(f"Baseline MLP not found at {BASELINE_MLP_PATH}")
    
    artifact = joblib.load(BASELINE_MLP_PATH)
    return artifact["model"], artifact["scaler"]


def evaluate_mlp(
    model: MLPClassifier,
    scaler: StandardScaler,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate MLP on given data at specified threshold."""
    X_scaled = scaler.transform(X)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    return compute_metrics(y.values, y_pred, y_proba)


def write_baseline_metrics_md(
    metrics: Dict[str, float],
    cm_str: str,
    path: Path = BASELINE_METRICS_MD,
) -> None:
    """Write baseline metrics to markdown file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# MLP Baseline Metrics\n\n",
        "These are the test-set metrics for the **locked baseline MLP** model.\n\n",
        "## Test Set Performance (threshold = 0.5)\n\n",
        "| Metric | Value |\n",
        "|--------|-------|\n",
        f"| Accuracy | {metrics['accuracy']:.4f} |\n",
        f"| Precision | {metrics['precision']:.4f} |\n",
        f"| Recall | {metrics['recall']:.4f} |\n",
        f"| F1 Score | {metrics['f1']:.4f} |\n",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |\n\n",
        "## Confusion Matrix\n\n",
        "```\n",
        "              Predicted\n",
        "              Neg    Pos\n",
        f"Actual Neg {cm_str.split(chr(10))[0]}\n",
        f"       Pos {cm_str.split(chr(10))[1]}\n",
        "```\n\n",
        "## Model Details\n\n",
        "- **Architecture:** (128, 64, 32) hidden layers\n",
        "- **Activation:** ReLU\n",
        "- **Optimizer:** Adam (lr=0.001)\n",
        "- **Early stopping:** Yes (validation_fraction=0.1)\n",
        "- **Lock status:** This model is frozen and will not be modified.\n",
    ]
    
    path.write_text("".join(lines), encoding="utf-8")
    print(f"Saved baseline metrics to: {path}")


def phase1_lock_baseline() -> Dict[str, float]:
    """Execute Phase 1: Train, save, and evaluate baseline MLP."""
    print("=" * 80)
    print("PHASE 1: BASELINE LOCK AND VERIFICATION")
    print("=" * 80)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    print(f"Loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Encode categorical features
    X_train, X_val, X_test = encode_categorical_features(X_train, X_val, X_test)
    print(f"Feature count after encoding: {X_train.shape[1]}")
    
    # Check if baseline already exists
    if BASELINE_MLP_PATH.exists():
        print(f"\nBaseline MLP already exists at {BASELINE_MLP_PATH}")
        print("Loading existing baseline (NOT overwriting)...")
        model, scaler = load_baseline_mlp()
    else:
        print("\nTraining baseline MLP...")
        model, scaler = train_baseline_mlp(X_train, y_train, X_val, y_val)
        save_baseline_mlp(model, scaler)
    
    # Evaluate on test set
    print("\nEvaluating baseline on test set...")
    X_test_scaled = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = compute_metrics(y_test.values, y_pred, y_proba)
    cm_str = get_confusion_matrix_str(y_test.values, y_pred)
    
    print("\nBaseline MLP - Test Set Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{cm_str}")
    
    # Save metrics to markdown
    write_baseline_metrics_md(metrics, cm_str)
    
    return metrics


# ---------------------------------------------------------------------------
# PHASE 2: Hyperparameter Search with Optuna
# ---------------------------------------------------------------------------


def create_mlp_from_params(params: Dict) -> MLPClassifier:
    """Create an MLP from hyperparameter dict."""
    # Build hidden layers
    n_layers = params.get("n_layers", 2)
    hidden_sizes = tuple(
        params.get(f"n_units_l{i}", 128)
        for i in range(n_layers)
    )
    
    return MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation=params.get("activation", "relu"),
        solver="adam",
        alpha=params.get("alpha", 0.0001),  # L2 regularization
        learning_rate_init=params.get("learning_rate", 0.001),
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        batch_size=params.get("batch_size", 128),
    )


def objective(
    trial: "optuna.Trial",
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    min_recall: float = 0.84,
) -> float:
    """Optuna objective function for MLP hyperparameter search.
    
    Maximizes validation accuracy subject to recall >= min_recall.
    """
    # Sample hyperparameters
    n_layers = trial.suggest_int("n_layers", 2, 4)
    
    hidden_sizes = []
    for i in range(n_layers):
        n_units = trial.suggest_categorical(f"n_units_l{i}", [64, 128, 256, 384, 512])
        hidden_sizes.append(n_units)
    
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    alpha = trial.suggest_float("alpha", 1e-6, 1e-3, log=True)  # L2 reg (weight decay)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # Create and train model
    model = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_sizes),
        activation=activation,
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        batch_size=batch_size,
    )
    
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"Training failed: {e}")
        return 0.0
    
    # Evaluate on validation set
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    # Store metrics for later analysis
    trial.set_user_attr("val_recall", val_recall)
    trial.set_user_attr("val_precision", val_precision)
    trial.set_user_attr("val_f1", val_f1)
    trial.set_user_attr("val_auc", val_auc)
    trial.set_user_attr("hidden_sizes", str(hidden_sizes))
    
    # Penalize if recall is too low
    if val_recall < min_recall:
        # Return a penalized score
        penalty = (min_recall - val_recall) * 2.0
        return val_acc - penalty
    
    return val_acc


def phase2_hyperparameter_search(
    n_trials: int = 100,
    min_recall: float = 0.84,
) -> pd.DataFrame:
    """Execute Phase 2: Hyperparameter search with Optuna."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter search")
    
    print("\n" + "=" * 80)
    print("PHASE 2: HYPERPARAMETER SEARCH")
    print("=" * 80)
    
    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    X_train, X_val, X_test = encode_categorical_features(X_train, X_val, X_test)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Target distribution (train): {y_train.value_counts().to_dict()}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="mlp_tuning",
    )
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print(f"\nRunning {n_trials} trials...")
    print(f"Objective: Maximize validation accuracy with recall >= {min_recall}")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, X_train_scaled, y_train.values, X_val_scaled, y_val.values, min_recall
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # Collect results
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = {
                "trial": trial.number,
                "val_accuracy": trial.value,
                "val_recall": trial.user_attrs.get("val_recall", 0),
                "val_precision": trial.user_attrs.get("val_precision", 0),
                "val_f1": trial.user_attrs.get("val_f1", 0),
                "val_auc": trial.user_attrs.get("val_auc", 0),
                "hidden_sizes": trial.user_attrs.get("hidden_sizes", ""),
                **trial.params,
            }
            results.append(row)
    
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TUNING_LOG_CSV, index=False)
    print(f"\nSaved tuning log to: {TUNING_LOG_CSV}")
    
    # Print best trials
    print("\nTop 5 trials by validation accuracy:")
    top5 = results_df.nlargest(5, "val_accuracy")
    print(top5[["trial", "val_accuracy", "val_recall", "hidden_sizes"]].to_string())
    
    return results_df


# ---------------------------------------------------------------------------
# PHASE 3: Candidate Selection and TEST Evaluation
# ---------------------------------------------------------------------------


def select_top_candidates(
    results_df: pd.DataFrame,
    n_candidates: int = 3,
    min_recall: float = 0.82,
) -> List[Dict]:
    """Select top N candidates that meet recall constraint."""
    # Filter by recall
    valid = results_df[results_df["val_recall"] >= min_recall]
    
    if len(valid) < n_candidates:
        print(f"Warning: Only {len(valid)} trials meet recall >= {min_recall}")
        # Relax constraint if needed
        valid = results_df.nlargest(n_candidates, "val_accuracy")
    
    # Select top by accuracy
    top = valid.nlargest(n_candidates, "val_accuracy")
    
    candidates = []
    for _, row in top.iterrows():
        params = {
            "n_layers": row.get("n_layers", 2),
            "activation": row.get("activation", "relu"),
            "learning_rate": row.get("learning_rate", 0.001),
            "alpha": row.get("alpha", 0.0001),
            "batch_size": int(row.get("batch_size", 128)),
        }
        # Extract hidden layer sizes
        for i in range(int(params["n_layers"])):
            key = f"n_units_l{i}"
            if key in row:
                params[key] = int(row[key])
        
        params["val_accuracy"] = row["val_accuracy"]
        params["val_recall"] = row["val_recall"]
        candidates.append(params)
    
    return candidates


def train_candidate_on_full_train(
    params: Dict,
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
) -> Tuple[MLPClassifier, StandardScaler]:
    """Train a candidate MLP on combined train+val data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_val)
    
    # Build hidden layers
    n_layers = params.get("n_layers", 2)
    hidden_sizes = tuple(
        params.get(f"n_units_l{i}", 128)
        for i in range(n_layers)
    )
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation=params.get("activation", "relu"),
        solver="adam",
        alpha=params.get("alpha", 0.0001),
        learning_rate_init=params.get("learning_rate", 0.001),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        batch_size=params.get("batch_size", 128),
    )
    
    model.fit(X_scaled, y_train_val)
    
    return model, scaler


def phase3_evaluate_candidates(
    results_df: pd.DataFrame,
    baseline_metrics: Dict[str, float],
) -> Tuple[List[Dict], Optional[Dict]]:
    """Execute Phase 3: Evaluate top candidates on test set."""
    print("\n" + "=" * 80)
    print("PHASE 3: CANDIDATE SELECTION AND TEST EVALUATION")
    print("=" * 80)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    X_train, X_val, X_test = encode_categorical_features(X_train, X_val, X_test)
    
    # Combine train + val
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    
    print(f"Combined train+val: {len(X_train_val)} samples")
    
    # Select top candidates
    candidates = select_top_candidates(results_df, n_candidates=3, min_recall=0.82)
    print(f"\nSelected {len(candidates)} candidates for test evaluation")
    
    candidate_results = []
    
    for i, params in enumerate(candidates, 1):
        print(f"\n--- Candidate {i} ---")
        print(f"  Val Accuracy: {params['val_accuracy']:.4f}")
        print(f"  Val Recall:   {params['val_recall']:.4f}")
        
        # Train on full train+val
        model, scaler = train_candidate_on_full_train(params, X_train_val, y_train_val)
        
        # Evaluate on test set
        X_test_scaled = scaler.transform(X_test)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        test_metrics = compute_metrics(y_test.values, y_pred, y_proba)
        cm_str = get_confusion_matrix_str(y_test.values, y_pred)
        
        print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Test Recall:    {test_metrics['recall']:.4f}")
        print(f"  Test Precision: {test_metrics['precision']:.4f}")
        print(f"  Test F1:        {test_metrics['f1']:.4f}")
        print(f"  Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        
        result = {
            "candidate_id": i,
            "params": params,
            "test_metrics": test_metrics,
            "confusion_matrix": cm_str,
            "model": model,
            "scaler": scaler,
        }
        candidate_results.append(result)
    
    # Find best candidate
    best_candidate = None
    baseline_acc = baseline_metrics["accuracy"]
    baseline_recall = baseline_metrics["recall"]
    
    for result in candidate_results:
        test_acc = result["test_metrics"]["accuracy"]
        test_recall = result["test_metrics"]["recall"]
        
        # Check if better than baseline
        acc_improvement = test_acc - baseline_acc
        recall_drop = baseline_recall - test_recall
        
        if acc_improvement >= 0.005 and recall_drop <= 0.02:
            if best_candidate is None or test_acc > best_candidate["test_metrics"]["accuracy"]:
                best_candidate = result
    
    return candidate_results, best_candidate


def write_candidates_comparison_md(
    baseline_metrics: Dict[str, float],
    candidate_results: List[Dict],
    best_candidate: Optional[Dict],
    path: Path = CANDIDATES_VS_BASELINE_MD,
) -> None:
    """Write comparison table to markdown."""
    lines = [
        "# MLP Candidates vs Baseline Comparison\n\n",
        "This table compares the locked baseline MLP against tuned candidates.\n\n",
        "## Test Set Metrics (threshold = 0.5)\n\n",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n",
        "|-------|----------|-----------|--------|----|---------|\n",
        f"| **Baseline_MLP** | {baseline_metrics['accuracy']:.4f} | ",
        f"{baseline_metrics['precision']:.4f} | {baseline_metrics['recall']:.4f} | ",
        f"{baseline_metrics['f1']:.4f} | {baseline_metrics['roc_auc']:.4f} |\n",
    ]
    
    for result in candidate_results:
        m = result["test_metrics"]
        lines.append(
            f"| Candidate_{result['candidate_id']} | {m['accuracy']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['roc_auc']:.4f} |\n"
        )
    
    lines.append("\n## Selection Result\n\n")
    
    if best_candidate:
        lines.append(
            f"**Candidate_{best_candidate['candidate_id']}** is selected as the new best model.\n\n"
            f"- Accuracy improvement: +{best_candidate['test_metrics']['accuracy'] - baseline_metrics['accuracy']:.4f}\n"
            f"- Recall change: {best_candidate['test_metrics']['recall'] - baseline_metrics['recall']:+.4f}\n"
        )
    else:
        lines.append(
            "**No candidate beats the baseline** with sufficient accuracy gain "
            "while maintaining recall. The baseline MLP remains the official model.\n"
        )
    
    path.write_text("".join(lines), encoding="utf-8")
    print(f"\nSaved comparison to: {path}")


# ---------------------------------------------------------------------------
# PHASE 4: Save Improved Model
# ---------------------------------------------------------------------------


def phase4_save_best_model(
    best_candidate: Optional[Dict],
    baseline_metrics: Dict[str, float],
) -> None:
    """Execute Phase 4: Save improved model if found."""
    print("\n" + "=" * 80)
    print("PHASE 4: SAVE BEST MODEL")
    print("=" * 80)
    
    if best_candidate is None:
        print("\nNo improved model found. Baseline MLP remains the official model.")
        write_best_summary_md(baseline_metrics, None)
        return
    
    # Save new best model
    artifact = {
        "model": best_candidate["model"],
        "scaler": best_candidate["scaler"],
        "params": best_candidate["params"],
        "test_metrics": best_candidate["test_metrics"],
        "created_at": datetime.now().isoformat(),
        "description": "Improved MLP v2 - tuned hyperparameters",
    }
    
    joblib.dump(artifact, MLP_V2_PATH)
    print(f"Saved improved MLP to: {MLP_V2_PATH}")
    
    write_best_summary_md(baseline_metrics, best_candidate)


def write_best_summary_md(
    baseline_metrics: Dict[str, float],
    best_candidate: Optional[Dict],
    path: Path = MLP_BEST_SUMMARY_MD,
) -> None:
    """Write final summary markdown."""
    lines = [
        "# MLP Best Model Summary\n\n",
        "## Baseline MLP Test Metrics\n\n",
        f"- Accuracy: {baseline_metrics['accuracy']:.4f}\n",
        f"- Precision: {baseline_metrics['precision']:.4f}\n",
        f"- Recall: {baseline_metrics['recall']:.4f}\n",
        f"- F1: {baseline_metrics['f1']:.4f}\n",
        f"- ROC-AUC: {baseline_metrics['roc_auc']:.4f}\n\n",
    ]
    
    if best_candidate:
        m = best_candidate["test_metrics"]
        lines.extend([
            "## Best Candidate MLP Test Metrics\n\n",
            f"- Accuracy: {m['accuracy']:.4f}\n",
            f"- Precision: {m['precision']:.4f}\n",
            f"- Recall: {m['recall']:.4f}\n",
            f"- F1: {m['f1']:.4f}\n",
            f"- ROC-AUC: {m['roc_auc']:.4f}\n\n",
            "## Final Choice\n\n",
            f"**mlp_v2_best** (Candidate_{best_candidate['candidate_id']})\n\n",
            "### Justification\n\n",
            f"The new MLP achieves {m['accuracy']:.4f} test accuracy "
            f"(+{m['accuracy'] - baseline_metrics['accuracy']:.4f} vs baseline) "
            f"while maintaining recall at {m['recall']:.4f} "
            f"(change: {m['recall'] - baseline_metrics['recall']:+.4f}), "
            "meeting my criteria for accuracy improvement without sacrificing sensitivity.\n",
        ])
    else:
        lines.extend([
            "## Best Candidate MLP Test Metrics\n\n",
            "No candidate improved upon the baseline.\n\n",
            "## Final Choice\n\n",
            "**Baseline_MLP** (mlp_baseline_locked.pkl)\n\n",
            "### Justification\n\n",
            "The baseline MLP already achieves strong performance with high accuracy and recall. "
            "None of the tuned candidates provided sufficient accuracy gains while maintaining recall, "
            "so I keep the original baseline as the official model.\n",
        ])
    
    path.write_text("".join(lines), encoding="utf-8")
    print(f"Saved summary to: {path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_full_pipeline(n_trials: int = 100) -> None:
    """Run the complete MLP tuning pipeline (Phases 1-4)."""
    print("=" * 80)
    print("CARDIODETECT MLP TUNING PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Phase 1: Lock baseline
    baseline_metrics = phase1_lock_baseline()
    
    # Phase 2: Hyperparameter search
    results_df = phase2_hyperparameter_search(n_trials=n_trials, min_recall=0.84)
    
    # Phase 3: Evaluate candidates
    candidate_results, best_candidate = phase3_evaluate_candidates(
        results_df, baseline_metrics
    )
    
    # Write comparison
    write_candidates_comparison_md(baseline_metrics, candidate_results, best_candidate)
    
    # Phase 4: Save best model
    phase4_save_best_model(best_candidate, baseline_metrics)
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nBaseline MLP locked at: {BASELINE_MLP_PATH}")
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Baseline Recall:   {baseline_metrics['recall']:.4f}")
    
    if best_candidate:
        m = best_candidate["test_metrics"]
        print(f"\nNew best MLP saved at: {MLP_V2_PATH}")
        print(f"New Accuracy: {m['accuracy']:.4f} (+{m['accuracy'] - baseline_metrics['accuracy']:.4f})")
        print(f"New Recall:   {m['recall']:.4f} ({m['recall'] - baseline_metrics['recall']:+.4f})")
        print(f"\nFinal choice: mlp_v2_best")
    else:
        print(f"\nNo improved model found.")
        print(f"Final choice: Baseline_MLP (unchanged)")


if __name__ == "__main__":
    run_full_pipeline(n_trials=100)
