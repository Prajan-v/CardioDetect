from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlp_tuning import MLP_V2_PATH  # noqa: E402  (may be useful for comparisons later)

# Paths
DATA_SPLIT_DIR = PROJECT_ROOT / "data" / "split"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Global constants
CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]
CLASS_WEIGHTS_OBJECTIVE = {0: 1.0, 1: 2.0, 2: 3.0}
N_TRIALS = 5000
RANDOM_STATE = 42
STUDY_PATH = EXPERIMENTS_DIR / "optuna_3class_study.pkl"

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_3class_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = DATA_SPLIT_DIR / "train_3class.csv"
    val_path = DATA_SPLIT_DIR / "val_3class.csv"
    test_path = DATA_SPLIT_DIR / "test_3class.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "3-class split files not found. Run experiments/create_3class_labels.py first."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare X/y from *_3class.csv dataframe."""
    y = df["risk_class_3"].astype(int)

    drop_cols = [
        "risk_class_3",
        "risk_target",
        "target",
        "TenYearCHD",
        "data_source",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y


def encode_categorical_features_3class(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical features consistently across splits."""
    combined = pd.concat(
        [X_train, X_val, X_test],
        keys=["train", "val", "test"],
        names=["split", None],
    )

    categorical_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        return X_train, X_val, X_test

    combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

    X_train_enc = combined_encoded.xs("train")
    X_val_enc = combined_encoded.xs("val")
    X_test_enc = combined_encoded.xs("test")

    return X_train_enc, X_val_enc, X_test_enc


def compute_class_weights(y: pd.Series) -> np.ndarray:
    classes = np.array(sorted(y.unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return weights


def compute_weighted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_weights: Dict[int, float] | None = None,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Compute per-class metrics and a custom weighted F1 score.

    weighted F1 uses CLASS_WEIGHTS_OBJECTIVE if class_weights is None.
    """
    if class_weights is None:
        class_weights = CLASS_WEIGHTS_OBJECTIVE

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    metrics = {
        "precision_per_class": {name: float(p) for name, p in zip(CLASS_NAMES, prec)},
        "recall_per_class": {name: float(r) for name, r in zip(CLASS_NAMES, rec)},
        "f1_per_class": {name: float(f) for name, f in zip(CLASS_NAMES, f1)},
        "support_per_class": {name: int(s) for name, s in zip(CLASS_NAMES, support)},
    }

    num = 0.0
    denom = 0.0
    for idx, name in enumerate(CLASS_NAMES):
        w = class_weights[idx]
        num += w * f1[idx]
        denom += w

    weighted_f1 = float(num / denom) if denom > 0 else 0.0
    return weighted_f1, metrics


def build_data() -> Dict[str, object]:
    """Load and preprocess data once, reused across trials."""
    print("Loading 3-class data splits...")
    train_df, val_df, test_df = load_3class_splits()

    X_train_raw, y_train = prepare_features_and_target(train_df)
    X_val_raw, y_val = prepare_features_and_target(val_df)
    X_test_raw, y_test = prepare_features_and_target(test_df)

    print("Encoding categorical features...")
    X_train_enc, X_val_enc, X_test_enc = encode_categorical_features_3class(
        X_train_raw, X_val_raw, X_test_raw
    )

    print("Train: %d samples" % len(X_train_enc))
    print("Val:   %d samples" % len(X_val_enc))
    print("Test:  %d samples" % len(X_test_enc))

    return {
        "X_train": X_train_enc,
        "y_train": y_train,
        "X_val": X_val_enc,
        "y_val": y_val,
        "X_test": X_test_enc,
        "y_test": y_test,
    }


def create_objective(data: Dict[str, object]):
    X_train: pd.DataFrame = data["X_train"]  # type: ignore
    y_train: pd.Series = data["y_train"]  # type: ignore
    X_val: pd.DataFrame = data["X_val"]  # type: ignore
    y_val: pd.Series = data["y_val"]  # type: ignore

    # Balanced class weights for training
    class_weights_balanced = compute_class_weights(y_train)
    classes = np.array(sorted(y_train.unique()))
    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights_balanced)}
    sample_weight_train = y_train.map(class_weight_dict).values

    # Global feature ranking (filter-based) using mutual information
    mi = mutual_info_classif(
        X_train,
        y_train,
        discrete_features="auto",
        random_state=RANDOM_STATE,
    )
    feature_ranking = [
        f for _, f in sorted(zip(mi, X_train.columns), reverse=True)
    ]

    def objective(trial: optuna.trial.Trial) -> float:
        # Hyperparameters
        hidden_layers = trial.suggest_categorical(
            "hidden_layers",
            [
                (128, 64, 32),
                (256, 128, 64),
                (256, 256, 128),
                (128, 64),
                (64, 32),
            ],
        )
        alpha = trial.suggest_categorical("alpha", [1e-5, 1e-4, 1e-3, 1e-2])
        learning_rate_init = trial.suggest_float(
            "learning_rate_init", 1e-4, 3e-3, log=True
        )
        batch_size = trial.suggest_categorical(
            "batch_size", ["auto", 32, 64, 128]
        )
        max_iter = trial.suggest_categorical("max_iter", [300, 500, 700])
        activation = trial.suggest_categorical(
            "activation",
            ["relu", "tanh", "logistic"],
        )
        feature_fraction = trial.suggest_categorical(
            "feature_fraction",
            [0.4, 0.6, 0.8, 1.0],
        )

        # Feature subset based on global ranking
        k = max(10, int(len(feature_ranking) * float(feature_fraction)))
        selected_features = feature_ranking[:k]

        X_train_sel = X_train[selected_features]
        X_val_sel = X_val[selected_features]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_val_scaled = scaler.transform(X_val_sel)

        # Model
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
            verbose=False,
        )

        try:
            mlp.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)

            y_val_pred = mlp.predict(X_val_scaled)
            acc = accuracy_score(y_val, y_val_pred)
            weighted_f1, metrics = compute_weighted_f1(y_val.values, y_val_pred)

            # High recall constraint (class 2 = HIGH)
            recall_high = metrics["recall_per_class"]["HIGH"]

            # Safety / quality constraints
            if recall_high < 0.90 or acc < 0.85:
                trial.set_user_attr("valid", False)
                trial.set_user_attr("recall_high", recall_high)
                trial.set_user_attr("accuracy", acc)
                trial.report(0.0, step=0)
                return 0.0

            trial.set_user_attr("valid", True)
            trial.set_user_attr("recall_high", recall_high)
            trial.set_user_attr("accuracy", acc)
            trial.set_user_attr("weighted_f1", weighted_f1)

            # Report to enable (coarse) pruning
            trial.report(weighted_f1, step=1)

            return weighted_f1

        except Exception as e:  # noqa: BLE001
            # Failed trial – return 0 score but continue study
            trial.set_user_attr("exception", str(e))
            trial.report(0.0, step=0)
            return 0.0

    return objective


def compute_test_metrics(
    model: MLPClassifier,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = float(accuracy_score(y_test, y_pred))
    weighted_f1, metrics = compute_weighted_f1(y_test.values, y_pred)

    print("\n======================================================================")
    print("TEST SET RESULTS (TUNED MODEL)")
    print("======================================================================")
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nPer-Class Metrics:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print("Rows = true, Cols = predicted (0=LOW,1=MEDIUM,2=HIGH)")
    print(cm)

    test_metrics = {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "precision_per_class": metrics["precision_per_class"],
        "recall_per_class": metrics["recall_per_class"],
        "f1_per_class": metrics["f1_per_class"],
        "support_per_class": metrics["support_per_class"],
    }
    return test_metrics


def evaluate_baseline(
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    """Evaluate existing mlp_v3_3class baseline for comparison, if present."""
    baseline_path = MODELS_DIR / "mlp_v3_3class.pkl"
    if not baseline_path.exists():
        print("Baseline mlp_v3_3class.pkl not found; skipping baseline comparison.")
        return {}

    artifact = joblib.load(baseline_path)
    model: MLPClassifier = artifact["model"]
    scaler: StandardScaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    X_test_base = X_test.copy()
    missing = set(feature_names) - set(X_test_base.columns)
    if missing:
        raise ValueError(
            f"X_test missing features required by baseline model: {sorted(missing)}"
        )

    X_test_base = X_test_base[feature_names]
    X_test_scaled = scaler.transform(X_test_base)
    y_pred = model.predict(X_test_scaled)

    acc = float(accuracy_score(y_test, y_pred))
    _, metrics = compute_weighted_f1(y_test.values, y_pred)

    print("\nBaseline mlp_v3_3class (untuned) on test set:")
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nPer-Class Metrics:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    baseline_metrics = {
        "accuracy": acc,
        "precision_per_class": metrics["precision_per_class"],
        "recall_per_class": metrics["recall_per_class"],
        "f1_per_class": metrics["f1_per_class"],
    }
    return baseline_metrics


def run_optimization(data: Dict[str, object]) -> optuna.Study:
    objective = create_objective(data)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="mlp_3class_tuning",
    )

    def log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Progress logging every 10 trials
        if (trial.number + 1) % 10 == 0:
            print(
                f"[Progress] Trial {trial.number + 1}/{N_TRIALS} - "
                f"Best weighted F1 so far: {study.best_value:.4f}"
            )
        # Save intermediate study every 20 trials
        if (trial.number + 1) % 20 == 0:
            joblib.dump(study, STUDY_PATH)
            print(f"[Info] Saved Optuna study checkpoint to {STUDY_PATH}")

    print(
        "\n======================================================================"\
        "\nOPTUNA HYPERPARAMETER TUNING - 3-CLASS RISK MODEL"\
        "\n======================================================================"
    )
    print("Starting Optuna optimization (%d trials)..." % N_TRIALS)

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_callback])

    # Final save
    joblib.dump(study, STUDY_PATH)
    print(f"\nSaved Optuna study to {STUDY_PATH}")

    return study


def resume_optimization(data: Dict[str, object]) -> optuna.Study:
    if not STUDY_PATH.exists():
        raise FileNotFoundError(f"Optuna study not found at {STUDY_PATH}")
    study: optuna.Study = joblib.load(STUDY_PATH)
    objective = create_objective(data)
    n_completed = len(study.trials)
    remaining = max(0, N_TRIALS - n_completed)

    def log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if len(study.trials) % 10 == 0:
            print(
                f"[Progress] Trial {len(study.trials)}/{N_TRIALS} - "
                f"Best weighted F1 so far: {study.best_value:.4f}"
            )
        if len(study.trials) % 20 == 0:
            joblib.dump(study, STUDY_PATH)
            print(f"[Info] Saved Optuna study checkpoint to {STUDY_PATH}")

    if remaining <= 0:
        print(
            f"Existing Optuna study already has {n_completed} trials "
            f"(N_TRIALS={N_TRIALS}); no additional trials will be run."
        )
        return study

    print(
        f"\nResuming Optuna optimization: {n_completed} completed trials, "
        f"running {remaining} additional trials to reach {N_TRIALS} total."
    )
    study.optimize(objective, n_trials=remaining, callbacks=[log_callback])

    joblib.dump(study, STUDY_PATH)
    print(f"\nSaved Optuna study to {STUDY_PATH}")

    return study


def main() -> None:
    data = build_data()

    reuse_study = "--reuse-study" in sys.argv
    resume_study = "--resume" in sys.argv

    if reuse_study and resume_study:
        raise ValueError("Cannot use --reuse-study and --resume together")

    if reuse_study and STUDY_PATH.exists():
        print(f"\nReusing existing Optuna study from {STUDY_PATH}")
        study: optuna.Study = joblib.load(STUDY_PATH)
    elif resume_study:
        print(
            f"\nResuming existing Optuna study from {STUDY_PATH} "
            f"towards {N_TRIALS} total trials"
        )
        study = resume_optimization(data)
    else:
        study = run_optimization(data)
    best_trial = study.best_trial

    print("\nBest trial (#%d):" % best_trial.number)
    print(f"Value (weighted F1): {best_trial.value:.4f}")
    print("Params:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    # Retrain best model on full training set (train + val)
    X_train: pd.DataFrame = data["X_train"]  # type: ignore
    y_train: pd.Series = data["y_train"]  # type: ignore
    X_val: pd.DataFrame = data["X_val"]  # type: ignore
    y_val: pd.Series = data["y_val"]  # type: ignore
    X_test: pd.DataFrame = data["X_test"]  # type: ignore
    y_test: pd.Series = data["y_test"]  # type: ignore

    X_full = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_full = pd.concat([y_train, y_val], axis=0, ignore_index=True)

    print("\nRetraining best model on full training set (train + val)...")

    # Recompute feature ranking on full training data
    mi_full = mutual_info_classif(
        X_full,
        y_full,
        discrete_features="auto",
        random_state=RANDOM_STATE,
    )
    feature_ranking_full = [
        f for _, f in sorted(zip(mi_full, X_full.columns), reverse=True)
    ]

    # Use best feature_fraction from Optuna to select top-K features
    best_params = best_trial.params.copy()
    feature_fraction_best = float(best_params.get("feature_fraction", 1.0))
    k_full = max(10, int(len(feature_ranking_full) * feature_fraction_best))
    selected_features = feature_ranking_full[:k_full]

    X_full_sel = X_full[selected_features]
    X_test_sel = X_test[selected_features]

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full_sel)
    X_test_scaled_ref = scaler.transform(X_test_sel)  # used only to verify dimensions
    feature_names = list(selected_features)

    # Class weights for final training (balanced)
    classes = np.array(sorted(y_full.unique()))
    class_weights_balanced = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_full
    )
    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights_balanced)}
    sample_weight_full = y_full.map(class_weight_dict).values

    # Instantiate best model
    mlp_best = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layers"],
        activation=best_params.get("activation", "relu"),
        solver="adam",
        alpha=best_params["alpha"],
        learning_rate_init=best_params["learning_rate_init"],
        batch_size=best_params["batch_size"],
        max_iter=best_params["max_iter"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    mlp_best.fit(X_full_scaled, y_full, sample_weight=sample_weight_full)

    # Compute test metrics for tuned model
    test_metrics_tuned = compute_test_metrics(
        mlp_best,
        scaler,
        X_test_sel,
        y_test,
    )

    # Baseline comparison (if available)
    baseline_metrics = evaluate_baseline(X_test, y_test)

    if baseline_metrics:
        print("\nKey Improvements vs Baseline (untuned mlp_v3_3class):")
        acc_base = baseline_metrics["accuracy"]
        acc_tuned = test_metrics_tuned["accuracy"]
        print(
            f"Overall accuracy: {acc_tuned - acc_base:+.4f} "
            f"(baseline: {acc_base:.4f} -> tuned: {acc_tuned:.4f})"
        )
        rec_base_med = baseline_metrics["recall_per_class"]["MEDIUM"]
        rec_tuned_med = test_metrics_tuned["recall_per_class"]["MEDIUM"]
        print(
            f"MEDIUM recall: {rec_base_med*100:.1f}% -> "
            f"{rec_tuned_med*100:.1f}% "
            f"(change: {(rec_tuned_med-rec_base_med)*100:.1f} pp)"
        )
        rec_base_high = baseline_metrics["recall_per_class"]["HIGH"]
        rec_tuned_high = test_metrics_tuned["recall_per_class"]["HIGH"]
        print(
            f"HIGH recall: {rec_base_high*100:.1f}% -> "
            f"{rec_tuned_high*100:.1f}%"
        )

    # Save tuned model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mlp_v3_tuned.pkl"

    artifact = {
        "model": mlp_best,
        "scaler": scaler,
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
        "best_params": best_trial.params,
        "val_score": float(best_trial.value),
        "test_metrics": test_metrics_tuned,
    }

    joblib.dump(artifact, model_path)
    print(f"\n\u2713 Model saved to {model_path}")


if __name__ == "__main__":
    main()
