"""
I/O utilities for saving and loading model artifacts and metadata.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================================
# MODEL ARTIFACTS
# ============================================================================

def save_model_artifact(
    model: Any,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None,
) -> Path:
    """Save sklearn model/pipeline with metadata.

    Args:
        model: Trained sklearn model or pipeline.
        model_name: Name for the model file (without extension).
        metadata: Optional metadata dictionary.
        models_dir: Directory to save model. Defaults to milestone_2/models.

    Returns:
        Path to saved model file.
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)

    # Save metadata
    if metadata is None:
        metadata = {}

    metadata["model_name"] = model_name
    metadata["saved_at"] = datetime.now().isoformat()
    metadata["model_path"] = str(model_path)

    meta_path = models_dir / f"{model_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {meta_path}")

    return model_path


def load_model_artifact(
    model_name: str,
    models_dir: Optional[Path] = None,
) -> tuple:
    """Load sklearn model/pipeline with metadata.

    Args:
        model_name: Name of the model file (without extension).
        models_dir: Directory containing model. Defaults to milestone_2/models.

    Returns:
        Tuple of (model, metadata dict).
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    model_path = models_dir / f"{model_name}.pkl"
    meta_path = models_dir / f"{model_name}_meta.json"

    model = joblib.load(model_path)

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return model, metadata


def list_saved_models(models_dir: Optional[Path] = None) -> List[str]:
    """List all saved model names.

    Args:
        models_dir: Directory containing models.

    Returns:
        List of model names (without .pkl extension).
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    if not models_dir.exists():
        return []

    return [p.stem for p in models_dir.glob("*.pkl")]


# ============================================================================
# METRICS AND REPORTS
# ============================================================================

def save_metrics(
    metrics: Dict[str, Any],
    run_name: str,
    reports_dir: Optional[Path] = None,
) -> Path:
    """Save evaluation metrics to JSON.

    Args:
        metrics: Dictionary of metric values.
        run_name: Name for this run/experiment.
        reports_dir: Directory to save report.

    Returns:
        Path to saved metrics file.
    """
    if reports_dir is None:
        reports_dir = REPORTS_DIR / "metrics"

    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics["run_name"] = run_name
    metrics["saved_at"] = datetime.now().isoformat()

    metrics_path = reports_dir / f"{run_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"Saved metrics: {metrics_path}")
    return metrics_path


def load_metrics(
    run_name: str,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load evaluation metrics from JSON.

    Args:
        run_name: Name of the run/experiment.
        reports_dir: Directory containing reports.

    Returns:
        Dictionary of metric values.
    """
    if reports_dir is None:
        reports_dir = REPORTS_DIR / "metrics"

    metrics_path = reports_dir / f"{run_name}_metrics.json"

    with open(metrics_path, "r") as f:
        return json.load(f)


def save_comparison_table(
    results: List[Dict[str, Any]],
    filename: str = "model_comparison.json",
    reports_dir: Optional[Path] = None,
) -> Path:
    """Save model comparison table.

    Args:
        results: List of result dictionaries (one per model).
        filename: Output filename.
        reports_dir: Directory to save report.

    Returns:
        Path to saved comparison file.
    """
    if reports_dir is None:
        reports_dir = REPORTS_DIR / "comparisons"

    reports_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "saved_at": datetime.now().isoformat(),
        "models": results,
    }

    output_path = reports_dir / filename
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Saved comparison: {output_path}")
    return output_path


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

def create_run_id(prefix: str = "run") -> str:
    """Create unique run identifier.

    Args:
        prefix: Prefix for run ID.

    Returns:
        Run ID string like 'run_20231204_153045'.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def save_experiment_config(
    config: Dict[str, Any],
    run_id: str,
    reports_dir: Optional[Path] = None,
) -> Path:
    """Save experiment configuration.

    Args:
        config: Configuration dictionary.
        run_id: Unique run identifier.
        reports_dir: Directory to save config.

    Returns:
        Path to saved config file.
    """
    if reports_dir is None:
        reports_dir = REPORTS_DIR / "configs"

    reports_dir.mkdir(parents=True, exist_ok=True)

    config_path = reports_dir / f"{run_id}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    return config_path
