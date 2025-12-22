"""
Risk scoring and categorization module.

Provides functions to categorize continuous risk into LOW/MODERATE/HIGH.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple


# ============================================================================
# DEFAULT THRESHOLDS
# ============================================================================

DEFAULT_THRESHOLDS = {
    "low": 10.0,   # <10% is LOW
    "high": 25.0,  # >=25% is HIGH, 10-25% is MODERATE
}


# ============================================================================
# RISK CATEGORIZATION
# ============================================================================

def categorize_risk(
    risk_percent: float,
    low_threshold: float = DEFAULT_THRESHOLDS["low"],
    high_threshold: float = DEFAULT_THRESHOLDS["high"],
) -> str:
    """Categorize risk percentage into LOW/MODERATE/HIGH.

    Args:
        risk_percent: Risk as percentage (0–100).
        low_threshold: Upper bound for LOW category (default 10%).
        high_threshold: Lower bound for HIGH category (default 25%).

    Returns:
        Category string: "LOW", "MODERATE", or "HIGH".
    """
    if risk_percent < low_threshold:
        return "LOW"
    elif risk_percent < high_threshold:
        return "MODERATE"
    else:
        return "HIGH"


def categorize_risk_from_probability(
    risk: float,
    low_threshold: float = DEFAULT_THRESHOLDS["low"] / 100.0,
    high_threshold: float = DEFAULT_THRESHOLDS["high"] / 100.0,
) -> str:
    """Categorize risk probability (0–1) into LOW/MODERATE/HIGH.

    Args:
        risk: Risk as probability (0–1).
        low_threshold: Upper bound for LOW category (default 0.10).
        high_threshold: Lower bound for HIGH category (default 0.25).

    Returns:
        Category string: "LOW", "MODERATE", or "HIGH".
    """
    return categorize_risk(risk * 100.0, low_threshold * 100.0, high_threshold * 100.0)


def get_category_color(category: str) -> str:
    """Get display color for risk category.

    Args:
        category: Risk category string.

    Returns:
        Color name for display.
    """
    colors = {
        "LOW": "green",
        "MODERATE": "orange",
        "HIGH": "red",
    }
    return colors.get(category, "gray")


def get_category_emoji(category: str) -> str:
    """Get emoji for risk category.

    Args:
        category: Risk category string.

    Returns:
        Emoji string.
    """
    emojis = {
        "LOW": "✅",
        "MODERATE": "⚠️",
        "HIGH": "🚨",
    }
    return emojis.get(category, "❓")


# ============================================================================
# THRESHOLD CONFIGURATION
# ============================================================================

def load_thresholds(config_path: Optional[Path] = None) -> Dict[str, float]:
    """Load threshold configuration from JSON file.

    Args:
        config_path: Path to JSON config file.

    Returns:
        Dictionary with 'low' and 'high' threshold values.
    """
    if config_path is None or not config_path.exists():
        return DEFAULT_THRESHOLDS.copy()

    with open(config_path, "r") as f:
        config = json.load(f)

    return {
        "low": config.get("low", DEFAULT_THRESHOLDS["low"]),
        "high": config.get("high", DEFAULT_THRESHOLDS["high"]),
    }


def save_thresholds(
    thresholds: Dict[str, float],
    config_path: Path,
) -> None:
    """Save threshold configuration to JSON file.

    Args:
        thresholds: Dictionary with 'low' and 'high' values.
        config_path: Path to save JSON config.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(thresholds, f, indent=2)


# ============================================================================
# RISK SCORE UTILITIES
# ============================================================================

def compute_risk_score(
    probability: float,
    scale: str = "percent",
) -> float:
    """Convert probability to risk score.

    Args:
        probability: Risk probability (0–1).
        scale: Output scale - 'percent' (0–100) or 'probability' (0–1).

    Returns:
        Risk score in specified scale.
    """
    if scale == "percent":
        return probability * 100.0
    return probability


def get_risk_summary(
    risk: float,
    low_threshold: float = DEFAULT_THRESHOLDS["low"],
    high_threshold: float = DEFAULT_THRESHOLDS["high"],
) -> Dict[str, any]:
    """Get comprehensive risk summary.

    Args:
        risk: Risk as probability (0–1).
        low_threshold: LOW/MODERATE boundary (%).
        high_threshold: MODERATE/HIGH boundary (%).

    Returns:
        Dictionary with risk details.
    """
    risk_percent = risk * 100.0
    category = categorize_risk(risk_percent, low_threshold, high_threshold)

    return {
        "risk_probability": round(risk, 4),
        "risk_percent": round(risk_percent, 2),
        "category": category,
        "color": get_category_color(category),
        "emoji": get_category_emoji(category),
        "thresholds": {
            "low": low_threshold,
            "high": high_threshold,
        },
    }


# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================

def analyze_threshold_impact(
    risks: list,
    low_values: list = [5, 7.5, 10, 12.5],
    high_values: list = [20, 25, 30],
) -> Dict[Tuple[float, float], Dict[str, int]]:
    """Analyze how different thresholds affect category distributions.

    Args:
        risks: List of risk probabilities (0–1).
        low_values: List of LOW thresholds to test (%).
        high_values: List of HIGH thresholds to test (%).

    Returns:
        Dictionary mapping (low, high) to category counts.
    """
    import numpy as np

    risks = np.asarray(risks) * 100.0  # Convert to percent
    results = {}

    for low in low_values:
        for high in high_values:
            if low >= high:
                continue

            low_count = int(np.sum(risks < low))
            mod_count = int(np.sum((risks >= low) & (risks < high)))
            high_count = int(np.sum(risks >= high))

            results[(low, high)] = {
                "LOW": low_count,
                "MODERATE": mod_count,
                "HIGH": high_count,
            }

    return results
