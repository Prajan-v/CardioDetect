"""
Target generation module for Milestone 2.

Provides functions to compute guideline-based 10-year CVD risk and
convert continuous risk into binary or 3-class labels.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# GUIDELINE RISK CALCULATION
# ============================================================================

def compute_guideline_risk(
    *,
    age: Optional[float],
    sex: Optional[int],  # 0=female, 1=male
    total_cholesterol: Optional[float],
    hdl_cholesterol: Optional[float] = None,
    systolic_bp: Optional[float],
    on_treatment: bool = False,
    smoking: Optional[float] = 0.0,
    diabetes: Optional[float] = 0.0,
) -> float:
    """Compute 10-year general CVD risk using Framingham-like coefficients.

    Based on D'Agostino et al. (2008) general CVD risk profile.

    Args:
        age: Age in years (30–74 in original study).
        sex: 0 for female, 1 for male.
        total_cholesterol: Total cholesterol (mg/dL).
        hdl_cholesterol: HDL cholesterol (mg/dL). Defaults to 50 if missing.
        systolic_bp: Systolic blood pressure (mmHg).
        on_treatment: Whether patient is on BP treatment.
        smoking: Indicator (0/1) for current smoking.
        diabetes: Indicator (0/1) for diabetes.

    Returns:
        Estimated 10-year CVD risk as float between 0 and 1.
    """
    if age is None or systolic_bp is None or total_cholesterol is None:
        return float("nan")

    if hdl_cholesterol is None:
        hdl_cholesterol = 50.0

    try:
        age = float(age)
        sex = int(sex) if sex is not None else 1
        total_cholesterol = float(total_cholesterol)
        hdl_cholesterol = float(hdl_cholesterol)
        systolic_bp = float(systolic_bp)
        smoking = 1.0 if (smoking is not None and float(smoking) > 0) else 0.0
        diabetes = 1.0 if (diabetes is not None and float(diabetes) > 0) else 0.0
        on_treatment = bool(on_treatment)
    except Exception:
        return float("nan")

    if age <= 0 or total_cholesterol <= 0 or hdl_cholesterol <= 0 or systolic_bp <= 0:
        return float("nan")

    ln_age = math.log(age)
    ln_tc = math.log(total_cholesterol)
    ln_hdl = math.log(hdl_cholesterol)
    ln_sbp = math.log(systolic_bp)

    if sex == 1:  # Male coefficients
        beta_age = 3.06117
        beta_tc = 1.12370
        beta_hdl = -0.93263
        beta_sbp_treated = 1.99881
        beta_sbp_untreated = 1.93303
        beta_smoking = 0.65451
        beta_diabetes = 0.57367
        s0 = 0.88936
        mean_xb = 23.9802
        beta_sbp = beta_sbp_treated if on_treatment else beta_sbp_untreated
    else:  # Female coefficients
        beta_age = 2.32888
        beta_tc = 1.20904
        beta_hdl = -0.70833
        beta_sbp_treated = 2.82263
        beta_sbp_untreated = 2.76157
        beta_smoking = 0.52873
        beta_diabetes = 0.69154
        s0 = 0.95012
        mean_xb = 26.1931
        beta_sbp = beta_sbp_treated if on_treatment else beta_sbp_untreated

    xb = (
        beta_age * ln_age
        + beta_tc * ln_tc
        + beta_hdl * ln_hdl
        + beta_sbp * ln_sbp
        + beta_smoking * smoking
        + beta_diabetes * diabetes
    )

    try:
        risk = 1.0 - s0 ** math.exp(xb - mean_xb)
    except OverflowError:
        risk = 1.0

    return float(np.clip(risk, 0.0, 1.0))


def compute_guideline_risk_for_df(df: pd.DataFrame) -> pd.Series:
    """Compute guideline risk for each row in a DataFrame.

    Args:
        df: DataFrame with required columns.

    Returns:
        Series of 10-year risk values.
    """
    def row_to_risk(row: pd.Series) -> float:
        return compute_guideline_risk(
            age=row.get("age"),
            sex=row.get("sex"),
            total_cholesterol=row.get("total_cholesterol"),
            hdl_cholesterol=row.get("hdl_cholesterol", None),
            systolic_bp=row.get("systolic_bp"),
            on_treatment=bool(row.get("bp_meds", 0) or row.get("hypertension", 0)),
            smoking=row.get("smoking"),
            diabetes=row.get("diabetes"),
        )

    return df.apply(row_to_risk, axis=1).astype(float)


# ============================================================================
# TARGET CONVERSION
# ============================================================================

def to_three_class(
    risk: np.ndarray,
    low_threshold: float = 0.10,
    high_threshold: float = 0.25,
) -> np.ndarray:
    """Convert continuous risk to 3-class labels.

    Args:
        risk: Array of continuous risk values (0–1).
        low_threshold: Upper bound for LOW class.
        high_threshold: Lower bound for HIGH class.

    Returns:
        Array of class labels: 0=LOW, 1=MODERATE, 2=HIGH.
    """
    risk = np.asarray(risk, dtype=float)
    labels = np.zeros_like(risk, dtype=int)
    labels[(risk >= low_threshold) & (risk < high_threshold)] = 1
    labels[risk >= high_threshold] = 2
    return labels


def to_binary(y_three_class: np.ndarray) -> np.ndarray:
    """Convert 3-class labels to binary: {0→0, 1→1, 2→1}.

    Args:
        y_three_class: Array of 3-class labels (0, 1, 2).

    Returns:
        Array of binary labels (0, 1).
    """
    y = np.asarray(y_three_class, dtype=int)
    return np.where(y >= 1, 1, 0)


def to_binary_from_continuous(
    risk: np.ndarray,
    threshold: float = 0.10,
) -> np.ndarray:
    """Convert continuous risk to binary labels.

    Args:
        risk: Array of continuous risk values (0–1).
        threshold: Threshold for positive class.

    Returns:
        Array of binary labels (0, 1).
    """
    risk = np.asarray(risk, dtype=float)
    return (risk >= threshold).astype(int)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def add_targets_to_df(
    df: pd.DataFrame,
    low_threshold: float = 0.10,
    high_threshold: float = 0.25,
) -> pd.DataFrame:
    """Add guideline risk and derived targets to DataFrame.

    Args:
        df: Input DataFrame.
        low_threshold: Threshold for LOW/MODERATE boundary.
        high_threshold: Threshold for MODERATE/HIGH boundary.

    Returns:
        DataFrame with added columns:
        - guideline_risk_10yr
        - risk_class_3 (0=LOW, 1=MODERATE, 2=HIGH)
        - risk_class_binary (0 or 1)
    """
    df = df.copy()

    df["guideline_risk_10yr"] = compute_guideline_risk_for_df(df)
    df["risk_class_3"] = to_three_class(
        df["guideline_risk_10yr"].values,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    df["risk_class_binary"] = to_binary(df["risk_class_3"].values)

    return df
