"""Guideline-based 10-year cardiovascular risk utilities.

This module implements a Framingham-like general cardiovascular disease
10-year risk function based on D'Agostino et al. (2008), using published
sex-specific coefficients for men and women.

The function returns an estimated 10-year risk as a float in [0, 1]. It is
intended for educational / model-training use, not as a clinical calculator.
"""

from __future__ import annotations

import math
from typing import Optional


def framingham_10yr_risk(
    *,
    age: Optional[float],
    sex: Optional[int],  # 0=female, 1=male
    total_cholesterol: Optional[float],
    hdl_cholesterol: Optional[float],
    systolic_bp: Optional[float],
    on_treatment: Optional[bool] = False,
    smoking: Optional[float] = 0.0,
    diabetes: Optional[float] = 0.0,
) -> float:
    """Estimate 10-year general CVD risk using Framingham-like coefficients.

    Args:
        age: Age in years (30–74 in original study).
        sex: 0 for female, 1 for male.
        total_cholesterol: Total cholesterol (mg/dL).
        hdl_cholesterol: HDL cholesterol (mg/dL). If missing, a default of
            50 mg/dL is used.
        systolic_bp: Systolic blood pressure (mmHg).
        on_treatment: Whether the patient is on blood pressure treatment.
        smoking: Indicator (0/1) for current smoking.
        diabetes: Indicator (0/1) for diabetes.

    Returns:
        Estimated 10-year CVD risk as a float between 0 and 1.

    Notes:
        This implementation follows the structure of the D'Agostino 2008
        "general CVD" risk profile. Coefficients and baseline survival are
        approximate and intended for educational use within CardioDetect.
    """

    # Basic validation
    if age is None or systolic_bp is None or total_cholesterol is None:
        return float("nan")

    if hdl_cholesterol is None:
        hdl_cholesterol = 50.0  # reasonable default

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

    # Guard against non-positive values for log terms
    if age <= 0 or total_cholesterol <= 0 or hdl_cholesterol <= 0 or systolic_bp <= 0:
        return float("nan")

    ln_age = math.log(age)
    ln_tc = math.log(total_cholesterol)
    ln_hdl = math.log(hdl_cholesterol)
    ln_sbp = math.log(systolic_bp)

    if sex == 1:  # Male coefficients
        # Coefficients from Framingham general CVD profile (approximate)
        beta_age = 3.06117
        beta_tc = 1.12370
        beta_hdl = -0.93263
        beta_sbp_treated = 1.99881
        beta_sbp_untreated = 1.93303
        beta_smoking = 0.65451
        beta_diabetes = 0.57367

        # Baseline survival and mean log-risk from reference population
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

    # Linear predictor
    xb = (
        beta_age * ln_age
        + beta_tc * ln_tc
        + beta_hdl * ln_hdl
        + beta_sbp * ln_sbp
        + beta_smoking * smoking
        + beta_diabetes * diabetes
    )

    # 10-year risk
    try:
        risk = 1.0 - s0 ** math.exp(xb - mean_xb)
    except OverflowError:
        risk = 1.0

    # Clamp to [0, 1]
    if risk < 0.0:
        risk = 0.0
    if risk > 1.0:
        risk = 1.0

    return float(risk)
