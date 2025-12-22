"""Production-Grade Risk Prediction Model.

This module provides the integration for the 3-class Risk Classification Model.
It handles feature engineering, prediction, and result formatting.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_classifier.pkl"
META_PATH = MODELS_DIR / "final_classifier_meta.json"


@dataclass
class RiskPrediction:
    """Complete risk prediction result."""
    risk_level: str  # LOW, MODERATE, HIGH
    risk_probability: float  # 0-100 (Probability of the predicted class)
    confidence: float  # 0-100
    
    # Class probabilities
    probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Explainability
    top_risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    
    # Flags
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    is_ood: bool = False


class ProductionModel:
    """Production-grade cardiovascular risk prediction model."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        meta_path: Path = META_PATH,
        enable_shap: bool = False, # SHAP temporarily disabled for new pipeline
        verbose: bool = False,
    ):
        """Initialize the production model."""
        self.verbose = verbose
        self._load_model(model_path, meta_path)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[ProductionModel] {msg}")

    def _load_model(self, model_path: Path, meta_path: Path) -> None:
        """Load the model and metadata."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._log(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
            self.feature_names = self.meta.get("feature_names", [])
            self._log(f"Loaded metadata (Expects {len(self.feature_names)} features)")
        else:
            self.meta = {}
            self.feature_names = []
            self._log("Metadata not found, proceeding without validation")

    def build_features(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """Build engineered features from patient data."""
        
        # 1. Defaults and Extraction
        defaults = {
            "sex": 1, "age": 50, "smoking": 0, "bp_meds": 0,
            "hypertension": 0, "diabetes": 0, "total_cholesterol": 200,
            "systolic_bp": 120, "diastolic_bp": 80, "bmi": 25,
            "heart_rate": 75, "fasting_glucose": 100
        }
        
        # Helper to safely get float
        def get_val(key, alt_key=None):
            val = patient_data.get(key)
            if val is None and alt_key:
                val = patient_data.get(alt_key)
            if val is not None:
                try: return float(val)
                except: pass
            return defaults.get(key, 0.0)

        data = {
            "sex": get_val("sex"),
            "age": get_val("age"),
            "smoking": 1 if str(patient_data.get("smoking", "")).lower() in ("1", "yes", "true") else 0,
            "bp_meds": 1 if str(patient_data.get("bp_meds", "")).lower() in ("1", "yes", "true") else 0,
            "diabetes": 1 if str(patient_data.get("diabetes", "")).lower() in ("1", "yes", "true") else 0,
            "total_cholesterol": get_val("total_cholesterol", "cholesterol"),
            "systolic_bp": get_val("systolic_bp"),
            "diastolic_bp": get_val("diastolic_bp"),
            "bmi": get_val("bmi"),
            "heart_rate": get_val("heart_rate"),
            "fasting_glucose": get_val("fasting_glucose", "glucose"),
        }
        
        # Hypertension derivation
        if data["systolic_bp"] >= 140 or data["diastolic_bp"] >= 90:
            data["hypertension"] = 1
        elif str(patient_data.get("hypertension", "")).lower() in ("1", "yes", "true"):
             data["hypertension"] = 1
        else:
             data["hypertension"] = 0

        # Create DataFrame
        df = pd.DataFrame([data])
        
        # 2. Feature Engineering (Matches Training Logic)
        
        # Derived
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        df["mean_arterial_pressure"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3)
        
        # Flags
        df["hypertension_flag"] = (df["systolic_bp"] >= 140).astype(int)
        df["high_cholesterol_flag"] = (df["total_cholesterol"] >= 240).astype(int)
        df["high_glucose_flag"] = (df["fasting_glucose"] >= 126).astype(int)
        df["obesity_flag"] = (df["bmi"] >= 30).astype(int)
        
        # Metabolic Score
        df["metabolic_syndrome_score"] = (
            df["hypertension_flag"] + 
            df["high_cholesterol_flag"] + 
            df["high_glucose_flag"] + 
            df["obesity_flag"] + 
            df["diabetes"]
        )
        
        # Age Groups
        age = data["age"]
        for ag in ["<40", "40-49", "50-59", "60-69", "70+"]:
            df[f"age_group_{ag}"] = 0
            
        if age < 40: df["age_group_<40"] = 1
        elif age < 50: df["age_group_40-49"] = 1
        elif age < 60: df["age_group_50-59"] = 1
        elif age < 70: df["age_group_60-69"] = 1
        else: df["age_group_70+"] = 1
        
        # BMI Categories
        bmi = data["bmi"]
        for cat in ["Underweight", "Normal", "Overweight", "Obese"]:
            df[f"bmi_cat_{cat}"] = 0
            
        if bmi < 18.5: df["bmi_cat_Underweight"] = 1
        elif bmi < 25: df["bmi_cat_Normal"] = 1
        elif bmi < 30: df["bmi_cat_Overweight"] = 1
        else: df["bmi_cat_Obese"] = 1
        
        # Log Transforms
        df["log_total_cholesterol"] = np.log1p(df["total_cholesterol"])
        df["log_fasting_glucose"] = np.log1p(df["fasting_glucose"])
        df["log_bmi"] = np.log1p(df["bmi"])
        
        # Interactions
        df["age_sbp_interaction"] = df["age"] * df["systolic_bp"]
        df["bmi_glucose_interaction"] = df["bmi"] * df["fasting_glucose"]
        df["age_smoking_interaction"] = df["age"] * df["smoking"]
        
        # 3. Align Columns with Model
        if self.feature_names:
            # Add missing columns with 0
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            # Reorder and select
            df = df[self.feature_names]
            
        return df

    def _generate_explanation(self, risk_level: str, patient_data: Dict[str, Any]) -> Tuple[str, List[Dict]]:
        """Generate rule-based explanation."""
        factors = []
        reasons = []

        # Age
        age = float(patient_data.get("age", 0))
        if age >= 65:
            factors.append({"feature": "Age", "direction": "increases", "importance": "high"})
            reasons.append(f"Patient age ({int(age)}) increases risk significantly")
        
        # BP
        sbp = float(patient_data.get("systolic_bp", 0))
        if sbp >= 140:
            factors.append({"feature": "Systolic BP", "direction": "increases", "importance": "high"})
            reasons.append(f"High blood pressure ({int(sbp)} mmHg)")
            
        # Cholesterol
        chol = float(patient_data.get("total_cholesterol", 0))
        if chol >= 240:
             factors.append({"feature": "Cholesterol", "direction": "increases", "importance": "medium"})
             reasons.append(f"High cholesterol ({int(chol)} mg/dL)")

        # Smoking
        smk = patient_data.get("smoking", 0)
        if str(smk).lower() in ("1", "yes", "true"):
            factors.append({"feature": "Smoking", "direction": "increases", "importance": "high"})
            reasons.append("Smoking is a major risk factor")

        if risk_level == "HIGH":
            text = "High cardiovascular risk detected. " + " ".join(reasons)
        elif risk_level == "MODERATE":
            text = "Moderate cardiovascular risk. " + " ".join(reasons)
        else:
            text = "Low cardiovascular risk. Maintain healthy lifestyle."
            
        return text, factors

    def _apply_clinical_override(
        self, 
        risk_level: str, 
        patient_data: Dict[str, Any],
        probabilities: np.ndarray
    ) -> Tuple[str, Optional[str]]:
        """
        Apply clinical safety rules to override ML predictions for edge cases.
        
        The ML model may miss high-risk young patients due to training data limitations
        (Framingham data has few CHD events in young patients). These rules provide
        a clinical safety net.
        
        Returns:
            Tuple of (final_risk_level, override_reason or None)
        """
        # Only override LOW predictions - never downgrade
        if risk_level != "LOW":
            return risk_level, None
        
        age = float(patient_data.get("age", 50))
        systolic_bp = float(patient_data.get("systolic_bp", 120))
        fasting_glucose = float(patient_data.get("fasting_glucose", 100))
        diabetes = patient_data.get("diabetes", 0)
        
        # Calculate metabolic score
        high_bp = 1 if systolic_bp >= 140 else 0
        high_chol = 1 if float(patient_data.get("total_cholesterol", 200)) >= 240 else 0
        high_glucose = 1 if fasting_glucose >= 126 else 0
        obese = 1 if float(patient_data.get("bmi", 25)) >= 30 else 0
        smoking = 1 if str(patient_data.get("smoking", 0)).lower() in ("1", "yes", "true") else 0
        diabetes_flag = 1 if str(diabetes).lower() in ("1", "yes", "true") else 0
        
        metabolic_score = high_bp + high_chol + high_glucose + obese + smoking + diabetes_flag
        
        # RULE 3: Extreme values - highest priority (medical emergency)
        if systolic_bp >= 180:
            return "MODERATE", "Hypertensive crisis (BP >= 180)"
        if fasting_glucose >= 200:
            return "MODERATE", "Severe hyperglycemia (glucose >= 200)"
        
        # RULE 2: Diabetes override - 36.7% CHD rate in diabetics
        if diabetes_flag == 1:
            return "MODERATE", "Diabetes is a major cardiovascular risk factor"
        
        # RULE 1: Young high metabolic risk - age < 50 with 3+ risk factors
        if age < 50 and metabolic_score >= 3:
            return "MODERATE", f"Multiple risk factors ({metabolic_score}) at young age"
        
        # No override needed
        return risk_level, None

    def predict(self, patient_data: Dict[str, Any]) -> RiskPrediction:
        """Run prediction."""
        try:
            # Build features
            df = self.build_features(patient_data)
            
            # Predict
            pred_idx = self.model.predict(df)[0]
            probabilities = self.model.predict_proba(df)[0]
            
            labels = ["LOW", "MODERATE", "HIGH"]
            risk_level = labels[pred_idx]
            confidence = probabilities[pred_idx] * 100
            
            # Apply clinical override rules for edge cases
            risk_level, override_reason = self._apply_clinical_override(
                risk_level, patient_data, probabilities
            )
            
            # Explanation
            expl_text, top_factors = self._generate_explanation(risk_level, patient_data)
            
            # Add override reason to explanation if applicable
            if override_reason:
                expl_text = f"{expl_text} [Clinical override: {override_reason}]"
            
            # Check for review
            needs_review = False
            reasons = []
            
            if confidence < 80:
                needs_review = True
                reasons.append(f"Model validation confidence is {confidence:.1f}%")
                
            if risk_level == "HIGH" or risk_level == "MODERATE":
                needs_review = True # Always review non-low cases
                reasons.append(f"Clinical review recommended for {risk_level} risk")

            return RiskPrediction(
                risk_level=risk_level,
                risk_probability=probabilities[2] * 100, # Probability of High Risk
                confidence=confidence,
                probabilities={
                    "LOW": probabilities[0],
                    "MODERATE": probabilities[1],
                    "HIGH": probabilities[2]
                },
                explanation=expl_text,
                top_risk_factors=top_factors,
                needs_review=needs_review,
                review_reasons=reasons
            )
            
        except Exception as e:
            self._log(f"Prediction failed: {e}")
            # Fallback
            return RiskPrediction(
                risk_level="ERROR",
                risk_probability=0.0,
                confidence=0.0,
                explanation=f"Error: {str(e)}",
                needs_review=True,
                review_reasons=["Model execution failed"]
            )

def predict_risk(patient_data: Dict[str, Any], verbose: bool = False) -> RiskPrediction:
    """Convenience function."""
    model = ProductionModel(verbose=verbose)
    return model.predict(patient_data)

if __name__ == "__main__":
    # Test
    sample = {"age": 72, "systolic_bp": 170, "sex": 1, "smoking": 1, "diabetes": 1, "total_cholesterol": 260}
    print(predict_risk(sample, verbose=True))
