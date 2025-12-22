"""
CardioDetect V3 End-to-End Pipeline
Document → OCR → Feature Engineering → Risk Prediction → JSON Output
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.universal_medical_ocr import UniversalMedicalOCREngine, OCRResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "split"


class CardioDetectV3:
    """
    End-to-end pipeline for cardiovascular risk prediction.
    
    Workflow:
    1. OCR: Extract structured fields from medical document
    2. Feature Engineering: Build 34-feature vector
    3. Risk Prediction: Run model and categorize risk
    4. Output: Comprehensive JSON with risk, confidence, audit trail
    """
    
    # Risk thresholds (as percentages) for 10-year risk
    THRESHOLD_LOW = 10.0
    THRESHOLD_HIGH = 25.0
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        ocr_verbose: bool = False,
        verbose: bool = False
    ):
        """
        Initialize pipeline.
        
        Args:
            model_path: Path to model artifact (default: mlp_v2.pkl or risk_regressor_v1.pkl)
            ocr_verbose: Enable OCR debug output
            verbose: Enable pipeline debug output
        """
        self.verbose = verbose
        
        # Initialize OCR engine
        self.ocr_engine = UniversalMedicalOCREngine(verbose=ocr_verbose)
        
        # Load model
        if model_path is None:
            # Default: guideline-based risk regressor (v2) from final models directory
            model_path = MODELS_DIR / "final" / "risk_regressor_v2.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._log(f"Loading model from {model_path}")
        artifact = joblib.load(model_path)

        # Expect dict artifact with at least a 'model' key
        model_obj: Any
        feature_names = None
        threshold_low = None
        threshold_high = None

        if isinstance(artifact, dict) and "model" in artifact:
            model_obj = artifact["model"]
            feature_names = artifact.get("feature_names")
            threshold_low = artifact.get("threshold_low")
            threshold_high = artifact.get("threshold_high")
        else:
            model_obj = artifact
            meta_path = model_path.with_name(model_path.stem + "_meta.json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                feature_names = meta.get("feature_names")
                threshold_low = meta.get("threshold_low")
                threshold_high = meta.get("threshold_high")

        self.model = model_obj
        self.model_version = model_path.stem

        # For guideline regressors, we store feature_names and thresholds in the artifact
        self.feature_columns = feature_names
        if threshold_low is not None:
            self.THRESHOLD_LOW = float(threshold_low)
        if threshold_high is not None:
            self.THRESHOLD_HIGH = float(threshold_high)

        # No external scaler is used for the guideline regressor
        self.scaler = None
        self.scaler_feature_names = None

        # Load baseline statistics for feature imputation
        self._load_baseline_stats()
        
        self._log("Pipeline initialized")
    
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[CardioDetectV3] {msg}")
    
    def _load_baseline_stats(self) -> None:
        """Load training data statistics for missing value imputation."""
        train_path = DATA_DIR / "train.csv"
        
        if train_path.exists():
            # Load actual training data when available to compute realistic medians
            train_df = pd.read_csv(train_path)
            drop_cols = ['risk_target', 'data_source']
            X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])

            # If the model artifact provided explicit feature names, align to them
            if self.feature_columns is not None:
                missing_cols = [c for c in self.feature_columns if c not in X_train.columns]
                for col in missing_cols:
                    X_train[col] = 0.0
                X_train = X_train[self.feature_columns]
            else:
                # Fallback: derive feature columns directly
                self.feature_columns = X_train.columns.tolist()

            # Ensure numeric dtypes before computing medians
            X_train = X_train.apply(pd.to_numeric, errors="coerce")

            self.baseline_median = X_train.median()
            self._log(f"Loaded baseline statistics: {len(self.feature_columns)} features")
            return

        # For the guideline regressor, the model artifact already provides the
        # exact feature_names. We only need a consistent index for missing-value
        # imputation; using zeros is sufficient and avoids version-specific
        # numeric conversion issues when reloading training CSVs.
        if getattr(self, "feature_columns", None) is not None:
            self.baseline_median = pd.Series(0.0, index=self.feature_columns)
            self._log(f"Initialized baseline statistics from model feature list: {len(self.feature_columns)} features")
            return

        if not train_path.exists():
            self._log("Warning: Training data not found, using hardcoded defaults")
            # Hardcoded reasonable defaults
            self.baseline_median = pd.Series({
                'age': 50.0,
                'sex': 0.5,
                'systolic_bp': 120.0,
                'diastolic_bp': 80.0,
                'bmi': 25.0,
                'total_cholesterol': 200.0,
                'hdl': 50.0,
                'ldl': 130.0,
                'triglycerides': 150.0,
                'fasting_glucose': 100.0,
                'hemoglobin': 14.0,
                'wbc': 7.0,
                'rbc': 4.5,
                'platelets': 250.0,
                'smoking': 0.0,
                'diabetes': 0.0,
                'heart_rate': 72.0,
            })
            self.feature_columns = list(self.baseline_median.index)
            return
    
    def build_feature_vector(self, ocr_fields: Dict[str, Any]) -> pd.Series:
        """
        Build 34-feature vector from OCR fields.
        Uses median imputation for missing values.
        
        Args:
            ocr_fields: Dictionary of extracted medical fields
            
        Returns:
            Feature vector aligned with model expectations
        """
        # Start with baseline (median values)
        features = self.baseline_median.copy()
        
        # Map OCR fields to feature names
        field_mapping = {
            'age': 'age',
            'sex': 'sex',
            'systolic_bp': 'systolic_bp',
            'diastolic_bp': 'diastolic_bp',
            'bmi': 'bmi',
            'total_cholesterol': 'total_cholesterol',
            'cholesterol': 'total_cholesterol',
            'hdl': 'hdl',
            'ldl': 'ldl',
            'triglycerides': 'triglycerides',
            'fasting_glucose': 'fasting_glucose',
            'glucose': 'fasting_glucose',
            'hemoglobin': 'hemoglobin',
            'wbc': 'wbc',
            'rbc': 'rbc',
            'platelets': 'platelets',
            'platelet': 'platelets',
            'smoking': 'smoking',
            'diabetes': 'diabetes',
            'heart_rate': 'heart_rate',
        }
        
        # Override with OCR values
        for ocr_key, feature_key in field_mapping.items():
            if ocr_key in ocr_fields and ocr_fields[ocr_key] is not None:
                if feature_key in features.index:
                    features[feature_key] = float(ocr_fields[ocr_key])
        
        # Compute engineered features if base features exist
        if 'systolic_bp' in features.index and 'diastolic_bp' in features.index:
            sbp = features['systolic_bp']
            dbp = features['diastolic_bp']
            
            if 'pulse_pressure' in features.index:
                features['pulse_pressure'] = sbp - dbp
            if 'mean_arterial_pressure' in features.index:
                features['mean_arterial_pressure'] = (sbp + 2 * dbp) / 3
        
        return features
    
    def categorize_risk(self, probability: float) -> str:
        """Categorize risk based on predicted 10-year risk probability.
        
        Args:
            probability: Estimated 10-year risk (0–1).
        
        Returns:
            str: Risk category ('LOW', 'MODERATE', 'HIGH').
        """
        risk_percent = probability * 100.0
        
        if risk_percent < self.THRESHOLD_LOW:
            return "LOW"
        elif risk_percent < self.THRESHOLD_HIGH:
            return "MODERATE"
        else:
            return "HIGH"
    
    def generate_recommendation(self, risk_category: str, risk_percent: float) -> str:
        """Generate human-readable recommendation."""
        if risk_category == "LOW":
            return (
                f"Low cardiovascular risk ({risk_percent:.1f}%). "
                "Continue healthy lifestyle practices. Regular check-ups recommended."
            )
        elif risk_category == "MODERATE":
            return (
                f"Moderate cardiovascular risk ({risk_percent:.1f}%). "
                "Consult healthcare provider for risk factor management. "
                "Consider lifestyle modifications and regular monitoring."
            )
        else:  # HIGH
            return (
                f"High cardiovascular risk ({risk_percent:.1f}%). "
                "Immediate medical consultation recommended. "
                "Comprehensive cardiovascular evaluation and intervention may be needed."
            )

    def _compute_risk_reasons(self, features: pd.Series) -> list[Dict[str, Any]]:
        reasons: list[Dict[str, Any]] = []

        def add_reason(feature_key: str, value: float, score: float, text: str) -> None:
            if value is None or pd.isna(value):
                return
            reasons.append(
                {
                    "feature": feature_key,
                    "value": float(value),
                    "direction": "increases" if score > 0 else "decreases",
                    "strength": float(abs(score)),
                    "text": text,
                }
            )

        age = features.get("age") if "age" in features.index else None
        if age is not None and not pd.isna(age):
            if age >= 65:
                score = (age - 65.0) / 10.0 + 1.0
                text = f"Age {age:.0f} years is high and increases cardiovascular risk."
                add_reason("age", age, score, text)
            elif age <= 45:
                score = (45.0 - age) / 10.0
                text = f"Age {age:.0f} years is relatively young and lowers cardiovascular risk."
                add_reason("age", age, -score, text)

        sbp = features.get("systolic_bp") if "systolic_bp" in features.index else None
        if sbp is not None and not pd.isna(sbp):
            if sbp >= 160:
                score = (sbp - 160.0) / 20.0 + 1.0
                text = f"Systolic blood pressure {sbp:.0f} mmHg is severely elevated and increases risk."
                add_reason("systolic_bp", sbp, score, text)
            elif sbp >= 140:
                score = (sbp - 140.0) / 20.0 + 0.5
                text = f"Systolic blood pressure {sbp:.0f} mmHg is elevated and increases risk."
                add_reason("systolic_bp", sbp, score, text)
            elif sbp <= 120:
                score = (120.0 - sbp) / 20.0
                text = f"Systolic blood pressure {sbp:.0f} mmHg is in a healthy range and lowers risk."
                add_reason("systolic_bp", sbp, -score, text)

        tc = None
        for key in ("total_cholesterol", "cholesterol"):
            if key in features.index:
                tc = features.get(key)
                break
        if tc is not None and not pd.isna(tc):
            if tc >= 240:
                score = (tc - 240.0) / 40.0 + 0.8
                text = f"Total cholesterol {tc:.0f} mg/dL is high and increases cardiovascular risk."
                add_reason("total_cholesterol", tc, score, text)
            elif tc >= 200:
                score = (tc - 200.0) / 40.0 + 0.4
                text = f"Total cholesterol {tc:.0f} mg/dL is borderline high and increases risk."
                add_reason("total_cholesterol", tc, score, text)
            elif tc <= 180:
                score = (180.0 - tc) / 40.0
                text = f"Total cholesterol {tc:.0f} mg/dL is in a favorable range and lowers risk."
                add_reason("total_cholesterol", tc, -score, text)

        smoking_val = features.get("smoking") if "smoking" in features.index else None
        if smoking_val is not None and not pd.isna(smoking_val):
            if float(smoking_val) >= 1.0:
                score = 1.5
                text = "Current smoking status strongly increases cardiovascular risk."
                add_reason("smoking", smoking_val, score, text)
            else:
                score = 0.5
                text = "Non-smoking status lowers cardiovascular risk."
                add_reason("smoking", smoking_val, -score, text)

        diabetes_val = features.get("diabetes") if "diabetes" in features.index else None
        if diabetes_val is not None and not pd.isna(diabetes_val):
            if float(diabetes_val) >= 1.0:
                score = 1.2
                text = "Diabetes is present and increases cardiovascular risk."
                add_reason("diabetes", diabetes_val, score, text)
            else:
                score = 0.4
                text = "Absence of diabetes lowers cardiovascular risk."
                add_reason("diabetes", diabetes_val, -score, text)

        reasons_sorted = sorted(reasons, key=lambda r: r["strength"], reverse=True)
        return reasons_sorted[:3]
    
    def run(self, document_path: str | Path) -> Dict[str, Any]:
        """Run complete pipeline on a medical document."""
        document_path = Path(document_path)

        result = {
            "success": False,
            "risk_score": 0.0,
            "risk_category": "UNKNOWN",
            "recommendation": "",
            "ocr_confidence": {"average": 0.0, "per_field": {}},
            "model_confidence": 0.0,
            "fields": {},
            "fields_used": [],
            "audit": {
                "engine": "universal_ocr_v3",
                "file_type": "unknown",
                "timestamp": datetime.now().isoformat(),
                "model_version": self.model_version,
                "document_path": str(document_path.name),
            },
            "errors": [],
            "warnings": [],
        }

        try:
            # Step 1: OCR
            self._log(f"Running OCR on {document_path.name}")
            ocr_result = self.ocr_engine.extract(document_path)

            # Always record OCR metadata, even on failure
            result["audit"]["engine"] = ocr_result.engine
            result["audit"]["file_type"] = ocr_result.file_type
            result["ocr_confidence"]["average"] = ocr_result.avg_ocr_confidence
            result["ocr_confidence"]["per_field"] = ocr_result.field_confidences
            result["fields"] = ocr_result.structured_fields

            if ocr_result.warnings:
                result["warnings"].extend(ocr_result.warnings)

            if not ocr_result.success:
                result["errors"].append("OCR extraction failed")
                result["errors"].extend(ocr_result.errors)
                return result

            self._log(f"OCR extracted {len(ocr_result.structured_fields)} fields")

            # Step 2: Build feature vector
            self._log("Building feature vector")
            features = self.build_feature_vector(ocr_result.structured_fields)
            result["fields_used"] = list(ocr_result.structured_fields.keys())

            # Step 3: Run model (guideline-based regressor)
            self._log("Running risk prediction model")

            # Align features with the model's expected feature order
            if self.feature_columns is not None:
                aligned_features = features.reindex(index=self.feature_columns)
            else:
                aligned_features = features

            # The milestone_2 regressor pipelines expect a pandas DataFrame
            # with named columns (used by ColumnTransformer selectors), not
            # a raw NumPy array. Convert the aligned Series into a 1-row
            # DataFrame preserving column names.
            input_df = aligned_features.to_frame().T

            risk = float(self.model.predict(input_df)[0])
            risk_percent = risk * 100.0

            # Step 4: Categorize and generate recommendation
            risk_category = self.categorize_risk(risk)
            recommendation = self.generate_recommendation(risk_category, risk_percent)

            # -----------------------------------------------------------------
            # SAFETY GUARD: very high clinical risk profile
            # If OCR indicates smoking and diabetes are both present, with
            # age ≥65 and systolic BP ≥160, then true risk is almost
            # certainly high even if the numeric estimate is lower.
            #
            # In that case, force the categorical risk to at least HIGH
            # and add a warning about likely underestimation.
            # -----------------------------------------------------------------
            try:
                fields = ocr_result.structured_fields or {}
                smoker_val = fields.get("smoking")
                diabetes_val = fields.get("diabetes")
                age_val = fields.get("age")
                sbp_val = fields.get("systolic_bp")

                if (
                    smoker_val is not None
                    and float(smoker_val) >= 1.0
                    and diabetes_val is not None
                    and float(diabetes_val) >= 1.0
                    and age_val is not None
                    and float(age_val) >= 65.0
                    and sbp_val is not None
                    and float(sbp_val) >= 160.0
                ):
                    if risk_percent < self.THRESHOLD_HIGH:
                        result["warnings"].append(
                            "Safety rule: age ≥65, systolic BP ≥160, and both smoking and diabetes present. "
                            "True cardiovascular risk is likely higher than the numeric estimate; "
                            "setting risk category to HIGH."
                        )
                        risk_category = "HIGH"
                        recommendation = self.generate_recommendation(risk_category, risk_percent)

                # Additional safety: elderly patients with very low numeric risk
                # or missing essential risk factors should not be labeled
                # confidently LOW without a warning.
                essential_keys = ["age", "systolic_bp", "cholesterol", "total_cholesterol", "glucose", "fasting_glucose", "smoking", "diabetes"]
                missing_essentials = []
                for key in essential_keys:
                    if key in ("cholesterol", "total_cholesterol"):
                        # Treat total_cholesterol/cholesterol as interchangeable
                        if "total_cholesterol" in fields or "cholesterol" in fields:
                            continue
                        missing_essentials.append("total_cholesterol")
                        continue
                    if fields.get(key) is None:
                        missing_essentials.append(key)

                if missing_essentials:
                    result["warnings"].append(
                        "Safety rule: some essential risk factors were not extracted (e.g., "
                        + ", ".join(sorted(set(missing_essentials)))
                        + "). Numeric risk may be underestimated."
                    )

                age_numeric = None
                if age_val is not None:
                    try:
                        age_numeric = float(age_val)
                    except (TypeError, ValueError):
                        age_numeric = None

                if age_numeric is not None and age_numeric >= 75.0:
                    # If the model reports a very low numeric risk for an
                    # elderly patient, upgrade at least to MODERATE to avoid
                    # over-confident LOW classification.
                    if risk_percent < max(5.0, float(self.THRESHOLD_LOW)) and risk_category == "LOW":
                        result["warnings"].append(
                            "Safety rule: age ≥75 with very low numeric risk. "
                            "Upgrading risk category from LOW to MODERATE."
                        )
                        risk_category = "MODERATE"
                        recommendation = self.generate_recommendation(risk_category, risk_percent)
            except Exception:
                # Fail open; do not interrupt pipeline if safety logic errors
                pass

            reasons = self._compute_risk_reasons(aligned_features)

            # Update result
            result["success"] = True
            result["risk_score"] = float(risk)
            result["risk_category"] = risk_category
            result["recommendation"] = recommendation
            result["model_confidence"] = float(risk)  # Same as risk score for now
            result["explanations"] = {
                "top_reasons": [r["text"] for r in reasons],
                "details": reasons,
            }

            self._log(f"Prediction complete: {risk_category} ({risk_percent:.1f}%)")

        except Exception as e:
            result["errors"].append(f"Pipeline error: {str(e)}")
            self._log(f"Error: {e}")

        return result
    
    def run_batch(self, document_paths: list[str | Path]) -> list[Dict[str, Any]]:
        """Run pipeline on multiple documents."""
        results = []
        
        for path in document_paths:
            self._log(f"\nProcessing {Path(path).name}")
            result = self.run(path)
            results.append(result)
        
        return results


def predict_risk(document_path: str | Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Quick prediction function.
    
    Args:
        document_path: Path to medical document
        verbose: Enable debug output
        
    Returns:
        Risk prediction result dictionary
    """
    pipeline = CardioDetectV3(verbose=verbose)
    return pipeline.run(document_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cardiodetect_v3_pipeline.py <document_path>")
        sys.exit(1)
    
    doc_path = sys.argv[1]
    
    print("\n" + "="*80)
    print("CARDIODETECT V3 PIPELINE")
    print("="*80)
    
    result = predict_risk(doc_path, verbose=True)
    
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(json.dumps(result, indent=2))
