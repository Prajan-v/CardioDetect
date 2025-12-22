"""
SHAP-based Explanations for CardioDetect Risk Predictions

This module provides interpretability for individual cardiovascular risk predictions
using SHAP (SHapley Additive exPlanations).

Usage:
    from milestone_2.src.explanations import RiskExplainer
    
    explainer = RiskExplainer(model, feature_names)
    explanation = explainer.explain(X_sample)
    print(explanation.text_summary())
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Optional SHAP import - graceful fallback if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


@dataclass
class FeatureContribution:
    """Represents a single feature's contribution to the prediction."""
    feature_name: str
    feature_value: Any
    shap_value: float
    direction: str  # 'increases' or 'decreases'
    
    @property
    def impact_percent(self) -> float:
        """Return impact as percentage points."""
        return abs(self.shap_value) * 100


@dataclass  
class PredictionExplanation:
    """Complete explanation for a single prediction."""
    risk_score: float
    risk_category: str
    base_value: float
    contributions: List[FeatureContribution]
    confidence: float
    
    def top_factors(self, n: int = 5) -> List[FeatureContribution]:
        """Return top N contributing factors by absolute impact."""
        sorted_contribs = sorted(
            self.contributions, 
            key=lambda x: abs(x.shap_value), 
            reverse=True
        )
        return sorted_contribs[:n]
    
    def text_summary(self, n_factors: int = 3) -> str:
        """Generate human-readable text explanation."""
        category_emoji = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴"}
        emoji = category_emoji.get(self.risk_category, "⚪")
        
        summary = f"{emoji} Risk is {self.risk_category} ({self.risk_score:.1%})"
        
        top = self.top_factors(n_factors)
        if top:
            increasing = [f for f in top if f.direction == "increases"]
            decreasing = [f for f in top if f.direction == "decreases"]
            
            if increasing:
                inc_text = ", ".join([
                    f"{f.feature_name} ({f.feature_value})" 
                    for f in increasing[:2]
                ])
                summary += f" mainly because {inc_text} increase risk"
            
            if decreasing:
                dec_text = ", ".join([
                    f"{f.feature_name}" 
                    for f in decreasing[:1]
                ])
                summary += f", while {dec_text} is protective"
        
        return summary + "."
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "risk_score": self.risk_score,
            "risk_category": self.risk_category,
            "base_value": self.base_value,
            "confidence": self.confidence,
            "top_factors": [
                {
                    "feature": c.feature_name,
                    "value": c.feature_value,
                    "impact": c.shap_value,
                    "direction": c.direction
                }
                for c in self.top_factors(5)
            ]
        }


class RiskExplainer:
    """
    SHAP-based explainer for CardioDetect risk predictions.
    
    Provides per-patient feature importance and generates
    human-readable explanations for risk predictions.
    """
    
    # Key features for Framingham-style risk
    CRITICAL_FEATURES = [
        'age', 'systolic_bp', 'total_cholesterol', 'hdl_cholesterol',
        'smoking', 'diabetes', 'sex', 'bp_treatment'
    ]
    
    # Human-readable feature names
    FEATURE_DISPLAY_NAMES = {
        'age': 'Age',
        'systolic_bp': 'Systolic BP',
        'diastolic_bp': 'Diastolic BP',
        'total_cholesterol': 'Total Cholesterol',
        'hdl_cholesterol': 'HDL Cholesterol',
        'ldl_cholesterol': 'LDL Cholesterol',
        'smoking': 'Smoking Status',
        'diabetes': 'Diabetes',
        'sex': 'Sex',
        'bmi': 'BMI',
        'bp_treatment': 'BP Treatment',
        'pulse_pressure': 'Pulse Pressure',
        'map': 'Mean Arterial Pressure',
        'cholesterol_ratio': 'Cholesterol Ratio',
    }
    
    def __init__(
        self, 
        model, 
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        n_background_samples: int = 100
    ):
        """
        Initialize the explainer.
        
        Args:
            model: Trained sklearn model (or pipeline)
            feature_names: List of feature names matching model input
            background_data: Background dataset for SHAP (optional)
            n_background_samples: Number of background samples to use
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
        if not SHAP_AVAILABLE:
            self.explainer = None
            self._use_fallback = True
            return
        
        self._use_fallback = False
        
        # Create SHAP explainer based on model type
        model_type = type(model).__name__
        
        if hasattr(model, 'estimators_') or 'Forest' in model_type or 'Gradient' in model_type:
            # Tree-based model - use TreeExplainer (fast)
            self.explainer = shap.TreeExplainer(model)
        else:
            # Other models - use KernelExplainer (slower but universal)
            if background_data is not None:
                # Subsample background data
                if len(background_data) > n_background_samples:
                    idx = np.random.choice(
                        len(background_data), 
                        n_background_samples, 
                        replace=False
                    )
                    background_data = background_data[idx]
                self.explainer = shap.KernelExplainer(
                    model.predict, 
                    background_data
                )
            else:
                self.explainer = None
                self._use_fallback = True
    
    def explain(
        self, 
        X: np.ndarray, 
        threshold_low: float = 0.10,
        threshold_high: float = 0.25
    ) -> PredictionExplanation:
        """
        Generate explanation for a single prediction.
        
        Args:
            X: Feature vector (1D or 2D with single row)
            threshold_low: LOW/MODERATE threshold (default 10%)
            threshold_high: MODERATE/HIGH threshold (default 25%)
            
        Returns:
            PredictionExplanation object with full explanation
        """
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get prediction
        risk_score = float(self.model.predict(X)[0])
        
        # Categorize
        if risk_score < threshold_low:
            category = "LOW"
        elif risk_score < threshold_high:
            category = "MODERATE"
        else:
            category = "HIGH"
        
        # Get SHAP values
        if self._use_fallback or self.explainer is None:
            # Fallback: use feature importance if available
            contributions = self._fallback_explanation(X, risk_score)
            base_value = risk_score / 2  # Approximate
        else:
            shap_values = self.explainer.shap_values(X)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if shap_values.ndim > 1:
                shap_values = shap_values[0]
            
            base_value = float(self.explainer.expected_value)
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            
            contributions = self._create_contributions(X[0], shap_values)
        
        # Calculate confidence based on feature completeness
        confidence = self._calculate_confidence(X[0])
        
        return PredictionExplanation(
            risk_score=risk_score,
            risk_category=category,
            base_value=base_value,
            contributions=contributions,
            confidence=confidence
        )
    
    def _create_contributions(
        self, 
        x: np.ndarray, 
        shap_values: np.ndarray
    ) -> List[FeatureContribution]:
        """Create FeatureContribution objects from SHAP values."""
        contributions = []
        
        for i, (name, value, shap_val) in enumerate(
            zip(self.feature_names, x, shap_values)
        ):
            display_name = self.FEATURE_DISPLAY_NAMES.get(name, name)
            direction = "increases" if shap_val > 0 else "decreases"
            
            contributions.append(FeatureContribution(
                feature_name=display_name,
                feature_value=self._format_value(name, value),
                shap_value=float(shap_val),
                direction=direction
            ))
        
        return contributions
    
    def _fallback_explanation(
        self, 
        X: np.ndarray, 
        risk_score: float
    ) -> List[FeatureContribution]:
        """
        Fallback explanation when SHAP is not available.
        Uses feature importance if available, otherwise returns empty.
        """
        contributions = []
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            for i, (name, imp) in enumerate(zip(self.feature_names, importances)):
                display_name = self.FEATURE_DISPLAY_NAMES.get(name, name)
                value = X[0, i] if X.ndim > 1 else X[i]
                
                # Approximate direction based on feature value vs median
                direction = "increases" if imp > np.median(importances) else "decreases"
                
                contributions.append(FeatureContribution(
                    feature_name=display_name,
                    feature_value=self._format_value(name, value),
                    shap_value=float(imp * risk_score),  # Approximate
                    direction=direction
                ))
        
        return contributions
    
    def _format_value(self, feature_name: str, value: Any) -> str:
        """Format feature value for display."""
        if 'smoking' in feature_name.lower() or 'diabetes' in feature_name.lower():
            return "Yes" if value == 1 else "No"
        elif 'sex' in feature_name.lower():
            return "Male" if value == 1 else "Female"
        elif isinstance(value, float):
            if abs(value) < 1:
                return f"{value:.3f}"
            else:
                return f"{value:.1f}"
        return str(value)
    
    def _calculate_confidence(self, x: np.ndarray) -> float:
        """
        Calculate confidence score based on feature completeness.
        Lower confidence if critical features appear to be imputed.
        """
        confidence = 1.0
        
        # Check for potential imputed values (exact median values are suspicious)
        for i, name in enumerate(self.feature_names):
            if any(crit in name.lower() for crit in ['age', 'bp', 'cholesterol']):
                # These are critical - if they look like round numbers, lower confidence
                if x[i] == 0 or (isinstance(x[i], float) and x[i] == int(x[i])):
                    confidence -= 0.05
        
        return max(0.5, min(1.0, confidence))
    
    def explain_batch(
        self, 
        X: np.ndarray,
        threshold_low: float = 0.10,
        threshold_high: float = 0.25
    ) -> List[PredictionExplanation]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            threshold_low: LOW/MODERATE threshold
            threshold_high: MODERATE/HIGH threshold
            
        Returns:
            List of PredictionExplanation objects
        """
        return [
            self.explain(X[i:i+1], threshold_low, threshold_high) 
            for i in range(len(X))
        ]


def generate_patient_report(
    explanation: PredictionExplanation,
    patient_id: Optional[str] = None
) -> str:
    """
    Generate a formatted patient risk report.
    
    Args:
        explanation: PredictionExplanation from RiskExplainer
        patient_id: Optional patient identifier
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CARDIOVASCULAR RISK ASSESSMENT REPORT")
    lines.append("=" * 60)
    
    if patient_id:
        lines.append(f"Patient ID: {patient_id}")
    
    lines.append("")
    lines.append(f"10-Year CVD Risk: {explanation.risk_score:.1%}")
    lines.append(f"Risk Category: {explanation.risk_category}")
    lines.append(f"Assessment Confidence: {explanation.confidence:.0%}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("TOP CONTRIBUTING FACTORS")
    lines.append("-" * 60)
    
    for i, factor in enumerate(explanation.top_factors(5), 1):
        arrow = "↑" if factor.direction == "increases" else "↓"
        impact = f"+{factor.impact_percent:.1f}%" if factor.direction == "increases" else f"-{factor.impact_percent:.1f}%"
        lines.append(f"{i}. {factor.feature_name}: {factor.feature_value}")
        lines.append(f"   Impact: {impact} {arrow}")
    
    lines.append("")
    lines.append("-" * 60)
    lines.append("SUMMARY")
    lines.append("-" * 60)
    lines.append(explanation.text_summary())
    lines.append("")
    lines.append("=" * 60)
    
    # Add clinical recommendations based on category
    lines.append("RECOMMENDED ACTIONS")
    lines.append("=" * 60)
    
    if explanation.risk_category == "LOW":
        lines.append("• Continue current healthy lifestyle")
        lines.append("• Routine checkups as recommended")
        lines.append("• Maintain healthy diet and exercise")
    elif explanation.risk_category == "MODERATE":
        lines.append("• Review modifiable risk factors with physician")
        lines.append("• Consider lifestyle interventions")
        lines.append("• More frequent monitoring recommended")
        lines.append("• Discuss preventive therapies if appropriate")
    else:  # HIGH
        lines.append("• Urgent consultation with healthcare provider")
        lines.append("• Comprehensive cardiovascular evaluation")
        lines.append("• Aggressive risk factor management")
        lines.append("• Consider pharmacological intervention")
    
    lines.append("")
    lines.append("Note: This is an educational tool. Consult a healthcare")
    lines.append("provider for medical decisions.")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# Convenience function for quick explanations
def explain_prediction(
    model,
    X: np.ndarray,
    feature_names: List[str],
    verbose: bool = True
) -> PredictionExplanation:
    """
    Quick function to explain a single prediction.
    
    Args:
        model: Trained model
        X: Feature vector
        feature_names: Feature names
        verbose: If True, print the explanation
        
    Returns:
        PredictionExplanation object
    """
    explainer = RiskExplainer(model, feature_names)
    explanation = explainer.explain(X)
    
    if verbose:
        print(generate_patient_report(explanation))
    
    return explanation
