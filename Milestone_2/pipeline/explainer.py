"""
SHAP Explainability Module for CardioDetect
============================================
Provides feature importance explanations for predictions
"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAP-based explainability for heart disease predictions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.explainer = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load model and create SHAP explainer"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Create TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def explain_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            features: Dictionary of input features
            
        Returns:
            Dictionary with feature contributions and explanation
        """
        if not self.explainer:
            return self._fallback_explanation(features)
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(df)
            
            # Handle multi-output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Create feature contributions
            contributions = []
            for i, (name, value) in enumerate(features.items()):
                if i < len(shap_values[0]):
                    impact = float(shap_values[0][i])
                    contributions.append({
                        'feature': name,
                        'value': value,
                        'impact': impact,
                        'direction': 'positive' if impact > 0 else 'negative',
                        'importance': abs(impact)
                    })
            
            # Sort by importance
            contributions.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'success': True,
                'contributions': contributions[:10],  # Top 10
                'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.5,
                'explanation_type': 'shap'
            }
            
        except Exception as e:
            return self._fallback_explanation(features)
    
    def _fallback_explanation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fallback when SHAP is unavailable.
        Provides clinical interpretation of risk factors.
        """
        contributions = []
        
        # Age impact
        age = features.get('age', 50)
        if age:
            impact = (age - 45) * 0.02 if age > 45 else 0
            contributions.append({
                'feature': 'Age',
                'value': age,
                'impact': impact,
                'direction': 'positive' if age > 45 else 'neutral',
                'importance': abs(impact),
                'explanation': f"Age {age} {'increases' if age > 45 else 'has minimal effect on'} risk"
            })
        
        # Blood pressure
        sbp = features.get('systolic_bp', 120)
        if sbp:
            impact = (sbp - 120) * 0.015 if sbp > 120 else 0
            stage = 'Normal' if sbp < 120 else 'Elevated' if sbp < 130 else 'Stage 1 HTN' if sbp < 140 else 'Stage 2 HTN'
            contributions.append({
                'feature': 'Blood Pressure',
                'value': f"{sbp} mmHg",
                'impact': impact,
                'direction': 'positive' if sbp > 130 else 'neutral',
                'importance': abs(impact),
                'explanation': f"BP {sbp} mmHg ({stage})"
            })
        
        # Cholesterol
        chol = features.get('total_cholesterol', 200)
        if chol:
            impact = (chol - 200) * 0.01 if chol > 200 else 0
            status = 'Optimal' if chol < 200 else 'Borderline' if chol < 240 else 'High'
            contributions.append({
                'feature': 'Cholesterol',
                'value': f"{chol} mg/dL",
                'impact': impact,
                'direction': 'positive' if chol > 200 else 'neutral',
                'importance': abs(impact),
                'explanation': f"Total cholesterol {status}"
            })
        
        # HDL (protective)
        hdl = features.get('hdl_cholesterol', 50)
        if hdl:
            impact = (40 - hdl) * 0.02 if hdl < 40 else -0.1 if hdl > 60 else 0
            contributions.append({
                'feature': 'HDL Cholesterol',
                'value': f"{hdl} mg/dL",
                'impact': impact,
                'direction': 'negative' if hdl > 60 else 'positive' if hdl < 40 else 'neutral',
                'importance': abs(impact),
                'explanation': f"HDL {'protective' if hdl > 60 else 'low risk' if hdl < 40 else 'normal'}"
            })
        
        # Smoking
        smoking = features.get('smoking', 0)
        if smoking:
            contributions.append({
                'feature': 'Smoking',
                'value': 'Yes',
                'impact': 0.3,
                'direction': 'positive',
                'importance': 0.3,
                'explanation': 'Smoking significantly increases cardiovascular risk'
            })
        
        # Diabetes
        diabetes = features.get('diabetes', 0)
        if diabetes:
            contributions.append({
                'feature': 'Diabetes',
                'value': 'Yes',
                'impact': 0.25,
                'direction': 'positive',
                'importance': 0.25,
                'explanation': 'Diabetes is a major cardiovascular risk factor'
            })
        
        # BMI
        bmi = features.get('bmi', 25)
        if bmi:
            impact = (bmi - 25) * 0.02 if bmi > 25 else 0
            status = 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'
            contributions.append({
                'feature': 'BMI',
                'value': bmi,
                'impact': impact,
                'direction': 'positive' if bmi > 25 else 'neutral',
                'importance': abs(impact),
                'explanation': f"BMI {bmi:.1f} ({status})"
            })
        
        # Sort by importance
        contributions.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'success': True,
            'contributions': contributions,
            'base_value': 0.5,
            'explanation_type': 'clinical_rules',
            'note': 'SHAP model not loaded - using clinical rules'
        }
    
    def get_feature_importance_chart(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get data formatted for a horizontal bar chart visualization.
        """
        explanation = self.explain_prediction(features)
        
        chart_data = []
        for contrib in explanation.get('contributions', []):
            chart_data.append({
                'name': contrib['feature'],
                'value': round(contrib['impact'] * 100, 1),  # Convert to percentage
                'color': '#ef4444' if contrib['direction'] == 'positive' else '#22c55e' if contrib['direction'] == 'negative' else '#64748b'
            })
        
        return {
            'chart_data': chart_data,
            'explanation_type': explanation.get('explanation_type'),
            'title': 'Feature Contributions to Risk'
        }


# Singleton instance
_explainer = None

def get_explainer() -> SHAPExplainer:
    """Get or create SHAP explainer instance"""
    global _explainer
    if _explainer is None:
        _explainer = SHAPExplainer()
        
        # Try to load prediction model
        model_paths = [
            Path(__file__).parent.parent / 'models' / 'Final_models' / 'prediction' / 'prediction_xgb.pkl',
            Path(__file__).parent / 'models' / 'prediction_xgb.pkl',
        ]
        
        for path in model_paths:
            if path.exists():
                _explainer.load_model(str(path))
                break
    
    return _explainer


def explain_prediction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to explain a prediction"""
    return get_explainer().explain_prediction(features)


if __name__ == '__main__':
    # Test
    explainer = SHAPExplainer()
    
    test_features = {
        'age': 55,
        'systolic_bp': 145,
        'total_cholesterol': 245,
        'hdl_cholesterol': 42,
        'smoking': 1,
        'diabetes': 0,
        'bmi': 28.5
    }
    
    result = explainer.explain_prediction(test_features)
    
    print("=== SHAP Explanation ===")
    print(f"Type: {result.get('explanation_type')}")
    print("\nTop Contributions:")
    for c in result.get('contributions', [])[:5]:
        print(f"  {c['feature']}: {c['value']} → {'↑' if c['direction'] == 'positive' else '↓'} {c['importance']:.3f}")
