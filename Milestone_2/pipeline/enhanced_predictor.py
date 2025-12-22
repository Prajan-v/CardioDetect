"""
Enhanced Model Predictions with Calibration and Explainability
===============================================================
Adds:
- Isotonic probability calibration
- SHAP feature explanations
- Data-driven thresholds (Youden index)
- Fairness metrics

This is an OPTIONAL enhancement - if issues arise, the system
falls back to the original clinical assessment.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

# Try to import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠ SHAP not installed. Run: pip install shap")

# Try to import calibration
try:
    from sklearn.calibration import CalibratedClassifierCV
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False


class EnhancedPredictor:
    """
    Enhanced prediction with calibration and explainability.
    
    Falls back to base model if enhancements fail.
    """
    
    def __init__(self, base_model, scaler=None, feature_names: List[str] = None):
        self.base_model = base_model
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.calibrated_model = None
        self.explainer = None
        self.optimal_threshold = 0.5  # Default, will be updated
        
        # Initialize enhancements
        self._setup_shap()
    
    def _setup_shap(self):
        """Initialize SHAP explainer."""
        if not HAS_SHAP or self.base_model is None:
            return
        
        try:
            # Use TreeExplainer for tree-based models (XGBoost, LightGBM, RF)
            if hasattr(self.base_model, 'get_booster'):
                self.explainer = shap.TreeExplainer(self.base_model)
            elif hasattr(self.base_model, 'estimators_'):
                self.explainer = shap.TreeExplainer(self.base_model)
            else:
                # Fallback to KernelExplainer (slower but universal)
                self.explainer = None
        except Exception as e:
            print(f"⚠ SHAP setup failed: {e}")
            self.explainer = None
    
    def calibrate(self, X_val, y_val, method: str = 'isotonic'):
        """
        Calibrate probabilities using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            method: 'isotonic' or 'sigmoid'
        """
        if not HAS_CALIBRATION or self.base_model is None:
            return False
        
        try:
            self.calibrated_model = CalibratedClassifierCV(
                self.base_model, 
                method=method, 
                cv='prefit'
            )
            self.calibrated_model.fit(X_val, y_val)
            print(f"✓ Model calibrated using {method} method")
            return True
        except Exception as e:
            print(f"⚠ Calibration failed: {e}")
            return False
    
    def find_optimal_threshold(self, X_val, y_val):
        """
        Find optimal classification threshold using Youden's J statistic.
        
        J = TPR - FPR = Sensitivity + Specificity - 1
        """
        try:
            from sklearn.metrics import roc_curve
            
            # Get probabilities
            if self.calibrated_model:
                y_proba = self.calibrated_model.predict_proba(X_val)[:, 1]
            else:
                y_proba = self.base_model.predict_proba(X_val)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_val, y_proba)
            
            # Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            self.optimal_threshold = thresholds[optimal_idx]
            
            print(f"✓ Optimal threshold: {self.optimal_threshold:.3f} (Youden J = {j_scores[optimal_idx]:.3f})")
            return self.optimal_threshold
        except Exception as e:
            print(f"⚠ Threshold optimization failed: {e}")
            return 0.5
    
    def predict_with_explanation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction with SHAP explanation.
        
        Returns:
            Dictionary with prediction, probability, and feature contributions
        """
        result = {
            'probability': 0.5,
            'prediction': 'MODERATE',
            'threshold_used': self.optimal_threshold,
            'calibrated': self.calibrated_model is not None,
            'explanations': [],
            'base_value': None,
        }
        
        if self.base_model is None:
            return result
        
        try:
            # Prepare features
            X = self._prepare_features(features)
            
            if X is None:
                return result
            
            # Scale if needed
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Get probability
            if self.calibrated_model is not None:
                prob = self.calibrated_model.predict_proba(X_scaled)[0, 1]
            else:
                prob = self.base_model.predict_proba(X_scaled)[0, 1]
            
            result['probability'] = float(prob)
            
            # Apply optimal threshold
            if prob >= self.optimal_threshold:
                result['prediction'] = 'HIGH'
            elif prob >= self.optimal_threshold * 0.6:
                result['prediction'] = 'MODERATE'
            else:
                result['prediction'] = 'LOW'
            
            # Get SHAP explanations
            if self.explainer is not None:
                try:
                    shap_values = self.explainer.shap_values(X_scaled)
                    
                    # Handle different SHAP output formats
                    if isinstance(shap_values, list):
                        values = shap_values[1][0]  # For binary classification
                    else:
                        values = shap_values[0]
                    
                    # Build explanation list
                    explanations = []
                    for i, (name, val) in enumerate(zip(self.feature_names, values)):
                        feature_value = X[0, i] if hasattr(X, 'shape') else features.get(name)
                        explanations.append({
                            'feature': name,
                            'value': float(feature_value) if feature_value is not None else None,
                            'impact': float(val),
                            'direction': 'increases' if val > 0 else 'decreases'
                        })
                    
                    # Sort by absolute impact
                    explanations.sort(key=lambda x: abs(x['impact']), reverse=True)
                    result['explanations'] = explanations[:5]  # Top 5
                    
                    # Base value (expected value)
                    if hasattr(self.explainer, 'expected_value'):
                        ev = self.explainer.expected_value
                        result['base_value'] = float(ev[1] if isinstance(ev, list) else ev)
                        
                except Exception as e:
                    print(f"⚠ SHAP explanation failed: {e}")
            
            return result
            
        except Exception as e:
            print(f"⚠ Prediction failed: {e}")
            return result
    
    def _prepare_features(self, features: Dict) -> Optional[np.ndarray]:
        """Prepare feature vector from dict."""
        if not self.feature_names:
            return None
        
        values = []
        for name in self.feature_names:
            val = features.get(name)
            if val is None:
                # Use default values for missing features
                defaults = {
                    'age': 55, 'sex': 1, 'systolic_bp': 130, 
                    'cholesterol': 200, 'heart_rate': 75, 'bmi': 25,
                    'smoking': 0, 'diabetes': 0, 'age_sq': 3025,
                    'elderly': 1, 'high_bp': 0
                }
                val = defaults.get(name, 0)
            values.append(val)
        
        return np.array([values])
    
    def fairness_audit(self, X, y, sensitive_feature: str = 'sex') -> Dict[str, Any]:
        """
        Perform fairness audit by sensitive attribute.
        
        Returns metrics broken down by group.
        """
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            
            # Get predictions
            if self.calibrated_model:
                y_pred = (self.calibrated_model.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)
            else:
                y_pred = (self.base_model.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)
            
            # Get sensitive feature index
            if sensitive_feature in self.feature_names:
                sens_idx = self.feature_names.index(sensitive_feature)
                sens_values = X[:, sens_idx]
            else:
                return {'error': f'Feature {sensitive_feature} not found'}
            
            # Calculate metrics by group
            groups = {}
            for group_val in np.unique(sens_values):
                mask = sens_values == group_val
                group_name = 'Male' if group_val == 1 else 'Female'
                
                groups[group_name] = {
                    'count': int(np.sum(mask)),
                    'accuracy': float(accuracy_score(y[mask], y_pred[mask])),
                    'precision': float(precision_score(y[mask], y_pred[mask], zero_division=0)),
                    'recall': float(recall_score(y[mask], y_pred[mask], zero_division=0)),
                    'positive_rate': float(np.mean(y_pred[mask]))
                }
            
            # Calculate disparate impact (ratio of positive rates)
            if len(groups) == 2:
                rates = [g['positive_rate'] for g in groups.values()]
                disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 1.0
            else:
                disparate_impact = None
            
            return {
                'by_group': groups,
                'disparate_impact': disparate_impact,
                'fair': disparate_impact is None or disparate_impact >= 0.8
            }
            
        except Exception as e:
            return {'error': str(e)}


# Convenience function
def create_enhanced_predictor(model_path: str = None) -> Optional[EnhancedPredictor]:
    """Create an enhanced predictor from saved model."""
    try:
        import joblib
        
        if model_path is None:
            # Use default path
            model_path = Path(__file__).parent.parent / 'models' / 'Final_models' / 'prediction' / 'prediction_xgb.pkl'
        
        model_data = joblib.load(model_path)
        
        if hasattr(model_data, 'predict'):
            model = model_data
            scaler = None
            features = []
        else:
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            features = model_data.get('feature_cols', [])
        
        return EnhancedPredictor(model, scaler, features)
        
    except Exception as e:
        print(f"⚠ Could not create enhanced predictor: {e}")
        return None


if __name__ == '__main__':
    print("=" * 60)
    print("ENHANCED PREDICTOR TEST")
    print("=" * 60)
    print(f"SHAP available: {HAS_SHAP}")
    print(f"Calibration available: {HAS_CALIBRATION}")
    
    # Try to create predictor
    predictor = create_enhanced_predictor()
    if predictor:
        print("✓ Predictor created")
        
        # Test prediction
        test_features = {
            'age': 60, 'sex': 1, 'systolic_bp': 150,
            'cholesterol': 240, 'heart_rate': 80
        }
        result = predictor.predict_with_explanation(test_features)
        print(f"\nTest prediction:")
        print(f"  Probability: {result['probability']:.2%}")
        print(f"  Category: {result['prediction']}")
        print(f"  Explanations: {len(result['explanations'])} features")
    else:
        print("✗ Could not create predictor")
