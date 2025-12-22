"""
Detection Pipeline - Heart Disease Detection
Accuracy: 91.45%

This pipeline detects current heart disease status using UCI Heart Disease dataset.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple


class DetectionPipeline:
    """Pipeline for detecting current heart disease status"""
    
    def __init__(self, model_dir: str = "../../models/detection"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self._load_models()
    
    def _load_models(self):
        """Load all detection models"""
        try:
            self.scaler = joblib.load(self.model_dir / "detection_scaler.pkl")
            self.models['xgb'] = joblib.load(self.model_dir / "detection_xgb.pkl")
            self.models['lgbm'] = joblib.load(self.model_dir / "detection_lgbm.pkl")
            self.models['rf'] = joblib.load(self.model_dir / "detection_rf.pkl")
            self.models['et'] = joblib.load(self.model_dir / "detection_et.pkl")
            print(f"✅ Loaded {len(self.models)} detection models")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
    
    def preprocess(self, data: Dict) -> np.ndarray:
        """Preprocess input data"""
        values = [data.get(f, 0) for f in self.features]
        X = np.array([values])
        if self.scaler:
            X = self.scaler.transform(X)
        return X
    
    def predict(self, data: Dict) -> Dict[str, Any]:
        """
        Predict heart disease status
        
        Args:
            data: Dictionary with patient features
            
        Returns:
            Dictionary with prediction results
        """
        X = self.preprocess(data)
        
        # Get predictions from all models
        predictions = {}
        probabilities = []
        
        for name, model in self.models.items():
            prob = model.predict_proba(X)[0, 1]
            predictions[name] = prob
            probabilities.append(prob)
        
        # Ensemble average
        avg_prob = np.mean(probabilities)
        
        # Classification
        has_disease = avg_prob >= 0.5
        
        result = {
            'has_disease': has_disease,
            'probability': avg_prob,
            'prediction': '⚠️ HEART DISEASE DETECTED' if has_disease else '✅ NO DISEASE DETECTED',
            'confidence': avg_prob if has_disease else 1 - avg_prob,
            'model_predictions': predictions
        }
        
        # Add Clinical Advisor recommendations (ACC/AHA 2017-2019, WHO 2020)
        try:
            from clinical_advisor import ClinicalAdvisor
            advisor = ClinicalAdvisor()
            patient_features = {
                'age': data.get('age', 55),
                'systolic_bp': data.get('trestbps', 130),
                'diastolic_bp': 80,
                'diabetes': data.get('fbs', 0),
                'smoking': 0,
                'allergies': data.get('allergies', []),
                'on_bp_meds': data.get('on_bp_meds', False)
            }
            result['clinical_recommendations'] = advisor.generate_recommendations(patient_features)
        except ImportError:
            result['clinical_recommendations'] = None
        
        return result
    
    def predict_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Predict for multiple patients from CSV"""
        df = pd.read_csv(csv_path)
        results = []
        
        for idx, row in df.iterrows():
            data = row.to_dict()
            pred = self.predict(data)
            results.append({
                'patient_id': idx,
                'has_disease': pred['has_disease'],
                'probability': pred['probability'],
                'prediction': pred['prediction']
            })
        
        return pd.DataFrame(results)


def main():
    """Test the detection pipeline"""
    print("=" * 60)
    print("🫀 DETECTION PIPELINE - Heart Disease Detection")
    print("   Accuracy: 91.45%")
    print("=" * 60)
    
    # Sample patient
    patient = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }
    
    print("\n📋 Patient Data:")
    for k, v in patient.items():
        print(f"   • {k}: {v}")
    
    # Initialize and predict
    pipeline = DetectionPipeline()
    result = pipeline.predict(patient)
    
    print("\n📊 Results:")
    print(f"   {result['prediction']}")
    print(f"   Probability: {result['probability']:.1%}")
    print(f"   Confidence: {result['confidence']:.1%}")


if __name__ == "__main__":
    main()
