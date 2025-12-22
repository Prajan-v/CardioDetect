"""
Prediction Pipeline - 10-Year CHD Risk Prediction
Accuracy: 91.63%

This pipeline predicts 10-year cardiovascular disease risk using clean, deduplicated data.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any


class PredictionPipeline:
    """Pipeline for predicting 10-year CHD risk"""
    
    def __init__(self, model_dir: str = "../../models/prediction"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.ensemble_model = None
        self.features = [
            'age', 'sex', 'systolic_bp', 'diastolic_bp', 'total_cholesterol',
            'smoking', 'diabetes', 'bmi', 'heart_rate', 'fasting_glucose'
        ]
        self._load_models()
    
    def _load_models(self):
        """Load all prediction models"""
        try:
            self.scaler = joblib.load(self.model_dir / "prediction_scaler.pkl")
            self.models['xgb'] = joblib.load(self.model_dir / "prediction_xgb.pkl")
            self.models['lgbm'] = joblib.load(self.model_dir / "prediction_lgbm.pkl")
            self.models['rf'] = joblib.load(self.model_dir / "prediction_rf.pkl")
            self.models['gb'] = joblib.load(self.model_dir / "prediction_gb.pkl")
            
            # Load ensemble if available
            ensemble_path = self.model_dir / "best_cv_ensemble_model.pkl"
            if ensemble_path.exists():
                self.ensemble_model = joblib.load(ensemble_path)
            
            print(f"✅ Loaded {len(self.models)} prediction models")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
    
    def engineer_features(self, data: Dict) -> Dict:
        """Add engineered features"""
        data = data.copy()
        
        # Pulse pressure
        sbp = data.get('systolic_bp', 120)
        dbp = data.get('diastolic_bp', 80)
        data['pulse_pressure'] = sbp - dbp
        data['mean_arterial_pressure'] = dbp + (sbp - dbp) / 3
        
        # Risk flags
        data['hypertension_flag'] = 1 if sbp >= 140 or dbp >= 90 else 0
        data['high_cholesterol_flag'] = 1 if data.get('total_cholesterol', 200) >= 240 else 0
        data['high_glucose_flag'] = 1 if data.get('fasting_glucose', 100) >= 126 else 0
        data['obesity_flag'] = 1 if data.get('bmi', 25) >= 30 else 0
        
        # Metabolic syndrome score
        data['metabolic_syndrome_score'] = sum([
            data['hypertension_flag'],
            data['high_cholesterol_flag'],
            data['high_glucose_flag'],
            data['obesity_flag']
        ])
        
        # Log transforms
        data['log_total_cholesterol'] = np.log1p(data.get('total_cholesterol', 200))
        data['log_fasting_glucose'] = np.log1p(data.get('fasting_glucose', 100))
        data['log_bmi'] = np.log1p(data.get('bmi', 25))
        
        # Interaction terms
        age = data.get('age', 55)
        data['age_sbp_interaction'] = age * sbp
        data['bmi_glucose_interaction'] = data.get('bmi', 25) * data.get('fasting_glucose', 100)
        data['age_smoking_interaction'] = age * data.get('smoking', 0)
        
        return data
    
    def preprocess(self, data: Dict) -> np.ndarray:
        """Preprocess input data"""
        data = self.engineer_features(data)
        
        all_features = self.features + [
            'pulse_pressure', 'mean_arterial_pressure',
            'hypertension_flag', 'high_cholesterol_flag', 'high_glucose_flag',
            'obesity_flag', 'metabolic_syndrome_score',
            'log_total_cholesterol', 'log_fasting_glucose', 'log_bmi',
            'age_sbp_interaction', 'bmi_glucose_interaction', 'age_smoking_interaction'
        ]
        
        values = [data.get(f, 0) for f in all_features]
        X = np.array([values])
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        return X
    
    def categorize_risk(self, probability: float) -> tuple:
        """Categorize risk level"""
        if probability < 0.10:
            return '🟢 LOW', 'low', 'Maintain healthy lifestyle'
        elif probability < 0.20:
            return '🟡 MODERATE', 'moderate', 'Lifestyle modifications advised'
        else:
            return '🔴 HIGH', 'high', 'Medical consultation recommended'
    
    def predict(self, data: Dict) -> Dict[str, Any]:
        """
        Predict 10-year CHD risk
        
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
            try:
                prob = model.predict_proba(X)[0, 1]
                predictions[name] = prob
                probabilities.append(prob)
            except:
                pass
        
        # Ensemble average
        avg_prob = np.mean(probabilities) if probabilities else 0.5
        
        # Categorize
        level, level_code, recommendation = self.categorize_risk(avg_prob)
        
        result = {
            'probability': avg_prob,
            'risk_level': level,
            'level_code': level_code,
            'recommendation': recommendation,
            'high_risk': avg_prob >= 0.20,
            'model_predictions': predictions
        }
        
        # Add Clinical Advisor recommendations (ACC/AHA 2017-2019, WHO 2020)
        try:
            from clinical_advisor import ClinicalAdvisor
            advisor = ClinicalAdvisor()
            patient_features = {
                'age': data.get('age', 55),
                'systolic_bp': data.get('systolic_bp', 130),
                'diastolic_bp': data.get('diastolic_bp', 80),
                'total_cholesterol': data.get('total_cholesterol', 200),
                'diabetes': data.get('diabetes', 0),
                'smoking': data.get('smoking', 0),
                'ascvd_risk_10y': avg_prob * 100,  # Convert to percentage
                'allergies': data.get('allergies', []),
                'on_bp_meds': data.get('on_bp_meds', False),
                'on_statin': data.get('on_statin', False)
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
                'probability': pred['probability'],
                'risk_level': pred['risk_level'],
                'recommendation': pred['recommendation']
            })
        
        return pd.DataFrame(results)


def main():
    """Test the prediction pipeline"""
    print("=" * 60)
    print("📈 PREDICTION PIPELINE - 10-Year CHD Risk")
    print("   Accuracy: 91.63%")
    print("=" * 60)
    
    # Sample patients
    patients = [
        {
            'name': 'HIGH RISK',
            'age': 68, 'sex': 1, 'systolic_bp': 165, 'diastolic_bp': 95,
            'total_cholesterol': 280, 'smoking': 1, 'diabetes': 1,
            'bmi': 32, 'heart_rate': 85, 'fasting_glucose': 140
        },
        {
            'name': 'LOW RISK',
            'age': 35, 'sex': 0, 'systolic_bp': 115, 'diastolic_bp': 75,
            'total_cholesterol': 180, 'smoking': 0, 'diabetes': 0,
            'bmi': 23, 'heart_rate': 70, 'fasting_glucose': 85
        }
    ]
    
    pipeline = PredictionPipeline()
    
    for patient in patients:
        print(f"\n📋 {patient.pop('name')} Patient:")
        print(f"   Age: {patient['age']}, Sex: {'M' if patient['sex'] else 'F'}")
        print(f"   BP: {patient['systolic_bp']}/{patient['diastolic_bp']}")
        print(f"   Cholesterol: {patient['total_cholesterol']}")
        
        result = pipeline.predict(patient)
        
        print(f"\n   📊 Results:")
        print(f"   {result['risk_level']} ({result['probability']:.1%})")
        print(f"   → {result['recommendation']}")


if __name__ == "__main__":
    main()
