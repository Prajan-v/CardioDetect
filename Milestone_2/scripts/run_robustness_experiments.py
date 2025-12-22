
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import joblib
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Add pipeline directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.prediction_pipeline import PredictionPipeline
from pipeline.integrated_pipeline import DualModelPipeline

def generate_test_report(data, filename):
    """Generate a simple PDF medical report for OCR testing"""
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, y, "MEDICAL LABORATORY REPORT")
    y -= 40
    c.setFont("Helvetica", 12)
    
    # Write fields that the OCR looks for
    c.drawString(100, y, f"Patient Age: {data['age']} Years")
    y -= 25
    c.drawString(100, y, f"Gender: {'Male' if data['sex']==1 else 'Female'}")
    y -= 25
    c.drawString(100, y, f"Blood Pressure: {data['systolic_bp']}/{data['diastolic_bp']} mmHg")
    y -= 25
    c.drawString(100, y, f"Total Cholesterol: {data['total_cholesterol']} mg/dL")
    y -= 25
    c.drawString(100, y, f"Fasting Glucose: {data['fasting_glucose']} mg/dL")
    y -= 25
    c.drawString(100, y, f"Heart Rate: {data['heart_rate']} bpm")
    y -= 25
    c.drawString(100, y, f"BMI: {data['bmi']}")
    y -= 25
    c.drawString(100, y, f"Smoking Status: {'Current' if data['smoking']==1 else 'Non-smoker'}")
    y -= 25
    c.drawString(100, y, f"Diabetes History: {'Yes' if data['diabetes']==1 else 'No'}")
    
    c.save()

def run_noise_stability(pipeline, dataset_path):
    print("\n🔬 EXPERIMENT 1: Synthetic Noise Stability Analysis")
    print("-" * 60)
    
    df = pd.read_csv(dataset_path)
    sample_size = 50
    # Use valid rows
    df = df.dropna().head(sample_size)
    
    perturb_features = ['systolic_bp', 'total_cholesterol', 'fasting_glucose', 'bmi', 'heart_rate']
    noise_level = 0.05 # 5% noise
    
    shifts = []
    flips = 0
    total_preds = 0
    
    print(f"Testing {sample_size} samples with +/- {noise_level*100}% random noise...")
    
    for _, row in df.iterrows():
        original_data = row.to_dict()
        base_pred = pipeline.predict(original_data)
        base_prob = base_pred['probability']
        base_risk = base_pred['risk_level']
        
        # Monte Carlo simulation per patient
        n_sims = 20
        patient_diffs = []
        
        for _ in range(n_sims):
            noisy_data = original_data.copy()
            for feat in perturb_features:
                val = noisy_data.get(feat, 0)
                noise = np.random.normal(0, val * noise_level)
                noisy_data[feat] = val + noise
            
            new_pred = pipeline.predict(noisy_data)
            patient_diffs.append(abs(new_pred['probability'] - base_prob))
            
            if new_pred['risk_level'] != base_risk:
                flips += 1
            total_preds += 1
            
        shifts.extend(patient_diffs)

    mean_shift = np.mean(shifts)
    flip_rate = (flips / total_preds) * 100
    
    print(f"✅ Prediction Stability Results:")
    print(f"   Mean Probability Shift: {mean_shift:.4f} (avg change in risk score)")
    print(f"   Risk Category Flip Rate: {flip_rate:.1f}% (cases where noise changed Low/Mod/High)")
    print(f"   Interpretation: {'Robust' if flip_rate < 15 else 'Sensitive'}")
    return mean_shift, flip_rate

def run_ocr_robustness(integrated_pipeline, prediction_pipeline, dataset_path):
    print("\n🔬 EXPERIMENT 2: OCR Robustness vs Perfect CSV")
    print("-" * 60)
    
    df = pd.read_csv(dataset_path).head(5) # 5 samples
    
    mae_features = []
    risk_diffs = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        true_data = row.to_dict()
        
        # Generate PDF
        pdf_path = f"test_patient_{i}.pdf"
        generate_test_report(true_data, pdf_path)
        
        # Run OCR Pipeline
        # Note: process_document usually takes path
        ocr_result = integrated_pipeline.process_document(pdf_path)
        extracted_data = ocr_result.get('extracted_data', {})
        
        # Ground Truth Prediction
        true_pred = prediction_pipeline.predict(true_data)
        
        # OCR Prediction
        ocr_pred_prob = ocr_result.get('prediction_pipeline', {}).get('probability', 0)
        
        # Calculate Errors
        diff = abs(true_pred['probability'] - ocr_pred_prob)
        risk_diffs.append(diff)
        
        print(f"Patient {i}: True Risk={true_pred['probability']:.2f}, OCR Risk={ocr_pred_prob:.2f}, Diff={diff:.4f}")
        
        # Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            
    mean_risk_diff = np.mean(risk_diffs)
    print(f"✅ OCR Robustness Results:")
    print(f"   Mean Absolute Error (Risk Probability): {mean_risk_diff:.4f}")
    print(f"   Impact: OCR errors cause ~{mean_risk_diff*100:.1f}% deviation in risk calculation")
    return mean_risk_diff


class PatchedPredictionPipeline(PredictionPipeline):
    def _load_models(self):
        """Patch to load scaler from within prediction_xgb.pkl if needed"""
        try:
            # point to prediction_xgb.pkl
            xgb_path = self.model_dir / "prediction_xgb.pkl"
            if xgb_path.exists():
                data = joblib.load(xgb_path)
                if isinstance(data, dict):
                    self.scaler = data.get('scaler')
                    self.models['xgb'] = data.get('model')
                    # Set defaults for others or ignore
                    print("✓ [Patched PP] Loaded model & scaler from dict")
                else:
                    self.models['xgb'] = data
                    # try to load scaler separately if possible, or fail gracefully
            
            print(f"✓ [Patched PP] Loaded {len(self.models)} prediction models")
        except Exception as e:
            print(f"⚠️ Error loading models in PatchedPP: {e}")

    def preprocess(self, data):
        """Override preprocess to match the 8-feature schema of the new 91.63% model"""
        # Features: age, sex, systolic_bp, total_cholesterol, heart_rate, age_sq, elderly, high_bp_flag
        
        # Get base values with defaults
        age = data.get('age', 55)
        sex = data.get('sex', 1) 
        # Note: 'sex' in data might be 0/1. If it's 'M'/'F', handle widely? 
        # The dataset has 0/1 for sex.
        
        systolic_bp = data.get('systolic_bp', 120)
        total_cholesterol = data.get('total_cholesterol', 200)
        heart_rate = data.get('heart_rate', 75)
        
        # Engineer features
        age_sq = age ** 2
        elderly = 1 if age >= 55 else 0
        high_bp_flag = 1 if systolic_bp >= 140 else 0
        
        # Create vector
        X = np.array([[
            age, sex, systolic_bp, total_cholesterol, heart_rate,
            age_sq, elderly, high_bp_flag
        ]])
        
        if self.scaler:
            X = self.scaler.transform(X)
            
        return X

class PatchedDualModelPipeline(DualModelPipeline):
    """
    Subclass to fix model paths at runtime without modifying the original file.
    Points to 'Final_models' instead of 'models'.
    """
    def _load_models(self):
        """Load detection and prediction models from correct Final_models location"""
        # Hardcoded correct path for experiment
        base_dir = Path("/Users/prajanv/CardioDetect/Milestone_2")
        models_dir = base_dir / "models/Final_models"
        
        # Detection models (UCI - current disease)
        self.detection_models = {}
        self.detection_scaler = None
        
        detection_dir = models_dir / "detection"
        if detection_dir.exists():
            for model_file in detection_dir.glob("detection_*.pkl"):
                if 'scaler' not in model_file.name:
                    name = model_file.stem.replace('detection_', '')
                    self.detection_models[name] = joblib.load(model_file)
            
            scaler_path = detection_dir / "detection_scaler.pkl"
            if scaler_path.exists():
                self.detection_scaler = joblib.load(scaler_path)
        
        # Prediction model (91.63% XGBoost)
        self.prediction_model = None
        self.prediction_scaler = None
        
        prediction_dir = models_dir / "prediction"
        pred_path = prediction_dir / "prediction_xgb.pkl"
        if pred_path.exists():
            model_data = joblib.load(pred_path)
            if hasattr(model_data, 'predict'):
                self.prediction_model = model_data
            else:
                self.prediction_model = model_data.get('model')
                self.prediction_scaler = model_data.get('scaler')
                self.prediction_threshold = model_data.get('threshold', 0.5)
        
        print(f"✓ [Patched] Loaded {len(self.detection_models)} detection models")
        print(f"✓ [Patched] Loaded prediction model: {self.prediction_model is not None}")

def main():
    try:
        # Paths
        base_dir = Path("/Users/prajanv/CardioDetect")
        milestone_dir = base_dir / "Milestone_2"
        data_path = base_dir / "data/final_dataset/prediction_data/final_risk_dataset_properly_cleaned.csv"
        
        # Initialize Pipelines
        print("Initializing pipelines...")
        # Point to Final_models/prediction
        pp = PatchedPredictionPipeline(model_dir=milestone_dir/"models/Final_models/prediction")
        
        # Use PATCHED pipeline
        ip = PatchedDualModelPipeline() 
        
        
        # Run Experiments
        run_noise_stability(pp, data_path)
        run_ocr_robustness(ip, pp, data_path)
        
        # Run User Specified Image
        syn_img = base_dir / "Milestone_2/Medical_report/Synthetic_report/SYN-002.png"
        if syn_img.exists():
            print(f"\n🔬 EXPERIMENT 3: Single Image Check ({syn_img.name})")
            print("-" * 60)
            res = ip.process_document(str(syn_img))
            print("Extracted:", res.get('fields', {}))
            print("Prediction:", res.get('prediction', {}))
        
    except Exception as e:
        print(f"❌ Experiment Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
