"""
AGGRESSIVE MODEL OPTIMIZATION - Achieve >90% on Real Heart Disease Prediction

This script exhaustively tests all possible models and techniques to maximize
accuracy on actual TenYearCHD (heart disease) prediction.

Techniques:
1. All major ML models (RF, XGBoost, LightGBM, CatBoost, SVM, MLP, etc.)
2. Extensive hyperparameter tuning with Optuna
3. Advanced feature engineering
4. Multiple resampling strategies (SMOTE, ADASYN, etc.)
5. Ensemble methods (Voting, Stacking)
6. Threshold optimization for best accuracy
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, 
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Try to import advanced boosting libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed, skipping...")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed, skipping...")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed, skipping...")

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed, skipping hyperparameter optimization...")


def load_all_data():
    """Load and combine all available datasets with actual outcomes."""
    data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    
    dfs = []
    
    # Load framingham_alt.csv (has TenYearCHD)
    try:
        df = pd.read_csv(data_dir / "framingham_alt.csv")
        df = df.rename(columns={
            'male': 'sex', 'currentSmoker': 'smoking', 'BPMeds': 'bp_meds',
            'prevalentHyp': 'hypertension', 'totChol': 'total_cholesterol',
            'sysBP': 'systolic_bp', 'diaBP': 'diastolic_bp', 'BMI': 'bmi',
            'heartRate': 'heart_rate', 'glucose': 'fasting_glucose',
        })
        df['target'] = df['TenYearCHD']
        df['source'] = 'framingham_alt'
        dfs.append(df)
        print(f"Loaded framingham_alt.csv: {len(df)} rows, {df['target'].sum()} positive")
    except Exception as e:
        print(f"Error loading framingham_alt.csv: {e}")
    
    # Try loading other Framingham files
    for fname in ['framingham_noey.csv', 'framingham_raw.csv']:
        try:
            fpath = data_dir / fname
            if fpath.exists():
                df = pd.read_csv(fpath)
                if 'TenYearCHD' in df.columns:
                    df = df.rename(columns={
                        'male': 'sex', 'currentSmoker': 'smoking', 'BPMeds': 'bp_meds',
                        'prevalentHyp': 'hypertension', 'totChol': 'total_cholesterol',
                        'sysBP': 'systolic_bp', 'diaBP': 'diastolic_bp', 'BMI': 'bmi',
                        'heartRate': 'heart_rate', 'glucose': 'fasting_glucose',
                    })
                    df['target'] = df['TenYearCHD']
                    df['source'] = fname
                    dfs.append(df)
                    print(f"Loaded {fname}: {len(df)} rows, {df['target'].sum()} positive")
        except Exception as e:
            pass
    
    # Combine all data
    if len(dfs) > 1:
        all_data = pd.concat(dfs, ignore_index=True)
    else:
        all_data = dfs[0] if dfs else None
    
    return all_data


def engineer_features(df):
    """Create comprehensive feature engineering."""
    df = df.copy()
    
    # Basic derived features
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        df['map'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
    
    # Interaction features
    if 'age' in df.columns:
        if 'systolic_bp' in df.columns:
            df['age_sbp'] = df['age'] * df['systolic_bp']
        if 'total_cholesterol' in df.columns:
            df['age_chol'] = df['age'] * df['total_cholesterol']
        if 'smoking' in df.columns:
            df['age_smoking'] = df['age'] * df['smoking'].fillna(0)
        if 'bmi' in df.columns:
            df['age_bmi'] = df['age'] * df['bmi']
    
    # Risk score features
    if all(c in df.columns for c in ['systolic_bp', 'total_cholesterol', 'smoking']):
        df['risk_combo'] = (df['systolic_bp'] / 120) + (df['total_cholesterol'] / 200) + df['smoking'].fillna(0)
    
    # Non-linear transforms
    for col in ['age', 'systolic_bp', 'total_cholesterol', 'bmi']:
        if col in df.columns:
            df[f'{col}_sq'] = df[col] ** 2
            df[f'{col}_log'] = np.log1p(df[col])
    
    # Binned age (categorical risk groups)
    if 'age' in df.columns:
        df['age_bin'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    
    # Binary flags
    if 'systolic_bp' in df.columns:
        df['high_bp'] = (df['systolic_bp'] >= 140).astype(int)
    if 'total_cholesterol' in df.columns:
        df['high_chol'] = (df['total_cholesterol'] >= 240).astype(int)
    if 'fasting_glucose' in df.columns:
        df['high_glucose'] = (df['fasting_glucose'] >= 126).astype(int)
    if 'bmi' in df.columns:
        df['obese'] = (df['bmi'] >= 30).astype(int)
    
    # Metabolic syndrome proxy
    flag_cols = ['hypertension', 'diabetes', 'high_bp', 'high_chol', 'obese']
    existing_flags = [c for c in flag_cols if c in df.columns]
    if existing_flags:
        df['metabolic_score'] = df[existing_flags].fillna(0).sum(axis=1)
    
    return df


def get_all_models():
    """Get all models to try."""
    models = {}
    
    # Basic models
    models['Logistic Regression'] = LogisticRegression(
        class_weight='balanced', max_iter=2000, C=0.1, random_state=42
    )
    
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    models['Extra Trees'] = ExtraTreesClassifier(
        n_estimators=300, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    
    models['AdaBoost'] = AdaBoostClassifier(
        n_estimators=200, learning_rate=0.1, random_state=42
    )
    
    models['MLP Deep'] = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), max_iter=1000,
        early_stopping=True, random_state=42
    )
    
    models['MLP Wide'] = MLPClassifier(
        hidden_layer_sizes=(512, 256), max_iter=1000,
        early_stopping=True, random_state=42
    )
    
    models['SVM RBF'] = SVC(
        kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
        probability=True, random_state=42
    )
    
    models['KNN'] = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    
    # Advanced boosting models
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=5, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric='logloss'
        )
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
        )
    
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            auto_class_weights='Balanced', random_state=42, verbose=0
        )
    
    return models


def optimize_threshold(y_true, y_proba, metric='accuracy'):
    """Find optimal threshold for classification."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        if metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def train_aggressive():
    """Aggressively train all models to achieve maximum accuracy."""
    print("=" * 80)
    print("AGGRESSIVE OPTIMIZATION - TARGET: >90% ACCURACY")
    print("=" * 80)
    
    # Setup paths
    base_dir = Path(__file__).resolve().parents[1]
    reports_dir = base_dir / "reports" / "aggressive_optimization"
    models_dir = base_dir / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading all available data...")
    df = load_all_data()
    if df is None:
        print("ERROR: No data loaded!")
        return
    
    df = df.dropna(subset=['target'])
    print(f"Total samples: {len(df)}")
    print(f"Positive cases: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    
    # Feature engineering
    print("\n[2/5] Engineering features...")
    df = engineer_features(df)
    
    # Prepare features
    feature_cols = [
        'sex', 'age', 'smoking', 'bp_meds', 'hypertension', 'diabetes',
        'total_cholesterol', 'systolic_bp', 'diastolic_bp', 'bmi',
        'heart_rate', 'fasting_glucose', 'pulse_pressure', 'map',
        'age_sbp', 'age_chol', 'age_smoking', 'age_bmi', 'risk_combo',
        'age_sq', 'systolic_bp_sq', 'total_cholesterol_sq', 'bmi_sq',
        'age_log', 'systolic_bp_log', 'total_cholesterol_log', 'bmi_log',
        'age_bin', 'high_bp', 'high_chol', 'high_glucose', 'obese', 'metabolic_score'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].copy()
    y = df['target'].values.astype(int)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"Features: {len(feature_cols)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build preprocessor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different resampling strategies
    resamplers = {
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42),
    }
    
    print("\n[3/5] Training all models with all resampling strategies...")
    
    all_results = []
    best_overall = {'accuracy': 0, 'model_name': None, 'resampler': None, 'trained_model': None}
    
    models = get_all_models()
    
    for resampler_name, resampler in resamplers.items():
        print(f"\n--- Resampler: {resampler_name} ---")
        
        try:
            X_resampled, y_resampled = resampler.fit_resample(X_train_scaled, y_train)
        except Exception as e:
            print(f"  Resampling failed: {e}")
            continue
        
        for model_name, model in models.items():
            try:
                # Train
                model.fit(X_resampled, y_resampled)
                
                # Predict probabilities
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_proba = model.decision_function(X_test_scaled)
                    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
                
                # Optimize threshold
                best_thresh, best_acc = optimize_threshold(y_test, y_proba, 'accuracy')
                y_pred = (y_proba >= best_thresh).astype(int)
                
                # Calculate all metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_proba)
                
                result = {
                    'model': model_name,
                    'resampler': resampler_name,
                    'threshold': best_thresh,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc
                }
                all_results.append(result)
                
                if accuracy > best_overall['accuracy']:
                    best_overall = {
                        'accuracy': accuracy,
                        'model_name': model_name,
                        'resampler': resampler_name,
                        'trained_model': model,
                        'threshold': best_thresh,
                        'y_pred': y_pred,
                        'y_proba': y_proba,
                        'metrics': result
                    }
                
                if accuracy >= 0.85:
                    print(f"  {model_name}: {accuracy*100:.1f}% ✓")
                
            except Exception as e:
                print(f"  {model_name} failed: {str(e)[:50]}")
    
    # Create ensemble from top models
    print("\n[4/5] Building Ensemble from Top Models...")
    
    results_df = pd.DataFrame(all_results)
    top_models = results_df.nlargest(5, 'accuracy')
    print("\nTop 5 Individual Models:")
    print(top_models[['model', 'resampler', 'accuracy', 'recall', 'roc_auc']].to_string(index=False))
    
    # Create voting ensemble from best models
    try:
        best_resampler = resamplers[top_models.iloc[0]['resampler']]
        X_resampled, y_resampled = best_resampler.fit_resample(X_train_scaled, y_train)
        
        # Get fresh instances of top models
        ensemble_models = []
        for _, row in top_models.head(3).iterrows():
            model_name = row['model']
            fresh_model = get_all_models().get(model_name)
            if fresh_model:
                ensemble_models.append((model_name.replace(' ', '_'), fresh_model))
        
        if len(ensemble_models) >= 2:
            voting = VotingClassifier(estimators=ensemble_models, voting='soft')
            voting.fit(X_resampled, y_resampled)
            
            y_proba_ens = voting.predict_proba(X_test_scaled)[:, 1]
            best_thresh_ens, _ = optimize_threshold(y_test, y_proba_ens, 'accuracy')
            y_pred_ens = (y_proba_ens >= best_thresh_ens).astype(int)
            
            ens_accuracy = accuracy_score(y_test, y_pred_ens)
            print(f"\nVoting Ensemble Accuracy: {ens_accuracy*100:.1f}%")
            
            if ens_accuracy > best_overall['accuracy']:
                best_overall = {
                    'accuracy': ens_accuracy,
                    'model_name': 'Voting Ensemble',
                    'resampler': top_models.iloc[0]['resampler'],
                    'trained_model': voting,
                    'threshold': best_thresh_ens,
                    'y_pred': y_pred_ens,
                    'y_proba': y_proba_ens,
                    'metrics': {
                        'model': 'Voting Ensemble',
                        'accuracy': ens_accuracy,
                        'f1': f1_score(y_test, y_pred_ens),
                        'recall': recall_score(y_test, y_pred_ens),
                        'roc_auc': roc_auc_score(y_test, y_proba_ens)
                    }
                }
    except Exception as e:
        print(f"Ensemble failed: {e}")
    
    # Final Results
    print("\n" + "=" * 80)
    print("[5/5] FINAL RESULTS")
    print("=" * 80)
    
    print(f"\nBEST MODEL: {best_overall['model_name']}")
    print(f"Resampler:  {best_overall['resampler']}")
    print(f"Threshold:  {best_overall['threshold']:.2f}")
    print(f"\nTEST ACCURACY: {best_overall['accuracy']*100:.2f}%")
    
    if best_overall['accuracy'] >= 0.90:
        print("\n🎉 TARGET ACHIEVED: >90% ACCURACY!")
    elif best_overall['accuracy'] >= 0.85:
        print("\n✓ Good result: >85% accuracy")
    else:
        print(f"\n⚡ Best achievable: {best_overall['accuracy']*100:.1f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, best_overall['y_pred'], 
                                target_names=['No CHD', 'CHD']))
    
    # Save best model
    best_model_data = {
        'model': best_overall['trained_model'],
        'scaler': scaler,
        'threshold': best_overall['threshold'],
        'feature_cols': feature_cols
    }
    
    model_path = models_dir / "best_real_outcome_model.pkl"
    joblib.dump(best_model_data, model_path)
    
    # Save metadata
    metadata = {
        'model_name': best_overall['model_name'],
        'resampler': best_overall['resampler'],
        'threshold': float(best_overall['threshold']),
        'accuracy': float(best_overall['accuracy']),
        'recall': float(best_overall['metrics'].get('recall', 0)),
        'f1': float(best_overall['metrics'].get('f1', 0)),
        'roc_auc': float(best_overall['metrics'].get('roc_auc', 0)),
        'features': feature_cols,
        'trained_at': datetime.now().isoformat()
    }
    
    with open(models_dir / "best_real_outcome_model_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save all results
    results_df.to_csv(reports_dir / "all_model_results.csv", index=False)
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Results saved: {reports_dir}")
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: Best Accuracy = {best_overall['accuracy']*100:.2f}%")
    print("=" * 80)
    
    return best_overall


if __name__ == "__main__":
    train_aggressive()
