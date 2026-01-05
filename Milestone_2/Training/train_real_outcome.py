"""
Train classification model using ACTUAL disease outcomes (TenYearCHD).

This model predicts REAL heart disease events, not calculated risk scores.
This avoids the circular dependency issue and creates a genuine predictive model.

Dataset: Framingham Heart Study (actual 10-year follow-up outcomes)
Target: TenYearCHD (1 = developed CHD within 10 years, 0 = did not)
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


def load_framingham_with_outcomes():
    """Load Framingham data with actual TenYearCHD outcomes."""
    data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    
    # Load framingham_alt.csv which has TenYearCHD
    df = pd.read_csv(data_dir / "framingham_alt.csv")
    
    # Rename columns to match our schema
    df = df.rename(columns={
        'male': 'sex',
        'currentSmoker': 'smoking',
        'BPMeds': 'bp_meds',
        'prevalentHyp': 'hypertension',
        'totChol': 'total_cholesterol',
        'sysBP': 'systolic_bp',
        'diaBP': 'diastolic_bp',
        'BMI': 'bmi',
        'heartRate': 'heart_rate',
        'glucose': 'fasting_glucose',
    })
    
    # Drop rows with missing target
    df = df.dropna(subset=['TenYearCHD'])
    
    # Handle missing values in features
    numeric_cols = ['age', 'total_cholesterol', 'systolic_bp', 'diastolic_bp', 
                    'bmi', 'heart_rate', 'fasting_glucose', 'cigsPerDay']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    binary_cols = ['sex', 'smoking', 'bp_meds', 'hypertension', 'diabetes']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def create_features(df):
    """Create features for the model."""
    feature_cols = [
        'sex', 'age', 'smoking', 'bp_meds', 'hypertension', 'diabetes',
        'total_cholesterol', 'systolic_bp', 'diastolic_bp', 'bmi', 
        'heart_rate', 'fasting_glucose'
    ]
    
    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].copy()
    y = df['TenYearCHD'].values
    
    # Add engineered features
    if 'systolic_bp' in X.columns and 'diastolic_bp' in X.columns:
        X['pulse_pressure'] = X['systolic_bp'] - X['diastolic_bp']
        X['map'] = X['diastolic_bp'] + (X['pulse_pressure'] / 3)
    
    if 'age' in X.columns and 'systolic_bp' in X.columns:
        X['age_sbp_interaction'] = X['age'] * X['systolic_bp']
    
    return X, y


def build_preprocessor(X):
    """Build preprocessing pipeline."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def plot_results(y_test, y_pred, y_proba, save_dir):
    """Generate and save result plots."""
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['No CHD', 'CHD']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix - Actual CHD Prediction',
           ylabel='True label',
           xlabel='Predicted label')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_dir / "real_outcome_confusion_matrix.png", dpi=150)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Actual CHD Prediction')
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(save_dir / "real_outcome_roc_curve.png", dpi=150)
    plt.close()
    
    return roc_auc


def train_real_outcome_model():
    """Train model on actual heart disease outcomes."""
    print("=" * 80)
    print("TRAINING MODEL ON ACTUAL DISEASE OUTCOMES (TenYearCHD)")
    print("This model predicts REAL heart disease events, not calculated risk scores")
    print("=" * 80)
    
    # Setup paths
    base_dir = Path(__file__).resolve().parents[1]
    reports_dir = base_dir / "reports" / "real_outcome"
    models_dir = base_dir / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading Framingham data with actual outcomes...")
    df = load_framingham_with_outcomes()
    print(f"Dataset size: {len(df)}")
    print(f"CHD positive cases: {df['TenYearCHD'].sum()} ({df['TenYearCHD'].mean()*100:.1f}%)")
    
    # Create features
    print("\n[2/6] Creating features...")
    X, y = create_features(df)
    print(f"Features: {X.shape[1]}")
    print(f"Feature list: {X.columns.tolist()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train CHD rate: {y_train.mean()*100:.1f}%")
    print(f"Test CHD rate: {y_test.mean()*100:.1f}%")
    
    # Build preprocessor
    preprocessor = build_preprocessor(X_train)
    
    # Define models with class weight balancing
    print("\n[3/6] Training models with SMOTE for class imbalance...")
    
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
        ),
    }
    
    results = []
    trained_models = {}
    
    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('prep', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', clf)
        ])
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Fit on full training data
        pipeline.fit(X_train, y_train)
        
        # Predict on test
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        
        print(f"  Test Accuracy:  {accuracy:.4f}")
        print(f"  Test Precision: {precision:.4f}")
        print(f"  Test Recall:    {recall:.4f}")
        print(f"  Test F1:        {f1:.4f}")
        print(f"  Test ROC-AUC:   {roc_auc:.4f}")
        
        results.append({
            'model_name': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std()
        })
        
        trained_models[name] = {
            'pipeline': pipeline,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    # Compare models
    print("\n[4/6] Model Comparison...")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roc_auc', ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by ROC-AUC)")
    print("=" * 70)
    print(results_df[['model_name', 'accuracy', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    print("=" * 70)
    
    # Select best model
    best_name = results_df.iloc[0]['model_name']
    best_model = trained_models[best_name]
    best_metrics = results_df.iloc[0]
    
    print(f"\n*** BEST MODEL: {best_name} ***")
    print(f"    ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"    Recall:  {best_metrics['recall']:.4f} (important for not missing CHD cases)")
    
    # Final evaluation
    print("\n[5/6] Final Evaluation...")
    print("\nClassification Report:")
    print(classification_report(y_test, best_model['y_pred'], 
                                target_names=['No CHD', 'CHD']))
    
    # Plot results
    roc_auc = plot_results(y_test, best_model['y_pred'], best_model['y_proba'], reports_dir)
    
    # Check for important metrics
    print("\n=== KEY METRICS FOR MEDICAL USE ===")
    cm = confusion_matrix(y_test, best_model['y_pred'])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"Sensitivity (Recall): {sensitivity:.4f} - Detects {sensitivity*100:.1f}% of CHD cases")
    print(f"Specificity:          {specificity:.4f}")
    print(f"PPV (Precision):      {ppv:.4f}")
    print(f"NPV:                  {npv:.4f}")
    print(f"Missed CHD cases:     {fn} out of {tp+fn} ({fn/(tp+fn)*100:.1f}%)")
    
    # Save model
    print("\n[6/6] Saving model...")
    
    model_path = models_dir / "real_outcome_classifier.pkl"
    joblib.dump(best_model['pipeline'], model_path)
    
    metadata = {
        'model_name': best_name,
        'target': 'TenYearCHD (actual heart disease outcome)',
        'accuracy': float(best_metrics['accuracy']),
        'precision': float(best_metrics['precision']),
        'recall': float(best_metrics['recall']),
        'f1': float(best_metrics['f1']),
        'roc_auc': float(best_metrics['roc_auc']),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'features': X.columns.tolist(),
        'trained_at': datetime.now().isoformat(),
        'note': 'This model predicts ACTUAL heart disease outcomes, not calculated risk scores'
    }
    
    with open(models_dir / "real_outcome_classifier_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    results_df.to_csv(reports_dir / "model_comparison.csv", index=False)
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Reports saved: {reports_dir}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - REAL OUTCOME PREDICTION MODEL")
    print("=" * 80)
    print(f"""
This model predicts ACTUAL heart disease events (TenYearCHD), not calculated risk.

Best Model: {best_name}
- Accuracy:  {best_metrics['accuracy']*100:.1f}%
- ROC-AUC:   {best_metrics['roc_auc']:.4f}
- Recall:    {best_metrics['recall']*100:.1f}% (detects this many CHD cases)

Note: This is a realistic accuracy for medical prediction.
Lower than 99% but based on REAL outcomes, not circular dependencies.
    """)
    print("=" * 80)
    
    return best_model['pipeline'], metadata


if __name__ == "__main__":
    train_real_outcome_model()
