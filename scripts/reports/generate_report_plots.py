import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.mlp_tuning import load_splits, encode_categorical_features

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLP_V2_PATH = MODELS_DIR / "mlp_v2.pkl"

def generate_plots():
    print("Loading data...")
    # Load test data
    _, _, _, _, X_test, y_test = load_splits()
    
    # We need to encode features just like in training
    # But load_splits returns raw dataframes for X.
    # We need the training set to fit the encoder if we were doing it properly,
    # but mlp_tuning.encode_categorical_features handles all splits together.
    X_train, _, X_val, _, X_test_raw, _ = load_splits()
    _, _, X_test_enc = encode_categorical_features(X_train, X_val, X_test_raw)
    
    print(f"Loading model from {MLP_V2_PATH}...")
    if not MLP_V2_PATH.exists():
        print(f"Error: Model file not found at {MLP_V2_PATH}")
        return

    artifact = joblib.load(MLP_V2_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    
    # Prepare test data
    X_test_scaled = scaler.transform(X_test_enc)
    
    # Predict
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrix_v2.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")
    
    # 2. ROC Curve
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = FIGURES_DIR / "roc_curve_v2.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curve to {roc_path}")

if __name__ == "__main__":
    generate_plots()
