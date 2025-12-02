import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# External libraries
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_splits(train_path="./data/split/train.csv",
                val_path="./data/split/val.csv",
                test_path="./data/split/test.csv"):
    """
    Loads train/val/test CSVs, splits into X/y, and drops non-feature columns.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print("="*80)
    print("LOADING DATA SPLITS")
    print("="*80)
    
    # Load CSVs
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"\nLoaded splits:")
    print(f"  Train: {train_df.shape}")
    print(f"  Val:   {val_df.shape}")
    print(f"  Test:  {test_df.shape}")
    
    # Separate features and target
    # Drop non-feature columns: risk_target (target) and data_source (metadata)
    drop_cols = ['risk_target', 'data_source']

    # Binarize target explicitly: 0 = no event, 1 = any event
    # This is important because some sources use multi-class encodings (e.g. 0-4)
    y_train = (train_df['risk_target'] > 0).astype(int)
    y_val = (val_df['risk_target'] > 0).astype(int)
    y_test = (test_df['risk_target'] > 0).astype(int)

    X_train = train_df.drop(columns=drop_cols)
    X_val = val_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)
    
    print(f"\nFeature matrix shape: {X_train.shape[1]} features")
    print(f"Target distribution (train): {y_train.value_counts().to_dict()}")
    print(f"Target distribution (test):  {y_test.value_counts().to_dict()}")
    print("="*80 + "\n")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_features(X_train, X_val, X_test):
    """
    Standardize features using StandardScaler (fit on train, transform all).
    
    Args:
        X_train, X_val, X_test: Feature matrices
    
    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def encode_categorical_features(X_train, X_val, X_test):
    """One-hot encode any non-numeric (object/category) features.

    I use this to ensure that all models receive purely numeric inputs.
    I fit the encoding jointly on train/val/test so that the columns
    stay aligned across all splits.

    Args:
        X_train, X_val, X_test: Feature DataFrames

    Returns:
        tuple: (X_train_encoded, X_val_encoded, X_test_encoded)
    """

    # Concatenate with a split key so I can split back later
    combined = pd.concat(
        [X_train, X_val, X_test],
        keys=["train", "val", "test"],
        names=["split", None]
    )

    # Identify categorical columns (object or pandas Categorical)
    categorical_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        # Nothing to encode, return as-is
        return X_train, X_val, X_test

    print("Encoding categorical features with one-hot encoding:")
    print(f"  Categorical columns: {categorical_cols}")

    # One-hot encode only the categorical columns, keep numeric as-is
    combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

    # Split back into train/val/test
    X_train_enc = combined_encoded.xs("train")
    X_val_enc = combined_encoded.xs("val")
    X_test_enc = combined_encoded.xs("test")

    print(f"  Original feature count: {X_train.shape[1]}")
    print(f"  Encoded feature count:  {X_train_enc.shape[1]}")

    return X_train_enc, X_val_enc, X_test_enc


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def compute_metrics(y_true, y_pred, y_proba, set_name="Test"):
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class
        set_name: Name of the dataset (for printing)
    
    Returns:
        dict: Metrics dictionary
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics


def print_metrics(metrics, model_name, set_name="Test"):
    """
    Print metrics in a formatted way.
    """
    print(f"\n{model_name} - {set_name} Set Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains Logistic Regression with class balancing.
    
    Returns:
        tuple: (fitted_model, metrics_dict)
    """
    print("\n" + "="*80)
    print("TRAINING: Logistic Regression")
    print("="*80)
    
    # Scale features (LR benefits from scaling)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Train model
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on all sets
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test")
    print_metrics(metrics, "Logistic Regression", "Test")
    
    # Store scaler with model
    model.scaler = scaler
    
    return model, metrics


def train_evaluate_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains Random Forest with class balancing.
    
    Returns:
        tuple: (fitted_model, metrics_dict)
    """
    print("\n" + "="*80)
    print("TRAINING: Random Forest")
    print("="*80)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_test_pred = model.predict(X_test.values)
    y_test_proba = model.predict_proba(X_test.values)[:, 1]
    
    metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test")
    print_metrics(metrics, "Random Forest", "Test")
    
    return model, metrics


def train_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains XGBoost with class balancing.
    
    Returns:
        tuple: (fitted_model, metrics_dict)
    """
    print("\n" + "="*80)
    print("TRAINING: XGBoost")
    print("="*80)
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Use NumPy arrays to avoid issues with column name encoding
    model.fit(X_train.values, y_train.values, verbose=False)
    
    # Evaluate
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test")
    print_metrics(metrics, "XGBoost", "Test")
    
    return model, metrics


def train_evaluate_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains LightGBM with class balancing.
    
    Returns:
        tuple: (fitted_model, metrics_dict)
    """
    print("\n" + "="*80)
    print("TRAINING: LightGBM")
    print("="*80)
    
    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test")
    print_metrics(metrics, "LightGBM", "Test")
    
    return model, metrics


def train_evaluate_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains SVM (RBF kernel) with class balancing.
    Note: SVM requires scaling and can be slow on large datasets.
    
    Returns:
        tuple: (fitted_model, metrics_dict)
    """
    print("\n" + "="*80)
    print("TRAINING: SVM (RBF)")
    print("="*80)
    
    # Scale features (SVM requires scaling)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Train model
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,  # Enable probability estimates
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test")
    print_metrics(metrics, "SVM (RBF)", "Test")
    
    # Store scaler with model
    model.scaler = scaler
    
    return model, metrics


def train_evaluate_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains Gradient Boosting Classifier (sklearn).
    
    Returns:
        tuple: (fitted_model, metrics_dict)
    """
    print("\n" + "="*80)
    print("TRAINING: Gradient Boosting")
    print("="*80)
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test")
    print_metrics(metrics, "Gradient Boosting", "Test")
    
    return model, metrics


def train_evaluate_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains Multi-Layer Perceptron (Neural Network).
    
    Returns:
        tuple: (fitted_model, metrics_dict)
    """
    print("\n" + "="*80)
    print("TRAINING: MLP (Neural Network)")
    print("="*80)
    
    # Scale features (MLPs require scaling)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Train model
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test")
    print_metrics(metrics, "MLP", "Test")
    
    # Store scaler with model
    model.scaler = scaler
    
    return model, metrics


def train_evaluate_ensemble(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains Soft-Voting Ensemble combining RF + XGBoost + LightGBM + MLP.
    
    Returns:
        tuple: (fitted_model, metrics_dict)
    """
    print("\n" + "="*80)
    print("TRAINING: Soft-Voting Ensemble (RF + XGB + LGBM + MLP)")
    print("="*80)
    
    # Scale features for MLP component
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Define base estimators
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    # Train individual models first (for MLP on scaled data)
    print("\n  Training base models...")
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train.values, y_train.values, verbose=False)
    lgbm.fit(X_train, y_train)
    mlp.fit(X_train_scaled, y_train)
    
    # Create soft-voting ensemble
    # Note: We'll manually combine predictions since MLP needs scaled data
    print("  Creating ensemble predictions...")
    
    # Get probabilities from each model
    rf_proba = rf.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test.values)[:, 1]
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
    mlp_proba = mlp.predict_proba(X_test_scaled)[:, 1]
    
    # Average probabilities (soft voting)
    y_test_proba = (rf_proba + xgb_proba + lgbm_proba + mlp_proba) / 4
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    
    metrics = compute_metrics(y_test, y_test_pred, y_test_proba, "Test")
    print_metrics(metrics, "Ensemble", "Test")
    
    # Package models together
    ensemble = {
        'rf': rf,
        'xgb': xgb_model,
        'lgbm': lgbm,
        'mlp': mlp,
        'scaler': scaler
    }
    
    return ensemble, metrics


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(results_dict):
    """
    Given a dict like {"LogReg": metrics, "RandomForest": metrics, ...},
    prints a neat summary table and returns the name of the best model.
    
    Selection rule:
      - Highest test accuracy.
      - If tie, choose higher test recall on positives.
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
    
    Returns:
        str: Name of the best model
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # Build comparison DataFrame
    comparison_data = []
    
    for model_name, metrics in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'ROC-AUC': metrics['roc_auc']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Sort by accuracy (descending), then recall (descending)
    df_comparison = df_comparison.sort_values(
        by=['Accuracy', 'Recall'],
        ascending=[False, False]
    ).reset_index(drop=True)
    
    # Print table
    print("\n" + df_comparison.to_string(index=False))
    
    # Best model
    best_model_name = df_comparison.iloc[0]['Model']
    best_accuracy = df_comparison.iloc[0]['Accuracy']
    best_recall = df_comparison.iloc[0]['Recall']
    
    print("\n" + "="*80)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    print(f"   Recall:   {best_recall:.4f}")
    print("="*80 + "\n")
    
    return best_model_name, df_comparison


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_all_models():
    """
    Loads the data splits, trains all models, evaluates them,
    prints a summary table, and returns:
    - best_model_name
    - best_model_object
    - all_results_dict
    
    Returns:
        tuple: (best_model_name, best_model_object, all_results_dict, comparison_df)
    """
    print("\n" + "="*80)
    print("CARDIODETECT - COMPREHENSIVE MODEL TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    # Ensure all features are numeric (one-hot encode categorical columns)
    X_train, X_val, X_test = encode_categorical_features(X_train, X_val, X_test)
    
    # Store all results
    all_results = {}
    all_models = {}
    
    # Train each model
    print("\n" + "="*80)
    print("TRAINING ALL MODELS")
    print("="*80)
    
    # 1. Logistic Regression
    model, metrics = train_evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['Logistic Regression'] = metrics
    all_models['Logistic Regression'] = model
    
    # 2. Random Forest
    model, metrics = train_evaluate_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['Random Forest'] = metrics
    all_models['Random Forest'] = model
    
    # 3. XGBoost
    model, metrics = train_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['XGBoost'] = metrics
    all_models['XGBoost'] = model
    
    # 4. LightGBM
    model, metrics = train_evaluate_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['LightGBM'] = metrics
    all_models['LightGBM'] = model
    
    # 5. SVM
    model, metrics = train_evaluate_svm(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['SVM (RBF)'] = metrics
    all_models['SVM (RBF)'] = model
    
    # 6. Gradient Boosting
    model, metrics = train_evaluate_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['Gradient Boosting'] = metrics
    all_models['Gradient Boosting'] = model
    
    # 7. MLP
    model, metrics = train_evaluate_mlp(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['MLP'] = metrics
    all_models['MLP'] = model
    
    # 8. Ensemble
    model, metrics = train_evaluate_ensemble(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['Ensemble'] = metrics
    all_models['Ensemble'] = model
    
    # Compare models
    best_model_name, comparison_df = compare_models(all_results)
    best_model = all_models[best_model_name]
    
    return best_model_name, best_model, all_results, comparison_df


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Run the full pipeline
    best_name, best_model, results, comparison = run_all_models()
    
    print("\n✅ Model training complete!")
    print(f"   Best model: {best_name}")
    print(f"   Results saved in returned dictionaries.")