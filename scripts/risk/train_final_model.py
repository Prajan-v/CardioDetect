import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score

def train_final_model():
    print("=" * 80)
    print("🚀 TRAINING FINAL DIAGNOSTIC MODEL")
    print("=" * 80)
    
    # 1. Load Data
    data_path = 'data/processed/diagnostic_2019.csv'
    print(f"   Loading {data_path}...")
    df = pd.read_csv(data_path)
    
    # Drop data_source
    if 'data_source' in df.columns:
        df = df.drop('data_source', axis=1)
        
    # Encoding
    obj_cols = df.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        le = LabelEncoder()
        for col in obj_cols:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    # Preprocess
    print("   Preprocessing...")
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save Preprocessor
    joblib.dump((imputer, scaler), 'models/diagnostic_preprocessor.pkl')
    
    # 2. Train with Optimized Hyperparameters
    print("   Training LightGBM (Optuna Optimized)...")
    params = {
        'n_estimators': 680,
        'max_depth': 9,
        'num_leaves': 150,
        'learning_rate': 0.1508,
        'min_child_samples': 52,
        'subsample': 0.706,
        'colsample_bytree': 0.741,
        'reg_alpha': 0.026,
        'reg_lambda': 7.44e-08,
        'random_state': 42,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    # 3. Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    
    print(f"\n✅ FINAL RESULTS:")
    print(f"   Accuracy:  {acc:.4f} ({acc:.2%})")
    print(f"   ROC-AUC:   {auc:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   Precision: {prec:.4f}")
    
    # Save Model
    joblib.dump(model, 'models/diagnostic_lightgbm.pkl')
    print("\n   Saved to models/diagnostic_lightgbm.pkl")
    print("   Saved to models/diagnostic_preprocessor.pkl")

if __name__ == "__main__":
    train_final_model()
