import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import optuna
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# PHASE 1: DATA & SPLITS
# ==========================================

def load_and_split_data(filepath):
    print("\n" + "="*40)
    print("PHASE 1: DATA & SPLITS")
    print("="*40)
    
    # Load
    df = pd.read_csv(filepath)
    print(f"Loaded data shape: {df.shape}")
    
    # Target distribution
    target_counts = df['TenYearCHD'].value_counts(normalize=True)
    print(f"\nTarget Distribution:\n{target_counts}")
    
    # Basic stats
    print(f"\nBasic Stats:\n{df[['age', 'sysBP', 'totChol', 'BMI', 'glucose']].describe().T[['mean', 'std', 'min', 'max']]}")
    
    # Cleaning
    # Drop rows where target is missing (if any)
    df = df.dropna(subset=['TenYearCHD'])
    
    # Split
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    
    # 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    
    # Split Temp into 15% Val, 15% Test (50% of 30% is 15%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    print(f"\nSplits:")
    print(f"Train: {X_train.shape[0]} ({y_train.mean():.1%} CHD)")
    print(f"Val:   {X_val.shape[0]} ({y_val.mean():.1%} CHD)")
    print(f"Test:  {X_test.shape[0]} ({y_test.mean():.1%} CHD)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ==========================================
# PHASE 2: FEATURE ENGINEERING
# ==========================================

def feature_engineering(X):
    X = X.copy()
    
    # Pulse Pressure
    X['pulse_pressure'] = X['sysBP'] - X['diaBP']
    
    # MAP
    X['map'] = X['diaBP'] + (X['pulse_pressure'] / 3)
    
    # BMI Categories
    X['bmi_cat'] = pd.cut(X['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Under', 'Normal', 'Over', 'Obese'])
    
    # Age Groups
    X['age_group'] = pd.cut(X['age'], bins=[20, 39, 49, 59, 69, 100], labels=['30s', '40s', '50s', '60s', '70+'])
    
    # Risk Flags
    X['high_bp'] = ((X['sysBP'] >= 140) | (X['diaBP'] >= 90)).astype(int)
    X['high_chol'] = (X['totChol'] >= 240).astype(int)
    X['high_glucose'] = (X['glucose'] >= 126).astype(int)
    X['high_bmi'] = (X['BMI'] >= 30).astype(int)
    
    # Metabolic Syndrome Count
    X['metabolic_syndrome'] = X['high_bp'] + X['high_chol'] + X['high_glucose'] + X['high_bmi']
    
    # Log Transforms
    for col in ['totChol', 'glucose', 'sysBP', 'BMI']:
        if col in X.columns:
            X[f'log_{col}'] = np.log1p(X[col])
            
    return X

def build_pipeline(X_train):
    # Identify columns
    # We need to run feature engineering once on a sample to get columns
    X_sample = feature_engineering(X_train.head())
    
    numeric_features = X_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Full Pipeline: Feature Engineering -> Preprocessing
    # Note: FunctionTransformer needs to handle the dataframe input
    fe_transformer = FunctionTransformer(feature_engineering, validate=False)
    
    # We can't put FE in Pipeline easily if it changes columns dynamically before ColumnTransformer
    # So we will apply FE manually first, then fit Pipeline.
    # Or we can use a custom class. For simplicity, we'll apply FE manually in the loop.
    
    return preprocessor, numeric_features, categorical_features

# ==========================================
# PHASE 3: BASELINE MODELS
# ==========================================

def train_baselines(X_train_fe, y_train, X_val_fe, y_val, preprocessor):
    print("\n" + "="*40)
    print("PHASE 3: BASELINE MODELS")
    print("="*40)
    
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    }
    
    results = []
    
    for name, model in models.items():
        # Create full pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train_fe, y_train)
        
        # Predict Probabilities
        y_val_proba = clf.predict_proba(X_val_fe)[:, 1]
        
        # Optimize Threshold for Accuracy (while Recall >= 0.6)
        best_acc = 0
        best_thresh = 0.5
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        for t in thresholds:
            y_pred_t = (y_val_proba >= t).astype(int)
            rec = recall_score(y_val, y_pred_t)
            acc = accuracy_score(y_val, y_pred_t)
            
            if rec >= 0.60 and acc > best_acc:
                best_acc = acc
                best_thresh = t
        
        # Evaluate at best threshold
        y_pred = (y_val_proba >= best_thresh).astype(int)
        
        acc = accuracy_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_val_proba)
        
        print(f"\n{name}:")
        print(f"  Best Threshold: {best_thresh:.2f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Recall:   {rec:.4f}")
        print(f"  AUC:      {auc:.4f}")
        
        results.append({'Model': name, 'Accuracy': acc, 'Recall': rec, 'AUC': auc, 'Threshold': best_thresh})
        
    return pd.DataFrame(results)

# ==========================================
# PHASE 4: OPTIMIZATION
# ==========================================

def optimize_lightgbm(X_train_fe, y_train, preprocessor):
    print("\n" + "="*40)
    print("PHASE 4: OPTIMIZATION (LightGBM)")
    print("="*40)
    
    # Transform data once for Optuna speed
    X_train_trans = preprocessor.fit_transform(X_train_fe)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMClassifier(**params)
        
        # 5-fold CV
        scores = cross_val_score(model, X_train_trans, y_train, cv=5, scoring='accuracy')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # 50 trials for speed
    
    print(f"\nBest Params: {study.best_params}")
    print(f"Best CV Accuracy: {study.best_value:.4f}")
    
    return study.best_params

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # 1. Load
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data('data/raw/framingham_raw.csv')
    
    # 2. Feature Engineering
    print("\nApplying Feature Engineering...")
    X_train_fe = feature_engineering(X_train)
    X_val_fe = feature_engineering(X_val)
    X_test_fe = feature_engineering(X_test)
    
    # Build Preprocessor
    preprocessor, _, _ = build_pipeline(X_train)
    
    # 3. Baselines
    results = train_baselines(X_train_fe, y_train, X_val_fe, y_val, preprocessor)
    print("\nBaseline Results:")
    print(results)
    
    # 4. Optimization
    best_params = optimize_lightgbm(X_train_fe, y_train, preprocessor)
    
    # Train Final Model
    print("\nTraining Final Model...")
    final_model = LGBMClassifier(**best_params)
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', final_model)])
    
    # Combine Train + Val for final training
    X_full = pd.concat([X_train_fe, X_val_fe])
    y_full = pd.concat([y_train, y_val])
    
    final_pipeline.fit(X_full, y_full)
    
    # Threshold Tuning on Test (Simulated Validation) - Wait, prompt says:
    # "Retrain final LightGBM on TRAIN + VALIDATION... Apply chosen threshold to TEST"
    # So we need to choose threshold on VALIDATION *before* combining.
    
    # Let's re-eval on Val to find threshold
    final_pipeline_val = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('classifier', LGBMClassifier(**best_params))])
    final_pipeline_val.fit(X_train_fe, y_train)
    y_val_proba = final_pipeline_val.predict_proba(X_val_fe)[:, 1]
    
    best_thresh = 0.1 # Start low to ensure some recall
    best_acc = 0
    
    for t in np.arange(0.05, 0.9, 0.01):
        y_pred_t = (y_val_proba >= t).astype(int)
        rec = recall_score(y_val, y_pred_t)
        acc = accuracy_score(y_val, y_pred_t)
        
        # We want max accuracy, BUT recall must be decent (e.g. >= 0.60)
        # If no threshold meets 0.60, we take the one with highest recall?
        # Or we prioritize recall.
        # Let's try to find max accuracy among those with recall >= 0.60
        if rec >= 0.60:
            if acc > best_acc:
                best_acc = acc
                best_thresh = t
        
    # Fallback: If no threshold met recall >= 0.60, pick threshold that gives recall closest to 0.60
    if best_acc == 0:
        print("Warning: No threshold met Recall >= 0.60. Maximizing Recall instead.")
        best_rec = 0
        for t in np.arange(0.05, 0.9, 0.01):
            y_pred_t = (y_val_proba >= t).astype(int)
            rec = recall_score(y_val, y_pred_t)
            if rec > best_rec:
                best_rec = rec
                best_thresh = t
            
    print(f"\nChosen Threshold (from Val): {best_thresh:.2f}")
    
    # 5. Final Evaluation
    print("\n" + "="*40)
    print("PHASE 5: FINAL EVALUATION")
    print("="*40)
    
    y_test_proba = final_pipeline.predict_proba(X_test_fe)[:, 1]
    y_test_pred = (y_test_proba >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Final Test Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Risk Categorization
    print("\nRisk Categorization Analysis:")
    risk_cats = []
    for p in y_test_proba:
        if p < 0.10:
            risk_cats.append('Low')
        elif p < 0.20:
            risk_cats.append('Moderate')
        else:
            risk_cats.append('High')
            
    df_res = pd.DataFrame({'Risk': risk_cats, 'Actual': y_test})
    summary = df_res.groupby('Risk')['Actual'].agg(['count', 'mean'])
    summary['mean'] = summary['mean'] * 100 # Convert to %
    print(summary)
    
    # Save Artifacts
    joblib.dump(final_pipeline, 'models/risk_model_pipeline.pkl')
    print("\nSaved pipeline to models/risk_model_pipeline.pkl")

if __name__ == "__main__":
    main()
