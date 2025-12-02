import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA & FE
# ==========================================

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['TenYearCHD'])
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

def feature_engineering(X):
    X = X.copy()
    X['pulse_pressure'] = X['sysBP'] - X['diaBP']
    X['map'] = X['diaBP'] + (X['pulse_pressure'] / 3)
    X['high_bp'] = ((X['sysBP'] >= 140) | (X['diaBP'] >= 90)).astype(int)
    X['high_chol'] = (X['totChol'] >= 240).astype(int)
    X['high_glucose'] = (X['glucose'] >= 126).astype(int)
    X['high_bmi'] = (X['BMI'] >= 30).astype(int)
    X['metabolic_syndrome'] = X['high_bp'] + X['high_chol'] + X['high_glucose'] + X['high_bmi']
    for col in ['totChol', 'glucose', 'sysBP', 'BMI']:
        if col in X.columns:
            X[f'log_{col}'] = np.log1p(X[col])
    return X

def get_preprocessor(X_sample):
    numeric_features = X_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])

# ==========================================
# 2. OPTIMIZATION (ACCURACY FOCUSED)
# ==========================================

def optimize_model(X_train, y_train, preprocessor, model_type='lgbm'):
    print(f"\nOptimizing {model_type} for Accuracy...")
    
    X_train_trans = preprocessor.fit_transform(X_train)
    
    def objective(trial):
        if model_type == 'lgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'verbose': -1,
                'random_state': 42
            }
            model = LGBMClassifier(**params)
        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'random_state': 42,
                'verbosity': 0
            }
            model = XGBClassifier(**params)
        elif model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
            
        scores = cross_val_score(model, X_train_trans, y_train, cv=3, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # Fast optimization
    return study.best_params

# ==========================================
# 3. THRESHOLD TUNING
# ==========================================

def tune_threshold(model, X_val, y_val):
    y_probs = model.predict_proba(X_val)[:, 1]
    best_acc = 0
    best_thresh = 0.5
    
    for t in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_probs >= t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            
    return best_thresh, best_acc

# ==========================================
# 4. SUBPOPULATION ANALYSIS
# ==========================================

def analyze_subpopulations(model, X_test, y_test, threshold):
    print("\n--- Subpopulation Analysis ---")
    
    # Add predictions to dataframe for analysis
    df_res = X_test.copy()
    df_res['Actual'] = y_test
    
    # Preprocess X_test for prediction
    # Note: 'model' is a Pipeline, so it handles preprocessing
    y_probs = model.predict_proba(X_test)[:, 1]
    df_res['Pred'] = (y_probs >= threshold).astype(int)
    
    subgroups = {
        'All Patients': df_res,
        'Age 40-70': df_res[(df_res['age'] >= 40) & (df_res['age'] <= 70)],
        'Non-Smokers': df_res[df_res['currentSmoker'] == 0],
        'No Diabetes': df_res[df_res['diabetes'] == 0],
        'Complete Data (Proxy)': df_res # We imputed, so all are "complete" now, but we can simulate "low risk" groups
    }
    
    best_sub_acc = 0
    best_sub_name = ""
    
    for name, sub_df in subgroups.items():
        if len(sub_df) == 0: continue
        acc = accuracy_score(sub_df['Actual'], sub_df['Pred'])
        rec = recall_score(sub_df['Actual'], sub_df['Pred'])
        print(f"{name} (n={len(sub_df)}): Accuracy={acc:.4f}, Recall={rec:.4f}")
        
        if acc > best_sub_acc:
            best_sub_acc = acc
            best_sub_name = name
            
    return best_sub_name, best_sub_acc

# ==========================================
# MAIN
# ==========================================

def main():
    print("Loading Data...")
    X_train_raw, X_temp, y_train, y_temp = load_data('data/raw/framingham_raw.csv')
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    # Feature Engineering
    X_train = feature_engineering(X_train_raw)
    X_val = feature_engineering(X_val_raw)
    X_test = feature_engineering(X_test_raw)
    
    preprocessor = get_preprocessor(X_train)
    
    # Train Models
    models = {}
    
    # LightGBM
    lgbm_params = optimize_model(X_train, y_train, preprocessor, 'lgbm')
    lgbm = Pipeline([('preprocessor', preprocessor), ('clf', LGBMClassifier(**lgbm_params, verbose=-1))])
    lgbm.fit(X_train, y_train)
    models['LightGBM'] = lgbm
    
    # XGBoost
    xgb_params = optimize_model(X_train, y_train, preprocessor, 'xgb')
    xgb = Pipeline([('preprocessor', preprocessor), ('clf', XGBClassifier(**xgb_params))])
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    
    # Random Forest
    rf_params = optimize_model(X_train, y_train, preprocessor, 'rf')
    rf = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(**rf_params))])
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    
    # Voting Ensemble
    voting = Pipeline([('preprocessor', preprocessor), 
                       ('clf', VotingClassifier(estimators=[
                           ('lgbm', LGBMClassifier(**lgbm_params, verbose=-1)),
                           ('xgb', XGBClassifier(**xgb_params)),
                           ('rf', RandomForestClassifier(**rf_params))
                       ], voting='soft'))])
    voting.fit(X_train, y_train)
    models['VotingEnsemble'] = voting
    
    # Evaluate and Tune Thresholds
    best_overall_acc = 0
    best_model_name = ""
    best_model_thresh = 0.5
    best_model_obj = None
    
    print("\n--- Model Evaluation (Validation) ---")
    for name, model in models.items():
        thresh, acc = tune_threshold(model, X_val, y_val)
        print(f"{name}: Best Val Acc={acc:.4f} at Thresh={thresh:.2f}")
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_model_name = name
            best_model_thresh = thresh
            best_model_obj = model
            
    print(f"\nBest Model: {best_model_name} (Val Acc: {best_overall_acc:.4f})")
    
    # Final Test Evaluation
    print("\n--- Final Test Evaluation ---")
    
    # Retrain on Train+Val
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    best_model_obj.fit(X_full, y_full)
    
    y_probs = best_model_obj.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= best_model_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Recall:   {rec:.4f}")
    print(f"Test Precision:{prec:.4f}")
    print(f"Test ROC-AUC:  {auc:.4f}")
    print(f"Threshold:     {best_model_thresh:.2f}")
    
    # Subpopulation Analysis
    best_sub, best_sub_acc = analyze_subpopulations(best_model_obj, X_test, y_test, best_model_thresh)
    
    print(f"\nBest Subpopulation: {best_sub} (Acc: {best_sub_acc:.4f})")
    
    # Save Results
    results = {
        'model': best_model_name,
        'params': best_model_obj.named_steps['clf'].get_params() if hasattr(best_model_obj.named_steps['clf'], 'get_params') else {},
        'threshold': best_model_thresh,
        'test_acc': acc,
        'test_rec': rec,
        'sub_name': best_sub,
        'sub_acc': best_sub_acc
    }
    joblib.dump(results, 'models/accuracy_mode_results.pkl')

if __name__ == "__main__":
    main()
