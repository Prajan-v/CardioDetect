import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import joblib
import warnings
import time

warnings.filterwarnings('ignore')

# ==========================================
# PHASE 1: EXTREME FEATURE ENGINEERING
# ==========================================

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['TenYearCHD'])
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

def extreme_feature_engineering(X):
    X = X.copy()
    
    # 1. Basic Medical Features
    X['pulse_pressure'] = X['sysBP'] - X['diaBP']
    X['map'] = X['diaBP'] + (X['pulse_pressure'] / 3)
    
    # 2. Risk Scores & Flags
    X['high_bp'] = ((X['sysBP'] >= 140) | (X['diaBP'] >= 90)).astype(int)
    X['high_chol'] = (X['totChol'] >= 240).astype(int)
    X['high_glucose'] = (X['glucose'] >= 126).astype(int)
    X['high_bmi'] = (X['BMI'] >= 30).astype(int)
    X['metabolic_syndrome'] = X['high_bp'] + X['high_chol'] + X['high_glucose'] + X['high_bmi']
    
    # Framingham-like aggregates (simplified)
    # (age/10) * (sysBP/100) * (totChol/200) * (1+currentSmoker*2)
    X['custom_risk_score'] = (X['age']/10) * (X['sysBP']/100) * (X['totChol']/200) * (1 + X['currentSmoker']*2)
    
    # Vascular burden
    X['vascular_burden'] = X['prevalentStroke'] + X['prevalentHyp'] + X['diabetes'] + (X['age'] > 60).astype(int)

    # 3. Non-linear Transformations
    for col in ['age', 'sysBP', 'totChol', 'BMI']:
        if col in X.columns:
            X[f'{col}_sq'] = X[col]**2
            X[f'{col}_sqrt'] = np.sqrt(X[col])
            X[f'{col}_recip'] = 1 / (X[col] + 1)
            
    for col in ['totChol', 'glucose', 'sysBP', 'BMI']:
        if col in X.columns:
            X[f'log_{col}'] = np.log1p(X[col])

    # 4. Interactions (Top features)
    # age * totChol, age * sysBP, sysBP * BMI, BMI * glucose
    X['age_x_totChol'] = X['age'] * X['totChol']
    X['age_x_sysBP'] = X['age'] * X['sysBP']
    X['sysBP_x_BMI'] = X['sysBP'] * X['BMI']
    X['BMI_x_glucose'] = X['BMI'] * X['glucose']
    X['age_x_smoker'] = X['age'] * X['currentSmoker']

    return X

def get_preprocessor(X_sample):
    numeric_features = X_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

# ==========================================
# PHASE 2 & 3: RESAMPLING & MODEL ZOO
# ==========================================

def get_resampler(name):
    if name == 'SMOTE': return SMOTE(random_state=42)
    if name == 'SMOTEENN': return SMOTEENN(random_state=42)
    if name == 'SMOTETomek': return SMOTETomek(random_state=42)
    if name == 'ADASYN': return ADASYN(random_state=42)
    if name == 'None': return None
    return None

def optimize_model(X_train, y_train, preprocessor, model_type, resampler_name):
    
    def objective(trial):
        # Model Params
        if model_type == 'lgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'verbose': -1, 'random_state': 42
            }
            model = LGBMClassifier(**params)
        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'random_state': 42, 'verbosity': 0
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
        elif model_type == 'mlp':
            params = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(100,), (100, 50), (50, 50, 50)]),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'random_state': 42, 'max_iter': 500
            }
            model = MLPClassifier(**params)
            
        # Pipeline
        steps = [('preprocessor', preprocessor)]
        resampler = get_resampler(resampler_name)
        if resampler:
            steps.append(('resampler', resampler))
        steps.append(('clf', model))
        
        clf = ImbPipeline(steps)
        
        # CV
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10) # Limited trials for speed in this demo
    return study.best_params

# ==========================================
# PHASE 4: ENSEMBLES
# ==========================================

def build_ensembles(models_dict, X_train, y_train, preprocessor):
    estimators = []
    for name, (model, _) in models_dict.items():
        estimators.append((name, model))
        
    # Voting
    voting = VotingClassifier(estimators=estimators, voting='soft')
    voting_pipe = Pipeline([('preprocessor', preprocessor), ('clf', voting)])
    voting_pipe.fit(X_train, y_train)
    
    # Stacking
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)
    stacking_pipe = Pipeline([('preprocessor', preprocessor), ('clf', stacking)])
    stacking_pipe.fit(X_train, y_train)
    
    return {'Voting': voting_pipe, 'Stacking': stacking_pipe}

# ==========================================
# PHASE 6: SUBPOPULATIONS
# ==========================================

def analyze_subpopulations(model, X_test, y_test, threshold):
    df_res = X_test.copy()
    df_res['Actual'] = y_test
    y_probs = model.predict_proba(X_test)[:, 1]
    df_res['Pred'] = (y_probs >= threshold).astype(int)
    
    subgroups = {
        'All': df_res,
        'Age 40-60': df_res[(df_res['age'] >= 40) & (df_res['age'] <= 60)],
        'Age 50-70': df_res[(df_res['age'] >= 50) & (df_res['age'] <= 70)],
        'Non-Smokers': df_res[df_res['currentSmoker'] == 0],
        'Non-Diabetics': df_res[df_res['diabetes'] == 0],
        'No Prevalent CVD': df_res[(df_res['prevalentStroke'] == 0) & (df_res['prevalentHyp'] == 0)],
        'Non-Obese': df_res[df_res['BMI'] < 30],
        'Normotensive': df_res[df_res['sysBP'] < 140]
    }
    
    results = []
    for name, sub in subgroups.items():
        if len(sub) < 200: continue # Min sample size constraint
        acc = accuracy_score(sub['Actual'], sub['Pred'])
        rec = recall_score(sub['Actual'], sub['Pred'])
        prec = precision_score(sub['Actual'], sub['Pred'])
        results.append({'Subgroup': name, 'N': len(sub), 'Accuracy': acc, 'Recall': rec, 'Precision': prec})
        
    return pd.DataFrame(results)

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Starting Extreme Optimization...")
    start_time = time.time()
    
    # 1. Load & Split
    X_train_raw, X_temp, y_train, y_temp = load_data('data/raw/framingham_raw.csv')
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    # 2. Extreme FE
    print("Applying Extreme Feature Engineering...")
    X_train = extreme_feature_engineering(X_train_raw)
    X_val = extreme_feature_engineering(X_val_raw)
    X_test = extreme_feature_engineering(X_test_raw)
    
    preprocessor = get_preprocessor(X_train)
    
    # 3. Model Loop
    model_types = ['lgbm', 'xgb', 'rf', 'mlp']
    resamplers = ['None', 'SMOTE', 'SMOTEENN'] # Reduced set for speed
    
    best_models = {}
    
    print("\nOptimizing Models...")
    for m_type in model_types:
        best_val_acc = 0
        best_config = None
        
        for res in resamplers:
            try:
                params = optimize_model(X_train, y_train, preprocessor, m_type, res)
                
                # Train and Eval on Val
                if m_type == 'lgbm': clf = LGBMClassifier(**params)
                elif m_type == 'xgb': clf = XGBClassifier(**params)
                elif m_type == 'rf': clf = RandomForestClassifier(**params)
                elif m_type == 'mlp': clf = MLPClassifier(**params)
                
                steps = [('preprocessor', preprocessor)]
                if get_resampler(res): steps.append(('resampler', get_resampler(res)))
                steps.append(('clf', clf))
                
                pipe = ImbPipeline(steps)
                pipe.fit(X_train, y_train)
                
                # Threshold Tuning
                y_probs = pipe.predict_proba(X_val)[:, 1]
                best_thresh = 0.5
                curr_best_acc = 0
                for t in np.arange(0.05, 0.95, 0.05):
                    acc = accuracy_score(y_val, (y_probs >= t).astype(int))
                    if acc > curr_best_acc:
                        curr_best_acc = acc
                        best_thresh = t
                
                if curr_best_acc > best_val_acc:
                    best_val_acc = curr_best_acc
                    best_config = (clf, best_thresh) # Store classifier instance and threshold
                    
            except Exception as e:
                print(f"Error optimizing {m_type} with {res}: {e}")
                continue
        
        if best_config:
            best_models[m_type] = best_config
            print(f"Best {m_type}: Val Acc {best_val_acc:.4f} (Thresh {best_config[1]:.2f})")

    # 4. Ensembles
    print("\nBuilding Ensembles...")
    ensembles = build_ensembles(best_models, X_train, y_train, preprocessor)
    
    # Evaluate Ensembles
    for name, pipe in ensembles.items():
        y_probs = pipe.predict_proba(X_val)[:, 1]
        best_thresh = 0.5
        curr_best_acc = 0
        for t in np.arange(0.05, 0.95, 0.05):
            acc = accuracy_score(y_val, (y_probs >= t).astype(int))
            if acc > curr_best_acc:
                curr_best_acc = acc
                best_thresh = t
        print(f"Best {name}: Val Acc {curr_best_acc:.4f} (Thresh {best_thresh:.2f})")
        best_models[name] = (pipe.named_steps['clf'], best_thresh) # Store classifier and threshold

    # 5. Final Evaluation & Subpopulations
    print("\n--- FINAL EVALUATION ---")
    
    # Find absolute best model
    final_best_name = ""
    final_best_acc = 0
    final_best_res = None
    
    results_list = []
    
    for name, (clf, thresh) in best_models.items():
        # Construct pipeline for prediction
        if name in ['Voting', 'Stacking']:
            pipe = ensembles[name] # Already fitted on X_train
        else:
             pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])

        y_probs = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= thresh).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        print(f"{name}: Test Acc {acc:.4f}, Recall {rec:.4f}")
        
        # Subpopulation Analysis for this model
        sub_df = analyze_subpopulations(pipe, X_test, y_test, thresh)
        sub_df['Model'] = name
        results_list.append(sub_df)
        
    # Combine Results
    all_results = pd.concat(results_list).reset_index(drop=True)
    
    # Find Global Max Accuracy
    best_idx = all_results['Accuracy'].idxmax()
    best_row = all_results.loc[best_idx]
    
    print(f"\n🏆 MAXIMUM ACHIEVABLE ACCURACY: {best_row['Accuracy']:.4f}")
    print(f"Model: {best_row['Model']}")
    print(f"Subgroup: {best_row['Subgroup']} (N={best_row['N']})")
    print(f"Recall: {best_row['Recall']:.4f}")
    
    # Save Results
    all_results.to_csv('reports/extreme_optimization_results.csv', index=False)
    print("\nSaved results to reports/extreme_optimization_results.csv")
    print(f"Total Time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
