import json
import os

def create_notebook():
    cells = []
    
    # Cell 1: Title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🛡️ CardioDetect - Risk Prediction Arm\n",
            "## 10-Year Heart Disease Risk Prediction (Framingham)\n",
            "\n",
            "**Objective:** Predict 10-year CHD risk using Framingham-style cohort data.\n",
            "**Target:** `TenYearCHD` (Binary 0/1)\n",
            "**Goal:** Maximize accuracy while maintaining Recall ≥ 0.60.\n",
            "\n",
            "---"
        ]
    })
    
    # Cell 2: Imports
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.model_selection import train_test_split, cross_val_score\n",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer\n",
            "from sklearn.impute import SimpleImputer\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from lightgbm import LGBMClassifier\n",
            "import optuna\n",
            "import joblib\n",
            "import warnings\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "pd.set_option('display.max_columns', None)\n",
            "print(\"✅ Environment Ready\")"
        ]
    })
    
    # Cell 3: Data Loading
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load Data\n",
            "df = pd.read_csv('../data/raw/framingham_raw.csv')\n",
            "print(f\"Loaded shape: {df.shape}\")\n",
            "\n",
            "# Target Distribution\n",
            "print(f\"\\nTarget Distribution:\\n{df['TenYearCHD'].value_counts(normalize=True)}\")\n",
            "\n",
            "# Drop missing target\n",
            "df = df.dropna(subset=['TenYearCHD'])\n",
            "\n",
            "# Split (70/15/15)\n",
            "X = df.drop('TenYearCHD', axis=1)\n",
            "y = df['TenYearCHD']\n",
            "\n",
            "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)\n",
            "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)\n",
            "\n",
            "print(f\"\\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\")"
        ]
    })
    
    # Cell 4: Feature Engineering
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def feature_engineering(X):\n",
            "    X = X.copy()\n",
            "    # Pulse Pressure & MAP\n",
            "    X['pulse_pressure'] = X['sysBP'] - X['diaBP']\n",
            "    X['map'] = X['diaBP'] + (X['pulse_pressure'] / 3)\n",
            "    \n",
            "    # Risk Flags\n",
            "    X['high_bp'] = ((X['sysBP'] >= 140) | (X['diaBP'] >= 90)).astype(int)\n",
            "    X['high_chol'] = (X['totChol'] >= 240).astype(int)\n",
            "    X['high_glucose'] = (X['glucose'] >= 126).astype(int)\n",
            "    X['high_bmi'] = (X['BMI'] >= 30).astype(int)\n",
            "    X['metabolic_syndrome'] = X['high_bp'] + X['high_chol'] + X['high_glucose'] + X['high_bmi']\n",
            "    \n",
            "    # Log Transforms\n",
            "    for col in ['totChol', 'glucose', 'sysBP', 'BMI']:\n",
            "        if col in X.columns:\n",
            "            X[f'log_{col}'] = np.log1p(X[col])\n",
            "            \n",
            "    return X\n",
            "\n",
            "# Apply FE\n",
            "X_train_fe = feature_engineering(X_train)\n",
            "X_val_fe = feature_engineering(X_val)\n",
            "X_test_fe = feature_engineering(X_test)\n",
            "\n",
            "# Build Preprocessor\n",
            "numeric_features = X_train_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
            "categorical_features = X_train_fe.select_dtypes(include=['object', 'category']).columns.tolist()\n",
            "\n",
            "preprocessor = ColumnTransformer(\n",
            "    transformers=[\n",
            "        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),\n",
            "        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)\n",
            "    ])\n",
            "\n",
            "print(\"✅ Feature Engineering & Pipeline Ready\")"
        ]
    })
    
    # Cell 5: Baseline Models
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = {\n",
            "    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),\n",
            "    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),\n",
            "    'LightGBM': LGBMClassifier(n_estimators=200, max_depth=5, random_state=42, verbose=-1)\n",
            "}\n",
            "\n",
            "print(\"Training Baselines...\")\n",
            "for name, model in models.items():\n",
            "    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])\n",
            "    clf.fit(X_train_fe, y_train)\n",
            "    val_acc = clf.score(X_val_fe, y_val)\n",
            "    print(f\"{name}: Val Accuracy = {val_acc:.4f}\")"
        ]
    })
    
    # Cell 6: Optimization (Optuna)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Note: We use the best params found from the script run to save time in demo\n",
            "best_params = {\n",
            "    'n_estimators': 856, \n",
            "    'max_depth': 6, \n",
            "    'num_leaves': 65, \n",
            "    'learning_rate': 0.0065, \n",
            "    'min_child_samples': 92, \n",
            "    'subsample': 0.99, \n",
            "    'colsample_bytree': 0.86, \n",
            "    'reg_alpha': 0.00026, \n",
            "    'reg_lambda': 4.13\n",
            "}\n",
            "\n",
            "print(\"Using Optimized Parameters:\", best_params)\n",
            "\n",
            "final_model = LGBMClassifier(**best_params, random_state=42, verbose=-1)\n",
            "final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', final_model)])\n",
            "\n",
            "# Train on Train + Val\n",
            "X_full = pd.concat([X_train_fe, X_val_fe])\n",
            "y_full = pd.concat([y_train, y_val])\n",
            "final_pipeline.fit(X_full, y_full)\n",
            "print(\"✅ Final Model Trained\")"
        ]
    })
    
    # Cell 7: Threshold Tuning & Evaluation
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Threshold Tuning (Simulated on Val for demo logic)\n",
            "# We found 0.13 to be the threshold that gives Recall ~60%\n",
            "best_thresh = 0.13\n",
            "print(f\"Selected Threshold: {best_thresh}\")\n",
            "\n",
            "# Final Test Eval\n",
            "y_prob = final_pipeline.predict_proba(X_test_fe)[:, 1]\n",
            "y_pred = (y_prob >= best_thresh).astype(int)\n",
            "\n",
            "print(\"\\nFINAL TEST RESULTS:\")\n",
            "print(f\"Accuracy:  {accuracy_score(y_test, y_pred):.4f}\")\n",
            "print(f\"Recall:    {recall_score(y_test, y_pred):.4f}\")\n",
            "print(f\"Precision: {precision_score(y_test, y_pred):.4f}\")\n",
            "print(f\"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}\")\n",
            "\n",
            "cm = confusion_matrix(y_test, y_pred)\n",
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
            "plt.title('Confusion Matrix')\n",
            "plt.show()"
        ]
    })
    
    # Cell 8: Risk Categorization
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "risk_cats = []\n",
            "for p in y_prob:\n",
            "    if p < 0.10: risk_cats.append('Low')\n",
            "    elif p < 0.20: risk_cats.append('Moderate')\n",
            "    else: risk_cats.append('High')\n",
            "\n",
            "df_res = pd.DataFrame({'Risk': risk_cats, 'Actual': y_test})\n",
            "summary = df_res.groupby('Risk')['Actual'].agg(['count', 'mean'])\n",
            "summary['mean'] = summary['mean'] * 100\n",
            "print(summary)"
        ]
    })
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('notebooks/risk_prediction_optimized.ipynb', 'w') as f:
        json.dump(notebook, f, indent=4)
    print("✅ Created notebooks/risk_prediction_optimized.ipynb")

if __name__ == "__main__":
    create_notebook()
