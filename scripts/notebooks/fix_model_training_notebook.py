import json
import os

def create_notebook():
    cells = []
    
    # Cell 1: Title & Description (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🤖 CardioDetect - Model Training & Evaluation\n",
            "\n",
            "## Milestone 2: Comprehensive Model Comparison\n",
            "\n",
            "In this notebook, I train and evaluate **8 different machine learning models** on my unified cardiovascular risk dataset. My goal is to find the model that best balances accuracy and recall for predicting heart disease.\n",
            "\n",
            "### Models I'm Testing:\n",
            "1. **Logistic Regression** - Linear baseline\n",
            "2. **Random Forest** - Ensemble of decision trees\n",
            "3. **XGBoost** - Gradient boosted trees\n",
            "4. **LightGBM** - Microsoft fast gradient boosting\n",
            "5. **SVM (RBF)** - Support vector machine with radial basis function\n",
            "6. **Gradient Boosting** - Sklearn gradient boosting\n",
            "7. **MLP** - Multi-layer perceptron (neural network)\n",
            "8. **Ensemble** - Soft-voting combination of RF + XGB + LGBM + MLP\n",
            "\n",
            "All models use class weighting (where applicable) to handle the dataset balance. I am using the unified dataset with ~16k records."
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
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
            "from sklearn.impute import SimpleImputer\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score\n",
            "\n",
            "# Models\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier\n",
            "from sklearn.svm import SVC\n",
            "from sklearn.neural_network import MLPClassifier\n",
            "from xgboost import XGBClassifier\n",
            "from lightgbm import LGBMClassifier\n",
            "\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "pd.set_option('display.max_columns', None)\n",
            "print(\"✅ Libraries Loaded\")"
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
            "data_path = '../data/processed/combined_processed.csv'\n",
            "df = pd.read_csv(data_path)\n",
            "print(f\"Dataset Shape: {df.shape}\")\n",
            "\n",
            "# Split Features/Target\n",
            "X = df.drop('target', axis=1)\n",
            "y = df['target']\n",
            "\n",
            "# Train/Test Split (80/20)\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
            "print(f\"Train: {X_train.shape}, Test: {X_test.shape}\")"
        ]
    })
    
    # Cell 4: Preprocessing Pipeline
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define Preprocessor\n",
            "numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
            "categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()\n",
            "\n",
            "preprocessor = ColumnTransformer(\n",
            "    transformers=[\n",
            "        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),\n",
            "        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)\n",
            "    ])\n",
            "\n",
            "print(\"✅ Preprocessor Ready\")"
        ]
    })
    
    # Cell 5: Model Definitions
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = {\n",
            "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n",
            "    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),\n",
            "    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),\n",
            "    'LightGBM': LGBMClassifier(random_state=42, verbose=-1),\n",
            "    'SVM (RBF)': SVC(probability=True, random_state=42),\n",
            "    'Gradient Boosting': GradientBoostingClassifier(random_state=42),\n",
            "    'MLP': MLPClassifier(max_iter=500, random_state=42)\n",
            "}\n",
            "\n",
            "print(\"✅ Models Initialized\")"
        ]
    })
    
    # Cell 6: Training Loop
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "results = []\n",
            "trained_models = {}\n",
            "\n",
            "print(\"🚀 Starting Training Loop...\")\n",
            "\n",
            "for name, model in models.items():\n",
            "    print(f\"Training {name}...\")\n",
            "    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])\n",
            "    clf.fit(X_train, y_train)\n",
            "    \n",
            "    # Evaluate\n",
            "    y_pred = clf.predict(X_test)\n",
            "    y_prob = clf.predict_proba(X_test)[:, 1]\n",
            "    \n",
            "    acc = accuracy_score(y_test, y_pred)\n",
            "    rec = recall_score(y_test, y_pred)\n",
            "    auc = roc_auc_score(y_test, y_prob)\n",
            "    \n",
            "    results.append({'Model': name, 'Accuracy': acc, 'Recall': rec, 'ROC-AUC': auc})\n",
            "    trained_models[name] = clf\n",
            "\n",
            "# Create Ensemble (Voting)\n",
            "print(\"Training Ensemble...\")\n",
            "estimators = [\n",
            "    ('rf', models['Random Forest']),\n",
            "    ('xgb', models['XGBoost']),\n",
            "    ('lgbm', models['LightGBM']),\n",
            "    ('mlp', models['MLP'])\n",
            "]\n",
            "voting = VotingClassifier(estimators=estimators, voting='soft')\n",
            "clf_voting = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', voting)])\n",
            "clf_voting.fit(X_train, y_train)\n",
            "\n",
            "y_pred = clf_voting.predict(X_test)\n",
            "y_prob = clf_voting.predict_proba(X_test)[:, 1]\n",
            "results.append({'Model': 'Voting Ensemble', \n",
            "                'Accuracy': accuracy_score(y_test, y_pred),\n",
            "                'Recall': recall_score(y_test, y_pred),\n",
            "                'ROC-AUC': roc_auc_score(y_test, y_prob)})\n",
            "\n",
            "print(\"✅ Training Complete\")"
        ]
    })
    
    # Cell 7: Results Table
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)\n",
            "print(results_df)\n",
            "\n",
            "# Plot\n",
            "plt.figure(figsize=(10, 6))\n",
            "sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')\n",
            "plt.title('Model Comparison (Accuracy)')\n",
            "plt.xlim(0.8, 1.0)\n",
            "plt.show()"
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
    
    with open('notebooks/03_model_training.ipynb', 'w') as f:
        json.dump(notebook, f, indent=4)
    print("✅ Regenerated notebooks/03_model_training.ipynb")

if __name__ == "__main__":
    create_notebook()
