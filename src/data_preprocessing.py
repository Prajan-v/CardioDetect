"""
CardioDetect - Data Preprocessing Pipeline
==========================================

This module handles:
1. Automatic detection of heart disease target columns across diverse datasets
2. Loading and harmonizing data from Framingham, NHANES, and custom sources
3. Building a unified risk prediction dataset with standardized features
4. Creating stratified train/val/test splits

Author: CardioDetect Project
Date: November 2025
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Target column candidates (in order of preference)
TARGET_COLUMN_CANDIDATES = [
    'TenYearCHD', 'TenYearCHD_risk', 'tenyearchd', 
    'CHD', 'CVD', 'cvd', 'chd',
    'target', 'outcome', 'label', 'class',
    'heart_disease', 'heartdisease', 'disease',
    'num', 'condition'  # UCI format
]

# Feature mapping: standardize column names across datasets
FEATURE_MAPPING = {
    # Age
    'age': 'age',
    'AGE': 'age',
    'RIDAGEYR': 'age',
    
    # Sex (0=Female, 1=Male)
    'sex': 'sex',
    'SEX': 'sex',
    'male': 'sex',
    'RIAGENDR': 'sex',
    
    # Blood Pressure
    'sysBP': 'systolic_bp',
    'sysbp': 'systolic_bp',
    'trestbps': 'systolic_bp',
    'BPXSY1': 'systolic_bp',
    
    'diaBP': 'diastolic_bp',
    'diabp': 'diastolic_bp',
    'BPXDI1': 'diastolic_bp',
    
    # Cholesterol
    'totChol': 'total_cholesterol',
    'totchol': 'total_cholesterol',
    'chol': 'total_cholesterol',
    'LBXTC': 'total_cholesterol',
    
    'HDLchol': 'hdl_cholesterol',
    'hdl': 'hdl_cholesterol',
    'LBDHDD': 'hdl_cholesterol',
    
    'LDLchol': 'ldl_cholesterol',
    'ldl': 'ldl_cholesterol',
    'LBDLDL': 'ldl_cholesterol',
    
    # Glucose
    'glucose': 'fasting_glucose',
    'fbs': 'fasting_glucose',
    'LBXGLU': 'fasting_glucose',
    
    # BMI
    'BMI': 'bmi',
    'bmi': 'bmi',
    'BMXBMI': 'bmi',
    
    # Smoking
    'currentSmoker': 'smoking',
    'cigsPerDay': 'smoking',
    'SMQ020': 'smoking',
    
    # Diabetes
    'diabetes': 'diabetes',
    'prevalentDiab': 'diabetes',
    'DIQ010': 'diabetes',
    
    # Heart Rate
    'heartRate': 'heart_rate',
    'thalach': 'heart_rate',
    'BPXPLS': 'heart_rate',
    
    # Medications
    'BPMeds': 'bp_meds',
    'prevalentHyp': 'hypertension',
}


# ============================================================================
# FUNCTION 1: SCAN AND INVENTORY RAW DATA
# ============================================================================

def scan_raw_datasets(raw_dir='./data/raw/'):
    """
    Scan all CSV files in the raw directory and create an inventory.
    
    Returns:
        dict: {filename: {'path': str, 'shape': tuple, 'columns': list}}
    """
    print("="*80)
    print("SCANNING RAW DATA DIRECTORY")
    print("="*80)
    
    inventory = {}
    raw_path = Path(raw_dir)
    
    # Find all CSV files (including in subdirectories)
    csv_files = list(raw_path.rglob('*.csv'))
    
    print(f"\nFound {len(csv_files)} CSV files:\n")
    
    for csv_file in sorted(csv_files):
        try:
            # Get relative path
            rel_path = csv_file.relative_to(raw_path.parent)
            
            # Try to read just the header to get column info
            df_sample = pd.read_csv(csv_file, nrows=5)
            
            inventory[csv_file.name] = {
                'path': str(csv_file),
                'relative_path': str(rel_path),
                'shape': (None, len(df_sample.columns)),  # rows unknown without full read
                'columns': df_sample.columns.tolist()
            }
            
            print(f"✓ {csv_file.name:40s} | {len(df_sample.columns):3d} columns")
            
        except Exception as e:
            print(f"✗ {csv_file.name:40s} | Error: {str(e)[:50]}")
    
    print(f"\n{'='*80}\n")
    return inventory


# ============================================================================
# FUNCTION 2: DETECT TARGET COLUMN
# ============================================================================

def detect_target_column(df, filename='unknown'):
    """
    Automatically detect the most likely target column for heart disease risk.
    
    Args:
        df: DataFrame to analyze
        filename: Name of the file (for logging)
    
    Returns:
        str: Name of the detected target column, or None
    """
    columns_lower = {col: col.lower() for col in df.columns}
    
    # Check each candidate
    for candidate in TARGET_COLUMN_CANDIDATES:
        candidate_lower = candidate.lower()
        
        # Exact match (case-insensitive)
        for orig_col, lower_col in columns_lower.items():
            if lower_col == candidate_lower:
                # Verify it's binary or categorical
                unique_vals = df[orig_col].dropna().unique()
                if len(unique_vals) <= 5:  # Binary or small categorical
                    print(f"  → Target detected: '{orig_col}' (from {filename})")
                    return orig_col
    
    # If no exact match, look for columns with 'heart', 'disease', 'risk' in name
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['heart', 'disease', 'chd', 'cvd', 'risk']):
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 5:
                print(f"  → Target detected (fuzzy): '{col}' (from {filename})")
                return col
    
    print(f"  ⚠ No target column detected in {filename}")
    return None


# ============================================================================
# FUNCTION 3: LOAD AND STANDARDIZE DATASETS
# ============================================================================

def load_raw_datasets(raw_dir='./data/raw/'):
    """
    Load all relevant heart disease datasets from raw directory.
    
    Returns:
        list of dict: [{'name': str, 'data': DataFrame, 'target_col': str, 'source_type': str}]
    """
    print("="*80)
    print("LOADING RAW DATASETS")
    print("="*80)
    
    datasets = []
    raw_path = Path(raw_dir)
    
    # -------------------------------------------------------------------------
    # FRAMINGHAM DATASETS
    # -------------------------------------------------------------------------
    print("\n[1/3] Loading Framingham datasets...")
    framingham_files = [
        'framingham_mahatir.csv',  # Largest
        'framingham_raw.csv',
        'framingham_alt.csv',
        'framingham_noey.csv',
        'framingham_christofel.csv'
    ]
    
    for fname in framingham_files:
        fpath = raw_path / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                target_col = detect_target_column(df, fname)
                
                if target_col:
                    datasets.append({
                        'name': fname,
                        'data': df,
                        'target_col': target_col,
                        'source_type': 'framingham'
                    })
                    print(f"  ✓ Loaded {fname}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"  ✗ Error loading {fname}: {e}")
    
    # -------------------------------------------------------------------------
    # NHANES DATASETS
    # -------------------------------------------------------------------------
    print("\n[2/3] Loading NHANES 2013-2014 datasets...")
    nhanes_dir = raw_path / 'nhanes_2013_2014'
    
    if nhanes_dir.exists():
        try:
            # Load key NHANES files
            demographic = pd.read_csv(nhanes_dir / 'demographic.csv')
            examination = pd.read_csv(nhanes_dir / 'examination.csv')
            labs = pd.read_csv(nhanes_dir / 'labs.csv')
            questionnaire = pd.read_csv(nhanes_dir / 'questionnaire.csv')
            
            # Merge on SEQN (patient ID)
            nhanes = demographic.merge(examination, on='SEQN', how='inner')
            nhanes = nhanes.merge(labs, on='SEQN', how='inner')
            nhanes = nhanes.merge(questionnaire, on='SEQN', how='inner')
            
            # NHANES doesn't have a direct 10-year CHD outcome
            # We'll create a synthetic risk indicator based on CVD questionnaire
            # For now, we'll skip NHANES for target but use it for features
            print(f"  ✓ Loaded NHANES (merged): {nhanes.shape[0]} rows, {nhanes.shape[1]} columns")
            print(f"  ⚠ NHANES has no direct CHD outcome - will use for feature enrichment only")
            
            # Store for potential feature enrichment
            datasets.append({
                'name': 'nhanes_2013_2014',
                'data': nhanes,
                'target_col': None,  # No outcome
                'source_type': 'nhanes'
            })
            
        except Exception as e:
            print(f"  ✗ Error loading NHANES: {e}")
    
    # -------------------------------------------------------------------------
    # CUSTOM / UCI DATASETS
    # -------------------------------------------------------------------------
    print("\n[3/3] Loading custom/UCI datasets...")
    custom_files = [
        'uci_va.csv',
        'uci_switzerland.csv',
        'kaggle_heart_1190.csv',
        'kaggle_combined_1190.csv',
        'new_data.csv',
        'new_heart.csv',
        'redwan_heart.csv',
        'utkarsh_heart.csv'
    ]
    
    for fname in custom_files:
        fpath = raw_path / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                target_col = detect_target_column(df, fname)
                
                if target_col:
                    datasets.append({
                        'name': fname,
                        'data': df,
                        'target_col': target_col,
                        'source_type': 'custom'
                    })
                    print(f"  ✓ Loaded {fname}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"  ✗ Error loading {fname}: {e}")
    
    print(f"\n{'='*80}")
    print(f"Total datasets loaded: {len(datasets)}")
    print(f"{'='*80}\n")
    
    return datasets


# ============================================================================
# FUNCTION 4: BUILD UNIFIED DATASET
# ============================================================================

def build_unified_dataset(datasets):
    """
    Merge multiple datasets into a unified schema with standardized features.
    
    Args:
        datasets: List of dataset dicts from load_raw_datasets()
    
    Returns:
        DataFrame: Unified dataset with standardized column names
    """
    print("="*80)
    print("BUILDING UNIFIED DATASET")
    print("="*80)
    
    unified_rows = []
    
    for dataset in datasets:
        name = dataset['name']
        df = dataset['data'].copy()
        target_col = dataset['target_col']
        source_type = dataset['source_type']
        
        # Skip datasets without targets (like NHANES for now)
        if target_col is None:
            print(f"\n⊘ Skipping {name} (no target column)")
            continue
        
        print(f"\nProcessing: {name} ({source_type})")
        print(f"  Original shape: {df.shape}")
        
        # Create standardized feature dictionary
        row_data = {}
        
        # Map target to 'risk_target'
        # Standardize to 0 (no disease/low risk) and 1 (disease/high risk)
        target_values = df[target_col].copy()
        
        # Handle different target encodings
        if target_values.dtype == 'object' or set(target_values.dropna().unique()) - {0, 1, 2, 3, 4}:
            # Categorical or multi-class - binarize
            # Assume: 0 or 'no' = healthy, anything else = disease
            target_values = target_values.apply(lambda x: 0 if x in [0, '0', 'no', 'No', 'NO'] else 1)
        else:
            # Numeric - binarize if multi-class
            unique_vals = target_values.dropna().unique()
            if len(unique_vals) > 2:
                # Multi-class (e.g., UCI 0-4 scale) -> binarize to 0 vs 1+
                target_values = (target_values > 0).astype(int)
        
        row_data['risk_target'] = target_values
        
        # Map features using FEATURE_MAPPING
        for orig_col in df.columns:
            if orig_col == target_col:
                continue
            
            # Check if this column should be mapped
            if orig_col in FEATURE_MAPPING:
                std_name = FEATURE_MAPPING[orig_col]
                row_data[std_name] = df[orig_col]
        
        # Create DataFrame for this source
        df_standardized = pd.DataFrame(row_data)
        df_standardized['data_source'] = name
        
        # Remove rows with missing target
        df_standardized = df_standardized.dropna(subset=['risk_target'])
        
        print(f"  Standardized shape: {df_standardized.shape}")
        print(f"  Features mapped: {df_standardized.shape[1] - 2}")  # Exclude target and source
        print(f"  Target distribution: {df_standardized['risk_target'].value_counts().to_dict()}")
        
        unified_rows.append(df_standardized)
    
    # Concatenate all datasets
    unified_df = pd.concat(unified_rows, axis=0, ignore_index=True)
    
    print(f"\n{'='*80}")
    print(f"UNIFIED DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Total rows: {len(unified_df)}")
    print(f"Total features: {unified_df.shape[1] - 2}")  # Exclude target and source
    print(f"\nFeature columns: {[col for col in unified_df.columns if col not in ['risk_target', 'data_source']]}")
    print(f"\nTarget distribution:")
    print(unified_df['risk_target'].value_counts())
    print(f"\nData sources:")
    print(unified_df['data_source'].value_counts())
    print(f"{'='*80}\n")
    
    return unified_df


# ============================================================================
# FUNCTION 5: FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """
    Create additional medically meaningful features from base features.
    
    Args:
        df: DataFrame with standardized features
    
    Returns:
        DataFrame: Enhanced with engineered features
    """
    print("="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df = df.copy()
    original_cols = df.shape[1]
    
    # -------------------------------------------------------------------------
    # DERIVED CARDIOVASCULAR FEATURES
    # -------------------------------------------------------------------------
    
    # Pulse Pressure (strong CHD predictor)
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        print("✓ Created: pulse_pressure")
    
    # Mean Arterial Pressure
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['systolic_bp'] - df['diastolic_bp']) / 3
        print("✓ Created: mean_arterial_pressure")
    
    # LDL/HDL Ratio (atherogenic index)
    if 'ldl_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
        df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)  # Avoid division by zero
        print("✓ Created: ldl_hdl_ratio")
    
    # Total/HDL Ratio
    if 'total_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
        df['total_hdl_ratio'] = df['total_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)
        print("✓ Created: total_hdl_ratio")
    
    # -------------------------------------------------------------------------
    # RISK FLAGS (Binary Indicators)
    # -------------------------------------------------------------------------
    
    # Hypertension (≥140/90 mmHg)
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['hypertension_flag'] = ((df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)).astype(int)
        print("✓ Created: hypertension_flag")
    
    # High Cholesterol (≥240 mg/dL)
    if 'total_cholesterol' in df.columns:
        df['high_cholesterol_flag'] = (df['total_cholesterol'] >= 240).astype(int)
        print("✓ Created: high_cholesterol_flag")
    
    # Prediabetes/Diabetes (≥126 mg/dL fasting glucose)
    if 'fasting_glucose' in df.columns:
        df['high_glucose_flag'] = (df['fasting_glucose'] >= 126).astype(int)
        print("✓ Created: high_glucose_flag")
    
    # Obesity (BMI ≥30)
    if 'bmi' in df.columns:
        df['obesity_flag'] = (df['bmi'] >= 30).astype(int)
        print("✓ Created: obesity_flag")
    
    # Metabolic Syndrome Score (sum of risk flags)
    risk_flags = ['hypertension_flag', 'high_cholesterol_flag', 'high_glucose_flag', 'obesity_flag']
    if all(flag in df.columns for flag in risk_flags):
        df['metabolic_syndrome_score'] = df[risk_flags].sum(axis=1)
        print("✓ Created: metabolic_syndrome_score")
    
    # -------------------------------------------------------------------------
    # AGE & BMI CATEGORIES
    # -------------------------------------------------------------------------
    
    # Age groups (clinical risk bands)
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], 
                                  bins=[0, 40, 50, 60, 70, 120], 
                                  labels=['<40', '40-49', '50-59', '60-69', '70+'],
                                  include_lowest=True)
        # One-hot encode
        age_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
        df = pd.concat([df, age_dummies], axis=1)
        df = df.drop('age_group', axis=1)
        print("✓ Created: age_group categories (one-hot)")
    
    # BMI categories
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(df['bmi'],
                                     bins=[0, 18.5, 25, 30, 100],
                                     labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                                     include_lowest=True)
        bmi_dummies = pd.get_dummies(df['bmi_category'], prefix='bmi_cat')
        df = pd.concat([df, bmi_dummies], axis=1)
        df = df.drop('bmi_category', axis=1)
        print("✓ Created: bmi_category (one-hot)")
    
    # -------------------------------------------------------------------------
    # LOG TRANSFORMS (for skewed distributions)
    # -------------------------------------------------------------------------
    
    for col in ['total_cholesterol', 'fasting_glucose', 'bmi']:
        if col in df.columns:
            # Ensure we're working with numeric data
            df[f'log_{col}'] = df[col].apply(lambda x: np.log1p(x) if pd.notna(x) else np.nan)
            print(f"✓ Created: log_{col}")
    
    # -------------------------------------------------------------------------
    # INTERACTION TERMS
    # -------------------------------------------------------------------------
    
    # Age × Systolic BP (risk increases with age and BP together)
    if 'age' in df.columns and 'systolic_bp' in df.columns:
        df['age_sbp_interaction'] = df['age'] * df['systolic_bp']
        print("✓ Created: age_sbp_interaction")
    
    # BMI × Glucose (diabetes-obesity interaction)
    if 'bmi' in df.columns and 'fasting_glucose' in df.columns:
        df['bmi_glucose_interaction'] = df['bmi'] * df['fasting_glucose']
        print("✓ Created: bmi_glucose_interaction")
    
    # Age × Smoking (compounding risk)
    if 'age' in df.columns and 'smoking' in df.columns:
        df['age_smoking_interaction'] = df['age'] * df['smoking']
        print("✓ Created: age_smoking_interaction")
    
    print(f"\n{'='*80}")
    print(f"Feature engineering complete!")
    print(f"Original features: {original_cols}")
    print(f"Engineered features: {df.shape[1] - original_cols}")
    print(f"Total features: {df.shape[1]}")
    print(f"{'='*80}\n")
    
    return df


# ============================================================================
# FUNCTION 6: PREPROCESS FEATURES
# ============================================================================

def preprocess_features(df):
    """
    Handle missing values, remove impossible values, and prepare for modeling.
    
    Args:
        df: DataFrame with all features
    
    Returns:
        DataFrame: Cleaned and preprocessed
    """
    print("="*80)
    print("PREPROCESSING FEATURES")
    print("="*80)
    
    df = df.copy()
    original_rows = len(df)
    
    # -------------------------------------------------------------------------
    # HANDLE IMPOSSIBLE VALUES
    # -------------------------------------------------------------------------
    print("\n[1/3] Handling impossible physiological values...")
    
    # Cholesterol: 0 or >400 is likely error
    if 'total_cholesterol' in df.columns:
        before = len(df)
        df.loc[df['total_cholesterol'] == 0, 'total_cholesterol'] = np.nan
        df.loc[df['total_cholesterol'] > 400, 'total_cholesterol'] = np.nan
        after = len(df)
        if before > after:
            print(f"  ✓ Set {before - after} impossible cholesterol values to NaN")
    
    # Blood Pressure: 0 or extreme values
    if 'systolic_bp' in df.columns:
        df.loc[df['systolic_bp'] == 0, 'systolic_bp'] = np.nan
        df.loc[df['systolic_bp'] > 250, 'systolic_bp'] = np.nan
        df.loc[df['systolic_bp'] < 70, 'systolic_bp'] = np.nan
    
    if 'diastolic_bp' in df.columns:
        df.loc[df['diastolic_bp'] == 0, 'diastolic_bp'] = np.nan
        df.loc[df['diastolic_bp'] > 150, 'diastolic_bp'] = np.nan
        df.loc[df['diastolic_bp'] < 40, 'diastolic_bp'] = np.nan
    
    # Glucose: 0 or >400
    if 'fasting_glucose' in df.columns:
        df.loc[df['fasting_glucose'] == 0, 'fasting_glucose'] = np.nan
        df.loc[df['fasting_glucose'] > 400, 'fasting_glucose'] = np.nan
    
    # BMI: <10 or >60
    if 'bmi' in df.columns:
        df.loc[df['bmi'] < 10, 'bmi'] = np.nan
        df.loc[df['bmi'] > 60, 'bmi'] = np.nan
    
    # Age: <18 or >120
    if 'age' in df.columns:
        df = df[(df['age'] >= 18) & (df['age'] <= 120)]
    
    # -------------------------------------------------------------------------
    # MISSING VALUE ANALYSIS
    # -------------------------------------------------------------------------
    print("\n[2/3] Analyzing missing values...")
    
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) > 0:
        print("\nFeatures with missing values:")
        for col, pct in missing_pct.head(10).items():
            print(f"  {col:30s}: {pct:5.1f}%")
    
    # Drop features with >70% missing data
    high_missing_cols = missing_pct[missing_pct > 70].index.tolist()
    if high_missing_cols:
        print(f"\n  ⚠ Dropping {len(high_missing_cols)} features with >70% missing data:")
        for col in high_missing_cols:
            print(f"    - {col}")
        df = df.drop(columns=high_missing_cols)
    
    # -------------------------------------------------------------------------
    # IMPUTATION
    # -------------------------------------------------------------------------
    print("\n[3/3] Imputing remaining missing values...")
    
    # Separate features by type
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['risk_target']]
    
    # Impute numeric features with median
    if numeric_features:
        imputer_numeric = SimpleImputer(strategy='median')
        df[numeric_features] = imputer_numeric.fit_transform(df[numeric_features])
        print(f"  ✓ Imputed {len(numeric_features)} numeric features with median")
    
    # Categorical features (if any)
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [col for col in categorical_features if col not in ['data_source']]
    
    if categorical_features:
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        df[categorical_features] = imputer_categorical.fit_transform(df[categorical_features])
        print(f"  ✓ Imputed {len(categorical_features)} categorical features with mode")
    
    # Drop rows with missing target (should be rare after earlier cleaning)
    df = df.dropna(subset=['risk_target'])
    
    print(f"\n{'='*80}")
    print(f"Preprocessing complete!")
    print(f"Rows before: {original_rows}")
    print(f"Rows after:  {len(df)}")
    print(f"Rows removed: {original_rows - len(df)} ({(original_rows - len(df))/original_rows*100:.1f}%)")
    print(f"{'='*80}\n")
    
    return df


# ============================================================================
# FUNCTION 7: SPLIT AND SAVE DATASETS
# ============================================================================

def split_and_save_datasets(df, 
                           train_ratio=0.70, 
                           val_ratio=0.15, 
                           test_ratio=0.15,
                           output_dir='./data/'):
    """
    Create stratified train/val/test splits and save to disk.
    
    Args:
        df: Preprocessed DataFrame
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        output_dir: Base directory for saving files
    
    Returns:
        dict: {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
    """
    print("="*80)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("="*80)
    
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
    
    # Create output directories
    processed_dir = Path(output_dir) / 'processed'
    final_dir = Path(output_dir) / 'final'
    split_dir = Path(output_dir) / 'split'
    
    for directory in [processed_dir, final_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STRATIFIED SPLIT
    # -------------------------------------------------------------------------
    
    # First split: train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df['risk_target'],
        random_state=42
    )
    
    # Second split: val vs test
    val_test_ratio = test_ratio / temp_ratio
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_test_ratio,
        stratify=temp_df['risk_target'],
        random_state=42
    )
    
    # -------------------------------------------------------------------------
    # SAVE DATASETS
    # -------------------------------------------------------------------------
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df):5d} ({len(train_df)/len(df)*100:.1f}%) | Target: {train_df['risk_target'].mean():.1%}")
    print(f"  Val:   {len(val_df):5d} ({len(val_df)/len(df)*100:.1f}%) | Target: {val_df['risk_target'].mean():.1%}")
    print(f"  Test:  {len(test_df):5d} ({len(test_df)/len(df)*100:.1f}%) | Target: {test_df['risk_target'].mean():.1%}")
    
    print(f"\nSaving datasets...")
    
    # 1. Save merged/processed dataset
    merged_path = processed_dir / 'merged_risk_dataset.csv'
    df.to_csv(merged_path, index=False)
    print(f"  ✓ Saved: {merged_path}")
    
    # 2. Save final dataset (same as merged for this pipeline)
    final_path = final_dir / 'final_risk_dataset.csv'
    df.to_csv(final_path, index=False)
    print(f"  ✓ Saved: {final_path}")
    
    # 3. Save splits
    train_path = split_dir / 'train.csv'
    val_path = split_dir / 'val.csv'
    test_path = split_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  ✓ Saved: {train_path}")
    print(f"  ✓ Saved: {val_path}")
    print(f"  ✓ Saved: {test_path}")
    
    print(f"\n{'='*80}")
    print("ALL DATASETS SAVED SUCCESSFULLY!")
    print(f"{'='*80}\n")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline(raw_dir='./data/raw/', output_dir='./data/'):
    """
    Execute the complete data preprocessing pipeline.
    
    Args:
        raw_dir: Directory containing raw CSV files
        output_dir: Directory for saving processed files
    
    Returns:
        dict: {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
    """
    print("\n" + "="*80)
    print("CARDIODETECT DATA PREPROCESSING PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Scan raw data
    inventory = scan_raw_datasets(raw_dir)
    
    # Step 2: Load datasets
    datasets = load_raw_datasets(raw_dir)
    
    # Step 3: Build unified dataset
    unified_df = build_unified_dataset(datasets)
    
    # Step 4: Feature engineering
    engineered_df = engineer_features(unified_df)
    
    # Step 5: Preprocessing
    preprocessed_df = preprocess_features(engineered_df)
    
    # Step 6: Split and save
    splits = split_and_save_datasets(preprocessed_df, output_dir=output_dir)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nFinal Dataset Summary:")
    print(f"  Total samples: {len(preprocessed_df)}")
    print(f"  Total features: {preprocessed_df.shape[1] - 2}")  # Exclude target and source
    print(f"  Target column: risk_target")
    print(f"  Positive cases: {preprocessed_df['risk_target'].sum()} ({preprocessed_df['risk_target'].mean()*100:.1f}%)")
    print(f"\nOutput locations:")
    print(f"  Merged dataset: ./data/processed/merged_risk_dataset.csv")
    print(f"  Final dataset:  ./data/final/final_risk_dataset.csv")
    print(f"  Train split:    ./data/split/train.csv")
    print(f"  Val split:      ./data/split/val.csv")
    print(f"  Test split:     ./data/split/test.csv")
    print("="*80 + "\n")
    
    return splits


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Run the full pipeline
    splits = run_full_pipeline()
    
    print("✅ Data preprocessing complete! Ready for modeling.")
