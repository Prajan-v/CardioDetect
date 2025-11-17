# Data Analysis Report

## Executive Summary
This integrated dataset contains 15920 patients spanning multiple sources and years. Numeric missingness averages 43.07%. Engineered risk factors and standardized preprocessing prepare the data for robust modeling.

## Composition by Source
- BRFSS: 15000
- UCI: 920

## Composition by Year
- 1988: 920
- 2011: 3000
- 2012: 3000
- 2013: 3000
- 2014: 3000
- 2015: 3000

## Key Correlations (numeric vs target)
- age vs target: 0.20
- trestbps vs target: -0.18
- chol vs target: -0.21
- bmi vs target: 0.08
- thalach vs target: -0.40
- oldpeak vs target: 0.39

## Data Quality
- Median imputation for numeric features; most-frequent for categoricals
- Standardization and one-hot encoding within a reproducible pipeline
- Outlier handling considered during modeling stage

## Recommendations for Milestone 2
- Train baselines (Logistic, RF, XGBoost) with calibration
- Evaluate with ROC-AUC and decision curves; define risk thresholds
- Perform feature selection (RFE/L1) and sensitivity analysis