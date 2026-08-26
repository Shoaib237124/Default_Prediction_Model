# Credit Risk Prediction System

## Overview

An end-to-end machine learning system for predicting loan default risk using the Home Credit dataset.

## Features

- Data preprocessing and feature engineering
- Aggregation of bureau and previous application history
- Logistic Regression baseline
- Random Forest baseline
- XGBoost modeling
- Hyperparameter tuning
- ROC-AUC and PR-AUC evaluation
- Threshold optimization
- SHAP explainability
- FastAPI inference backend
- Streamlit interactive dashboard

## Architecture
```text
Streamlit
    ↓
FastAPI
    ↓
Preprocessing Pipeline
    ↓
Tuned XGBoost
    ↓
Risk Probability
    ↓
SHAP Explanation
```
## Model Performance

| Metric | Score |
|---|---:|
| ROC-AUC | 0.7881 |
| PR-AUC | 0.2885 |
| Precision | 0.1900 |
| Recall | 0.6985 |
| F1 | 0.2987 |

## Tech Stack

* Python
* XGBoost
* Scikit-learn
* Pandas
* SHAP
* FastAPI
* Streamlit
