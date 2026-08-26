# 🏦 Credit Risk Prediction & Decisioning System

An end-to-end **Machine Learning Credit Risk Prediction System** that predicts the probability of loan default using a tuned **XGBoost classifier**.

The project combines application-level information with engineered historical credit features from multiple financial datasets. A **FastAPI backend** serves real-time predictions, while a **Streamlit dashboard** provides an interactive interface for evaluating loan applications and understanding model decisions through **SHAP explainability**.

---

## 🚀 Project Overview

Credit risk prediction is a highly imbalanced classification problem where accurately identifying potential defaulters is more important than relying solely on overall accuracy.

This project implements a complete machine learning workflow:

**Raw Financial Data → Preprocessing → Feature Engineering → Historical Data Aggregation → Baseline Models → XGBoost → Hyperparameter Tuning → Final Model → FastAPI → Streamlit Dashboard**

The system provides both **credit risk prediction** and **model explainability** for individual applications.

---

## ✨ Features

- 📊 Exploratory Data Analysis
- 🧹 Data cleaning and preprocessing
- 🔧 Feature engineering
- 🏦 Historical financial data aggregation
- ⚖️ Handling highly imbalanced classification
- 🤖 Logistic Regression baseline
- 🌲 Random Forest baseline
- 🚀 XGBoost classification
- 🔍 XGBoost hyperparameter tuning
- 📈 ROC-AUC and PR-AUC evaluation
- 🎯 Decision threshold optimization
- 🧠 SHAP model explainability
- ⚡ FastAPI prediction API
- 🎨 Interactive Streamlit dashboard
- ☁️ Deployment-ready architecture

---

# 📂 Dataset

This project uses the **Home Credit Default Risk** dataset.

The dataset contains information about loan applicants, including:

- Income
- Credit amount
- Annuity
- Goods price
- Age
- Employment information
- External credit scores
- Education
- Family status
- Previous applications
- Bureau credit history
- Credit card history
- Installment payment history
- POS/Cash loan history

Historical datasets were aggregated using the applicant identifier:

**SK_ID_CURR**

The final modeling dataset contains approximately:

**307,511 applicants and 256 features**

---

# 🛠️ Tech Stack

## Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP

## Backend

- FastAPI
- Uvicorn
- Pydantic

## Dashboard

- Streamlit
- Requests
- Pandas

## Development

- Jupyter Notebook
- Git
- GitHub

---

# 🤖 Machine Learning Pipeline

## 1. Data Preprocessing

The original application dataset was cleaned and prepared for machine learning.

Major preprocessing operations included:

- Missing value analysis
- Numerical feature handling
- Categorical feature encoding
- Feature scaling where required
- Removal of constant features
- Removal of duplicate columns
- Data type optimization

## 2. Feature Engineering

Additional features were created to capture meaningful financial relationships.

Examples include:

- Age in years
- Employment duration in years
- Credit-to-income ratio
- Credit-to-annuity ratio
- Annuity-to-income ratio
- Payment-related indicators
- Historical default and payment behavior

## 3. Historical Data Aggregation

Additional information was aggregated from:

- Bureau
- Bureau Balance
- Previous Applications
- POS/Cash Balance
- Credit Card Balance
- Installment Payments

Aggregation was performed at the applicant level using **SK_ID_CURR**.

This allowed the final model to use both current application information and historical financial behavior.

---

# 📊 Baseline Model Comparison

Three baseline models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

XGBoost provided the strongest overall performance and was selected for further optimization.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7339 | 0.2127 | 0.1526 | 0.6705 | 0.2486 |
| Random Forest | 0.7454 | 0.2232 | 0.2024 | 0.4832 | 0.2853 |
| XGBoost | 0.7693 | 0.2608 | 0.1862 | 0.6445 | 0.2889 |

Because the dataset is highly imbalanced, ROC-AUC and PR-AUC were emphasized instead of relying only on accuracy.

---

# 🚀 Final Tuned XGBoost

After aggregating the historical datasets, XGBoost was optimized using randomized hyperparameter search.

### Best Cross-Validation Score

**ROC-AUC: approximately 0.783**

### Final Validation Performance

| Metric | Score |
|---|---:|
| ROC-AUC | **0.7881** |
| PR-AUC | **0.2885** |
| Precision | **0.1900** |
| Recall | **0.6985** |
| F1 Score | **0.2987** |

### Confusion Matrix

| | Predicted Non-Default | Predicted Default |
|---|---:|---:|
| Actual Non-Default | 41,754 | 14,784 |
| Actual Default | 1,497 | 3,468 |

---

# ⚖️ Class Imbalance

The target variable is highly imbalanced:

| Target | Count |
|---|---:|
| Non-Default (0) | 282,686 |
| Default (1) | 24,825 |

Therefore, accuracy alone is not a reliable measure of model quality.

The project focuses on:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1 Score
- Confusion Matrix

Recall is particularly important because failing to identify an actual defaulter can represent significant financial risk.

---

# 🎯 Risk Decisioning

The model generates a probability of default.

A configurable threshold is then used to convert the probability into a credit decision.

Example:

**Default Probability: 82.05%**

**Decision: REJECT**

The system therefore separates:

**Risk Prediction → Probability → Threshold → Credit Decision**

This allows the decision threshold to be adjusted depending on the desired balance between detecting potential defaults and approving applicants.

---

# 🧠 SHAP Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

SHAP provides insight into which features contribute most strongly to an applicant's predicted credit risk.

Example top risk drivers:

| Feature | SHAP Value |
|---|---:|
| LATE_PAYMENT_RATE | +0.2349 |
| PREV_REFUSAL_RATE | +0.2184 |
| AMT_GOODS_PRICE | +0.1663 |
| AMT_ANNUITY | +0.1333 |
| CODE_GENDER_M | +0.1065 |

Positive SHAP values generally push the prediction toward higher default risk, while negative SHAP values push the prediction toward lower risk.

The dashboard displays the strongest local SHAP drivers for each prediction.

---

# 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard for credit risk evaluation.

Users can provide applicant information such as:

### Financial Information

- External Source 1 Score
- External Source 2 Score
- External Source 3 Score
- Requested Credit
- Annuity Amount
- Goods Price
- Annual Income

### Applicant Information

- Age
- Employment Years
- Gender
- Family Status
- Education Level

### Historical Information

- Late Payment Rate
- Bureau Debt Ratio
- Previous Application Refusal Rate
- Previous Down Payment
- POS Installments
- POS Months Balance

The dashboard communicates with the FastAPI backend and displays:

- Default probability
- Risk level
- Credit decision
- Top SHAP risk drivers

---

# 🖥️ Dashboard Screenshots

## Main Dashboard

![Credit Risk Dashboard](images/dashboard.png)



---

## Credit Risk Prediction

![Credit Risk Prediction](images/prediction.png)



---

## SHAP Explainability

![SHAP Explainability](images/shap.png)



---
