# Customer Churn Analytics & Prediction Platform

> Production-quality end-to-end analytics pipeline built for an American Express MIS & Advanced Analytics internship portfolio.

---

## Project Overview

A **modular Python application** that ingests a Telco customer churn dataset, cleans and validates it, performs deep feature engineering, stores data in SQLite, executes **30 business SQL analytics queries**, runs **Exploratory Data Analysis**, trains **4 Machine Learning models**, generates **23 professional visualizations**, and exports executive-level reports — all via a single command.

---

## Architecture

```
Data → Clean → Feature Eng → SQLite → 30 SQL Queries
                                ↓
                              EDA → ML Training → Best Model
                                ↓
                         23 Visualizations → Reports & CSV Exports
```

---

## Features

| Category | Details |
|---|---|
| **Data Pipeline** | Load → Clean → Validate → Feature Engineer |
| **Database** | SQLite with PKs, 9 indexes, 4 views, constraints, bulk insert |
| **SQL Analytics** | 30 queries covering KPIs, window functions, CTEs, subqueries, CASE |
| **EDA** | Correlation matrix, distributions, segment analysis, churn breakdown |
| **Machine Learning** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| **Visualizations** | 23 charts: ROC curves, confusion matrix, Pareto, heatmaps, boxplots |
| **Exports** | Executive summary, business report, all query CSVs, model metrics |

---

## Dataset

**Source:** IBM Telco Customer Churn  
**Rows:** 7,043 customers  
**Columns:** 33 features (demographics, services, contract, financials, churn label)  
**Location:** `data/customers.csv`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data | Pandas 2.x, NumPy |
| Database | SQLite 3 (stdlib) |
| ML | Scikit-learn (LR, DT, RF, GB) |
| Visualizations | Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Code Quality | PEP 8, type hints, logging, comments |

---

## Folder Structure

```
customer_churn_analytics/
├── data/
│   └── customers.csv              # Source dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # CSV → raw DataFrame
│   ├── data_cleaner.py            # Validation, imputation, deduplication
│   ├── feature_engineering.py     # 14 derived features
│   ├── database.py                # SQLite schema, bulk insert, views
│   ├── queries.py                 # 30 business SQL queries
│   ├── eda.py                     # EDA aggregations
│   ├── model_training.py          # Train 4 models, evaluate, select best
│   ├── prediction.py              # Single + batch churn prediction
│   ├── visualizations.py          # 23 Matplotlib / Seaborn charts
│   └── exporter.py                # Executive summary, reports, CSV
├── models/
│   └── churn_model.pkl            # Best trained model (auto-generated)
├── output/
│   ├── plots/                     # 23 PNG charts
│   ├── csv/                       # 30 SQL query CSVs
│   └── reports/                   # Text reports + cleaned dataset
├── docs/
│   ├── data_dictionary.md
│   └── query_reference.md
├── main.py                        # 🚀 Entry point
├── predict.py                     # CLI churn predictor
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

Pipeline steps (printed in real time):
1. Load data
2. Clean & validate
3. Feature engineering
4. Load SQLite
5. Run 30 SQL queries
6. EDA
7. Train ML models
8. Generate 23 charts
9. Export reports

### 3. Predict a single customer

```bash
python predict.py
```

Prompts for customer attributes and outputs:
```
  Churn Probability    : 82.3%
  Prediction           : Likely to Churn
  Risk Classification  : Very High Risk — Likely to Churn
```

---

## Business Insights

1. **Month-to-month contracts** drive the highest churn (~43%) vs only ~3% for two-year contracts
2. **Electronic check users** churn at ~45% — encourage auto-pay migration
3. **Fiber optic customers** churn at 2× the rate of DSL customers despite higher ARPU
4. **Short-tenure customers (0–12 months)** are the highest-risk cohort; early intervention is critical
5. **Senior citizens** churn ~11 points more than non-seniors
6. **High-risk, not-yet-churned** customers represent a high-priority retention target
7. **Top 20% of customers** generate ~60%+ of total revenue (Pareto effect)

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.81 | ~0.67 | ~0.56 | ~0.61 | ~0.85 |
| Decision Tree | ~0.79 | ~0.62 | ~0.58 | ~0.60 | ~0.75 |
| Random Forest | ~0.82 | ~0.69 | ~0.54 | ~0.61 | ~0.86 |
| **Gradient Boosting** | **~0.83** | **~0.72** | **~0.58** | **~0.64** | **~0.87** |

> Exact values printed to console after each run. Best model auto-selected by ROC-AUC.

---

## Screenshots

> After running `python main.py`, charts are saved to `output/plots/`.

| Chart | File |
|---|---|
| Churn Distribution | `output/plots/01_churn_distribution.png` |
| Monthly Charges | `output/plots/02_monthly_charges_distribution.png` |
| ROC Curves | `output/plots/17_roc_curves.png` |
| Feature Importance | `output/plots/18_feature_importance.png` |
| Confusion Matrix | `output/plots/19_confusion_matrix.png` |
| Pareto Revenue | `output/plots/23_revenue_concentration.png` |

---

## Future Improvements

- [ ] XGBoost / LightGBM / CatBoost models
- [ ] SHAP explainability values per prediction
- [ ] Streamlit or FastAPI dashboard
- [ ] Hyperparameter tuning with Optuna
- [ ] Real-time scoring API endpoint
- [ ] Customer cohort survival analysis (Kaplan-Meier)
- [ ] Automated retraining pipeline with model drift detection

---


