# Customer Churn Analysis & Prediction

An end-to-end **customer churn analytics and prediction** project built with Python and Streamlit.  
This repository combines data preparation, machine learning, interactive risk scoring, batch processing, and historical prediction tracking in one business-ready application.

## Live Demo
- Streamlit App: https://customer-churn.streamlit.app/

## Project Overview
This project predicts whether telecom customers are likely to churn, using the **Telco Customer Churn** dataset and a tuned **Random Forest** classifier.

The solution includes:
- Feature engineering and preprocessing pipeline
- Model training and artifact persistence (`joblib`)
- Single-customer prediction flow with threshold control
- Batch scoring for CSV/Excel files
- Risk segmentation (**Low / Medium / High**)
- SQLite-based prediction history with filtering and deletion tools
- Interactive business visualizations with Plotly

## Key Features
- **Single Prediction:** Manual form-based scoring for one customer
- **Batch Prediction:** Upload `.csv`, `.xlsx`, or `.xls` files for bulk predictions
- **Download Outputs:** Export scored records as CSV/Excel
- **History Dashboard:** Track previous runs and monitor risk trends
- **Threshold Tuning:** Adjust decision threshold (default: `0.30`) based on precision-recall trade-offs
- **Model Transparency:** Feature-importance and threshold-analysis visual references

## Tech Stack
- **Language:** Python 3.11
- **App Framework:** Streamlit
- **ML/Analytics:** scikit-learn, pandas, numpy, imbalanced-learn
- **Visualization:** Plotly
- **Persistence:** SQLite (`artifacts/prediction_history.db`)
- **Model Serialization:** joblib

## Repository Structure
```text
Customer-churn/
├── app.py                         # Streamlit application
├── train_model.py                 # Training entry point
├── src/
│   └── churn_backend.py           # Core training, inference, and history logic
├── artifacts/
│   ├── churn_model.joblib         # Saved model bundle
│   └── prediction_history.db      # Prediction history database
├── Powerbi_exports/               # Benchmark and threshold analysis CSVs
├── shap_plots/                    # Model interpretation visual artifacts
├── Telco_customer_churn.csv       # Input dataset
├── feature_importance.csv         # Persisted feature importance
└── README.md
```

## Model Snapshot
From `Powerbi_exports/01_model_metrics.csv`:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7835 | 0.5800 | 0.6684 | 0.6211 | 0.8447 |
| CatBoost | 0.7878 | 0.6027 | 0.5882 | 0.5954 | 0.8405 |
| XGBoost | 0.7864 | 0.6000 | 0.5856 | 0.5927 | 0.8341 |
| LightGBM | 0.7807 | 0.5915 | 0.5615 | 0.5761 | 0.8306 |

> The deployed application uses a Random Forest model and supports threshold-based decision tuning.

## Getting Started
### 1) Clone the repository
```bash
git clone https://github.com/AmnaNaveed189/Customer-churn.git
cd Customer-churn
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Train or refresh model artifacts
```bash
python train_model.py
```

### 5) Run the Streamlit application
```bash
streamlit run app.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

## Required Batch Input Columns
Batch files must include these fields:

- `Gender`
- `Senior Citizen`
- `Partner`
- `Dependents`
- `Tenure Months`
- `Phone Service`
- `Multiple Lines`
- `Internet Service`
- `Online Security`
- `Online Backup`
- `Device Protection`
- `Tech Support`
- `Streaming TV`
- `Streaming Movies`
- `Contract`
- `Paperless Billing`
- `Payment Method`
- `Monthly Charges`
- `Total Charges`
- `CLTV`

## How Prediction Output Is Enriched
For each scored customer, the app adds:
- `Churn Probability`
- `Predicted Churn` (0/1)
- `Prediction Label` (Churn/Stay)
- `Risk Tier` (Low/Medium/High)
- `Threshold Used`
- `Run ID`
- `Prediction Time`

## Business Use Cases
- Proactively identify customers at high churn risk
- Prioritize retention campaigns and outreach lists
- Evaluate threshold impact before operational rollout
- Monitor prediction behavior over time through history analytics

## Notes
- If `openpyxl` is unavailable, Excel import/export features may be limited.
- Prediction history is stored locally in SQLite under `artifacts/`.

## Screenshots
Project screenshots are available in the repository root:
- `Customer-Churn-01.png`
- `Customer-Churn-02.png`
- `Customer-Churn-03.png`
- `Customer-Churn-04.png`

## Author
**Amna Naveed**

---
If you find this project useful, consider giving the repository a ⭐.
