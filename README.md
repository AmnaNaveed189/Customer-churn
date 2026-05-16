# Customer Churn Analysis & Prediction

An end-to-end machine learning project that predicts telecom customer churn and turns model output into business-ready decision support through a deployed Streamlit app and a Power BI report.

## Why this project is recruiter-ready
- Solves a real business problem: identifying customers likely to churn
- Covers the full lifecycle: data prep, model training, deployment, BI storytelling
- Includes both product-facing delivery (Streamlit) and stakeholder reporting (Power BI)
- Demonstrates practical MLOps habits: artifact persistence, threshold tuning, and prediction history tracking

## Deployed Streamlit App
- **Live URL:** https://customer-churn.streamlit.app/
- **What it shows:**
  - Single-customer churn prediction with probability + risk tier
  - Batch scoring from CSV/Excel with downloadable outputs
  - Threshold control for precision/recall trade-off decisions
  - Prediction history dashboard stored in SQLite for operational tracking

## Tech Stack
- **Language:** Python 3.11
- **ML:** scikit-learn (Random Forest), imbalanced-learn
- **Data:** pandas, numpy
- **Web App:** Streamlit
- **Visualization:** Plotly
- **Storage:** SQLite (`artifacts/prediction_history.db`)
- **Model Artifact:** joblib (`artifacts/churn_model.joblib`)
- **BI Layer:** Power BI (`Customer Churn.pbix`) with CSV feeds from `Powerbi_exports/`

## Model Snapshot
From `Powerbi_exports/01_model_metrics.csv`:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7835 | 0.5800 | 0.6684 | 0.6211 | 0.8447 |
| CatBoost | 0.7878 | 0.6027 | 0.5882 | 0.5954 | 0.8405 |
| XGBoost | 0.7864 | 0.6000 | 0.5856 | 0.5927 | 0.8341 |
| LightGBM | 0.7807 | 0.5915 | 0.5615 | 0.5761 | 0.8306 |

> The deployed Streamlit application currently serves the Random Forest model with adjustable thresholding.

## Power BI Project
- **File:** `Customer Churn.pbix`
- **Data sources used inside report:**  
  `Powerbi_exports/01_model_metrics.csv`, `Powerbi_exports/02_customer_predictions.csv`, `Powerbi_exports/05_threshold_analysis.csv`

### Short analysis of each dashboard
1. **Model Benchmark Dashboard**
   - Compares Random Forest, CatBoost, XGBoost, and LightGBM across Accuracy, Recall, F1, and ROC AUC.
   - Insight: all models are close, with Random Forest offering strong ROC-AUC (0.8447) and balanced operational performance for deployment.

2. **Customer Risk Dashboard**
   - Uses prediction-level data to segment customers into **Low / Medium / High** risk tiers and compare predicted vs actual churn.
   - Insight: high-risk customers become the primary retention target group, while medium-risk customers are ideal for cost-sensitive interventions.

3. **Threshold Strategy Dashboard**
   - Visualizes how Precision, Recall, F1, Accuracy, and Specificity shift as threshold changes from 0.10 to 0.90.
   - Insight: lower thresholds maximize recall (catch more churners), while higher thresholds reduce false positives and improve precision.

4. **Decision Support / Operations Dashboard**
   - Combines model and risk outputs into business-facing decision views for campaign prioritization.
   - Insight: teams can align retention actions by risk tier and threshold policy instead of relying on a single fixed metric.

## Key Features in the Streamlit Product
- **Single Prediction:** Form-based scoring for one customer profile
- **Batch Prediction:** Upload `.csv`, `.xlsx`, or `.xls` files for bulk predictions
- **Downloads:** Export scored records to CSV/Excel
- **History Analytics:** Track previous runs, filter risk tiers, and manage stored records
- **Model Explainability:** Feature importance and threshold trade-off charts

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
├── Powerbi_exports/               # CSV exports for Power BI report
├── shap_plots/                    # Model interpretation visual artifacts
├── Customer Churn.pbix            # Power BI report
├── Telco_customer_churn.csv       # Input dataset
├── feature_importance.csv         # Persisted feature importance
└── README.md
```

## Run Locally
```bash
git clone https://github.com/AmnaNaveed189/Customer-churn.git
cd Customer-churn
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Screenshots
- `Customer-Churn-01.png`
- `Customer-Churn-02.png`
- `Customer-Churn-03.png`
- `Customer-Churn-04.png`

## Author
**Amna Naveed**

---
If you find this project useful, consider giving the repository a ⭐.
