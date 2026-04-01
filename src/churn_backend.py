from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


APP_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = APP_ROOT / "artifacts"
DATASET_PATH = APP_ROOT / "Telco_customer_churn.csv"
MODEL_PATH = ARTIFACTS_DIR / "churn_model.joblib"
HISTORY_DB_PATH = ARTIFACTS_DIR / "prediction_history.db"
FEATURE_IMPORTANCE_PATH = APP_ROOT / "feature_importance.csv"
DEFAULT_THRESHOLD = 0.30

DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
]

MODEL_INPUT_COLUMNS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]

SERVICE_COLUMNS = [
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

BINARY_COLUMNS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Paperless Billing",
]

NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "avg_monthly_spend",
    "Charge per Service",
]

DEFAULT_FORM_VALUES = {
    "CustomerID": "CUST-1001",
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 79.90,
    "Total Charges": 958.80,
    "CLTV": 3200,
}

SELECT_OPTIONS = {
    "Gender": ["Female", "Male"],
    "Senior Citizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "Phone Service": ["No", "Yes"],
    "Multiple Lines": ["No", "Yes", "No phone service"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["No", "Yes", "No internet service"],
    "Online Backup": ["No", "Yes", "No internet service"],
    "Device Protection": ["No", "Yes", "No internet service"],
    "Tech Support": ["No", "Yes", "No internet service"],
    "Streaming TV": ["No", "Yes", "No internet service"],
    "Streaming Movies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["No", "Yes"],
    "Payment Method": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


@dataclass
class ModelBundle:
    model: RandomForestClassifier
    scaler: StandardScaler
    feature_columns: list[str]
    threshold: float
    total_charges_median: float
    metadata: dict[str, Any]


def _ensure_directories() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _label_to_int(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1, "female": 0, "male": 1})
        .fillna(pd.to_numeric(series, errors="coerce"))
        .fillna(0)
        .astype(int)
    )


def _normalize_service_value(value: Any) -> int:
    return 1 if str(value).strip().lower() == "yes" else 0


def risk_tier(probabilities: pd.Series | np.ndarray) -> pd.Series:
    probs = pd.Series(probabilities, dtype=float)
    return pd.cut(
        probs,
        bins=[0.0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)


def build_template_dataframe(rows: int = 5) -> pd.DataFrame:
    template_rows = []
    for idx in range(rows):
        row = DEFAULT_FORM_VALUES.copy()
        row["CustomerID"] = f"CUST-{1001 + idx}"
        row["Tenure Months"] = int(DEFAULT_FORM_VALUES["Tenure Months"]) + idx
        row["Total Charges"] = round(
            float(DEFAULT_FORM_VALUES["Monthly Charges"]) * (idx + 1), 2
        )
        template_rows.append(row)
    return pd.DataFrame(template_rows)


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(dataframe: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="predictions")
    output.seek(0)
    return output.read()


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _prepare_features(
    dataframe: pd.DataFrame,
    *,
    scaler: StandardScaler | None = None,
    feature_columns: list[str] | None = None,
    total_charges_median: float | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, StandardScaler, float]:
    frame = dataframe.copy()
    for column in MODEL_INPUT_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"Missing required column: {column}")

    frame["Total Charges"] = pd.to_numeric(frame["Total Charges"], errors="coerce")
    fill_value = (
        float(frame["Total Charges"].median())
        if fit or total_charges_median is None
        else float(total_charges_median)
    )
    frame["Total Charges"] = frame["Total Charges"].fillna(fill_value)

    for column in BINARY_COLUMNS:
        frame[column] = _label_to_int(frame[column])

    for column in SERVICE_COLUMNS:
        frame[column] = frame[column].apply(_normalize_service_value).astype(int)

    contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    frame["Contract"] = frame["Contract"].map(contract_mapping).fillna(0).astype(int)
    frame["Tenure Months"] = pd.to_numeric(frame["Tenure Months"], errors="coerce").fillna(0)
    frame["Monthly Charges"] = pd.to_numeric(frame["Monthly Charges"], errors="coerce").fillna(0)
    frame["CLTV"] = pd.to_numeric(frame["CLTV"], errors="coerce").fillna(0)

    frame["tenure_group"] = pd.cut(
        frame["Tenure Months"],
        bins=[-1, 12, 36, float("inf")],
        labels=[0, 1, 2],
    ).astype(int)
    frame["avg_monthly_spend"] = np.where(
        frame["Tenure Months"] > 0,
        frame["Total Charges"] / frame["Tenure Months"],
        frame["Monthly Charges"],
    )
    frame["avg_monthly_spend"] = frame["avg_monthly_spend"].round(2)
    frame["High Value Customer"] = (frame["avg_monthly_spend"] > 70).astype(int)
    frame["Add-on Services"] = frame[SERVICE_COLUMNS[1:]].sum(axis=1)
    frame["Charge per Service"] = frame["Monthly Charges"] / (frame["Add-on Services"] + 1)

    frame = pd.get_dummies(
        frame,
        columns=["Internet Service", "Payment Method"],
        drop_first=True,
        dtype=int,
    )

    scaler_to_use = scaler if scaler is not None else StandardScaler()
    if fit:
        frame[NUMERICAL_FEATURES] = scaler_to_use.fit_transform(frame[NUMERICAL_FEATURES])
    else:
        if scaler is None:
            raise ValueError("Scaler is required when fit=False.")
        frame[NUMERICAL_FEATURES] = scaler_to_use.transform(frame[NUMERICAL_FEATURES])

    if feature_columns is None:
        feature_columns = list(frame.columns)

    missing_columns = [column for column in feature_columns if column not in frame.columns]
    for column in missing_columns:
        frame[column] = 0

    extra_columns = [column for column in frame.columns if column not in feature_columns]
    if extra_columns:
        frame = frame.drop(columns=extra_columns)

    return frame[feature_columns], scaler_to_use, fill_value


def _prepare_training_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = dataframe.copy().drop(columns=DROP_COLUMNS, errors="ignore")
    y = _label_to_int(frame["Churn Label"])
    X = frame.drop(columns=["Churn Label"], errors="ignore")
    return X, y


def _maybe_resample(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, str]:
    try:
        from imblearn.combine import SMOTETomek

        sampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, "SMOTETomek"
    except Exception:
        return X, y, "class_weight_only"


def train_and_save_model(force: bool = False) -> ModelBundle:
    _ensure_directories()
    if MODEL_PATH.exists() and not force:
        return load_model_bundle()

    raw_df = pd.read_csv(DATASET_PATH)
    X_raw, y = _prepare_training_data(raw_df)
    X_processed, scaler, total_charges_median = _prepare_features(X_raw, fit=True)
    feature_columns = list(X_processed.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train_fit, y_train_fit, sampling_strategy = _maybe_resample(X_train, y_train)

    eval_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    eval_model.fit(X_train_fit, y_train_fit)
    y_prob = eval_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DEFAULT_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "specificity": float(round(tn / (tn + fp), 4)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    X_full_fit, y_full_fit, _ = _maybe_resample(X_processed, y)
    final_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    final_model.fit(X_full_fit, y_full_fit)

    feature_importance = pd.DataFrame(
        {"Feature": feature_columns, "Importance": final_model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    payload = {
        "model": final_model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "threshold": DEFAULT_THRESHOLD,
        "total_charges_median": total_charges_median,
        "metadata": {
            "model_name": "Random Forest",
            "sampling_strategy": sampling_strategy,
            "trained_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "dataset_rows": int(len(raw_df)),
        },
    }
    joblib.dump(payload, MODEL_PATH)
    return ModelBundle(**payload)


def load_model_bundle() -> ModelBundle:
    _ensure_directories()
    if not MODEL_PATH.exists():
        return train_and_save_model(force=True)
    payload = joblib.load(MODEL_PATH)
    return ModelBundle(**payload)


def predict_customers(
    dataframe: pd.DataFrame,
    *,
    threshold: float | None = None,
    save_history: bool = False,
    run_type: str = "single",
    source_name: str | None = None,
) -> tuple[pd.DataFrame, str]:
    bundle = load_model_bundle()
    model_threshold = bundle.threshold if threshold is None else float(threshold)

    display_df = dataframe.copy()
    if "CustomerID" not in display_df.columns:
        display_df["CustomerID"] = [f"Customer-{idx + 1}" for idx in range(len(display_df))]

    processed_df, _, _ = _prepare_features(
        display_df[MODEL_INPUT_COLUMNS].copy(),
        scaler=bundle.scaler,
        feature_columns=bundle.feature_columns,
        total_charges_median=bundle.total_charges_median,
        fit=False,
    )
    probabilities = bundle.model.predict_proba(processed_df)[:, 1]
    predicted_churn = (probabilities >= model_threshold).astype(int)

    results = display_df.copy()
    results["Churn Probability"] = np.round(probabilities, 4)
    results["Predicted Churn"] = predicted_churn
    results["Prediction Label"] = np.where(predicted_churn == 1, "Churn", "Stay")
    results["Risk Tier"] = risk_tier(probabilities)
    results["Threshold Used"] = model_threshold
    results["Run ID"] = f"{run_type[:1].upper()}-{uuid.uuid4().hex[:10]}"
    results["Prediction Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if save_history:
        save_prediction_history(results, run_type=run_type, source_name=source_name)

    return results, str(results["Run ID"].iloc[0])


def validate_batch_columns(dataframe: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [column for column in MODEL_INPUT_COLUMNS if column not in dataframe.columns]
    return len(missing) == 0, missing


def init_history_db() -> None:
    _ensure_directories()
    with sqlite3.connect(HISTORY_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                run_type TEXT NOT NULL,
                source_name TEXT,
                created_at TEXT NOT NULL,
                customer_id TEXT,
                prediction_label TEXT NOT NULL,
                predicted_churn INTEGER NOT NULL,
                churn_probability REAL NOT NULL,
                risk_tier TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                input_payload TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_prediction_history(
    predictions: pd.DataFrame,
    *,
    run_type: str,
    source_name: str | None = None,
) -> None:
    init_history_db()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = []
    for row in predictions.to_dict(orient="records"):
        payload = {column: row.get(column) for column in MODEL_INPUT_COLUMNS + ["CustomerID"]}
        records.append(
            (
                row["Run ID"],
                run_type,
                source_name or "",
                created_at,
                str(row.get("CustomerID", "")),
                row["Prediction Label"],
                int(row["Predicted Churn"]),
                float(row["Churn Probability"]),
                row["Risk Tier"],
                float(row["Threshold Used"]),
                json.dumps(payload),
            )
        )

    with sqlite3.connect(HISTORY_DB_PATH) as connection:
        connection.executemany(
            """
            INSERT INTO prediction_history (
                run_id, run_type, source_name, created_at, customer_id,
                prediction_label, predicted_churn, churn_probability,
                risk_tier, threshold_value, input_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.commit()


def load_history() -> pd.DataFrame:
    init_history_db()
    with sqlite3.connect(HISTORY_DB_PATH) as connection:
        return pd.read_sql_query(
            "SELECT * FROM prediction_history ORDER BY id DESC",
            connection,
        )


def delete_history_rows(record_ids: list[int]) -> int:
    if not record_ids:
        return 0
    init_history_db()
    with sqlite3.connect(HISTORY_DB_PATH) as connection:
        placeholders = ",".join("?" for _ in record_ids)
        cursor = connection.execute(
            f"DELETE FROM prediction_history WHERE id IN ({placeholders})",
            record_ids,
        )
        connection.commit()
        return cursor.rowcount


def delete_history_run(run_id: str) -> int:
    init_history_db()
    with sqlite3.connect(HISTORY_DB_PATH) as connection:
        cursor = connection.execute(
            "DELETE FROM prediction_history WHERE run_id = ?",
            (run_id,),
        )
        connection.commit()
        return cursor.rowcount


def history_summary(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {
            "total_predictions": 0,
            "single_runs": 0,
            "batch_runs": 0,
            "high_risk": 0,
        }
    return {
        "total_predictions": int(len(history)),
        "single_runs": int(history.loc[history["run_type"] == "single", "run_id"].nunique()),
        "batch_runs": int(history.loc[history["run_type"] == "batch", "run_id"].nunique()),
        "high_risk": int((history["risk_tier"] == "High").sum()),
    }
