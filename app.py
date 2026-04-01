from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.churn_backend import (
    DEFAULT_FORM_VALUES,
    DEFAULT_THRESHOLD,
    FEATURE_IMPORTANCE_PATH,
    MODEL_INPUT_COLUMNS,
    SELECT_OPTIONS,
    build_template_dataframe,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    delete_history_rows,
    delete_history_run,
    history_summary,
    load_csv_if_exists,
    load_history,
    train_and_save_model,
    predict_customers,
    validate_batch_columns,
)


MODEL_METRICS_PATH = Path("Powerbi_exports") / "01_model_metrics.csv"
THRESHOLD_ANALYSIS_PATH = Path("Powerbi_exports") / "05_threshold_analysis.csv"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(24, 119, 242, 0.10), transparent 22%),
                linear-gradient(180deg, #f7fbff 0%, #eef6ff 35%, #ffffff 100%);
            color: #0f2747;
        }
        .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f4ea8 0%, #0b66c3 100%); }
        [data-testid="stSidebar"] * { color: white !important; }
        .hero-card, .metric-card, .panel-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(28, 102, 202, 0.10);
            border-radius: 22px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 12px 34px rgba(15, 78, 168, 0.08);
        }
        .hero-title { font-size: 2.1rem; font-weight: 700; color: #0f4ea8; margin-bottom: 0.2rem; }
        .hero-subtitle { color: #45698f; font-size: 1rem; margin-bottom: 0; }
        .risk-pill {
            display: inline-block; border-radius: 999px; padding: 0.35rem 0.75rem;
            font-weight: 700; font-size: 0.88rem;
        }
        .risk-low { background: #e8f4ff; color: #1363c6; }
        .risk-medium { background: #dff1ff; color: #0b66c3; }
        .risk-high { background: #d6e9ff; color: #083d77; }
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(90deg, #0f4ea8 0%, #2391ff 100%);
            color: white; border: 0; border-radius: 12px; font-weight: 600;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(15, 78, 168, 0.08); border-radius: 999px; padding: 0.35rem 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def ensure_bundle():
    return train_and_save_model(force=False)


@st.cache_data(show_spinner=False)
def cached_frame(path_str: str) -> pd.DataFrame:
    return load_csv_if_exists(Path(path_str))


def metric_card(title: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"<div class='metric-card'><h4>{title}</h4><h2>{value}</h2><p>{help_text}</p></div>",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Customer Churn Intelligence Hub</div>
            <p class="hero-subtitle">
                Predict churn for a single customer, score full customer files, monitor history,
                and turn your churn model into a polished business-facing product.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    bundle = ensure_bundle()
    history = load_history()
    summary = history_summary(history)
    metrics = bundle.metadata.get("metrics", {})

    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Single Prediction", "Batch Prediction", "History"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Model Snapshot")
    st.sidebar.metric("Model", bundle.metadata.get("model_name", "Random Forest"))
    st.sidebar.metric("Threshold", f"{bundle.threshold:.2f}")
    st.sidebar.metric("ROC-AUC", metrics.get("roc_auc", "-"))
    st.sidebar.metric("Recall", metrics.get("recall", "-"))
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Usage Snapshot")
    st.sidebar.metric("Predictions Saved", summary["total_predictions"])
    st.sidebar.metric("Single Runs", summary["single_runs"])
    st.sidebar.metric("Batch Runs", summary["batch_runs"])
    st.sidebar.metric("High Risk Cases", summary["high_risk"])
    return page


def create_single_input_dataframe() -> pd.DataFrame:
    with st.form("single_prediction_form", clear_on_submit=False):
        st.markdown("### Customer Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            customer_id = st.text_input("Customer ID", value=DEFAULT_FORM_VALUES["CustomerID"])
            gender = st.selectbox("Gender", SELECT_OPTIONS["Gender"])
            senior = st.selectbox("Senior Citizen", SELECT_OPTIONS["Senior Citizen"])
            partner = st.selectbox("Partner", SELECT_OPTIONS["Partner"], index=1)
            dependents = st.selectbox("Dependents", SELECT_OPTIONS["Dependents"])
            tenure = st.slider("Tenure Months", min_value=0, max_value=72, value=12)
            contract = st.selectbox("Contract", SELECT_OPTIONS["Contract"])
            payment = st.selectbox("Payment Method", SELECT_OPTIONS["Payment Method"], index=2)
        with col2:
            phone = st.selectbox("Phone Service", SELECT_OPTIONS["Phone Service"], index=1)
            multiple = st.selectbox("Multiple Lines", SELECT_OPTIONS["Multiple Lines"])
            internet = st.selectbox("Internet Service", SELECT_OPTIONS["Internet Service"], index=1)
            security = st.selectbox("Online Security", SELECT_OPTIONS["Online Security"])
            backup = st.selectbox("Online Backup", SELECT_OPTIONS["Online Backup"], index=1)
            device = st.selectbox("Device Protection", SELECT_OPTIONS["Device Protection"])
            support = st.selectbox("Tech Support", SELECT_OPTIONS["Tech Support"])
        with col3:
            tv = st.selectbox("Streaming TV", SELECT_OPTIONS["Streaming TV"], index=1)
            movies = st.selectbox("Streaming Movies", SELECT_OPTIONS["Streaming Movies"], index=1)
            paperless = st.selectbox("Paperless Billing", SELECT_OPTIONS["Paperless Billing"], index=1)
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=79.90, step=0.10)
            total = st.number_input("Total Charges", min_value=0.0, value=958.80, step=0.10)
            cltv = st.number_input("CLTV", min_value=0, value=3200, step=50)
            threshold = st.slider(
                "Decision Threshold",
                min_value=0.10,
                max_value=0.90,
                value=float(DEFAULT_THRESHOLD),
                step=0.05,
            )
        submitted = st.form_submit_button("Predict Churn")

    if not submitted:
        return pd.DataFrame()

    st.session_state["single_threshold"] = threshold
    return pd.DataFrame(
        [
            {
                "CustomerID": customer_id,
                "Gender": gender,
                "Senior Citizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "Tenure Months": tenure,
                "Phone Service": phone,
                "Multiple Lines": multiple,
                "Internet Service": internet,
                "Online Security": security,
                "Online Backup": backup,
                "Device Protection": device,
                "Tech Support": support,
                "Streaming TV": tv,
                "Streaming Movies": movies,
                "Contract": contract,
                "Paperless Billing": paperless,
                "Payment Method": payment,
                "Monthly Charges": monthly,
                "Total Charges": total,
                "CLTV": cltv,
            }
        ]
    )


def render_single_prediction_page() -> None:
    st.markdown("## Single Customer Prediction")
    st.caption("Manual scoring flow for one customer, with confidence, business charts, and model insights.")

    input_df = create_single_input_dataframe()
    if input_df.empty:
        if "single_result" in st.session_state:
            results = st.session_state["single_result"].copy()
            record = results.iloc[0]
            risk_class = str(record["Risk Tier"]).lower()
            st.markdown(
                f"""
                <div class="panel-card">
                    <h3>Latest Prediction Result</h3>
                    <p><span class="risk-pill risk-{risk_class}">{record["Risk Tier"]} Risk</span></p>
                    <h2>{record["Prediction Label"]}</h2>
                    <p>Customer <strong>{record["CustomerID"]}</strong> has a churn probability of
                    <strong>{record["Churn Probability"]:.1%}</strong>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        feature_importance = cached_frame(str(FEATURE_IMPORTANCE_PATH))
        if not feature_importance.empty:
            fig = px.bar(
                feature_importance.head(10).sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#cde7ff", "#0f4ea8"],
                title="Top Model Drivers",
            )
            fig.update_layout(height=420, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        return

    threshold = st.session_state.get("single_threshold", DEFAULT_THRESHOLD)
    results, _ = predict_customers(
        input_df,
        threshold=threshold,
        save_history=True,
        run_type="single",
        source_name="manual_form",
    )
    st.session_state["single_result"] = results
    record = results.iloc[0]
    risk_class = str(record["Risk Tier"]).lower()

    st.markdown(
        f"""
        <div class="panel-card">
            <h3>Prediction Result</h3>
            <p><span class="risk-pill risk-{risk_class}">{record["Risk Tier"]} Risk</span></p>
            <h2>{record["Prediction Label"]}</h2>
            <p>Customer <strong>{record["CustomerID"]}</strong> has a churn probability of
            <strong>{record["Churn Probability"]:.1%}</strong> using threshold <strong>{threshold:.2f}</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Churn Probability", f"{record['Churn Probability']:.1%}", "Probability from the churn model.")
    with col2:
        metric_card("Prediction", record["Prediction Label"], "Final decision after thresholding.")
    with col3:
        metric_card("Risk Tier", record["Risk Tier"], "Low <= 30%, Medium <= 60%, High > 60%.")

    chart_col1, chart_col2 = st.columns([1.1, 1])
    with chart_col1:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(record["Churn Probability"]) * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0f4ea8"},
                    "steps": [
                        {"range": [0, 30], "color": "#dceeff"},
                        {"range": [30, 60], "color": "#8ac8ff"},
                        {"range": [60, 100], "color": "#2e7de9"},
                    ],
                    "threshold": {
                        "line": {"color": "#083d77", "width": 4},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
                title={"text": "Churn Probability Gauge"},
            )
        )
        gauge.update_layout(height=350, margin=dict(l=20, r=20, t=70, b=20))
        st.plotly_chart(gauge, width="stretch")
    with chart_col2:
        profile_df = pd.DataFrame(
            {
                "Metric": ["Monthly Charges", "Total Charges", "CLTV", "Tenure Months"],
                "Value": [
                    record["Monthly Charges"],
                    record["Total Charges"],
                    record["CLTV"],
                    record["Tenure Months"],
                ],
            }
        )
        fig = px.bar(
            profile_df,
            x="Metric",
            y="Value",
            color="Metric",
            color_discrete_sequence=["#0f4ea8", "#2391ff", "#6ab7ff", "#badfff"],
            title="Customer Commercial Profile",
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    threshold_df = cached_frame(str(THRESHOLD_ANALYSIS_PATH))
    feature_importance = cached_frame(str(FEATURE_IMPORTANCE_PATH))
    lower_col, upper_col = st.columns(2)
    with lower_col:
        if not threshold_df.empty:
            threshold_fig = px.line(
                threshold_df,
                x="Threshold",
                y=["Precision", "Recall", "F1_Score"],
                markers=True,
                color_discrete_sequence=["#0f4ea8", "#2391ff", "#7bc3ff"],
                title="Threshold Trade-off Reference",
            )
            threshold_fig.add_vline(x=threshold, line_width=2, line_dash="dash", line_color="#083d77")
            threshold_fig.update_layout(height=380)
            st.plotly_chart(threshold_fig, width="stretch")
    with upper_col:
        if not feature_importance.empty:
            importance_fig = px.bar(
                feature_importance.head(8).sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#d6ebff", "#0f4ea8"],
                title="Top Features Behind the Model",
            )
            importance_fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(importance_fig, width="stretch")


def read_uploaded_batch(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file format. Upload CSV or Excel.")


def render_batch_prediction_page() -> None:
    st.markdown("## Batch Prediction")
    st.caption("Upload a customer file, score every row, explore charts, and download the enriched output.")

    sample_df = build_template_dataframe()
    sample_col1, sample_col2 = st.columns([1.2, 1])
    with sample_col1:
        st.markdown("### Expected Upload Columns")
        st.dataframe(sample_df.head(3), width="stretch", hide_index=True)
    with sample_col2:
        st.download_button(
            "Download CSV Template",
            data=dataframe_to_csv_bytes(sample_df),
            file_name="churn_batch_template.csv",
            mime="text/csv",
        )
        try:
            st.download_button(
                "Download Excel Template",
                data=dataframe_to_excel_bytes(sample_df),
                file_name="churn_batch_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            st.info("Excel export becomes available after installing `openpyxl`.")

    uploaded_file = st.file_uploader("Upload customer CSV or Excel file", type=["csv", "xlsx", "xls"])
    threshold = st.slider(
        "Batch Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(DEFAULT_THRESHOLD),
        step=0.05,
        key="batch_threshold",
    )

    if uploaded_file is None:
        return

    try:
        batch_df = read_uploaded_batch(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        return

    is_valid, missing_columns = validate_batch_columns(batch_df)
    if not is_valid:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return

    st.markdown("### Uploaded Preview")
    st.dataframe(batch_df.head(10), width="stretch", hide_index=True)

    if st.button("Run Batch Prediction", type="primary"):
        results, run_id = predict_customers(
            batch_df,
            threshold=threshold,
            save_history=True,
            run_type="batch",
            source_name=uploaded_file.name,
        )
        st.session_state["batch_results"] = results
        st.session_state["batch_run_id"] = run_id

    if "batch_results" not in st.session_state:
        return

    results = st.session_state["batch_results"].copy()
    churn_count = int(results["Predicted Churn"].sum())
    high_risk = int((results["Risk Tier"] == "High").sum())
    avg_prob = float(results["Churn Probability"].mean())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Records Scored", f"{len(results)}", "Total customers processed.")
    with m2:
        metric_card("Predicted Churn", f"{churn_count}", "Customers above the chosen threshold.")
    with m3:
        metric_card("High Risk", f"{high_risk}", "Customers with probability above 60%.")
    with m4:
        metric_card("Average Probability", f"{avg_prob:.1%}", "Average churn likelihood across the file.")

    st.markdown("### Scored Customer File")
    st.dataframe(results, width="stretch", hide_index=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        risk_fig = px.pie(
            results,
            names="Risk Tier",
            title="Risk Tier Mix",
            color="Risk Tier",
            color_discrete_map={"Low": "#badfff", "Medium": "#4ba6ff", "High": "#0f4ea8"},
            hole=0.45,
        )
        risk_fig.update_layout(height=360)
        st.plotly_chart(risk_fig, width="stretch")
    with chart_right:
        hist_fig = px.histogram(
            results,
            x="Churn Probability",
            nbins=20,
            color="Prediction Label",
            color_discrete_map={"Churn": "#0f4ea8", "Stay": "#9dcfff"},
            title="Probability Distribution",
        )
        hist_fig.add_vline(x=threshold, line_dash="dash", line_color="#083d77")
        hist_fig.update_layout(height=360, bargap=0.08)
        st.plotly_chart(hist_fig, width="stretch")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        contract_summary = results.groupby(["Contract", "Prediction Label"]).size().reset_index(name="Customers")
        contract_fig = px.bar(
            contract_summary,
            x="Contract",
            y="Customers",
            color="Prediction Label",
            barmode="group",
            color_discrete_map={"Churn": "#0f4ea8", "Stay": "#8ac8ff"},
            title="Predictions by Contract Type",
        )
        contract_fig.update_layout(height=360)
        st.plotly_chart(contract_fig, width="stretch")
    with lower_right:
        top_risk = results.sort_values("Churn Probability", ascending=False).head(15)
        risk_rank_fig = px.bar(
            top_risk,
            x="CustomerID",
            y="Churn Probability",
            color="Risk Tier",
            color_discrete_map={"Low": "#badfff", "Medium": "#4ba6ff", "High": "#0f4ea8"},
            title="Highest-Risk Customers",
        )
        risk_rank_fig.update_layout(height=360, xaxis_title="Customer ID")
        st.plotly_chart(risk_rank_fig, width="stretch")

    st.download_button(
        "Download Predicted CSV",
        data=dataframe_to_csv_bytes(results),
        file_name="batch_churn_predictions.csv",
        mime="text/csv",
    )
    try:
        st.download_button(
            "Download Predicted Excel",
            data=dataframe_to_excel_bytes(results),
            file_name="batch_churn_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.info("Excel download becomes available after installing `openpyxl`.")


def render_history_page() -> None:
    st.markdown("## Prediction History")
    st.caption("Review all saved single and batch runs, inspect charts, and remove records when needed.")

    history = load_history()
    if history.empty:
        st.info("No history is saved yet. Run a single or batch prediction first.")
        return

    summary = history_summary(history)
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        metric_card("Saved Predictions", f"{summary['total_predictions']}", "Every stored customer-level prediction.")
    with h2:
        metric_card("Single Runs", f"{summary['single_runs']}", "Manual form submissions saved to history.")
    with h3:
        metric_card("Batch Runs", f"{summary['batch_runs']}", "Uploaded customer files processed in bulk.")
    with h4:
        metric_card("High Risk Cases", f"{summary['high_risk']}", "History rows labelled as high risk.")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        selected_types = st.multiselect("Run Type", options=sorted(history["run_type"].unique()), default=sorted(history["run_type"].unique()))
    with filter_col2:
        selected_risk = st.multiselect("Risk Tier", options=sorted(history["risk_tier"].unique()), default=sorted(history["risk_tier"].unique()))
    with filter_col3:
        search_term = st.text_input("Search Customer ID", value="")

    filtered = history.copy()
    if selected_types:
        filtered = filtered[filtered["run_type"].isin(selected_types)]
    if selected_risk:
        filtered = filtered[filtered["risk_tier"].isin(selected_risk)]
    if search_term:
        filtered = filtered[filtered["customer_id"].str.contains(search_term, case=False, na=False)]

    chart_left, chart_right = st.columns(2)
    with chart_left:
        trend = filtered.copy()
        trend["date"] = pd.to_datetime(trend["created_at"]).dt.date
        trend_df = trend.groupby(["date", "run_type"]).size().reset_index(name="Predictions")
        trend_fig = px.area(
            trend_df,
            x="date",
            y="Predictions",
            color="run_type",
            color_discrete_map={"single": "#8ac8ff", "batch": "#0f4ea8"},
            title="Prediction Volume Over Time",
        )
        trend_fig.update_layout(height=340)
        st.plotly_chart(trend_fig, width="stretch")
    with chart_right:
        mix_fig = px.histogram(
            filtered,
            x="risk_tier",
            color="run_type",
            barmode="group",
            color_discrete_map={"single": "#8ac8ff", "batch": "#0f4ea8"},
            title="History Mix by Risk Tier",
        )
        mix_fig.update_layout(height=340, xaxis_title="Risk Tier")
        st.plotly_chart(mix_fig, width="stretch")

    display = filtered[
        ["id", "run_id", "run_type", "source_name", "created_at", "customer_id", "prediction_label", "churn_probability", "risk_tier", "threshold_value"]
    ].copy()
    display["Delete"] = False
    st.markdown("### Saved Records")
    edited = st.data_editor(
        display,
        width="stretch",
        hide_index=True,
        disabled=["id", "run_id", "run_type", "source_name", "created_at", "customer_id", "prediction_label", "churn_probability", "risk_tier", "threshold_value"],
        column_config={
            "Delete": st.column_config.CheckboxColumn("Delete", help="Select rows to remove."),
            "churn_probability": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    delete_col1, delete_col2 = st.columns([1, 1.2])
    with delete_col1:
        rows_to_delete = edited.loc[edited["Delete"], "id"].tolist()
        if st.button("Delete Selected Rows"):
            deleted = delete_history_rows([int(row_id) for row_id in rows_to_delete])
            if deleted:
                st.success(f"Deleted {deleted} history rows.")
                st.rerun()
            st.warning("No rows were selected.")
    with delete_col2:
        selected_run = st.selectbox("Delete Entire Run", options=[""] + sorted(filtered["run_id"].unique()))
        if st.button("Delete Selected Run"):
            if selected_run:
                deleted = delete_history_run(selected_run)
                st.success(f"Deleted {deleted} rows from run {selected_run}.")
                st.rerun()
            st.warning("Choose a run before deleting.")


def render_footer() -> None:
    metrics_df = cached_frame(str(MODEL_METRICS_PATH))
    if metrics_df.empty:
        return
    with st.expander("Model Benchmark Snapshot"):
        st.dataframe(metrics_df, width="stretch", hide_index=True)


def main() -> None:
    inject_styles()
    ensure_bundle()
    render_hero()
    page = render_sidebar()
    if page == "Single Prediction":
        render_single_prediction_page()
    elif page == "Batch Prediction":
        render_batch_prediction_page()
    else:
        render_history_page()
    render_footer()


if __name__ == "__main__":
    main()
