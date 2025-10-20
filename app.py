# app_no_shap.py — Streamlit Dashboard without SHAP explainability

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import classification_report
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

st.set_page_config(page_title="Credit Risk Bias Dashboard", layout="wide")
st.title("Credit Risk Bias & Fairness Detection System (SHAP Disabled)")

uploaded_file = st.file_uploader("Upload dataset with predictions (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Data Preview", df.head())

    sensitive_col = 'gender'  # Set appropriately
    target_col = 'default'

    if target_col not in df.columns or sensitive_col not in df.columns:
        st.error("Missing required columns: target or sensitive feature")
        st.stop()

    y_true = df[target_col]
    y_pred_baseline = df['y_pred_baseline']
    y_pred_fair = df['y_pred_fair']
    sensitive = df[sensitive_col]

    # Bias metrics
    dp_base = demographic_parity_difference(y_true, y_pred_baseline, sensitive)
    eo_base = equalized_odds_difference(y_true, y_pred_baseline, sensitive)
    dp_fair = demographic_parity_difference(y_true, y_pred_fair, sensitive)
    eo_fair = equalized_odds_difference(y_true, y_pred_fair, sensitive)

    st.subheader("Bias Metrics Before & After Mitigation")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Demographic Parity", "Equalized Odds"],
                         y=[dp_base, eo_base],
                         name="Before Mitigation",
                         marker_color='red'))
    fig.add_trace(go.Bar(x=["Demographic Parity", "Equalized Odds"],
                         y=[dp_fair, eo_fair],
                         name="After Mitigation",
                         marker_color='green'))
    fig.update_layout(barmode='group', yaxis_title="Bias Metric (closer to 0 is better)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Baseline Model")
        st.text(classification_report(y_true, y_pred_baseline))
    with col2:
        st.write("Fair Model")
        st.text(classification_report(y_true, y_pred_fair))

    st.subheader("Group-wise Positive Prediction Rate")
    baseline_rates = df.groupby(sensitive_col)['y_pred_baseline'].mean()
    fair_rates = df.groupby(sensitive_col)['y_pred_fair'].mean()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=baseline_rates.index,
                          y=baseline_rates,
                          name="Before Mitigation",
                          marker_color='red'))
    fig2.add_trace(go.Bar(x=fair_rates.index,
                          y=fair_rates,
                          name="After Mitigation",
                          marker_color='green'))
    fig2.update_layout(barmode='group', yaxis_title="Positive Prediction Rate")
    st.plotly_chart(fig2, use_container_width=True)

    # SHAP explainability disabled
    st.info("Model explainability using SHAP is currently disabled due to installation issues. Re-enable once SHAP is installed.")

else:
    st.info("Please upload your dataset CSV with predictions to view the dashboard.")
