import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import pickle

# Setup
st.set_page_config(page_title="Credit Risk Bias Dashboard", layout="wide")
st.title("Credit Risk Bias & Fairness Detection with Input Form")

# Sidebar upload and inputs
st.sidebar.header("Data Upload & Manual Input")

uploaded_file = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])

st.sidebar.subheader("Manual Applicant Input")
def manual_input():
    age = st.sidebar.slider("Age", 18, 80, 35)
    income = st.sidebar.number_input("Annual Income ($)", 10000, 200000, 50000)
    loan_to_income = st.sidebar.slider("Loan to Income Ratio", 0.1, 5.0, 1.0)
    delinquency_ratio = st.sidebar.slider("Delinquency Ratio", 0.0, 1.0, 0.1)
    avg_dpd_per_delinquency = st.sidebar.slider("Avg DPD per Delinquency", 0.0, 20.0, 5.0)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    
    data = {
        "age": age,
        "income": income,
        "loan_to_income": loan_to_income,
        "delinquency_ratio": delinquency_ratio,
        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
        "gender": gender
    }
    return pd.DataFrame([data])

manual_df = manual_input()

# Load trained model and scaler (replace with your own pickle paths)
@st.cache_resource
def load_model_scaler():
    model = pickle.load(open("best_logistic_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

try:
    model, scaler = load_model_scaler()
except Exception as e:
    st.warning("Model or scaler pickle not found. Please train and save them first.")
    model, scaler = None, None

# Prepare manual input for prediction
if model and scaler:
    # Preprocess manual input
    manual_features = manual_df.drop(columns=["gender"])
    manual_scaled = scaler.transform(manual_features)
    # Assume gender encoded as binary 0/1
    manual_gender = np.array([1 if manual_df.iloc[0]['gender'] == "Female" else 0]).reshape(-1,1)
    manual_input_final = np.hstack((manual_scaled, manual_gender))
    
    score = model.predict_proba(manual_input_final)[:,1][0]
    st.sidebar.markdown(f"### Predicted Default Risk Score: {score:.3f}")

# Show uploaded data if provided
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Extract columns
    y_true = df["default"]
    y_pred_baseline = df["y_pred_baseline"]
    y_pred_fair = df["y_pred_fair"]
    sensitive = df["gender"]  # or make dynamic
    
    # Fairness metrics
    dp_before = demographic_parity_difference(y_true, y_pred_baseline, sensitive)
    dp_after = demographic_parity_difference(y_true, y_pred_fair, sensitive)
    eo_before = equalized_odds_difference(y_true, y_pred_baseline, sensitive)
    eo_after = equalized_odds_difference(y_true, y_pred_fair, sensitive)
    
    st.subheader("Bias Metrics Before & After Mitigation")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Demographic Parity", "Equalized Odds"],
        y=[abs(dp_before), abs(eo_before)],
        name="Before Mitigation",
        marker_color="indianred"
    ))
    fig.add_trace(go.Bar(
        x=["Demographic Parity", "Equalized Odds"],
        y=[abs(dp_after), abs(eo_after)],
        name="After Mitigation",
        marker_color="seagreen"
    ))
    fig.update_layout(barmode="group", yaxis_title="Absolute Bias (Lower Better)")
    st.plotly_chart(fig)

    st.subheader("Prediction Rates by Gender")
    baseline_rates = df.groupby("gender")["y_pred_baseline"].mean()
    fair_rates = df.groupby("gender")["y_pred_fair"].mean()
    groups = baseline_rates.index.tolist()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=groups, y=baseline_rates.values, name="Before Mitigation", marker_color="tomato"))
    fig2.add_trace(go.Bar(x=groups, y=fair_rates.values, name="After Mitigation", marker_color="mediumseagreen"))
    fig2.update_layout(barmode="group", yaxis_title="Positive Prediction Rate")
    st.plotly_chart(fig2)

    st.subheader("Classification Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Baseline Model")
        st.text(classification_report(y_true, y_pred_baseline))
    with col2:
        st.write("Fair Model")
        st.text(classification_report(y_true, y_pred_fair))

else:
    st.info("Upload your dataset CSV with predictions to display bias and performance metrics.")

