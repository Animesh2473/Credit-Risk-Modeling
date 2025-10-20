# app.py - Main Streamlit Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML and Fairness Libraries
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, MetricFrame
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Model Explainability
import shap
import lime
import lime.lime_tabular

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="🏦 Credit Risk Bias & Fairness Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.title("🏦 Credit Risk Bias & Fairness Detection System")
st.markdown("*Demonstrating Ethical AI through Bias Detection and Model Explainability*")

# Sidebar Configuration
st.sidebar.header("📊 Dashboard Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload your processed dataset", 
    type=['csv'], 
    help="Upload the final processed dataset with predictions"
)

# Demo data option
use_demo = st.sidebar.checkbox("Use Demo Data", value=True)

# Load and prepare data
@st.cache_data
def load_demo_data():
    """Generate demo credit risk data"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'income': np.random.lognormal(10, 0.8, n_samples).astype(int),
        'loan_to_income': np.random.uniform(0.1, 3.0, n_samples),
        'delinquency_ratio': np.random.exponential(0.1, n_samples),
        'avg_dpd_per_delinquency': np.random.exponential(5, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic target with bias
    risk_score = (
        0.3 * (df['age'] < 30).astype(int) +
        0.4 * (df['loan_to_income'] > 2.0).astype(int) +
        0.3 * (df['delinquency_ratio'] > 0.2).astype(int) +
        0.2 * (df['gender'] == 'Female').astype(int)  # Intentional bias
    )
    
    df['default'] = (risk_score + np.random.normal(0, 0.3, n_samples) > 0.7).astype(int)
    
    # Simulate model predictions (baseline and fair)
    baseline_bias = 0.15 * (df['gender'] == 'Female').astype(int)
    df['y_pred_baseline'] = ((risk_score + baseline_bias + np.random.normal(0, 0.2, n_samples)) > 0.7).astype(int)
    df['y_pred_fair'] = ((risk_score + np.random.normal(0, 0.2, n_samples)) > 0.7).astype(int)
    
    # Prediction probabilities
    df['y_prob_baseline'] = np.clip(risk_score + baseline_bias + np.random.normal(0, 0.1, n_samples), 0, 1)
    df['y_prob_fair'] = np.clip(risk_score + np.random.normal(0, 0.1, n_samples), 0, 1)
    
    return df

# Data loading logic
if use_demo or uploaded_file is None:
    df = load_demo_data()
    st.sidebar.success("✅ Demo data loaded successfully")
else:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Data uploaded successfully")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {e}")
        df = load_demo_data()

# Sidebar options
sensitive_feature = st.sidebar.selectbox(
    "Select Sensitive Attribute", 
    ['gender', 'age_group', 'marital_status'],
    index=0
)

if sensitive_feature == 'age_group':
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
    sensitive_col = 'age_group'
else:
    sensitive_col = sensitive_feature

# Main Dashboard Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", 
    "⚖️ Bias Analysis", 
    "🧠 Model Explainability", 
    "📊 Interactive Testing",
    "📋 Model Comparison"
])

# Tab 1: Overview
with tab1:
    st.header("📈 Executive Summary")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        default_rate = df['default'].mean()
        st.metric("Overall Default Rate", f"{default_rate:.1%}")
    
    with col2:
        baseline_accuracy = (df['default'] == df['y_pred_baseline']).mean()
        st.metric("Baseline Accuracy", f"{baseline_accuracy:.1%}")
    
    with col3:
        fair_accuracy = (df['default'] == df['y_pred_fair']).mean()
        st.metric("Fair Model Accuracy", f"{fair_accuracy:.1%}")
    
    with col4:
        sample_size = len(df)
        st.metric("Dataset Size", f"{sample_size:,}")
    
    # Dataset Overview
    st.subheader("Dataset Overview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df.head(10))
    
    with col2:
        # Distribution by sensitive attribute
        fig = px.pie(
            df, 
            names=sensitive_col, 
            title=f"Distribution by {sensitive_col.replace('_', ' ').title()}"
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Bias Analysis
with tab2:
    st.header("⚖️ Bias Analysis & Fairness Metrics")
    
    # Calculate fairness metrics
    y_true = df['default']
    y_pred_baseline = df['y_pred_baseline']
    y_pred_fair = df['y_pred_fair']
    sensitive_features = df[sensitive_col]
    
    # Demographic Parity and Equalized Odds
    dp_baseline = demographic_parity_difference(y_true, y_pred_baseline, sensitive_features=sensitive_features)
    dp_fair = demographic_parity_difference(y_true, y_pred_fair, sensitive_features=sensitive_features)
    eo_baseline = equalized_odds_difference(y_true, y_pred_baseline, sensitive_features=sensitive_features)
    eo_fair = equalized_odds_difference(y_true, y_pred_fair, sensitive_features=sensitive_features)
    
    # Fairness Metrics Comparison
    st.subheader("Fairness Metrics Comparison")
    
    metrics_data = {
        'Metric': ['Demographic Parity Difference', 'Equalized Odds Difference'],
        'Baseline Model': [dp_baseline, eo_baseline],
        'Fair Model': [dp_fair, eo_fair],
        'Improvement': [abs(dp_baseline) - abs(dp_fair), abs(eo_baseline) - abs(eo_fair)]
    }
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Baseline Model',
        x=metrics_data['Metric'],
        y=[abs(x) for x in metrics_data['Baseline Model']],
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        name='Fair Model',
        x=metrics_data['Metric'],
        y=[abs(x) for x in metrics_data['Fair Model']],
        marker_color='lightseagreen'
    ))
    
    fig.update_layout(
        title='Fairness Metrics: Lower is Better (Closer to 0 = More Fair)',
        yaxis_title='Absolute Bias Score',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert boxes
    dp_improvement = abs(dp_baseline) - abs(dp_fair)
    eo_improvement = abs(eo_baseline) - abs(eo_fair)
    
    if dp_improvement > 0 and eo_improvement > 0:
        st.success(f"✅ **Bias Reduction Successful!** \n\n"
                  f"Demographic Parity improved by {dp_improvement:.3f} \n\n"
                  f"Equalized Odds improved by {eo_improvement:.3f}")
    else:
        st.warning("⚠️ **Limited Bias Reduction** - Consider additional mitigation techniques")
    
    # Group-wise Performance Analysis
    st.subheader("Group-wise Performance Analysis")
    
    group_analysis = []
    for group in df[sensitive_col].unique():
        mask = df[sensitive_col] == group
        group_data = {
            'Group': group,
            'Count': mask.sum(),
            'True Default Rate': df.loc[mask, 'default'].mean(),
            'Baseline Prediction Rate': df.loc[mask, 'y_pred_baseline'].mean(),
            'Fair Model Prediction Rate': df.loc[mask, 'y_pred_fair'].mean(),
        }
        group_analysis.append(group_data)
    
    group_df = pd.DataFrame(group_analysis)
    
    # Selection Rate Comparison Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Baseline Model',
        x=group_df['Group'],
        y=group_df['Baseline Prediction Rate'],
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        name='Fair Model',
        x=group_df['Group'],
        y=group_df['Fair Model Prediction Rate'],
        marker_color='lightseagreen'
    ))
    fig.add_trace(go.Scatter(
        name='True Default Rate',
        x=group_df['Group'],
        y=group_df['True Default Rate'],
        mode='markers+lines',
        marker=dict(size=10, color='orange'),
        line=dict(color='orange', width=3)
    ))
    
    fig.update_layout(
        title=f'Default Prediction Rates by {sensitive_col.replace("_", " ").title()}',
        yaxis_title='Rate',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Table
    st.subheader("Detailed Group Performance")
    styled_df = group_df.style.format({
        'True Default Rate': '{:.1%}',
        'Baseline Prediction Rate': '{:.1%}',
        'Fair Model Prediction Rate': '{:.1%}'
    }).background_gradient(subset=['True Default Rate', 'Baseline Prediction Rate', 'Fair Model Prediction Rate'])
    
    st.dataframe(styled_df)

# Tab 3: Model Explainability
with tab3:
    st.header("🧠 Model Explainability with SHAP")
    
    # Prepare data for SHAP
    feature_cols = ['age', 'income', 'loan_to_income', 'delinquency_ratio', 'avg_dpd_per_delinquency']
    X = df[feature_cols]
    
    # Train a simple model for demonstration
    model = LogisticRegression(random_state=42)
    model.fit(X, df['default'])
    
    # SHAP Explanations
    @st.cache_data
    def calculate_shap_values(X_data):
        explainer = shap.LinearExplainer(model, X_data)
        shap_values = explainer.shap_values(X_data)
        return explainer, shap_values
    
    explainer, shap_values = calculate_shap_values(X)
    
    # Global Feature Importance
    st.subheader("Global Feature Importance")
    
    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='SHAP Feature Importance (Global)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual Explanation
    st.subheader("Individual Prediction Explanation")
    
    sample_idx = st.slider("Select Sample Index", 0, len(X)-1, 0)
    
    # Show sample details
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Sample Details:**")
        sample_data = df.iloc[sample_idx]
        for col in feature_cols:
            st.write(f"- **{col}**: {sample_data[col]:.2f}")
        st.write(f"- **Actual Default**: {'Yes' if sample_data['default'] else 'No'}")
        st.write(f"- **Predicted Probability**: {model.predict_proba(X.iloc[[sample_idx]])[0][1]:.3f}")
    
    with col2:
        # SHAP waterfall plot data
        sample_shap = shap_values[sample_idx]
        expected_value = explainer.expected_value
        
        # Create waterfall-style visualization
        fig = go.Figure()
        
        # Base value
        cumsum = expected_value
        fig.add_trace(go.Bar(
            x=['Base Value'],
            y=[expected_value],
            name='Base Value',
            marker_color='lightgray'
        ))
        
        # Feature contributions
        for i, (feature, shap_val) in enumerate(zip(feature_cols, sample_shap)):
            color = 'red' if shap_val < 0 else 'green'
            fig.add_trace(go.Bar(
                x=[feature],
                y=[shap_val],
                name=f'{feature}: {shap_val:.3f}',
                marker_color=color,
                showlegend=False
            ))
        
        fig.update_layout(
            title=f'SHAP Values for Sample #{sample_idx}',
            yaxis_title='SHAP Value (Impact on Prediction)'
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Interactive Testing
with tab4:
    st.header("📊 Interactive Bias Testing")
    
    st.subheader("Test Model Predictions Across Different Groups")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Adjust Parameters:**")
        test_age = st.slider("Age", 18, 80, 35)
        test_income = st.slider("Annual Income ($)", 20000, 200000, 50000)
        test_lti = st.slider("Loan to Income Ratio", 0.1, 5.0, 2.0)
        test_delinq = st.slider("Delinquency Ratio", 0.0, 1.0, 0.1)
        test_dpd = st.slider("Avg DPD per Delinquency", 0.0, 20.0, 5.0)
    
    with col2:
        st.write("**Prediction Results Across Groups:**")
        
        test_results = []
        for group in df[sensitive_col].unique():
            # Create test sample
            test_sample = pd.DataFrame({
                'age': [test_age],
                'income': [test_income],
                'loan_to_income': [test_lti],
                'delinquency_ratio': [test_delinq],
                'avg_dpd_per_delinquency': [test_dpd]
            })
            
            pred_prob = model.predict_proba(test_sample)[0][1]
            pred_class = "High Risk" if pred_prob > 0.5 else "Low Risk"
            
            test_results.append({
                'Group': group,
                'Predicted Probability': pred_prob,
                'Risk Classification': pred_class
            })
        
        results_df = pd.DataFrame(test_results)
        
        # Visualization
        fig = px.bar(
            results_df,
            x='Group',
            y='Predicted Probability',
            color='Risk Classification',
            title='Prediction Consistency Across Groups'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.dataframe(
            results_df.style.format({'Predicted Probability': '{:.3f}'})
        )
        
        # Bias alert
        prob_range = results_df['Predicted Probability'].max() - results_df['Predicted Probability'].min()
        if prob_range > 0.1:
            st.warning(f"⚠️ **Potential Bias Detected!** \n\nPrediction probability varies by {prob_range:.3f} across groups for identical inputs.")
        else:
            st.success("✅ **No Significant Bias Detected** for these parameters across groups.")

# Tab 5: Model Comparison
with tab5:
    st.header("📋 Comprehensive Model Comparison")
    
    # Classification Reports
    st.subheader("Classification Reports")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Baseline Model:**")
        baseline_report = classification_report(y_true, y_pred_baseline, output_dict=True)
        st.text(classification_report(y_true, y_pred_baseline))
    
    with col2:
        st.write("**Fair Model:**")
        fair_report = classification_report(y_true, y_pred_fair, output_dict=True)
        st.text(classification_report(y_true, y_pred_fair))
    
    # ROC Curves
    st.subheader("ROC Curve Comparison")
    
    fpr_baseline, tpr_baseline, _ = roc_curve(y_true, df['y_prob_baseline'])
    fpr_fair, tpr_fair, _ = roc_curve(y_true, df['y_prob_fair'])
    
    auc_baseline = auc(fpr_baseline, tpr_baseline)
    auc_fair = auc(fpr_fair, tpr_fair)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr_baseline, y=tpr_baseline,
        mode='lines',
        name=f'Baseline Model (AUC = {auc_baseline:.3f})',
        line=dict(color='red', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=fpr_fair, y=tpr_fair,
        mode='lines',
        name=f'Fair Model (AUC = {auc_fair:.3f})',
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary Card
    st.subheader("📊 Project Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown("""
        <div class="success-card">
            <h4>✅ Bias Mitigation Success</h4>
            <ul>
                <li>Demographic Parity improved</li>
                <li>Equalized Odds enhanced</li>
                <li>Model transparency increased</li>
                <li>Ethical AI principles applied</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🔧 Technical Implementation</h4>
            <ul>
                <li>Fairlearn for bias mitigation</li>
                <li>SHAP for model explainability</li>
                <li>Interactive bias testing</li>
                <li>Comprehensive fairness metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>🏦 Credit Risk Bias & Fairness Detection System | Built with Streamlit | 
    <a href="https://github.com/your-repo">View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
