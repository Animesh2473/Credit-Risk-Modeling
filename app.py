import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and components
@st.cache_resource
def load_model_components():
    """Load model components - using baseline model instead of fairlearn"""
    try:
        st.info("📦 Loading model components...")
        
        # Try loading the complete package first to extract components
        package_path = 'model/complete_model_package_20251020_113604.pkl'
        baseline_path = 'model/baseline_model_20251020_113604.pkl'
        
        model = None
        scaler = None
        gender_encoder = None
        metadata = None
        feature_columns = None
        scaled_columns = None
        
        # Try baseline model first (doesn't require fairlearn)
        if os.path.exists(baseline_path):
            st.info(f"Loading baseline model from: {baseline_path}")
            model = joblib.load(baseline_path)
            st.success("✅ Baseline model loaded!")
        
        # Load other components
        if os.path.exists('scaler/scaler_for_streamlit.pkl'):
            scaler = joblib.load('scaler/scaler_for_streamlit.pkl')
            st.success("✅ Scaler loaded!")
        
        if os.path.exists('model/gender_encoder_for_streamlit.pkl'):
            gender_encoder = joblib.load('model/gender_encoder_for_streamlit.pkl')
            st.success("✅ Gender encoder loaded!")
        
        if os.path.exists('model/feature_info.pkl'):
            feature_info = joblib.load('model/feature_info.pkl')
            feature_columns = feature_info['feature_columns']
            scaled_columns = feature_info['scaled_columns']
            st.success("✅ Feature info loaded!")
        
        # Load metadata
        metadata_path = 'model_metadata_20251020_113604.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            st.success("✅ Metadata loaded!")
        
        if model is None:
            st.error("❌ Could not load model. Please check file paths.")
            st.info("Available files:")
            if os.path.exists('model'):
                st.write(os.listdir('model'))
            return None, None, None, None, None, None
        
        return model, scaler, gender_encoder, metadata, feature_columns, scaled_columns
        
    except Exception as e:
        st.error(f"❌ Error loading model components: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None, None

def engineer_features(data):
    """Apply feature engineering as done during training"""
    df = data.copy()
    
    # Loan to Income Ratio
    if 'loan_amount' in df.columns and 'income' in df.columns:
        df['loan_to_income'] = (df['loan_amount'] / df['income']).round(2)
    
    # Delinquency Ratio
    if 'delinquent_months' in df.columns and 'total_loan_months' in df.columns:
        df['delinquency_ratio'] = (df['delinquent_months'] * 100 / df['total_loan_months']).round(1)
    
    # Average DPD per Delinquency
    if 'total_dpd' in df.columns and 'delinquent_months' in df.columns:
        df['avg_dpd_per_delinquency'] = np.where(
            df['delinquent_months'] != 0,
            (df['total_dpd'] / df['delinquent_months']).round(1),
            0
        )
    
    return df

def preprocess_input(input_data, gender_encoder, scaler, feature_columns, scaled_columns):
    """Preprocess input data to match training pipeline"""
    try:
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply feature engineering
        df = engineer_features(df)
        
        # Encode gender if encoder exists
        if 'gender' in df.columns and gender_encoder is not None:
            try:
                df['gender'] = gender_encoder.transform(df[['gender']])
            except:
                # Manual encoding if needed
                df['gender'] = df['gender'].map({'M': 0, 'F': 1})
        
        # Drop features that were dropped during training
        features_to_drop = ['disbursal_date', 'installment_start_dt', 'loan_amount', 
                          'income', 'total_loan_months', 'delinquent_months', 'total_dpd']
        df = df.drop(columns=[col for col in features_to_drop if col in df.columns], errors='ignore')
        
        # If feature_columns provided, ensure correct structure
        if feature_columns is not None:
            # Add missing features with default value
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Select and order features correctly
            df = df[feature_columns]
        
        # Scale numerical features
        if scaled_columns is not None and scaler is not None:
            cols_to_scale = [col for col in scaled_columns if col in df.columns]
            if cols_to_scale:
                df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        
        return df
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        raise

# Title
st.markdown('<p class="main-header">🏦 Loan Default Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

# Load model components
with st.spinner("Loading model components..."):
    model, scaler, gender_encoder, metadata, feature_columns, scaled_columns = load_model_components()

if model is None:
    st.error("⚠️ Failed to load model. Please check if all model files exist.")
    st.info("""
    **Required files:**
    - model/baseline_model_20251020_113604.pkl
    - scaler/scaler_for_streamlit.pkl
    - model/gender_encoder_for_streamlit.pkl
    - model/feature_info.pkl
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.header("ℹ️ Model Information")
    
    if metadata:
        st.markdown(f"**Model Type:** Logistic Regression")
        st.markdown(f"**Training Date:** {metadata.get('training_date', 'N/A')}")
        
        st.markdown("### 📊 Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F1-Score", f"{metadata.get('test_f1_score', 0):.3f}")
            st.metric("Precision", f"{metadata.get('test_precision', 0):.3f}")
        with col2:
            st.metric("Recall", f"{metadata.get('test_recall', 0):.3f}")
    else:
        st.info("Using baseline model")

# Main prediction interface
st.header("📝 Enter Loan Application Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Personal Information")
        gender = st.selectbox("Gender", options=["M", "F"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        residence_type = st.selectbox("Residence Type", 
                                     options=["Owned", "Rented", "Mortgage", "Other"])
    
    with col2:
        st.subheader("💰 Financial Information")
        income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=10000, step=1000)
        sanction_amount = st.number_input("Sanction Amount ($)", min_value=1000, value=10000, step=1000)
    
    with col3:
        st.subheader("📋 Credit History")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        total_loan_months = st.number_input("Total Loan Months", min_value=0, value=36)
        delinquent_months = st.number_input("Delinquent Months", min_value=0, value=0)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.subheader("🏦 Loan Details")
        loan_purpose = st.selectbox("Loan Purpose",
                                   options=["Personal", "Business", "Education", 
                                          "Home", "Auto", "Medical", "Other"])
        loan_type = st.selectbox("Loan Type", options=["Unsecured", "Secured"])
    
    with col5:
        st.subheader("💳 Fees")
        processing_fee = st.number_input("Processing Fee ($)", min_value=0.0, value=100.0)
        gst = st.number_input("GST ($)", min_value=0.0, value=18.0)
        net_disbursement = st.number_input("Net Disbursement ($)", min_value=0, value=9882)
    
    with col6:
        st.subheader("📈 Payment History")
        total_dpd = st.number_input("Total Days Past Due", min_value=0, value=0)
        principal_outstanding = st.number_input("Principal Outstanding ($)", min_value=0, value=5000)
    
    submitted = st.form_submit_button("🔮 Predict Default Risk", type="primary", use_container_width=True)

if submitted:
    try:
        with st.spinner("Analyzing application..."):
            # Prepare input
            input_data = {
                'gender': gender,
                'age': age,
                'income': income,
                'loan_amount': loan_amount,
                'sanction_amount': sanction_amount,
                'processing_fee': processing_fee,
                'gst': gst,
                'net_disbursement': net_disbursement,
                'principal_outstanding': principal_outstanding,
                'credit_score': credit_score,
                'loan_purpose': loan_purpose,
                'loan_type': loan_type,
                'residence_type': residence_type,
                'total_loan_months': total_loan_months,
                'delinquent_months': delinquent_months,
                'total_dpd': total_dpd
            }
            
            # Preprocess
            processed_data = preprocess_input(input_data, gender_encoder, scaler,
                                             feature_columns, scaled_columns)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            
            # Get probabilities
            try:
                prediction_proba = model.predict_proba(processed_data)[0]
                default_prob = prediction_proba[1] * 100
                no_default_prob = prediction_proba[0] * 100
            except:
                default_prob = None
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col2:
                if prediction == 1:
                    st.markdown("""
                    <div class="danger-box">
                        <h2 style="text-align: center; color: #dc3545;">⚠️ HIGH RISK</h2>
                        <p style="text-align: center;">Loan Likely to Default</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <h2 style="text-align: center; color: #28a745;">✅ LOW RISK</h2>
                        <p style="text-align: center;">Loan Unlikely to Default</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if default_prob is not None:
                st.markdown("### Probability Breakdown")
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    st.metric("No Default Probability", f"{no_default_prob:.2f}%")
                
                with prob_col2:
                    st.metric("Default Probability", f"{default_prob:.2f}%")
                
                # Risk assessment
                st.markdown("### Risk Assessment")
                if default_prob < 30:
                    st.markdown("""
                    <div class="success-box">
                        <h4>📈 Low Risk Profile</h4>
                        <p>Strong indicators for loan repayment. Recommended for approval.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif default_prob < 60:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>📊 Medium Risk Profile</h4>
                        <p>Mixed indicators. Additional review recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="danger-box">
                        <h4>📉 High Risk Profile</h4>
                        <p>Concerning indicators for potential default. Thorough review required.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Key metrics
                st.markdown("### 📊 Key Financial Ratios")
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    lti = loan_amount / income if income > 0 else 0
                    st.metric("Loan-to-Income", f"{lti:.2f}")
                
                with m2:
                    if total_loan_months > 0:
                        delinq = (delinquent_months / total_loan_months) * 100
                        st.metric("Delinquency Rate", f"{delinq:.1f}%")
                
                with m3:
                    util = (principal_outstanding / loan_amount * 100) if loan_amount > 0 else 0
                    st.metric("Credit Utilization", f"{util:.1f}%")
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>💡 This prediction is based on machine learning and should be used as a decision support tool.</p>
        <p>Always conduct thorough due diligence before making final lending decisions.</p>
    </div>
""", unsafe_allow_html=True)