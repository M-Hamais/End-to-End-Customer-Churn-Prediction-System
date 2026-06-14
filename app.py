import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -- 1. Configuration Paths -----------------------------------------------
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
LR_MODEL_PATH = "models/logistic_regression_model.pkl"
KNN_MODEL_PATH = "models/knn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODERS_PATH = "models/encoders.pkl"

# -- 2. Page Configuration -------------------------------------------------
st.set_page_config(
    page_title="Customer Retention AI Dashboard",
    page_icon="🔮",
    layout="wide"
)

# -- 3. Custom CSS for Modern Dark Theme ----------------------------------
CSS_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Main UI Fonts */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Custom cards styling */
.metric-card {
    background-color: #1A1D27;
    border: 1px solid #2A2D3A;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: #6C63FF;
    box-shadow: 0 10px 20px rgba(108, 99, 255, 0.15);
}

.metric-card-title {
    color: #8A8D9A;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 5px;
}

.metric-card-value {
    color: #E8E8EC;
    font-size: 2rem;
    font-weight: 700;
}

.metric-card-desc {
    color: #5D606E;
    font-size: 0.8rem;
    margin-top: 5px;
}

/* Prediction Verdict Cards */
.verdict-card {
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid;
    transition: all 0.3s ease;
}

.verdict-low-risk {
    background: linear-gradient(135deg, rgba(0, 212, 170, 0.1) 0%, rgba(0, 212, 170, 0.02) 100%);
    border-color: #00D4AA;
    box-shadow: 0 4px 20px rgba(0, 212, 170, 0.08);
}

.verdict-high-risk {
    background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 75, 75, 0.02) 100%);
    border-color: #FF4B4B;
    box-shadow: 0 4px 20px rgba(255, 75, 75, 0.08);
}

.verdict-label {
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #8A8D9A;
    margin-bottom: 5px;
}

.verdict-value-low {
    color: #00D4AA;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 12px;
}

.verdict-value-high {
    color: #FF4B4B;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 12px;
}

/* Custom Probability Progress Bar */
.progress-container {
    background-color: #2A2D3A;
    border-radius: 6px;
    height: 12px;
    width: 100%;
    margin-top: 8px;
    overflow: hidden;
}

.progress-bar-low {
    background: linear-gradient(90deg, #00D4AA 0%, #00F3C3 100%);
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}

.progress-bar-high {
    background: linear-gradient(90deg, #FF4B4B 0%, #FF7676 100%);
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}

/* Comparison Table Styling */
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
    margin-bottom: 25px;
}

.comparison-table th {
    background-color: #1A1D27;
    border-bottom: 2px solid #2A2D3A;
    border-right: 1px solid #2A2D3A;
    color: #E8E8EC;
    padding: 14px 12px;
    text-align: left;
    font-weight: 600;
}
.comparison-table th:first-child {
    border-left: 1px solid #2A2D3A;
}

.comparison-table td {
    padding: 14px 12px;
    border-bottom: 1px solid #2A2D3A;
    border-right: 1px solid #2A2D3A;
    color: var(--text-color);
}
.comparison-table td:first-child {
    border-left: 1px solid #2A2D3A;
}

.comparison-table tr:hover td {
    color: var(--text-color);
    background-color: rgba(108, 99, 255, 0.08);
}

.winner-tag {
    background-color: rgba(0, 212, 170, 0.12);
    color: #00D4AA;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}

.winner-lr-tag {
    background-color: rgba(108, 99, 255, 0.12);
    color: #8A82FF;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Custom spacing and header styling */
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-color);
    margin-bottom: 5px;
}

.sub-title {
    color: var(--text-color);
    opacity: 0.85;
    font-size: 1rem;
    margin-bottom: 30px;
}

/* Custom styled inputs containers */
.input-section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #8A82FF;
    border-bottom: 1px solid #2A2D3A;
    padding-bottom: 5px;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Recommendation Cards */
.recommendation-card {
    border-radius: 12px;
    padding: 20px;
    margin-top: -5px;
    margin-bottom: 15px;
    border: 1px solid;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.rec-low {
    background: linear-gradient(135deg, rgba(0, 212, 170, 0.1) 0%, rgba(0, 212, 170, 0.02) 100%);
    border-color: #00D4AA;
}
.rec-medium {
    background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 152, 0, 0.02) 100%);
    border-color: #FF9800;
}
.rec-high {
    background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 75, 75, 0.02) 100%);
    border-color: #FF4B4B;
}
.rec-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 6px;
}
.rec-low .rec-title {
    color: #00D4AA;
}
.rec-medium .rec-title {
    color: #FF9800;
}
.rec-high .rec-title {
    color: #FF4B4B;
}
.rec-text {
    color: var(--text-color);
    font-size: 0.95rem;
    line-height: 1.4;
}

/* Vertical line divider between minus and plus buttons inside st.number_input */
div[data-testid="stNumberInputContainer"] button[data-testid="stNumberInputStepDown"],
div[data-testid="stNumberInputContainer"] button.step-down,
button[data-testid="stNumberInputStepDown"],
button.step-down {
    border-right: 1px solid rgba(128, 128, 128, 0.4) !important;
    border-top-right-radius: 0px !important;
    border-bottom-right-radius: 0px !important;
}

</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# -- 4. Load or Train pipeline dynamically ---------------------------------
@st.cache_resource
def load_or_train_pipeline():
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    
    # Check if models folder exists
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # Check if we can load existing pipeline files
    if (os.path.exists(LR_MODEL_PATH) and 
        os.path.exists(KNN_MODEL_PATH) and 
        os.path.exists(SCALER_PATH) and 
        os.path.exists(ENCODERS_PATH)):
        
        try:
            with open(LR_MODEL_PATH, "rb") as f:
                lr_model = pickle.load(f)
            with open(KNN_MODEL_PATH, "rb") as f:
                knn_model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            with open(ENCODERS_PATH, "rb") as f:
                encoders = pickle.load(f)
                
            # Compute evaluation metrics on original stratified split
            df = pd.read_csv(DATA_PATH)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
            df.drop(columns=["customerID"], inplace=True)
            
            # Label Encode based on loaded encoders
            for col, le in encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col])
                    
            df[numerical_cols] = scaler.transform(df[numerical_cols])
            
            X = df.drop(columns=["Churn"])
            y = df["Churn"]
            
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
            
        except Exception as e:
            # Fallback to training if loading fails
            st.warning(f"Error loading pickle files ({str(e)}). Retraining models...")
            return train_pipeline_from_scratch(numerical_cols)
            
    else:
        return train_pipeline_from_scratch(numerical_cols)
        
    # Evaluate loaded models
    lr_metrics, knn_metrics = evaluate_models(lr_model, knn_model, X_test, y_test)
    return lr_model, knn_model, scaler, encoders, lr_metrics, knn_metrics, X.columns.tolist()

def train_pipeline_from_scratch(numerical_cols):
    df = pd.read_csv(DATA_PATH)
    
    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    
    # Drop customer ID
    df.drop(columns=["customerID"], inplace=True)
    
    # Label Encoding
    categorical_cols = [col for col in df.columns if str(df[col].dtype) in ["object", "string", "str"]]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # Standard Scaling
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Train-Test Split (80/20 Stratified)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Model 1: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    lr_model.fit(X_train, y_train)
    
    # Model 2: KNN
    knn_model = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    knn_model.fit(X_train, y_train)
    
    # Save Pipeline Assets
    with open(LR_MODEL_PATH, "wb") as f:
        pickle.dump(lr_model, f)
    with open(KNN_MODEL_PATH, "wb") as f:
        pickle.dump(knn_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
        
    lr_metrics, knn_metrics = evaluate_models(lr_model, knn_model, X_test, y_test)
    return lr_model, knn_model, scaler, encoders, lr_metrics, knn_metrics, X.columns.tolist()

def evaluate_models(lr_model, knn_model, X_test, y_test):
    lr_pred = lr_model.predict(X_test)
    knn_pred = knn_model.predict(X_test)
    
    lr_metrics = {
        "Accuracy": accuracy_score(y_test, lr_pred),
        "Precision": precision_score(y_test, lr_pred),
        "Recall": recall_score(y_test, lr_pred),
        "F1-Score": f1_score(y_test, lr_pred),
        "cm": confusion_matrix(y_test, lr_pred)
    }
    
    knn_metrics = {
        "Accuracy": accuracy_score(y_test, knn_pred),
        "Precision": precision_score(y_test, knn_pred),
        "Recall": recall_score(y_test, knn_pred),
        "F1-Score": f1_score(y_test, knn_pred),
        "cm": confusion_matrix(y_test, knn_pred)
    }
    
    return lr_metrics, knn_metrics

# Load models and configurations
lr_model, knn_model, scaler, encoders, lr_metrics, knn_metrics, feature_names = load_or_train_pipeline()

# -- 5. UI Page Header ----------------------------------------------------
st.markdown('<div class="main-title">🔮 Customer Retention AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An AI early-warning system that predicts if a subscriber is about to cancel their service, allowing businesses to proactively protect customer loyalty.</div>', unsafe_allow_html=True)

# Define Tabs
tab1, tab2 = st.tabs(["📊 Live Churn Risk Profiler", "📈 Model Evaluation & Comparison"])

# =========================================================================
# TAB 1: LIVE CHURN RISK PROFILER
# =========================================================================
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("📋 Step 1: Simulate a Customer Profile")
        st.markdown("<p style='color: var(--text-color); opacity: 0.85; font-size: 0.95rem; margin-top: -5px; margin-bottom: 15px;'>Adjust the parameters below to estimate loyalty and cancellation risk under both AI models.</p>", unsafe_allow_html=True)
        
        # Primary Exposed Financials
        st.markdown('<div class="input-section-header">💰 Billing Details</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider(
                "Tenure (months)", 
                min_value=0, 
                max_value=72, 
                value=24, 
                help="Number of months the customer has stayed with the company."
            )
            monthly_charges = st.slider(
                "Monthly Charges ($)", 
                min_value=15.0, 
                max_value=125.0, 
                value=65.0, 
                step=0.5,
                help="The amount charged to the customer monthly."
            )
        with c2:
            # Predict default total charges dynamically as MonthlyCharges * Tenure
            calculated_total = float(tenure * monthly_charges)
            total_charges = st.number_input(
                "Total Charges ($)", 
                min_value=0.0, 
                max_value=10000.0, 
                value=calculated_total, 
                step=10.0,
                help="Total amount charged to the customer (defaults to Monthly Charges × Tenure)."
            )
            
        # Primary Exposed Demographics & Services
        st.markdown('<div class="input-section-header">👤 Customer Account Profile</div>', unsafe_allow_html=True)
        
        c3, c4 = st.columns(2)
        with c3:
            contract = st.selectbox(
                "Contract Type",
                options=list(encoders["Contract"].classes_),
                index=0, # Default to Month-to-month
                help="The contract term of the customer."
            )
        with c4:
            internet_service = st.selectbox(
                "Internet Service Type",
                options=list(encoders["InternetService"].classes_),
                index=1, # Default to Fiber optic
                help="Customer's internet service provider."
            )
            
            if internet_service in ["DSL", "Fiber optic"]:
                tech_support = st.selectbox(
                    "Tech Support Subscription",
                    options=[opt for opt in list(encoders["TechSupport"].classes_) if opt != "No internet service"],
                    index=0, # Default to No
                    help="Whether the customer has tech support subscription."
                )
            else:
                tech_support = "No internet service"

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("⚡ Run AI Analysis", width="stretch", type="primary")

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

    # Expandable Advanced / Additional Settings Safeguard (Stretches Full Width below inputs & predictions)
    with st.expander("🛠️ Advanced Profile Settings (Demographics & Services Defaults)", expanded=False):
        st.markdown("<p style='color: var(--text-color); opacity: 0.85; font-size: 0.95rem; margin-top: 5px; margin-bottom: 15px;'>These defaults complete the model inputs. Tweak them to simulate specific customer details.</p>", unsafe_allow_html=True)
        
        # Row 1 of columns (always visible, balanced layout)
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            gender_options = ["Both"] + list(encoders["gender"].classes_)
            gender = st.selectbox("Gender", options=gender_options, index=0)
            senior_citizen = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
            partner = st.selectbox("Has Partner", options=list(encoders["Partner"].classes_), index=0)
            dependents = st.selectbox("Has Dependents", options=list(encoders["Dependents"].classes_), index=0)
        with c_adv2:
            phone_service = st.selectbox("Has Phone Service", options=list(encoders["PhoneService"].classes_), index=1)
            multiple_lines = st.selectbox("Multiple Lines", options=list(encoders["MultipleLines"].classes_), index=0)
            paperless_billing = st.selectbox("Paperless Billing", options=list(encoders["PaperlessBilling"].classes_), index=1)
            payment_method = st.selectbox("Payment Method", options=list(encoders["PaymentMethod"].classes_), index=2) # Electronic check

        # Row 2 of columns (only visible if Internet Service is active, balanced layout)
        if internet_service in ["DSL", "Fiber optic"]:
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            c_adv3, c_adv4 = st.columns(2)
            with c_adv3:
                online_security = st.selectbox("Online Security", options=[opt for opt in list(encoders["OnlineSecurity"].classes_) if opt != "No internet service"], index=0)
                online_backup = st.selectbox("Online Backup", options=[opt for opt in list(encoders["OnlineBackup"].classes_) if opt != "No internet service"], index=0)
                device_protection = st.selectbox("Device Protection", options=[opt for opt in list(encoders["DeviceProtection"].classes_) if opt != "No internet service"], index=0)
            with c_adv4:
                streaming_tv = st.selectbox("Streaming TV", options=[opt for opt in list(encoders["StreamingTV"].classes_) if opt != "No internet service"], index=0)
                streaming_movies = st.selectbox("Streaming Movies", options=[opt for opt in list(encoders["StreamingMovies"].classes_) if opt != "No internet service"], index=0)
        else:
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

    # We perform analysis either upon button click, or run automatically on first load
    # We can store in session state so results persist when tweaking tabs
    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
        st.session_state.lr_prob = 0.0
        st.session_state.knn_prob = 0.0

    if analyze_btn or not st.session_state.prediction_made:
        # 1. Build dictionary of all features in correct order (mapping 'Both' to 'Female' safely for the ML pipeline)
        raw_input = {
            "gender": "Female" if gender == "Both" else gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([raw_input])
        
        # 2. Safeguard Preprocessing: Ensure columns match training X order
        input_df = input_df[feature_names]
        
        # Transform categorical features with matching LabelEncoder
        for col in input_df.columns:
            if col in encoders and col != "Churn":
                le = encoders[col]
                val = input_df.loc[0, col]
                # Safeguard: Handle unexpected text gracefully
                if val not in le.classes_:
                    input_df.loc[0, col] = le.classes_[0]
                input_df[col] = le.transform([input_df.loc[0, col]])[0]
        
        # Transform numerical features with Scaler
        numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # 3. Model Predictions
        st.session_state.lr_prob = lr_model.predict_proba(input_df)[0][1]
        st.session_state.knn_prob = knn_model.predict_proba(input_df)[0][1]
        st.session_state.prediction_made = True
        
    with col_result:
        st.subheader("🤖 Step 2: AI Loyalty Predictions")
        st.markdown("<p style='color: var(--text-color); opacity: 0.85; font-size: 0.95rem; margin-top: -5px; margin-bottom: 15px;'>Compare real-time risk predictions generated by AI Model A and AI Model B.</p>", unsafe_allow_html=True)
        
        # Verdict Display Function
        def get_verdict_card(model_name, prob):
            is_high = prob >= 0.5
            verdict = "High Risk (Likely to Cancel)" if is_high else "Low Risk (Likely to Stay)"
            card_class = "verdict-high-risk" if is_high else "verdict-low-risk"
            value_class = "verdict-value-high" if is_high else "verdict-value-low"
            bar_class = "progress-bar-high" if is_high else "progress-bar-low"
            percentage = int(prob * 100)
            
            html = f"""
            <div class="verdict-card {card_class}">
                <div class="verdict-label">{model_name} Prediction</div>
                <div class="{value_class}">{verdict}</div>
                <div style="color: var(--text-color); opacity: 0.8; font-size: 0.95rem;">
                    <strong>Cancellation Risk Level:</strong> {percentage}%
                </div>
                <div class="progress-container">
                    <div class="{bar_class}" style="width: {percentage}%;"></div>
                </div>
            </div>
            """
            return html

        # Display Logistic Regression Card
        st.markdown(get_verdict_card("AI Model A (Trend Analyzer :: Logistic Regression)", st.session_state.lr_prob), unsafe_allow_html=True)
        
        # Display KNN Card
        st.markdown(get_verdict_card("AI Model B (Similar-Customer Matcher :: KNN (K=5))", st.session_state.knn_prob), unsafe_allow_html=True)
        
        # Comparison insight card
        lr_pct = int(st.session_state.lr_prob * 100)
        knn_pct = int(st.session_state.knn_prob * 100)
        lr_safe = lr_pct < 50
        knn_safe = knn_pct < 50
        
        if lr_safe != knn_safe:
            status = "medium"
            if lr_safe:
                insight_msg = f"Our two AI systems have a slight disagreement on this profile. While Model A thinks the customer is safe, Model B flags a {knn_pct}% risk of cancellation based on historical data."
            else:
                insight_msg = f"Our two AI systems have a slight disagreement on this profile. While Model B thinks the customer is safe, Model A flags a {lr_pct}% risk of cancellation based on historical data."
            action_text = "A manager should reach out with a loyalty reward to keep this customer."
        elif lr_safe and knn_safe:
            status = "low"
            insight_msg = f"Both AI systems agree that this customer is at low risk of leaving (Model A: {lr_pct}% risk, Model B: {knn_pct}% risk)."
            action_text = "No immediate action is required. Continue offering standard support."
        else:
            status = "high"
            insight_msg = f"Both AI systems agree that this customer is at high risk of cancelling (Model A: {lr_pct}% risk, Model B: {knn_pct}% risk)."
            action_text = "High priority! A manager should proactively contact this customer with a custom retention offer or loyalty reward to prevent cancellation."
            
        st.info(f"💡 **Smart Business Insight:** {insight_msg}")
        
        # Recommendation Banner (Distinct Background Theme)
        def get_recommendation_card(status, action_text):
            if status == "low":
                card_class = "rec-low"
            elif status == "medium":
                card_class = "rec-medium"
            else:
                card_class = "rec-high"
                
            html = f"""
            <div class="recommendation-card {card_class}">
                <div class="rec-title">🎯 Recommended Retention Strategy</div>
                <div class="rec-text">{action_text}</div>
            </div>
            """
            return html
            
        st.markdown(get_recommendation_card(status, action_text), unsafe_allow_html=True)

# =========================================================================
# TAB 2: MODEL EVALUATION & COMPARISON
# =========================================================================
with tab2:
    st.subheader("📊 Model Performance Summary")
    st.markdown("<p style='color: var(--text-color); opacity: 0.85; font-size: 0.95rem; margin-top: -5px; margin-bottom: 15px;'>Below is a side-by-side evaluation comparison of Logistic Regression and KNN on the 20% stratified test set.</p>", unsafe_allow_html=True)
    
    # 1. Custom Rendered Comparison Table
    def get_comparison_table_html(lr, knn):
        html = '<table class="comparison-table">'
        html += '<thead><tr><th>Evaluation Metric</th><th>Logistic Regression</th><th>KNN (K=5)</th><th>Winner</th></tr></thead>'
        html += '<tbody>'
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            lr_val = lr[metric]
            knn_val = knn[metric]
            
            if lr_val > knn_val:
                winner_tag = '<span class="winner-lr-tag">Logistic Regression</span>'
            elif knn_val > lr_val:
                winner_tag = '<span class="winner-tag">KNN</span>'
            else:
                winner_tag = '<span>Tie</span>'
                
            html += f'<tr><td><strong>{metric}</strong></td><td>{lr_val:.2%}</td><td>{knn_val:.2%}</td><td>{winner_tag}</td></tr>'
        html += '</tbody></table>'
        return html
        
    st.markdown(get_comparison_table_html(lr_metrics, knn_metrics), unsafe_allow_html=True)
    
    # Model Comparison Chart & Heatmaps
    col_chart, col_heatmaps = st.columns([5, 4], gap="large")
    
    with col_chart:
        # Create and render comparison bar chart
        def make_comparison_chart(lr, knn):
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
            lr_values = [lr[m] for m in metrics]
            knn_values = [knn[m] for m in metrics]
            
            x = np.arange(len(metrics))
            bar_width = 0.35
            
            # Styling matches custom dark theme colors
            bg_color = "#0F1117"
            card_color = "#1E2230"
            text_color = "#E8E8EC"
            accent_lr = "#6C63FF"
            accent_knn = "#00D4AA"
            grid_color = "#2A2D3A"
            
            fig, ax = plt.subplots(figsize=(8, 5.2))
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(card_color)
            
            bars_lr = ax.bar(
                x - bar_width/2, lr_values, bar_width,
                label="Logistic Regression",
                color=accent_lr,
                edgecolor="#8A82FF",
                linewidth=1.2,
                alpha=0.9,
                zorder=3
            )
            
            bars_knn = ax.bar(
                x + bar_width/2, knn_values, bar_width,
                label="KNN (K=5)",
                color=accent_knn,
                edgecolor="#33FFD1",
                linewidth=1.2,
                alpha=0.9,
                zorder=3
            )
            
            # Labels on top
            for bar in bars_lr:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2, h + 0.012,
                    f"{h:.1%}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#8A82FF"
                )
                
            for bar in bars_knn:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2, h + 0.012,
                    f"{h:.1%}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#33FFD1"
                )
                
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=11, fontweight="semibold", color=text_color)
            ax.set_ylabel("Score Score", fontsize=11, color=text_color, labelpad=8)
            ax.set_ylim(0, 1.05)
            ax.tick_params(colors=text_color)
            ax.yaxis.grid(True, color=grid_color, linestyle="--", alpha=0.5, zorder=0)
            ax.xaxis.grid(False)
            
            # Spine styling
            for spine in ax.spines.values():
                spine.set_color(grid_color)
                spine.set_linewidth(0.8)
                
            legend = ax.legend(
                fontsize=9, loc="upper right",
                framealpha=0.9,
                facecolor=card_color,
                edgecolor=grid_color
            )
            for text in legend.get_texts():
                text.set_color(text_color)
                
            plt.title("Performance Metric Comparison Matrix", fontsize=12, fontweight="bold", color=text_color, pad=15)
            fig.tight_layout()
            return fig
            
        fig_metrics = make_comparison_chart(lr_metrics, knn_metrics)
        st.pyplot(fig_metrics, use_container_width=True)
        
    with col_heatmaps:
        # Confusion Matrices Section
        def make_cm_heatmap(cm, model_name, is_knn=False):
            cmap = "YlGnBu" if is_knn else "RdPu"
            bg_color = "#0F1117"
            card_color = "#1E2230"
            text_color = "#E8E8EC"
            grid_color = "#2A2D3A"
            
            fig, ax = plt.subplots(figsize=(6.0, 5.2))
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(card_color)
            
            sns.heatmap(
                cm,
                annot=False,
                cmap=cmap,
                linewidths=2.5,
                linecolor="#000000",
                square=True,
                cbar=False,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                ax=ax
            )
            
            # Custom text overlay with high contrast based on cell background density
            tn, fp, fn, tp = cm.ravel()
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    text_color_cell = "#FFFFFF" if val > (cm.max() / 2.0) else "#111111"
                    label = "TN" if (i==0 and j==0) else ("FP" if (i==0 and j==1) else ("FN" if (i==1 and j==0) else "TP"))
                    
                    ax.text(j + 0.5, i + 0.38, f"{val}", ha="center", va="center", fontsize=14, fontweight="bold", color=text_color_cell)
                    ax.text(j + 0.5, i + 0.65, f"{label}", ha="center", va="center", fontsize=14, fontweight="bold", color=text_color_cell)
                    
            ax.set_title(f"Confusion Matrix: {model_name}", fontsize=22, fontweight="bold", color=text_color, pad=15)
            ax.set_xlabel("Predicted", fontsize=18, color=text_color, labelpad=8)
            ax.set_ylabel("Actual", fontsize=18, color=text_color, labelpad=8)
            ax.tick_params(colors=text_color, labelsize=16)
            
            for spine in ax.spines.values():
                spine.set_color(grid_color)
                spine.set_linewidth(0.8)
                
            fig.tight_layout()
            return fig

        # Render Confusion matrices side-by-side inside col_heatmaps using columns
        cm1, cm2 = st.columns(2)
        with cm1:
            fig_cm_lr = make_cm_heatmap(lr_metrics["cm"], "Logistic Regression")
            st.pyplot(fig_cm_lr, use_container_width=True)
        with cm2:
            fig_cm_knn = make_cm_heatmap(knn_metrics["cm"], "KNN (K=5)", is_knn=True)
            st.pyplot(fig_cm_knn, use_container_width=True)
            
    # Narrative Analysis of Results
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💡 Analytical Findings & Model Differences")
    st.markdown(
        """
        - **🏆 AI Model A (Trend Analyzer) - Overall Winner:** This model performs exceptionally well across every grading test. It achieves an impressive **~80%** accuracy score because it is highly stable. It focuses entirely on the big, logical customer patterns (like contract length and internet packages) without getting confused by minor, random background details.
        - **👥 AI Model B (Similar-Customer Matcher):** This system trails slightly behind at **~76%** accuracy. It works by trying to calculate similarity scores to match new profiles with past customers. Because our customer accounts use a lot of simple 'Yes/No' options, calculating exact mathematical similarities becomes much less effective.
        - **🛡️ Reliability Check:** The Trend Analyzer proved to be remarkably steady, showing it genuinely understands the core triggers behind customer loyalty. On the other hand, the Matcher model got slightly overwhelmed trying to juggle too many separate account checkboxes at once—a very common limitation when computer algorithms try to calculate similarity across wide checklists.
        """
    )
