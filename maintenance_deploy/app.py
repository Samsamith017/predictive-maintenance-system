import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="🛠️",
    layout="wide"
)

# --- 2. MODEL LOADING LOGIC ---
# This section ensures the app finds 'model.pkl' regardless of the environment
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'model.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file 'model.pkl' not found in {base_path}. Please upload it to GitHub.")
        st.stop()
    
    return joblib.load(model_path)

model = load_model()

# --- 3. SIDEBAR / INPUTS ---
st.sidebar.header("🔧 Machine Sensor Inputs")
st.sidebar.markdown("Enter the real-time sensor readings below:")

def user_input_features():
    # Categorical Input
    type_input = st.sidebar.selectbox("Machine Type", ("Low (L)", "Medium (M)", "High (H)"))
    
    # Numerical Inputs
    air_temp = st.sidebar.number_input("Air Temperature [K]", min_value=200.0, max_value=400.0, value=300.0)
    proc_temp = st.sidebar.number_input("Process Temperature [K]", min_value=200.0, max_value=400.0, value=310.0)
    speed = st.sidebar.number_input("Rotational Speed [rpm]", min_value=0, max_value=3000, value=1500)
    torque = st.sidebar.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0)
    tool_wear = st.sidebar.number_input("Tool Wear [min]", min_value=0, max_value=300, value=0)

    # Convert Machine Type to One-Hot Encoding (matching the model training)
    type_l = 1 if type_input == "Low (L)" else 0
    type_m = 1 if type_input == "Medium (M)" else 0
    # Note: Type_H is dropped to avoid the dummy variable trap (Standard practice)

    data = {
        'Air temperature [K]': air_temp,
        'Process temperature [K]': proc_temp,
        'Rotational speed [rpm]': speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Type_L': type_l,
        'Type_M': type_m
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 4. MAIN DASHBOARD ---
st.title("🛠️ Intelligent Predictive Maintenance Dashboard")
st.markdown("""
This system uses a **Calibrated Gradient Boosting** model to monitor industrial equipment health. 
It analyzes sensor patterns to predict potential failures before they occur.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Sensor Snapshot")
    st.dataframe(input_df.style.highlight_max(axis=0, color='#2980b9'))

with col2:
    st.subheader("Diagnostic Prediction")
    
    # Get Prediction and Probabilities
    prediction = model.predict(input_df)[0]
    # Use index [1] for the probability of "Failure"
    risk_probability = model.predict_proba(input_df)[0][1] * 100 

    # Display Result
    if prediction == 0:
        st.success("✅ Status: Machine Healthy")
    else:
        st.error("⚠️ Status: Maintenance Required")

    # Risk Meter Logic
    st.metric(label="Failure Risk Score", value=f"{risk_probability:.1f}%")
    
    if risk_probability < 30:
        st.info("Risk Level: **Low** (Normal Operation)")
    elif 30 <= risk_probability < 70:
        st.warning("Risk Level: **Medium** (Schedule Inspection)")
    else:
        st.error("Risk Level: **High** (Immediate Action Required)")

# --- 5. TECHNICAL INSIGHTS ---
st.divider()
with st.expander("ℹ️ About this System"):
    st.markdown("""
    **Developer:** Samith S P  
    **Model:** Gradient Boosting Classifier with SMOTE & Calibration.  
    **Hardware Compatibility:** CNC Machines, Wind Turbines, Electric Motors, and Industrial Pumps.
    
    *During the development phase, AI tools were utilized for code optimization, UI styling (CSS), and troubleshooting model calibration errors.*
    """)
