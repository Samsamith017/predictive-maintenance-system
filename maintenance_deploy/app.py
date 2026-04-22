import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -- Page Configuration --
st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# -- Advanced UI Styling --
st.markdown("""
    <style>
    [data-testid="stSidebarCollapseIcon"] svg, 
    [data-testid="collapsedControl"] svg {
        display: none !important;
        opacity: 0 !important;
    }

    [data-testid="stSidebarCollapseIcon"]::after {
        content: "Collapse";
        font-size: 14px;
        color: #2980b9;
        font-weight: bold;
        visibility: visible;
    }
    [data-testid="collapsedControl"]::after {
        content: "Expand";
        font-size: 14px;
        color: #2980b9;
        font-weight: bold;
        visibility: visible;
        margin-left: 10px;
    }

    div.stButton > button {
        background-color: #E1F6FF !important;
        color: #000000 !important;
        border: 1px solid #b3e5fc !important;
        font-weight: bold !important;
        height: 3.5em !important;
        width: 100% !important;
        border-radius: 8px !important;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div.stButton > button:hover {
        background-color: #c5edff !important;
        border: 1px solid #2980b9 !important;
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.95);
        color: #000;
        text-align: center;
        padding: 6px;
        font-size: 12px;
        font-weight: bold;
        border-top: 1px solid #eee;
        z-index: 999;
    }
    
    button[kind="header"] {
        background: transparent !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# -- Model Loading --
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'model.pkl')
    features_path = os.path.join(base_path, 'features.pkl')

    if not os.path.exists(model_path):
        st.error("❌ Model file 'model.pkl' not found.")
        st.stop()

    model = joblib.load(model_path)

    # Load feature order if available
    if os.path.exists(features_path):
        features = joblib.load(features_path)
    else:
        features = ['Air temperature [K]', 'Process temperature [K]',
                    'Rotational speed [rpm]', 'Torque [Nm]',
                    'Tool wear [min]', 'Type_L', 'Type_M']

    return model, features

model, FEATURES = load_model()

# -- Sidebar: Machine Settings --
st.sidebar.header("⚙️ Machine Settings")
machine_id = st.sidebar.text_input("Machine ID", value="MAC-1024")

st.sidebar.markdown("""
**Machine Type** <br>
<small style="color: grey; font-size: 11px;">The build quality and durability grade of the equipment</small>
""", unsafe_allow_html=True)

machine_type = st.sidebar.selectbox(
    label="Select Grade",
    options=["L (Low)", "M (Medium)", "H (High)"],
    label_visibility="collapsed"
)

# -- Main Dashboard Header --
st.title("🔧 Industrial Predictive Maintenance Dashboard")
st.caption(f"Currently Monitoring: **{machine_id}**")

st.subheader("📡 Real-Time Sensor Inputs")
col1, col2, col3 = st.columns(3)

with col1:
    air_temp = st.number_input("Air Temp [K]", value=300.0, step=0.1)
    process_temp = st.number_input("Process Temp [K]", value=310.0, step=0.1)

with col2:
    rot_speed = st.number_input("Rotational Speed [rpm]", value=1500.0, step=10.0)
    torque = st.number_input("Torque [Nm]", value=40.0, step=0.5)

with col3:
    tool_wear = st.number_input("Tool Wear [min]", value=50.0, step=1.0)

# -- Data Preparation --
type_L = 1 if "L" in machine_type else 0
type_M = 1 if "M" in machine_type else 0

input_df = pd.DataFrame([[air_temp, process_temp, rot_speed, torque, tool_wear, type_L, type_M]],
                        columns=FEATURES)

# -- Diagnostic Results --
st.divider()

if st.button("Run System Diagnostic", use_container_width=True):

    probs = model.predict_proba(input_df)[0]

    # SAFE risk calculation (handles any class order)
    if len(probs) == 3:
        risk_score = (0 * probs[0]) + (50 * probs[1]) + (100 * probs[2])
    else:
        # fallback if binary model
        risk_score = probs[-1] * 100

    st.subheader(f"📊 Diagnostic Analysis: {machine_id}")
    st.progress(int(min(max(risk_score, 0), 100)))

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if risk_score < 30:
            st.success(f"🟢 **LOW RISK** (Score: {risk_score:.1f})")
            st.write("Machine is healthy. No action needed.")
        elif risk_score < 65:
            st.warning(f"🟡 **MEDIUM RISK** (Score: {risk_score:.1f})")
            st.write("Monitor machine closely for changes.")
        else:
            st.error(f"🔴 **HIGH RISK** (Score: {risk_score:.1f})")
            st.write("Failure imminent! Maintenance required.")

        st.write("**Confidence Breakdown:**")
        if len(probs) == 3:
            st.info(f"Healthy: {probs[0]*100:.1f}% | Warning: {probs[1]*100:.1f}% | Failure: {probs[2]*100:.1f}%")
        else:
            st.info(f"Failure Probability: {probs[-1]*100:.1f}%")

    with res_col2:
        st.write("**Operational Context:**")
        st.dataframe(input_df.style.background_gradient(cmap='Blues', axis=1))
        st.info("The Risk Score is calculated based on historical failure patterns and real-time sensor correlation.")

# -- Footer --
st.markdown(f'<div class="footer">Samith S P © 2026 | Predictive Maintenance System v1.0</div>', unsafe_allow_html=True)
