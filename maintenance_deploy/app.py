import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

# 2. Advanced CSS to Hide Arrows and Add Custom Labels
st.markdown("""
    <style>
    /* HIDE THE ACTUAL ARROW ICONS COMPLETELY */
    [data-testid="stSidebarCollapseIcon"] svg, 
    [data-testid="collapsedControl"] svg {
        display: none !important;
        opacity: 0 !important;
    }

    /* ADD 'COLLAPSE' TEXT TO THE OPEN SIDEBAR BUTTON */
    [data-testid="stSidebarCollapseIcon"]::after {
        content: "Collapse";
        font-size: 14px;
        color: #666;
        font-weight: bold;
        visibility: visible;
    }

    /* ADD 'EXPAND' TEXT TO THE CLOSED SIDEBAR BUTTON */
    [data-testid="collapsedControl"]::after {
        content: "Expand";
        font-size: 14px;
        color: #666;
        font-weight: bold;
        visibility: visible;
        margin-left: 10px;
    }

    /* REMOVE THE OVERLAY CIRCLE AROUND THE BUTTON */
    button[kind="header"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* PROFESSIONAL FOOTER */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #888;
        text-align: center;
        padding: 5px;
        font-size: 12px;
        font-weight: bold;
        border-top: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load the Model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Model file not found. Please ensure 'model.pkl' is in the folder.")
    st.stop()

# 4. Sidebar: Machine Identity
st.sidebar.header("Machine Settings")
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

# 5. Main Content
st.title("🔧 Industrial Predictive Maintenance Dashboard")
st.caption("Real-time Predictive Maintenance powered by Calibrated AI.")

st.subheader("📡 Sensor Inputs")
col1, col2, col3 = st.columns(3)

with col1:
    air_temp = st.number_input("Air Temp [K]", value=300.0, step=0.1)
    process_temp = st.number_input("Process Temp [K]", value=310.0, step=0.1)

with col2:
    rot_speed = st.number_input("Rotational Speed [rpm]", value=1500.0, step=10.0)
    torque = st.number_input("Torque [Nm]", value=40.0, step=0.5)

with col3:
    tool_wear = st.number_input("Tool Wear [min]", value=50.0, step=1.0)

# 6. Prediction Logic
type_L = 1 if "L" in machine_type else 0
type_M = 1 if "M" in machine_type else 0

columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
           'Torque [Nm]', 'Tool wear [min]', 'Type_L', 'Type_M']

input_df = pd.DataFrame([[air_temp, process_temp, rot_speed, torque, tool_wear, type_L, type_M]], columns=columns)

st.divider()
if st.button("Check Machine Health", type="primary", use_container_width=True):
    probs = model.predict_proba(input_df)[0]
    risk_score = (0 * probs[0]) + (50 * probs[1]) + (100 * probs[2])
    
    st.subheader(f"📊 Diagnostic Analysis: {machine_id}")
    st.progress(int(risk_score))
    
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

    with res_col2:
        st.write("**Confidence Breakdown:**")
        st.info(f"Healthy: {probs[0]*100:.1f}% | Warning: {probs[1]*100:.1f}% | Failure: {probs[2]*100:.1f}%")

# 7. Professional Footer
st.markdown("""
    <div class="footer">
        Samith S P © 2026 | Predictive Maintenance System v1.0
    </div>
""", unsafe_allow_html=True)