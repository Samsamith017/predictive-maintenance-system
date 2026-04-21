import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION & CUSTOM STYLING ---
st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="🛠️",
    layout="wide"
)

# CSS for Sidebar
st.markdown("""
    <style>
    /* HIDE THE ACTUAL ARROW ICONS COMPLETELY to prevent ghosting */
    [data-testid="stSidebarCollapseIcon"] svg, 
    [data-testid="collapsedControl"] svg {
        display: none !important;
        opacity: 0 !important;
    }

    /* OPEN SIDEBAR BUTTON */
    [data-testid="stSidebarCollapseIcon"]::after {
        content: "Collapse";
        font-size: 14px;
        color: #2980b9;
        font-weight: bold;
        visibility: visible;
    }

    /* CLOSE SIDEBAR BUTTON */
    [data-testid="collapsedControl"]::after {
        content: "Expand";
        font-size: 14px;
        color: #2980b9;
        font-weight: bold;
        visibility: visible;
        margin-left: 10px;
    }

    /* FOOTER */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.9);
        color: #888;
        text-align: center;
        padding: 5px;
        font-size: 12px;
        font-weight: bold;
        border-top: 1px solid #eee;
        z-index: 999;
    }
    
    /* button hover circles */
    button[kind="header"] {
        background: transparent !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- SAFE MODEL LOADING ---
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'model.pkl')
    
    if not os.path.exists(model_path):
        st.error("❌ Model file 'model.pkl' not found. Please ensure it is in your GitHub repository.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# --- SIDEBAR: MACHINE SETTINGS ---
st.sidebar.header("⚙️ Machine Settings")
machine_id = st.sidebar.text_input("Machine ID", value="MAC-1024")

st.sidebar.markdown("""
**Machine Type** <br>
<small style="color: grey; font-size: 11px;">The build quality and durability grade of the equipment</small>
""", unsafe_allow_html=True)

machine_type_input = st.sidebar.selectbox(
    label="Select Grade",
    options=["Low (L)", "Medium (M)", "High (H)"],
    label_visibility="collapsed"
)

# --- MAIN CONTENT & SENSOR INPUTS ---
st.title("🔧 Industrial Predictive Maintenance Dashboard")
st.caption(f"Currently Monitoring: **{machine_id}**")

st.markdown("""
This system uses a **Calibrated Gradient Boosting** model to monitor industrial equipment health. 
Adjust the sensors below to run a diagnostic.
""")

st.subheader("📡 Real-Time Sensor Inputs")
col1, col2, col3 = st.columns(3)

with col1:
    air_temp = st.number_input("Air Temp [K]", min_value=200.0, max_value=400.0, value=300.0, step=0.1)
    proc_temp = st.number_input("Process Temp [K]", min_value=200.0, max_value=400.0, value=310.0, step=0.1)

with col2:
    rot_speed = st.number_input("Rotational Speed [rpm]", min_value=0.0, max_value=5000.0, value=1500.0, step=10.0)
    torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.5)

with col3:
    tool_wear = st.number_input("Tool Wear [min]", min_value=0.0, max_value=300.0, value=50.0, step=1.0)

# --- DATA PREPARATION ---
# Convert inputs to match model features
type_L = 1 if "Low" in machine_type_input else 0
type_M = 1 if "Medium" in machine_type_input else 0

columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
           'Torque [Nm]', 'Tool wear [min]', 'Type_L', 'Type_M']

input_df = pd.DataFrame([[air_temp, proc_temp, rot_speed, torque, tool_wear, type_L, type_M]], columns=columns)

# --- DIAGNOSTIC RESULTS ---
st.divider()

if st.button("Run System Diagnostic", type="primary", use_container_width=True):
    # Getting the Probability of Failure (class index 1)
    risk_probability = model.predict_proba(input_df)[0][1] * 100
    
    st.subheader(f"📊 Analysis Results for {machine_id}")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.metric(label="Failure Risk Score", value=f"{risk_probability:.1f}%")
        st.progress(int(risk_probability))
        
        if risk_probability < 30:
            st.success("🟢 **LOW RISK**: Machine is healthy. Continue normal operation.")
        elif 30 <= risk_probability < 70:
            st.warning("🟡 **MEDIUM RISK**: Potential anomaly detected. Schedule inspection.")
        else:
            st.error("🔴 **HIGH RISK**: Failure imminent! Immediate maintenance required.")

    with res_col2:
        st.write("**Operational Context:**")
        # Displaying the input data
        st.dataframe(input_df.style.background_gradient(cmap='Blues', axis=1))
        st.info("The Risk Score is calculated based on historical failure patterns and real-time sensor correlation.")

# --- FOOTER & ABOUT ---
st.divider()
with st.expander("ℹ️ Technical System Details"):
    st.markdown("""
    **Developer:** Samith S P  
    **Framework:** Streamlit + Scikit-Learn  
    **Model:** Calibrated Gradient Boosting with SMOTE  
    
    *During the development phase, AI tools were utilized for code optimization, UI styling (CSS), and troubleshooting model calibration errors.*
    """)

st.markdown("""
    <div class="footer">
        Samith S P © 2026 | Predictive Maintenance System v1.0
    </div>
""", unsafe_allow_html=True)
