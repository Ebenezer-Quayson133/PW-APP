import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------
st.set_page_config(
    page_title="ROP Prediction and Optimization Tool",
    layout="wide"
)

# ---------------------------------
# LOAD TRAINED MODEL
# ---------------------------------
rf = joblib.load("rop_rf_model.pkl")

# ---------------------------------
# TITLE AND DESCRIPTION
# ---------------------------------
st.title("🛢️ AI-Based Drilling Optimization Tool")
st.markdown(
    """
    This application predicts **Rate of Penetration (ROP)** using a  
    **Random Forest machine learning model** trained on real drilling data.
    It supports **performance evaluation**, **sensitivity analysis**, and  
    **engineering decision-making**.
    """
)

# ---------------------------------
# SIDEBAR INPUTS
# ---------------------------------
st.sidebar.header("Drilling Parameters")

depth = st.sidebar.slider("Depth (ft)", 0, 15000, 5000)
wob = st.sidebar.slider("Weight on Bit (k-lbs)", 0.0, 60.0, 20.0)
rpm = st.sidebar.slider("Rotary Speed (rpm)", 0, 300, 120)
pump_press = st.sidebar.slider("Pump Pressure (psi)", 0, 5000, 2500)
flow_rate = st.sidebar.slider("Flow Rate (gpm)", 0, 1200, 600)
torque = st.sidebar.slider("Surface Torque (psi)", 0, 4000, 1500)
hookload = st.sidebar.slider("Hookload (k-lbs)", 0, 800, 300)
whp = st.sidebar.slider("Wellhead Pressure (psi)", 0, 3000, 800)

# ---------------------------------
# FEATURE ENGINEERING (Physics-informed)
# ---------------------------------
WOB_RPM = wob * rpm
Hydraulic_Power = pump_press * flow_rate
Torque_RPM = torque * rpm
Depth_WOB = depth * wob

input_data = pd.DataFrame([[
    depth, wob, pump_press, hookload, torque,
    rpm, flow_rate, whp,
    WOB_RPM, Hydraulic_Power, Torque_RPM, Depth_WOB
]], columns=[
    "Depth(ft)", "weight on bit (k-lbs)", "Pump Press (psi)",
    "Hookload (k-lbs)", "Surface Torque (psi)",
    "Rotary Speed (rpm)", "Flow In (gal/min)", "WH Pressure (psi)",
    "WOB_RPM", "Hydraulic_Power", "Torque_RPM", "Depth_WOB"
])

# ---------------------------------
# ROP PREDICTION
# ---------------------------------
log_rop = rf.predict(input_data)[0]
rop = np.expm1(log_rop)

st.subheader("📈 Predicted Drilling Performance")
st.metric("Predicted ROP (ft/hr)", f"{rop:.2f}")

# ---------------------------------
# FEATURE IMPORTANCE
# ---------------------------------
st.subheader("🔍 Feature Importance Analysis")

feat_imp = pd.DataFrame({
    "Feature": input_data.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig1, ax1 = plt.subplots()
ax1.barh(feat_imp["Feature"], feat_imp["Importance"])
ax1.invert_yaxis()
ax1.set_xlabel("Relative Importance")
ax1.set_ylabel("Input Parameters")
ax1.grid(True)

st.pyplot(fig1)

# ---------------------------------
# SENSITIVITY ANALYSIS
# ---------------------------------
st.subheader("📊 Sensitivity Analysis")

param = st.selectbox(
    "Select parameter to vary",
    ["weight on bit (k-lbs)", "Rotary Speed (rpm)", "Flow In (gal/min)"]
)

values = np.linspace(
    input_data[param].values[0] * 0.5,
    input_data[param].values[0] * 1.5,
    30
)

rop_sens = []

for v in values:
    temp = input_data.copy()
    temp[param] = v

    temp["WOB_RPM"] = temp["weight on bit (k-lbs)"] * temp["Rotary Speed (rpm)"]
    temp["Hydraulic_Power"] = temp["Pump Press (psi)"] * temp["Flow In (gal/min)"]
    temp["Torque_RPM"] = temp["Surface Torque (psi)"] * temp["Rotary Speed (rpm)"]
    temp["Depth_WOB"] = temp["Depth(ft)"] * temp["weight on bit (k-lbs)"]

    log_pred = rf.predict(temp)[0]
    rop_sens.append(np.expm1(log_pred))

fig2, ax2 = plt.subplots()
ax2.plot(values, rop_sens)
ax2.set_xlabel(param)
ax2.set_ylabel("ROP (ft/hr)")
ax2.grid(True)

st.pyplot(fig2)

# ---------------------------------
# DEPTH-WISE ROP PROFILE
# ---------------------------------
st.subheader("🧱 Depth-wise ROP Profile")

depth_range = np.linspace(0, 15000, 50)
rop_depth = []

for d in depth_range:
    temp = input_data.copy()
    temp["Depth(ft)"] = d
    temp["Depth_WOB"] = d * temp["weight on bit (k-lbs)"]

    log_pred = rf.predict(temp)[0]
    rop_depth.append(np.expm1(log_pred))

fig3, ax3 = plt.subplots()
ax3.plot(depth_range, rop_depth)
ax3.set_xlabel("Depth (ft)")
ax3.set_ylabel("ROP (ft/hr)")
ax3.invert_xaxis()
ax3.grid(True)

st.pyplot(fig3)

# ---------------------------------
# MODEL SUMMARY
# ---------------------------------
st.subheader("📘 Model Summary")
st.markdown(
    """
    **Developed by:** Ebenezer Quayson, Ernest Eric Adams, Evans Datani, and Charles Sarfo Arhin
    
    **Model Type:** Random Forest Regressor  
    **Approach:** Data-driven with physics-informed features  
    **Application:** Drilling optimization and performance prediction  
    
    """
)
