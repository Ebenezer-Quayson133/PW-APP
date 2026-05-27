import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="ROP Prediction and Optimisation App",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Custom CSS for better look --------------------
st.markdown(
    """
    <style>
    /* Global styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stNumberInput, .stSlider {
        background-color: rgba(255,255,255,0.8);
        border-radius: 10px;
        padding: 5px 10px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return joblib.load('rop_rf_model.pkl')

model = load_model()

# Feature names as used during training (11 features)
feature_names = [
    'Depth(ft)',
    'weight on bit (k-lbs)',
    'Pump Press (psi)',
    'Hookload (k-lbs)',
    'Surface Torque (psi)',
    'Rotary Speed (rpm)',
    'Flow In (gal/min)',
    'WOB_RPM',
    'Hydraulic_Power',
    'Torque_RPM',
    'Depth_WOB'
]

# -------------------- Optimisation Function (unchanged) --------------------
def find_optimal_wob_rpm_fast(model, fixed_params, wob_range=(10,47), rpm_range=(30,120), steps=30):
    wob_vals = np.linspace(wob_range[0], wob_range[1], steps)
    rpm_vals = np.linspace(rpm_range[0], rpm_range[1], steps)
    W, R = np.meshgrid(wob_vals, rpm_vals)
    wob_flat = W.ravel()
    rpm_flat = R.ravel()
    n_points = len(wob_flat)

    wob_rpm = wob_flat * rpm_flat
    torque_rpm = fixed_params['Surface Torque (psi)'] * rpm_flat
    depth_wob = fixed_params['Depth(ft)'] * wob_flat
    hydraulic_power = fixed_params['Pump Press (psi)'] * fixed_params['Flow In (gal/min)']

    df_all = pd.DataFrame({
        'Depth(ft)': fixed_params['Depth(ft)'],
        'weight on bit (k-lbs)': wob_flat,
        'Pump Press (psi)': fixed_params['Pump Press (psi)'],
        'Hookload (k-lbs)': fixed_params['Hookload (k-lbs)'],
        'Surface Torque (psi)': fixed_params['Surface Torque (psi)'],
        'Rotary Speed (rpm)': rpm_flat,
        'Flow In (gal/min)': fixed_params['Flow In (gal/min)'],
        'WOB_RPM': wob_rpm,
        'Hydraulic_Power': hydraulic_power,
        'Torque_RPM': torque_rpm,
        'Depth_WOB': depth_wob
    }, columns=feature_names)

    pred_log = model.predict(df_all)
    pred_rop = np.expm1(pred_log)

    idx_max = np.argmax(pred_rop)
    best_wob = wob_flat[idx_max]
    best_rpm = rpm_flat[idx_max]
    max_rop = pred_rop[idx_max]

    results_df = pd.DataFrame({
        'WOB': wob_flat,
        'RPM': rpm_flat,
        'Predicted_ROP': pred_rop
    })

    return best_wob, best_rpm, max_rop, results_df

# -------------------- Sidebar Inputs --------------------
st.sidebar.title("⛰️ Drilling Parameters")
st.sidebar.markdown("Adjust the values below. ROP updates in real time.")

# Use session state to store values for random generator
if 'depth' not in st.session_state:
    st.session_state.depth = 5000.0
if 'wob' not in st.session_state:
    st.session_state.wob = 25.0
if 'pump_press' not in st.session_state:
    st.session_state.pump_press = 2000.0
if 'hookload' not in st.session_state:
    st.session_state.hookload = 150.0
if 'torque' not in st.session_state:
    st.session_state.torque = 5000.0
if 'rpm' not in st.session_state:
    st.session_state.rpm = 60.0
if 'flow' not in st.session_state:
    st.session_state.flow = 400.0

# Function to set random example
def random_example():
    st.session_state.depth = np.random.uniform(3000, 10000)
    st.session_state.wob = np.random.uniform(10, 47)
    st.session_state.pump_press = np.random.uniform(1500, 3000)
    st.session_state.hookload = np.random.uniform(100, 200)
    st.session_state.torque = np.random.uniform(3000, 8000)
    st.session_state.rpm = np.random.uniform(30, 120)
    st.session_state.flow = np.random.uniform(300, 600)

if st.sidebar.button("🎲 Random Example", help="Fill with random realistic values"):
    random_example()

st.sidebar.number_input("Depth (ft)", min_value=0.0, key='depth', step=100.0, format="%.1f")
st.sidebar.slider("Weight on Bit (k-lbs)", 10.0, 47.0, key='wob', step=0.5, format="%.1f")
st.sidebar.number_input("Pump Pressure (psi)", min_value=0.0, key='pump_press', step=50.0, format="%.1f")
st.sidebar.number_input("Hookload (k-lbs)", min_value=0.0, key='hookload', step=5.0, format="%.1f")
st.sidebar.number_input("Surface Torque (psi)", min_value=0.0, key='torque', step=100.0, format="%.1f")
st.sidebar.slider("Rotary Speed (RPM)", 30.0, 120.0, key='rpm', step=1.0, format="%.1f")
st.sidebar.number_input("Flow In (gal/min)", min_value=0.0, key='flow', step=10.0, format="%.1f")

# -------------------- Real-time Prediction --------------------
# Compute engineered features from current session state
wob_rpm = st.session_state.wob * st.session_state.rpm
torque_rpm = st.session_state.torque * st.session_state.rpm
depth_wob = st.session_state.depth * st.session_state.wob
hydraulic_power = st.session_state.pump_press * st.session_state.flow

input_df = pd.DataFrame([[
    st.session_state.depth,
    st.session_state.wob,
    st.session_state.pump_press,
    st.session_state.hookload,
    st.session_state.torque,
    st.session_state.rpm,
    st.session_state.flow,
    wob_rpm,
    hydraulic_power,
    torque_rpm,
    depth_wob
]], columns=feature_names)

pred_log = model.predict(input_df)[0]
pred_rop = np.expm1(pred_log)

# -------------------- Main Panel --------------------
st.title("⛰️ ROP Prediction and Optimisation App")
st.markdown("##### *Intelligent drilling advisory tool*")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Current ROP", f"{pred_rop:.2f} ft/hr")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("WOB", f"{st.session_state.wob:.1f} k-lbs")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("RPM", f"{st.session_state.rpm:.1f}")
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Depth", f"{st.session_state.depth:.0f} ft")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- Optimisation Section --------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("⚙️ Optimise WOB and RPM")
    st.markdown("Find the best combination of Weight on Bit and Rotary Speed to maximise ROP, given your current fixed parameters (Depth, Pump Pressure, Hookload, Torque, Flow).")

    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        if st.button("🚀 Run Optimisation", use_container_width=True):
            fixed_vals = {
                'Depth(ft)': st.session_state.depth,
                'Pump Press (psi)': st.session_state.pump_press,
                'Hookload (k-lbs)': st.session_state.hookload,
                'Surface Torque (psi)': st.session_state.torque,
                'Flow In (gal/min)': st.session_state.flow
            }

            with st.spinner("Optimising... (grid search in progress)"):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Simulate work (remove in production if not needed)
                    progress_bar.progress(i + 1)
                best_wob, best_rpm, max_rop, results_df = find_optimal_wob_rpm_fast(
                    model, fixed_vals, steps=30
                )
                progress_bar.empty()

            st.session_state.opt_results = {
                'best_wob': best_wob,
                'best_rpm': best_rpm,
                'max_rop': max_rop,
                'results_df': results_df
            }
            st.success("Optimisation complete!")

    with col_opt2:
        if 'opt_results' in st.session_state:
            res = st.session_state.opt_results
            st.markdown(f"**Optimal WOB:** `{res['best_wob']:.2f} k-lbs`  \n**Optimal RPM:** `{res['best_rpm']:.2f}`  \n**Max ROP:** `{res['max_rop']:.2f} ft/hr`")
            # Download button for results
            csv = res['results_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Optimisation Results",
                data=csv,
                file_name='rop_optimisation_results.csv',
                mime='text/csv',
            )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Visualisation --------------------
if 'opt_results' in st.session_state:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Optimisation Surface")
    tab1, tab2 = st.tabs(["Contour Plot", "3D Surface"])

    results_df = st.session_state.opt_results['results_df']
    best_wob = st.session_state.opt_results['best_wob']
    best_rpm = st.session_state.opt_results['best_rpm']
    max_rop = st.session_state.opt_results['max_rop']

    with tab1:
        # Interactive Plotly contour
        pivot = results_df.pivot(index='RPM', columns='WOB', values='Predicted_ROP')
        fig = go.Figure(data=
            go.Contour(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='Viridis',
                contours=dict(showlabels=True),
                colorbar=dict(title="ROP (ft/hr)")
            )
        )
        fig.add_trace(go.Scatter(
            x=[best_wob],
            y=[best_rpm],
            mode='markers',
            marker=dict(symbol='star', size=15, color='red'),
            name=f'Optimum: WOB={best_wob:.1f}, RPM={best_rpm:.1f}'
        ))
        fig.update_layout(
            xaxis_title="WOB (k-lbs)",
            yaxis_title="RPM",
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # 3D surface
        fig3d = go.Figure(data=[
            go.Surface(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='Viridis',
                colorbar=dict(title="ROP (ft/hr)")
            )
        ])
        fig3d.add_trace(go.Scatter3d(
            x=[best_wob],
            y=[best_rpm],
            z=[max_rop],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='Optimum'
        ))
        fig3d.update_layout(
            scene=dict(
                xaxis_title='WOB (k-lbs)',
                yaxis_title='RPM',
                zaxis_title='ROP (ft/hr)'
            ),
            height=500
        )
        st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Info Section --------------------
with st.expander("ℹ️ About this tool"):
    st.markdown("""
    This app uses a **Random Forest model (R^2 = 0.844) ** trained on drilling data to predict Rate of Penetration (ROP).
    - **Real-time prediction** updates as you change parameters.
    - **Optimisation** performs a grid search over WOB and RPM to find the combination that yields the highest ROP, keeping other parameters fixed.
    - The model was trained with engineered features: `WOB_RPM`, `Hydraulic_Power`, `Torque_RPM`, `Depth_WOB`.
    
    """)

# -------------------- Footer with Developer Credits --------------------
st.markdown("---")
st.caption(
    "Developed by: **Ebenezer Quayson, Ernest Eric Adams, Evans Datani, Charles Sarfo Arhin** | "
    "Supervised by: **Assoc. Prof. Richard Amorin** | "

)
