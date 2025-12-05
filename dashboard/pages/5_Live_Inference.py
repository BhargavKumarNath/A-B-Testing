import streamlit as st
import pandas as pd
import numpy as np
import pickle
import utils

st.set_page_config(page_title="Live Inference", page_icon="⚡", layout="wide")
utils.load_css()

st.title("⚡ Live Prediction API")

st.markdown("""
<div class="insight-box">
    <strong>Production Simulator:</strong> This module loads the actual distilled <code>.pkl</code> model saved by the pipeline. 
    Adjust the sliders to simulate a user profile and see the model's bidding decision in real-time (< 1ms).
</div>
""", unsafe_allow_html=True)

model = None
try:
    with open(utils.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.success(f"Loaded Production Model: {utils.MODEL_PATH.name}")
except FileNotFoundError:
    st.error("Model file not found. Please run main.py first to generate 'production_uplift_model.pkl'.")
    st.stop()

# Inputs
st.markdown('<div class="section-header">User Profile (Context)</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    f4 = st.slider("f4 (Persuadability Signal)", -5.0, 20.0, 12.0, help="High values indicate persuadability")
    f3 = st.slider("f3 (Sure Thing Signal)", -5.0, 10.0, 2.0, help="Low values indicate persuadability")
    f0 = st.slider("f0", -5.0, 20.0, 10.0)
    f1 = st.slider("f1", -5.0, 20.0, 10.0)

with col2:
    f2 = st.slider("f2", -5.0, 20.0, 8.0)
    f5 = st.slider("f5", -5.0, 20.0, 4.0)
    f6 = st.slider("f6", -20.0, 5.0, -8.0)
    f7 = st.slider("f7", -5.0, 20.0, 5.0)

with col3:
    f8 = st.slider("f8", 0.0, 10.0, 3.0)
    f9 = st.slider("f9", 0.0, 50.0, 25.0)
    f10 = st.slider("f10", 0.0, 10.0, 5.0)
    f11 = st.slider("f11", -2.0, 2.0, -0.1)

# Const Feature Vec.
features = np.array([[f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]])

# Prediction
st.markdown('<div class="section-header">Model Decision</div>', unsafe_allow_html=True)

if st.button("🔮 Predict Uplift"):
    # 1. Inference
    uplift_score = model.predict(features)[0]

    # 2. Economics 
    conversion_value = 10.0
    ad_cost = 0.10
    expected_profit = (uplift_score * conversion_value) - ad_cost
    decision = "BID" if expected_profit > 0 else "NO BID"

    # 3. Display
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Predicted Uplift (CATE)", f"{uplift_score:.4f}", delta="High Persuadability" if uplift_score > 0.05 else "Low Persuadability")
    
    with c2:
        color = "green" if decision == "BID" else "red"
        st.markdown(f"""
        <div style="text-align: center; border: 2px solid {color}; padding: 10px; border-radius: 10px;">
            <h2 style="color: {color} !important; margin:0;">{decision}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.metric("Expected ROI", f"${expected_profit:.4f}", delta="Positive" if expected_profit > 0 else "Negative")

    # Explanation
    if decision == "BID":
        st.success(f"Logic: Uplift ({uplift_score:.3f}) × Value ($10) > Cost ($0.10). **Profitable impression.**")
    else:
        st.warning(f"Logic: Uplift ({uplift_score:.3f}) × Value ($10) < Cost ($0.10). **Save budget.**")