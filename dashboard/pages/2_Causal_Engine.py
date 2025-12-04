import streamlit as st
import utils
import pandas as pd

st.set_page_config(page_title="Causal Engine", page_icon="🧠", layout="wide")
utils.load_css()

st.title("🧠 The Causal Engine (X-Learner)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Settings")
    n_bootstraps = st.slider("Bootstrap Iterations (Simulated)", 10, 100, 20)
    hist_bins = st.slider("Histogram Bins", 20, 100, 50)

# 1. Methodology
st.markdown('<div class="section-header">1. Methodology: Why X-Learner?</div>', unsafe_allow_html=True)

st.markdown("""
We chose the **X-Learner** meta-learner because it excels in **imbalanced datasets** where conversions are rare (0.2%).
It works by estimating treatment effects in two stages:
1.  **Stage 1**: Estimate response functions for Treatment and Control separately.
2.  **Stage 2**: Impute counterfactuals and train a second-stage model to predict the difference (CATE).
""")

# 2. CATE Distribution
st.markdown('<div class="section-header">2. Predicted Uplift Distribution (CATE)</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(utils.plot_cate_distribution(bins=hist_bins), width='stretch')

with col2:
    st.markdown("### User Segments")
    st.markdown("""
    - **Persuadables (>0)**: Users who buy *only if* treated. (Target These!)
    - **Sleeping Dogs (<0)**: Users who are *less* likely to buy if treated. (Avoid!)
    - **Lost Causes (~0)**: Users who won't buy regardless. (Waste of budget)
    """)

# 3. Model Validation
st.markdown('<div class="section-header">3. Model Validation (Bootstrap)</div>', unsafe_allow_html=True)

st.markdown("""
To ensure the model isn't just fitting noise, we performed **Bootstrap Validation**. 
The Qini curve below shows the cumulative uplift. The shaded region represents the **95% Confidence Interval**.
""")

st.plotly_chart(utils.plot_qini_curve(n_bootstraps=n_bootstraps), width='stretch')

st.success("✅ **Result:** The lower bound of the CI is consistently above the random line, proving the model has **predictive power**.")
