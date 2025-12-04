import streamlit as st
import utils
import pandas as pd
import numpy as np

st.set_page_config(page_title="Knowledge Distillation", page_icon="⚡", layout="wide")
utils.load_css()

st.title("⚡ Knowledge Distillation & Segmentation")

# Sidebar
with st.sidebar:
    st.header("⚙️ Filters")
    st.multiselect("Filter Segments", ["Persuadables", "Sleeping Dogs", "Lost Causes"], default=["Persuadables"])

# 1. Distillation Overview
st.markdown('<div class="section-header">1. Model Distillation</div>', unsafe_allow_html=True)

dist_stats = utils.get_distillation_stats()

st.markdown("""
The X-Learner is a complex ensemble of models (slow). For production, we **distilled** its knowledge into a single, interpretable **Decision Tree** (Surrogate Model).
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Original Latency", dist_stats["latency"]["X-Learner"])
with col2:
    st.metric("Distilled Latency", dist_stats["latency"]["Distilled Tree"])
with col3:
    st.metric("Speedup Factor", dist_stats["latency"]["Speedup"], delta="120x")

# 2. Interpretable Rules
st.markdown('<div class="section-header">2. Interpretable Segment Rules</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Decision Logic")
    st.code(dist_stats["tree_rules"], language="text")
    st.caption("These rules are extracted from the surrogate decision tree.")

with col2:
    st.markdown("### Feature Importance (SHAP)")
    df_imp = dist_stats["feature_importance"]
    fig = utils.px.bar(df_imp, x="Importance", y="Feature", orientation='h', title="Top Features Driving Uplift")
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, width='stretch')

# 3. Segment Explorer
st.markdown('<div class="section-header">3. Segment Explorer</div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
    <strong>Strategic Insight:</strong> The "Persuadable" segment is characterized by <strong>High `f2`</strong> and <strong>Low `f7`</strong>. 
    Marketing efforts should focus exclusively on this group to maximize ROI.
</div>
""", unsafe_allow_html=True)

# Dummy scatter plot for segments
df_seg = pd.DataFrame({
    "f2": np.random.rand(100),
    "f7": np.random.rand(100),
    "Segment": np.random.choice(["Persuadables", "Sleeping Dogs", "Lost Causes"], 100)
})

fig_seg = utils.px.scatter(df_seg, x="f2", y="f7", color="Segment", 
                     title="Segment Clusters (f2 vs f7)",
                     color_discrete_map={"Persuadables": "#10B981", "Sleeping Dogs": "#EF4444", "Lost Causes": "#6B7280"})
fig_seg.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_seg, width='stretch')
