import streamlit as st
import utils

# Page Config
st.set_page_config(
    page_title="Criteo Uplift | Home",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
utils.load_css()

# Sidebar
with st.sidebar:
    st.title("Navigation")
    st.info("Select a page above to explore specific analysis modules.")
    st.markdown("---")
    st.markdown("### Project Info")
    st.caption("Dataset: Criteo Uplift (14M rows)")
    st.caption("Model: X-Learner (XGBoost)")
    st.caption("Status: Production Ready")

# Main Content
st.title("🚀 Criteo Uplift: Algorithmic Profit Optimization")

st.markdown("""
<div class="insight-box">
    <strong>Executive Summary:</strong> We transitioned from a standard A/B testing approach (which was losing money) 
    to an algorithmic Uplift Modeling strategy. This shift turned a <strong>$0.05 loss per user</strong> into a 
    <strong>$0.09 profit per user</strong>, unlocking significant ROI at scale.
</div>
""", unsafe_allow_html=True)

# Project At A Glance
st.markdown('<div class="section-header">Project At A Glance</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

metrics = utils.get_notebook_metrics()

with col1:
    st.markdown(utils.plot_metric_card("Total Dataset", "14M+", help_text="Total rows in Criteo Uplift dataset"), unsafe_allow_html=True)
with col2:
    st.markdown(utils.plot_metric_card("Baseline Lift", "59.5%", delta=59.45, help_text="Conversion lift from A/B test"), unsafe_allow_html=True)
with col3:
    st.markdown(utils.plot_metric_card("Net Profit (A/B)", "-$0.05", delta=-0.07, help_text="Net profit per user with standard targeting"), unsafe_allow_html=True)
with col4:
    st.markdown(utils.plot_metric_card("Net Profit (Uplift)", "+$0.09", delta=0.09, help_text="Net profit per user with Uplift Model"), unsafe_allow_html=True)

# The Journey
st.markdown('<div class="section-header">The Optimization Journey</div>', unsafe_allow_html=True)

st.markdown("""
This dashboard guides you through the complete lifecycle of the project. Select a page from the sidebar to dive deep:

1.  **Statistical Foundation**: Understanding why the A/B test failed despite high lift (The "Profitability Trap").
2.  **Causal Engine**: How we used **X-Learners** to predict individual treatment effects (CATE).
3.  **Profit Optimization**: Using **Bandit Analysis** to simulate policies and maximize ROI.
4.  **Knowledge Distillation**: How we distilled the complex ensemble into a fast **Decision Tree** for production (<1ms latency).
""")

# Key Takeaways
st.markdown('<div class="section-header">Key Technical Wins</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.success("✅ **Solved Imbalanced Data**")
    st.markdown("Standard T-Learners failed due to low conversion rates (0.2%). The **X-Learner** architecture handled this efficiently.")

with col2:
    st.success("✅ **Production Latency**")
    st.markdown("Distilled the heavy meta-learner into a lightweight **Surrogate Model**, reducing inference time by **120x**.")

st.markdown("---")
st.caption("Built with Streamlit • Causal Inference • Plotly")
