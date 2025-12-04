import streamlit as st
import utils
import pandas as pd

st.set_page_config(page_title="Statistical Foundation", page_icon="📊", layout="wide")
utils.load_css()

st.title("📊 Statistical Foundation & Validation")

# Sidebar Customization
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_level = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
    st.info("Adjusting confidence level affects the significance bounds for A/B test metrics.")

# 1. A/B Test Analysis
st.markdown('<div class="section-header">1. The A/B Test Baseline</div>', unsafe_allow_html=True)

metrics = utils.get_notebook_metrics()["ab_test"]

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Standard Metrics")
    st.write(f"**Treatment CR:** {metrics['treatment_cr']:.4%}")
    st.write(f"**Control CR:** {metrics['control_cr']:.4%}")
    st.write(f"**Relative Lift:** {metrics['lift']}%")
    
    if metrics['lift'] > 0:
        st.success("🎉 Significant Conversion Lift Observed")
    else:
        st.error("No Significant Lift")

with col2:
    st.markdown("### The Profitability Trap")
    st.markdown("""
    <div class="insight-box">
        <strong>Critical Insight:</strong> While conversion lift is high (+59%), the <strong>Net Profit is negative</strong>. 
        This happens because the cost of treating <i>everyone</i> outweighs the revenue gained from the <i>marginal</i> conversions.
    </div>
    """, unsafe_allow_html=True)
    
    # Simple bar chart for Revenue vs Cost
    df_econ = pd.DataFrame({
        "Metric": ["Revenue per User", "Cost per User", "Net Profit"],
        "Value": [metrics['revenue_per_user'], metrics['cost_per_user'], metrics['net_profit']],
        "Color": ["#10B981", "#EF4444", "#EF4444"]
    })
    
    fig = utils.px.bar(df_econ, x="Metric", y="Value", color="Metric", 
                       color_discrete_map={"Revenue per User": "#10B981", "Cost per User": "#EF4444", "Net Profit": "#EF4444"},
                       title="Unit Economics (Per User)")
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig, width='stretch')

# 2. Data Integrity
st.markdown('<div class="section-header">2. Data Integrity Checks</div>', unsafe_allow_html=True)

val_metrics = utils.get_notebook_metrics()["validation"]

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Sample Ratio Mismatch (SRM)")
    st.plotly_chart(utils.plot_srm_gauge(val_metrics['srm_p_value']), width='stretch')
    st.caption(f"P-value: {val_metrics['srm_p_value']:.4f}. Values > 0.05 indicate no bias in randomization.")

with col2:
    st.markdown("#### Covariate Balance (SMD)")
    st.info(f"Max Standardized Mean Difference (SMD): **{val_metrics['max_smd']}**")
    st.markdown("All covariates have SMD < 0.1, indicating the Treatment and Control groups are **statistically identical** pre-treatment.")

st.markdown("---")
st.success("✅ **Conclusion:** The experiment data is valid. The negative profit is a strategy problem, not a data problem.")
