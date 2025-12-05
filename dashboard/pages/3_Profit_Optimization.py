import streamlit as st
import utils
import pandas as pd

st.set_page_config(page_title="Profit Optimization", page_icon="💰", layout="wide")
utils.load_css()

st.title("💰 Profit Optimization & Bandit Analysis")

# Sidebar
with st.sidebar:
    st.header("⚙️ Economics")
    ad_cost = st.slider("Ad Cost ($)", 0.0, 5.0, 0.10, 0.01)
    conv_value = st.slider("Conversion Value ($)", 0.0, 50.0, 10.0, 0.5)
    budget = st.number_input("Budget Cap ($)", value=10000)

# 1. Bandit Analysis
st.markdown('<div class="section-header">1. Bandit Analysis</div>', unsafe_allow_html=True)

st.markdown("""
We compared three policies:
1.  **Random (A/B)**: Treat everyone (Exploration).
2.  **Uplift Greedy**: Treat top k% based on model score (Exploitation).
3.  **Thompson Sampling**: Bandit approach balancing exploration and exploitation.
""")

bandit_stats = utils.get_bandit_stats()
df_bandit = bandit_stats["policy_comparison"]

col1, col2 = st.columns([2, 1])

with col1:
    fig = utils.px.bar(
        df_bandit,
        x="Policy", y="Avg Profit per User", color="Policy",
        title="Average Profit per User by Policy",
        color_discrete_sequence=["#EF4444", "#10B981", "#6366F1"]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(df_bandit, hide_index=True)
    st.info("The **Bandit Policy** (Thompson Sampling) yields the highest long-term profit by dynamically adjusting to new data.")

# 2. Profit Simulator
st.markdown('<div class="section-header">2. Interactive Profit Simulator</div>', unsafe_allow_html=True)

st.markdown("Adjust the **Ad Cost** and **Conversion Value** in the sidebar to see how the optimal targeting threshold changes.")

fig_sim = utils.plot_profit_simulation(ad_cost, conv_value)
st.plotly_chart(fig_sim, use_container_width=True)

# 3. ROI Calculator
st.markdown('<div class="section-header">3. ROI Calculator</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

optimal_users = 10000 * 0.2
cost = optimal_users * ad_cost
revenue = optimal_users * (0.002 * 3) * conv_value
profit = revenue - cost
roi = (profit / cost) * 100 if cost > 0 else 0

with col1:
    st.metric("Projected Cost", f"${cost:,.2f}")
with col2:
    st.metric("Projected Revenue", f"${revenue:,.2f}")
with col3:
    st.metric("Projected ROI", f"{roi:.1f}%", delta=roi)

# 4. Strategic Breakeven Analysis
st.markdown('<div class="section-header">4. Strategic Breakeven Analysis</div>', unsafe_allow_html=True)
st.markdown("Red zones indicate market conditions where we should pause the campaign. Blue zones are profitable.")
st.plotly_chart(utils.plot_profit_heatmap(), use_container_width=True)
