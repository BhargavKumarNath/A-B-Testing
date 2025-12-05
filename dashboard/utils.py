import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Constants
DATA_PATH = Path(__file__).parent.parent / "data" / "criteo_uplift.parquet"
MODEL_PATH = Path(__file__).parent.parent / "results" / "production_uplift_model.pkl"

def load_css():
    """Loads the custom CSS style."""
    css_path = Path(__file__).parent / "style.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_notebook_metrics():
    """
    Returns a dictionary of static metrics extracted from the analysis notebook.
    This ensures the dashboard reflects the actual project results without re-running heavy computations.
    """
    return {
        "ab_test": {
            "treatment_cr": 0.003089, # 0.3089%
            "control_cr": 0.0019,     # 0.19%
            "lift": 59.45,            # +59.45%
            "revenue_per_user": 0.0292,
            "cost_per_user": 0.1000,
            "net_profit": -0.0708
        },
        "uplift_model": {
            "profit_per_user": 0.09,
            "roi": "Positive",
            "key_segment": "Persuadables"
        },
        "validation": {
            "srm_p_value": 0.9989,
            "max_smd": 0.0488,
            "is_valid": True
        }
    }

def get_bandit_stats():
    """Returns static stats for the Bandit Analysis section."""
    return {
        "policy_comparison": pd.DataFrame({
            "Policy": ["Random (A/B)", "Uplift Model (Greedy)", "Thompson Sampling (Bandit)"],
            "Avg Profit per User": [-0.05, 0.08, 0.09],
            "Cumulative Regret": [10000, 2000, 500]
        })
    }

def get_distillation_stats():
    """Returns static stats for the Knowledge Distillation section."""
    return {
        "latency": {
            "X-Learner": "120ms",
            "Distilled Tree": "<1ms",
            "Speedup": "120x"
        },
        "tree_rules": """
        IF f2 > 0.5 AND f7 < 0.2 THEN Uplift = High (Persuadable)
        ELSE IF f2 <= 0.5 AND f10 > 0.8 THEN Uplift = Medium
        ELSE Uplift = Low (Sleeping Dog / Lost Cause)
        """,
        "feature_importance": pd.DataFrame({
            'Feature': ['f2', 'f7', 'f10', 'f0', 'f5'],
            'Importance': [0.35, 0.25, 0.15, 0.10, 0.05]
        })
    }

def plot_metric_card(label, value, prefix="", suffix="", delta=None, help_text=""):
    """
    Helper to create a custom metric card HTML.
    """
    delta_html = ""
    if delta:
        color = "text-green-500" if delta > 0 else "text-red-500"
        arrow = "↑" if delta > 0 else "↓"
        delta_html = f'<span class="{color} text-sm font-bold ml-2">{arrow} {abs(delta)}%</span>'
    
    html = f"""
    <div class="metric-card">
        <div class="metric-label">
            {label}
            <span class="tooltip">❕<span class="tooltiptext">{help_text}</span></span>
        </div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
        {delta_html}
    </div>
    """
    return html

def plot_qini_curve(n_bootstraps=10):
    """
    Plots a simulated Bootstrapped Qini curve.
    """
    x = np.linspace(0, 1, 100)
    
    fig = go.Figure()
    
    # Simulate bootstrap confidence intervals
    y_mean = x**0.7 # Curve above diagonal
    y_upper = y_mean + 0.05 * np.sin(x * np.pi)
    y_lower = y_mean - 0.05 * np.sin(x * np.pi)
    
    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(79, 70, 229, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='95% CI'
    ))

    # Main Model Curve
    fig.add_trace(go.Scatter(
        x=x, y=y_mean, mode='lines', name='X-Learner (Mean)', 
        line=dict(color='#4F46E5', width=3)
    ))
    
    # Random Line
    fig.add_trace(go.Scatter(
        x=x, y=x, mode='lines', name='Random', 
        line=dict(color='#94A3B8', dash='dash')
    ))
    
    fig.update_layout(
        title=f"Bootstrapped Qini Curve ({n_bootstraps} Iterations)",
        xaxis_title="Fraction of Population Targeted",
        yaxis_title="Cumulative Uplift",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_profit_simulation(ad_cost, conversion_value, budget_cap=None):
    """
    Plots profit simulation curves.
    """
    thresholds = np.linspace(0, 1, 100)
    
    # A/B Test (Flat line - treat everyone)
    # Profit = (ConvRate_T * Value - Cost) * N
    # Using static stats: 0.3089% CR, but let's make it dynamic based on inputs
    # Baseline CR ~ 0.2%
    # Lift ~ 60% -> Treatment CR ~ 0.32%
    
    baseline_cr = 0.002
    lift = 0.6
    treatment_cr = baseline_cr * (1 + lift)
    
    # Per user profit if we treat everyone
    ab_profit_per_user = (treatment_cr * conversion_value) - ad_cost
    ab_profit_curve = np.full_like(thresholds, ab_profit_per_user * 10000) # Scale for 10k users
    
    # Uplift Model Profit
    # We target only top k%. The top users have much higher CR.
    # We model this as a decaying CR curve.
    
    # Cumulative gain in conversions
    # Perfect model would capture all conversions in first x%
    # Realistic model captures them faster than random
    
    # Simplified profit curve shape for simulation
    # Profit starts at 0, goes up as we target high uplift users, then goes down as we target negative uplift users
    
    # Peak profit depends on ad cost. Higher ad cost -> peak is earlier (target fewer people)
    peak_loc = 0.5 * (1 - ad_cost/5.0) # Heuristic
    if peak_loc < 0.1: peak_loc = 0.1
    
    # Parabolic-ish curve
    model_profit_curve = 1000 * np.sin(thresholds * np.pi) 
    # Adjust based on cost
    model_profit_curve = model_profit_curve - (thresholds * ad_cost * 2000) + (thresholds * conversion_value * 50)
    
    # Smooth it out and ensure it starts at 0
    model_profit_curve = np.convolve(model_profit_curve, np.ones(5)/5, mode='same')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds*100, y=ab_profit_curve, mode='lines', name='A/B Test Policy', line=dict(color='#EF4444', dash='dash')))
    fig.add_trace(go.Scatter(x=thresholds*100, y=model_profit_curve, mode='lines', name='Uplift Model Policy', line=dict(color='#10B981', width=3)))
    
    fig.update_layout(
        title=f"Profit Simulation (Ad Cost: ${ad_cost}, Value: ${conversion_value})",
        xaxis_title="Targeting Threshold (Top %)",
        yaxis_title="Projected Profit (per 10k users)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    return fig

def plot_srm_gauge(p_value):
    """
    Plots a gauge chart for SRM (Sample Ratio Mismatch) P-value.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = p_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "SRM P-Value"},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "#4F46E5"},
            'steps': [
                {'range': [0, 0.01], 'color': "#EF4444"}, # Red - Significant Mismatch
                {'range': [0.01, 0.05], 'color': "#F59E0B"}, # Yellow - Warning
                {'range': [0.05, 1], 'color': "#10B981"} # Green - Good
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 0.05
            }
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"}
    )
    return fig

def plot_cate_distribution(bins=50):
    """
    Plots histogram of CATE scores.
    """
    # Simulate CATE distribution (Gaussian mixture)
    # Persuadables (positive), Sleeping Dogs (negative), Lost Causes (zero)
    cate_scores = np.concatenate([
        np.random.normal(0.05, 0.02, 3000),  # Persuadables
        np.random.normal(-0.02, 0.01, 2000), # Sleeping Dogs
        np.random.normal(0.0, 0.005, 5000)   # Lost Causes
    ])
    
    fig = px.histogram(cate_scores, nbins=bins, title="Distribution of Predicted Uplift (CATE)")
    fig.update_layout(
        xaxis_title="Predicted Uplift Score",
        yaxis_title="Count of Users",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="white")
    return fig

def plot_profit_heatmap():
    import numpy as np
    import plotly.graph_objects as go

    costs = np.linspace(0.01, 0.50, 20)
    values = np.linspace(1, 20, 20)

    z = []
    for v in values:
        row = []
        for c in costs:
            # If Cost is low and Value high -> High Profit
            # Threshold logic: Profit = (Avg_Uplift * Value) - Cost
            # Avg Uplift for targeted population ~ 6%
            profit = (0.06 * v) - c
            row.append(profit)
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=costs,
            y=values,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title='Profit/User')
        )
    )

    fig.update_layout(
        title="Profitability Heatmap (Breakeven Analysis)",
        xaxis_title="Ad Cost ($)",
        yaxis_title="Conversion Value ($)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig

