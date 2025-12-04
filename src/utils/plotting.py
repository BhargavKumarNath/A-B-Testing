import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
from pathlib import Path

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12

def plot_uplift_by_decile(stats: pl.DataFrame, save_path: str):
    """Plots Actual Lift per Decile."""
    setup_style()
    df_plot = stats.to_pandas()
    df_plot['bin'] = df_plot['bin'].astype(str)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='bin', y='actual_lift', hue='bin', palette='viridis', legend=False)
    
    global_ate = df_plot['actual_lift'].mean()
    plt.axhline(y=global_ate, color='r', linestyle='--', label=f'Avg Lift ({global_ate:.4f})')
    
    plt.title('Actual Lift by Predicted Decile (Validation)', fontsize=16)
    plt.xlabel('Decile (High Uplift -> Low Uplift)')
    plt.ylabel('Actual Lift')
    plt.legend()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_bootstrapped_qini(qini_data: dict, save_path: str):
    """
    Plots the Qini Curve with 95% Confidence Intervals.
    """
    setup_style()
    plt.figure(figsize=(10, 10))
    
    # Extract data
    x = qini_data['x']
    mean_y = qini_data['y_mean']
    lower_y = qini_data['y_lower']
    upper_y = qini_data['y_upper']
    
    # Plot Mean Curve
    plt.plot(x, mean_y, label=f'Uplift Model (AUUC: {qini_data["auuc_mean"]:.2f})', linewidth=2.5, color='blue')
    
    # Plot Confidence Interval
    plt.fill_between(x, lower_y, upper_y, color='blue', alpha=0.2, label='95% Confidence Interval')
    
    # Plot Random Line
    plt.plot([0, 1], [0, mean_y[-1]], 'k--', label='Random Targeting')
    
    plt.title('Qini Curve with Uncertainty (Bootstrapped)', fontsize=16)
    plt.xlabel('Fraction of Population Targeted')
    plt.ylabel('Incremental Conversions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_bandit_performance(bandit_history: list, global_ctr: float, save_path: str):
    """Plots Cumulative Reward/Profit."""
    setup_style()
    n_events = len(bandit_history)
    x_axis = np.arange(n_events)
    bandit_curve = np.array(bandit_history)
    
    # Baseline assumes constant performance
    baseline_curve = np.cumsum(np.ones(n_events) * global_ctr)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, bandit_curve, label='LinUCB Bandit (Adaptive)', linewidth=2, color='green')
    plt.plot(x_axis, baseline_curve, label='Fixed Strategy (Baseline)', linewidth=2, linestyle='--', color='gray')
    
    plt.fill_between(x_axis, bandit_curve, baseline_curve, where=(bandit_curve > baseline_curve), 
                     interpolate=True, color='green', alpha=0.1, label='Incremental Value')
    
    plt.title('Adaptive Experimentation Performance', fontsize=16)
    plt.xlabel('Number of Impressions')
    plt.ylabel('Cumulative Value (Profit/Conversions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()