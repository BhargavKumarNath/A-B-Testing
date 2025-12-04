import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_confidence_interval(p, n, z=1.96):
    """
    Calculates the margin of error for a proportion (95% CI).
    Standard Error = sqrt( p * (1-p) / n )
    Margin of Error = z * SE
    """
    se = np.sqrt((p * (1 - p)) / n)
    return z * se

def main():
    DATA_PATH = "data/criteo_uplift.parquet"
    OUTPUT_IMG = "plots/ate_results.png"
    if not os.path.exists(DATA_PATH):
        logger.error(f"File not found: {DATA_PATH}")
        return

    logger.info(f"Loading data from {DATA_PATH}...")
    df = pl.read_parquet(DATA_PATH)

    # 1. Aggregate Data
    logger.info("Calculating aggregate statistics...")
    
    stats = (
        df.group_by("treatment")
        .agg([
            pl.len().alias("n_obs"),
            pl.col("conversion").sum().alias("conversions"),
            pl.col("conversion").mean().alias("conversion_rate")
        ])
        .sort("treatment")
    )
    
    # Note: sort("treatment") ensures index 0 is Control, 1 is Treatment
    ctrl_row = stats.filter(pl.col("treatment") == 0)
    treat_row = stats.filter(pl.col("treatment") == 1)

    n_c = ctrl_row["n_obs"][0]
    conv_c = ctrl_row["conversions"][0]
    cr_c = ctrl_row["conversion_rate"][0]

    n_t = treat_row["n_obs"][0]
    conv_t = treat_row["conversions"][0]
    cr_t = treat_row["conversion_rate"][0]

    # 2. Calculate ATE and Lift
    ate = cr_t - cr_c
    relative_lift = (ate / cr_c) * 100

    # Calculate Confidence Intervals (95%)
    ci_c = calculate_confidence_interval(cr_c, n_c)
    ci_t = calculate_confidence_interval(cr_t, n_t)

    # 3. Print Business Metrics
    print("\n" + "="*40)
    print("      BASELINE A/B TEST RESULTS      ")
    print("="*40)
    print(f"{'Metric':<20} | {'Control (No Ad)':<15} | {'Treatment (Ad)':<15}")
    print("-" * 56)
    print(f"{'Sample Size':<20} | {n_c:,}<15 | {n_t:,}<15")
    print(f"{'Conversions':<20} | {conv_c:,}<15 | {conv_t:,}<15")
    print(f"{'Conv. Rate (CR)':<20} | {cr_c:.4%}<15 | {cr_t:.4%}<15")
    print("-" * 56)
    print(f"Absolute Lift (ATE):  {ate:.4%}")
    print(f"Relative Lift:        {relative_lift:.2f}%")
    print("="*40 + "\n")

    # 4. Plotting
    logger.info(f"Generating plot -> {OUTPUT_IMG}")
    
    labels = ['Control', 'Treatment']
    rates = [cr_c, cr_t]
    errors = [ci_c, ci_t] # 95% CI bars
    colors = ['#bdc3c7', '#2ecc71'] # Grey for Control, Green for Treatment

    plt.figure(figsize=(8, 6))
    
    # Create bars with error bars
    bars = plt.bar(labels, rates, yerr=errors, capsize=10, color=colors, alpha=0.9)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (height*0.01), 
                 f'{height:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylabel('Conversion Rate')
    plt.title(f'Global Treatment Effect\nLift: +{relative_lift:.2f}%', fontsize=14)
    plt.ylim(0, max(rates) * 1.2) # Give some headroom for text
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Save and Show
    plt.savefig(OUTPUT_IMG, dpi=300)
    logger.info("Plot saved successfully.")
    
    # If running in VS Code with a display, this might show the window. 
    # If not, the file is saved.
    try:
        plt.show() 
    except:
        pass

if __name__ == "__main__":
    main()