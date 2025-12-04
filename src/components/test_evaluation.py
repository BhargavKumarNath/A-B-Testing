import sys
import os
import pytest
import polars as pl
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.evaluation import UpliftEvaluator

def test_decile_analysis():
    """
    Robust test for UpliftEvaluator:
    Ensures top predicted uplift bins have higher actual lift than bottom bins.
    """
    n = 10000
    
    # 1. Create monotonic predicted uplift (no shuffling)
    pred_uplift = np.linspace(-0.1, 0.5, n)  # low -> high
    
    # 2. Treatment assignment
    treatment = np.random.randint(0, 2, n)
    
    # 3. Simulate conversion probabilities
    base_prob = 0.2
    prob = np.where(treatment == 1, base_prob + pred_uplift, base_prob)
    prob = np.clip(prob, 0, 1)
    
    conversion = np.random.binomial(1, prob)
    
    # 4. Build dataframe
    df = pl.DataFrame({
        "treatment": treatment,
        "conversion": conversion
    })
    
    # 5. Evaluate deciles
    evaluator = UpliftEvaluator(df)
    stats = evaluator.get_decile_stats(pred_uplift, n_bins=5)
    
    print("\nDecile Stats:")
    print(stats)
    
    # 6. Assertions
    # Since we sorted descending in the evaluator:
    # Row 0 -> highest predicted uplift
    # Row -1 -> lowest predicted uplift
    top_bin_lift = stats["actual_lift"][0]
    bottom_bin_lift = stats["actual_lift"][-1]
    
    print(f"Top Bin Lift: {top_bin_lift:.4f}")
    print(f"Bottom Bin Lift: {bottom_bin_lift:.4f}")
    
    # Robust checks
    assert top_bin_lift > bottom_bin_lift       # top bin has higher lift
    assert top_bin_lift > 0.3                   # sanity check
    assert bottom_bin_lift < 0.2                # bottom bin near baseline

if __name__ == "__main__":
    test_decile_analysis()
    print("Evaluation test passed.")
