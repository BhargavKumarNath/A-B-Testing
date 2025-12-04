import sys
import os
import pytest
import polars as pl
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.statistics import FrequentistEngine

def test_ate_calculation():
    """Test standard ATE calculation."""
    # Create a clear signal: Treatment = Control + 0.1
    n = 10000
    df = pl.DataFrame({
        "treatment": np.concatenate([np.zeros(5000), np.ones(5000)]).astype(int),
        "conversion": np.concatenate([np.random.normal(0.5, 1, 5000), np.random.normal(0.6, 1, 5000)])
    })
    
    engine = FrequentistEngine(df)
    result = engine.calculate_ate("conversion")
    
    assert 0.05 < result.absolute_effect < 0.15 # Should be around 0.1
    assert result.p_value < 0.05
    assert result.variant == "Simple Difference"

def test_cuped_variance_reduction():
    """Test that CUPED reduces Standard Error when a correlated covariate exists."""
    n = 10000
    treatment = np.random.randint(0, 2, n)
    
    # Pre-experiment covariate (e.g., pre-conversion rate)
    X = np.random.normal(10, 2, n)
    
    # Outcome is highly correlated with X
    # Y = X + Treatment_Effect + Noise
    Y = X + (treatment * 0.5) + np.random.normal(0, 0.5, n)
    
    df = pl.DataFrame({
        "treatment": treatment,
        "conversion": Y,
        "f0": X
    })
    
    engine = FrequentistEngine(df)
    
    # 1. Standard ATE
    std_result = engine.calculate_ate("conversion")
    
    # 2. CUPED ATE
    cuped_result = engine.calculate_cuped("conversion", covariates=["f0"])
    
    print(f"\nStandard SE: {std_result.std_error:.5f}")
    print(f"CUPED SE:    {cuped_result.std_error:.5f}")
    
    # CUPED SE should be significantly lower because X explains much of the variance in Y
    assert cuped_result.std_error < std_result.std_error
    assert cuped_result.variant == "CUPED Adjusted"
    
    # Effect size should be similar (unbiased)
    assert abs(cuped_result.absolute_effect - 0.5) < 0.1

if __name__ == "__main__":
    test_ate_calculation()
    test_cuped_variance_reduction()
    print("All statistical tests passed.")