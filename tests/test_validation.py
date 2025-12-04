import sys
import os
import pytest
import polars as pl
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.validation import ExperimentValidator

def test_srm_check_pass():
    """Test validation with a balanced dataset matching expectations."""
    # Create 1000 rows, 85% treatment
    n = 10000
    # Precise counts to pass chi-square easily
    treat = np.ones(8500, dtype=int)
    ctrl = np.zeros(1500, dtype=int)
    treatment = np.concatenate([treat, ctrl])
    np.random.shuffle(treatment)
    
    df = pl.DataFrame({
        "treatment": treatment,
        "f0": np.random.normal(0, 1, n) # Random noise
    })
    
    validator = ExperimentValidator(df, expected_ratio=0.85)
    result = validator.check_srm()
    
    assert result['valid'] is True
    assert result['observed_counts']['treatment'] == 8500

def test_srm_check_fail():
    """Test validation fails when ratios are wrong."""
    # 50/50 split but we expect 0.85
    n = 1000
    df = pl.DataFrame({
        "treatment": np.random.randint(0, 2, n),
        "f0": np.random.normal(0, 1, n)
    })
    
    validator = ExperimentValidator(df, expected_ratio=0.85)
    result = validator.check_srm()
    
    assert result['valid'] is False

def test_covariate_balance():
    """Test SMD calculation."""
    # Create a situation where f0 is balanced, f1 is imbalanced
    n = 1000
    treatment = np.random.randint(0, 2, n)
    
    # f0: same distribution
    f0 = np.random.normal(10, 2, n)
    
    # f1: Treatment group has mean 100, Control has mean 0 (Huge SMD)
    f1 = np.where(treatment == 1, 
                  np.random.normal(100, 5, n), 
                  np.random.normal(0, 5, n))
    
    df = pl.DataFrame({
        "treatment": treatment,
        "f0": f0,
        "f1": f1
    })
    
    validator = ExperimentValidator(df)
    smd_df = validator.check_covariate_balance()
    
    # f0 should be balanced (|SMD| < 0.1)
    # f1 should be imbalanced (|SMD| > 0.1)
    
    f0_smd = smd_df.filter(pl.col("feature") == "f0")["smd"][0]
    f1_smd = smd_df.filter(pl.col("feature") == "f1")["smd"][0]
    
    assert abs(f0_smd) < 0.15  # generous buffer for random noise
    assert abs(f1_smd) > 1.0   # massive imbalance

if __name__ == "__main__":
    # Manual runner
    test_srm_check_pass()
    test_srm_check_fail()
    test_covariate_balance()
    print("All validation tests passed.")