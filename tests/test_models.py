import sys
import os
import pytest
import polars as pl
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.models import TLearner

def test_tlearner_heterogeneity():
    """
    Verifies that the T-Learner can detect heterogeneous treatment effects.
    Scenario:
    - Feature 'f0' is the driver.
    - If f0 > 0: Treatment adds +0.5 to conversion prob.
    - If f0 <= 0: Treatment has NO effect (0.0).
    """
    n = 5000
    
    # 1. Generate Synthetic Data
    f0 = np.random.uniform(-1, 1, n)
    treatment = np.random.randint(0, 2, n)
    
    # Base conversion rate (random noise)
    base_prob = 0.2
    
    # Treatment Effect: 0.5 if f0 > 0 else 0
    cate_true = np.where(f0 > 0, 0.5, 0.0)
    
    # Outcome Y
    # Prob = Base + Treatment * CATE
    prob = base_prob + (treatment * cate_true)
    
    # Clip prob to [0,1] just in case
    prob = np.clip(prob, 0, 1)
    
    # Bernoulli sampling
    y = np.random.binomial(1, prob)
    
    df = pl.DataFrame({
        "f0": f0,
        "treatment": treatment,
        "conversion": y
    })
    
    # 2. Train Model
    learner = TLearner(features=["f0"], n_estimators=50) # Low estimators for speed in test
    learner.fit(df, treatment_col="treatment", target_col="conversion")
    
    # 3. Predict Uplift on Test Set
    # We create pure test cases
    test_df = pl.DataFrame({"f0": [-0.5, 0.5, -0.8, 0.8], "treatment": [0,0,0,0], "conversion": [0,0,0,0]})
    pred_uplift = learner.predict(test_df)
    
    print("\nPredicted Uplift for f0 = [-0.5, 0.5, -0.8, 0.8]:")
    print(pred_uplift)
    
    # 4. Assertions
    # f0 = -0.5 should have uplift ~ 0
    assert abs(pred_uplift[0]) < 0.15, f"Expected ~0 uplift for negative f0, got {pred_uplift[0]}"
    
    # f0 = 0.5 should have uplift ~ 0.5
    assert abs(pred_uplift[1] - 0.5) < 0.2, f"Expected ~0.5 uplift for positive f0, got {pred_uplift[1]}"
    
    # Check ranking
    assert pred_uplift[1] > pred_uplift[0]
    
    print("Test Passed: Model correctly identified heterogeneity.")

if __name__ == "__main__":
    test_tlearner_heterogeneity()