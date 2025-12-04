import sys
import os
import pytest
import polars as pl
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.evaluation import UpliftEvaluator

def test_bootstrap_qini():
    """Verify that bootstrapping produces valid confidence intervals"""
    n = 2000
    df = pl.DataFrame({
        "treatment": np.random.randint(0, 2, n),
        "conversion": np.random.randint(0, 2, n)
    })
    scores = np.random.rand(n)

    evaluator = UpliftEvaluator(df)

    # Run with small bootstraps for speed
    res = evaluator.get_bootstrapped_qini(scores, n_bootstraps=10)

    # Checks
    assert len(res["x"]) == 1000
    assert len(res["y_mean"]) == 1000
    assert len(res["y_lower"]) == 1000
    assert len(res["y_upper"]) == 1000

    # Logic check: Upper >= Mean => Lower
    # We check the middle of the curve (index 500)
    idx = 500
    assert res["y_upper"][idx] >= res["y_mean"][idx]
    assert res["y_mean"][idx] >= res["y_lower"][idx]

    print(f"\nBootstrap Test Passed")
    print(f"AUUC: {res['auuc_mean']:.4f} ± {res['auuc_std']:.4f}")

if __name__ == "__main__":
    test_bootstrap_qini()