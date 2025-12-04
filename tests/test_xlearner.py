import sys
import os
import pytest
import polars as pl
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.models import XLearner
import logging
logging.basicConfig(level=logging.INFO)

def test_xlearner_fit_predict():
    """Test X-Learner training and prediction pipeline"""
    n = 1000
    df = pl.DataFrame({
        "f0": np.random.rand(n),
        "f1": np.random.rand(n),
        "treatment": np.random.randint(0, 2, n),
        "conversion": np.random.randint(0, 2, n)
    })

    learner = XLearner(features=["f0", "f1"], n_estimators=10)

    # 1. Fit
    learner.fit(df, "treatment", "conversion")

    assert learner.propensity is not None
    assert learner.tau0 is not None
    assert learner.tau1 is not None

    # 2. Predict 
    preds = learner.predict(df)

    assert len(preds) == n
    assert isinstance(preds, np.ndarray)
    assert not np.isnan(preds).any()

    print(f"\nX-Learner Test Passed")

if __name__ == "__main__":
    test_xlearner_fit_predict()