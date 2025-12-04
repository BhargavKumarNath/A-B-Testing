import sys
import os
import pytest
import polars as pl
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.distillation import DistillationEngine

class MockTeacher:
    """A fake teacher that just returns the sum of features as the prediction"""
    def predict(self, df: pl.DataFrame)-> np.ndarray:
        return df["f0"].to_numpy() + df["f1"].to_numpy()

def test_distillation_fidelity():
    """
    Test if the student can learn a simple function from the teacher.
    """
    n = 1000
    df = pl.DataFrame({
        "f0": np.random.rand(n),
        "f1": np.random.rand(n),
        "f2": np.random.rand(n) # Irrelevant feature
    })
    
    teacher = MockTeacher()
    
    # We want the student to learn CATE = f0 + f1
    # A decision tree should approximate this reasonably well
    
    engine = DistillationEngine(teacher, ["f0", "f1", "f2"])
    
    # Train Student
    r2 = engine.train_student(df, max_depth=10)
    
    print(f"\nStudent R2: {r2:.4f}")
    
    # The relationship is linear, so a deep tree should capture it well (R2 > 0.8)
    assert r2 > 0.80

    # Test Prediction
    preds = engine.predict(df)
    assert len(preds) == n

if __name__ == "__main__":
    test_distillation_fidelity()
