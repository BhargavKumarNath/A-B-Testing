import sys
import os
import pytest
import polars as pl
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.segmentation import SegmentAnalyzer

def test_segment_profiling():
    """Test that the analyzer correctly identifies differentiating features."""
    n = 1000
    
    # Create features
    # f0 is the differentiator
    f0 = np.concatenate([np.random.normal(10, 1, 500), np.random.normal(0, 1, 500)])
    f1 = np.random.normal(5, 5, n) # Noise
    
    df = pl.DataFrame({"f0": f0, "f1": f1})
    
    # Uplift correlates perfectly with f0
    uplift_scores = f0 * 0.1 
    
    analyzer = SegmentAnalyzer(df, ["f0", "f1"])
    
    # 1. Test Profile
    profile = analyzer.get_segment_profiles(uplift_scores, top_percentile=0.5)
    print("\nSegment Profile:")
    print(profile)
    
    # f0 should have a large Diff %, f1 should be small
    assert profile.loc["f0", "Diff %"] > 50
    assert abs(profile.loc["f1", "Diff %"]) < 50
    
    # 2. Test Surrogate Rule
    rules = analyzer.explain_with_surrogate(uplift_scores, max_depth=2)
    print("\nSurrogate Rules:")
    print(rules)
    
    # The rule should split on f0
    assert "f0" in rules

if __name__ == "__main__":
    test_segment_profiling()