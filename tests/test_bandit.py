import sys
import os
import pytest
import polars as pl
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.bandit import LinUCBAgent, BanditSimulator

def test_linucb_learning():
    """Test that LinUCB learns to pick the optimal arm."""
    n_features = 2
    agent = LinUCBAgent(n_features)
    
    # Context is constant for simplicity
    x = np.array([1.0, 0.5])
    
    # Scenario: Arm 1 gives reward 1.0, Arm 0 gives reward 0.0
    # Train it a few times
    for _ in range(50):
        # Force feed it "Arm 1 is good"
        agent.update(arm=1, x=x, reward=1.0)
        # Force feed it "Arm 0 is bad"
        agent.update(arm=0, x=x, reward=0.0)
        
    # Now ask it to choose
    choice = agent.select_arm(x)
    
    # It should choose Arm 1
    assert choice == 1
    print("\nTest Passed: LinUCB learned to pick the high-reward arm.")

def test_replay_simulation():
    """Test the replay mechanics."""
    df = pl.DataFrame({
        "f0": np.random.rand(100),
        "f1": np.random.rand(100),
        "treatment": np.random.randint(0, 2, 100),
        "conversion": np.random.randint(0, 2, 100)
    })
    
    sim = BanditSimulator(df, ["f0", "f1"], "conversion", "treatment")
    result = sim.run_replay(sample_size=50)
    
    assert result['aligned_events'] > 0
    assert "ctr" in result
    print(f"Simulation CTR: {result['ctr']:.2%}")

if __name__ == "__main__":
    test_linucb_learning()
    test_replay_simulation()