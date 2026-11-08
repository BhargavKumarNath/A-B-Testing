import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.bandit import BanditSimulator
from src.components.uncertainty import bootstrap_mean_ci


def _profits(n: int, seed: int) -> np.ndarray:
    """Per-user profits shaped like the pipeline's: four distinct values."""
    rng = np.random.default_rng(seed)
    treated = rng.random(n) < 0.85
    converted = rng.random(n) < 0.03
    return converted * 10.0 - treated * 0.10


def test_same_seed_same_interval():
    values = _profits(5_000, seed=0)
    assert bootstrap_mean_ci(values, n_boot=2_000, seed=7) == bootstrap_mean_ci(values, n_boot=2_000, seed=7)


def test_interval_brackets_the_mean():
    values = _profits(5_000, seed=1)
    ci = bootstrap_mean_ci(values, n_boot=2_000)
    assert ci.lower < ci.mean < ci.upper
    assert ci.mean == pytest.approx(values.mean())
    assert ci.n == values.size


def test_matches_row_resampling_bootstrap():
    """The multinomial shortcut must give the same distribution as resampling rows."""
    values = _profits(4_000, seed=2)
    n_boot = 4_000
    rng = np.random.default_rng(3)
    row_means = values[rng.integers(0, values.size, (n_boot, values.size))].mean(axis=1)
    ci = bootstrap_mean_ci(values, n_boot=n_boot, seed=4)
    lo, hi = np.quantile(row_means, [0.025, 0.975])
    width = hi - lo
    # Two independent Monte Carlo estimates of the same quantiles.
    assert ci.lower == pytest.approx(lo, abs=0.05 * width)
    assert ci.upper == pytest.approx(hi, abs=0.05 * width)


def test_interval_width_matches_normal_theory():
    values = _profits(20_000, seed=5)
    ci = bootstrap_mean_ci(values, n_boot=5_000)
    se = values.std(ddof=1) / np.sqrt(values.size)
    assert (ci.upper - ci.lower) == pytest.approx(2 * 1.96 * se, rel=0.05)


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        bootstrap_mean_ci(np.array([]))


def test_bandit_returns_one_profit_per_matched_event():
    n = 500
    rng = np.random.default_rng(0)
    df = pl.DataFrame({
        "f0": rng.random(n),
        "f1": rng.random(n),
        "conversion": rng.binomial(1, 0.1, n),
        "treatment": rng.integers(0, 2, n),
    })
    res = BanditSimulator(df, ["f0", "f1"], "conversion", "treatment").run_replay(
        sample_size=n, conversion_value=10.0, cost_per_ad=1.0
    )
    profits = res["matched_profits"]
    assert profits.shape == (res["aligned_events"],)
    assert profits.sum() == pytest.approx(res["cumulative_reward"])
    assert profits.mean() == pytest.approx(res["avg_profit"])
