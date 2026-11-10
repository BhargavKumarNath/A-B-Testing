"""IPW policy value: unbiased on a randomised log where the replay average is not."""
import numpy as np

from src.components.bandit import BanditSimulator


def _log(n=400_000, p=0.85, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random(n) < 0.5                     # context: "persuadable" half
    t = (rng.random(n) < p).astype(np.float64)
    # conversion prob: persuadables 0.05 base +0.06 uplift when treated; others 0.01, no uplift
    prob = np.where(x, 0.05 + 0.06 * t, 0.01)
    y = (rng.random(n) < prob).astype(np.float64)
    return x, t, y


def test_ipw_matches_true_value_of_known_policy():
    x, t, y = _log()
    arms = x.astype(np.float64)                 # treat exactly the persuadables
    got = BanditSimulator.ipw_profit(arms, t, y)["bandit"]
    # truth: half the users treated: 0.5*(10*0.11-0.10) + 0.5*(10*0.01)
    truth = 0.5 * (10 * 0.11 - 0.10) + 0.5 * (10 * 0.01)
    assert abs(got - truth) < 0.02


def test_replay_average_is_biased_but_ipw_is_not():
    x, t, y = _log()
    arms = x.astype(np.float64)
    matched = arms == t
    prof = y * 10.0 - np.where(t == 1, 0.10, 0.0)
    replay = prof[matched].mean()
    truth = 0.5 * (10 * 0.11 - 0.10) + 0.5 * (10 * 0.01)
    ipw = BanditSimulator.ipw_profit(arms, t, y)["bandit"]
    assert abs(ipw - truth) < abs(replay - truth)
    assert abs(replay - truth) > 0.1            # replay over-weights treated contexts under an 85/15 log


def test_fixed_policies():
    x, t, y = _log()
    r = BanditSimulator.ipw_profit(np.zeros(len(t)), t, y)
    assert r["treat_none"] == r["bandit"] and r["share_treated"] == 0.0
