"""Phase 4 addendum: score the LinUCB bandit's decisions with the IPW estimator instead of replay.

Same data, split, stream and agent as pipeline.py. Shapes: X (n, 12), arms/T/Y (n,). Writes results/phase4/bandit_ipw.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.components.bandit import LinUCBAgent  # noqa: E402
from src.components.data_loader import DataLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("p4b")
FEATURES = [f"f{i}" for i in range(12)]
V, C, N = 10.0, 0.10, 1_000_000


def main() -> None:
    df = DataLoader("data/criteo_uplift.parquet").load()
    u = np.random.default_rng(42).random(df.height)
    test = df.filter(pl.Series(u >= 0.8))
    stream = test.sample(n=N, shuffle=True, seed=42)  # identical call to BanditSimulator.run_replay
    X = stream.select(FEATURES).to_numpy()
    y = stream["conversion"].to_numpy().reshape(-1).astype(np.float64)
    t = stream["treatment"].to_numpy().reshape(-1).astype(np.float64)
    assert X.shape == (N, 12) and y.shape == t.shape == (N,)
    p = float(t.mean())
    agent = LinUCBAgent(n_features=12, alpha=1.0)
    arms = np.empty(N, dtype=np.float64)
    matched_profit, n_match = 0.0, 0
    for i in range(N):
        a = agent.select_arm(X[i])
        arms[i] = a
        if a == t[i]:
            prof = y[i] * V - (C if t[i] == 1 else 0.0)
            agent.update(a, X[i], (prof + C) / (V + C))
            matched_profit += prof
            n_match += 1
        if i % 100000 == 0:
            log.info("event %d, matched %d", i, n_match)
    terms = lambda pi: pi * t / p * (V * y - C) + (1 - pi) * (1 - t) / (1 - p) * V * y  # noqa: E731
    rng = np.random.default_rng(0)
    cols = {"bandit": terms(arms), "treat_all": terms(np.ones(N)), "treat_none": terms(np.zeros(N))}
    boots = {k: np.empty(1000) for k in cols}
    for b in range(1000):
        idx = rng.integers(0, N, N)
        for k, v in cols.items():
            boots[k][b] = v[idx].mean()
    out = {"events": N, "p_treated": p, "replay_matched": n_match, "replay_profit_per_matched": matched_profit / n_match,
           "share_bandit_picks_treat": float(arms.mean()),
           "ipw": {k: {"mean": float(v.mean()), "lo": float(np.percentile(boots[k], 2.5)), "hi": float(np.percentile(boots[k], 97.5))}
                   for k, v in cols.items()}}
    d = boots["bandit"] - boots["treat_none"]
    out["ipw_bandit_minus_treat_none"] = {"mean": float(cols["bandit"].mean() - cols["treat_none"].mean()),
                                          "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
    Path("results/phase4/bandit_ipw.json").write_text(json.dumps(out, indent=2))
    log.info("%s", out)


if __name__ == "__main__":
    main()
