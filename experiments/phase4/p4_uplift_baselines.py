"""Phase 4: X-Learner vs T-Learner vs response model vs random, plus CATE-threshold sensitivity.

Protocol: docs/phase4_protocol.md. Writes results/phase4/*.json.
Shapes: n_test rows; scores are (n_test,) float; policies are (n_test,) bool.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.components.data_loader import DataLoader  # noqa: E402
from src.components.evaluation import UpliftEvaluator  # noqa: E402
from src.components.models import TLearner, XLearner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("p4")

FEATURES = [f"f{i}" for i in range(12)]
V, C = 10.0, 0.10
BUDGETS = [0.10, 0.20, 0.30, 0.50]
TAUS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06]
N_BOOT, N_BOOT_AUUC = 1000, 200
OUT = Path("results/phase4")


def ipw_terms(pi: np.ndarray, t: np.ndarray, y: np.ndarray, p: float) -> np.ndarray:
    """Per-row terms whose mean is the policy's profit per user. pi may be 0/1 or a probability."""
    assert pi.shape == t.shape == y.shape
    return pi * t / p * (V * y - C) + (1 - pi) * (1 - t) / (1 - p) * V * y


def top_k_policy(score: np.ndarray, k: float) -> np.ndarray:
    n_treat = int(round(k * len(score)))
    pol = np.zeros(len(score), dtype=np.float32)
    pol[np.argsort(-score, kind="stable")[:n_treat]] = 1.0
    return pol


def boot_ci(terms: np.ndarray, names: list[str], n_boot: int, seed: int = 0) -> dict:
    """terms: (n, m) per-row values. Returns mean and 95% CI per column, and the stored resample means (n_boot, m)."""
    n, m = terms.shape
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot, m), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = terms[idx].mean(axis=0, dtype=np.float64)
    point = terms.mean(axis=0, dtype=np.float64)
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return {"names": names, "point": point, "lo": lo, "hi": hi, "boot": means}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = DataLoader("data/criteo_uplift.parquet").load()
    u = np.random.default_rng(42).random(df.height)
    df = df.with_columns(pl.Series("u", u))
    fit, val, test = (df.filter(pl.col("u") < 0.6), df.filter((pl.col("u") >= 0.6) & (pl.col("u") < 0.8)),
                      df.filter(pl.col("u") >= 0.8))
    log.info("rows fit/val/test = %d / %d / %d", fit.height, val.height, test.height)
    assert fit.height + val.height + test.height == df.height

    # --- fit (fit rows only) ---
    x_l = XLearner(features=FEATURES, n_estimators=100)
    x_l.fit(fit, "treatment", "conversion")
    t_l = TLearner(features=FEATURES, n_estimators=100)
    t_l.fit(fit, "treatment", "conversion")
    resp = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_iterations": 100, "verbosity": -1,
                      "num_threads": -1},
                     lgb.Dataset(fit.select(FEATURES).to_numpy(), label=fit["conversion"].to_numpy()))
    log.info("fitted in %.0fs", time.time() - t0)

    def scores(d: pl.DataFrame) -> dict[str, np.ndarray]:
        return {"x_learner": x_l.predict(d), "t_learner": t_l.predict(d),
                "response": resp.predict(d.select(FEATURES).to_numpy())}

    s_test, s_val = scores(test), scores(val)
    for s in s_test.values():
        assert s.shape == (test.height,)

    t = test["treatment"].to_numpy().astype(np.float32)
    y = test["conversion"].to_numpy().astype(np.float32)
    p = float(t.mean())
    tv = val["treatment"].to_numpy().astype(np.float32)
    yv = val["conversion"].to_numpy().astype(np.float32)
    pv = float(tv.mean())

    # --- budgets table ---
    cols, names = [], []
    for k in BUDGETS:
        for m, s in s_test.items():
            cols.append(ipw_terms(top_k_policy(s, k), t, y, p)); names.append(f"{m}@{k}")
        cols.append(ipw_terms(np.full(len(t), k, dtype=np.float32), t, y, p)); names.append(f"random@{k}")
    cols.append(ipw_terms(np.ones(len(t), dtype=np.float32), t, y, p)); names.append("treat_all")
    cols.append(ipw_terms(np.zeros(len(t), dtype=np.float32), t, y, p)); names.append("treat_none")
    # threshold policies
    for tau in TAUS:
        cols.append(ipw_terms((s_test["x_learner"] > tau).astype(np.float32), t, y, p)); names.append(f"tau@{tau}")
    terms = np.stack(cols, axis=1).astype(np.float32)
    log.info("bootstrapping %d columns", terms.shape[1])
    bc = boot_ci(terms, names, N_BOOT)
    ix = {n: i for i, n in enumerate(names)}

    def row(n: str) -> dict:
        i = ix[n]
        return {"mean": float(bc["point"][i]), "lo": float(bc["lo"][i]), "hi": float(bc["hi"][i])}

    def diff(a: str, b: str) -> dict:
        d = bc["boot"][:, ix[a]] - bc["boot"][:, ix[b]]
        return {"mean": float(bc["point"][ix[a]] - bc["point"][ix[b]]), "lo": float(np.percentile(d, 2.5)),
                "hi": float(np.percentile(d, 97.5))}

    budgets = {}
    for k in BUDGETS:
        budgets[str(k)] = {
            "profit": {m: row(f"{m}@{k}") for m in [*s_test, "random"]},
            "x_learner_minus": {m: diff(f"x_learner@{k}", f"{m}@{k}") for m in ["t_learner", "response", "random"]},
        }

    # --- thresholds: test curve, and tau chosen on validation ---
    val_profit = {}
    for tau in TAUS:
        pol = (s_val["x_learner"] > tau).astype(np.float32)
        val_profit[tau] = float(ipw_terms(pol, tv, yv, pv).mean())
    best_tau = max(val_profit, key=lambda k: val_profit[k])
    thr = {str(tau): {**row(f"tau@{tau}"), "share_treated": float((s_test["x_learner"] > tau).mean()),
                      "val_profit": val_profit[tau]} for tau in TAUS}
    thr_summary = {
        "val_best_tau": best_tau,
        "profit_at_val_best": row(f"tau@{best_tau}"),
        "profit_at_0.02": row("tau@0.02"),
        "diff_0.02_minus_valbest": diff("tau@0.02", f"tau@{best_tau}"),
        "profit_treat_all": row("treat_all"), "profit_treat_none": row("treat_none"),
    }

    # --- AUUC, paired bootstrap ---
    ev = UpliftEvaluator(test)
    all_scores = {**s_test, "random": np.random.default_rng(1).random(len(t))}
    auuc_pt = {}
    for m, s in all_scores.items():
        _, yc, a = ev._calculate_qini_curve(t, y, s)
        auuc_pt[m] = {"auuc": float(a), "over_random_line": float(a - 0.5 * yc[-1])}
    rng = np.random.default_rng(0)
    boots = {m: [] for m in all_scores}
    for _ in range(N_BOOT_AUUC):
        idx = rng.integers(0, len(t), len(t))
        for m, s in all_scores.items():
            _, yc, a = ev._calculate_qini_curve(t[idx], y[idx], s[idx])
            boots[m].append(a - 0.5 * yc[-1])
    auuc = {}
    for m in all_scores:
        b = np.array(boots[m])
        auuc[m] = {**auuc_pt[m], "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}
        if m != "x_learner":
            d = np.array(boots["x_learner"]) - b
            auuc[m]["x_learner_minus_this"] = {"mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
                                               "hi": float(np.percentile(d, 97.5))}

    res = {"protocol": "docs/phase4_protocol.md", "rows": {"fit": fit.height, "val": val.height, "test": test.height},
           "p_treated_test": p, "value": V, "cost": C, "n_boot": N_BOOT, "n_boot_auuc": N_BOOT_AUUC,
           "budgets": budgets, "threshold_sweep": thr, "threshold_summary": thr_summary, "auuc": auuc,
           "profit_treat_all": row("treat_all"), "profit_treat_none": row("treat_none"),
           "seconds": round(time.time() - t0)}
    (OUT / "uplift_baselines.json").write_text(json.dumps(res, indent=2))
    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
