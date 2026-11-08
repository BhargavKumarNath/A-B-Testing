"""Bootstrap confidence intervals for per-user means.

The profit numbers this pipeline reports (bandit replay, treat-everyone,
treat-nobody) are means of a variable that takes only a handful of distinct
values (e.g. 9.90, -0.10, 10.00, 0.00 dollars). For such a variable a
nonparametric bootstrap, which resamples n rows with replacement, is exactly
equivalent to drawing the count of each distinct value from a multinomial with
the observed proportions. Drawing the counts directly gives the identical
bootstrap distribution in O(k) per replicate instead of O(n), which is what
makes 10,000 replicates over millions of rows cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class MeanCI:
    mean: float
    lower: float
    upper: float
    n: int
    n_boot: int
    level: float
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 42,
) -> MeanCI:
    """Percentile bootstrap CI for the mean of ``values``.

    Uses the multinomial form of the bootstrap described in the module
    docstring, so the result is the same distribution as resampling rows.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    if n == 0:
        raise ValueError("values is empty")
    uniq, counts = np.unique(values, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, counts / n, size=n_boot)  # (n_boot, k)
    boot_means = draws @ uniq / n
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return MeanCI(
        mean=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
        n=int(n),
        n_boot=int(n_boot),
        level=float(level),
        seed=int(seed),
    )
