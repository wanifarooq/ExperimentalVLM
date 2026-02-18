"""Statistical test helpers for hypothesis validation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import stats


def spearman_correlation(x, y) -> Tuple[float, float]:
    """Spearman rank correlation coefficient and p-value."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return 0.0, 1.0
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def pearson_correlation(x, y) -> Tuple[float, float]:
    """Pearson linear correlation coefficient and p-value."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return 0.0, 1.0
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def one_way_anova(*groups) -> Tuple[float, float]:
    """One-way ANOVA F-statistic and p-value.

    Args:
        *groups: Two or more arrays of observations.

    Returns:
        ``(F_statistic, p_value)``.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(arrays) < 2:
        return 0.0, 1.0
    f_stat, p = stats.f_oneway(*arrays)
    return float(f_stat), float(p)


def cohens_d(group1, group2) -> float:
    """Cohen's d effect size (pooled standard deviation)."""
    g1, g2 = np.asarray(group1, dtype=float), np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled_std)


def paired_ttest(x, y) -> Tuple[float, float]:
    """Paired t-test on matched samples."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0, 1.0
    t, p = stats.ttest_rel(x, y)
    return float(t), float(p)


def monotonicity_test(values: List[float]) -> Tuple[bool, float]:
    """Test if values are monotonically increasing.

    Returns:
        ``(is_monotonic, kendall_tau)`` where Kendall's tau measures
        the ordinal association with index position.
    """
    vals = np.asarray(values, dtype=float)
    if len(vals) < 3:
        return True, 1.0
    indices = np.arange(len(vals))
    tau, _ = stats.kendalltau(indices, vals)
    is_mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    return is_mono, float(tau)


def bootstrap_ci(
    data,
    statistic=np.mean,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval.

    Returns:
        ``(point_estimate, ci_lower, ci_upper)``.
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.RandomState(seed)
    point = float(statistic(data))
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(float(statistic(sample)))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return point, lo, hi
