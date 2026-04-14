"""Statistical test helpers for hypothesis validation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    return float(np.nan_to_num(t, nan=0.0)), float(np.nan_to_num(p, nan=1.0))


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


def paired_wordy_mirror_ttests(
    points: Sequence[Dict[str, Any]],
    *,
    y_keys: Sequence[str],
    level_key: str = "level",
    group_key: str = "image_id",
    level_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Paired tests comparing terse levels L1-L4 against wordy mirrors L5-L8.

    The reported stabilization effect is ``base - wordy``. Positive values mean
    the wordy prompt reduced the outcome relative to its terse semantic mirror.
    """

    pairs = tuple(
        level_pairs
        or (
            ("L1_COARSE", "L5_WORDY_SIMPLETON"),
            ("L2_MEDIUM", "L6_WORDY_MEDIUM"),
            ("L3_FINE", "L7_WORDY_FINE"),
            ("L4_VERY_FINE", "L8_WORDY_VERY_FINE"),
        )
    )
    grouped_values: Dict[Tuple[str, str, str], List[float]] = {}
    for point in points:
        group = point.get(group_key)
        level = point.get(level_key)
        if group is None or level is None:
            continue
        for y_key in y_keys:
            value = point.get(y_key)
            if value is None:
                continue
            value_float = float(value)
            if not np.isfinite(value_float):
                continue
            grouped_values.setdefault((str(group), str(level), y_key), []).append(value_float)

    def _mean_for(group: str, level: str, y_key: str) -> Optional[float]:
        values = grouped_values.get((group, level, y_key))
        if not values:
            return None
        return float(np.mean(values))

    groups = sorted({key[0] for key in grouped_values})
    results: Dict[str, Any] = {
        "definition": "stabilization_effect = base_level_value - wordy_control_value; positive means the wordy prompt reduced the measured drop or drift.",
        "group_key": group_key,
        "level_key": level_key,
        "level_pairs": [{"base": base, "wordy": wordy} for base, wordy in pairs],
        "outcomes": {},
    }

    for y_key in y_keys:
        pair_results: Dict[str, Any] = {}
        all_base: List[float] = []
        all_wordy: List[float] = []
        all_deltas: List[float] = []

        for base_level, wordy_level in pairs:
            base_values: List[float] = []
            wordy_values: List[float] = []
            for group in groups:
                base_value = _mean_for(group, base_level, y_key)
                wordy_value = _mean_for(group, wordy_level, y_key)
                if base_value is None or wordy_value is None:
                    continue
                base_values.append(base_value)
                wordy_values.append(wordy_value)
            base_arr = np.asarray(base_values, dtype=float)
            wordy_arr = np.asarray(wordy_values, dtype=float)
            deltas = base_arr - wordy_arr
            n = int(deltas.size)
            if n >= 2:
                t_stat, p_value = paired_ttest(base_arr, wordy_arr)
                delta_mean = float(np.mean(deltas))
                delta_std = float(np.std(deltas, ddof=1))
                sem = float(delta_std / np.sqrt(n)) if delta_std > 0 else 0.0
                crit = float(stats.t.ppf(0.975, n - 1))
                ci_lower = delta_mean - crit * sem
                ci_upper = delta_mean + crit * sem
                cohen_dz = float(delta_mean / delta_std) if delta_std > 0 else 0.0
            elif n == 1:
                t_stat, p_value = 0.0, 1.0
                delta_mean = float(deltas[0])
                ci_lower = ci_upper = delta_mean
                cohen_dz = 0.0
            else:
                t_stat, p_value = 0.0, 1.0
                delta_mean = ci_lower = ci_upper = cohen_dz = 0.0

            pair_key = f"{base_level}__{wordy_level}"
            pair_results[pair_key] = {
                "base_level": base_level,
                "wordy_level": wordy_level,
                "n_pairs": n,
                "mean_base": float(np.mean(base_arr)) if n else 0.0,
                "mean_wordy": float(np.mean(wordy_arr)) if n else 0.0,
                "mean_stabilization_effect": float(delta_mean),
                "ci_95_lower": float(ci_lower),
                "ci_95_upper": float(ci_upper),
                "paired_t_statistic": float(t_stat),
                "paired_p_value": float(p_value),
                "cohens_dz": float(cohen_dz),
            }
            all_base.extend(base_values)
            all_wordy.extend(wordy_values)
            all_deltas.extend(deltas.tolist())

        base_all = np.asarray(all_base, dtype=float)
        wordy_all = np.asarray(all_wordy, dtype=float)
        deltas_all = np.asarray(all_deltas, dtype=float)
        n_all = int(deltas_all.size)
        if n_all >= 2:
            t_stat, p_value = paired_ttest(base_all, wordy_all)
            delta_mean = float(np.mean(deltas_all))
            delta_std = float(np.std(deltas_all, ddof=1))
            sem = float(delta_std / np.sqrt(n_all)) if delta_std > 0 else 0.0
            crit = float(stats.t.ppf(0.975, n_all - 1))
            ci_lower = delta_mean - crit * sem
            ci_upper = delta_mean + crit * sem
            cohen_dz = float(delta_mean / delta_std) if delta_std > 0 else 0.0
        elif n_all == 1:
            t_stat, p_value = 0.0, 1.0
            delta_mean = float(deltas_all[0])
            ci_lower = ci_upper = delta_mean
            cohen_dz = 0.0
        else:
            t_stat, p_value = 0.0, 1.0
            delta_mean = ci_lower = ci_upper = cohen_dz = 0.0

        results["outcomes"][y_key] = {
            "pairs": pair_results,
            "overall": {
                "n_pairs": n_all,
                "mean_base": float(np.mean(base_all)) if n_all else 0.0,
                "mean_wordy": float(np.mean(wordy_all)) if n_all else 0.0,
                "mean_stabilization_effect": float(delta_mean),
                "ci_95_lower": float(ci_lower),
                "ci_95_upper": float(ci_upper),
                "paired_t_statistic": float(t_stat),
                "paired_p_value": float(p_value),
                "cohens_dz": float(cohen_dz),
            },
        }

    return results
