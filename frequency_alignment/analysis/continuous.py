"""Helpers for continuous-score trend analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
from scipy import stats

from .statistics import pearson_correlation, spearman_correlation

_PRIMARY_LEVEL_NAMES = {"L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE"}


def attach_linear_residual(
    points: Sequence[Dict[str, Any]],
    *,
    target_key: str,
    control_key: str,
    residual_key: str,
    fit_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    min_slope: Optional[float] = None,
) -> Dict[str, Any]:
    """Attach residuals from target ~ control to each point in-place.

    Returns summary metadata for the fitted residualization model.
    """

    valid_indices: List[int] = []
    fit_indices: List[int] = []
    xs: List[float] = []
    ys: List[float] = []
    fit_xs: List[float] = []
    fit_ys: List[float] = []
    for idx, point in enumerate(points):
        x_val = point.get(control_key)
        y_val = point.get(target_key)
        if x_val is None or y_val is None:
            continue
        x_float = float(x_val)
        y_float = float(y_val)
        if not np.isfinite(x_float) or not np.isfinite(y_float):
            continue
        valid_indices.append(idx)
        xs.append(x_float)
        ys.append(y_float)
        if fit_filter is None or bool(fit_filter(point)):
            fit_indices.append(idx)
            fit_xs.append(x_float)
            fit_ys.append(y_float)

    for point in points:
        point[residual_key] = None

    if not valid_indices:
        return {
            "target_key": target_key,
            "control_key": control_key,
            "residual_key": residual_key,
            "n": 0,
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "fit_n": 0,
            "applied_n": 0,
        }

    if not fit_indices:
        fit_indices = list(valid_indices)
        fit_xs = list(xs)
        fit_ys = list(ys)

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    fit_x_arr = np.asarray(fit_xs, dtype=np.float64)
    fit_y_arr = np.asarray(fit_ys, dtype=np.float64)
    fit_design = np.column_stack([np.ones(fit_x_arr.size, dtype=np.float64), fit_x_arr])
    beta, _, _, _ = np.linalg.lstsq(fit_design, fit_y_arr, rcond=None)
    if min_slope is not None and float(beta[1]) < float(min_slope):
        beta[1] = float(min_slope)
        beta[0] = float(np.mean(fit_y_arr) - beta[1] * np.mean(fit_x_arr))

    fitted_fit = fit_design @ beta
    fit_residuals = fit_y_arr - fitted_fit
    fit_y_mean = float(np.mean(fit_y_arr))
    tss = float(np.sum((fit_y_arr - fit_y_mean) ** 2))
    rss = float(np.sum(fit_residuals ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else 0.0

    design = np.column_stack([np.ones(x_arr.size, dtype=np.float64), x_arr])
    residuals = y_arr - (design @ beta)
    for idx, residual in zip(valid_indices, residuals):
        points[idx][residual_key] = float(residual)

    return {
        "target_key": target_key,
        "control_key": control_key,
        "residual_key": residual_key,
        "n": int(len(valid_indices)),
        "fit_n": int(len(fit_indices)),
        "applied_n": int(len(valid_indices)),
        "slope": float(beta[1]),
        "intercept": float(beta[0]),
        "r_squared": float(r_squared),
    }


def attach_complexity_residual(
    points: Sequence[Dict[str, Any]],
    *,
    complexity_key: str = "complexity_score",
    prompt_key: str = "prompt_complexity_score",
    residual_key: str = "complexity_score_residual",
    fit_primary_levels: bool = True,
) -> Dict[str, Any]:
    """Residualize semantic complexity against prompt load in-place.

    When level labels are available, the prompt-load correction is fit on the
    primary L1-L4 ladder and then applied to wordy mirrors. The slope is kept
    non-negative so extra filler cannot make a same-semantics wordy control
    look more semantically complex than its terse base.
    """

    use_primary_fit = bool(fit_primary_levels) and any(
        str(point.get("level")) in _PRIMARY_LEVEL_NAMES for point in points
    )
    fit_filter = (
        (lambda point: str(point.get("level")) in _PRIMARY_LEVEL_NAMES)
        if use_primary_fit
        else None
    )

    summary = attach_linear_residual(
        points,
        target_key=complexity_key,
        control_key=prompt_key,
        residual_key=residual_key,
        fit_filter=fit_filter,
        min_slope=0.0,
    )
    summary["fit_scope"] = "primary_levels_L1_L4" if use_primary_fit else "all_valid_points"
    summary["slope_constraint"] = "non_negative"
    return summary


def summarize_linear_trend(
    points: Sequence[Dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> Dict[str, Any]:
    """Summarize a linear / monotonic relationship between two numeric keys."""

    xs: List[float] = []
    ys: List[float] = []
    for point in points:
        x_val = point.get(x_key)
        y_val = point.get(y_key)
        if x_val is None or y_val is None:
            continue
        x_float = float(x_val)
        y_float = float(y_val)
        if not np.isfinite(x_float) or not np.isfinite(y_float):
            continue
        xs.append(x_float)
        ys.append(y_float)

    if len(xs) < 3 or len(set(xs)) < 2:
        return {
            "n": len(xs),
            "x_key": x_key,
            "y_key": y_key,
            "pearson_r": 0.0,
            "pearson_p_value": 1.0,
            "spearman_rho": 0.0,
            "spearman_p_value": 1.0,
            "slope": 0.0,
            "intercept": float(np.mean(ys)) if ys else 0.0,
            "x_min": float(min(xs)) if xs else None,
            "x_max": float(max(xs)) if xs else None,
            "y_min": float(min(ys)) if ys else None,
            "y_max": float(max(ys)) if ys else None,
        }

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    pearson_r, pearson_p = pearson_correlation(x_arr, y_arr)
    spearman_rho, spearman_p = spearman_correlation(x_arr, y_arr)
    return {
        "n": len(xs),
        "x_key": x_key,
        "y_key": y_key,
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
        "slope": float(slope),
        "intercept": float(intercept),
        "x_min": float(np.min(x_arr)),
        "x_max": float(np.max(x_arr)),
        "y_min": float(np.min(y_arr)),
        "y_max": float(np.max(y_arr)),
    }


def summarize_by_score(
    points: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    value_key: str,
) -> Dict[str, Dict[str, float]]:
    """Aggregate a metric by exact complexity score."""

    grouped: Dict[float, List[float]] = defaultdict(list)
    for point in points:
        score = point.get(score_key)
        value = point.get(value_key)
        if score is None or value is None:
            continue
        score_float = float(score)
        value_float = float(value)
        if not np.isfinite(score_float) or not np.isfinite(value_float):
            continue
        grouped[score_float].append(value_float)

    summary: Dict[str, Dict[str, float]] = {}
    for score in sorted(grouped):
        values = np.asarray(grouped[score], dtype=np.float64)
        summary[str(score)] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": int(values.size),
        }
    return summary


def _collect_rows(
    points: Sequence[Dict[str, Any]],
    *,
    x_keys: Sequence[str],
    y_key: str,
    group_key: str | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for point in points:
        y_val = point.get(y_key)
        if y_val is None:
            continue
        y_float = float(y_val)
        if not np.isfinite(y_float):
            continue
        row: Dict[str, Any] = {y_key: y_float}
        valid = True
        for x_key in x_keys:
            x_val = point.get(x_key)
            if x_val is None:
                valid = False
                break
            x_float = float(x_val)
            if not np.isfinite(x_float):
                valid = False
                break
            row[x_key] = x_float
        if not valid:
            continue
        if group_key is not None:
            group_val = point.get(group_key)
            if group_val is None:
                continue
            row[group_key] = str(group_val)
        rows.append(row)
    return rows


def summarize_fixed_effects_trend(
    points: Sequence[Dict[str, Any]],
    *,
    group_key: str,
    x_key: str,
    y_key: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Within-group regression: fit y~x per group, then summarize slopes."""

    rows = _collect_rows(points, x_keys=[x_key], y_key=y_key, group_key=group_key)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)

    slopes: List[float] = []
    intercepts: List[float] = []
    within_r: List[float] = []
    group_sizes: List[int] = []

    for group_rows in grouped.values():
        xs = np.asarray([row[x_key] for row in group_rows], dtype=np.float64)
        ys = np.asarray([row[y_key] for row in group_rows], dtype=np.float64)
        if xs.size < 2 or np.unique(xs).size < 2:
            continue
        slope, intercept = np.polyfit(xs, ys, 1)
        slopes.append(float(slope))
        intercepts.append(float(intercept))
        group_sizes.append(int(xs.size))
        if xs.size >= 3 and np.unique(ys).size > 1:
            r, _ = pearson_correlation(xs, ys)
            within_r.append(float(r))

    if not slopes:
        return {
            "group_key": group_key,
            "x_key": x_key,
            "y_key": y_key,
            "n_points": len(rows),
            "n_groups_total": len(grouped),
            "n_groups_used": 0,
            "mean_points_per_group": 0.0,
            "mean_slope": 0.0,
            "std_slope": 0.0,
            "mean_intercept": 0.0,
            "mean_within_group_pearson_r": 0.0,
            "slope_t_statistic": 0.0,
            "slope_p_value": 1.0,
            "mean_slope_ci_95_lower": 0.0,
            "mean_slope_ci_95_upper": 0.0,
        }

    slope_arr = np.asarray(slopes, dtype=np.float64)
    intercept_arr = np.asarray(intercepts, dtype=np.float64)
    if slope_arr.size >= 2:
        t_stat, p_value = stats.ttest_1samp(slope_arr, 0.0)
        t_stat = float(np.nan_to_num(t_stat, nan=0.0))
        p_value = float(np.nan_to_num(p_value, nan=1.0))
    else:
        t_stat, p_value = 0.0, 1.0

    rng = np.random.default_rng(seed)
    if slope_arr.size >= 2:
        boot_means = []
        for _ in range(n_boot):
            sample = rng.choice(slope_arr, size=slope_arr.size, replace=True)
            boot_means.append(float(np.mean(sample)))
        lo = float(np.percentile(boot_means, 2.5))
        hi = float(np.percentile(boot_means, 97.5))
    else:
        lo = hi = float(slope_arr[0])

    return {
        "group_key": group_key,
        "x_key": x_key,
        "y_key": y_key,
        "n_points": len(rows),
        "n_groups_total": len(grouped),
        "n_groups_used": int(slope_arr.size),
        "mean_points_per_group": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "mean_slope": float(np.mean(slope_arr)),
        "std_slope": float(np.std(slope_arr)),
        "mean_intercept": float(np.mean(intercept_arr)),
        "mean_within_group_pearson_r": float(np.mean(within_r)) if within_r else 0.0,
        "slope_t_statistic": t_stat,
        "slope_p_value": p_value,
        "mean_slope_ci_95_lower": lo,
        "mean_slope_ci_95_upper": hi,
    }


def summarize_multivariate_regression(
    points: Sequence[Dict[str, Any]],
    *,
    y_key: str,
    x_keys: Sequence[str],
    group_key: str | None = None,
    demean_by_group: bool = False,
    level_filter: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """OLS summary for pooled or within-group demeaned regression."""

    filtered_points: Sequence[Dict[str, Any]]
    if level_filter is None:
        filtered_points = points
    else:
        filtered_points = [point for point in points if str(point.get("level")) in level_filter]

    rows = _collect_rows(
        filtered_points,
        x_keys=x_keys,
        y_key=y_key,
        group_key=group_key if demean_by_group else None,
    )
    if demean_by_group:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row[group_key]].append(row)
        transformed_rows: List[Dict[str, Any]] = []
        for _, group_rows in grouped.items():
            if len(group_rows) < 2:
                continue
            y_mean = float(np.mean([row[y_key] for row in group_rows]))
            x_means = {
                x_key: float(np.mean([row[x_key] for row in group_rows]))
                for x_key in x_keys
            }
            for row in group_rows:
                transformed = {y_key: row[y_key] - y_mean}
                for x_key in x_keys:
                    transformed[x_key] = row[x_key] - x_means[x_key]
                transformed_rows.append(transformed)
        rows = transformed_rows

    if len(rows) < max(4, len(x_keys) + 1):
        return {
            "mode": "within_image_fixed_effects" if demean_by_group else "pooled_ols",
            "y_key": y_key,
            "x_keys": list(x_keys),
            "level_filter": sorted(level_filter) if level_filter is not None else None,
            "n": len(rows),
            "r_squared": 0.0,
            "adjusted_r_squared": 0.0,
            "predictors": {},
        }

    y_arr = np.asarray([row[y_key] for row in rows], dtype=np.float64)
    x_arr = np.asarray([[row[x_key] for x_key in x_keys] for row in rows], dtype=np.float64)
    include_intercept = not demean_by_group
    if include_intercept:
        design = np.column_stack([np.ones(len(rows), dtype=np.float64), x_arr])
        coef_names = ["intercept"] + list(x_keys)
    else:
        design = x_arr
        coef_names = list(x_keys)

    rank = int(np.linalg.matrix_rank(design))
    beta, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    fitted = design @ beta
    residuals = y_arr - fitted
    dof = max(0, len(rows) - rank)
    rss = float(np.sum(residuals ** 2))
    y_center = y_arr - np.mean(y_arr) if include_intercept else y_arr
    tss = float(np.sum(y_center ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else 0.0
    adjusted_r_squared = (
        1.0 - (1.0 - r_squared) * (len(rows) - 1) / dof
        if include_intercept and dof > 0
        else r_squared
    )

    if dof > 0:
        sigma2 = rss / dof
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
        t_stats = np.divide(beta, stderr, out=np.zeros_like(beta), where=stderr > 0)
        p_values = 2.0 * stats.t.sf(np.abs(t_stats), dof)
    else:
        stderr = np.zeros_like(beta)
        t_stats = np.zeros_like(beta)
        p_values = np.ones_like(beta)

    y_std = float(np.std(y_arr))
    predictors: Dict[str, Any] = {}
    start_idx = 1 if include_intercept else 0
    for idx, x_key in enumerate(x_keys, start=start_idx):
        x_std = float(np.std(x_arr[:, idx - start_idx]))
        standardized_beta = float(beta[idx] * x_std / y_std) if y_std > 0 and x_std > 0 else 0.0
        predictors[x_key] = {
            "beta": float(beta[idx]),
            "standardized_beta": standardized_beta,
            "stderr": float(stderr[idx]),
            "t_statistic": float(t_stats[idx]),
            "p_value": float(p_values[idx]),
        }

    result: Dict[str, Any] = {
        "mode": "within_image_fixed_effects" if demean_by_group else "pooled_ols",
        "y_key": y_key,
        "x_keys": list(x_keys),
        "level_filter": sorted(level_filter) if level_filter is not None else None,
        "n": int(len(rows)),
        "rank": rank,
        "degrees_of_freedom": int(dof),
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "predictors": predictors,
    }
    if include_intercept:
        result["intercept"] = {
            "beta": float(beta[0]),
            "stderr": float(stderr[0]),
            "t_statistic": float(t_stats[0]),
            "p_value": float(p_values[0]),
        }
    return result
