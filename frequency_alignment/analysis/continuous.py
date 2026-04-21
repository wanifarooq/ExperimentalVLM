"""Helpers for continuous-score trend analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    cluster_key: str | None = None,
    extra_keys: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    extras = tuple(extra_keys or ())
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
        if cluster_key is not None and cluster_key != group_key:
            cluster_val = point.get(cluster_key)
            if cluster_val is None:
                continue
            row[cluster_key] = str(cluster_val)
        for extra in extras:
            if extra in (y_key, group_key, cluster_key) or extra in x_keys:
                continue
            row[extra] = point.get(extra)
        rows.append(row)
    return rows


def _cluster_robust_covariance(
    design: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: Sequence[str],
) -> Tuple[np.ndarray, int]:
    """CR1 cluster-robust sandwich covariance estimator.

    V = c * (X'X)^{-1} * B * (X'X)^{-1}
    where
        B = sum over clusters g of (X_g' e_g)(X_g' e_g)'
        c = (G / (G-1)) * ((n-1) / (n-k))  -- Stata-style small-sample correction
    and the effective degrees of freedom for t-tests is G - 1.
    """
    n, k = design.shape
    cluster_index: Dict[str, List[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        cluster_index[cid].append(i)
    G = len(cluster_index)
    xtx_inv = np.linalg.pinv(design.T @ design)
    score_sum = np.zeros((k, k), dtype=np.float64)
    for idxs in cluster_index.values():
        idx_arr = np.asarray(idxs, dtype=np.int64)
        Xg = design[idx_arr]
        eg = residuals[idx_arr]
        ug = Xg.T @ eg
        score_sum += np.outer(ug, ug)
    if G > 1 and n > k:
        correction = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    else:
        correction = 1.0
    cov = correction * xtx_inv @ score_sum @ xtx_inv
    return cov, G


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
    cluster_key: str | None = None,
    level_filter: Optional[Set[str]] = None,
    dummy_key: str | None = None,
) -> Dict[str, Any]:
    """OLS summary with optional within-group demeaning, cluster-robust SEs,
    and categorical fixed-effects via ``dummy_key``.

    Parameters
    ----------
    cluster_key : optional
        When supplied, report CR1 cluster-robust standard errors with
        degrees of freedom ``G - 1`` (number of clusters minus one). This
        is the correct standard-error treatment when observations within
        a cluster share unobserved structure (e.g. multiple perturbation
        rows from the same image). Classical OLS SEs are still reported
        under ``classical_`` prefixes for transparency.
    dummy_key : optional
        Name of a categorical column whose distinct values are expanded
        into K-1 dummy columns appended to the design matrix. Used to
        absorb perturbation-type fixed effects in the long-format
        regression. These dummy coefficients are fit but not returned
        as ``predictors`` to keep output compact.
    """

    filtered_points: Sequence[Dict[str, Any]]
    if level_filter is None:
        filtered_points = points
    else:
        filtered_points = [point for point in points if str(point.get("level")) in level_filter]

    extras: List[str] = []
    if cluster_key is not None:
        extras.append(cluster_key)
    if dummy_key is not None:
        extras.append(dummy_key)

    rows = _collect_rows(
        filtered_points,
        x_keys=x_keys,
        y_key=y_key,
        group_key=group_key if demean_by_group else None,
        cluster_key=cluster_key,
        extra_keys=extras,
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
                if cluster_key is not None:
                    transformed[cluster_key] = row.get(cluster_key)
                if dummy_key is not None:
                    transformed[dummy_key] = row.get(dummy_key)
                transformed_rows.append(transformed)
        rows = transformed_rows

    if len(rows) < max(4, len(x_keys) + 1):
        return {
            "mode": "within_image_fixed_effects" if demean_by_group else "pooled_ols",
            "y_key": y_key,
            "x_keys": list(x_keys),
            "cluster_key": cluster_key,
            "dummy_key": dummy_key,
            "level_filter": sorted(level_filter) if level_filter is not None else None,
            "n": len(rows),
            "r_squared": 0.0,
            "adjusted_r_squared": 0.0,
            "predictors": {},
        }

    y_arr = np.asarray([row[y_key] for row in rows], dtype=np.float64)
    x_arr = np.asarray([[row[x_key] for x_key in x_keys] for row in rows], dtype=np.float64)

    include_intercept = not demean_by_group

    dummy_matrix: np.ndarray = np.zeros((len(rows), 0), dtype=np.float64)
    dummy_levels: List[str] = []
    if dummy_key is not None:
        raw_levels = [str(row.get(dummy_key)) for row in rows]
        uniq = sorted({lvl for lvl in raw_levels if lvl is not None and lvl != "None"})
        if len(uniq) >= 2:
            # drop the first level as the reference category
            dummy_levels = uniq[1:]
            dummy_matrix = np.zeros((len(rows), len(dummy_levels)), dtype=np.float64)
            idx_of = {lvl: i for i, lvl in enumerate(dummy_levels)}
            for r_idx, lvl in enumerate(raw_levels):
                col = idx_of.get(lvl)
                if col is not None:
                    dummy_matrix[r_idx, col] = 1.0

    if include_intercept:
        design_parts = [np.ones(len(rows), dtype=np.float64).reshape(-1, 1), x_arr]
        coef_names = ["intercept"] + list(x_keys)
    else:
        design_parts = [x_arr]
        coef_names = list(x_keys)
    if dummy_matrix.shape[1] > 0:
        design_parts.append(dummy_matrix)
        coef_names = coef_names + [f"{dummy_key}={lvl}" for lvl in dummy_levels]
    design = np.column_stack(design_parts) if len(design_parts) > 1 else design_parts[0]

    rank = int(np.linalg.matrix_rank(design))
    beta, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    fitted = design @ beta
    residuals = y_arr - fitted
    classical_dof = max(0, len(rows) - rank)
    rss = float(np.sum(residuals ** 2))
    y_center = y_arr - np.mean(y_arr) if include_intercept else y_arr
    tss = float(np.sum(y_center ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else 0.0
    adjusted_r_squared = (
        1.0 - (1.0 - r_squared) * (len(rows) - 1) / classical_dof
        if include_intercept and classical_dof > 0
        else r_squared
    )

    if classical_dof > 0:
        sigma2 = rss / classical_dof
        classical_cov = sigma2 * np.linalg.pinv(design.T @ design)
        classical_stderr = np.sqrt(np.maximum(np.diag(classical_cov), 0.0))
        classical_t = np.divide(beta, classical_stderr, out=np.zeros_like(beta), where=classical_stderr > 0)
        classical_p = 2.0 * stats.t.sf(np.abs(classical_t), classical_dof)
    else:
        classical_stderr = np.zeros_like(beta)
        classical_t = np.zeros_like(beta)
        classical_p = np.ones_like(beta)

    cluster_stderr = None
    cluster_t = None
    cluster_p = None
    n_clusters = None
    cluster_dof = None
    if cluster_key is not None:
        cluster_ids = [str(row.get(cluster_key)) for row in rows]
        if any(cid is None or cid == "None" for cid in cluster_ids):
            cluster_stderr = np.zeros_like(beta)
            cluster_t = np.zeros_like(beta)
            cluster_p = np.ones_like(beta)
            n_clusters = 0
            cluster_dof = 0
        else:
            cluster_cov, n_clusters = _cluster_robust_covariance(design, residuals, cluster_ids)
            cluster_stderr = np.sqrt(np.maximum(np.diag(cluster_cov), 0.0))
            cluster_dof = max(1, n_clusters - 1)
            cluster_t = np.divide(beta, cluster_stderr, out=np.zeros_like(beta), where=cluster_stderr > 0)
            cluster_p = 2.0 * stats.t.sf(np.abs(cluster_t), cluster_dof)

    # Report cluster-robust inference when available; else classical.
    if cluster_stderr is not None:
        stderr = cluster_stderr
        t_stats = cluster_t
        p_values = cluster_p
        inference_mode = "cluster_robust_cr1"
    else:
        stderr = classical_stderr
        t_stats = classical_t
        p_values = classical_p
        inference_mode = "classical_ols"

    y_std = float(np.std(y_arr))
    predictors: Dict[str, Any] = {}
    start_idx = 1 if include_intercept else 0
    for idx, x_key in enumerate(x_keys, start=start_idx):
        x_col = x_arr[:, idx - start_idx]
        x_std = float(np.std(x_col))
        standardized_beta = float(beta[idx] * x_std / y_std) if y_std > 0 and x_std > 0 else 0.0
        entry: Dict[str, Any] = {
            "beta": float(beta[idx]),
            "standardized_beta": standardized_beta,
            "stderr": float(stderr[idx]),
            "t_statistic": float(t_stats[idx]),
            "p_value": float(p_values[idx]),
            "classical_stderr": float(classical_stderr[idx]),
            "classical_t_statistic": float(classical_t[idx]),
            "classical_p_value": float(classical_p[idx]),
        }
        if cluster_stderr is not None:
            entry["cluster_robust_stderr"] = float(cluster_stderr[idx])
            entry["cluster_robust_t_statistic"] = float(cluster_t[idx])
            entry["cluster_robust_p_value"] = float(cluster_p[idx])
            if classical_stderr[idx] > 0:
                entry["design_effect"] = float((cluster_stderr[idx] / classical_stderr[idx]) ** 2)
        predictors[x_key] = entry

    result: Dict[str, Any] = {
        "mode": "within_image_fixed_effects" if demean_by_group else "pooled_ols",
        "inference_mode": inference_mode,
        "y_key": y_key,
        "x_keys": list(x_keys),
        "cluster_key": cluster_key,
        "dummy_key": dummy_key,
        "n_dummy_levels": len(dummy_levels),
        "level_filter": sorted(level_filter) if level_filter is not None else None,
        "n": int(len(rows)),
        "rank": rank,
        "degrees_of_freedom": int(cluster_dof if cluster_stderr is not None else classical_dof),
        "classical_degrees_of_freedom": int(classical_dof),
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "predictors": predictors,
    }
    if cluster_stderr is not None:
        result["n_clusters"] = int(n_clusters or 0)
    if include_intercept:
        result["intercept"] = {
            "beta": float(beta[0]),
            "stderr": float(stderr[0]),
            "t_statistic": float(t_stats[0]),
            "p_value": float(p_values[0]),
        }
    return result


def _pearson_ci_95(r_value: float, n: int) -> Tuple[float, float]:
    """Approximate 95% CI for Pearson r using Fisher z transform."""

    if n <= 3:
        return float(r_value), float(r_value)
    clipped = float(np.clip(r_value, -0.999999, 0.999999))
    z_value = float(np.arctanh(clipped))
    se = 1.0 / np.sqrt(n - 3)
    lo = float(np.tanh(z_value - 1.96 * se))
    hi = float(np.tanh(z_value + 1.96 * se))
    return lo, hi


def _predictor_target_rows(
    points: Sequence[Dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    xs: List[float] = []
    ys: List[float] = []
    rows: List[Dict[str, Any]] = []
    for point in points:
        x_val = point.get(x_key)
        y_val = point.get(y_key)
        if x_val is None or y_val is None:
            continue
        try:
            x_float = float(x_val)
            y_float = float(y_val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x_float) or not np.isfinite(y_float):
            continue
        xs.append(x_float)
        ys.append(y_float)
        rows.append(point)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), rows


def _level_mean_pairs(
    rows: Sequence[Dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    level_order: Sequence[str],
) -> Dict[str, Any]:
    per_level: Dict[str, Dict[str, List[float]]] = {
        level: {"x": [], "y": []}
        for level in level_order
    }
    for row in rows:
        level = str(row.get("level"))
        if level not in per_level:
            continue
        per_level[level]["x"].append(float(row[x_key]))
        per_level[level]["y"].append(float(row[y_key]))

    level_means: Dict[str, Dict[str, float]] = {}
    x_means: List[float] = []
    y_means: List[float] = []
    ranks: List[int] = []
    for idx, level in enumerate(level_order):
        xs = per_level[level]["x"]
        ys = per_level[level]["y"]
        if not xs or not ys:
            continue
        x_mean = float(np.mean(xs))
        y_mean = float(np.mean(ys))
        level_means[level] = {
            "x_mean": x_mean,
            "y_mean": y_mean,
            "n": int(min(len(xs), len(ys))),
        }
        x_means.append(x_mean)
        y_means.append(y_mean)
        ranks.append(idx + 1)

    if len(ranks) >= 3:
        tau_level_target, p_level_target = stats.kendalltau(ranks, y_means)
        tau_level_predictor, p_level_predictor = stats.kendalltau(ranks, x_means)
        tau_predictor_target, p_predictor_target = stats.kendalltau(x_means, y_means)
    else:
        tau_level_target = tau_level_predictor = tau_predictor_target = 0.0
        p_level_target = p_level_predictor = p_predictor_target = 1.0

    return {
        "level_order": list(level_order),
        "level_means": level_means,
        "kendall_tau_level_target": float(np.nan_to_num(tau_level_target, nan=0.0)),
        "kendall_tau_level_target_p_value": float(np.nan_to_num(p_level_target, nan=1.0)),
        "kendall_tau_level_predictor": float(np.nan_to_num(tau_level_predictor, nan=0.0)),
        "kendall_tau_level_predictor_p_value": float(np.nan_to_num(p_level_predictor, nan=1.0)),
        "kendall_tau_predictor_target_by_level": float(np.nan_to_num(tau_predictor_target, nan=0.0)),
        "kendall_tau_predictor_target_by_level_p_value": float(np.nan_to_num(p_predictor_target, nan=1.0)),
    }


def summarize_vif_diagnostic(
    points: Sequence[Dict[str, Any]],
    *,
    x_keys: Sequence[str] = ("question_complexity_score", "prompt_complexity_score"),
) -> Dict[str, Any]:
    """VIF diagnostic for predictor collinearity."""

    if not x_keys:
        return {
            "x_keys": [],
            "n": 0,
            "pairwise_pearson_r": 0.0,
            "collinearity_percent": 0.0,
            "vif": {},
        }

    rows = _collect_rows(points, x_keys=x_keys, y_key=x_keys[0])
    if len(x_keys) < 2 or len(rows) < 3:
        return {
            "x_keys": list(x_keys),
            "n": len(rows),
            "pairwise_pearson_r": 0.0,
            "collinearity_percent": 0.0,
            "vif": {key: 1.0 for key in x_keys},
        }

    matrix = np.asarray([[row[key] for key in x_keys] for row in rows], dtype=np.float64)
    vif: Dict[str, float] = {}
    for idx, key in enumerate(x_keys):
        target = matrix[:, idx]
        others = np.delete(matrix, idx, axis=1)
        design = np.column_stack([np.ones(others.shape[0], dtype=np.float64), others])
        beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        fitted = design @ beta
        residuals = target - fitted
        tss = float(np.sum((target - np.mean(target)) ** 2))
        rss = float(np.sum(residuals ** 2))
        r_squared = 1.0 - rss / tss if tss > 0 else 0.0
        denom = max(1e-12, 1.0 - r_squared)
        vif[key] = float(1.0 / denom)

    first = matrix[:, 0]
    second = matrix[:, 1]
    if np.std(first) > 0 and np.std(second) > 0:
        pair_r, pair_p = pearson_correlation(first, second)
    else:
        pair_r, pair_p = 0.0, 1.0
    return {
        "x_keys": list(x_keys),
        "n": int(matrix.shape[0]),
        "pairwise_pearson_r": float(pair_r),
        "pairwise_pearson_p_value": float(pair_p),
        "collinearity_percent": float(abs(pair_r) * 100.0),
        "vif": vif,
    }


def summarize_marginal_relationships(
    points: Sequence[Dict[str, Any]],
    *,
    y_key: str,
    x_keys: Sequence[str],
    level_filter: Optional[Set[str]] = None,
    level_order: Sequence[str] = ("L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE"),
) -> Dict[str, Any]:
    """Primary-ladder marginal correlations without multivariate betas."""

    filtered_points: Sequence[Dict[str, Any]]
    if level_filter is None:
        filtered_points = points
    else:
        filtered_points = [point for point in points if str(point.get("level")) in level_filter]

    marginal_predictors: Dict[str, Any] = {}
    for x_key in x_keys:
        xs, ys, rows = _predictor_target_rows(filtered_points, x_key=x_key, y_key=y_key)
        if xs.size >= 3 and np.unique(xs).size >= 2 and np.unique(ys).size >= 2:
            pearson_r, pearson_p = pearson_correlation(xs, ys)
            spearman_rho, spearman_p = spearman_correlation(xs, ys)
            ci_lower, ci_upper = _pearson_ci_95(float(pearson_r), int(xs.size))
        else:
            pearson_r = spearman_rho = 0.0
            pearson_p = spearman_p = 1.0
            ci_lower = ci_upper = 0.0
        level_payload = _level_mean_pairs(rows, x_key=x_key, y_key=y_key, level_order=level_order)
        marginal_predictors[x_key] = {
            "n": int(xs.size),
            "pearson_r": float(pearson_r),
            "pearson_p_value": float(pearson_p),
            "pearson_ci_95_lower": float(ci_lower),
            "pearson_ci_95_upper": float(ci_upper),
            "spearman_rho": float(spearman_rho),
            "spearman_p_value": float(spearman_p),
            **level_payload,
        }

    vif = summarize_vif_diagnostic(
        filtered_points,
        x_keys=[key for key in ("question_complexity_score", "prompt_complexity_score") if key in x_keys],
    )
    return {
        "mode": "primary_marginal_summary",
        "y_key": y_key,
        "x_keys": list(x_keys),
        "level_filter": sorted(level_filter) if level_filter is not None else None,
        "n": int(max((stats.get("n", 0) for stats in marginal_predictors.values()), default=0)),
        "marginal_predictors": marginal_predictors,
        "vif_diagnostic": vif,
        "caption": (
            "Primary L1-L4 marginal correlations; multivariate decomposition "
            "requires the wordy-mirror design available in pooled/wordy views."
        ),
    }


def summarize_horse_race_view(
    points: Sequence[Dict[str, Any]],
    *,
    y_key: str,
    x_keys: Sequence[str],
    view_name: str,
    level_filter: Optional[Set[str]] = None,
    group_key: str = "image_id",
    cluster_key: Optional[str] = "image_id",
) -> Dict[str, Any]:
    """Return primary marginal summary or full wordy/pooled horse race.

    The pooled and within-image regressions report cluster-robust SEs on
    ``cluster_key`` (default ``image_id``) alongside classical SEs, which
    is the correct standard-error treatment when each image contributes
    multiple rows (one per level, and possibly one per perturbation).
    """

    view_points = (
        list(points)
        if level_filter is None
        else [point for point in points if str(point.get("level")) in level_filter]
    )
    base_payload: Dict[str, Any] = {
        "level_filter": sorted(level_filter) if level_filter is not None else None,
        "n_points": len(view_points),
    }
    if view_name == "primary":
        base_payload["marginal"] = summarize_marginal_relationships(
            view_points,
            y_key=y_key,
            x_keys=x_keys,
            level_filter=None,
        )
        return base_payload

    base_payload["pooled"] = summarize_multivariate_regression(
        view_points,
        y_key=y_key,
        x_keys=x_keys,
        cluster_key=cluster_key,
        level_filter=None,
    )
    base_payload["within_image"] = summarize_multivariate_regression(
        view_points,
        y_key=y_key,
        x_keys=x_keys,
        group_key=group_key,
        demean_by_group=True,
        cluster_key=cluster_key,
        level_filter=None,
    )
    return base_payload


# ---------------------------------------------------------------------------
# Long-format and ΔF-stratified horse races
# ---------------------------------------------------------------------------


def classify_delta_f_family(
    delta_f: Sequence[float],
    *,
    zero_threshold: float = 1e-6,
    low_frac: float = 1.0 / 3.0,
    high_frac: float = 2.0 / 3.0,
) -> str:
    """Classify a radial ΔF spectrum into one of four families.

    - ``zero``      : ||ΔF|| is numerically negligible (shift-only or DC-only
                      perturbations — translation under a shift-equivariant encoder,
                      pure brightness at DC in log-space, etc.). Theory predicts
                      β_Csem ≈ 0 here because spectral overlap with any W_t is ~0.
    - ``low_freq``  : ΔF energy peaks in the lower third of the band axis.
    - ``high_freq`` : ΔF energy peaks in the upper third of the band axis.
    - ``broadband`` : ΔF energy is spread across the middle third (flat spectra
                      such as additive Gaussian noise).

    Parameters
    ----------
    zero_threshold : float
        Relative L1 magnitude below which a spectrum is treated as zero.
    """
    arr = np.asarray(delta_f, dtype=np.float64)
    if arr.size == 0:
        return "zero"
    total = float(np.sum(np.abs(arr)))
    if total <= zero_threshold:
        return "zero"
    num_bands = arr.size
    weights = np.abs(arr) / total
    centroid = float(np.sum(weights * np.arange(num_bands)))
    low_cut = low_frac * num_bands
    high_cut = high_frac * num_bands
    if centroid < low_cut:
        return "low_freq"
    if centroid > high_cut:
        return "high_freq"
    return "broadband"


def summarize_long_format_horse_race(
    per_sample_rows: Sequence[Dict[str, Any]],
    *,
    y_key: str,
    x_keys: Sequence[str],
    cluster_key: str = "image_id",
    perturbation_key: str = "perturbation_type",
    level_filter: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Long-format regression: one row per (image, level, perturbation).

    Absorbs per-perturbation baseline toxicity with K-1 fixed-effect dummies,
    so the recovered β_Csem / β_Cprompt are identified by *within-perturbation
    variation*. Standard errors are cluster-robust on ``image_id``.

    Returns
    -------
    dict with fields ``pooled`` (regression with perturbation FE) and
    ``pooled_no_fe`` (regression without FE, for comparison).
    """
    rows = per_sample_rows
    if level_filter is not None:
        rows = [row for row in rows if str(row.get("level")) in level_filter]
    pooled_fe = summarize_multivariate_regression(
        rows,
        y_key=y_key,
        x_keys=x_keys,
        cluster_key=cluster_key,
        dummy_key=perturbation_key,
    )
    pooled_no_fe = summarize_multivariate_regression(
        rows,
        y_key=y_key,
        x_keys=x_keys,
        cluster_key=cluster_key,
    )
    return {
        "mode": "long_format_with_perturbation_fe",
        "y_key": y_key,
        "x_keys": list(x_keys),
        "perturbation_key": perturbation_key,
        "cluster_key": cluster_key,
        "level_filter": sorted(level_filter) if level_filter is not None else None,
        "n_rows_total": len(rows),
        "pooled_with_perturbation_fe": pooled_fe,
        "pooled_no_perturbation_fe": pooled_no_fe,
    }


def summarize_per_perturbation_horse_race(
    per_sample_rows: Sequence[Dict[str, Any]],
    *,
    y_key: str,
    x_keys: Sequence[str],
    perturbation_key: str = "perturbation_type",
    cluster_key: str = "image_id",
    level_filter: Optional[Set[str]] = None,
    min_rows_per_group: int = 50,
) -> Dict[str, Any]:
    """Fit the horse race separately for each perturbation family.

    Produces β_Csem and β_Cprompt estimates per perturbation, plus a
    sign-stability summary that asks whether the sign of the pooled
    β survives in each individual perturbation. Standard errors are
    cluster-robust on ``image_id`` within each per-perturbation fit.
    """
    rows = per_sample_rows
    if level_filter is not None:
        rows = [row for row in rows if str(row.get("level")) in level_filter]

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pert = str(row.get(perturbation_key, "unknown"))
        groups[pert].append(row)

    per_pert: Dict[str, Dict[str, Any]] = {}
    beta_signs: Dict[str, List[int]] = {x: [] for x in x_keys}
    beta_values: Dict[str, List[float]] = {x: [] for x in x_keys}
    for pert_name, group_rows in sorted(groups.items()):
        if len(group_rows) < max(min_rows_per_group, len(x_keys) + 2):
            continue
        fit = summarize_multivariate_regression(
            group_rows,
            y_key=y_key,
            x_keys=x_keys,
            cluster_key=cluster_key,
        )
        n_rows = fit.get("n", 0)
        meta = {
            "n_rows": n_rows,
            "n_clusters": fit.get("n_clusters"),
            "r_squared": fit.get("r_squared"),
            "predictors": fit.get("predictors", {}),
        }
        per_pert[pert_name] = meta
        preds = fit.get("predictors", {})
        for x in x_keys:
            info = preds.get(x) or {}
            b = info.get("beta")
            if b is None or not np.isfinite(b):
                continue
            beta_signs[x].append(1 if b > 0 else (-1 if b < 0 else 0))
            beta_values[x].append(float(b))

    sign_stability: Dict[str, Dict[str, Any]] = {}
    for x in x_keys:
        signs = beta_signs[x]
        if not signs:
            sign_stability[x] = {"n_perturbations": 0, "fraction_positive": 0.0, "fraction_negative": 0.0}
            continue
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        total = len(signs)
        sign_stability[x] = {
            "n_perturbations": total,
            "fraction_positive": float(pos / total),
            "fraction_negative": float(neg / total),
            "mean_beta": float(np.mean(beta_values[x])),
            "median_beta": float(np.median(beta_values[x])),
            "std_beta": float(np.std(beta_values[x])),
        }

    return {
        "mode": "per_perturbation_horse_race",
        "y_key": y_key,
        "x_keys": list(x_keys),
        "perturbation_key": perturbation_key,
        "cluster_key": cluster_key,
        "level_filter": sorted(level_filter) if level_filter is not None else None,
        "n_perturbations_fit": len(per_pert),
        "per_perturbation": per_pert,
        "sign_stability": sign_stability,
    }


def summarize_by_delta_f_family_horse_race(
    per_sample_rows: Sequence[Dict[str, Any]],
    *,
    y_key: str,
    x_keys: Sequence[str],
    family_key: str = "delta_f_family",
    cluster_key: str = "image_id",
    level_filter: Optional[Set[str]] = None,
    min_rows_per_family: int = 30,
) -> Dict[str, Any]:
    """Stratify the horse race by ΔF-family and fit separately within each.

    Theory predicts that perturbations with near-zero ΔF (shift-only / DC-only)
    should produce β_Csem ≈ 0 because spectral overlap with any W_t is ~0.
    Perturbations with high-frequency ΔF should drive the dual-force signal
    most strongly for fine-grained tasks. This stratified view tests that
    falsifiable prediction directly.
    """
    rows = per_sample_rows
    if level_filter is not None:
        rows = [row for row in rows if str(row.get("level")) in level_filter]
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fam = str(row.get(family_key, "unknown"))
        by_family[fam].append(row)
    family_fits: Dict[str, Dict[str, Any]] = {}
    for fam, family_rows in sorted(by_family.items()):
        if len(family_rows) < max(min_rows_per_family, len(x_keys) + 2):
            family_fits[fam] = {
                "n_rows": len(family_rows),
                "skipped": True,
            }
            continue
        family_fits[fam] = summarize_multivariate_regression(
            family_rows,
            y_key=y_key,
            x_keys=x_keys,
            cluster_key=cluster_key,
        )
    return {
        "mode": "delta_f_family_stratified_horse_race",
        "y_key": y_key,
        "x_keys": list(x_keys),
        "family_key": family_key,
        "cluster_key": cluster_key,
        "level_filter": sorted(level_filter) if level_filter is not None else None,
        "family_row_counts": {k: len(v) for k, v in by_family.items()},
        "family_fits": family_fits,
    }
