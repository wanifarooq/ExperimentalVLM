"""Helpers for continuous-score trend analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .statistics import pearson_correlation, spearman_correlation


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
