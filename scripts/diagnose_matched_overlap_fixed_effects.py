#!/usr/bin/env python3
"""Matched fixed-effect diagnostics for Exp 5 overlap laws.

This is an experimental analysis script, not part of the production pipeline.
It tests the overlap law within matched image/perturbation/severity groups so
that image content and perturbation strength are held fixed while only the task
level/filter changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


LEVEL_ORDER = [
    "L1_COARSE",
    "L2_MEDIUM",
    "L3_FINE",
    "L4_VERY_FINE",
    "L5_WORDY_SIMPLETON",
    "L6_WORDY_MEDIUM",
    "L7_WORDY_FINE",
    "L8_WORDY_VERY_FINE",
]
GROUP_ORDER = ["overall", "early", "mid", "late", "last_2"]
FILTER_2D_DIRS = {
    "option_conditioned": "filters_2d",
    "question_only": "filters_2d_question_only",
}
DOMAIN_SPECS = {
    "image_space": {
        "delta_radial_raw": "delta_f",
        "delta_radial_relative": "delta_f_relative",
        "delta_2d_dir": "delta_f_2d",
        "delta_2d_file": "delta_f_2d_file",
        "delta_2d_raw": "delta_power_2d",
        "delta_2d_relative": "delta_power_2d_relative",
    },
    "vision_feature_space": {
        "delta_radial_raw": "delta_f_vision",
        "delta_radial_relative": "delta_f_vision_relative",
        "delta_2d_dir": "delta_f_vision_2d",
        "delta_2d_file": "delta_f_vision_2d_file",
        "delta_2d_raw": "delta_power_vision_2d",
        "delta_2d_relative": "delta_power_vision_2d_relative",
    },
}
SPACE_ORDER = ["radial_first_order", "two_d_first_order"]
CENTERING_ORDER = ["raw", "delta_centered", "filter_centered", "both_centered"]
EPS = 1e-12


def _parse_csv(value: Optional[str], default: Sequence[str]) -> List[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_name(value: Any) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned.strip("_") or "unknown"


def _target_value(level_data: Dict[str, Any], perturbation: Dict[str, Any], target_key: str) -> Optional[float]:
    if target_key == "loglik_volatility":
        drift = _as_float(perturbation.get("loglik_drift"))
        return abs(drift) if drift is not None else None
    if target_key == "loglik_erosion":
        drift = _as_float(perturbation.get("loglik_drift"))
        return max(drift, 0.0) if drift is not None else None
    if target_key == "loglik_recovery":
        drift = _as_float(perturbation.get("loglik_drift"))
        return min(drift, 0.0) if drift is not None else None
    if target_key == "loglik_drift":
        return _as_float(perturbation.get("loglik_drift"))
    if target_key == "accuracy_drop":
        value = _as_float(perturbation.get("accuracy_drop"))
        if value is not None:
            return value
        clean_correct = bool(level_data.get("clean", {}).get("correct", False))
        perturbed_correct = bool(perturbation.get("correct", False))
        return 1.0 if clean_correct and not perturbed_correct else 0.0
    if target_key in {"net_drop", "net_change"}:
        clean_correct = bool(level_data.get("clean", {}).get("correct", False))
        perturbed_correct = bool(perturbation.get("correct", False))
        ci = 1.0 if clean_correct and not perturbed_correct else 0.0
        ic = 1.0 if (not clean_correct) and perturbed_correct else 0.0
        return ci - ic
    return _as_float(perturbation.get(target_key))


def _rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    sorted_values = arr[order]
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3 or np.std(x_arr) <= EPS or np.std(y_arr) <= EPS:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3:
        return None
    return _pearson(_rankdata(x_arr), _rankdata(y_arr))


def _load_npz_array(path: Path, key: str, cache: Dict[Tuple[str, str], Optional[np.ndarray]]) -> Optional[np.ndarray]:
    cache_key = (str(path), key)
    if cache_key in cache:
        return cache[cache_key]
    if not path.exists():
        cache[cache_key] = None
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            value = np.asarray(data[key], dtype=np.float64)
    except Exception:
        value = None
    cache[cache_key] = value
    return value


def _resize_power_preserve_sum(power_2d: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(power_2d, dtype=np.float64)
    if arr.shape == target_shape:
        return arr
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros(target_shape, dtype=np.float64)
    source_sum = float(np.sum(arr))
    image = Image.fromarray(arr.astype(np.float32))
    resized = np.asarray(
        image.resize((int(target_shape[1]), int(target_shape[0])), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    resized_sum = float(np.sum(resized))
    if resized_sum > EPS:
        resized *= source_sum / resized_sum
    return resized


def _load_aligned_npz_array(
    path: Path,
    key: str,
    target_shape: Tuple[int, int],
    npz_cache: Dict[Tuple[str, str], Optional[np.ndarray]],
    aligned_cache: Dict[Tuple[str, str, Tuple[int, int]], Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    cache_key = (str(path), key, target_shape)
    if cache_key in aligned_cache:
        return aligned_cache[cache_key]
    arr = _load_npz_array(path, key, npz_cache)
    if arr is None:
        aligned_cache[cache_key] = None
        return None
    aligned = _resize_power_preserve_sum(arr, target_shape)
    aligned_cache[cache_key] = aligned
    return aligned


def _load_filter_bank(run_dir: Path, groups: Sequence[str]) -> Dict[Tuple[str, str, str], np.ndarray]:
    power_path = run_dir / "exp2" / "power_spectra.json"
    if not power_path.exists():
        raise FileNotFoundError(f"Missing {power_path}")
    records = _load_json(power_path)
    bank: Dict[Tuple[str, str, str], np.ndarray] = {}
    for record in records:
        image_id = str(record.get("image_id"))
        for level_key, level_data in (record.get("levels") or {}).items():
            if "overall" in groups and level_data.get("W_t"):
                bank[(image_id, level_key, "overall")] = np.asarray(level_data["W_t"], dtype=np.float64)
            layer_groups = level_data.get("layer_groups") or {}
            for group_name in groups:
                if group_name == "overall":
                    continue
                group_data = layer_groups.get(group_name) or {}
                if group_data.get("W_t"):
                    bank[(image_id, level_key, group_name)] = np.asarray(group_data["W_t"], dtype=np.float64)
    return bank


def _centered_vectors(w: np.ndarray, d: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    w_arr = np.asarray(w, dtype=np.float64).reshape(-1)
    d_arr = np.asarray(d, dtype=np.float64).reshape(-1)
    usable = min(w_arr.size, d_arr.size)
    w_arr = w_arr[:usable]
    d_arr = d_arr[:usable]
    if mode in {"filter_centered", "both_centered"}:
        w_arr = w_arr - float(np.mean(w_arr))
    if mode in {"delta_centered", "both_centered"}:
        d_arr = d_arr - float(np.mean(d_arr))
    return w_arr, d_arr


def _first_order_overlap(w: np.ndarray, d: np.ndarray, mode: str) -> Optional[float]:
    w_arr, d_arr = _centered_vectors(w, d, mode)
    if w_arr.size == 0:
        return None
    value = float(np.sum(w_arr * d_arr))
    return value if math.isfinite(value) else None


def _accumulate_row(
    rows: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]],
    *,
    domain: str,
    group_name: str,
    overlap_space: str,
    centering: str,
    image_id: str,
    level_key: str,
    perturbation_name: str,
    severity: Any,
    score: Optional[float],
    target: Optional[float],
) -> None:
    if score is None or target is None:
        return
    if not math.isfinite(score) or not math.isfinite(target):
        return
    rows[(domain, group_name, overlap_space, centering)].append(
        {
            "image_id": image_id,
            "level": level_key,
            "level_index": LEVEL_ORDER.index(level_key) if level_key in LEVEL_ORDER else None,
            "perturbation": str(perturbation_name),
            "severity": str(severity),
            "match_key": f"{image_id}::{perturbation_name}::sev{severity}",
            "score": float(score),
            "target": float(target),
        }
    )


def _collect_rows(
    run_dir: Path,
    *,
    domains: Sequence[str],
    groups: Sequence[str],
    centering_modes: Sequence[str],
    target_key: str,
    normalization: str,
    filter_mode: str,
) -> Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]:
    exp1_path = run_dir / "exp1" / "per_sample.jsonl"
    filter_dir_name = FILTER_2D_DIRS.get(filter_mode)
    if filter_dir_name is None:
        raise ValueError(f"Unknown filter mode {filter_mode!r}")
    filters_2d_dir = run_dir / "exp2" / filter_dir_name
    if not exp1_path.exists():
        raise FileNotFoundError(f"Missing {exp1_path}")
    if not filters_2d_dir.exists():
        raise FileNotFoundError(f"Missing {filters_2d_dir}")

    radial_filters = _load_filter_bank(run_dir, groups)
    npz_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    aligned_cache: Dict[Tuple[str, str, Tuple[int, int]], Optional[np.ndarray]] = {}
    rows: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for record in _iter_jsonl(exp1_path):
        image_id = str(record.get("image_id"))
        for level_key, level_data in (record.get("levels") or {}).items():
            if level_key not in LEVEL_ORDER:
                continue
            for perturbation in level_data.get("perturbations", []):
                target = _target_value(level_data, perturbation, target_key)
                if target is None:
                    continue
                perturbation_name = str(perturbation.get("name", "unknown"))
                severity = perturbation.get("severity", "unknown")

                for domain in domains:
                    spec = DOMAIN_SPECS[domain]
                    radial_delta_key = str(spec[f"delta_radial_{normalization}"])
                    radial_delta = perturbation.get(radial_delta_key)
                    delta_2d_file = perturbation.get(str(spec["delta_2d_file"]))
                    delta_2d_dir = run_dir / "exp1" / str(spec["delta_2d_dir"])
                    delta_2d_key = str(spec[f"delta_2d_{normalization}"])
                    delta_2d_path = delta_2d_dir / str(delta_2d_file) if delta_2d_file else None

                    for group_name in groups:
                        radial_w = radial_filters.get((image_id, level_key, group_name))
                        w_2d = _load_npz_array(
                            filters_2d_dir / f"{image_id}_{level_key}_{group_name}.npz",
                            "W_t_2d",
                            npz_cache,
                        )

                        if filter_mode == "option_conditioned" and radial_w is not None and radial_delta:
                            radial_delta_arr = np.asarray(radial_delta, dtype=np.float64)
                            for centering in centering_modes:
                                _accumulate_row(
                                    rows,
                                    domain=domain,
                                    group_name=group_name,
                                    overlap_space="radial_first_order",
                                    centering=centering,
                                    image_id=image_id,
                                    level_key=level_key,
                                    perturbation_name=perturbation_name,
                                    severity=severity,
                                    score=_first_order_overlap(radial_w, radial_delta_arr, centering),
                                    target=target,
                                )

                        if w_2d is not None and delta_2d_path is not None:
                            aligned_delta = _load_aligned_npz_array(
                                delta_2d_path,
                                delta_2d_key,
                                (int(w_2d.shape[0]), int(w_2d.shape[1])),
                                npz_cache,
                                aligned_cache,
                            )
                            if aligned_delta is None:
                                continue
                            for centering in centering_modes:
                                _accumulate_row(
                                    rows,
                                    domain=domain,
                                    group_name=group_name,
                                    overlap_space="two_d_first_order",
                                    centering=centering,
                                    image_id=image_id,
                                    level_key=level_key,
                                    perturbation_name=perturbation_name,
                                    severity=severity,
                                    score=_first_order_overlap(w_2d, aligned_delta, centering),
                                    target=target,
                                )
    return rows


def _group_by_match(rows: Sequence[Dict[str, Any]], min_levels: int) -> List[List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["match_key"])].append(row)
    groups: List[List[Dict[str, Any]]] = []
    for bucket_rows in buckets.values():
        levels = {row["level"] for row in bucket_rows}
        if len(levels) < min_levels:
            continue
        groups.append(sorted(bucket_rows, key=lambda row: row.get("level_index") or 999))
    return groups


def _demeaned_xy(groups: Sequence[Sequence[Dict[str, Any]]]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for rows in groups:
        x = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
        if x.size < 2:
            continue
        xs.extend((x - float(np.mean(x))).tolist())
        ys.extend((y - float(np.mean(y))).tolist())
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _group_demeaned_stats(groups: Sequence[Sequence[Dict[str, Any]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx2: List[float] = []
    sy2: List[float] = []
    sxy: List[float] = []
    for rows in groups:
        x = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
        if x.size < 2:
            continue
        x_dm = x - float(np.mean(x))
        y_dm = y - float(np.mean(y))
        sx2.append(float(np.dot(x_dm, x_dm)))
        sy2.append(float(np.dot(y_dm, y_dm)))
        sxy.append(float(np.dot(x_dm, y_dm)))
    return np.asarray(sx2, dtype=np.float64), np.asarray(sy2, dtype=np.float64), np.asarray(sxy, dtype=np.float64)


def _slope_no_intercept(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    denom = float(np.dot(x, x))
    if denom <= EPS:
        return None
    return float(np.dot(x, y) / denom)


def _standardized_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 3 or np.std(x) <= EPS or np.std(y) <= EPS:
        return None
    z_x = (x - float(np.mean(x))) / float(np.std(x))
    z_y = (y - float(np.mean(y))) / float(np.std(y))
    return _slope_no_intercept(z_x, z_y)


def _bootstrap_fe(
    groups: Sequence[Sequence[Dict[str, Any]]],
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Optional[float]]:
    if not groups or n_bootstrap <= 0:
        return {
            "fe_pearson_ci_low": None,
            "fe_pearson_ci_high": None,
            "fe_slope_ci_low": None,
            "fe_slope_ci_high": None,
        }
    rng = np.random.default_rng(seed)
    sx2, sy2, sxy = _group_demeaned_stats(groups)
    if sx2.size == 0:
        return {
            "fe_pearson_ci_low": None,
            "fe_pearson_ci_high": None,
            "fe_slope_ci_low": None,
            "fe_slope_ci_high": None,
        }
    pearsons = np.empty(n_bootstrap, dtype=np.float64)
    slopes = np.empty(n_bootstrap, dtype=np.float64)
    n_groups = int(sx2.size)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_groups, size=n_groups)
        total_sx2 = float(np.sum(sx2[indices]))
        total_sy2 = float(np.sum(sy2[indices]))
        total_sxy = float(np.sum(sxy[indices]))
        pearsons[_] = (
            total_sxy / math.sqrt(total_sx2 * total_sy2)
            if total_sx2 > EPS and total_sy2 > EPS
            else np.nan
        )
        slopes[_] = total_sxy / total_sx2 if total_sx2 > EPS else np.nan
    pearsons = pearsons[np.isfinite(pearsons)]
    slopes = slopes[np.isfinite(slopes)]
    return {
        "fe_pearson_ci_low": float(np.percentile(pearsons, 2.5)) if pearsons.size else None,
        "fe_pearson_ci_high": float(np.percentile(pearsons, 97.5)) if pearsons.size else None,
        "fe_slope_ci_low": float(np.percentile(slopes, 2.5)) if slopes.size else None,
        "fe_slope_ci_high": float(np.percentile(slopes, 97.5)) if slopes.size else None,
    }


def _summarize_variant(
    key: Tuple[str, str, str, str],
    rows: Sequence[Dict[str, Any]],
    *,
    target_key: str,
    normalization: str,
    filter_mode: str,
    min_levels: int,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    groups = _group_by_match(rows, min_levels)
    x_dm, y_dm = _demeaned_xy(groups)
    within_pearsons: List[float] = []
    within_spearmans: List[float] = []
    for group_rows in groups:
        x = [float(row["score"]) for row in group_rows]
        y = [float(row["target"]) for row in group_rows]
        pearson = _pearson(x, y)
        spearman = _spearman(x, y)
        if pearson is not None:
            within_pearsons.append(pearson)
        if spearman is not None:
            within_spearmans.append(spearman)
    domain, group_name, overlap_space, centering = key
    positive_pearsons = [value for value in within_pearsons if value > 0.0]
    summary = {
        "domain": domain,
        "group": group_name,
        "overlap_space": overlap_space,
        "centering": centering,
        "target": target_key,
        "normalization": normalization,
        "filter_mode": filter_mode,
        "matched_unit": "image_id x perturbation_name x severity",
        "min_levels_per_unit": min_levels,
        "n_rows": int(len(rows)),
        "n_matched_units": int(len(groups)),
        "n_demeaned_rows": int(x_dm.size),
        "fe_pearson_r": _pearson(x_dm, y_dm),
        "fe_spearman_rho": _spearman(x_dm, y_dm),
        "fe_slope": _slope_no_intercept(x_dm, y_dm),
        "fe_standardized_beta": _standardized_slope(x_dm, y_dm),
        "within_pearson_n": int(len(within_pearsons)),
        "within_pearson_mean": float(np.mean(within_pearsons)) if within_pearsons else None,
        "within_pearson_median": float(np.median(within_pearsons)) if within_pearsons else None,
        "within_pearson_positive_rate": (
            float(len(positive_pearsons) / len(within_pearsons)) if within_pearsons else None
        ),
        "within_spearman_n": int(len(within_spearmans)),
        "within_spearman_mean": float(np.mean(within_spearmans)) if within_spearmans else None,
        "within_spearman_median": float(np.median(within_spearmans)) if within_spearmans else None,
    }
    summary.update(_bootstrap_fe(groups, n_bootstrap=n_bootstrap, seed=seed))
    return summary


def _write_outputs(out_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "notes": [
            "Rows are matched within image_id x perturbation_name x severity.",
            "Fixed-effect statistics demean predicted overlap and observed target inside each matched unit.",
            "This controls image content and perturbation strength before testing whether level-specific W_t predicts volatility.",
            "Within-unit correlations are computed over the available L1-L8 points and summarized as a distribution.",
        ],
        "summaries": list(summaries),
    }
    with (out_dir / "matched_overlap_fixed_effects_summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    fieldnames = [
        "domain",
        "group",
        "overlap_space",
        "centering",
        "target",
        "normalization",
        "filter_mode",
        "matched_unit",
        "min_levels_per_unit",
        "n_rows",
        "n_matched_units",
        "n_demeaned_rows",
        "fe_pearson_r",
        "fe_spearman_rho",
        "fe_slope",
        "fe_standardized_beta",
        "fe_pearson_ci_low",
        "fe_pearson_ci_high",
        "fe_slope_ci_low",
        "fe_slope_ci_high",
        "within_pearson_n",
        "within_pearson_mean",
        "within_pearson_median",
        "within_pearson_positive_rate",
        "within_spearman_n",
        "within_spearman_mean",
        "within_spearman_median",
    ]
    with (out_dir / "matched_overlap_fixed_effects_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _plot_summary(out_dir: Path, summaries: Sequence[Dict[str, Any]], target_key: str) -> None:
    rows = [
        row
        for row in summaries
        if row["centering"] == "raw" and row["group"] == "late" and row["fe_pearson_r"] is not None
    ]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), sharey=True)
    for ax, domain in zip(axes, ["image_space", "vision_feature_space"]):
        domain_rows = [row for row in rows if row["domain"] == domain]
        labels: List[str] = []
        values: List[float] = []
        low_errors: List[float] = []
        high_errors: List[float] = []
        colors: List[str] = []
        for overlap_space in SPACE_ORDER:
            row = next((item for item in domain_rows if item["overlap_space"] == overlap_space), None)
            if row is None:
                continue
            value = float(row["fe_pearson_r"])
            ci_low = row.get("fe_pearson_ci_low")
            ci_high = row.get("fe_pearson_ci_high")
            labels.append(overlap_space.replace("_first_order", "").replace("_", " "))
            values.append(value)
            low_errors.append(value - float(ci_low) if ci_low is not None else 0.0)
            high_errors.append(float(ci_high) - value if ci_high is not None else 0.0)
            colors.append("#6b7280" if overlap_space == "radial_first_order" else "#2563eb")
        x = np.arange(len(values), dtype=np.float64)
        ax.bar(x, values, color=colors, alpha=0.84)
        if values:
            ax.errorbar(
                x,
                values,
                yerr=np.vstack([low_errors, high_errors]),
                fmt="none",
                ecolor="#111827",
                capsize=4,
                linewidth=1.2,
            )
        ax.axhline(0.0, color="#111827", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(domain.replace("_", " "))
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(f"Fixed-effect Pearson r vs {target_key}")
    fig.suptitle("Matched overlap law: same image and perturbation, level filter changes", fontsize=14)
    fig.text(
        0.01,
        0.01,
        "Each statistic demeans overlap and target inside image_id x perturbation_name x severity. "
        "Error bars are bootstrap 95% CIs over matched units. Shown: late attention group, raw first-order overlap.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_dir / "matched_overlap_fixed_effects_late_raw.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched fixed-effect diagnostics for Exp5 overlap laws."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Frequency-alignment run output directory containing exp1/ and exp2/.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to <output-dir>/plots_overlap_matched_fe.",
    )
    parser.add_argument(
        "--domains",
        default="image_space,vision_feature_space",
        help="Comma-separated domains: image_space,vision_feature_space.",
    )
    parser.add_argument(
        "--groups",
        default=",".join(GROUP_ORDER),
        help="Comma-separated attention groups.",
    )
    parser.add_argument(
        "--centering",
        default="raw",
        help="Comma-separated centering modes: raw,delta_centered,filter_centered,both_centered.",
    )
    parser.add_argument(
        "--target",
        default="loglik_volatility",
        help="Observed target, e.g. loglik_volatility or accuracy_drop.",
    )
    parser.add_argument(
        "--normalization",
        default="relative",
        choices=["relative", "raw"],
        help="Use raw or energy-relative perturbation spectra.",
    )
    parser.add_argument(
        "--filter-mode",
        default="option_conditioned",
        choices=sorted(FILTER_2D_DIRS),
        help=(
            "Which saved 2D attention filters to use. question_only reads "
            "exp2/filters_2d_question_only and omits radial rows because no "
            "question-only radial filter bank is saved."
        ),
    )
    parser.add_argument(
        "--min-levels",
        type=int,
        default=8,
        help="Minimum distinct levels required inside each matched unit.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap iterations for FE confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    run_dir = args.output_dir
    default_subdir = "plots_overlap_matched_fe"
    if args.filter_mode != "option_conditioned":
        default_subdir = f"{default_subdir}_{args.filter_mode}"
    out_dir = args.plots_dir or (run_dir / default_subdir)
    domains = _parse_csv(args.domains, list(DOMAIN_SPECS))
    groups = _parse_csv(args.groups, GROUP_ORDER)
    centering_modes = _parse_csv(args.centering, ["raw"])
    unknown_domains = sorted(set(domains) - set(DOMAIN_SPECS))
    unknown_centering = sorted(set(centering_modes) - set(CENTERING_ORDER))
    if unknown_domains:
        raise ValueError(f"Unknown domains: {unknown_domains}")
    if unknown_centering:
        raise ValueError(f"Unknown centering modes: {unknown_centering}")
    if args.min_levels < 3:
        raise ValueError("--min-levels must be at least 3")

    rows_by_variant = _collect_rows(
        run_dir,
        domains=domains,
        groups=groups,
        centering_modes=centering_modes,
        target_key=args.target,
        normalization=args.normalization,
        filter_mode=args.filter_mode,
    )
    summaries = [
        _summarize_variant(
            key,
            rows,
            target_key=args.target,
            normalization=args.normalization,
            filter_mode=args.filter_mode,
            min_levels=args.min_levels,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        for key, rows in sorted(rows_by_variant.items())
    ]
    _write_outputs(out_dir, summaries)
    _plot_summary(out_dir, summaries, args.target)

    print(f"Wrote matched fixed-effect diagnostics to {out_dir}")
    print(f"CSV: {out_dir / 'matched_overlap_fixed_effects_summary.csv'}")
    print("Late raw fixed-effect Pearson r:")
    for row in summaries:
        if row["group"] != "late" or row["centering"] != "raw":
            continue
        r_value = row["fe_pearson_r"]
        r_text = "NA" if r_value is None else f"{r_value:.4f}"
        ci_low = row.get("fe_pearson_ci_low")
        ci_high = row.get("fe_pearson_ci_high")
        ci_text = ""
        if ci_low is not None and ci_high is not None:
            ci_text = f" CI=[{ci_low:.4f}, {ci_high:.4f}]"
        print(
            f"  {row['domain']} {row['overlap_space']}: "
            f"r={r_text}{ci_text} matched_units={row['n_matched_units']}"
        )


if __name__ == "__main__":
    main()
