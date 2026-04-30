#!/usr/bin/env python3
"""Offline Exp5 overlap-law probes on existing results.

This is intentionally outside ``frequency_alignment``. It does not modify the
production pipeline and does not run model inference.

The script compares the saved radial overlap law against two diagnostics:
1. coarser aggregation views that average away image noise;
2. an area-corrected radial approximation to true 2D spectral overlap.

True 2D overlap requires saved full FFT maps for W_t(u,v) and DeltaF(u,v).
The current result folders only store radial vectors, so this script reports
2D overlap as unavailable and tests the best possible radial proxy:

    S_uniform2d = sum_b W_b * DeltaF_b / N_b

where N_b is the number of FFT cells in radial band b. This is exactly what
you get if energy is distributed uniformly inside each radial band before doing
the true 2D dot product.

It also reports a diagnostic density-product variant:

    S_density2 = sum_b W_b * DeltaF_b / N_b**2

This is not the true 2D dot product; it is a stress test for whether wide
radial annuli are dominating the current overlap score.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DIRS = (
    "frequency_alignment_outputs_local_200samples_20260428_125813",
    "frequency_alignment_outputs_local_200samples_20260428_131910",
)
TARGETS = ("loglik_volatility", "loglik_erosion", "accuracy_drop")
PREDICTORS = (
    "predicted_first_order",
    "predicted_linear",
    "predicted_quadratic",
    "predicted_uniform2d_radial",
    "predicted_uniform2d_radial_scaled",
    "predicted_density2_radial",
    "predicted_density2_radial_scaled",
)
EPS = 1e-12


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    sorted_vals = arr[order]
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _corr(x_values: Sequence[Any], y_values: Sequence[Any], *, spearman: bool = False) -> Dict[str, Any]:
    pairs: List[Tuple[float, float]] = []
    for x_raw, y_raw in zip(x_values, y_values):
        x = _safe_float(x_raw, float("nan"))
        y = _safe_float(y_raw, float("nan"))
        if not math.isnan(x) and not math.isnan(y):
            pairs.append((x, y))
    if len(pairs) < 3:
        return {"r": None, "n": len(pairs)}
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if spearman:
        x = _rankdata(x)
        y = _rankdata(y)
    if float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return {"r": None, "n": int(len(x)), "degenerate": True}
    return {"r": float(np.corrcoef(x, y)[0, 1]), "n": int(len(x))}


def _zscore(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray([_safe_float(v, float("nan")) for v in values], dtype=np.float64)
    mask = ~np.isnan(arr)
    out = np.zeros_like(arr)
    if not np.any(mask):
        return out
    std = float(np.std(arr[mask]))
    if std <= EPS:
        return out
    out[mask] = (arr[mask] - float(np.mean(arr[mask]))) / std
    return out


def _standardized_ols(
    rows: List[Dict[str, Any]],
    *,
    target_key: str,
    predicted_key: str,
    controls: Sequence[str],
    family_fixed_effects: bool,
) -> Dict[str, Any]:
    if len(rows) < 5:
        return {"skipped": "too_few_rows", "n": len(rows)}

    y_raw = np.asarray([_safe_float(row.get(target_key), float("nan")) for row in rows])
    columns = [np.ones(len(rows), dtype=np.float64)]
    names = ["intercept"]

    pred = [math.log1p(max(0.0, _safe_float(row.get(predicted_key), 0.0))) for row in rows]
    columns.append(_zscore(pred))
    names.append(f"z_log1p_{predicted_key}")

    for control in controls:
        col = np.asarray([_safe_float(row.get(control), float("nan")) for row in rows])
        if np.any(np.isnan(col)) or float(np.std(col)) <= EPS:
            continue
        columns.append(_zscore(col))
        names.append(control)

    if family_fixed_effects:
        families = sorted({str(row.get("perturbation_family", "unknown")) for row in rows})
        for family in families[1:]:
            columns.append(
                np.asarray(
                    [1.0 if str(row.get("perturbation_family", "unknown")) == family else 0.0 for row in rows],
                    dtype=np.float64,
                )
            )
            names.append(f"FE:{family}")

    x_all = np.column_stack(columns)
    mask = ~np.isnan(y_raw)
    y = _zscore(y_raw[mask])
    x = x_all[mask]
    if len(y) < x.shape[1] + 2:
        return {"skipped": "too_few_rows_or_too_many_predictors", "n": int(len(y)), "p": int(x.shape[1])}

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    pred_name = f"z_log1p_{predicted_key}"
    return {
        "n": int(len(y)),
        "p": int(x.shape[1]),
        "target": target_key,
        "predicted": predicted_key,
        "family_fixed_effects": bool(family_fixed_effects),
        "predicted_beta_std": float(beta[names.index(pred_name)]),
        "r2": float(1.0 - ss_res / max(ss_tot, EPS)),
        "controls_used": [
            name for name in names if name not in {"intercept", pred_name} and not name.startswith("FE:")
        ],
        "num_family_fixed_effects": sum(1 for name in names if name.startswith("FE:")),
    }


def _softmax_from_loglik(scores: Dict[str, Any]) -> Dict[str, float]:
    if not scores:
        return {}
    labels = list(scores.keys())
    vals = np.asarray([_safe_float(scores[label], -1e9) for label in labels], dtype=np.float64)
    vals -= float(np.max(vals))
    exp_vals = np.exp(vals)
    probs = exp_vals / max(float(np.sum(exp_vals)), EPS)
    return {label: float(prob) for label, prob in zip(labels, probs)}


def _bin_cell_counts(
    patch_grid: Sequence[int],
    num_bands: int,
    *,
    suppress_dc: bool,
) -> np.ndarray:
    if not patch_grid or len(patch_grid) < 2:
        return np.ones(int(num_bands), dtype=np.float64)
    h, w = int(patch_grid[0]), int(patch_grid[1])
    if h <= 0 or w <= 0:
        return np.ones(int(num_bands), dtype=np.float64)
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    max_dist = float(np.sqrt(cy ** 2 + cx ** 2))
    if max_dist <= EPS:
        counts = np.zeros(int(num_bands), dtype=np.float64)
        if num_bands > 0:
            counts[0] = 1.0
        return counts

    edges = np.linspace(0.0, max_dist, int(num_bands) + 1)
    counts = np.zeros(int(num_bands), dtype=np.float64)
    for band_idx in range(int(num_bands)):
        if band_idx == int(num_bands) - 1:
            mask = (dist >= edges[band_idx]) & (dist <= edges[band_idx + 1])
        else:
            mask = (dist >= edges[band_idx]) & (dist < edges[band_idx + 1])
        counts[band_idx] = float(mask.sum())

    # If DC is suppressed, the center cell contributes zero energy. For the
    # uniform-within-band approximation, remove it from the effective area.
    if suppress_dc and len(counts) > 0:
        counts[0] = max(1.0, counts[0] - 1.0)
    return np.maximum(counts, 1.0)


def _load_exp2_maps(exp2_dir: Path, group: str) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], List[int]]]:
    filters_dir = exp2_dir / "filters"
    filter_map: Dict[Tuple[str, str], np.ndarray] = {}
    if filters_dir.exists():
        suffix = f"_{group}.npy"
        for path in filters_dir.glob(f"*{suffix}"):
            stem = path.stem
            if stem.startswith("average_") or stem.endswith("_l2"):
                continue
            base = stem[: -len(f"_{group}")]
            parts = base.split("_", 1)
            if len(parts) != 2:
                continue
            image_id, level = parts
            filter_map[(image_id, level)] = np.load(path).astype(np.float64)

    patch_grid_map: Dict[Tuple[str, str], List[int]] = {}
    power_path = exp2_dir / "power_spectra.json"
    if power_path.exists():
        for record in _load_json(power_path):
            image_id = str(record.get("image_id"))
            for level, level_data in (record.get("levels") or {}).items():
                patch_grid = level_data.get("patch_grid")
                if patch_grid:
                    patch_grid_map[(image_id, str(level))] = [int(patch_grid[0]), int(patch_grid[1])]
    return filter_map, patch_grid_map


def _build_exp1_maps(exp1_dir: Path) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
    perturbation_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    level_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    path = exp1_dir / "per_sample.jsonl"
    if not path.exists():
        return perturbation_map, level_map
    for record in _load_jsonl(path):
        image_id = str(record.get("image_id"))
        for level, level_data in (record.get("levels") or {}).items():
            level_key = str(level)
            clean = level_data.get("clean", {}) or {}
            scores = clean.get("scores", {}) or {}
            answer = str(level_data.get("answer_label", ""))
            probs = _softmax_from_loglik(scores)
            sorted_scores = sorted((_safe_float(v, -1e9) for v in scores.values()), reverse=True)
            margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0
            level_map[(image_id, level_key)] = {
                "clean_answer_prob": probs.get(answer, 0.0),
                "clean_margin": float(margin),
                "clean_prediction_entropy": _safe_float(
                    clean.get("prediction_entropy", level_data.get("prediction_entropy")), 0.0
                ),
            }
            for perturbation in level_data.get("perturbations", []) or []:
                name = str(perturbation.get("name"))
                perturbation_map[(image_id, level_key, name)] = {
                    "delta_f": perturbation.get("delta_f") or [],
                    "delta_f_relative": perturbation.get("delta_f_relative") or [],
                    "severity": perturbation.get("severity"),
                    "perturbation_prediction_entropy": _safe_float(
                        perturbation.get("prediction_entropy"), 0.0
                    ),
                }
    return perturbation_map, level_map


def _load_sample_rows(
    output_dir: Path,
    *,
    max_images: int,
    group: str,
    delta_key: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sample_path = output_dir / "exp5" / "sample_scatter_data.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {sample_path}")

    rows = _load_json(sample_path)
    image_ids = sorted({str(row.get("image_id")) for row in rows})
    keep_images = set(image_ids[:max_images])
    rows = [dict(row) for row in rows if str(row.get("image_id")) in keep_images]

    cfg = _load_json(output_dir / "config_snapshot.json")
    suppress_dc = bool(cfg.get("analysis", {}).get("suppress_dc", False))
    num_bands = int(cfg.get("analysis", {}).get("num_bands", 0) or 0)

    filters, patch_grids = _load_exp2_maps(output_dir / "exp2", group)
    perturbations, level_controls = _build_exp1_maps(output_dir / "exp1")

    unavailable_2d_reasons = {
        "full_attention_power_2d_saved": False,
        "full_delta_power_2d_saved": False,
        "reason": (
            "Existing artifacts contain radial vectors only. True 2D overlap "
            "needs saved W_t(u,v) and DeltaF(u,v) maps from a future run."
        ),
    }
    density_rows = 0
    missing_density = 0
    current_recompute_errors: List[float] = []

    for row in rows:
        image_id = str(row.get("image_id"))
        level = str(row.get("level"))
        perturbation = str(row.get("perturbation"))
        row.update(level_controls.get((image_id, level), {}))
        pert_info = perturbations.get((image_id, level, perturbation), {})
        row.update(
            {
                "severity": pert_info.get("severity"),
                "perturbation_prediction_entropy": pert_info.get(
                    "perturbation_prediction_entropy", 0.0
                ),
            }
        )
        w = filters.get((image_id, level))
        delta = np.asarray(pert_info.get(delta_key) or [], dtype=np.float64)
        patch_grid = patch_grids.get((image_id, level))
        if w is None or delta.size == 0 or not patch_grid:
            missing_density += 1
            continue

        m = min(len(w), len(delta), num_bands if num_bands > 0 else len(w))
        if m <= 0:
            missing_density += 1
            continue
        w_use = np.asarray(w[:m], dtype=np.float64)
        d_use = np.asarray(delta[:m], dtype=np.float64)
        counts = _bin_cell_counts(patch_grid, m, suppress_dc=suppress_dc)[:m]

        predicted_current = float(np.sum(w_use * d_use))
        row["predicted_recomputed_first_order"] = predicted_current
        row["predicted_uniform2d_radial"] = float(np.sum(w_use * d_use / counts))
        row["predicted_uniform2d_radial_scaled"] = float(
            np.sum(w_use * d_use / counts) * np.sum(counts)
        )
        # Diagnostic only: product of per-cell densities without multiplying
        # back by bin area. True uniform 2D overlap is /N_b, not /N_b**2, but
        # this tests whether strongly penalizing wide annuli helps empirically.
        row["predicted_density2_radial"] = float(np.sum(w_use * d_use / (counts ** 2)))
        row["predicted_density2_radial_scaled"] = float(
            np.sum(w_use * d_use / (counts ** 2)) * np.sum(counts)
        )
        row["radial_bin_count_min"] = float(np.min(counts))
        row["radial_bin_count_max"] = float(np.max(counts))
        row["patch_grid_h"] = int(patch_grid[0])
        row["patch_grid_w"] = int(patch_grid[1])
        current_recompute_errors.append(
            abs(predicted_current - _safe_float(row.get("predicted_first_order"), 0.0))
        )
        density_rows += 1

    diagnostics = {
        "true_2d_overlap": unavailable_2d_reasons,
        "density_rows": density_rows,
        "missing_density_rows": missing_density,
        "max_abs_error_recomputed_first_order_vs_saved": (
            float(max(current_recompute_errors)) if current_recompute_errors else None
        ),
        "mean_abs_error_recomputed_first_order_vs_saved": (
            float(np.mean(current_recompute_errors)) if current_recompute_errors else None
        ),
    }
    return rows, diagnostics


def _group_rows(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    numeric_keys = set(PREDICTORS) | set(TARGETS) | {
        "clean_correct",
        "clean_answer_prob",
        "clean_margin",
        "clean_prediction_entropy",
        "perturbation_prediction_entropy",
        "is_binary",
        "option_hardness_score",
        "prompt_complexity_score",
        "question_complexity_score",
    }
    buckets: Dict[Tuple[Any, ...], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    meta: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        meta.setdefault(key, {k: row.get(k) for k in keys})
        for numeric_key in numeric_keys:
            if numeric_key in row:
                buckets[key][numeric_key].append(_safe_float(row.get(numeric_key), 0.0))
    grouped: List[Dict[str, Any]] = []
    for key, values in buckets.items():
        out = dict(meta[key])
        out["n"] = max((len(v) for v in values.values()), default=0)
        for numeric_key, vals in values.items():
            out[numeric_key] = float(np.mean(vals)) if vals else 0.0
        grouped.append(out)
    return grouped


def _correlation_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    table: Dict[str, Any] = {}
    for target in TARGETS:
        table[target] = {}
        for predictor in PREDICTORS:
            available = [row for row in rows if predictor in row and target in row]
            table[target][predictor] = {
                "pearson": _corr(
                    [row.get(predictor) for row in available],
                    [row.get(target) for row in available],
                ),
                "spearman": _corr(
                    [row.get(predictor) for row in available],
                    [row.get(target) for row in available],
                    spearman=True,
                ),
            }
    return table


def _correlation_table_by_level(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One macro correlation per level, over perturbation/severity cells."""
    output: Dict[str, Any] = {}
    for level in sorted({str(row.get("level")) for row in rows if row.get("level") is not None}):
        level_rows = [row for row in rows if str(row.get("level")) == level]
        output[level] = {
            "n_cells": len(level_rows),
            "correlations": _correlation_table(level_rows),
        }
    return output


def _within_family(rows: List[Dict[str, Any]], *, target: str, predictor: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    families = sorted({str(row.get("perturbation_family", "unknown")) for row in rows})
    for family in families:
        family_rows = [
            row
            for row in rows
            if str(row.get("perturbation_family", "unknown")) == family
            and predictor in row
            and target in row
        ]
        if len(family_rows) < 10:
            continue
        out[family] = {
            "n": len(family_rows),
            "pearson": _corr(
                [row.get(predictor) for row in family_rows],
                [row.get(target) for row in family_rows],
            ),
            "spearman": _corr(
                [row.get(predictor) for row in family_rows],
                [row.get(target) for row in family_rows],
                spearman=True,
            ),
        }
    return out


def _within_strata_correlations(
    rows: List[Dict[str, Any]],
    *,
    keys: Sequence[str],
    target: str,
    predictors: Sequence[str],
    min_rows: int = 8,
) -> Dict[str, Any]:
    """Correlate predicted vs observed within each fixed stratum across images."""
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)

    per_stratum: Dict[str, Any] = {}
    summary: Dict[str, Any] = {
        predictor: {
            "num_strata": 0,
            "num_positive_pearson": 0,
            "mean_pearson": None,
            "median_pearson": None,
            "weighted_mean_pearson": None,
            "mean_spearman": None,
            "median_spearman": None,
            "weighted_mean_spearman": None,
        }
        for predictor in predictors
    }
    pearson_values: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    spearman_values: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

    for key, bucket_rows in sorted(buckets.items()):
        label = "|".join(str(part) for part in key)
        stratum_payload: Dict[str, Any] = {
            key_name: key_value for key_name, key_value in zip(keys, key)
        }
        stratum_payload["n_rows"] = len(bucket_rows)
        stratum_payload["predictors"] = {}
        for predictor in predictors:
            usable = [row for row in bucket_rows if predictor in row and target in row]
            if len(usable) < min_rows:
                continue
            pearson = _corr(
                [row.get(predictor) for row in usable],
                [row.get(target) for row in usable],
            )
            spearman = _corr(
                [row.get(predictor) for row in usable],
                [row.get(target) for row in usable],
                spearman=True,
            )
            stratum_payload["predictors"][predictor] = {
                "pearson": pearson,
                "spearman": spearman,
            }
            if pearson.get("r") is not None:
                pearson_values[predictor].append((float(pearson["r"]), int(pearson["n"])))
            if spearman.get("r") is not None:
                spearman_values[predictor].append((float(spearman["r"]), int(spearman["n"])))
        if stratum_payload["predictors"]:
            per_stratum[label] = stratum_payload

    for predictor in predictors:
        p_vals = pearson_values.get(predictor, [])
        s_vals = spearman_values.get(predictor, [])
        if p_vals:
            p_arr = np.asarray([value for value, _ in p_vals], dtype=np.float64)
            p_weights = np.asarray([weight for _, weight in p_vals], dtype=np.float64)
            summary[predictor].update(
                {
                    "num_strata": int(len(p_vals)),
                    "num_positive_pearson": int(np.sum(p_arr > 0)),
                    "fraction_positive_pearson": float(np.mean(p_arr > 0)),
                    "mean_pearson": float(np.mean(p_arr)),
                    "median_pearson": float(np.median(p_arr)),
                    "weighted_mean_pearson": float(np.average(p_arr, weights=p_weights)),
                }
            )
        if s_vals:
            s_arr = np.asarray([value for value, _ in s_vals], dtype=np.float64)
            s_weights = np.asarray([weight for _, weight in s_vals], dtype=np.float64)
            summary[predictor].update(
                {
                    "num_positive_spearman": int(np.sum(s_arr > 0)),
                    "fraction_positive_spearman": float(np.mean(s_arr > 0)),
                    "mean_spearman": float(np.mean(s_arr)),
                    "median_spearman": float(np.median(s_arr)),
                    "weighted_mean_spearman": float(np.average(s_arr, weights=s_weights)),
                }
            )

    return {
        "target": target,
        "stratum_keys": list(keys),
        "min_rows": int(min_rows),
        "summary": summary,
        "per_stratum": per_stratum,
    }


def _residualized_target_corr(rows: List[Dict[str, Any]], *, target: str, predictor: str) -> Dict[str, Any]:
    controls = (
        "clean_answer_prob",
        "clean_margin",
        "clean_prediction_entropy",
        "perturbation_prediction_entropy",
        "is_binary",
        "option_hardness_score",
        "prompt_complexity_score",
    )
    usable = [row for row in rows if predictor in row and target in row]
    if len(usable) < 10:
        return {"skipped": "too_few_rows", "n": len(usable)}

    y = np.asarray([_safe_float(row.get(target), float("nan")) for row in usable])
    columns = [np.ones(len(usable), dtype=np.float64)]
    used_controls: List[str] = []
    for control in controls:
        col = np.asarray([_safe_float(row.get(control), float("nan")) for row in usable])
        if np.any(np.isnan(col)) or float(np.std(col)) <= EPS:
            continue
        columns.append(_zscore(col))
        used_controls.append(control)
    mask = ~np.isnan(y)
    x = np.column_stack(columns)[mask]
    y_valid = y[mask]
    if len(y_valid) < x.shape[1] + 2:
        return {"skipped": "too_few_rows_after_controls", "n": int(len(y_valid))}
    beta, *_ = np.linalg.lstsq(x, y_valid, rcond=None)
    residual = y_valid - x @ beta
    pred = [row.get(predictor) for row, keep in zip(usable, mask) if keep]
    return {
        "n": int(len(y_valid)),
        "controls_used": used_controls,
        "pearson": _corr(pred, residual),
        "spearman": _corr(pred, residual, spearman=True),
    }


def _severity_from_perturbation(text: Any) -> Optional[int]:
    match = re.search(r"\|sev(\d+)", str(text or ""))
    if not match:
        return None
    return int(match.group(1))


def analyze_output_dir(
    output_dir: Path,
    *,
    max_images: int,
    group: str,
    delta_key: str,
) -> Dict[str, Any]:
    cfg = _load_json(output_dir / "config_snapshot.json")
    rows, availability = _load_sample_rows(
        output_dir,
        max_images=max_images,
        group=group,
        delta_key=delta_key,
    )
    for row in rows:
        if row.get("severity") is None:
            row["severity"] = _severity_from_perturbation(row.get("perturbation"))

    grouped_image_level_family = _group_rows(rows, ("image_id", "level", "perturbation_family"))
    grouped_level_family = _group_rows(rows, ("level", "perturbation_family"))
    grouped_level_family_perturbation = _group_rows(
        rows,
        ("level", "perturbation_family", "perturbation"),
    )
    grouped_level_family_severity = _group_rows(
        rows,
        ("level", "perturbation_family", "severity"),
    )

    controls = (
        "clean_answer_prob",
        "clean_margin",
        "clean_prediction_entropy",
        "perturbation_prediction_entropy",
        "is_binary",
        "option_hardness_score",
        "prompt_complexity_score",
    )

    return {
        "output_dir": str(output_dir),
        "suppress_dc": cfg.get("analysis", {}).get("suppress_dc"),
        "num_bands": cfg.get("analysis", {}).get("num_bands"),
        "num_bands_resolution": cfg.get("analysis", {}).get("num_bands_resolution"),
        "layer_group": group,
        "delta_key": delta_key,
        "max_images_requested": max_images,
        "num_images_used": len({str(row.get("image_id")) for row in rows}),
        "num_sample_rows": len(rows),
        "num_image_level_family_rows": len(grouped_image_level_family),
        "num_level_family_rows": len(grouped_level_family),
        "num_level_family_perturbation_rows": len(grouped_level_family_perturbation),
        "num_level_family_severity_rows": len(grouped_level_family_severity),
        "spectral_availability": availability,
        "sample_exact_correlations": _correlation_table(rows),
        "image_level_family_correlations": _correlation_table(grouped_image_level_family),
        "level_family_macro_correlations": _correlation_table(grouped_level_family),
        "level_family_perturbation_macro_correlations": _correlation_table(
            grouped_level_family_perturbation
        ),
        "level_family_severity_macro_correlations": _correlation_table(
            grouped_level_family_severity
        ),
        "level_family_severity_macro_correlations_by_level": _correlation_table_by_level(
            grouped_level_family_severity
        ),
        "within_family_volatility_first_order": _within_family(
            rows,
            target="loglik_volatility",
            predictor="predicted_first_order",
        ),
        "within_family_volatility_uniform2d_radial": _within_family(
            rows,
            target="loglik_volatility",
            predictor="predicted_uniform2d_radial_scaled",
        ),
        "within_level_family_severity_across_images": _within_strata_correlations(
            rows,
            keys=("level", "perturbation_family", "severity"),
            target="loglik_volatility",
            predictors=PREDICTORS,
            min_rows=8,
        ),
        "residualized_volatility": {
            predictor: _residualized_target_corr(
                rows,
                target="loglik_volatility",
                predictor=predictor,
            )
            for predictor in PREDICTORS
        },
        "family_fe_horse_race_volatility": {
            predictor: _standardized_ols(
                [row for row in rows if predictor in row],
                target_key="loglik_volatility",
                predicted_key=predictor,
                controls=controls,
                family_fixed_effects=True,
            )
            for predictor in PREDICTORS
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_result(result: Dict[str, Any]) -> None:
    print(f"\n=== {result['output_dir']} ===")
    print(
        "suppress_dc="
        f"{result.get('suppress_dc')} images={result['num_images_used']} "
        f"sample_rows={result['num_sample_rows']} group={result['layer_group']}"
    )
    print("true_2d_overlap:", result["spectral_availability"]["true_2d_overlap"]["reason"])
    for block in (
        "sample_exact_correlations",
        "image_level_family_correlations",
        "level_family_macro_correlations",
        "level_family_perturbation_macro_correlations",
        "level_family_severity_macro_correlations",
    ):
        print(f"\n{block} | target=loglik_volatility")
        for predictor in PREDICTORS:
            stats = result[block]["loglik_volatility"][predictor]
            pearson = stats["pearson"]
            spearman = stats["spearman"]
            print(
                f"  {predictor:34s} "
                f"r={_fmt(pearson.get('r'))} n={pearson.get('n')} "
                f"rho={_fmt(spearman.get('r'))}"
            )
    print("\nfamily-FE beta_std | target=loglik_volatility")
    for predictor, stats in result["family_fe_horse_race_volatility"].items():
        print(
            f"  {predictor:34s} beta={_fmt(stats.get('predicted_beta_std'))} "
            f"r2={_fmt(stats.get('r2'))} n={stats.get('n')}"
        )
    print("\nper-level macro correlation over family×severity cells | target=loglik_volatility")
    per_level = result["level_family_severity_macro_correlations_by_level"]
    for level, payload in per_level.items():
        stats = payload["correlations"]["loglik_volatility"]["predicted_first_order"]
        pearson = stats["pearson"]
        spearman = stats["spearman"]
        print(
            f"  {level:24s} cells={payload['n_cells']:3d} "
            f"first_order r={_fmt(pearson.get('r'))} "
            f"rho={_fmt(spearman.get('r'))}"
        )
    print("\nwithin level×family×severity across images | target=loglik_volatility")
    stratum_summary = result["within_level_family_severity_across_images"]["summary"]
    for predictor in PREDICTORS:
        stats = stratum_summary[predictor]
        print(
            f"  {predictor:34s} "
            f"mean_r={_fmt(stats.get('mean_pearson'))} "
            f"median_r={_fmt(stats.get('median_pearson'))} "
            f"frac_pos={_fmt(stats.get('fraction_positive_pearson'))} "
            f"strata={stats.get('num_strata')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        default=[path for path in DEFAULT_DIRS if Path(path).exists()],
        help="Existing output directories to inspect.",
    )
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--group", default="late", help="Exp2 layer group to use for W_t.")
    parser.add_argument(
        "--delta-key",
        default="delta_f_relative",
        choices=("delta_f", "delta_f_relative"),
        help="Perturbation spectrum stored in Exp1 to use.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("exp5_overlap_diagnostics_50.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output_dirs:
        raise SystemExit("No output dirs found/provided.")
    results = [
        analyze_output_dir(
            Path(output_dir),
            max_images=args.max_images,
            group=args.group,
            delta_key=args.delta_key,
        )
        for output_dir in args.output_dirs
    ]
    payload = {
        "description": (
            "Offline Exp5 overlap diagnostics. This file is not consumed by the "
            "main pipeline. True 2D overlap is reported unavailable when full "
            "2D spectra are absent from existing artifacts."
        ),
        "max_images": args.max_images,
        "layer_group": args.group,
        "delta_key": args.delta_key,
        "predictors": list(PREDICTORS),
        "targets": list(TARGETS),
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    for result in results:
        _print_result(result)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
