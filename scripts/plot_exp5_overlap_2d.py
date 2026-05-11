#!/usr/bin/env python3
"""Experimental plots for Exp 5 2D overlap diagnostics.

This script intentionally lives outside the production plotting pipeline. It
reads an existing run directory, uses or recomputes 2D overlap keys, and writes
plots to a separate folder for inspection before merging anything into
``frequency_alignment/plotting``. It supports both image-space 2D spectra and
vision-feature-space 2D spectra when the corresponding Exp 1 sidecars exist.
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
VARIANTS = {
    "radial_first_order": "predicted_first_order",
    "two_d_first_order": "predicted_2d_first_order",
}
VARIANT_LABELS = {
    "radial_first_order": "Radial first-order",
    "two_d_first_order": "2D first-order",
}
TARGET_LABELS = {
    "accuracy_drop": "Accuracy Drop",
    "loglik_erosion": "Log-Likelihood Erosion",
    "loglik_volatility": "Log-Likelihood Volatility",
    "net_drop": "Net Drop",
    "relative_accuracy_drop": "Relative Accuracy Drop",
}
COLORS = {
    "overall": "#525252",
    "early": "#3182bd",
    "mid": "#31a354",
    "late": "#de2d26",
    "last_2": "#9333ea",
    "radial_first_order": "#6b7280",
    "two_d_first_order": "#2563eb",
}
EPS = 1e-12
DOMAIN_SPECS = {
    "image_space": {
        "source": "image_space",
        "delta_dir": "delta_f_2d",
        "delta_file_key": "delta_f_2d_file",
        "delta_key_raw": "delta_power_2d",
        "delta_key_relative": "delta_power_2d_relative",
        "description": "image/pixel FFT perturbation spectrum",
    },
    "vision_feature_space": {
        "source": "vision_feature_space",
        "delta_dir": "delta_f_vision_2d",
        "delta_file_key": "delta_f_vision_2d_file",
        "delta_key_raw": "delta_power_vision_2d",
        "delta_key_relative": "delta_power_vision_2d_relative",
        "description": "vision-token feature-grid FFT perturbation spectrum",
    },
}


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


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned.strip("_") or "unknown"


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _perturbation_family_name(name: Any) -> str:
    text = str(name or "unknown")
    text = text.split("|sev", 1)[0]
    text = text.split("(", 1)[0]
    return text.strip() or "unknown"


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
    """Average ranks for ties, 1-based like scipy.stats.rankdata."""
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    sorted_values = arr[order]
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3 or y_arr.size < 3:
        return None
    if np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3 or y_arr.size < 3:
        return None
    return _pearson(_rankdata(x_arr), _rankdata(y_arr))


def _extract_xy(
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    y_key: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    xs: List[float] = []
    ys: List[float] = []
    kept: List[Dict[str, Any]] = []
    for row in rows:
        x_val = _as_float(row.get(x_key))
        y_val = _as_float(row.get(y_key))
        if x_val is None or y_val is None:
            continue
        xs.append(x_val)
        ys.append(y_val)
        kept.append(row)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), kept


def _summary_for_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    variant_name: str,
    predicted_key: str,
    target_key: str,
    group_name: str,
    level: Optional[str] = None,
    perturbation_family: Optional[str] = None,
) -> Dict[str, Any]:
    x, y, _ = _extract_xy(rows, predicted_key, target_key)
    return {
        "group": group_name,
        "variant": variant_name,
        "predicted_key": predicted_key,
        "target": target_key,
        "level": level,
        "perturbation_family": perturbation_family,
        "n": int(x.size),
        "pearson_r": _pearson(x, y),
        "spearman_rho": _spearman(x, y),
        "x_mean": float(np.mean(x)) if x.size else None,
        "x_std": float(np.std(x)) if x.size else None,
        "y_mean": float(np.mean(y)) if y.size else None,
        "y_std": float(np.std(y)) if y.size else None,
    }


def _sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _add_metadata(
    fig: plt.Figure,
    *,
    lines: Sequence[str],
) -> None:
    fig.text(
        0.01,
        0.01,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#374151",
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_npz_array(
    path: Path,
    key: str,
    cache: Dict[Tuple[str, str], Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    cache_key = (str(path), key)
    if cache_key in cache:
        return cache[cache_key]
    if not path.exists():
        cache[cache_key] = None
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            arr = np.asarray(data[key], dtype=np.float64)
    except Exception:
        arr = None
    cache[cache_key] = arr
    return arr


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


def _compute_2d_variants(W_t_2d: np.ndarray, delta_2d: np.ndarray) -> Dict[str, float]:
    w = np.asarray(W_t_2d, dtype=np.float64)
    d = np.asarray(delta_2d, dtype=np.float64)
    if w.ndim != 2 or d.ndim != 2 or w.size == 0 or d.size == 0:
        return {}
    if d.shape != w.shape:
        d = _resize_power_preserve_sum(d, (int(w.shape[0]), int(w.shape[1])))
    dot = float(np.sum(w * d))
    denom = float(np.linalg.norm(w) * np.linalg.norm(d))
    return {
        "predicted_2d_first_order": dot,
        "predicted_2d_first_order_recomputed": dot,
        "predicted_2d_linear": float(np.sum(w * d ** 2)),
        "predicted_2d_quadratic": float(np.sum((w ** 2) * (d ** 2))),
        "predicted_2d_cosine": float(dot / denom) if denom > EPS else 0.0,
        "predicted_2d_cosine_recomputed": float(dot / denom) if denom > EPS else 0.0,
    }


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _compute_grouped_2d_variant_rows(
    run_dir: Path,
    *,
    groups: Sequence[str],
    target_key: str,
    domain: str,
    delta_key: str,
) -> Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]]:
    """Recompute 2D linear/quadratic variants from Exp1/Exp2 sidecars."""
    spec = DOMAIN_SPECS[domain]
    exp1_jsonl = run_dir / "exp1" / "per_sample.jsonl"
    delta_dir = run_dir / "exp1" / str(spec["delta_dir"])
    filter_dir = run_dir / "exp2" / "filters_2d"
    if not exp1_jsonl.exists() or not delta_dir.exists() or not filter_dir.exists():
        return {}

    npz_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    accum: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]] = {
        group: defaultdict(lambda: defaultdict(list)) for group in groups
    }

    for record in _iter_jsonl(exp1_jsonl):
        image_id = str(record.get("image_id"))
        for level_key, level_data in (record.get("levels") or {}).items():
            w_by_group: Dict[str, Optional[np.ndarray]] = {}
            for group_name in groups:
                w_path = filter_dir / f"{image_id}_{level_key}_{group_name}.npz"
                w_by_group[group_name] = _load_npz_array(w_path, "W_t_2d", npz_cache)
            if not any(w is not None for w in w_by_group.values()):
                continue

            for perturbation in level_data.get("perturbations", []):
                delta_file = perturbation.get(str(spec["delta_file_key"]))
                target_value = _target_value(level_data, perturbation, target_key)
                if not delta_file or target_value is None:
                    continue
                delta_2d = _load_npz_array(delta_dir / str(delta_file), delta_key, npz_cache)
                if delta_2d is None:
                    continue
                perturbation_name = str(perturbation.get("name", "unknown"))
                family = _perturbation_family_name(perturbation_name)
                row_key = (image_id, str(level_key), family)
                for group_name, W_t_2d in w_by_group.items():
                    if W_t_2d is None:
                        continue
                    variants = _compute_2d_variants(W_t_2d, delta_2d)
                    if not variants:
                        continue
                    bucket = accum[group_name][row_key]
                    bucket["image_id"].append(image_id)
                    bucket["level"].append(str(level_key))
                    bucket["perturbation_family"].append(family)
                    bucket[target_key].append(float(target_value))
                    bucket["actual"].append(float(target_value))
                    bucket["predicted_2d_first_order"].append(
                        variants["predicted_2d_first_order"]
                    )
                    bucket["predicted_2d_first_order_recomputed"].append(
                        variants["predicted_2d_first_order_recomputed"]
                    )
                    bucket["predicted_2d_linear"].append(variants["predicted_2d_linear"])
                    bucket["predicted_2d_quadratic"].append(variants["predicted_2d_quadratic"])
                    bucket["predicted_2d_cosine"].append(variants["predicted_2d_cosine"])
                    bucket["predicted_2d_cosine_recomputed"].append(
                        variants["predicted_2d_cosine_recomputed"]
                    )

    grouped: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]] = {}
    for group_name, group_accum in accum.items():
        grouped[group_name] = {}
        for row_key, values in group_accum.items():
            image_id, level_key, family = row_key
            grouped[group_name][row_key] = {
                "image_id": image_id,
                "level": level_key,
                "perturbation": family,
                "perturbation_family": family,
                target_key: _mean_or_none(values[target_key]),
                "actual": _mean_or_none(values["actual"]),
                "predicted_2d_first_order": _mean_or_none(
                    values["predicted_2d_first_order"]
                ),
                "predicted_2d_first_order_recomputed": _mean_or_none(
                    values["predicted_2d_first_order_recomputed"]
                ),
                "predicted_2d_linear": _mean_or_none(values["predicted_2d_linear"]),
                "predicted_2d_quadratic": _mean_or_none(values["predicted_2d_quadratic"]),
                "predicted_2d_cosine": _mean_or_none(values["predicted_2d_cosine"]),
                "predicted_2d_cosine_recomputed": _mean_or_none(
                    values["predicted_2d_cosine_recomputed"]
                ),
                "n_2d_recomputed": len(values["actual"]),
                "overlap_2d_domain": domain,
                "overlap_2d_delta_key": delta_key,
            }
    return grouped


def _augment_grouped_rows_with_computed_2d(
    grouped_rows: Dict[str, List[Dict[str, Any]]],
    computed_rows: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]],
) -> None:
    """Merge recomputed 2D rows, including groups absent from Exp5 production."""
    for group_name, rows in grouped_rows.items():
        computed_group = computed_rows.get(group_name, {})
        for row in rows:
            key = (
                str(row.get("image_id")),
                str(row.get("level")),
                str(row.get("perturbation_family", row.get("perturbation", "unknown"))),
            )
            computed = computed_group.get(key)
            if computed:
                row.update(computed)
    for group_name, computed_group in computed_rows.items():
        existing_rows = grouped_rows.setdefault(group_name, [])
        existing_keys = {
            (
                str(row.get("image_id")),
                str(row.get("level")),
                str(row.get("perturbation_family", row.get("perturbation", "unknown"))),
            )
            for row in existing_rows
        }
        for row_key, computed in computed_group.items():
            if row_key in existing_keys:
                continue
            existing_rows.append(dict(computed))


def _plot_scatter(
    rows: Sequence[Dict[str, Any]],
    *,
    group_name: str,
    variant_name: str,
    predicted_key: str,
    target_key: str,
    out_path: Path,
    max_points: int,
    seed: int,
    log_x: bool,
) -> Dict[str, Any]:
    x, y, kept = _extract_xy(rows, predicted_key, target_key)
    summary = _summary_for_rows(
        kept,
        variant_name=variant_name,
        predicted_key=predicted_key,
        target_key=target_key,
        group_name=group_name,
    )
    if x.size < 3:
        return summary

    idx = _sample_indices(x.size, max_points, seed)
    x_plot = x[idx]
    y_plot = y[idx]
    level_plot = [str(kept[int(i)].get("level", "unknown")) for i in idx]

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for level in LEVEL_ORDER:
        mask = np.asarray([item == level for item in level_plot], dtype=bool)
        if not np.any(mask):
            continue
        ax.scatter(
            x_plot[mask],
            y_plot[mask],
            s=12,
            alpha=0.45,
            linewidths=0,
            label=level.replace("_", " "),
            rasterized=True,
        )
    if log_x and np.all(x_plot >= 0):
        ax.set_xscale("symlog", linthresh=max(float(np.percentile(x_plot, 5)), 1e-12))
    ax.set_xlabel(VARIANT_LABELS.get(variant_name, predicted_key))
    ax.set_ylabel(TARGET_LABELS.get(target_key, target_key))
    ax.set_title(
        f"Exp 5 2D Overlap Diagnostic: {VARIANT_LABELS.get(variant_name, variant_name)} ({group_name})"
    )
    ax.legend(fontsize=7, ncol=2, loc="best")
    _style_axis(ax)
    _add_metadata(
        fig,
        lines=[
            f"Target={target_key}",
            f"Group={group_name}",
            f"Variant={variant_name}",
            f"Pearson r={summary['pearson_r']:.3f}" if summary["pearson_r"] is not None else "Pearson r=NA",
            f"Spearman rho={summary['spearman_rho']:.3f}" if summary["spearman_rho"] is not None else "Spearman rho=NA",
            f"n={summary['n']}; plotted={len(idx)}",
            "Experimental script output; production plots unchanged.",
        ],
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return summary


def _plot_level_grid(
    rows: Sequence[Dict[str, Any]],
    *,
    group_name: str,
    variant_name: str,
    predicted_key: str,
    target_key: str,
    out_path: Path,
    max_points_per_level: int,
    seed: int,
    log_x: bool,
) -> List[Dict[str, Any]]:
    present_levels = [
        level for level in LEVEL_ORDER if any(str(row.get("level")) == level for row in rows)
    ]
    if not present_levels:
        return []
    ncols = 4
    nrows = int(math.ceil(len(present_levels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.0 * nrows), squeeze=False)
    summaries: List[Dict[str, Any]] = []
    for axis in axes.ravel()[len(present_levels):]:
        axis.axis("off")
    for panel_idx, (axis, level) in enumerate(zip(axes.ravel(), present_levels)):
        level_rows = [row for row in rows if str(row.get("level")) == level]
        x, y, kept = _extract_xy(level_rows, predicted_key, target_key)
        summary = _summary_for_rows(
            kept,
            variant_name=variant_name,
            predicted_key=predicted_key,
            target_key=target_key,
            group_name=group_name,
            level=level,
        )
        summaries.append(summary)
        if x.size >= 3:
            idx = _sample_indices(x.size, max_points_per_level, seed + panel_idx)
            families = [str(kept[int(i)].get("perturbation_family", "unknown")) for i in idx]
            unique_families = sorted(set(families))
            cmap = plt.get_cmap("tab20", max(1, len(unique_families)))
            for fam_idx, family in enumerate(unique_families):
                mask = np.asarray([item == family for item in families], dtype=bool)
                axis.scatter(
                    x[idx][mask],
                    y[idx][mask],
                    s=12,
                    alpha=0.42,
                    linewidths=0,
                    color=cmap(fam_idx),
                    rasterized=True,
                )
            if log_x and np.all(x[idx] >= 0):
                axis.set_xscale("symlog", linthresh=max(float(np.percentile(x[idx], 5)), 1e-12))
        axis.set_title(
            f"{level.replace('_', ' ')}\n"
            f"r={summary['pearson_r']:.3f}" if summary["pearson_r"] is not None else level,
            fontsize=10,
        )
        axis.set_xlabel(VARIANT_LABELS.get(variant_name, predicted_key), fontsize=9)
        axis.set_ylabel(TARGET_LABELS.get(target_key, target_key), fontsize=9)
        _style_axis(axis)
    fig.suptitle(
        f"Exp 5 2D Overlap by Level: {VARIANT_LABELS.get(variant_name, variant_name)} ({group_name})",
        fontsize=14,
    )
    _add_metadata(
        fig,
        lines=[
            f"Target={target_key}",
            f"Group={group_name}",
            f"Variant={variant_name}",
            "Panels=task levels; points=grouped perturbation families within image and level.",
            "Experimental script output; production plots unchanged.",
        ],
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return summaries


def _plot_correlation_bars(
    summaries: Sequence[Dict[str, Any]],
    *,
    target_key: str,
    out_path: Path,
) -> None:
    group_names = [group for group in GROUP_ORDER if any(s["group"] == group for s in summaries)]
    variant_names = [variant for variant in VARIANTS if any(s["variant"] == variant for s in summaries)]
    if not group_names or not variant_names:
        return

    x = np.arange(len(group_names), dtype=np.float64)
    width = 0.75 / max(1, len(variant_names))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx, variant_name in enumerate(variant_names):
        values = []
        for group_name in group_names:
            match = next(
                (
                    s
                    for s in summaries
                    if s["group"] == group_name
                    and s["variant"] == variant_name
                    and s.get("level") is None
                    and s.get("perturbation_family") is None
                ),
                None,
            )
            values.append(np.nan if match is None or match["pearson_r"] is None else match["pearson_r"])
        offset = (idx - (len(variant_names) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS.get(variant_name),
            alpha=0.86,
            label=VARIANT_LABELS.get(variant_name, variant_name),
        )
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([group.capitalize() for group in group_names])
    ax.set_ylabel(f"Pearson r vs {TARGET_LABELS.get(target_key, target_key)}")
    ax.set_title("Exp 5 Overlap Law Correlation: Radial vs 2D Variants")
    ax.legend(fontsize=9)
    _style_axis(ax)
    _add_metadata(
        fig,
        lines=[
            f"Target={target_key}",
            "Bars=Pearson r over grouped perturbation-family rows.",
            "Grouped rows=perturbation families averaged within image and level.",
            "Experimental script output; production plots unchanged.",
        ],
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_family_correlations(
    rows: Sequence[Dict[str, Any]],
    *,
    group_name: str,
    variant_name: str,
    predicted_key: str,
    target_key: str,
    out_path: Path,
    min_n: int,
) -> List[Dict[str, Any]]:
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("perturbation_family", "unknown"))].append(row)
    summaries: List[Dict[str, Any]] = []
    for family, family_rows in sorted(by_family.items()):
        if len(family_rows) < min_n:
            continue
        summaries.append(
            _summary_for_rows(
                family_rows,
                variant_name=variant_name,
                predicted_key=predicted_key,
                target_key=target_key,
                group_name=group_name,
                perturbation_family=family,
            )
        )
    summaries = [s for s in summaries if s["pearson_r"] is not None]
    if not summaries:
        return []
    summaries.sort(key=lambda item: float(item["pearson_r"]))
    labels = [str(item["perturbation_family"]) for item in summaries]
    values = [float(item["pearson_r"]) for item in summaries]
    fig_height = max(5.0, 0.28 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=COLORS.get(variant_name, "#2563eb"), alpha=0.84)
    ax.axvline(0.0, color="#111827", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Pearson r vs {TARGET_LABELS.get(target_key, target_key)}")
    ax.set_title(
        f"2D Overlap Correlation by Perturbation Family ({group_name}, {VARIANT_LABELS.get(variant_name, variant_name)})"
    )
    _style_axis(ax)
    _add_metadata(
        fig,
        lines=[
            f"Target={target_key}",
            f"Group={group_name}",
            f"Variant={variant_name}",
            f"Minimum n per family={min_n}",
            "Experimental script output; production plots unchanged.",
        ],
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return summaries


def _write_summary_files(
    out_dir: Path,
    summaries: Sequence[Dict[str, Any]],
    family_summaries: Sequence[Dict[str, Any]],
) -> None:
    payload = {
        "correlations": list(summaries),
        "perturbation_family_correlations": list(family_summaries),
    }
    with (out_dir / "exp5_overlap2d_plot_summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    with (out_dir / "exp5_overlap2d_correlations.csv").open("w", newline="") as handle:
        fieldnames = [
            "domain",
            "source",
            "delta_key",
            "group",
            "variant",
            "target",
            "level",
            "perturbation_family",
            "n",
            "pearson_r",
            "spearman_rho",
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in list(summaries) + list(family_summaries):
            writer.writerow({field: row.get(field) for field in fieldnames})


def _load_grouped_rows(exp5_dir: Path, source_name: str) -> Dict[str, List[Dict[str, Any]]]:
    by_group_path = exp5_dir / "scatter_data_by_group.json"
    if by_group_path.exists():
        data = _load_json(by_group_path)
        source_payload = data.get(source_name, {}) if isinstance(data, dict) else {}
        if isinstance(source_payload, dict):
            return {
                group: rows
                for group, rows in source_payload.items()
                if isinstance(rows, list)
            }
    fallback_path = exp5_dir / "scatter_data.json"
    if not fallback_path.exists():
        raise FileNotFoundError(f"No grouped Exp5 scatter data found under {exp5_dir}")
    return {"late": _load_json(fallback_path)}


def _resolve_domain_and_source(args: argparse.Namespace) -> Tuple[str, str]:
    domain = args.domain or args.source or "image_space"
    if domain not in DOMAIN_SPECS:
        raise ValueError(
            f"Unknown 2D overlap domain {domain!r}; expected one of {sorted(DOMAIN_SPECS)}"
        )
    source = args.source or str(DOMAIN_SPECS[domain]["source"])
    return domain, source


def _resolve_delta_key(domain: str, delta_key: str) -> str:
    spec = DOMAIN_SPECS[domain]
    key = str(delta_key or "relative")
    if key == "relative":
        return str(spec["delta_key_relative"])
    if key == "raw":
        return str(spec["delta_key_raw"])
    if key in {spec["delta_key_raw"], spec["delta_key_relative"]}:
        return key
    image_keys = {DOMAIN_SPECS["image_space"]["delta_key_raw"], DOMAIN_SPECS["image_space"]["delta_key_relative"]}
    vision_keys = {
        DOMAIN_SPECS["vision_feature_space"]["delta_key_raw"],
        DOMAIN_SPECS["vision_feature_space"]["delta_key_relative"],
    }
    if domain == "vision_feature_space" and key in image_keys:
        return str(spec["delta_key_relative"]) if key.endswith("_relative") else str(spec["delta_key_raw"])
    if domain == "image_space" and key in vision_keys:
        return str(spec["delta_key_relative"]) if key.endswith("_relative") else str(spec["delta_key_raw"])
    raise ValueError(f"Unknown delta key {delta_key!r} for domain {domain!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate experimental Exp5 overlap_2d plots from an existing run."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Frequency-alignment run output directory containing exp5/.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to <output-dir>/plots_overlap_2d_experimental.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        choices=sorted(DOMAIN_SPECS),
        help=(
            "2D sidecar domain to recompute: image_space uses exp1/delta_f_2d; "
            "vision_feature_space uses exp1/delta_f_vision_2d."
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Exp5 radial source key to load. Defaults to the matching source for --domain."
        ),
    )
    parser.add_argument(
        "--target",
        default="loglik_volatility",
        help="Observed target column, e.g. loglik_volatility or accuracy_drop.",
    )
    parser.add_argument(
        "--groups",
        default=",".join(GROUP_ORDER),
        help="Comma-separated filter groups to plot.",
    )
    parser.add_argument(
        "--variants",
        default="radial_first_order,two_d_first_order",
        help="Comma-separated variants to plot.",
    )
    parser.add_argument(
        "--delta-key",
        default="relative",
        choices=[
            "relative",
            "raw",
            "delta_power_2d",
            "delta_power_2d_relative",
            "delta_power_vision_2d",
            "delta_power_vision_2d_relative",
        ],
        help="Which saved 2D perturbation spectrum to use for recomputed variants.",
    )
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--max-points-per-level", type=int, default=1800)
    parser.add_argument("--min-family-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-log-x",
        action="store_true",
        help="Disable symlog x-axis for scatter plots.",
    )
    args = parser.parse_args()

    run_dir = args.output_dir
    exp5_dir = run_dir / "exp5"
    domain, source = _resolve_domain_and_source(args)
    delta_key = _resolve_delta_key(domain, args.delta_key)
    plots_dir = args.plots_dir or (run_dir / "plots_overlap_2d_experimental" / domain)
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped_rows = _load_grouped_rows(exp5_dir, source)
    groups = _parse_csv(args.groups, GROUP_ORDER)
    variants = _parse_csv(args.variants, list(VARIANTS))
    # The two_d_linear / two_d_quadratic / two_d_cosine variants were retired
    # in the camera-ready cleanup. The script no longer recomputes them; only
    # radial_first_order and two_d_first_order are produced.

    summaries: List[Dict[str, Any]] = []
    family_summaries: List[Dict[str, Any]] = []
    for group_name in groups:
        rows = grouped_rows.get(group_name)
        if not rows:
            continue
        for variant_name in variants:
            predicted_key = VARIANTS.get(variant_name, variant_name)
            if not any(row.get(predicted_key) is not None for row in rows):
                continue
            summaries.append(
                _plot_scatter(
                    rows,
                    group_name=group_name,
                    variant_name=variant_name,
                    predicted_key=predicted_key,
                    target_key=args.target,
                    out_path=plots_dir
                    / f"exp5_overlap2d_scatter_{_safe_name(domain)}_{_safe_name(variant_name)}_{_safe_name(group_name)}_{_safe_name(args.target)}.png",
                    max_points=args.max_points,
                    seed=args.seed,
                    log_x=not args.no_log_x,
                )
            )
            summaries.extend(
                _plot_level_grid(
                    rows,
                    group_name=group_name,
                    variant_name=variant_name,
                    predicted_key=predicted_key,
                    target_key=args.target,
                    out_path=plots_dir
                    / f"exp5_overlap2d_level_grid_{_safe_name(domain)}_{_safe_name(variant_name)}_{_safe_name(group_name)}_{_safe_name(args.target)}.png",
                    max_points_per_level=args.max_points_per_level,
                    seed=args.seed,
                    log_x=not args.no_log_x,
                )
            )
            if variant_name.startswith("two_d_"):
                family_summaries.extend(
                    _plot_family_correlations(
                        rows,
                        group_name=group_name,
                        variant_name=variant_name,
                        predicted_key=predicted_key,
                        target_key=args.target,
                        out_path=plots_dir
                        / f"exp5_overlap2d_family_correlations_{_safe_name(domain)}_{_safe_name(variant_name)}_{_safe_name(group_name)}_{_safe_name(args.target)}.png",
                        min_n=args.min_family_n,
                    )
                )

    _plot_correlation_bars(
        summaries,
        target_key=args.target,
        out_path=plots_dir
        / f"exp5_overlap2d_correlation_bars_{_safe_name(domain)}_{_safe_name(args.target)}.png",
    )
    for row in summaries:
        row["domain"] = domain
        row["source"] = source
        row["delta_key"] = delta_key
        row["domain_description"] = DOMAIN_SPECS[domain]["description"]
    for row in family_summaries:
        row["domain"] = domain
        row["source"] = source
        row["delta_key"] = delta_key
        row["domain_description"] = DOMAIN_SPECS[domain]["description"]
    _write_summary_files(plots_dir, summaries, family_summaries)

    print(f"Wrote overlap_2d experimental plots to {plots_dir}")
    print(f"Domain: {domain}  Source: {source}  Delta key: {delta_key}")
    print(f"Summary JSON: {plots_dir / 'exp5_overlap2d_plot_summary.json'}")


if __name__ == "__main__":
    main()
