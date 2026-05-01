#!/usr/bin/env python3
"""Aggregate Exp 2 2D filters to expose broad-vs-narrow frequency trends.

This is an experimental visualization script. It does not modify experiment
outputs or rerun inference. It reads saved ``exp2/filters_2d/*.npz`` files,
normalizes each 2D task filter, and writes shared-scale plots comparing the
eight query levels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
LEVEL_LABELS = {
    "L1_COARSE": "L1 Coarse",
    "L2_MEDIUM": "L2 Medium",
    "L3_FINE": "L3 Fine",
    "L4_VERY_FINE": "L4 Very Fine",
    "L5_WORDY_SIMPLETON": "L5 Wordy L1",
    "L6_WORDY_MEDIUM": "L6 Wordy L2",
    "L7_WORDY_FINE": "L7 Wordy L3",
    "L8_WORDY_VERY_FINE": "L8 Wordy L4",
}
LEVEL_COLORS = {
    "L1_COARSE": "#1f77b4",
    "L2_MEDIUM": "#ff7f0e",
    "L3_FINE": "#2ca02c",
    "L4_VERY_FINE": "#d62728",
    "L5_WORDY_SIMPLETON": "#1f77b4",
    "L6_WORDY_MEDIUM": "#ff7f0e",
    "L7_WORDY_FINE": "#2ca02c",
    "L8_WORDY_VERY_FINE": "#d62728",
}
EPS = 1e-12


def _load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def _parse_csv(value: Optional[str], default: Sequence[str]) -> List[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _safe_name(value: Any) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned.strip("_") or "unknown"


def _normalize_mass(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.maximum(data, 0.0)
    total = float(np.sum(data))
    if total <= EPS:
        return data
    return data / total


def _load_w_t_2d(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "W_t_2d" not in data:
                return None
            return _normalize_mass(np.asarray(data["W_t_2d"], dtype=np.float64))
    except Exception:
        return None


def _resize_2d(arr: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Resize for visualization on normalized coordinates, then renormalize."""

    data = _normalize_mass(arr)
    if data.ndim != 2 or data.size == 0:
        return np.zeros(output_shape, dtype=np.float64)
    out_h, out_w = output_shape
    h, w = data.shape
    old_x = np.linspace(-1.0, 1.0, w)
    old_y = np.linspace(-1.0, 1.0, h)
    new_x = np.linspace(-1.0, 1.0, out_w)
    new_y = np.linspace(-1.0, 1.0, out_h)

    x_resized = np.empty((h, out_w), dtype=np.float64)
    for row_idx in range(h):
        x_resized[row_idx] = np.interp(new_x, old_x, data[row_idx])

    resized = np.empty((out_h, out_w), dtype=np.float64)
    for col_idx in range(out_w):
        resized[:, col_idx] = np.interp(new_y, old_y, x_resized[:, col_idx])

    return _normalize_mass(resized)


def _radius_grid(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_radius = float(np.max(radius))
    if max_radius <= EPS:
        return np.zeros_like(radius, dtype=np.float64)
    return radius / max_radius


def _radial_profile(arr: np.ndarray, num_bins: int) -> np.ndarray:
    data = _normalize_mass(arr)
    if data.ndim != 2 or data.size == 0:
        return np.zeros(num_bins, dtype=np.float64)
    radius = _radius_grid(data.shape)
    edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
    profile = np.zeros(int(num_bins), dtype=np.float64)
    for idx in range(int(num_bins)):
        if idx == int(num_bins) - 1:
            mask = (radius >= edges[idx]) & (radius <= edges[idx + 1])
        else:
            mask = (radius >= edges[idx]) & (radius < edges[idx + 1])
        if np.any(mask):
            profile[idx] = float(np.sum(data[mask]))
    total = float(np.sum(profile))
    if total > EPS:
        profile /= total
    return profile


def _centroid(profile: np.ndarray) -> float:
    p = np.asarray(profile, dtype=np.float64)
    total = float(np.sum(p))
    if total <= EPS:
        return float("nan")
    centers = (np.arange(len(p), dtype=np.float64) + 0.5) / max(len(p), 1)
    return float(np.sum(centers * p) / total)


def _entropy(profile: np.ndarray) -> float:
    p = np.asarray(profile, dtype=np.float64)
    total = float(np.sum(p))
    if total <= EPS or len(p) <= 1:
        return float("nan")
    p = p / total
    valid = p > EPS
    return float(-np.sum(p[valid] * np.log(p[valid])) / math.log(len(p)))


def _sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[axis] <= 1:
        return np.zeros(arr.shape[1:], dtype=np.float64)
    return np.nanstd(arr, axis=axis, ddof=1) / math.sqrt(arr.shape[axis])


def _selected_records(
    records: Sequence[Dict[str, Any]],
    sample_ids: Sequence[str],
    max_samples: int,
) -> List[Dict[str, Any]]:
    if sample_ids:
        wanted = {str(item) for item in sample_ids}
        return [record for record in records if str(record.get("image_id")) in wanted]
    if int(max_samples) > 0:
        return list(records[: int(max_samples)])
    return list(records)


def _collect_filters(
    *,
    run_dir: Path,
    records: Sequence[Dict[str, Any]],
    levels: Sequence[str],
    group: str,
    radial_bins: int,
    canvas_size: int,
) -> Dict[str, Dict[str, Any]]:
    filters_dir = run_dir / "exp2" / "filters_2d"
    out: Dict[str, Dict[str, Any]] = {}
    for level in levels:
        out[level] = {
            "profiles": [],
            "resized": [],
            "sample_ids": [],
            "shapes": [],
            "missing": 0,
        }

    for record in records:
        sample_id = str(record.get("image_id"))
        available_levels = record.get("levels") or {}
        for level in levels:
            if level not in available_levels:
                out[level]["missing"] += 1
                continue
            path = filters_dir / f"{sample_id}_{level}_{group}.npz"
            w_t = _load_w_t_2d(path)
            if w_t is None:
                out[level]["missing"] += 1
                continue
            out[level]["profiles"].append(_radial_profile(w_t, radial_bins))
            out[level]["resized"].append(_resize_2d(w_t, (canvas_size, canvas_size)))
            out[level]["sample_ids"].append(sample_id)
            out[level]["shapes"].append([int(w_t.shape[0]), int(w_t.shape[1])])

    return out


def _mean_stack(items: Sequence[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    if not items:
        return np.zeros(shape, dtype=np.float64)
    return np.mean(np.stack(items, axis=0), axis=0)


def _plot_mean_2d_grid(
    data: Dict[str, Dict[str, Any]],
    levels: Sequence[str],
    group: str,
    out_path: Path,
) -> None:
    mean_maps = {
        level: _mean_stack(data[level]["resized"], (64, 64))
        for level in levels
    }
    vmax = max(float(np.max(arr)) for arr in mean_maps.values()) if mean_maps else 0.0
    vmax = max(vmax, EPS)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), constrained_layout=True)
    for ax, level in zip(axes.ravel(), levels):
        arr = mean_maps[level]
        image = ax.imshow(
            arr,
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
            extent=(-1, 1, -1, 1),
        )
        ax.set_title(f"{LEVEL_LABELS.get(level, level)}\nn={len(data[level]['profiles'])}")
        ax.set_xlabel("normalized f_x")
        ax.set_ylabel("normalized f_y")
        ax.axhline(0.0, color="white", alpha=0.22, linewidth=0.7)
        ax.axvline(0.0, color="white", alpha=0.22, linewidth=0.7)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015)
    cbar.set_label("Mean normalized W_t(u,v), shared scale")
    fig.suptitle(
        f"Mean 2D task-frequency filters by query level ({group})",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_radial_profiles(
    data: Dict[str, Dict[str, Any]],
    levels: Sequence[str],
    group: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    x = None
    for level in levels:
        profiles = data[level]["profiles"]
        if not profiles:
            continue
        arr = np.stack(profiles, axis=0)
        mean = np.mean(arr, axis=0)
        err = _sem(arr, axis=0)
        x = (np.arange(len(mean), dtype=np.float64) + 0.5) / len(mean)
        linestyle = "-" if level.startswith(("L1", "L2", "L3", "L4")) else "--"
        ax.plot(
            x,
            mean,
            label=f"{LEVEL_LABELS.get(level, level)} (n={arr.shape[0]})",
            color=LEVEL_COLORS.get(level),
            linestyle=linestyle,
            linewidth=2.1,
        )
        ax.fill_between(
            x,
            mean - err,
            mean + err,
            color=LEVEL_COLORS.get(level),
            alpha=0.10,
            linewidth=0,
        )

    ax.set_title(f"Normalized radial frequency profiles by level ({group})")
    ax.set_xlabel("Normalized radial frequency: center/DC -> edge/Nyquist")
    ax.set_ylabel("Mean W_t mass per radial bin")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(ncol=2, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        -0.18,
        "Interpretation: broader filters put more mass at larger radii; narrower filters concentrate mass near center/mid frequencies. "
        "Profiles are normalized per sample before averaging, so this is a shape comparison.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_radial_cdf(
    data: Dict[str, Dict[str, Any]],
    levels: Sequence[str],
    group: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    for level in levels:
        profiles = data[level]["profiles"]
        if not profiles:
            continue
        arr = np.stack(profiles, axis=0)
        cdf = np.cumsum(arr, axis=1)
        mean = np.mean(cdf, axis=0)
        err = _sem(cdf, axis=0)
        x = (np.arange(len(mean), dtype=np.float64) + 0.5) / len(mean)
        linestyle = "-" if level.startswith(("L1", "L2", "L3", "L4")) else "--"
        ax.plot(
            x,
            mean,
            label=LEVEL_LABELS.get(level, level),
            color=LEVEL_COLORS.get(level),
            linestyle=linestyle,
            linewidth=2.1,
        )
        ax.fill_between(
            x,
            mean - err,
            mean + err,
            color=LEVEL_COLORS.get(level),
            alpha=0.10,
            linewidth=0,
        )

    ax.set_title(f"Cumulative radial mass by level ({group})")
    ax.set_xlabel("Normalized radial frequency threshold")
    ax.set_ylabel("Fraction of W_t mass inside threshold")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(ncol=2, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        -0.18,
        "Interpretation: a faster-rising curve means the level concentrates more filter mass at lower/mid frequencies.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_band_masses(
    data: Dict[str, Dict[str, Any]],
    levels: Sequence[str],
    group: str,
    out_path: Path,
) -> None:
    labels = ["center-low\n0-0.33", "mid\n0.33-0.66", "high\n0.66-1.0"]
    values = []
    errors = []
    for level in levels:
        profiles = data[level]["profiles"]
        if not profiles:
            values.append([0.0, 0.0, 0.0])
            errors.append([0.0, 0.0, 0.0])
            continue
        arr = np.stack(profiles, axis=0)
        n_bins = arr.shape[1]
        centers = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
        bands = [
            np.sum(arr[:, centers < 1.0 / 3.0], axis=1),
            np.sum(arr[:, (centers >= 1.0 / 3.0) & (centers < 2.0 / 3.0)], axis=1),
            np.sum(arr[:, centers >= 2.0 / 3.0], axis=1),
        ]
        values.append([float(np.mean(band)) for band in bands])
        errors.append([float(np.std(band, ddof=1) / math.sqrt(len(band))) if len(band) > 1 else 0.0 for band in bands])

    values_arr = np.asarray(values, dtype=np.float64)
    errors_arr = np.asarray(errors, dtype=np.float64)
    x = np.arange(len(levels), dtype=np.float64)
    width = 0.25

    fig, ax = plt.subplots(figsize=(13.4, 6.8))
    colors = ["#3b82f6", "#f59e0b", "#ef4444"]
    for idx, label in enumerate(labels):
        ax.bar(
            x + (idx - 1) * width,
            values_arr[:, idx],
            width=width,
            yerr=errors_arr[:, idx],
            label=label,
            color=colors[idx],
            alpha=0.82,
            capsize=2,
        )
    ax.set_title(f"Frequency-band mass by query level ({group})")
    ax.set_ylabel("Mean normalized W_t mass")
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS.get(level, level) for level in levels], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_broadness_metrics(
    data: Dict[str, Dict[str, Any]],
    levels: Sequence[str],
    group: str,
    out_path: Path,
) -> None:
    metrics = defaultdict(list)
    for level in levels:
        for profile in data[level]["profiles"]:
            metrics[(level, "centroid")].append(_centroid(profile))
            metrics[(level, "entropy")].append(_entropy(profile))

    x = np.arange(len(levels), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.9), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "centroid", "Spectral centroid (0=center, 1=edge)"),
        (axes[1], "entropy", "Normalized radial entropy"),
    ]:
        means = []
        errs = []
        for level in levels:
            vals = np.asarray(metrics[(level, metric)], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if len(vals) else 0.0)
            errs.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        ax.errorbar(x, means, yerr=errs, marker="o", linewidth=2.0, capsize=3, color="#111827")
        ax.set_title(metric.capitalize())
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_LABELS.get(level, level) for level in levels], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Broadness diagnostics for 2D task filters ({group})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_contrast_maps(
    data: Dict[str, Dict[str, Any]],
    group: str,
    out_path: Path,
) -> None:
    pairs = [
        ("L1_COARSE", "L4_VERY_FINE", "Primary: L1 - L4"),
        ("L5_WORDY_SIMPLETON", "L8_WORDY_VERY_FINE", "Wordy mirror: L5 - L8"),
        ("L1_COARSE", "L5_WORDY_SIMPLETON", "Length effect: L1 - L5"),
        ("L4_VERY_FINE", "L8_WORDY_VERY_FINE", "Length effect: L4 - L8"),
    ]
    diffs = []
    for left, right, _ in pairs:
        left_mean = _mean_stack(data.get(left, {}).get("resized", []), (64, 64))
        right_mean = _mean_stack(data.get(right, {}).get("resized", []), (64, 64))
        diffs.append(left_mean - right_mean)
    max_abs = max(float(np.max(np.abs(diff))) for diff in diffs) if diffs else EPS
    max_abs = max(max_abs, EPS)

    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.8), constrained_layout=True)
    for ax, diff, (_, _, title) in zip(axes, diffs, pairs):
        image = ax.imshow(
            diff,
            origin="lower",
            cmap="coolwarm",
            interpolation="nearest",
            vmin=-max_abs,
            vmax=max_abs,
            extent=(-1, 1, -1, 1),
        )
        ax.set_title(title)
        ax.set_xlabel("normalized f_x")
        ax.set_ylabel("normalized f_y")
        ax.axhline(0.0, color="black", alpha=0.22, linewidth=0.7)
        ax.axvline(0.0, color="black", alpha=0.22, linewidth=0.7)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.020, pad=0.018)
    cbar.set_label("Difference in mean normalized W_t")
    fig.suptitle(
        f"Direct 2D filter contrasts ({group}); red = first level has more mass",
        fontsize=13.5,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_summary(
    data: Dict[str, Dict[str, Any]],
    levels: Sequence[str],
    group: str,
    out_path: Path,
) -> None:
    summary: Dict[str, Any] = {
        "group": group,
        "levels": {},
        "interpretation": {
            "spectral_centroid": "Higher means more mass at larger normalized radial frequencies.",
            "radial_entropy": "Higher means a broader radial spread.",
            "band_masses": "center_low/mid/high are thirds of normalized radial frequency.",
        },
    }
    for level in levels:
        profiles = data[level]["profiles"]
        level_summary: Dict[str, Any] = {
            "n": len(profiles),
            "missing": int(data[level]["missing"]),
            "source_shapes": data[level]["shapes"][:20],
        }
        if profiles:
            arr = np.stack(profiles, axis=0)
            centers = (np.arange(arr.shape[1], dtype=np.float64) + 0.5) / arr.shape[1]
            level_summary.update(
                {
                    "mean_radial_profile": np.mean(arr, axis=0).tolist(),
                    "sem_radial_profile": _sem(arr, axis=0).tolist(),
                    "mean_cdf": np.mean(np.cumsum(arr, axis=1), axis=0).tolist(),
                    "mean_spectral_centroid": float(np.mean([_centroid(p) for p in profiles])),
                    "mean_radial_entropy": float(np.mean([_entropy(p) for p in profiles])),
                    "center_low_mass": float(np.mean(np.sum(arr[:, centers < 1.0 / 3.0], axis=1))),
                    "mid_mass": float(
                        np.mean(
                            np.sum(
                                arr[:, (centers >= 1.0 / 3.0) & (centers < 2.0 / 3.0)],
                                axis=1,
                            )
                        )
                    ),
                    "high_mass": float(np.mean(np.sum(arr[:, centers >= 2.0 / 3.0], axis=1))),
                }
            )
        summary["levels"][level] = level_summary
    with out_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot aggregate level-wise 2D frequency-filter contrasts from Exp2 outputs."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Run output directory containing exp2/power_spectra.json and exp2/filters_2d/.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory. Defaults to "
            "<output-dir>/plots_overlap_2d_experimental/exp2_level_frequency_contrast."
        ),
    )
    parser.add_argument(
        "--groups",
        default="late",
        help="Comma-separated attention groups to aggregate, e.g. overall,early,mid,late.",
    )
    parser.add_argument(
        "--levels",
        default=",".join(LEVEL_ORDER),
        help="Comma-separated query levels to include.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Cap number of samples; 0 means all available samples.",
    )
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Optional comma-separated image IDs. Overrides --num-samples.",
    )
    parser.add_argument("--radial-bins", type=int, default=24)
    parser.add_argument("--canvas-size", type=int, default=64)
    args = parser.parse_args()

    run_dir = args.output_dir
    power_path = run_dir / "exp2" / "power_spectra.json"
    filters_dir = run_dir / "exp2" / "filters_2d"
    if not power_path.exists():
        raise FileNotFoundError(f"Missing {power_path}")
    if not filters_dir.exists():
        raise FileNotFoundError(f"Missing {filters_dir}")

    records = _load_json(power_path)
    selected = _selected_records(
        records=records,
        sample_ids=_parse_csv(args.sample_ids, []),
        max_samples=args.num_samples,
    )
    levels = _parse_csv(args.levels, LEVEL_ORDER)
    groups = _parse_csv(args.groups, ["late"])
    plots_dir = args.plots_dir or (
        run_dir / "plots_overlap_2d_experimental" / "exp2_level_frequency_contrast"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    total_plots = 0
    for group in groups:
        data = _collect_filters(
            run_dir=run_dir,
            records=selected,
            levels=levels,
            group=group,
            radial_bins=args.radial_bins,
            canvas_size=args.canvas_size,
        )
        suffix = _safe_name(group)
        _plot_mean_2d_grid(
            data,
            levels,
            group,
            plots_dir / f"exp2_mean_2d_filters_by_level_{suffix}.png",
        )
        _plot_radial_profiles(
            data,
            levels,
            group,
            plots_dir / f"exp2_radial_profiles_by_level_{suffix}.png",
        )
        _plot_radial_cdf(
            data,
            levels,
            group,
            plots_dir / f"exp2_radial_cdf_by_level_{suffix}.png",
        )
        _plot_band_masses(
            data,
            levels,
            group,
            plots_dir / f"exp2_band_masses_by_level_{suffix}.png",
        )
        _plot_broadness_metrics(
            data,
            levels,
            group,
            plots_dir / f"exp2_broadness_metrics_by_level_{suffix}.png",
        )
        _plot_contrast_maps(
            data,
            group,
            plots_dir / f"exp2_2d_filter_contrasts_{suffix}.png",
        )
        _write_summary(
            data,
            levels,
            group,
            plots_dir / f"exp2_level_frequency_contrast_summary_{suffix}.json",
        )
        total_plots += 6

    print(f"Wrote {total_plots} aggregate Exp2 frequency contrast plots to {plots_dir}")
    print(f"Groups: {', '.join(groups)}")
    print(f"Samples used: {len(selected)}")


if __name__ == "__main__":
    main()
