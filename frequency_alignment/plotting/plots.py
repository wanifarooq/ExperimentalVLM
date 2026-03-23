from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..perturbations.frequency_sweep import compute_critical_cutoff

logger = logging.getLogger(__name__)

_LEVEL_COLORS = {
    "L1_COARSE": "#1f77b4",
    "L2_MEDIUM": "#2ca02c",
    "L3_FINE": "#ff7f0e",
    "L4_VERY_FINE": "#d62728",
}
_LEVEL_LABELS = {
    "L1_COARSE": "L1 (Coarse)",
    "L2_MEDIUM": "L2 (Medium)",
    "L3_FINE": "L3 (Fine)",
    "L4_VERY_FINE": "L4 (Very Fine)",
}
_LEVEL_ORDER = ["L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE"]
_SEG_LEVEL_ORDER = ["L1_COARSE", "L2_MEDIUM", "L3_FINE"]


def _apply_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 250) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ordered_levels(keys: Iterable[str], segmentation: bool = False) -> List[str]:
    order = _SEG_LEVEL_ORDER if segmentation else _LEVEL_ORDER
    key_set = set(keys)
    return [level for level in order if level in key_set]


def _level_label(level_key: str) -> str:
    return _LEVEL_LABELS.get(level_key, level_key)


def _short_sample_id(sample_id: str, max_chars: int = 18) -> str:
    sample_id = str(sample_id)
    if len(sample_id) <= max_chars:
        return sample_id
    return f"{sample_id[:8]}...{sample_id[-7:]}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def _sort_metric_rows(
    row_labels: Sequence[str],
    matrix: np.ndarray,
    max_rows: int = 10,
) -> Tuple[List[str], np.ndarray]:
    if matrix.size == 0:
        return list(row_labels), matrix
    scores = np.nanmean(matrix, axis=1)
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    order = order[: min(max_rows, len(order))]
    return [row_labels[idx] for idx in order], matrix[order]


def _plot_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    annotate: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    if matrix.size == 0 or not row_labels or not col_labels:
        return

    fig_width = max(8.0, 2.5 + 0.55 * len(col_labels))
    fig_height = max(4.5, 2.0 + 0.45 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=13)
    _apply_style(ax)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(colorbar_label, fontsize=10)

    if annotate and len(row_labels) <= 12 and len(col_labels) <= 16:
        finite = matrix[np.isfinite(matrix)]
        mean_value = float(np.mean(finite)) if finite.size else 0.0
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if np.isnan(value):
                    continue
                color = "white" if np.nan_to_num(value, nan=0.0) > mean_value else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    _save_fig(fig, out_path)


def _plot_grouped_bars(
    series: Dict[str, Sequence[float]],
    x_labels: Sequence[str],
    out_path: Path,
    title: str,
    ylabel: str,
    colors: Optional[Sequence[str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    if not series or not x_labels:
        return

    fig, ax = plt.subplots(figsize=(max(8.0, 2.0 + len(x_labels) * 1.1), 5.2))
    x = np.arange(len(x_labels))
    n_series = len(series)
    width = 0.8 / max(1, n_series)

    for idx, (label, values) in enumerate(series.items()):
        offset = (idx - (n_series - 1) / 2) * width
        color = colors[idx] if colors and idx < len(colors) else None
        ax.bar(x + offset, values, width=width * 0.92, label=label, alpha=0.85, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9)
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def _plot_distribution_with_points(
    grouped_values: Dict[str, Sequence[float]],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    present = _ordered_levels(grouped_values.keys())
    if not present:
        return

    datasets = [list(grouped_values[level]) for level in present]
    if not any(datasets):
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    box = ax.boxplot(datasets, patch_artist=True, widths=0.55, showfliers=False)
    for patch, level in zip(box["boxes"], present):
        patch.set_facecolor(_LEVEL_COLORS.get(level, "#999"))
        patch.set_alpha(0.45)

    rng = np.random.default_rng(42)
    for idx, level in enumerate(present):
        values = np.asarray(grouped_values[level], dtype=float)
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        ax.scatter(
            np.full(values.size, idx + 1) + jitter,
            values,
            s=18,
            color=_LEVEL_COLORS.get(level, "#999"),
            alpha=0.65,
            edgecolors="white",
            linewidth=0.4,
        )

    ax.set_xticks(np.arange(1, len(present) + 1))
    ax.set_xticklabels([_level_label(level) for level in present], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def _plot_sample_profile_grid(
    sample_id: str,
    per_level_series: Dict[str, Dict[str, Sequence[float]]],
    out_path: Path,
    title: str,
    ylabel: str,
    overlay_by_level: Optional[Dict[str, Sequence[float]]] = None,
) -> None:
    present_levels = _ordered_levels(per_level_series.keys())
    if not present_levels:
        return

    ncols = 2
    nrows = int(np.ceil(len(present_levels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.1 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for axis in axes_arr.ravel()[len(present_levels):]:
        axis.axis("off")

    for axis, level in zip(axes_arr.ravel(), present_levels):
        series = per_level_series[level]
        if not series:
            axis.axis("off")
            continue

        labels = list(series.keys())
        cmap = plt.cm.get_cmap("tab20", max(1, len(labels)))
        for idx, label in enumerate(labels):
            values = np.asarray(series[label], dtype=float)
            axis.plot(
                np.arange(len(values)),
                values,
                linewidth=1.5,
                alpha=0.9,
                label=label,
                color=cmap(idx),
            )

        overlay = None if overlay_by_level is None else overlay_by_level.get(level)
        if overlay is not None:
            overlay_values = np.asarray(overlay, dtype=float)
            axis.plot(
                np.arange(len(overlay_values)),
                overlay_values,
                linewidth=2.4,
                linestyle="--",
                color="black",
                alpha=0.85,
                label="W_t",
            )

        axis.set_title(_level_label(level), fontsize=11)
        axis.set_xlabel("Frequency Band", fontsize=10)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.grid(alpha=0.2, linewidth=0.6)
        _apply_style(axis)

    handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle(f"{title}: {_short_sample_id(sample_id, 32)}", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    _save_fig(fig, out_path)


def plot_granularity_curves(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Accuracy Degradation vs Task Granularity",
) -> None:
    per_level = summary.get("per_level", {})
    present = _ordered_levels(per_level.keys())
    if not present:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(present))
    width = 0.36

    clean_acc = [per_level[level]["clean_accuracy"] for level in present]
    pert_acc = [per_level[level]["perturbed_accuracy"] for level in present]
    axes[0].bar(x - width / 2, clean_acc, width, color="#2ca02c", alpha=0.85, label="Clean")
    axes[0].bar(x + width / 2, pert_acc, width, color="#d62728", alpha=0.85, label="Perturbed")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_level_label(level) for level in present], fontsize=9)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Clean vs Perturbed Accuracy", fontsize=12)
    axes[0].legend(fontsize=9)
    _apply_style(axes[0])

    drops = [per_level[level]["mean_accuracy_drop"] for level in present]
    stds = [per_level[level]["std_accuracy_drop"] for level in present]
    axes[1].bar(
        x,
        drops,
        yerr=stds,
        capsize=4,
        color=[_LEVEL_COLORS.get(level, "#999") for level in present],
        alpha=0.9,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_level_label(level) for level in present], fontsize=9)
    axes[1].set_ylabel("Mean Accuracy Drop", fontsize=11)
    axes[1].set_title("Degradation by Level", fontsize=12)
    _apply_style(axes[1])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_attention_power_spectrum(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Attention Power Spectrum by Granularity",
) -> None:
    per_level = summary.get("per_level", {})
    present = _ordered_levels(per_level.keys())
    if not present:
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for level in present:
        values = per_level[level].get("W_t_average", [])
        if not values:
            continue
        ax.plot(
            np.arange(len(values)),
            values,
            marker="o",
            linewidth=2.0,
            color=_LEVEL_COLORS.get(level, "#999"),
            label=_level_label(level),
        )
    ax.set_xlabel("Frequency Band (low to high)", fontsize=11)
    ax.set_ylabel("Normalized Attention Power", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_effective_bandwidth(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Effective Bandwidth by Task Granularity",
) -> None:
    per_level = summary.get("per_level", {})
    present = _ordered_levels(per_level.keys())
    if not present:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(present))
    means = [per_level[level]["mean_bandwidth"] for level in present]
    stds = [per_level[level]["std_bandwidth"] for level in present]
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[_LEVEL_COLORS.get(level, "#999") for level in present],
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_level_label(level) for level in present], fontsize=10)
    ax.set_ylabel("Effective Bandwidth", fontsize=11)
    ax.set_title(title, fontsize=13)
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_amplification_heatmap(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Drift Amplification Ratio by Level",
) -> None:
    per_level = summary.get("per_level", {})
    present = _ordered_levels(per_level.keys())
    if not present:
        return

    matrix = []
    for level in present:
        values = per_level[level].get("mean_amplification_ratio", [])
        if values:
            matrix.append(values)
    if not matrix:
        return

    _plot_heatmap(
        np.asarray(matrix, dtype=float),
        [_level_label(level) for level in present],
        [f"B{i}" for i in range(len(matrix[0]))],
        out_path,
        title,
        "Amplification Ratio",
        cmap="YlOrRd",
    )


def plot_frequency_threshold_curves(
    accuracy_data: Dict[str, Any],
    out_path: Path,
    mode: str = "lowpass",
    title: str = "Accuracy vs Frequency Cutoff",
) -> None:
    mode_data = accuracy_data.get("data", {}).get(mode, {})
    cutoffs = accuracy_data.get("cutoffs", [])
    present = _ordered_levels(mode_data.keys())
    if not present or not cutoffs:
        return

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    for level in present:
        curve = mode_data[level].get("accuracy_curve", [])
        if not curve:
            continue
        ax.plot(
            cutoffs[: len(curve)],
            curve,
            marker="o",
            markersize=4,
            linewidth=2,
            color=_LEVEL_COLORS.get(level, "#999"),
            label=_level_label(level),
        )
        omega_c = mode_data[level].get("critical_cutoff")
        if omega_c is not None:
            ax.axvline(
                omega_c,
                color=_LEVEL_COLORS.get(level, "#999"),
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
            )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="50% threshold")
    ax.set_xlabel(f"{mode.capitalize()} cutoff", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title} ({mode})", fontsize=13)
    ax.legend(fontsize=9)
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_overlap_scatter(
    scatter_data: List[Dict[str, Any]],
    pearson_r: float,
    out_path: Path,
    title: str = "Predicted vs Actual Sensitivity",
    point_size: float = 40,
    alpha: float = 0.7,
) -> None:
    if not scatter_data:
        return

    fig, ax = plt.subplots(figsize=(7.1, 6.0))
    for item in scatter_data:
        level = item.get("level")
        if level is None:
            label = item.get("label", "")
            level = label.split("|", 1)[0] if "|" in label else "unknown"
        ax.scatter(
            item["predicted"],
            item["actual"],
            s=point_size,
            alpha=alpha,
            color=_LEVEL_COLORS.get(level, "#999"),
            edgecolors="white",
            linewidth=0.5,
        )

    preds = np.asarray([item["predicted"] for item in scatter_data], dtype=float)
    actuals = np.asarray([item["actual"] for item in scatter_data], dtype=float)
    if len(preds) > 2 and np.unique(preds).size > 1:
        fit = np.polyfit(preds, actuals, 1)
        curve = np.poly1d(fit)
        xs = np.linspace(preds.min(), preds.max(), 100)
        ax.plot(xs, curve(xs), "k--", linewidth=1.5, alpha=0.65)

    ax.set_xlabel("Predicted Sensitivity (spectral overlap)", fontsize=11)
    ax.set_ylabel("Observed Accuracy Drop", fontsize=11)
    ax.set_title(f"{title} (r = {pearson_r:.3f})", fontsize=13)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=_level_label(level), markerfacecolor=color, markersize=8)
        for level, color in _LEVEL_COLORS.items()
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_segmentation_granularity(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Segmentation Degradation by Hierarchy Level",
) -> None:
    per_model = summary.get("per_model", {})
    if not per_model:
        return

    models = list(per_model)
    levels = _SEG_LEVEL_ORDER
    x = np.arange(len(levels))
    width = 0.8 / max(1, len(models))
    colors = {"sam3": "#d62728", "sam2": "#1f77b4"}

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for idx, model_name in enumerate(models):
        values = [per_model[model_name].get(level, {}).get("mean_drop", 0.0) for level in levels]
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            label=model_name.upper(),
            color=colors.get(model_name, f"C{idx}"),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_level_label(level) for level in levels], fontsize=10)
    ax.set_ylabel("Mean mIoU Drop", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def _exp1_level_perturbation_matrix(detail: Dict[str, Any]) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _ordered_levels(detail.keys())
    perturbations = sorted({name for level in levels for name in detail.get(level, {})})
    matrix = np.full((len(levels), len(perturbations)), np.nan, dtype=float)
    for row, level in enumerate(levels):
        for col, perturbation in enumerate(perturbations):
            stats = detail.get(level, {}).get(perturbation)
            if stats:
                matrix[row, col] = float(stats.get("mean_drop", np.nan))
    return levels, perturbations, matrix


def _exp1_sample_level_matrix(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _LEVEL_ORDER
    sample_ids: List[str] = []
    rows: List[List[float]] = []
    for record in records:
        row: List[float] = []
        for level in levels:
            perturbations = record.get("levels", {}).get(level, {}).get("perturbations", [])
            values = [float(item.get("accuracy_drop", 0.0)) for item in perturbations]
            row.append(float(np.mean(values)) if values else np.nan)
        if not np.all(np.isnan(row)):
            sample_ids.append(str(record.get("image_id")))
            rows.append(row)
    if not rows:
        return [], [], np.empty((0, 0))
    return sample_ids, levels, np.asarray(rows, dtype=float)


def _select_exp1_samples(records: List[Dict[str, Any]], max_samples: int = 3) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        values = []
        for level_data in record.get("levels", {}).values():
            values.extend(float(item.get("accuracy_drop", 0.0)) for item in level_data.get("perturbations", []))
        if values:
            scored.append((float(np.mean(values)), record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:max_samples]]


def _exp1_sample_spectra(record: Dict[str, Any], key: str) -> Dict[str, Dict[str, Sequence[float]]]:
    spectra: Dict[str, Dict[str, Sequence[float]]] = {}
    for level in _ordered_levels(record.get("levels", {}).keys()):
        per_perturbation: Dict[str, Sequence[float]] = {}
        for perturbation in record.get("levels", {}).get(level, {}).get("perturbations", []):
            values = perturbation.get(key)
            if values:
                per_perturbation[str(perturbation.get("name", "unknown"))] = values
        if per_perturbation:
            spectra[level] = per_perturbation
    return spectra


def _exp2_bandwidth_values(records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {level: [] for level in _LEVEL_ORDER}
    for record in records:
        for level, level_data in record.get("levels", {}).items():
            bandwidth = level_data.get("bandwidth")
            if bandwidth is not None:
                values.setdefault(level, []).append(float(bandwidth))
    return {level: items for level, items in values.items() if items}


def _exp2_sample_level_matrix(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], np.ndarray]:
    sample_ids: List[str] = []
    rows: List[List[float]] = []
    for record in records:
        row = []
        for level in _LEVEL_ORDER:
            bandwidth = record.get("levels", {}).get(level, {}).get("bandwidth")
            row.append(float(bandwidth) if bandwidth is not None else np.nan)
        if not np.all(np.isnan(row)):
            sample_ids.append(str(record.get("image_id")))
            rows.append(row)
    if not rows:
        return [], [], np.empty((0, 0))
    return sample_ids, _LEVEL_ORDER, np.asarray(rows, dtype=float)


def _select_exp2_samples(records: List[Dict[str, Any]], max_samples: int = 3) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        values = [float(level_data.get("bandwidth")) for level_data in record.get("levels", {}).values() if level_data.get("bandwidth") is not None]
        if values:
            scored.append((float(np.mean(values)), record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:max_samples]]


def _exp3_level_perturbation_matrix(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _LEVEL_ORDER
    perturbations = sorted(
        {
            str(item.get("perturbation"))
            for record in records
            for level_data in record.get("levels", {}).values()
            for item in level_data.get("perturbations", [])
        }
    )
    matrix = np.full((len(levels), len(perturbations)), np.nan, dtype=float)
    for row, level in enumerate(levels):
        for col, perturbation in enumerate(perturbations):
            values = []
            for record in records:
                for item in record.get("levels", {}).get(level, {}).get("perturbations", []):
                    if item.get("perturbation") == perturbation:
                        values.append(float(item.get("mean_amplification", 0.0)))
            if values:
                matrix[row, col] = float(np.mean(values))
    valid_rows = [idx for idx in range(len(levels)) if not np.all(np.isnan(matrix[idx]))]
    valid_levels = [levels[idx] for idx in valid_rows]
    return valid_levels, perturbations, matrix[valid_rows]


def _exp3_sample_level_matrix(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], np.ndarray]:
    sample_ids: List[str] = []
    rows: List[List[float]] = []
    for record in records:
        row = []
        for level in _LEVEL_ORDER:
            values = [
                float(item.get("mean_amplification", 0.0))
                for item in record.get("levels", {}).get(level, {}).get("perturbations", [])
            ]
            row.append(float(np.mean(values)) if values else np.nan)
        if not np.all(np.isnan(row)):
            sample_ids.append(str(record.get("image_id")))
            rows.append(row)
    if not rows:
        return [], [], np.empty((0, 0))
    return sample_ids, _LEVEL_ORDER, np.asarray(rows, dtype=float)


def _select_exp3_samples(records: List[Dict[str, Any]], max_samples: int = 2) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        values = []
        for level_data in record.get("levels", {}).values():
            values.extend(float(item.get("mean_amplification", 0.0)) for item in level_data.get("perturbations", []))
        if values:
            scored.append((float(np.mean(values)), record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:max_samples]]


def _exp3_sample_profiles(record: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Sequence[float]]], Dict[str, Sequence[float]]]:
    profiles: Dict[str, Dict[str, Sequence[float]]] = {}
    overlays: Dict[str, Sequence[float]] = {}
    for level in _ordered_levels(record.get("levels", {}).keys()):
        per_perturbation: Dict[str, Sequence[float]] = {}
        overlay: Optional[Sequence[float]] = None
        for item in record.get("levels", {}).get(level, {}).get("perturbations", []):
            values = item.get("amplification_ratio")
            if values:
                per_perturbation[str(item.get("perturbation", "unknown"))] = values
            if overlay is None and item.get("W_t"):
                overlay = item.get("W_t")
        if per_perturbation:
            profiles[level] = per_perturbation
        if overlay:
            overlays[level] = overlay
    return profiles, overlays


def _exp4_sample_cutoffs(
    records: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, Tuple[List[str], List[str], np.ndarray]]:
    results: Dict[str, Tuple[List[str], List[str], np.ndarray]] = {}
    for mode in ("lowpass", "highpass"):
        sample_ids: List[str] = []
        rows: List[List[float]] = []
        for record in records:
            mode_data = record.get("modes", {}).get(mode)
            if not mode_data:
                continue
            cutoffs = mode_data.get("cutoffs", [])
            row = []
            for level in _LEVEL_ORDER:
                correct = mode_data.get("levels", {}).get(level, {}).get("correct_at_cutoff", [])
                if cutoffs and correct:
                    accuracy = [1.0 if value else 0.0 for value in correct]
                    row.append(float(compute_critical_cutoff(accuracy, cutoffs, threshold=threshold)))
                else:
                    row.append(np.nan)
            if not np.all(np.isnan(row)):
                sample_ids.append(str(record.get("image_id")))
                rows.append(row)
        if rows:
            results[mode] = (sample_ids, _LEVEL_ORDER, np.asarray(rows, dtype=float))
    return results


def _select_exp4_samples(records: List[Dict[str, Any]], max_samples: int = 3) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        values = []
        for mode_data in record.get("modes", {}).values():
            cutoffs = mode_data.get("cutoffs", [])
            for level_data in mode_data.get("levels", {}).values():
                correct = level_data.get("correct_at_cutoff", [])
                if cutoffs and correct:
                    accuracy = [1.0 if item else 0.0 for item in correct]
                    values.append(float(compute_critical_cutoff(accuracy, cutoffs)))
        if values:
            scored.append((float(np.mean(values)), record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:max_samples]]


def _exp5_heatmap_matrix(grouped_pairs: List[Dict[str, Any]], key: str) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _ordered_levels({pair.get("level") for pair in grouped_pairs if pair.get("level")})
    perturbations = sorted({str(pair.get("perturbation")) for pair in grouped_pairs if pair.get("perturbation")})
    matrix = np.full((len(levels), len(perturbations)), np.nan, dtype=float)
    for row, level in enumerate(levels):
        for col, perturbation in enumerate(perturbations):
            for pair in grouped_pairs:
                if pair.get("level") == level and pair.get("perturbation") == perturbation:
                    matrix[row, col] = float(pair.get(key, np.nan))
                    break
    return levels, perturbations, matrix


def _exp6_level_perturbation_matrix(
    records: List[Dict[str, Any]],
    model_name: str,
) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _SEG_LEVEL_ORDER
    perturbations = sorted(
        {
            str(perturbation.get("perturbation"))
            for record in records
            for level_data in record.get("models", {}).get(model_name, {}).get("levels", {}).values()
            for perturbation in level_data.get("perturbations", [])
        }
    )
    matrix = np.full((len(levels), len(perturbations)), np.nan, dtype=float)
    for row, level in enumerate(levels):
        for col, perturbation in enumerate(perturbations):
            values = []
            for record in records:
                items = record.get("models", {}).get(model_name, {}).get("levels", {}).get(level, {}).get("perturbations", [])
                for item in items:
                    if item.get("perturbation") == perturbation:
                        values.append(float(item.get("miou_drop", 0.0)))
            if values:
                matrix[row, col] = float(np.mean(values))
    valid_rows = [idx for idx in range(len(levels)) if not np.all(np.isnan(matrix[idx]))]
    return [levels[idx] for idx in valid_rows], perturbations, matrix[valid_rows]


def _exp6_sample_level_matrix(
    records: List[Dict[str, Any]],
    model_name: str,
) -> Tuple[List[str], List[str], np.ndarray]:
    sample_ids: List[str] = []
    rows: List[List[float]] = []
    for record in records:
        row = []
        for level in _SEG_LEVEL_ORDER:
            values = [
                float(item.get("miou_drop", 0.0))
                for item in record.get("models", {}).get(model_name, {}).get("levels", {}).get(level, {}).get("perturbations", [])
            ]
            row.append(float(np.mean(values)) if values else np.nan)
        if not np.all(np.isnan(row)):
            sample_ids.append(str(record.get("image_id")))
            rows.append(row)
    if not rows:
        return [], [], np.empty((0, 0))
    return sample_ids, _SEG_LEVEL_ORDER, np.asarray(rows, dtype=float)


def _plot_exp2_level_band_heatmap(summary: Dict[str, Any], out_path: Path) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    if not levels:
        return
    matrix = np.asarray([per_level[level].get("W_t_average", []) for level in levels], dtype=float)
    if matrix.size == 0:
        return
    _plot_heatmap(
        matrix,
        [_level_label(level) for level in levels],
        [f"B{i}" for i in range(matrix.shape[1])],
        out_path,
        "Attention Filter W_t by Level and Band",
        "W_t(omega)",
        cmap="magma",
    )


def _plot_exp2_selected_samples(records: List[Dict[str, Any]], out_dir: Path) -> None:
    selected = _select_exp2_samples(records)
    if not selected:
        return

    ncols = min(2, len(selected))
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.5 * nrows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis in axes_arr.ravel()[len(selected):]:
        axis.axis("off")

    for axis, record in zip(axes_arr.ravel(), selected):
        sample_id = str(record.get("image_id"))
        for level in _ordered_levels(record.get("levels", {}).keys()):
            values = record.get("levels", {}).get(level, {}).get("W_t")
            if not values:
                continue
            axis.plot(
                np.arange(len(values)),
                values,
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                color=_LEVEL_COLORS.get(level, "#999"),
                label=_level_label(level),
            )
        axis.set_title(_short_sample_id(sample_id, 28), fontsize=11)
        axis.set_xlabel("Frequency Band", fontsize=10)
        axis.set_ylabel("W_t(omega)", fontsize=10)
        axis.grid(alpha=0.2, linewidth=0.6)
        _apply_style(axis)

    handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("Representative Sample Attention Spectra", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    _save_fig(fig, out_dir / "exp2_sample_spectra.png")


def _plot_exp3_pre_post(summary: Dict[str, Any], out_path: Path) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    if not levels:
        return
    _plot_grouped_bars(
        {
            "Pre-fusion": [per_level[level].get("mean_pre_drift", 0.0) for level in levels],
            "Post-fusion": [per_level[level].get("mean_post_drift", 0.0) for level in levels],
        },
        [_level_label(level) for level in levels],
        out_path,
        "Pre- vs Post-Fusion Drift by Level",
        "Mean Drift",
        colors=["#6baed6", "#fb6a4a"],
    )


def _plot_exp4_critical_cutoffs(summary: Dict[str, Any], out_path: Path) -> None:
    per_mode = summary.get("per_mode", {})
    levels = _ordered_levels({level for mode_data in per_mode.values() for level in mode_data})
    if not levels:
        return
    series = {
        mode.capitalize(): [per_mode.get(mode, {}).get(level, {}).get("critical_cutoff", np.nan) for level in levels]
        for mode in ("lowpass", "highpass")
        if mode in per_mode
    }
    _plot_grouped_bars(
        series,
        [_level_label(level) for level in levels],
        out_path,
        "Critical Cutoff by Level",
        "Critical Cutoff",
        colors=["#3182bd", "#e6550d"],
        ylim=(0.0, 0.55),
    )


def _plot_exp4_sample_curves(records: List[Dict[str, Any]], out_dir: Path) -> None:
    for mode in ("lowpass", "highpass"):
        selected = _select_exp4_samples(records)
        if not selected:
            continue
        ncols = min(2, len(selected))
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.2 * nrows), sharex=True, sharey=True)
        axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
        for axis in axes_arr.ravel()[len(selected):]:
            axis.axis("off")

        for axis, record in zip(axes_arr.ravel(), selected):
            mode_data = record.get("modes", {}).get(mode)
            if not mode_data:
                axis.axis("off")
                continue
            cutoffs = mode_data.get("cutoffs", [])
            for level in _ordered_levels(mode_data.get("levels", {}).keys()):
                correct = mode_data.get("levels", {}).get(level, {}).get("correct_at_cutoff", [])
                if not cutoffs or not correct:
                    continue
                accuracy = [1.0 if item else 0.0 for item in correct]
                axis.plot(
                    cutoffs[: len(accuracy)],
                    accuracy,
                    marker="o",
                    linewidth=1.8,
                    markersize=3.8,
                    color=_LEVEL_COLORS.get(level, "#999"),
                    label=_level_label(level),
                )
            axis.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
            axis.set_title(_short_sample_id(str(record.get("image_id")), 28), fontsize=11)
            axis.set_xlabel("Cutoff", fontsize=10)
            axis.set_ylabel("Accuracy", fontsize=10)
            axis.set_ylim(-0.05, 1.05)
            _apply_style(axis)

        handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.suptitle(f"Sample Accuracy Curves ({mode})", fontsize=14, y=0.99)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        _save_fig(fig, out_dir / f"exp4_sample_curves_{mode}.png")


def _plot_exp5_heatmaps(grouped_pairs: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    levels, perturbations, predicted = _exp5_heatmap_matrix(grouped_pairs, "predicted")
    _, _, actual = _exp5_heatmap_matrix(grouped_pairs, "actual")
    if predicted.size == 0 or actual.size == 0:
        return

    residual = actual - predicted
    fig, axes = plt.subplots(1, 3, figsize=(max(15.0, 6.0 + 0.7 * len(perturbations)), 5.3))
    panels = [
        (predicted, "Predicted Overlap", "magma"),
        (actual, "Observed Drop", "viridis"),
        (residual, "Residual (actual - predicted)", "coolwarm"),
    ]

    for axis, (matrix, panel_title, cmap) in zip(axes, panels):
        masked = np.ma.masked_invalid(matrix)
        im = axis.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
        axis.set_xticks(np.arange(len(perturbations)))
        axis.set_xticklabels(perturbations, rotation=40, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(levels)))
        axis.set_yticklabels([_level_label(level) for level in levels], fontsize=9)
        axis.set_title(panel_title, fontsize=11)
        _apply_style(axis)
        fig.colorbar(im, ax=axis, shrink=0.8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_fig(fig, out_path)


def _plot_exp5_per_level_corr(summary: Dict[str, Any], out_path: Path) -> None:
    image_corr = summary.get("image_space", {}).get("per_level_correlation", {})
    vision_corr = summary.get("vision_feature_space", {}).get("per_level_correlation", {})
    levels = _ordered_levels(set(image_corr) | set(vision_corr))
    if not levels:
        return
    _plot_grouped_bars(
        {
            "Image-space": [image_corr.get(level, {}).get("pearson_r", np.nan) for level in levels],
            "Vision-feature": [vision_corr.get(level, {}).get("pearson_r", np.nan) for level in levels],
        },
        [_level_label(level) for level in levels],
        out_path,
        "Per-Level Overlap Correlation",
        "Pearson r",
        colors=["#6baed6", "#fd8d3c"],
        ylim=(-1.0, 1.0),
    )


def generate_all_plots(results_dir: Path) -> None:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    exp1_summary = _load_json(results_dir / "exp1" / "summary.json")
    exp1_detail = _load_json(results_dir / "exp1" / "degradation_by_level.json")
    exp1_samples = _load_jsonl(results_dir / "exp1" / "per_sample.jsonl")
    if exp1_summary:
        plot_granularity_curves(exp1_summary, plots_dir / "exp1_granularity_curves.png")
    if exp1_detail:
        levels, perturbations, matrix = _exp1_level_perturbation_matrix(exp1_detail)
        _plot_heatmap(
            matrix,
            [_level_label(level) for level in levels],
            perturbations,
            plots_dir / "exp1_level_perturbation_drop.png",
            "Accuracy Drop by Level and Perturbation",
            "Mean Accuracy Drop",
            cmap="YlOrRd",
        )
    if exp1_samples:
        sample_ids, levels, matrix = _exp1_sample_level_matrix(exp1_samples)
        sample_ids, matrix = _sort_metric_rows(sample_ids, matrix, max_rows=10)
        _plot_heatmap(
            matrix,
            [_short_sample_id(sample_id) for sample_id in sample_ids],
            [_level_label(level) for level in levels],
            plots_dir / "exp1_sample_level_drop.png",
            "Per-Sample Mean Drop by Level",
            "Mean Accuracy Drop",
            cmap="YlGnBu",
        )
        for record in _select_exp1_samples(exp1_samples):
            sample_id = str(record.get("image_id"))
            delta_f = _exp1_sample_spectra(record, "delta_f")
            if delta_f:
                _plot_sample_profile_grid(
                    sample_id,
                    delta_f,
                    plots_dir / f"exp1_sample_delta_f_{_safe_name(sample_id)}.png",
                    "Image-Space delta_f by Perturbation",
                    "delta_f(omega)",
                )
            delta_f_vision = _exp1_sample_spectra(record, "delta_f_vision")
            if delta_f_vision:
                _plot_sample_profile_grid(
                    sample_id,
                    delta_f_vision,
                    plots_dir / f"exp1_sample_delta_f_vision_{_safe_name(sample_id)}.png",
                    "Vision-Feature delta_f by Perturbation",
                    "delta_f_vision(omega)",
                )

    exp2_summary = _load_json(results_dir / "exp2" / "summary.json")
    exp2_samples = _load_json(results_dir / "exp2" / "power_spectra.json")
    if exp2_summary:
        plot_attention_power_spectrum(exp2_summary, plots_dir / "exp2_power_spectrum.png")
        plot_effective_bandwidth(exp2_summary, plots_dir / "exp2_bandwidth.png")
        _plot_exp2_level_band_heatmap(exp2_summary, plots_dir / "exp2_band_heatmap.png")
    if exp2_samples:
        _plot_distribution_with_points(
            _exp2_bandwidth_values(exp2_samples),
            plots_dir / "exp2_bandwidth_distribution.png",
            "Bandwidth Distribution Across Samples",
            "Bandwidth",
        )
        sample_ids, levels, matrix = _exp2_sample_level_matrix(exp2_samples)
        sample_ids, matrix = _sort_metric_rows(sample_ids, matrix, max_rows=10)
        _plot_heatmap(
            matrix,
            [_short_sample_id(sample_id) for sample_id in sample_ids],
            [_level_label(level) for level in levels],
            plots_dir / "exp2_sample_bandwidth.png",
            "Per-Sample Bandwidth by Level",
            "Bandwidth",
            cmap="plasma",
        )
        _plot_exp2_selected_samples(exp2_samples, plots_dir)

    exp3_summary = _load_json(results_dir / "exp3" / "summary.json")
    exp3_samples = _load_json(results_dir / "exp3" / "amplification.json")
    if exp3_summary:
        plot_amplification_heatmap(exp3_summary, plots_dir / "exp3_amplification.png")
        _plot_exp3_pre_post(exp3_summary, plots_dir / "exp3_pre_post_drift.png")
    if exp3_samples:
        levels, perturbations, matrix = _exp3_level_perturbation_matrix(exp3_samples)
        _plot_heatmap(
            matrix,
            [_level_label(level) for level in levels],
            perturbations,
            plots_dir / "exp3_level_perturbation_amplification.png",
            "Amplification by Level and Perturbation",
            "Mean Amplification",
            cmap="YlOrRd",
        )
        sample_ids, levels, matrix = _exp3_sample_level_matrix(exp3_samples)
        sample_ids, matrix = _sort_metric_rows(sample_ids, matrix, max_rows=10)
        _plot_heatmap(
            matrix,
            [_short_sample_id(sample_id) for sample_id in sample_ids],
            [_level_label(level) for level in levels],
            plots_dir / "exp3_sample_amplification.png",
            "Per-Sample Amplification by Level",
            "Mean Amplification",
            cmap="PuRd",
        )
        for record in _select_exp3_samples(exp3_samples):
            sample_id = str(record.get("image_id"))
            profiles, overlays = _exp3_sample_profiles(record)
            if profiles:
                _plot_sample_profile_grid(
                    sample_id,
                    profiles,
                    plots_dir / f"exp3_sample_profiles_{_safe_name(sample_id)}.png",
                    "Amplification Ratio by Perturbation",
                    "R(omega)",
                    overlay_by_level=overlays,
                )

    exp4_summary = _load_json(results_dir / "exp4" / "summary.json")
    exp4_curves = _load_json(results_dir / "exp4" / "accuracy_curves.json")
    exp4_samples = _load_json(results_dir / "exp4" / "per_sample.json")
    if exp4_curves:
        for mode in ("lowpass", "highpass"):
            plot_frequency_threshold_curves(
                exp4_curves,
                plots_dir / f"exp4_threshold_{mode}.png",
                mode=mode,
            )
    if exp4_summary:
        _plot_exp4_critical_cutoffs(exp4_summary, plots_dir / "exp4_critical_cutoffs.png")
    if exp4_samples:
        for mode, (sample_ids, levels, matrix) in _exp4_sample_cutoffs(exp4_samples).items():
            sample_ids, matrix = _sort_metric_rows(sample_ids, matrix, max_rows=10)
            _plot_heatmap(
                matrix,
                [_short_sample_id(sample_id) for sample_id in sample_ids],
                [_level_label(level) for level in levels],
                plots_dir / f"exp4_sample_cutoffs_{mode}.png",
                f"Per-Sample Critical Cutoff ({mode})",
                "Critical Cutoff",
                cmap="cividis",
                vmin=0.0,
                vmax=0.5,
            )
        _plot_exp4_sample_curves(exp4_samples, plots_dir)

    exp5_summary = _load_json(results_dir / "exp5" / "summary.json")
    exp5_grouped = _load_json(results_dir / "exp5" / "scatter_data.json")
    exp5_samples = _load_json(results_dir / "exp5" / "sample_scatter_data.json")
    exp5_vision_grouped = _load_json(results_dir / "exp5" / "vision_scatter_data.json")
    exp5_vision_samples = _load_json(results_dir / "exp5" / "vision_sample_scatter_data.json")
    if exp5_grouped and exp5_summary:
        image_summary = exp5_summary.get("image_space", exp5_summary)
        plot_overlap_scatter(
            exp5_grouped,
            image_summary.get("pearson_r_grouped", 0.0),
            plots_dir / "exp5_overlap_scatter.png",
            title="Overlap vs Accuracy Drop (grouped, image-space)",
        )
        _plot_exp5_heatmaps(
            exp5_grouped,
            plots_dir / "exp5_grouped_heatmap_image.png",
            "Grouped Overlap vs Drop (image-space)",
        )
    if exp5_samples and exp5_summary:
        image_summary = exp5_summary.get("image_space", exp5_summary)
        plot_overlap_scatter(
            exp5_samples,
            image_summary.get("pearson_r_sample", 0.0),
            plots_dir / "exp5_overlap_scatter_sample.png",
            title="Overlap vs Accuracy Drop (per sample, image-space)",
            point_size=20,
            alpha=0.45,
        )
    if exp5_vision_grouped and exp5_summary:
        vision_summary = exp5_summary.get("vision_feature_space", {})
        plot_overlap_scatter(
            exp5_vision_grouped,
            vision_summary.get("pearson_r_grouped", 0.0),
            plots_dir / "exp5_overlap_scatter_vision.png",
            title="Overlap vs Accuracy Drop (grouped, vision-feature)",
        )
        _plot_exp5_heatmaps(
            exp5_vision_grouped,
            plots_dir / "exp5_grouped_heatmap_vision.png",
            "Grouped Overlap vs Drop (vision-feature)",
        )
    if exp5_vision_samples and exp5_summary:
        vision_summary = exp5_summary.get("vision_feature_space", {})
        plot_overlap_scatter(
            exp5_vision_samples,
            vision_summary.get("pearson_r_sample", 0.0),
            plots_dir / "exp5_overlap_scatter_vision_sample.png",
            title="Overlap vs Accuracy Drop (per sample, vision-feature)",
            point_size=20,
            alpha=0.45,
        )
    if exp5_summary:
        _plot_exp5_per_level_corr(exp5_summary, plots_dir / "exp5_per_level_correlation.png")

    exp6_summary = _load_json(results_dir / "exp6" / "summary.json")
    exp6_samples = _load_jsonl(results_dir / "exp6" / "per_sample.jsonl")
    if exp6_summary:
        plot_segmentation_granularity(exp6_summary, plots_dir / "exp6_segmentation.png")
    if exp6_samples:
        for model_name in sorted({name for record in exp6_samples for name in record.get("models", {})}):
            levels, perturbations, matrix = _exp6_level_perturbation_matrix(exp6_samples, model_name)
            if matrix.size:
                _plot_heatmap(
                    matrix,
                    [_level_label(level) for level in levels],
                    perturbations,
                    plots_dir / f"exp6_level_perturbation_{_safe_name(model_name)}.png",
                    f"mIoU Drop by Level and Perturbation ({model_name})",
                    "Mean mIoU Drop",
                    cmap="YlOrRd",
                )
            sample_ids, levels, matrix = _exp6_sample_level_matrix(exp6_samples, model_name)
            sample_ids, matrix = _sort_metric_rows(sample_ids, matrix, max_rows=10)
            if matrix.size:
                _plot_heatmap(
                    matrix,
                    [_short_sample_id(sample_id) for sample_id in sample_ids],
                    [_level_label(level) for level in levels],
                    plots_dir / f"exp6_sample_level_{_safe_name(model_name)}.png",
                    f"Per-Sample mIoU Drop ({model_name})",
                    "Mean mIoU Drop",
                    cmap="PuRd",
                )

    logger.info("All plots generated in %s", plots_dir)
