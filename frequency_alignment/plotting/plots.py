"""Publication-quality plots for all frequency alignment experiments.

All functions take pre-computed results (dicts/arrays) and produce
matplotlib figures.  Designed for direct inclusion in papers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Consistent styling
_LEVEL_COLORS = {
    "L1_COARSE": "#2196F3",
    "L2_MEDIUM": "#4CAF50",
    "L3_FINE": "#FF9800",
    "L4_VERY_FINE": "#F44336",
}
_LEVEL_LABELS = {
    "L1_COARSE": "L1 (Coarse)",
    "L2_MEDIUM": "L2 (Medium)",
    "L3_FINE": "L3 (Fine)",
    "L4_VERY_FINE": "L4 (Very Fine)",
}
_LEVEL_ORDER = ["L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE"]


def _apply_style(ax: plt.Axes) -> None:
    """Apply consistent publication styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Save figure and close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


# =========================================================================
# Plot 1: Granularity Scaling Curves (Experiment 1)
# =========================================================================

def plot_granularity_curves(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Accuracy Degradation vs. Task Granularity",
) -> None:
    """Bar chart of mean accuracy drop per granularity level.

    Shows that finer tasks (L4) degrade more than coarse tasks (L1).
    """
    per_level = summary.get("per_level", {})
    present = [lk for lk in _LEVEL_ORDER if lk in per_level]
    if not present:
        logger.warning("No data for granularity curves plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: clean vs perturbed accuracy
    ax = axes[0]
    x = np.arange(len(present))
    width = 0.35
    clean_accs = [per_level[lk]["clean_accuracy"] for lk in present]
    pert_accs = [per_level[lk]["perturbed_accuracy"] for lk in present]
    bars1 = ax.bar(x - width / 2, clean_accs, width, label="Clean", color="#4CAF50", alpha=0.8)
    bars2 = ax.bar(x + width / 2, pert_accs, width, label="Perturbed", color="#F44336", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_LEVEL_LABELS.get(lk, lk) for lk in present], fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Clean vs. Perturbed Accuracy", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    _apply_style(ax)

    # Right: mean accuracy drop with error bars
    ax = axes[1]
    drops = [per_level[lk]["mean_accuracy_drop"] for lk in present]
    stds = [per_level[lk]["std_accuracy_drop"] for lk in present]
    colors = [_LEVEL_COLORS.get(lk, "#999") for lk in present]
    ax.bar(x, drops, color=colors, alpha=0.85, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([_LEVEL_LABELS.get(lk, lk) for lk in present], fontsize=9)
    ax.set_ylabel("Mean Accuracy Drop", fontsize=11)
    ax.set_title("Degradation Scales with Granularity", fontsize=12)
    _apply_style(ax)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_path)


# =========================================================================
# Plot 2: Attention Power Spectrum (Experiment 2)
# =========================================================================

def plot_attention_power_spectrum(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Cross-Attention Power Spectrum by Granularity",
) -> None:
    """Line plot of radial power spectrum W_t(ω) per granularity level.

    Shows that coarse tasks have power concentrated at low frequencies,
    while fine tasks spread across higher bands.
    """
    per_level = summary.get("per_level", {})
    present = [lk for lk in _LEVEL_ORDER if lk in per_level]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for lk in present:
        W_t = per_level[lk].get("W_t_average", [])
        if not W_t:
            continue
        bands = np.arange(len(W_t))
        ax.plot(
            bands, W_t,
            marker="o", markersize=4, linewidth=2,
            color=_LEVEL_COLORS.get(lk, "#999"),
            label=_LEVEL_LABELS.get(lk, lk),
        )

    ax.set_xlabel("Frequency Band (low → high)", fontsize=11)
    ax.set_ylabel("W_t(ω) (normalized power)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


# =========================================================================
# Plot 3: Effective Bandwidth G(t) Bar Chart (Experiment 2)
# =========================================================================

def plot_effective_bandwidth(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Effective Bandwidth G(t) by Task Granularity",
) -> None:
    """Bar chart showing G(t) increases with task granularity."""
    per_level = summary.get("per_level", {})
    present = [lk for lk in _LEVEL_ORDER if lk in per_level]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(present))
    bws = [per_level[lk]["mean_bandwidth"] for lk in present]
    stds = [per_level[lk]["std_bandwidth"] for lk in present]
    colors = [_LEVEL_COLORS.get(lk, "#999") for lk in present]

    ax.bar(x, bws, color=colors, alpha=0.85, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([_LEVEL_LABELS.get(lk, lk) for lk in present], fontsize=10)
    ax.set_ylabel("Effective Bandwidth G(t)", fontsize=11)
    ax.set_title(title, fontsize=13)
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


# =========================================================================
# Plot 4: Drift Amplification Heatmap (Experiment 3)
# =========================================================================

def plot_amplification_heatmap(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Drift Amplification Ratio R(ω) by Level",
) -> None:
    """Heatmap of R(ω) per frequency band per level."""
    per_level = summary.get("per_level", {})
    present = [lk for lk in _LEVEL_ORDER if lk in per_level]
    if not present:
        return

    # Build matrix: rows = levels, cols = frequency bands
    data = []
    for lk in present:
        R = per_level[lk].get("mean_amplification_ratio", [])
        if R:
            data.append(R)
    if not data:
        return

    matrix = np.array(data)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([_LEVEL_LABELS.get(lk, lk) for lk in present], fontsize=10)
    ax.set_xlabel("Frequency Band (low → high)", fontsize=11)
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax, label="R(ω)")
    fig.tight_layout()
    _save_fig(fig, out_path)


# =========================================================================
# Plot 5: Frequency Threshold Curves (Experiment 4)
# =========================================================================

def plot_frequency_threshold_curves(
    accuracy_data: Dict[str, Any],
    out_path: Path,
    mode: str = "lowpass",
    title: str = "Accuracy vs. Frequency Cutoff",
) -> None:
    """Line plot of accuracy vs cutoff for each level.

    The critical cutoff ω_c* shifts right (higher) for finer tasks.
    """
    mode_data = accuracy_data.get("data", {}).get(mode, {})
    cutoffs = accuracy_data.get("cutoffs", [])
    if not mode_data or not cutoffs:
        return

    present = [lk for lk in _LEVEL_ORDER if lk in mode_data]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for lk in present:
        acc_curve = mode_data[lk].get("accuracy_curve", [])
        if not acc_curve:
            continue
        ax.plot(
            cutoffs[:len(acc_curve)], acc_curve,
            marker=".", markersize=3, linewidth=2,
            color=_LEVEL_COLORS.get(lk, "#999"),
            label=_LEVEL_LABELS.get(lk, lk),
        )
        # Mark critical cutoff
        omega_c = mode_data[lk].get("critical_cutoff")
        if omega_c:
            ax.axvline(omega_c, color=_LEVEL_COLORS.get(lk, "#999"),
                        linestyle="--", alpha=0.5, linewidth=1)

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="50% threshold")
    ax.set_xlabel(f"Frequency Cutoff ({mode})", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"{title} ({mode.capitalize()})", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


# =========================================================================
# Plot 6: Overlap Scatter Plot (Experiment 5)
# =========================================================================

def plot_overlap_scatter(
    scatter_data: List[Dict[str, Any]],
    pearson_r: float,
    out_path: Path,
    title: str = "Predicted vs. Actual Sensitivity",
) -> None:
    """Scatter plot of predicted (spectral overlap) vs actual (accuracy drop).

    Each point is one (level, perturbation) pair, colored by level.
    """
    if not scatter_data:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    for item in scatter_data:
        label = item["label"]
        level = label.split("|")[0] if "|" in label else "unknown"
        color = _LEVEL_COLORS.get(level, "#999")
        ax.scatter(
            item["predicted"], item["actual"],
            c=color, s=40, alpha=0.7, edgecolors="white", linewidth=0.5,
        )

    # Add regression line
    preds = np.array([d["predicted"] for d in scatter_data])
    actuals = np.array([d["actual"] for d in scatter_data])
    if len(preds) > 2:
        z = np.polyfit(preds, actuals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(preds.min(), preds.max(), 100)
        ax.plot(x_line, p(x_line), "k--", linewidth=1.5, alpha=0.6)

    ax.set_xlabel("Predicted Sensitivity (spectral overlap)", fontsize=11)
    ax.set_ylabel("Actual Sensitivity (accuracy drop)", fontsize=11)
    ax.set_title(f"{title} (r = {pearson_r:.3f})", fontsize=13)

    # Legend for levels
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=c, label=_LEVEL_LABELS.get(lk, lk))
                    for lk, c in _LEVEL_COLORS.items()]
    ax.legend(handles=legend_items, fontsize=9, loc="upper left")
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


# =========================================================================
# Plot 7: Segmentation Granularity (Experiment 6)
# =========================================================================

def plot_segmentation_granularity(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Segmentation Degradation by Hierarchy Level",
) -> None:
    """Grouped bar chart: SAM3 should fan out, SAM2 should stay flat."""
    per_model = summary.get("per_model", {})
    if not per_model:
        return

    level_order = ["L1_COARSE", "L2_MEDIUM", "L3_FINE"]
    models = list(per_model.keys())
    n_levels = len(level_order)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_levels)
    width = 0.35 if n_models == 2 else 0.5
    model_colors = {"sam3": "#F44336", "sam2": "#2196F3"}

    for mi, model_name in enumerate(models):
        model_data = per_model[model_name]
        drops = [model_data.get(lk, {}).get("mean_drop", 0) for lk in level_order]
        stds = [model_data.get(lk, {}).get("std_drop", 0) for lk in level_order]
        offset = (mi - (n_models - 1) / 2) * width
        color = model_colors.get(model_name, f"C{mi}")
        ax.bar(x + offset, drops, width * 0.9, label=model_name.upper(),
               color=color, alpha=0.8, yerr=stds, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([_LEVEL_LABELS.get(lk, lk) for lk in level_order], fontsize=10)
    ax.set_ylabel("Mean Mask Degradation (1 - IoU)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    _apply_style(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


# =========================================================================
# Master plotting function
# =========================================================================

def generate_all_plots(results_dir: Path) -> None:
    """Generate all plots from experiment output directory.

    Reads JSON results from each exp{N}/ subdirectory and produces plots.
    """
    import json

    def _load(path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    plots_dir = results_dir / "plots"

    # Exp 1: Granularity curves
    exp1_summary = _load(results_dir / "exp1" / "summary.json")
    if exp1_summary:
        plot_granularity_curves(exp1_summary, plots_dir / "exp1_granularity_curves.png")

    # Exp 2: Power spectrum + bandwidth
    exp2_summary = _load(results_dir / "exp2" / "summary.json")
    if exp2_summary:
        plot_attention_power_spectrum(exp2_summary, plots_dir / "exp2_power_spectrum.png")
        plot_effective_bandwidth(exp2_summary, plots_dir / "exp2_bandwidth.png")

    # Exp 3: Amplification heatmap
    exp3_summary = _load(results_dir / "exp3" / "summary.json")
    if exp3_summary:
        plot_amplification_heatmap(exp3_summary, plots_dir / "exp3_amplification.png")

    # Exp 4: Frequency threshold curves
    exp4_curves = _load(results_dir / "exp4" / "accuracy_curves.json")
    if exp4_curves:
        for mode in ["lowpass", "highpass"]:
            plot_frequency_threshold_curves(
                exp4_curves, plots_dir / f"exp4_threshold_{mode}.png", mode=mode,
            )

    # Exp 5: Overlap scatter
    exp5_scatter = _load(results_dir / "exp5" / "scatter_data.json")
    exp5_summary = _load(results_dir / "exp5" / "summary.json")
    if exp5_scatter and exp5_summary:
        plot_overlap_scatter(
            exp5_scatter,
            exp5_summary.get("pearson_r", 0),
            plots_dir / "exp5_overlap_scatter.png",
        )

    # Exp 6: Segmentation granularity
    exp6_summary = _load(results_dir / "exp6" / "summary.json")
    if exp6_summary:
        plot_segmentation_granularity(exp6_summary, plots_dir / "exp6_segmentation.png")

    logger.info("All plots generated in %s", plots_dir)
