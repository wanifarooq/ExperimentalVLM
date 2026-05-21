#!/usr/bin/env python3
"""Compose publication-quality cross-run figures for the paper.

Five figures, all generated with a single shared matplotlib style so they
look consistent in the PDF (same fonts, same palette, same dimensions, same
grid style, same caption-friendly aspect ratios):

  N1 — per-layer bandwidth G(t) trajectory (2×2 grid)
  N2 — ΔF × W_t signature volatility matrix (2×2 grid, shared colourbar)
  N3 — overlap-law Pearson r across three aggregation scales (grouped bars)
  N4 — W_t mass per radial band at last_2 (2×2 grid, shared y-axis)
  N5 — designed perturbations sorted by volatility (horizontal bars, coloured
       by measured ΔF signature)

Inputs are read straight from the four `qwen_*` run directories.
Outputs are written to `paper/figures/n*.{png,pdf}`.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import numpy as np

# ---------------------------------------------------------------------------
# Shared paper style (single source of truth for every figure)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Computer Modern", "Times New Roman"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "legend.frameon":     True,
    "legend.framealpha":  0.92,
    "legend.fancybox":    False,
    "legend.edgecolor":   "0.8",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.25,
    "grid.color":         "0.6",
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "lines.linewidth":    1.6,
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
})

# Colour-blind-safe palette (used across all figures)
PALETTE = {
    "CLEVR-2B": "#4c72b0",
    "CLEVR-8B": "#1f4068",
    "GQA-2B":   "#dd8452",
    "GQA-8B":   "#a04415",
    "primary":  "#4c72b0",
    "wordy":    "#dd8452",
    "L1":       "#4c72b0",
    "L4":       "#a04415",
    "L5":       "#4c72b0",
    "L8":       "#a04415",
    "low_freq":  "#4c72b0",
    "mid":       "#937860",
    "high_freq": "#55a868",
    "broadband": "#dd8452",
    "grouped":   "#4c72b0",
    "fe":        "#dd8452",
    "within":    "#55a868",
}

ROOT = Path("/home/farooq/Public/vlm-robustness")
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "CLEVR-2B": "qwen_clevr_2B/frequency_alignment_outputs_clevr_2B_500_20260516_132931",
    "CLEVR-8B": "qwen_clevr_8B/frequency_alignment_outputs_clevr_8B_500_20260516_131942",
    "GQA-2B":   "qwen_gpa_2B/frequency_alignment_outputs_gqa_2B_500_20260517_032513",
    "GQA-8B":   "qwen_gpa_8B/frequency_alignment_outputs_gqa_8B_500_20260516_125653",
}
RUN_ORDER = ["CLEVR-2B", "CLEVR-8B", "GQA-2B", "GQA-8B"]
LEVELS = ['L1_COARSE', 'L2_MEDIUM', 'L3_FINE', 'L4_VERY_FINE',
          'L5_WORDY_SIMPLETON', 'L6_WORDY_MEDIUM', 'L7_WORDY_FINE', 'L8_WORDY_VERY_FINE']


def _load(path: Path) -> Any:
    return json.load(open(path))


def _save(fig, name: str) -> None:
    """Save both PNG and PDF with consistent settings."""
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}  +  {out.with_suffix('.pdf').name}")


# ---------------------------------------------------------------------------
# N1 — per-layer bandwidth trajectory (2×2 grid)
# ---------------------------------------------------------------------------

def fig_n1_trajectory() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharey=False)
    # First collect global y-range
    all_y = []
    data: Dict[str, Dict[str, list]] = {}
    for label in RUN_ORDER:
        s = _load(ROOT / RUNS[label] / "exp2" / "summary.json")
        plb = s["per_layer_bandwidth"]
        n_layers = len(plb[LEVELS[0]]["layer_indices"])
        primary = [statistics.mean(plb[lv]["mean_gt"][i] for lv in LEVELS[:4]) for i in range(n_layers)]
        wordy = [statistics.mean(plb[lv]["mean_gt"][i] for lv in LEVELS[4:]) for i in range(n_layers)]
        data[label] = {"layers": list(range(n_layers)), "primary": primary, "wordy": wordy,
                       "B": int(s["effective_num_bands"])}
        all_y.extend(primary); all_y.extend(wordy)
    y_lo, y_hi = min(all_y) - 0.3, max(all_y) + 0.5

    for ax, label in zip(axes.flat, RUN_ORDER):
        d = data[label]
        ax.plot(d["layers"], d["primary"], color=PALETTE["primary"], lw=1.6,
                label="primary (L1–L4 mean)")
        ax.plot(d["layers"], d["wordy"],   color=PALETTE["wordy"],   lw=1.6, linestyle="--",
                label="wordy (L5–L8 mean)")
        ax.fill_between(d["layers"], d["primary"], d["wordy"],
                        where=[w > p for p, w in zip(d["primary"], d["wordy"])],
                        color=PALETTE["wordy"], alpha=0.12, interpolate=True,
                        label="wordy broader")
        gaps = [w - p for p, w in zip(d["primary"], d["wordy"])]
        peak = int(np.argmax(gaps))
        ax.axvline(peak, color="black", lw=0.5, ls=":", alpha=0.6)
        ax.scatter([peak], [d["wordy"][peak]], color="black", s=18, zorder=5)
        ax.annotate(f"peak Δ={gaps[peak]:+.2f}\nlayer {peak}",
                    xy=(peak, d["wordy"][peak]),
                    xytext=(6, -1), textcoords="offset points",
                    fontsize=7, color="0.2",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.4))
        ax.set_title(f"{label}   (B={d['B']})")
        ax.set_xlabel("decoder layer index")
        ax.set_ylabel(r"$G(t)$  (IPR bandwidth)")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(-0.5, len(d["layers"]) - 0.5)
        if label == RUN_ORDER[0]:
            ax.legend(loc="lower right", fontsize=7)
    fig.suptitle(r"Layer-wise mean $G(t)$ — primary L1–L4 vs wordy L5–L8",
                 y=1.005, fontsize=10.5)
    fig.tight_layout()
    _save(fig, "n1_trajectory_grid")


# ---------------------------------------------------------------------------
# N2 — ΔF × W_t signature volatility matrix (2×2 grid, shared colorbar)
# ---------------------------------------------------------------------------

def _read_signature_matrix(label: str) -> Dict[str, Dict[str, float]]:
    p = ROOT / RUNS[label] / "plots_delta_wt_signature_matrix" / "image_last_2" / "delta_wt_signature_matrix.csv"
    out: Dict[str, Dict[str, float]] = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out.setdefault(row["delta_signature"], {})[row["wt_signature"]] = float(row["mean_loglik_volatility"])
    return out


def fig_n2_signature_matrices() -> None:
    delta_order = ["low", "mid", "high", "broadband"]
    wt_order = ["narrow", "broadband"]
    matrices = {label: _read_signature_matrix(label) for label in RUN_ORDER}
    arr = {label: np.array([[matrices[label].get(d, {}).get(w, np.nan) for w in wt_order]
                            for d in delta_order]) for label in RUN_ORDER}
    vmax = max(np.nanmax(arr[label]) for label in RUN_ORDER)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.6), sharey=True)
    for ax, label in zip(axes.flat, RUN_ORDER):
        ax.grid(False)
        im = ax.imshow(arr[label], cmap="magma_r", aspect="auto", vmin=0, vmax=vmax)
        for i, d in enumerate(delta_order):
            for j, w in enumerate(wt_order):
                v = arr[label][i, j]
                if np.isnan(v):
                    continue
                colour = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=colour, fontsize=9, weight="bold")
        ax.set_xticks(range(len(wt_order))); ax.set_xticklabels(wt_order)
        ax.set_yticks(range(len(delta_order))); ax.set_yticklabels(delta_order)
        ax.set_xlabel(r"$W_t$ shape")
        if label in (RUN_ORDER[0], RUN_ORDER[2]):
            ax.set_ylabel(r"$\Delta F$ family")
        ax.set_title(label)
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(r"Mean log-likelihood volatility per $(\Delta F,\,W_t)$ cell", y=0.99, fontsize=10.5)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.93, bottom=0.08, wspace=0.18, hspace=0.30)
    cbar_ax = fig.add_axes([0.89, 0.18, 0.020, 0.66])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("mean log-likelihood volatility", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, "n2_signature_matrices_grid")


# ---------------------------------------------------------------------------
# N3 — overlap-law Pearson r bar chart (3 aggregation scales × 4 runs)
# ---------------------------------------------------------------------------

def fig_n3_overlap_law_bars() -> None:
    grouped_r, fe_r, within_med = [], [], []
    for label in RUN_ORDER:
        h = _load(ROOT / RUNS[label] / "exp5" / "hypothesis_tests.json")
        grouped_r.append(float(h["pearson_predicted_vs_actual_grouped"]["r"]))
        m = _load(ROOT / RUNS[label] / "plots_overlap_matched_fe" / "matched_overlap_fixed_effects_summary.json")
        s = m["summaries"][0]  # image-space last_2 radial first-order
        fe_r.append(float(s["fe_pearson_r"]))
        within_med.append(float(s["within_pearson_median"]))
    x = np.arange(len(RUN_ORDER))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bars1 = ax.bar(x - width, grouped_r,  width, label="grouped (population)", color=PALETTE["grouped"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x,         fe_r,       width, label="matched-FE (full demeaned)", color=PALETTE["fe"], edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, within_med, width, label="within-cell median",  color=PALETTE["within"], edgecolor="white", linewidth=0.5)
    for bars in (bars1, bars2, bars3):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(RUN_ORDER)
    ax.set_ylabel(r"Pearson $r$  (overlap $\langle W_t,\,\Delta F\rangle$ vs volatility)")
    ax.set_ylim(0, max(grouped_r + fe_r + within_med) * 1.18)
    ax.legend(loc="upper right", ncol=3, fontsize=7.5,
              bbox_to_anchor=(1.0, 1.13), frameon=False)
    ax.set_title("Overlap-law predictive strength across the four 500-sample runs", pad=22)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.85, bottom=0.12)
    _save(fig, "n3_overlap_law_bars")


# ---------------------------------------------------------------------------
# N4 — W_t mass per radial band (2×2 grid, shared y-axis)
# ---------------------------------------------------------------------------

def fig_n4_band_mass() -> None:
    # Pull data for shared axis
    data = {}
    all_y = []
    for label in RUN_ORDER:
        s = _load(ROOT / RUNS[label] / "exp2" / "summary.json")
        B = int(s["effective_num_bands"])
        d: Dict[str, list] = {"B": B}
        for lv in ("L1_COARSE", "L4_VERY_FINE", "L5_WORDY_SIMPLETON", "L8_WORDY_VERY_FINE"):
            wt = s["per_level"][lv]["layer_groups"]["last_2"]["W_t_average"]
            total = sum(wt)
            norm = [x / total for x in wt] if total > 0 else wt
            d[lv] = norm; all_y.extend(norm)
        data[label] = d
    y_max = max(all_y) * 1.10
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), sharey=True)
    level_styles = {
        "L1_COARSE":           ("L1 primary",   PALETTE["L1"], "-",  1.6),
        "L5_WORDY_SIMPLETON":  ("L5 wordy",     PALETTE["L1"], "--", 1.4),
        "L4_VERY_FINE":        ("L4 primary",   PALETTE["L4"], "-",  1.6),
        "L8_WORDY_VERY_FINE":  ("L8 wordy",     PALETTE["L4"], "--", 1.4),
    }
    for ax, label in zip(axes.flat, RUN_ORDER):
        d = data[label]
        B = d["B"]
        for lv, (lbl, c, ls, lw) in level_styles.items():
            ax.plot(range(B), d[lv], color=c, linestyle=ls, lw=lw, label=lbl, marker="o", markersize=3)
        # Highlight highest band
        ax.axvspan(B - 1.5, B - 0.5, color="#cc4d4d", alpha=0.10, zorder=0)
        ax.set_xticks(range(B))
        ax.set_title(f"{label}   (B={B})")
        ax.set_xlabel(r"radial band $\omega$  (0 = DC, $B-1$ = highest)")
        ax.set_ylabel(r"$W_t(\omega)$ mass")
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
        if label == RUN_ORDER[0]:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle(r"$W_t^{(\mathrm{last\_2})}$ mass per radial band — the highest band carries only 3–5 % of the mass",
                 y=1.005, fontsize=10.5)
    fig.tight_layout()
    _save(fig, "n4_band_mass")


# ---------------------------------------------------------------------------
# N5 — designed perturbations: horizontal bar chart, GQA-8B
# ---------------------------------------------------------------------------

def fig_n5_designed_vs_actual() -> None:
    """Two-panel figure on GQA-8B:
        left  — per-operator mean volatility (horizontal bar, sorted)
        right — STACKED bar showing the share of each ΔF signature within each
                operator (so the reader sees that JPEG is broadband but with a
                pinch of low; LowBandNoise is high_freq with a small broadband
                tail; etc. — multiple ops can carry the same signature)
    """
    base = ROOT / RUNS["GQA-8B"] / "plots_perturbation_signature_volatility" / "image"
    rows = list(csv.DictReader(open(base / "volatility_by_designed_perturbation.csv")))
    by_designed: Dict[str, Dict[str, int]] = {}
    for r in csv.DictReader(open(base / "volatility_by_designed_and_actual_signature.csv")):
        by_designed.setdefault(r["designed_perturbation"], {})[r["actual_signature"]] = int(r["n"])

    # Sort by volatility ascending for horizontal bars
    rows.sort(key=lambda r: float(r["mean_loglik_volatility"]))
    names = [r["designed_perturbation"] for r in rows]
    vols  = [float(r["mean_loglik_volatility"]) for r in rows]
    dominant = {n: max(sigs.items(), key=lambda kv: kv[1])[0] for n, sigs in by_designed.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.8),
                             gridspec_kw={"width_ratios": [1.0, 0.85], "wspace": 0.05})
    ax = axes[0]
    y = np.arange(len(names))
    colours = [PALETTE.get(dominant.get(n, "broadband"), "gray") for n in names]
    ax.barh(y, vols, color=colours, edgecolor="white", linewidth=0.5)
    for yi, v in zip(y, vols):
        ax.text(v + 0.07, yi, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(r"mean log-likelihood volatility  (GQA-8B)")
    ax.set_xlim(0, max(vols) * 1.13)
    ax.set_title("Mean volatility per operator   (bar colour = dominant $\\Delta F$ signature)",
                 fontsize=9.5, pad=8)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    # Right panel: stacked bar showing signature share per operator (same y order)
    ax2 = axes[1]
    sig_order = ["low_freq", "broadband", "high_freq"]
    left = np.zeros(len(names))
    totals = np.array([sum(by_designed.get(n, {}).values()) for n in names], dtype=float)
    for sig in sig_order:
        widths = np.array([by_designed.get(n, {}).get(sig, 0) for n in names], dtype=float)
        frac = np.where(totals > 0, widths / totals, 0.0)
        ax2.barh(y, frac, left=left, color=PALETTE[sig], edgecolor="white", linewidth=0.4, label=sig)
        left += frac
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel(r"fraction of perturbed images with that measured $\Delta F$ signature")
    ax2.set_title(r"Signature mix per operator", fontsize=9.5, pad=8)
    ax2.legend(loc="lower right", fontsize=8, frameon=True, ncol=1)
    ax2.grid(axis="x", alpha=0.25)
    ax2.grid(axis="y", visible=False)
    fig.suptitle(r"Designed perturbation $\rightarrow$ measured $\Delta F$ signature (GQA-8B)",
                 y=1.00, fontsize=10.5, weight="bold")
    fig.tight_layout()
    _save(fig, "n5_designed_vs_actual")


if __name__ == "__main__":
    fig_n1_trajectory()
    fig_n2_signature_matrices()
    fig_n3_overlap_law_bars()
    fig_n4_band_mass()
    fig_n5_designed_vs_actual()
    print(f"\nAll five cross-run figures regenerated in {FIG_DIR.relative_to(ROOT)}/")
