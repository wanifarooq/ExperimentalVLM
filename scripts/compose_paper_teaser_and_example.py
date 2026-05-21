#!/usr/bin/env python3
"""Compose two additional paper figures around the worked example (image 2332974):

  T1 — paper teaser  (Figure 1, opening):
       a single visual showing the headline wordy-stabilises finding.
       Image on the left + four 4-cell mini-bars on the right showing
       per-level |drift| (terse vs wordy on L1, L4) for both 2B and 8B.

  E1 — appendix question-generation pipeline:
       schematic of scene-graph → L1-L4 questions → wordy template → L5-L8,
       laid out next to the image for the same example.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image

# --- shared style (same source as compose_cross_run_figures.py) -------------
plt.rcParams.update({
    "font.family":   "serif",
    "font.serif":    ["DejaVu Serif", "Times New Roman"],
    "font.size":     9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":     True,
    "axes.axisbelow": True,
    "grid.alpha":    0.25,
    "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
    "figure.dpi":    150,
})

PALETTE = {
    "primary": "#4c72b0",
    "wordy":   "#dd8452",
    "2B": "#4c72b0",
    "8B": "#a04415",
}

ROOT     = Path("/home/farooq/Public/vlm-robustness")
FIG_DIR  = ROOT / "paper" / "figures"
IMG_PATH = FIG_DIR / "example_gqa_clean.png"
IMAGE_ID = "2410848"

LEVEL_QS = {
    "L1": "L1: object presence",
    "L2": "L2: attribute query",
    "L3": "L3: spatial relation",
    "L4": "L4: compositional",
}
LEVEL_QS_FULL = {
    "L1": "Is there a bed in this image?",
    "L2": "What color is the wall?",
    "L3": "Is the outlet to the right of the wall?",
    "L4": "What color is the wall that is to the right of the outlet?",
}
LEVEL_GOLD = {"L1": "yes (A)", "L2": "green (A)", "L3": "yes (A)", "L4": "green (A)"}


def _pull_drifts(run: str) -> dict:
    """Return per-level |Δ| averaged across the perturbation bank for image 2332974."""
    found = None
    for line in open(ROOT / run / "exp1" / "per_sample.jsonl"):
        d = json.loads(line)
        if d["image_id"] == IMAGE_ID:
            found = d
            break
    if found is None:
        raise FileNotFoundError(f"image {IMAGE_ID} not found in {run}")
    out = {}
    for lv in ["L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE",
               "L5_WORDY_SIMPLETON", "L6_WORDY_MEDIUM", "L7_WORDY_FINE", "L8_WORDY_VERY_FINE"]:
        d = found["levels"][lv]
        perts = d.get("perturbations", [])
        if not perts:
            out[lv] = 0.0; continue
        absdrifts = [abs(p.get("loglik_drift", 0.0)) for p in perts]
        out[lv] = sum(absdrifts) / len(absdrifts)
    return out


# ---------------------------------------------------------------------------
# T1 — Teaser figure (Figure 1)
# ---------------------------------------------------------------------------

def fig_teaser() -> None:
    drift_2b = _pull_drifts("qwen_gpa_2B/frequency_alignment_outputs_gqa_2B_500_20260517_032513")
    drift_8b = _pull_drifts("qwen_gpa_8B/frequency_alignment_outputs_gqa_8B_500_20260516_125653")
    img = np.asarray(Image.open(IMG_PATH))

    fig = plt.figure(figsize=(8.4, 4.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.05, 0.02, 1.0, 0.04, 1.0], wspace=0.30)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for s in ("top","right","left","bottom"): ax_img.spines[s].set_visible(False)
    ax_img.set_title(r"GQA image 2410848", fontsize=9.5, pad=4)
    ax_img.set_xlabel("two Shiba Inus on a bed", fontsize=8, color="0.4")

    # Per-pair bars (L1/L5, L4/L8) for both models
    for axi, (mtitle, drifts) in enumerate(((r"Qwen3-VL-2B", drift_2b),
                                            (r"Qwen3-VL-8B", drift_8b))):
        ax = fig.add_subplot(gs[0, 2 + 2*axi])
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        ax.grid(axis="x", visible=False)
        x = np.arange(2)
        bw = 0.35
        terse_vals = [drifts["L1_COARSE"], drifts["L4_VERY_FINE"]]
        wordy_vals = [drifts["L5_WORDY_SIMPLETON"], drifts["L8_WORDY_VERY_FINE"]]
        b1 = ax.bar(x - bw/2, terse_vals,  bw, color=PALETTE["primary"], label="terse (L1, L4)",  edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x + bw/2, wordy_vals, bw, color=PALETTE["wordy"],   label="wordy (L5, L8)", edgecolor="white", linewidth=0.5)
        ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=7)
        ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(["L1: presence", "L4: compositional"])
        ax.set_ylabel(r"mean $|\Delta|$  (perturbation drift)")
        ax.set_title(mtitle, fontsize=10)
        ax.set_ylim(0, max(terse_vals + wordy_vals) * 1.30)
        if axi == 0:
            ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.suptitle("Verbose prompts stabilise the model under image perturbation.\n"
                 "Same image, identical semantic content; only the prompt wording changes.\n"
                 "(Bars: mean $|\\Delta|$ over the full 78-perturbation bank.)",
                 y=1.10, fontsize=10, weight="bold")
    out = FIG_DIR / "teaser_fig1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# E1 — Question-generation pipeline figure for the appendix
# ---------------------------------------------------------------------------

def fig_pipeline() -> None:
    """Three-row layout, no box-text overflow.

    Row 1: image + scene-graph snippet (side by side).
    Row 2: 4 primary question boxes L1-L4.
    Row 3: 4 wordy-mirror boxes L5-L8.
    """
    import textwrap as _tw

    img = np.asarray(Image.open(IMG_PATH))
    fig = plt.figure(figsize=(13.0, 8.5))
    gs = fig.add_gridspec(3, 4,
                          height_ratios=[1.0, 1.0, 1.0],
                          hspace=0.50, wspace=0.18)

    # Row 1, col 0: image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for s in ("top", "right", "left", "bottom"): ax_img.spines[s].set_visible(False)
    ax_img.set_title("Source image", fontsize=10)

    # Row 1, cols 1-3: scene-graph snippet
    ax_sg = fig.add_subplot(gs[0, 1:])
    ax_sg.axis("off")
    ax_sg.set_xlim(0, 1); ax_sg.set_ylim(0, 1)
    ax_sg.set_title("Scene-graph snippet (input to the generator)", fontsize=10, pad=10)
    ax_sg.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.85,
                                   boxstyle="round,pad=0.015", fc="#fafaf7",
                                   ec="0.55", lw=0.6, transform=ax_sg.transAxes))
    sg_lines = [
        ("objects",    "{dog (x2), bed, wall, pillow, outlet, sheet, ...}"),
        ("attributes", "wall.color = white;   pillow.color = blue;   bed = large"),
        ("relations",  "dog on bed;   pillow on bed;   wall to-the-right-of outlet"),
    ]
    y0 = 0.78
    for i, (k, v) in enumerate(sg_lines):
        ax_sg.text(0.04, y0 - i * 0.22, k + ":", fontsize=10, family="monospace",
                   color="#2c4a7d", weight="bold", va="top")
        ax_sg.text(0.22, y0 - i * 0.22, v, fontsize=10, family="monospace",
                   color="0.18", va="top")
    ax_sg.text(0.04, 0.13,
               "L1/L3 alternate positive and negative phrasings by image index; "
               "L4 reuses L2's options unchanged (v8 design).",
               fontsize=9, color="0.35", style="italic")

    # Row 2: primary L1-L4
    primary = [
        ("L1", "object presence",         "Is there a bed in this image?",                          "(A) yes",   None),
        ("L2", "attribute query",         "What color is the wall?",                                "(C) white", "options: {white, +3 distractors}"),
        ("L3", "spatial relation (neg.)", "Is the outlet to the right of the wall?",                "(B) no",    None),
        ("L4", "compositional",           "What color is the wall that is to the right of the outlet?", "(C) white", "options inherited from L2"),
    ]
    for col, (lvl, title, q, gold, extra) in enumerate(primary):
        ax = fig.add_subplot(gs[1, col])
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.92,
                                    boxstyle="round,pad=0.02", fc="#eef3fb", ec="#4c72b0", lw=0.9,
                                    transform=ax.transAxes))
        ax.text(0.07, 0.90, f"{lvl} — {title}", fontsize=10, weight="bold",
                color="#2c4a7d", va="top")
        q_wrapped = _tw.fill(f'"{q}"', width=32)
        ax.text(0.07, 0.72, q_wrapped, fontsize=8.5, style="italic", va="top",
                color="0.15", linespacing=1.25)
        ax.text(0.07, 0.22, f"gold: {gold}", fontsize=9, weight="bold",
                color="#2c4a7d", va="top")
        if extra:
            ax.text(0.07, 0.12, extra, fontsize=7.5, color="0.4", va="top")

    # Row 3: wordy mirrors L5-L8 (compact, one consolidated text block)
    wordy_template_short = (
        '"Please read the question carefully and answer it directly. '
        '⟨primary question⟩ The extra wording is only polite framing."'
    )
    for col, (lvl, src) in enumerate([("L5", "L1"), ("L6", "L2"), ("L7", "L3"), ("L8", "L4")]):
        ax = fig.add_subplot(gs[2, col])
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.92,
                                    boxstyle="round,pad=0.02", fc="#fdf2e9", ec="#dd8452", lw=0.9,
                                    transform=ax.transAxes))
        ax.text(0.07, 0.90, f"{lvl} — wordy mirror of {src}", fontsize=10, weight="bold",
                color="#a04415", va="top")
        wrapped = _tw.fill(wordy_template_short, width=33)
        ax.text(0.07, 0.72, wrapped, fontsize=7.5, style="italic", va="top",
                color="0.15", linespacing=1.30)
        ax.text(0.07, 0.12, "same options, same gold", fontsize=7.5, color="0.4", va="top")

    fig.suptitle("Question-generation pipeline",
                 y=0.99, fontsize=12, weight="bold")
    out = FIG_DIR / "appendix_pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    fig_teaser()
    fig_pipeline()
    print(f"\nBoth figures regenerated in {FIG_DIR.relative_to(ROOT)}/")
