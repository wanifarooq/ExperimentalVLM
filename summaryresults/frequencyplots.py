#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse VLM frequency + Dirichlet analysis text report and make plots.

Usage:
    python plot_freq_dirichlet.py report.txt --outdir plots

Expected structure (like your file):

====== FREQUENCY ANALYSIS (Δ band energy: pert - base) ======
...
====== DIRICHLET ANALYSIS (vision token smoothness) ======
...
"""

import argparse
import os
import re
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# File IO
# ---------------------------

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------
# Parsing: FREQUENCY ANALYSIS
# ---------------------------

def parse_frequency_section(text: str) -> Dict[str, Any]:
    """
    Parse the FREQUENCY ANALYSIS section.

    Returns:
        freq_data[name] = {
            "all": {
                "low": {"mean", "std", "rel_mean", "rel_std"},
                "mid": {...},
                "high": {...},
                "n": int,
            },
            "flips": { ... same structure ... }
        }
    """
    freq_data: Dict[str, Any] = {}

    # Extract only the frequency block
    freq_match = re.search(
        r"====== FREQUENCY ANALYSIS.*?====== DIRICHLET ANALYSIS",
        text,
        flags=re.DOTALL
    )
    if not freq_match:
        raise ValueError("Could not find FREQUENCY ANALYSIS section.")
    freq_block = freq_match.group(0)

    # Main line regex (all samples)
    main_line_re = re.compile(
        r"""^
        (?P<name>[A-Za-z+/]+)\s+
        low=\s*(?P<low_mean>[-\d\.eE]+)\s*±\s*(?P<low_std>[-\d\.eE]+)\s*
        \(\s*(?P<low_rel_mean>[-\d\.eE]+)\s*±\s*(?P<low_rel_std>[-\d\.eE]+)\s*rel\)\s*\|
        \s*mid=\s*(?P<mid_mean>[-\d\.eE]+)\s*±\s*(?P<mid_std>[-\d\.eE]+)\s*
        \(\s*(?P<mid_rel_mean>[-\d\.eE]+)\s*±\s*(?P<mid_rel_std>[-\d\.eE]+)\s*rel\)\s*\|
        \s*high=\s*(?P<high_mean>[-\d\.eE]+)\s*±\s*(?P<high_std>[-\d\.eE]+)\s*
        \(\s*(?P<high_rel_mean>[-\d\.eE]+)\s*±\s*(?P<high_rel_std>[-\d\.eE]+)\s*rel\)
        \s*\(n=(?P<n>\d+)\)
        """,
        flags=re.VERBOSE
    )

    # Flips line regex
    flips_line_re = re.compile(
        r"""^\s*\[flips\]\s*
        low=\s*(?P<low_mean>[-\d\.eE]+)\s*±\s*(?P<low_std>[-\d\.eE]+)\s*
        \(\s*(?P<low_rel_mean>[-\d\.eE]+)\s*±\s*(?P<low_rel_std>[-\d\.eE]+)\s*rel\)\s*\|
        \s*mid=\s*(?P<mid_mean>[-\d\.eE]+)\s*±\s*(?P<mid_std>[-\d\.eE]+)\s*
        \(\s*(?P<mid_rel_mean>[-\d\.eE]+)\s*±\s*(?P<mid_rel_std>[-\d\.eE]+)\s*rel\)\s*\|
        \s*high=\s*(?P<high_mean>[-\d\.eE]+)\s*±\s*(?P<high_std>[-\d\.eE]+)\s*
        \(\s*(?P<high_rel_mean>[-\d\.eE]+)\s*±\s*(?P<high_rel_std>[-\d\.eE]+)\s*rel\)
        \s*\(n=(?P<n>\d+)\)
        """,
        flags=re.VERBOSE
    )

    lines = freq_block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        m = main_line_re.match(line)
        if m:
            name = m.group("name")
            freq_data[name] = {}
            freq_data[name]["all"] = {
                "low": {
                    "mean": float(m.group("low_mean")),
                    "std": float(m.group("low_std")),
                    "rel_mean": float(m.group("low_rel_mean")),
                    "rel_std": float(m.group("low_rel_std")),
                },
                "mid": {
                    "mean": float(m.group("mid_mean")),
                    "std": float(m.group("mid_std")),
                    "rel_mean": float(m.group("mid_rel_mean")),
                    "rel_std": float(m.group("mid_rel_std")),
                },
                "high": {
                    "mean": float(m.group("high_mean")),
                    "std": float(m.group("high_std")),
                    "rel_mean": float(m.group("high_rel_mean")),
                    "rel_std": float(m.group("high_rel_std")),
                },
                "n": int(m.group("n")),
            }

            # Next line: flips
            if i + 1 < len(lines):
                flips_line = lines[i + 1].rstrip()
                mf = flips_line_re.match(flips_line)
                if mf:
                    freq_data[name]["flips"] = {
                        "low": {
                            "mean": float(mf.group("low_mean")),
                            "std": float(mf.group("low_std")),
                            "rel_mean": float(mf.group("low_rel_mean")),
                            "rel_std": float(mf.group("low_rel_std")),
                        },
                        "mid": {
                            "mean": float(mf.group("mid_mean")),
                            "std": float(mf.group("mid_std")),
                            "rel_mean": float(mf.group("mid_rel_mean")),
                            "rel_std": float(mf.group("mid_rel_std")),
                        },
                        "high": {
                            "mean": float(mf.group("high_mean")),
                            "std": float(mf.group("high_std")),
                            "rel_mean": float(mf.group("high_rel_mean")),
                            "rel_std": float(mf.group("high_rel_std")),
                        },
                        "n": int(mf.group("n")),
                    }
                    i += 1  # skip flips line
        i += 1

    return freq_data


# ---------------------------
# Parsing: DIRICHLET ANALYSIS
# ---------------------------

def _parse_dirichlet_block(dir_block: str) -> Dict[str, Any]:
    """
    Internal: parse just the Dirichlet block (starting from '====== DIRICHLET...' line)
    using a robust, line-based approach.
    """
    dir_data: Dict[str, Any] = {}
    lines = dir_block.splitlines()

    # We'll start after the first two header lines:
    # '====== DIRICHLET ANALYSIS ...' and the descriptive line.
    i = 0
    # Skip until we see a line starting with '====== DIRICHLET'
    while i < len(lines) and not lines[i].startswith("====== DIRICHLET"):
        i += 1
    # skip header and description
    i += 2

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue

        # Look for lines like:
        # 'Translation  ΔE mean=   10.341± 67.487 (n=48000) | flips mean=   15.263± 68.559 (n=3111)'
        if "mean=" in line and "flips mean" in line:
            name = line.split()[0]

            # Find all 'mean = ... ± ... (n=...)' occurrences on this line
            matches = re.findall(
                r"mean=\s*([-\d\.eE]+)\s*±\s*([-\d\.eE]+)\s*\(n=(\d+)\)",
                line
            )
            if len(matches) >= 2:
                (all_mean, all_std, all_n), (flips_mean, flips_std, flips_n) = matches[:2]

                dir_data[name] = {
                    "deltaE": {
                        "all_mean": float(all_mean),
                        "all_std": float(all_std),
                        "all_n": int(all_n),
                        "flips_mean": float(flips_mean),
                        "flips_std": float(flips_std),
                        "flips_n": int(flips_n),
                    },
                    "corr_all": {},
                    "corr_nonflip": {},
                    "corr_flips": {},
                }

            # Next line: global correlations
            if i + 1 < len(lines) and "corr(dE," in lines[i + 1]:
                corr_line = lines[i + 1]
                pairs = re.findall(
                    r"corr\(dE,\s*([^)=]+)\)=\s*([-\d\.eE]+)",
                    corr_line
                )
                corr_all = {}
                for key, val in pairs:
                    key = key.strip()
                    if key == "drift":
                        k = "drift"
                    elif key == "Δlow":
                        k = "dlow"
                    elif key == "Δhigh":
                        k = "dhigh"
                    elif key == "Δlow/Δhigh":
                        k = "ratio"
                    else:
                        k = key
                    corr_all[k] = float(val)
                dir_data[name]["corr_all"] = corr_all

            # Next next line: [non-flip] corr(dE, Δlogprob_base)=...
            if i + 2 < len(lines) and "[non-flip]" in lines[i + 2]:
                nf_line = lines[i + 2]
                m = re.search(
                    r"corr\(dE,\s*Δlogprob_base\)=\s*([-\d\.eE]+)\s*\(n=(\d+)\)",
                    nf_line
                )
                if m:
                    dir_data[name]["corr_nonflip"] = {
                        "dlogprob": float(m.group(1)),
                        "n": int(m.group(2)),
                    }

            # Flips-only correlations
            if i + 3 < len(lines) and "[flips]" in lines[i + 3]:
                fl_line = lines[i + 3]
                pairs = re.findall(
                    r"corr\(dE,\s*([^)=]+)\)=\s*([-\d\.eE]+)",
                    fl_line
                )
                corr_flips = {}
                for key, val in pairs:
                    key = key.strip()
                    if key == "drift":
                        k = "drift"
                    elif key == "Δlow":
                        k = "dlow"
                    elif key == "Δhigh":
                        k = "dhigh"
                    elif key == "Δlow/Δhigh":
                        k = "ratio"
                    else:
                        k = key
                    corr_flips[k] = float(val)
                dir_data[name]["corr_flips"] = corr_flips

            i += 4
        else:
            i += 1

    return dir_data


def parse_dirichlet_section(text: str) -> Dict[str, Any]:
    """
    Extract and parse the DIRICHLET ANALYSIS section from the full text.
    """
    # Grab everything from the Dirichlet heading to the end of the file
    dir_match = re.search(
        r"====== DIRICHLET ANALYSIS.*",
        text,
        flags=re.DOTALL
    )
    if not dir_match:
        raise ValueError("Could not find DIRICHLET ANALYSIS section.")
    dir_block = dir_match.group(0)
    return _parse_dirichlet_block(dir_block)


# ---------------------------
# Plotting helpers
# ---------------------------

def _ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)


def plot_frequency_relative(freq_data: Dict[str, Any], outdir: str):
    """
    Plot ONLY relative band deltas (normalized by base band power):
    low_rel, mid_rel, high_rel, for all vs flips.
    """
    _ensure_outdir(outdir)

    names = list(freq_data.keys())
    x = np.arange(len(names))
    width = 0.35

    def gather_values(mode: str, band: str):
        return np.array([freq_data[name][mode][band]["rel_mean"] for name in names])

    for band in ["low", "mid", "high"]:
        all_vals = gather_values("all", band)
        flip_vals = gather_values("flips", band)

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, all_vals, width, label="all")
        plt.bar(x + width / 2, flip_vals, width, label="flips")

        plt.xticks(x, names, rotation=45, ha="right")
        plt.axhline(0.0, linestyle="--", linewidth=0.8)
        plt.grid(axis="y", linestyle=":", linewidth=0.5)

        plt.title(f"Relative Δ{band} band energy (normalized by base)")
        plt.ylabel("Relative Δ (pert - base) / base")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"freq_rel_{band}.png"), dpi=200)
        plt.close()


def plot_dirichlet_deltaE(dir_data: Dict[str, Any], outdir: str):
    """Bar plot of ΔE_dir means (all vs flips) per perturbation."""
    _ensure_outdir(outdir)

    names = list(dir_data.keys())
    x = np.arange(len(names))
    width = 0.35

    all_means = np.array([dir_data[name]["deltaE"]["all_mean"] for name in names])
    flip_means = np.array([dir_data[name]["deltaE"]["flips_mean"] for name in names])

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, all_means, width, label="all")
    plt.bar(x + width / 2, flip_means, width, label="flips")

    plt.xticks(x, names, rotation=45, ha="right")
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.grid(axis="y", linestyle=":", linewidth=0.5)

    plt.ylabel("ΔE_dir mean")
    plt.title("Dirichlet ΔE_dir (pert - base) means")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dirichlet_deltaE_means.png"), dpi=200)
    plt.close()


def plot_dirichlet_correlations(dir_data: Dict[str, Any], outdir: str):
    """Plot correlation bars for all-sample and flip-sample correlations."""
    _ensure_outdir(outdir)

    names = list(dir_data.keys())
    x = np.arange(len(names))
    width = 0.18

    corr_keys = ["drift", "dlow", "dhigh", "ratio"]
    labels = [
        "corr(dE, drift)",
        "corr(dE, Δlow)",
        "corr(dE, Δhigh)",
        "corr(dE, Δlow/Δhigh)",
    ]

    # All samples
    plt.figure(figsize=(10, 5))
    for i, (ck, lab) in enumerate(zip(corr_keys, labels)):
        vals = np.array([dir_data[name]["corr_all"].get(ck, np.nan) for name in names])
        plt.bar(x + (i - 1.5) * width, vals, width, label=lab)

    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.grid(axis="y", linestyle=":", linewidth=0.5)
    plt.ylabel("Correlation")
    plt.title("Dirichlet correlations (all samples)")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dirichlet_correlations_all.png"), dpi=200)
    plt.close()

    # Flips only
    plt.figure(figsize=(10, 5))
    for i, (ck, lab) in enumerate(zip(corr_keys, labels)):
        vals = np.array([dir_data[name]["corr_flips"].get(ck, np.nan) for name in names])
        plt.bar(x + (i - 1.5) * width, vals, width, label=lab)

    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.grid(axis="y", linestyle=":", linewidth=0.5)
    plt.ylabel("Correlation")
    plt.title("Dirichlet correlations (flip subset)")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dirichlet_correlations_flips.png"), dpi=200)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot VLM frequency + Dirichlet analysis from a text report."
    )
    parser.add_argument("input", help="Path to text report file.")
    parser.add_argument("--outdir", default="plots", help="Directory to save plots.")
    args = parser.parse_args()

    text = load_text(args.input)

    freq_data = parse_frequency_section(text)
    dir_data = parse_dirichlet_section(text)

    print("Frequency perturbations parsed:", list(freq_data.keys()))
    print("Dirichlet perturbations parsed:", list(dir_data.keys()))

    plot_frequency_relative(freq_data, args.outdir)
    plot_dirichlet_deltaE(dir_data, args.outdir)
    plot_dirichlet_correlations(dir_data, args.outdir)

    print(f"Plots saved in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
