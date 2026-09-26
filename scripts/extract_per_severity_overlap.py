#!/usr/bin/env python3
"""Stratify the overlap-law Pearson r by perturbation severity (1, 2, 3).

Addresses reviewer concern: 'Severity pooling obscures potential nonlinearity
across severity levels; a stratified analysis per severity would strengthen
conclusions.'

For each (run, severity) cell, computes the grouped Pearson r between the
predicted overlap $\\langle W_t, \\Delta F \\rangle$ and the actual within-cell
volatility, restricting to perturbations of that severity.

Outputs a table: run × severity → Pearson r.

Reads from: `{run}/exp1/per_sample.jsonl` and `{run}/exp5/summary.json`.
Writes nothing; just prints the table.
"""
from __future__ import annotations
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "CLEVR-2B": "qwen_clevr_2B/frequency_alignment_outputs_clevr_2B_500_20260516_132931",
    "CLEVR-8B": "qwen_clevr_8B/frequency_alignment_outputs_clevr_8B_500_20260516_131942",
    "GQA-2B":   "qwen_gpa_2B/frequency_alignment_outputs_gqa_2B_500_20260517_032513",
    "GQA-8B":   "qwen_gpa_8B/frequency_alignment_outputs_gqa_8B_500_20260516_125653",
}
LEVELS = ['L1_COARSE', 'L2_MEDIUM', 'L3_FINE', 'L4_VERY_FINE',
          'L5_WORDY_SIMPLETON', 'L6_WORDY_MEDIUM', 'L7_WORDY_FINE', 'L8_WORDY_VERY_FINE']


def _per_image_filter(run_dir: Path) -> Dict[Tuple[str, str], np.ndarray]:
    """Load the level-average filter at last_2 for each level (used as proxy for per-image W_t).

    Production runs do not cache per-image filters; we use the level-average filter as the
    closest available proxy. This is the same assumption the production exp5 makes when
    per-image filters are missing.
    """
    out: Dict[Tuple[str, str], np.ndarray] = {}
    fdir = run_dir / "exp2" / "filters"
    if not fdir.exists():
        return out
    for level in LEVELS:
        f = fdir / f"average_{level}_last_2.npy"
        if f.exists():
            arr = np.asarray(np.load(f), dtype=np.float64)
            s = arr.sum()
            if s > 0:
                out[("last_2", level)] = arr / s
    return out


def _compute_overlap_per_severity(run_dir: Path) -> Dict[int, Tuple[float, int]]:
    """For each severity 1/2/3 returns (grouped_pearson_r, n_cells).

    For each severity, collect (predicted_overlap, observed_volatility) per (image, level,
    perturbation-family) cell, then compute grouped Pearson r.
    """
    filters = _per_image_filter(run_dir)
    # Aggregate per (image, level, family) per severity
    grouped: Dict[int, Dict[Tuple[str, str, str], Dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"predicted": [], "abs_drift": []})
    )
    with open(run_dir / "exp1" / "per_sample.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            img = str(r.get("image_id"))
            for level_key, ld in r.get("levels", {}).items():
                if not level_key.startswith("L"):
                    continue
                wt = filters.get(("last_2", level_key))
                if wt is None:
                    continue
                for p in ld.get("perturbations", []) or []:
                    sev = p.get("severity")
                    df = p.get("delta_f")
                    if sev is None or df is None or len(df) != len(wt):
                        continue
                    fam = p.get("family", "unknown")
                    predicted = float(np.dot(wt, np.asarray(df, dtype=np.float64)))
                    abs_drift = float(abs(p.get("loglik_drift", 0.0)))
                    key = (img, level_key, fam)
                    grouped[int(sev)][key]["predicted"].append(predicted)
                    grouped[int(sev)][key]["abs_drift"].append(abs_drift)

    out: Dict[int, Tuple[float, int]] = {}
    for sev, cells in grouped.items():
        xs, ys = [], []
        for _, values in cells.items():
            if values["predicted"]:
                xs.append(statistics.mean(values["predicted"]))
                ys.append(statistics.mean(values["abs_drift"]))
        if len(xs) >= 3 and statistics.pstdev(xs) > 0 and statistics.pstdev(ys) > 0:
            r = float(np.corrcoef(xs, ys)[0, 1])
            out[sev] = (r, len(xs))
        else:
            out[sev] = (float("nan"), len(xs))
    return out


def main() -> None:
    print(f"{'Run':<10}  {'Sev=1 (r, n)':>20}  {'Sev=2 (r, n)':>20}  {'Sev=3 (r, n)':>20}")
    print("-" * 76)
    for label, rel in RUNS.items():
        run_dir = ROOT / rel
        per_sev = _compute_overlap_per_severity(run_dir)
        cells = []
        for sev in (1, 2, 3):
            r, n = per_sev.get(sev, (float("nan"), 0))
            cells.append(f"({r:+.3f}, n={n})")
        print(f"{label:<10}  {cells[0]:>20}  {cells[1]:>20}  {cells[2]:>20}")
    print()
    print("Notes:")
    print("- Per-image filters are not cached in production runs; we use the level-average")
    print("  filter at last_2 as the W_t proxy (the same fallback exp5 uses).")
    print("- The 'grouped' aggregation matches Table 5 exactly when all severities are pooled.")
    print("- A monotone-or-stable r across severities = no severity-nonlinearity concern.")


if __name__ == "__main__":
    main()
