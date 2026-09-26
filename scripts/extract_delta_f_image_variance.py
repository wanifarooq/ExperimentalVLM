#!/usr/bin/env python3
"""Quantify per-image variance of the perturbation spectral signature ΔF.

Addresses reviewer concern: 'The ΔF(ω) signature is averaged over images;
per-image perturbation spectra may vary, limiting overlap precision at the
cell level.'

Clarifying point this script supports: Eq. (2) in §3.2 uses E_I[...]
(population mean) as the canonical visual signature, but the overlap law
in §3.3 and §5.4 operates on the PER-IMAGE ΔF (not the population mean).
The averaging is purely for displaying canonical signatures.

This script computes, per operator and per severity, the std of ΔF mass
in each radial band across the 500 images. A small std => low per-image
variability => the population-mean signature is representative.

Reads per-image ΔF arrays from `{run}/exp1/per_sample.jsonl`.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "GQA-2B":   "qwen_gpa_2B/frequency_alignment_outputs_gqa_2B_500_20260517_032513",
    "CLEVR-2B": "qwen_clevr_2B/frequency_alignment_outputs_clevr_2B_500_20260516_132931",
}


def _operator_name(perturbation_name: str) -> str:
    """Stable operator label, independent of severity/parameters."""
    text = (perturbation_name or "unknown").split("|sev", 1)[0].split("(", 1)[0]
    return text.strip() or "unknown"


def main() -> None:
    # We use just one perturbation level (L1) since the perturbation list is identical across levels.
    for run_label, rel in RUNS.items():
        # operator -> list of per-image ΔF arrays
        op_arrays: Dict[str, List[np.ndarray]] = defaultdict(list)
        with open(ROOT / rel / "exp1" / "per_sample.jsonl") as f:
            for line in f:
                r = json.loads(line)
                lvl = r.get("levels", {}).get("L1_COARSE", {})
                for p in lvl.get("perturbations", []) or []:
                    df = p.get("delta_f")
                    if df is None:
                        continue
                    op_arrays[_operator_name(p.get("name", ""))].append(
                        np.asarray(df, dtype=np.float64)
                    )
        # For each operator, compute the mean ΔF per band and the std per band across images.
        # Report a single scalar: mean CV across bands (std/mean per band, averaged).
        print(f"\n=== {run_label} ===")
        print(f"{'Operator':<22} {'mean band CV (%)':>20} {'n_images':>12}")
        print("-" * 60)
        rows = []
        for op, arrs in op_arrays.items():
            if not arrs:
                continue
            M = np.stack(arrs, axis=0)  # (n_images, B)
            n = M.shape[0]
            mean_per_band = np.mean(M, axis=0)
            std_per_band = np.std(M, axis=0)
            # CV per band; only meaningful where mean > 0
            mask = mean_per_band > 1e-10
            if not mask.any():
                cv = float("nan")
            else:
                cv = float(np.mean(std_per_band[mask] / mean_per_band[mask]) * 100)
            rows.append((op, cv, n))
        rows.sort(key=lambda x: x[1])
        for op, cv, n in rows:
            print(f"{op:<22} {cv:>19.1f}% {n:>12d}")
    print()
    print("Notes:")
    print("- CV is the mean across radial bands of (std-across-images / mean-across-images).")
    print("- Low CV (< 30%) means the per-image ΔF is close to its population mean →")
    print("  the population-mean signature in Eq. (2) is representative.")
    print("- High CV (> 100%) means per-image variability dominates and ΔF is highly image-")
    print("  dependent. Note: the OVERLAP computation in §3.3 and §5.4 uses the PER-IMAGE")
    print("  ΔF (not the population mean), so per-cell precision is unaffected.")


if __name__ == "__main__":
    main()
