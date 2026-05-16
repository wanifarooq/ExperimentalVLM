#!/usr/bin/env python3
"""Reproducible verification of two drift-comparison confounds.

Pulls clean log-likelihood baselines and per-perturbation drifts from any saved
exp1 run (per_sample.jsonl) and reports whether either of the two confounds
that could compromise a "wordy is more robust" claim materialises in the data:

  Concern A (reviewer-style):
      "Maybe wordy starts at a LOWER clean log p (already drifted) and its
      smaller drift just reflects less room to fall."
      -> Falsified if wordy_clean > terse_clean in >>50% of cells.

  Concern B (ceiling-style):
      "Maybe wordy starts at a HIGHER clean log p (near ceiling) and its
      smaller drift just reflects mechanical compression near log p = 0."
      -> Empirically the direction that DOES hold in our runs; addressed by
      using volatility (shift-invariant std of drift) and z-scored partial
      regression slopes against attention-derived predictors (not log-probs).

For each (terse, wordy) pair this script reports:
  * mean clean log p on the gold answer, terse vs wordy
  * fraction of cells with wordy_clean > terse_clean
  * mean |drift| terse vs wordy, plus mean perturbed log p (end-state)
  * fraction of cells matching Concern A's worry scenario
  * fraction of cells matching Concern B's ceiling scenario (wordy clean above
    a chosen ceiling threshold)

Outputs:
  - prints a markdown-formatted summary to stdout
  - writes <run>/baseline_concerns_report.json and .md alongside the run

Usage:
  python scripts/verify_drift_baseline_concerns.py <run_dir>
  # or default to the saved 100-sample run:
  python scripts/verify_drift_baseline_concerns.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

DEFAULT_RUN = Path(
    "/home/farooq/Public/vlm-robustness/"
    "frequency_alignment_outputs_server_2d_100_20260512_023014"
)

LEVEL_PAIRS: List[Tuple[str, str]] = [
    ("L1_COARSE", "L5_WORDY_SIMPLETON"),
    ("L2_MEDIUM", "L6_WORDY_MEDIUM"),
    ("L3_FINE", "L7_WORDY_FINE"),
    ("L4_VERY_FINE", "L8_WORDY_VERY_FINE"),
]


def load_per_sample(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "exp1" / "per_sample.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing per_sample.jsonl at {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_cells(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (image_id, level-pair). Drops cells where either level is
    missing or has no clean score for its gold answer."""
    cells = []
    for d in records:
        for terse, wordy in LEVEL_PAIRS:
            lt = d["levels"].get(terse)
            lw = d["levels"].get(wordy)
            if not lt or not lw:
                continue
            at = lt.get("answer_label")
            aw = lw.get("answer_label")
            ct_scores = lt.get("clean", {}).get("scores") or {}
            cw_scores = lw.get("clean", {}).get("scores") or {}
            if at not in ct_scores or aw not in cw_scores:
                continue
            ct = float(ct_scores[at])
            cw = float(cw_scores[aw])
            drifts_t = [
                float(p["loglik_drift"]) for p in lt.get("perturbations", [])
                if "loglik_drift" in p
            ]
            drifts_w = [
                float(p["loglik_drift"]) for p in lw.get("perturbations", [])
                if "loglik_drift" in p
            ]
            if not drifts_t or not drifts_w:
                continue
            cells.append({
                "image_id": d["image_id"],
                "pair": f"{terse}/{wordy}",
                "terse_level": terse,
                "wordy_level": wordy,
                "clean_terse": ct,
                "clean_wordy": cw,
                "mean_abs_drift_terse": float(np.mean(np.abs(drifts_t))),
                "mean_abs_drift_wordy": float(np.mean(np.abs(drifts_w))),
                "std_drift_terse": float(np.std(drifts_t)),    # volatility
                "std_drift_wordy": float(np.std(drifts_w)),
                "n_perts_terse": len(drifts_t),
                "n_perts_wordy": len(drifts_w),
            })
    return cells


def summarise_pair(cells: List[Dict[str, Any]], pair_label: str,
                   ceiling_threshold: float = -1.0) -> Dict[str, Any]:
    rs = [c for c in cells if c["pair"] == pair_label]
    if not rs:
        return {"pair": pair_label, "n": 0}
    ct = np.array([c["clean_terse"] for c in rs])
    cw = np.array([c["clean_wordy"] for c in rs])
    dt = np.array([c["mean_abs_drift_terse"] for c in rs])
    dw = np.array([c["mean_abs_drift_wordy"] for c in rs])
    vol_t = np.array([c["std_drift_terse"] for c in rs])
    vol_w = np.array([c["std_drift_wordy"] for c in rs])
    end_t = ct - dt
    end_w = cw - dw
    return {
        "pair": pair_label,
        "n": int(len(rs)),
        "mean_clean_terse": float(ct.mean()),
        "mean_clean_wordy": float(cw.mean()),
        "delta_clean": float((cw - ct).mean()),
        "frac_wordy_clean_higher": float((cw > ct).mean()),
        "mean_abs_drift_terse": float(dt.mean()),
        "mean_abs_drift_wordy": float(dw.mean()),
        "mean_volatility_terse": float(vol_t.mean()),
        "mean_volatility_wordy": float(vol_w.mean()),
        "frac_wordy_drift_smaller": float((dw < dt).mean()),
        "mean_end_terse": float(end_t.mean()),
        "mean_end_wordy": float(end_w.mean()),
        "frac_wordy_end_higher": float((end_w > end_t).mean()),
        # Concern A worry: small wordy drift BUT worse wordy end-state.
        "frac_concernA_materialises": float(
            ((dw < dt) & (end_w < end_t)).mean()
        ),
        # Concern B near-ceiling: wordy clean above threshold (e.g. -1.0).
        "frac_wordy_near_ceiling": float((cw > ceiling_threshold).mean()),
        "ceiling_threshold": float(ceiling_threshold),
    }


def render_markdown(run_dir: Path, summaries: List[Dict[str, Any]],
                    overall: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Drift-baseline confound report\n")
    lines.append(f"Run directory: `{run_dir.name}`\n")
    lines.append(f"N image-level-pair cells used: **{overall['n_total']}**\n")
    lines.append("\n## Concern A (reviewer-style)\n")
    lines.append("> *Maybe wordy starts at a LOWER clean log p and its smaller drift "
                 "just reflects less room to fall.*\n")
    lines.append(f"- Cells with wordy clean log p **higher** than terse clean log p: "
                 f"**{overall['frac_wordy_higher']:.1%}**.\n")
    lines.append(f"- Cells matching the worry scenario (small wordy drift **and** "
                 f"worse wordy end-state): **{overall['frac_concernA']:.1%}**.\n")
    if overall["frac_wordy_higher"] > 0.5:
        lines.append("- **Verdict:** the direction Concern A worries about is NOT what "
                     "we observe; the opposite (wordy clean above terse clean) is the "
                     "dominant pattern. Concern A is empirically falsified.\n")
    else:
        lines.append("- **Verdict:** Concern A may be active in this run; flag for "
                     "manual review.\n")

    lines.append("\n## Concern B (ceiling effect)\n")
    lines.append("> *Maybe wordy starts at a HIGHER clean log p (near ceiling) and its "
                 "smaller drift just reflects mechanical compression.*\n")
    lines.append(f"- Cells with wordy clean log p above ceiling threshold "
                 f"(log p > {overall['ceiling_threshold']:.1f}): "
                 f"**{overall['frac_wordy_near_ceiling']:.1%}**.\n")
    lines.append("- **Resolution:** the paper's response variable is volatility "
                 "(within-cell std of drift), which is *shift-invariant*; and the "
                 "regression reports z-scored partial slopes against an "
                 "attention-derived predictor (first-order overlap or IPR bandwidth), "
                 "which has no log-prob units at all. Multiplicative compression of "
                 "drift cancels under z-scoring.\n")

    lines.append("\n## Per-pair summary\n")
    lines.append("| Pair | N | clean terse | clean wordy | Δclean | |drift| terse | "
                 "|drift| wordy | volatility terse | volatility wordy | "
                 "wordy↑clean | wordy↑end | concernA |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        if s["n"] == 0:
            continue
        lines.append(
            f"| {s['pair']} | {s['n']} | {s['mean_clean_terse']:.2f} | "
            f"{s['mean_clean_wordy']:.2f} | {s['delta_clean']:+.2f} | "
            f"{s['mean_abs_drift_terse']:.2f} | {s['mean_abs_drift_wordy']:.2f} | "
            f"{s['mean_volatility_terse']:.2f} | {s['mean_volatility_wordy']:.2f} | "
            f"{s['frac_wordy_clean_higher']:.0%} | {s['frac_wordy_end_higher']:.0%} | "
            f"{s['frac_concernA_materialises']:.0%} |"
        )
    lines.append("\nKey columns:")
    lines.append("- `Δclean` — mean (wordy clean − terse clean). Positive ⇒ wordy "
                 "lives higher up the log-prob axis.")
    lines.append("- `wordy↑clean` — share of cells where wordy clean > terse clean.")
    lines.append("- `wordy↑end` — share of cells where wordy *end-state* log p "
                 "(clean − mean|drift|) is also higher than terse.")
    lines.append("- `concernA` — share of cells matching the worry pattern: wordy "
                 "drift smaller AND wordy end-state worse. The reviewer-style "
                 "concern fires only if this is non-trivially large.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_RUN,
                        help="Path to a saved frequency_alignment_outputs_* directory")
    parser.add_argument("--ceiling-threshold", type=float, default=-1.0,
                        help="Wordy clean log p above this counts as near-ceiling")
    parser.add_argument("--quiet", action="store_true",
                        help="Skip stdout markdown")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    records = load_per_sample(run_dir)
    cells = extract_cells(records)
    if not cells:
        print("ERROR: no eligible (terse, wordy) cells found.", file=sys.stderr)
        sys.exit(2)

    summaries = [
        summarise_pair(cells, f"{t}/{w}", args.ceiling_threshold)
        for (t, w) in LEVEL_PAIRS
    ]

    n_total = len(cells)
    overall = {
        "n_total": n_total,
        "frac_wordy_higher": float(np.mean([
            c["clean_wordy"] > c["clean_terse"] for c in cells
        ])),
        "frac_wordy_drift_smaller": float(np.mean([
            c["mean_abs_drift_wordy"] < c["mean_abs_drift_terse"] for c in cells
        ])),
        "frac_concernA": float(np.mean([
            (c["mean_abs_drift_wordy"] < c["mean_abs_drift_terse"]) and
            ((c["clean_wordy"] - c["mean_abs_drift_wordy"]) <
             (c["clean_terse"] - c["mean_abs_drift_terse"]))
            for c in cells
        ])),
        "frac_wordy_near_ceiling": float(np.mean([
            c["clean_wordy"] > args.ceiling_threshold for c in cells
        ])),
        "ceiling_threshold": float(args.ceiling_threshold),
    }

    md = render_markdown(run_dir, summaries, overall)
    if not args.quiet:
        print(md)

    out_json = run_dir / "baseline_concerns_report.json"
    out_md = run_dir / "baseline_concerns_report.md"
    out_json.write_text(json.dumps({
        "run_dir": str(run_dir),
        "overall": overall,
        "per_pair": summaries,
        "n_cells": n_total,
        "level_pairs": LEVEL_PAIRS,
    }, indent=2))
    out_md.write_text(md)
    print(f"\nWrote {out_json}\nWrote {out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
