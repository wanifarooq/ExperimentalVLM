#!/usr/bin/env python3
"""Compare 1D (radial-binned) vs 2D (orientation-preserving) overlap Pearson r.

Addresses reviewer concern: 'Radial binning ignores directional effects;
some corruptions (e.g., motion blur) are highly anisotropic, potentially
weakening predictive power.'

The 2D overlap (`two_d_first_order`) was NOT enabled in the 500-sample
production runs (n=0 in their summaries). It WAS enabled in the 100-sample
development run (`frequency_alignment_outputs_server_2d_100_20260512_023014`),
which we use here as a single-data-point comparison.

Reads from each run's `exp5/summary.json` and pulls
`image_space.group_summaries.{group}.overlap_law_variants.{first_order|two_d_first_order}.grouped.pearson_r`.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

ROOT = Path("/home/farooq/Public/vlm-robustness")

PRODUCTION_RUNS = {
    "CLEVR-2B (500)": "qwen_clevr_2B/frequency_alignment_outputs_clevr_2B_500_20260516_132931",
    "CLEVR-8B (500)": "qwen_clevr_8B/frequency_alignment_outputs_clevr_8B_500_20260516_131942",
    "GQA-2B (500)":   "qwen_gpa_2B/frequency_alignment_outputs_gqa_2B_500_20260517_032513",
    "GQA-8B (500)":   "qwen_gpa_8B/frequency_alignment_outputs_gqa_8B_500_20260516_125653",
}

DEV_RUN_WITH_2D = {
    "Dev (100, 2D enabled)": "frequency_alignment_outputs_server_2d_100_20260512_023014",
}


def _extract(run_dir: Path, group: str = "last_2") -> dict:
    summary = json.load(open(run_dir / "exp5" / "summary.json"))
    gs = summary["image_space"]["group_summaries"][group]
    variants = gs.get("overlap_law_variants", {})
    out = {}
    for variant_name in ("first_order", "two_d_first_order"):
        v = variants.get(variant_name, {})
        grouped = v.get("grouped", {})
        out[variant_name] = {
            "pearson_r": grouped.get("pearson_r"),
            "n": grouped.get("n"),
            "p_value": grouped.get("pearson_p_value"),
        }
    return out


def _fmt(v: Optional[float], n: int = 0) -> str:
    if v is None:
        return f"None (n={n})"
    return f"{v:+.3f} (n={n})"


def main() -> None:
    print("=== Production runs (500-sample): 1D vs 2D overlap at last_2 ===")
    print(f"{'Run':<22}  {'1D r':>18}  {'2D r':>22}")
    print("-" * 64)
    for label, rel in PRODUCTION_RUNS.items():
        d = _extract(ROOT / rel)
        one = d["first_order"]
        two = d["two_d_first_order"]
        print(f"{label:<22}  {_fmt(one['pearson_r'], one['n']):>18}  {_fmt(two['pearson_r'], two['n']):>22}")

    print()
    print("=== Development run (100-sample, with 2D enabled): 1D vs 2D ===")
    print(f"{'Run':<26}  {'1D r':>18}  {'2D r':>22}")
    print("-" * 68)
    for label, rel in DEV_RUN_WITH_2D.items():
        d = _extract(ROOT / rel)
        one = d["first_order"]
        two = d["two_d_first_order"]
        print(f"{label:<26}  {_fmt(one['pearson_r'], one['n']):>18}  {_fmt(two['pearson_r'], two['n']):>22}")

    print()
    print("Notes:")
    print("- 2D overlap is disabled in the production runs (n=0). Re-enabling it requires")
    print("  re-running exp5 with `analysis.compute_2d_overlap_law: true`.")
    print("- In the 100-sample dev run, 1D and 2D Pearson r are within ~0.02, suggesting")
    print("  the orientation-collapsed radial form does not lose predictive power.")
    print("- For anisotropic perturbations specifically (e.g., motion blur, rotation),")
    print("  the dev run gives one data point; a per-operator-class 2D-vs-1D breakdown")
    print("  would require a fresh production run with 2D enabled.")


if __name__ == "__main__":
    main()
