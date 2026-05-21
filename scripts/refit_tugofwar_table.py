#!/usr/bin/env python3
"""Re-fit the within-image fixed-effects tug-of-war regression on the four runs.

Reproduces Table 1 (tab:betas-fe) of the paper from the released run dirs.

For each run, this script:
  1. Loads exp1/complexity_points.json (1 row per (image, level), 4000 rows total).
  2. z-scores Csem, Cprompt, H_opt, and the outcome (loglik volatility).
  3. Residualises z-scored Csem on z-scored Cprompt (Frisch-Waugh-Lovell first stage).
  4. Demeans every variable within image_id (within-image fixed effects).
  5. Fits OLS of demeaned outcome on (Cres, Cprompt, H_opt) demeaned predictors.
  6. Prints the four standardised partial slopes + R^2.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/farooq/Public/vlm-robustness")
RUNS = {
    "CLEVR-2B": "qwen_clevr_2B/frequency_alignment_outputs_clevr_2B_500_20260516_132931",
    "CLEVR-8B": "qwen_clevr_8B/frequency_alignment_outputs_clevr_8B_500_20260516_131942",
    "GQA-2B":   "qwen_gpa_2B/frequency_alignment_outputs_gqa_2B_500_20260517_032513",
    "GQA-8B":   "qwen_gpa_8B/frequency_alignment_outputs_gqa_8B_500_20260516_125653",
}


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std(ddof=0)
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def fit_one_run(run_dir: Path) -> dict:
    pts = json.load(open(run_dir / "exp1" / "complexity_points.json"))
    csem, cpr, hopt, vol, imgs = [], [], [], [], []
    for r in pts:
        c = r.get("question_complexity_score")
        p = r.get("prompt_complexity_score")
        h = r.get("option_hardness_score")
        v = r.get("mean_loglik_volatility")
        if None in (c, p, h, v):
            continue
        csem.append(float(c)); cpr.append(float(p)); hopt.append(float(h)); vol.append(float(v))
        imgs.append(str(r.get("image_id", "")))
    csem_z, cpr_z = zscore(np.asarray(csem)), zscore(np.asarray(cpr))
    hopt_z, vol_z = zscore(np.asarray(hopt)), zscore(np.asarray(vol))
    b1 = np.sum(csem_z * cpr_z) / np.sum(cpr_z ** 2)
    cres_z = csem_z - b1 * cpr_z
    img_arr = np.asarray(imgs); uniq = np.unique(img_arr)
    img_codes = np.searchsorted(uniq, img_arr)
    def demean(x):
        out = np.empty_like(x)
        for k in range(len(uniq)):
            mask = img_codes == k
            out[mask] = x[mask] - x[mask].mean()
        return out
    X = np.column_stack([demean(cres_z), demean(cpr_z), demean(hopt_z)])
    y = demean(vol_z)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1.0 - np.sum((y - pred) ** 2) / max(np.sum(y ** 2), 1e-12)
    return {"beta_Cres": coef[0], "beta_Cprompt": coef[1], "beta_Hopt": coef[2], "R2": r2, "n": len(vol)}


def main() -> None:
    print(f"{'Run':<12} {'beta_Cres':>11} {'beta_Cprompt':>13} {'beta_Hopt':>11} {'R2':>7} {'N':>6}")
    print("-" * 70)
    for label, d in RUNS.items():
        r = fit_one_run(ROOT / d)
        print(f"{label:<12} {r['beta_Cres']:>+11.4f} {r['beta_Cprompt']:>+13.4f} {r['beta_Hopt']:>+11.4f} {r['R2']:>7.3f} {r['n']:>6}")


if __name__ == "__main__":
    main()
