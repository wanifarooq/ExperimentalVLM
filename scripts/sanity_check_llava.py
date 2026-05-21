#!/usr/bin/env python3
"""Smoke-test a LLaVA-OneVision run before committing to a 500-sample sweep.

Runs the experiment pipeline with --max-samples=5 (or user-specified),
tees the stderr+stdout to a log file, then post-processes the run dir:

  1. Counts `LLaVA patch grid for n_tokens=...` warnings emitted by the
     adapter's exact-vs-padded fallback. If a non-trivial fraction of images
     hit the padded path, the per-image spectral W_t carries FFT zero-padding
     bias and the per-image overlap correlations will be noisier.
  2. Verifies W_t invariants on every level in exp2/summary.json:
       sum(W_t) == 1, all W_t >= 0, IPR in [1, B].
  3. Reports patch-grid distribution (how many distinct grids, exact vs padded).
  4. Confirms every sample produced cross-attention (no silent extract failures).
  5. Prints a single PASS/FAIL line you can grep in CI.

Usage:
  python scripts/sanity_check_llava.py \
      --config frequency_alignment/configs/llava_onevision_7b.yaml \
      --samples 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

PADDED_GRID_WARN_RE = re.compile(
    r"LLaVA patch grid for n_tokens=(\d+) has no in-band exact factor"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path,
                   help="Path to LLaVA yaml config")
    p.add_argument("--samples", type=int, default=5,
                   help="Number of samples to run (default 5)")
    p.add_argument("--experiment", default="all",
                   help="Which experiment(s) to run (default: all)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override out_dir; default is auto-generated sanity dir")
    p.add_argument("--keep-log", action="store_true",
                   help="Keep the captured stdout/stderr log alongside the run dir")
    p.add_argument("--skip-run", action="store_true",
                   help="Skip subprocess; only post-process the existing --out-dir")
    return p.parse_args()


def make_default_out_dir(config_path: Path, samples: int) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    model_id = cfg.get("model", {}).get("primary", "unknown")
    short = model_id.split("/")[-1].replace(".", "").replace("-hf", "")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / f"frequency_alignment_outputs_sanity_{short}_{samples}samples_{stamp}"


def run_pipeline(config: Path, samples: int, experiment: str, out_dir: Path,
                 log_path: Path) -> int:
    """Launch the experiment pipeline as a subprocess, tee stdout+stderr to log."""
    cmd = [
        sys.executable, "-m", "frequency_alignment.run_experiment",
        "--config", str(config),
        "--experiment", experiment,
        "--max-samples", str(samples),
        "--out-dir", str(out_dir),
        "--verbose",
    ]
    print(f"[sanity] launching: {' '.join(cmd)}", flush=True)
    print(f"[sanity] log -> {log_path}", flush=True)
    with log_path.open("w") as logf:
        logf.write(f"# command: {' '.join(cmd)}\n# started: {datetime.now()}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        rc = proc.wait()
        logf.write(f"\n# finished: {datetime.now()}  rc={rc}\n")
    return rc


def count_padding_warnings(log_path: Path) -> Tuple[int, List[int]]:
    n_tokens_seen: List[int] = []
    for line in log_path.read_text().splitlines():
        m = PADDED_GRID_WARN_RE.search(line)
        if m:
            n_tokens_seen.append(int(m.group(1)))
    return len(n_tokens_seen), n_tokens_seen


def check_wt_invariants(summary_path: Path) -> Dict[str, Dict[str, float]]:
    """Per-level: sum, IPR, min, all-finite, all-nonneg."""
    if not summary_path.exists():
        return {"_error": {"reason": f"missing {summary_path}"}}
    d = json.loads(summary_path.read_text())
    out: Dict[str, Dict[str, float]] = {}
    for lvl, ld in d.get("per_level", {}).items():
        wt = ld.get("W_t_average")
        if not isinstance(wt, list):
            out[lvl] = {"reason": "no W_t_average"}
            continue
        arr = np.asarray(wt, dtype=np.float64)
        if arr.size == 0:
            out[lvl] = {"reason": "empty W_t"}
            continue
        sum_w = float(arr.sum())
        sq = float((arr * arr).sum())
        ipr = (1.0 / sq) if sq > 0 else float("inf")
        out[lvl] = {
            "B": int(arr.size),
            "sum": sum_w,
            "min": float(arr.min()),
            "ipr": ipr,
            "nonneg": bool(np.all(arr >= -1e-9)),
            "finite": bool(np.all(np.isfinite(arr))),
            "sum_ok": bool(abs(sum_w - 1.0) < 1e-4),
            "ipr_in_range": bool(1.0 - 1e-6 <= ipr <= arr.size + 1e-6),
        }
    return out


def collect_patch_grids(out_dir: Path) -> Dict[str, int]:
    """Tally patch_grid values across Exp2 power_spectra artifacts."""
    grids: Dict[str, int] = {}
    for p in out_dir.rglob("power_spectra.json"):
        try:
            records = json.loads(p.read_text())
        except Exception:
            continue
        for record in records:
            for level_payload in record.get("levels", {}).values():
                pg = level_payload.get("patch_grid")
                key = str(pg) if pg else "(none)"
                grids[key] = grids.get(key, 0) + 1
        if grids:
            return grids

    # Legacy fallback for older diagnostic outputs.
    for p in out_dir.rglob("per_sample.jsonl"):
        with p.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pg = r.get("patch_grid") or r.get("levels", {}).get("L1_COARSE", {}).get("patch_grid")
                key = str(pg) if pg else "(none)"
                grids[key] = grids.get(key, 0) + 1
    return grids


def count_attention_failures(out_dir: Path) -> Tuple[int, int]:
    """How many Exp2 (sample, level) cells are missing a usable W_t."""
    total = 0
    failed = 0
    for p in out_dir.rglob("power_spectra.json"):
        try:
            records = json.loads(p.read_text())
        except Exception:
            continue
        for record in records:
            for level_payload in record.get("levels", {}).values():
                total += 1
                wt = level_payload.get("W_t")
                patch_grid = level_payload.get("patch_grid")
                if not wt or not patch_grid:
                    failed += 1
        if total:
            return failed, total

    # Legacy fallback for older diagnostic outputs.
    for p in out_dir.rglob("per_sample.jsonl"):
        with p.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for lvl, ld in r.get("levels", {}).items():
                    total += 1
                    # exp2 stores per-sample W_t under "filters" or "spectral_features";
                    # exp1 stores "vision_feature_status". Failures show as empty/None.
                    if ld is None:
                        failed += 1
    return failed, total


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"[sanity] ERROR: config not found: {args.config}", file=sys.stderr)
        return 2

    out_dir = args.out_dir or make_default_out_dir(args.config, args.samples)
    out_dir = out_dir.resolve()
    log_path = out_dir.parent / f"{out_dir.name}.log"

    if not args.skip_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rc = run_pipeline(args.config, args.samples, args.experiment, out_dir, log_path)
        elapsed = time.time() - t0
        print(f"\n[sanity] pipeline rc={rc}, wall={elapsed:.1f}s", flush=True)
        if rc != 0:
            print(f"[sanity] FAIL: pipeline returned non-zero rc; inspect {log_path}", flush=True)
            return rc
    else:
        if not log_path.exists():
            print(f"[sanity] WARNING: --skip-run set but {log_path} not present; padding-warning check will be empty",
                  file=sys.stderr)
            log_path.touch()
        if not out_dir.exists():
            print(f"[sanity] ERROR: --skip-run set but {out_dir} does not exist", file=sys.stderr)
            return 2

    # -- POST-PROCESS --------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY REPORT")
    print("=" * 70)

    # 1. Padding-warning count
    n_warn, n_tokens_padded = count_padding_warnings(log_path)
    print(f"\n[1] Padded-grid fallback (FFT zero-padding bias) warnings: {n_warn}")
    if n_tokens_padded:
        from collections import Counter
        ctr = Counter(n_tokens_padded)
        for n, c in ctr.most_common(5):
            print(f"      n_tokens={n}  hit {c}x")

    # 2. Patch grid distribution
    grids = collect_patch_grids(out_dir)
    print(f"\n[2] Distinct patch grids observed in Exp2 artifacts: {len(grids)}")
    for g, c in sorted(grids.items(), key=lambda t: -t[1])[:6]:
        print(f"      {g}: {c} samples")

    # 3. W_t invariants
    print("\n[3] W_t invariants (exp2/summary.json):")
    exp2_summary = out_dir / "exp2" / "summary.json"
    inv = check_wt_invariants(exp2_summary)
    if "_error" in inv:
        print(f"      ERROR: {inv['_error']['reason']}")
        wt_pass = False
    else:
        wt_pass = True
        for lvl, m in inv.items():
            if "reason" in m:
                print(f"      {lvl:25s} {m['reason']}")
                wt_pass = False
                continue
            flags = []
            if not m["sum_ok"]: flags.append(f"sum={m['sum']:.4f}!=1")
            if not m["nonneg"]: flags.append(f"min={m['min']:.2e}<0")
            if not m["finite"]: flags.append("nonfinite")
            if not m["ipr_in_range"]: flags.append(f"IPR={m['ipr']:.3f} oor [1,{m['B']}]")
            ok = not flags
            wt_pass = wt_pass and ok
            tag = "OK " if ok else "FAIL"
            print(f"      {lvl:25s} B={m['B']:>3}  sum={m['sum']:.4f}  IPR={m['ipr']:>6.3f}  [{tag}]"
                  + (f"  {'; '.join(flags)}" if flags else ""))

    # 4. Attention failures (silent extract failures)
    fail, total = count_attention_failures(out_dir)
    print(f"\n[4] Exp2 sample-level cells with missing W_t/patch_grid: {fail}/{total}")

    # 5. Verdict
    fraction_padded = (n_warn / max(total, 1)) if total else 0.0
    padding_ok = fraction_padded < 0.20  # tolerate <20% padded
    overall = wt_pass and padding_ok and fail == 0
    print("\n" + "=" * 70)
    print(f"VERDICT: {'PASS' if overall else 'FAIL'}")
    print(f"  W_t invariants:        {'PASS' if wt_pass else 'FAIL'}")
    print(f"  Padded fraction <20%:  {'PASS' if padding_ok else 'FAIL'}  ({fraction_padded:.1%})")
    print(f"  No missing Exp2 cells: {'PASS' if fail == 0 else 'FAIL'}")
    print("=" * 70)
    print(f"\nrun dir: {out_dir}")
    print(f"log:     {log_path}")
    if not args.keep_log and overall and not args.skip_run:
        pass  # leave the log; cheap

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
