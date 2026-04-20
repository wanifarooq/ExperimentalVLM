#!/usr/bin/env python3
"""Main CLI entry point for frequency alignment experiments.

Usage:
    python -m frequency_alignment.run_experiment --experiment 1 --config frequency_alignment/configs/default.yaml
    python -m frequency_alignment.run_experiment --experiment all --config frequency_alignment/configs/local_test.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

from .utils.io import load_yaml, merge_configs, save_json, ensure_dir
from .utils.device import select_device, get_dtype, gpu_memory_gb
from .utils.band_selection import resolve_num_bands_config
from .data.base import ExperimentResult

logger = logging.getLogger("frequency_alignment")

# Default config path (relative to repo root)
_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.yaml"

# Experiment execution order respecting dependencies:
# {1, 2, 4, 6} are independent; 3 needs Exp2; 5 needs Exp1+Exp2
_INDEPENDENT = [1, 2, 4, 6]
_DEPENDS_ON_2 = [3]
_DEPENDS_ON_1_AND_2 = [5]
_ALL_ORDER = _INDEPENDENT + _DEPENDS_ON_2 + _DEPENDS_ON_1_AND_2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frequency Alignment Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="all",
        help="Experiment to run: 1-6 or 'all' (default: all)",
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="YAML config file path",
    )
    p.add_argument("--model-id", type=str, default=None, help="Override model ID")
    p.add_argument("--device", type=str, default=None, help="Override device")
    p.add_argument("--max-samples", type=int, default=None, help="Override max samples")
    p.add_argument("--out-dir", type=str, default=None, help="Override output directory")
    p.add_argument("--cache-dir", type=str, default=None, help="Override cache directory")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    p.add_argument("--offline", action="store_true", help="HF offline mode")
    p.add_argument("--plot-only", action="store_true",
                   help="Generate plots from existing results (no experiments)")
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Load YAML config and apply CLI overrides."""
    cfg = load_yaml(args.config)

    # CLI overrides take precedence
    if args.model_id:
        cfg.setdefault("model", {})["primary"] = args.model_id
    if args.device:
        cfg["device"] = args.device
    if args.max_samples is not None:
        cfg.setdefault("data", {})["max_samples"] = args.max_samples
        # Also override per-experiment max_samples
        for exp_key in cfg.get("experiments", {}):
            cfg["experiments"][exp_key]["max_samples"] = args.max_samples
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.cache_dir:
        cfg["cache_dir"] = args.cache_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.offline:
        cfg["offline"] = True

    return cfg


def resolve_experiments(experiment_arg: str) -> List[int]:
    """Parse --experiment flag into list of experiment IDs."""
    if experiment_arg == "all":
        return list(_ALL_ORDER)
    parts = experiment_arg.replace(",", " ").split()
    exp_ids = []
    for part in parts:
        eid = int(part)
        if eid < 1 or eid > 6:
            raise ValueError(f"Experiment ID must be 1-6, got {eid}")
        exp_ids.append(eid)
    return exp_ids


def run_single_experiment(
    exp_id: int,
    cfg: dict,
    out_dir: Path,
    results_so_far: Dict[int, ExperimentResult],
) -> ExperimentResult:
    """Dispatch to the appropriate experiment runner."""

    exp_cfg = cfg.get("experiments", {}).get(f"exp{exp_id}", {})
    if not exp_cfg.get("enabled", True):
        logger.info("Experiment %d is disabled in config, skipping.", exp_id)
        return ExperimentResult(
            experiment_id=exp_id,
            experiment_name=f"exp{exp_id}",
            config=exp_cfg,
            metrics={"skipped": True},
        )

    exp_out = ensure_dir(out_dir / f"exp{exp_id}")

    if exp_id == 1:
        from .experiments.exp1_granularity import run_exp1
        return run_exp1(cfg, exp_out, results_so_far)
    elif exp_id == 2:
        from .experiments.exp2_attention_frequency import run_exp2
        return run_exp2(cfg, exp_out, results_so_far)
    elif exp_id == 3:
        from .experiments.exp3_fusion_drift import run_exp3
        return run_exp3(cfg, exp_out, results_so_far)
    elif exp_id == 4:
        from .experiments.exp4_frequency_ablation import run_exp4
        return run_exp4(cfg, exp_out, results_so_far)
    elif exp_id == 5:
        from .experiments.exp5_overlap_prediction import run_exp5
        return run_exp5(cfg, exp_out, results_so_far)
    elif exp_id == 6:
        from .experiments.exp6_segmentation_granularity import run_exp6
        return run_exp6(cfg, exp_out, results_so_far)
    else:
        raise ValueError(f"Unknown experiment ID: {exp_id}")


def main() -> None:
    args = parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = build_config(args)

    # Plot-only mode: skip experiments, just regenerate plots
    if args.plot_only:
        out_dir = Path(cfg.get("out_dir", "frequency_alignment_outputs"))
        logger.info("Plot-only mode: generating plots from %s", out_dir)
        from .plotting import generate_all_plots
        generate_all_plots(out_dir, cfg)
        return

    exp_ids = resolve_experiments(args.experiment)
    cfg = resolve_num_bands_config(cfg, exp_ids)

    # Device info
    device = select_device(cfg.get("device", "auto"))
    dtype = get_dtype(device)
    vram = gpu_memory_gb(device)
    logger.info("Device: %s  dtype: %s  VRAM: %.1f GB", device, dtype, vram)
    num_bands_resolution = cfg.get("analysis", {}).get("num_bands_resolution", {})
    if num_bands_resolution:
        logger.info(
            "Spectral bins: requested=%s resolved=%s mode=%s status=%s",
            num_bands_resolution.get("requested_num_bands", cfg.get("analysis", {}).get("num_bands")),
            num_bands_resolution.get("resolved_num_bands", cfg.get("analysis", {}).get("num_bands")),
            num_bands_resolution.get("mode", cfg.get("analysis", {}).get("num_bands_mode", "fixed")),
            num_bands_resolution.get("status", "unknown"),
        )

    # Output directory
    out_dir = ensure_dir(cfg.get("out_dir", "frequency_alignment_outputs"))
    logger.info("Output directory: %s", out_dir)
    logger.info("Experiments to run: %s", exp_ids)

    # Save config snapshot
    save_json(cfg, out_dir / "config_snapshot.json")

    # Run experiments in dependency order
    results: Dict[int, ExperimentResult] = {}
    for exp_id in exp_ids:
        logger.info("=" * 60)
        logger.info("Starting Experiment %d", exp_id)
        logger.info("=" * 60)
        t0 = time.time()
        try:
            result = run_single_experiment(exp_id, cfg, out_dir, results)
            results[exp_id] = result
            elapsed = time.time() - t0
            logger.info(
                "Experiment %d completed in %.1f s", exp_id, elapsed
            )
        except Exception:
            logger.exception("Experiment %d failed", exp_id)
            raise

    # Save combined results
    combined = {
        f"exp{eid}": {
            "metrics": r.metrics,
            "hypothesis_tests": r.hypothesis_tests,
        }
        for eid, r in results.items()
    }
    save_json(combined, out_dir / "combined" / "all_results.json")

    # Generate plots from results
    try:
        from .plotting import generate_all_plots
        generate_all_plots(out_dir, cfg)
    except Exception:
        logger.exception("Plot generation failed (non-fatal)")

    logger.info("All experiments complete. Results at: %s", out_dir)


if __name__ == "__main__":
    main()
