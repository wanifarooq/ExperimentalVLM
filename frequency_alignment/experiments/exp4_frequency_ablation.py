"""Experiment 4: Synthetic Frequency Ablation.

Sweep a lowpass or highpass cutoff from near-DC to Nyquist, producing
filtered images at each step.  For each cutoff × granularity level, score
the MCQ and find the critical frequency ω_c* where accuracy drops to 50%.

The hypothesis is that ω_c* increases with task granularity:
    ω_c*(L1) < ω_c*(L2) < ω_c*(L3) < ω_c*(L4)

because finer tasks require higher-frequency information to answer correctly.

Outputs:
    exp4/summary.json             -- critical cutoffs and hypothesis tests
    exp4/accuracy_curves.json     -- accuracy vs cutoff per level
    exp4/hypothesis_tests.json    -- monotonicity and Spearman tests
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from ..analysis.statistics import (
    monotonicity_test,
    spearman_correlation,
)
from ..data.base import ExperimentResult, GranularityLevel
from ..data.gqa import build_granularity_dataset
from ..models import get_adapter
from ..perturbations.frequency_sweep import compute_critical_cutoff, frequency_sweep
from ..utils.device import select_device
from ..utils.io import save_json

logger = logging.getLogger(__name__)


def run_exp4(
    cfg: dict,
    out_dir: Path,
    results_so_far: Dict[int, ExperimentResult],
) -> ExperimentResult:
    """Run Experiment 4: Synthetic Frequency Ablation.

    Sweep lowpass/highpass cutoff, score MCQ at each step, find critical
    cutoff ω_c* per level.
    """
    logger.info("=" * 50)
    logger.info("Experiment 4: Synthetic Frequency Ablation")
    logger.info("=" * 50)

    exp_cfg = cfg.get("experiments", {}).get("exp4", {})
    max_samples = exp_cfg.get("max_samples", 500)
    sweep_steps = exp_cfg.get("sweep_steps", 20)
    sweep_modes = exp_cfg.get("sweep_modes", ["lowpass", "highpass"])
    num_bands = cfg.get("analysis", {}).get("num_bands", 10)
    seed = cfg.get("seed", 42)
    accuracy_threshold = exp_cfg.get("accuracy_threshold", 0.5)

    # --- Load model ---
    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("primary", "Qwen/Qwen3-VL-8B-Instruct")
    device = select_device(cfg.get("device", "auto"))
    quantization = model_cfg.get("quantization")

    logger.info("Loading model: %s", model_id)
    adapter = get_adapter(model_id)
    adapter.load(
        model_id=model_id,
        device=device,
        cache_dir=cfg.get("cache_dir"),
        quantization=quantization,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    # --- Load dataset ---
    cache_dir = Path(cfg.get("cache_dir", ".hf_cache"))
    data_cfg = cfg.get("data", {})
    samples = build_granularity_dataset(
        cache_dir=cache_dir,
        max_samples=max_samples,
        seed=seed,
        allow_download=data_cfg.get("allow_download", True),
    )
    logger.info("Loaded %d samples", len(samples))

    if not samples:
        return ExperimentResult(
            experiment_id=4,
            experiment_name="exp4_frequency_ablation",
            config=exp_cfg,
            metrics={"error": "no_samples"},
        )

    # --- Evaluation loop ---
    # Collect accuracy at each (mode, cutoff, level)
    # Structure: {mode: {level: {cutoff_idx: [correct_bools]}}}
    accuracy_data: Dict[str, Dict[str, Dict[int, List[bool]]]] = {}
    for mode in sweep_modes:
        accuracy_data[mode] = {}
        for level in GranularityLevel:
            accuracy_data[mode][level.name] = defaultdict(list)

    per_sample: List[Dict[str, Any]] = []
    t0 = time.time()

    for idx, sample in enumerate(samples):
        logger.info("Processing sample %d/%d: %s", idx + 1, len(samples), sample.image_id)

        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as e:
            logger.warning("Cannot open image %s: %s", sample.image_path, e)
            continue

        sample_record: Dict[str, Any] = {
            "image_id": sample.image_id,
            "modes": {},
        }

        for mode in sweep_modes:
            # Generate filtered images at each cutoff
            sweep_results = frequency_sweep(image, num_steps=sweep_steps, mode=mode)
            cutoffs = [c for c, _ in sweep_results]

            mode_record: Dict[str, Any] = {"cutoffs": cutoffs, "levels": {}}

            for level in GranularityLevel:
                level_data = sample.levels.get(level)
                if level_data is None:
                    continue
                level_key = level.name

                # Score at each cutoff
                accuracies_at_cutoff: List[bool] = []
                for ci, (cutoff, filtered_img) in enumerate(sweep_results):
                    try:
                        scores = adapter.score_options(
                            filtered_img, level_data.question, level_data.options,
                        )
                        pred = max(scores, key=scores.get)
                        correct = pred == level_data.answer_label
                    except Exception:
                        correct = False

                    accuracies_at_cutoff.append(correct)
                    accuracy_data[mode][level_key][ci].append(correct)

                mode_record["levels"][level_key] = {
                    "correct_at_cutoff": accuracies_at_cutoff,
                }

            sample_record["modes"][mode] = mode_record

        per_sample.append(sample_record)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info("Progress: %d/%d (%.2f samples/s)", idx + 1, len(samples),
                        (idx + 1) / elapsed)

    total_time = time.time() - t0
    logger.info("Ablation complete: %d samples in %.1fs", len(per_sample), total_time)

    # --- Aggregate accuracy curves ---
    # Get cutoff values from sweep
    reference_cutoffs = np.linspace(0.02, 0.5, sweep_steps).tolist()

    agg: Dict[str, Any] = {"per_mode": {}}
    critical_cutoffs: Dict[str, Dict[str, float]] = {}

    for mode in sweep_modes:
        agg["per_mode"][mode] = {}
        critical_cutoffs[mode] = {}

        level_order = ["L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE"]
        for lk in level_order:
            # Compute mean accuracy at each cutoff
            acc_curve = []
            for ci in range(sweep_steps):
                bools = accuracy_data[mode].get(lk, {}).get(ci, [])
                acc = float(np.mean(bools)) if bools else 0.0
                acc_curve.append(acc)

            # Find critical cutoff
            omega_c = compute_critical_cutoff(
                acc_curve, reference_cutoffs, threshold=accuracy_threshold,
            )

            agg["per_mode"][mode][lk] = {
                "accuracy_curve": acc_curve,
                "cutoffs": reference_cutoffs,
                "critical_cutoff": omega_c,
                "num_samples": len(accuracy_data[mode].get(lk, {}).get(0, [])),
            }
            critical_cutoffs[mode][lk] = omega_c

    # --- Hypothesis tests ---
    tests: Dict[str, Any] = {}

    for mode in sweep_modes:
        present_levels = [lk for lk in level_order if lk in critical_cutoffs.get(mode, {})]
        if len(present_levels) < 2:
            continue

        cutoff_values = [critical_cutoffs[mode][lk] for lk in present_levels]

        # H1: Monotonicity of ω_c* across levels
        is_mono, tau = monotonicity_test(cutoff_values)
        tests[f"monotonicity_{mode}"] = {
            "is_monotonic": is_mono,
            "kendall_tau": tau,
            "values": dict(zip(present_levels, cutoff_values)),
            "passed": tau > 0.6,
        }

        # H2: Spearman correlation
        ranks = list(range(1, len(present_levels) + 1))
        rho, p = spearman_correlation(ranks, cutoff_values)
        tests[f"spearman_cutoff_vs_granularity_{mode}"] = {
            "rho": rho,
            "p_value": p,
            "passed": rho > 0.8,
            "values": dict(zip(present_levels, cutoff_values)),
        }

    # Summary: lowpass is the main test
    lp_mono = tests.get("monotonicity_lowpass", {}).get("passed", False)
    lp_spear = tests.get("spearman_cutoff_vs_granularity_lowpass", {}).get("passed", False)
    tests["hypothesis_supported"] = lp_mono or lp_spear

    # --- Log ---
    logger.info("-" * 40)
    for mode in sweep_modes:
        logger.info("Mode: %s", mode)
        for lk in level_order:
            if lk in critical_cutoffs.get(mode, {}):
                logger.info("  %s: ω_c* = %.4f", lk, critical_cutoffs[mode][lk])
    sp = tests.get("spearman_cutoff_vs_granularity_lowpass", {})
    logger.info(
        "  Spearman(granularity, ω_c*): rho=%.3f, p=%.4f [%s]",
        sp.get("rho", 0), sp.get("p_value", 1),
        "PASS" if sp.get("passed") else "FAIL",
    )
    logger.info("-" * 40)

    # --- Save ---
    save_json(agg, out_dir / "summary.json")
    save_json(tests, out_dir / "hypothesis_tests.json")
    save_json(
        {"cutoffs": reference_cutoffs, "data": agg["per_mode"]},
        out_dir / "accuracy_curves.json",
    )

    try:
        adapter.unload()
    except Exception:
        pass

    metrics = {
        "num_samples": len(per_sample),
        "total_time_s": total_time,
        "sweep_steps": sweep_steps,
    }
    for mode in sweep_modes:
        for lk in level_order:
            if lk in critical_cutoffs.get(mode, {}):
                metrics[f"omega_c_{mode}_{lk}"] = critical_cutoffs[mode][lk]

    return ExperimentResult(
        experiment_id=4,
        experiment_name="exp4_frequency_ablation",
        config=exp_cfg,
        metrics=metrics,
        per_sample=per_sample,
        hypothesis_tests=tests,
    )
