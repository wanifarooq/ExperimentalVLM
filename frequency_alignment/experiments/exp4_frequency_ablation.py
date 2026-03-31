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
    exp4/per_sample.json          -- per-sample cutoff sweeps
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

from ..analysis.continuous import summarize_by_score, summarize_linear_trend
from ..analysis.statistics import (
    monotonicity_test,
    spearman_correlation,
)
from ..data.base import ExperimentResult, GranularityLevel
from ..data.complexity import ensure_level_complexity
from ..data.loaders import load_multilevel_vqa_dataset
from ..models import get_adapter
from ..perturbations.frequency_sweep import compute_critical_cutoff, frequency_sweep
from ..utils.device import select_device
from ..utils.io import save_json

logger = logging.getLogger(__name__)


def _build_complexity_points(
    per_sample: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for record in per_sample:
        image_id = str(record.get("image_id"))
        for mode, mode_data in record.get("modes", {}).items():
            cutoffs = mode_data.get("cutoffs", [])
            if not cutoffs:
                continue
            for level_key, level_data in mode_data.get("levels", {}).items():
                correct = level_data.get("correct_at_cutoff", [])
                if not correct:
                    continue
                accuracy = [1.0 if value else 0.0 for value in correct]
                points.append(
                    {
                        "image_id": image_id,
                        "mode": mode,
                        "level": level_key,
                        "question": level_data.get("question"),
                        "question_type": level_data.get("question_type"),
                        "complexity_score": float(
                            level_data.get("complexity_score", 0.0) or 0.0
                        ),
                        "question_complexity_score": float(
                            level_data.get("question_complexity_score", 0.0) or 0.0
                        ),
                        "prompt_complexity_score": float(
                            level_data.get("prompt_complexity_score", 0.0) or 0.0
                        ),
                        "critical_cutoff": float(
                            compute_critical_cutoff(accuracy, cutoffs, threshold=threshold)
                        ),
                    }
                )
    return points


def _summarize_complexity(
    points: List[Dict[str, Any]],
    sweep_modes: List[str],
) -> Dict[str, Any]:
    if not points:
        return {}
    complexity_values = [
        float(point.get("complexity_score", 0.0))
        for point in points
        if point.get("complexity_score") is not None
    ]
    summary: Dict[str, Any] = {
        "score_name": "complexity_score",
        "score_definition": "semantic prompt atoms (question semantics plus non-boolean option semantics)",
        "score_min": float(min(complexity_values)) if complexity_values else 0.0,
        "score_max": float(max(complexity_values)) if complexity_values else 0.0,
        "num_points": len(points),
        "per_mode": {},
    }
    for mode in sweep_modes:
        mode_points = [point for point in points if point.get("mode") == mode]
        summary["per_mode"][mode] = {
            "mean_critical_cutoff_by_score": summarize_by_score(
                mode_points,
                score_key="complexity_score",
                value_key="critical_cutoff",
            )
        }
    return summary


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
        device_map=model_cfg.get("device_map"),
        local_files_only=cfg.get("offline", False),
        attn_implementation=model_cfg.get("attn_implementation"),
        attention_extract_implementation=model_cfg.get(
            "attention_extract_implementation", "eager"
        ),
    )

    # --- Load dataset ---
    samples = load_multilevel_vqa_dataset(cfg, max_samples=max_samples)
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
                complexity = ensure_level_complexity(level_data)

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
                    "question": level_data.question,
                    "question_type": level_data.question_type,
                    "complexity_score": complexity["complexity_score"],
                    "question_complexity_score": complexity["question_complexity_score"],
                    "prompt_complexity_score": complexity["prompt_complexity_score"],
                    "semantic_atoms": complexity["semantic_atoms"],
                    "prompt_semantic_atoms": complexity["prompt_semantic_atoms"],
                    "semantic_atom_counts": complexity["semantic_atom_counts"],
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

    complexity_points = _build_complexity_points(per_sample, accuracy_threshold)
    agg["complexity_analysis"] = _summarize_complexity(complexity_points, sweep_modes)

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

    tests["continuous_complexity"] = {}
    for mode in sweep_modes:
        mode_points = [point for point in complexity_points if point.get("mode") == mode]
        tests["continuous_complexity"][mode] = summarize_linear_trend(
            mode_points,
            x_key="complexity_score",
            y_key="critical_cutoff",
        )

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
    save_json(complexity_points, out_dir / "complexity_points.json")
    save_json(
        {"cutoffs": reference_cutoffs, "data": agg["per_mode"]},
        out_dir / "accuracy_curves.json",
    )
    save_json(per_sample, out_dir / "per_sample.json")

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
