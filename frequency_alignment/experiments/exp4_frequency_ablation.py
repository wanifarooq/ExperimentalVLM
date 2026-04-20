"""Experiment 4: Synthetic Frequency Ablation.

Sweep a lowpass or highpass cutoff from near-DC to Nyquist, producing
filtered images at each step.  For each cutoff × granularity level, score
the MCQ and find the critical frequency ω_c* where accuracy drops to 50%.

The hypothesis gate is that ω_c* decreases with task granularity:
    ω_c*(L1) > ω_c*(L2) > ω_c*(L3) > ω_c*(L4)

because finer tasks become fragile when less of the required frequency support remains.

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

from ..analysis.continuous import (
    attach_complexity_residual,
    summarize_by_score,
    summarize_fixed_effects_trend,
    summarize_horse_race_view,
    summarize_linear_trend,
    summarize_multivariate_regression,
)
from ..analysis.gate_thresholds import GATES
from ..analysis.level_views import LEVEL_VIEW_ORDER, LEVEL_VIEWS
from ..analysis.statistics import (
    monotonicity_test,
    spearman_correlation,
)
from ..data.base import ExperimentResult, GranularityLevel, ALL_VQA_LEVEL_NAMES, PRIMARY_VQA_LEVEL_NAMES
from ..data.complexity import ensure_level_complexity
from ..data.complexity import (
    OPTION_HARDNESS_SCORE_DEFINITION,
    OPTION_HARDNESS_SCORE_NAME,
    PROMPT_COMPLEXITY_SCORE_DEFINITION,
    PROMPT_COMPLEXITY_SCORE_NAME,
    SEMANTIC_COMPLEXITY_SCORE_DEFINITION,
    SEMANTIC_COMPLEXITY_SCORE_NAME,
)
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
                        "option_hardness_score": float(
                            level_data.get("option_hardness_score", 0.0) or 0.0
                        ),
                        "critical_cutoff": float(
                            compute_critical_cutoff(accuracy, cutoffs, threshold=threshold)
                        ),
                    }
                )
    attach_complexity_residual(points)
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
        "score_name": SEMANTIC_COMPLEXITY_SCORE_NAME,
        "score_definition": SEMANTIC_COMPLEXITY_SCORE_DEFINITION,
        "control_score_name": PROMPT_COMPLEXITY_SCORE_NAME,
        "control_score_definition": PROMPT_COMPLEXITY_SCORE_DEFINITION,
        "option_hardness_score_name": OPTION_HARDNESS_SCORE_NAME,
        "option_hardness_score_definition": OPTION_HARDNESS_SCORE_DEFINITION,
        "logic_residual_key": "complexity_score_residual",
        "logic_residual_definition": "Residual of semantic complexity after linear regression on prompt load.",
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
            ),
            "mean_critical_cutoff_by_prompt_load": summarize_by_score(
                mode_points,
                score_key="prompt_complexity_score",
                value_key="critical_cutoff",
            ),
            "mean_critical_cutoff_by_option_hardness": summarize_by_score(
                mode_points,
                score_key="option_hardness_score",
                value_key="critical_cutoff",
            ),
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
                    "option_hardness_score": float(getattr(level_data, "option_hardness_score", 0.0) or 0.0),
                    "semantic_atoms": complexity["semantic_atoms"],
                    "prompt_semantic_atoms": complexity["prompt_semantic_atoms"],
                    "semantic_atom_counts": complexity["semantic_atom_counts"],
                    "option_hardness_components": dict(
                        getattr(level_data, "option_hardness_components", {}) or {}
                    ),
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

        level_order = list(ALL_VQA_LEVEL_NAMES)
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

    # Compatibility summary used by shared plot/report checks.
    primary_mode = "lowpass" if "lowpass" in sweep_modes else sweep_modes[0]
    agg["per_level"] = {}
    agg["per_level_perturbation"] = {}
    for lk in level_order:
        if lk not in critical_cutoffs.get(primary_mode, {}):
            continue
        mode_entries: Dict[str, Any] = {}
        cutoff_values: List[float] = []
        for mode in sweep_modes:
            mode_payload = agg["per_mode"].get(mode, {}).get(lk)
            if mode_payload is None:
                continue
            cutoff_value = float(mode_payload["critical_cutoff"])
            cutoff_values.append(cutoff_value)
            mode_entries[mode] = {
                "critical_cutoff": cutoff_value,
                "mean_accuracy": float(np.mean(mode_payload["accuracy_curve"])),
                "n": int(mode_payload["num_samples"]),
            }
        agg["per_level"][lk] = {
            "critical_cutoff": float(critical_cutoffs[primary_mode][lk]),
            "mean_critical_cutoff": float(np.mean(cutoff_values)) if cutoff_values else 0.0,
            "primary_mode": primary_mode,
            "num_modes": len(mode_entries),
            "num_samples": int(agg["per_mode"][primary_mode][lk]["num_samples"]),
        }
        agg["per_level_perturbation"][lk] = mode_entries

    complexity_points = _build_complexity_points(per_sample, accuracy_threshold)
    agg["complexity_analysis"] = _summarize_complexity(complexity_points, sweep_modes)

    # --- Hypothesis tests ---
    tests: Dict[str, Any] = {}

    for mode in sweep_modes:
        present_levels = [lk for lk in PRIMARY_VQA_LEVEL_NAMES if lk in critical_cutoffs.get(mode, {})]
        if len(present_levels) < 2:
            continue

        cutoff_values = [critical_cutoffs[mode][lk] for lk in present_levels]

        # H1: Monotonicity of ω_c* across levels
        is_mono, tau = monotonicity_test(cutoff_values)
        is_decreasing = all(
            cutoff_values[idx] >= cutoff_values[idx + 1]
            for idx in range(len(cutoff_values) - 1)
        )
        tests[f"monotonicity_{mode}"] = {
            "is_monotonic": is_decreasing,
            "kendall_tau": tau,
            "values": dict(zip(present_levels, cutoff_values)),
            "target": f"kendall_tau < -{GATES['monotonicity_kendall_tau_min']} (critical cutoff decreases with granularity)",
            "passed": tau < -GATES["monotonicity_kendall_tau_min"],
        }

        # H2: Spearman correlation
        ranks = list(range(1, len(present_levels) + 1))
        rho, p = spearman_correlation(ranks, cutoff_values)
        tests[f"spearman_cutoff_vs_granularity_{mode}"] = {
            "rho": rho,
            "p_value": p,
            "target": f"rho < -{GATES['strong_spearman_rho_min']} (critical cutoff decreases with granularity)",
            "passed": rho < -GATES["strong_spearman_rho_min"],
            "values": dict(zip(present_levels, cutoff_values)),
            "caveat": (
                "Spearman p-value gating over four level means is underpowered; "
                "directional cutoff monotonicity is the primary gate."
            ),
        }

        wordy_present = [
            level
            for level in ALL_VQA_LEVEL_NAMES
            if level in (LEVEL_VIEWS["wordy"] or set()) and level in critical_cutoffs.get(mode, {})
        ]
        if len(wordy_present) >= 2:
            wordy_values = [critical_cutoffs[mode][level] for level in wordy_present]
            wordy_mono, wordy_tau = monotonicity_test(wordy_values)
            wordy_rho, wordy_p = spearman_correlation(range(1, len(wordy_values) + 1), wordy_values)
            tests[f"monotonicity_{mode}_wordy"] = {
                "is_monotonic": all(
                    wordy_values[idx] >= wordy_values[idx + 1]
                    for idx in range(len(wordy_values) - 1)
                ),
                "kendall_tau": wordy_tau,
                "values": dict(zip(wordy_present, wordy_values)),
                "target": f"kendall_tau < -{GATES['monotonicity_kendall_tau_min']}",
                "passed": wordy_tau < -GATES["monotonicity_kendall_tau_min"],
            }
            tests[f"spearman_cutoff_vs_granularity_{mode}_wordy"] = {
                "rho": wordy_rho,
                "p_value": wordy_p,
                "passed": wordy_rho < -GATES["secondary_spearman_rho_min"],
                "values": dict(zip(wordy_present, wordy_values)),
                "target": f"rho < -{GATES['secondary_spearman_rho_min']} (secondary wordy-ladder monotonicity only; no pooled monotonicity is run)",
            }

    tests["continuous_complexity"] = {}
    for mode in sweep_modes:
        mode_points = [point for point in complexity_points if point.get("mode") == mode]
        tests["continuous_complexity"][mode] = {
            "critical_cutoff_vs_complexity": summarize_linear_trend(
                mode_points,
                x_key="complexity_score",
                y_key="critical_cutoff",
            ),
            "critical_cutoff_vs_prompt_load": summarize_linear_trend(
                mode_points,
                x_key="prompt_complexity_score",
                y_key="critical_cutoff",
            ),
            "critical_cutoff_vs_option_hardness": summarize_linear_trend(
                mode_points,
                x_key="option_hardness_score",
                y_key="critical_cutoff",
            ),
            "critical_cutoff_vs_complexity_fixed_effects": summarize_fixed_effects_trend(
                mode_points,
                group_key="image_id",
                x_key="complexity_score",
                y_key="critical_cutoff",
            ),
            "horse_race_critical_cutoff": summarize_multivariate_regression(
                mode_points,
                y_key="critical_cutoff",
                x_keys=[
                    "question_complexity_score",
                    "prompt_complexity_score",
                    "option_hardness_score",
                ],
            ),
            "horse_race_critical_cutoff_within_image": summarize_multivariate_regression(
                mode_points,
                y_key="critical_cutoff",
                x_keys=[
                    "question_complexity_score",
                    "prompt_complexity_score",
                    "option_hardness_score",
                ],
                group_key="image_id",
                demean_by_group=True,
            ),
        }
        view_payload: Dict[str, Any] = {}
        for view_name in LEVEL_VIEW_ORDER:
            level_filter = LEVEL_VIEWS[view_name]
            view_points = (
                mode_points
                if level_filter is None
                else [point for point in mode_points if str(point.get("level")) in level_filter]
            )
            view_entry = summarize_horse_race_view(
                view_points,
                y_key="critical_cutoff",
                x_keys=[
                    "question_complexity_score",
                    "prompt_complexity_score",
                    "option_hardness_score",
                ],
                view_name=view_name,
                level_filter=None,
            )
            view_entry.update({
                "level_filter": sorted(level_filter) if level_filter is not None else None,
                "n_points": len(view_points),
            })
            view_payload[view_name] = view_entry
            if view_name == "primary":
                tests["continuous_complexity"][mode][f"horse_race_critical_cutoff_{view_name}"] = (
                    view_entry["marginal"]
                )
            else:
                tests["continuous_complexity"][mode][f"horse_race_critical_cutoff_{view_name}"] = (
                    view_entry["pooled"]
                )
                tests["continuous_complexity"][mode][f"horse_race_critical_cutoff_within_image_{view_name}"] = (
                    view_entry["within_image"]
                )
        tests["continuous_complexity"][mode]["horse_race_critical_cutoff_views"] = view_payload

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
