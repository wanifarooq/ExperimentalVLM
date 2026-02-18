"""Experiment 3: Pre/Post Fusion Drift.

For a subset of perturbations, hook the vision encoder output (pre-fusion)
and early decoder layers (post-fusion), then measure how the perturbation-
induced drift is amplified or attenuated across frequency bands.

The amplification ratio R(ω) = ||ΔZ(ω)|| / ||ΔV(ω)|| should correlate
with the task-specific filter W_t(ω) from Experiment 2, confirming that
language conditioning selectively amplifies drift in its attended bands.

Depends on: Experiment 2 outputs (W_t filters).

Outputs:
    exp3/summary.json           -- aggregate amplification ratios
    exp3/hypothesis_tests.json  -- correlation R(ω) vs W_t(ω)
    exp3/amplification.json     -- per-sample amplification data
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from ..analysis.drift import (
    compute_amplification_ratio,
    compute_band_drift,
    compute_scalar_drift,
)
from ..analysis.statistics import (
    pearson_correlation,
    spearman_correlation,
    bootstrap_ci,
)
from ..data.base import ExperimentResult, GranularityLevel
from ..data.gqa import build_granularity_dataset
from ..models import get_adapter
from ..perturbations import build_perturbation_suite
from ..utils.device import select_device
from ..utils.io import save_json

logger = logging.getLogger(__name__)


def _load_W_t_from_exp2(
    results_so_far: Dict[int, ExperimentResult],
    exp2_out_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load average W_t filters from Experiment 2 outputs."""
    filters: Dict[str, np.ndarray] = {}

    # Try loading from saved npy files
    filters_dir = exp2_out_dir / "filters"
    for level_key in ["L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE"]:
        path = filters_dir / f"average_{level_key}.npy"
        if path.exists():
            filters[level_key] = np.load(path)

    if filters:
        logger.info("Loaded W_t filters from %s: %s", filters_dir, list(filters.keys()))
        return filters

    # Try from results_so_far
    exp2_result = results_so_far.get(2)
    if exp2_result and exp2_result.metrics.get("error") is None:
        # Extract from hypothesis_tests or per_sample
        logger.info("Attempting to extract W_t from exp2 in-memory results")
        # This is a fallback; in practice the npy files should exist
        pass

    logger.warning("No W_t filters found from Experiment 2")
    return filters


def run_exp3(
    cfg: dict,
    out_dir: Path,
    results_so_far: Dict[int, ExperimentResult],
) -> ExperimentResult:
    """Run Experiment 3: Pre/Post Fusion Drift.

    Hook pre-fusion (vision encoder) and post-fusion (decoder) features,
    measure band-decomposed drift under perturbations, compute amplification
    ratio R(ω), and correlate with W_t(ω).
    """
    logger.info("=" * 50)
    logger.info("Experiment 3: Pre/Post Fusion Drift")
    logger.info("=" * 50)

    exp_cfg = cfg.get("experiments", {}).get("exp3", {})
    max_samples = exp_cfg.get("max_samples", 200)
    pert_subset = exp_cfg.get("perturbation_subset", 10)
    num_bands = cfg.get("analysis", {}).get("num_bands", 10)
    seed = cfg.get("seed", 42)

    # Load W_t from Experiment 2
    base_out = Path(cfg.get("out_dir", "frequency_alignment_outputs"))
    W_t_filters = _load_W_t_from_exp2(results_so_far, base_out / "exp2")

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
            experiment_id=3,
            experiment_name="exp3_fusion_drift",
            config=exp_cfg,
            metrics={"error": "no_samples"},
        )

    # --- Perturbation config ---
    pert_cfg_global = cfg.get("perturbations", {})
    severity_levels = pert_cfg_global.get("severity_levels", [1, 2, 3])

    # --- Evaluation loop ---
    per_sample: List[Dict[str, Any]] = []
    # Collect amplification ratios per level
    level_amplifications: Dict[str, List[np.ndarray]] = defaultdict(list)
    level_pre_drifts: Dict[str, List[float]] = defaultdict(list)
    level_post_drifts: Dict[str, List[float]] = defaultdict(list)
    t0 = time.time()

    for idx, sample in enumerate(samples):
        logger.info("Processing sample %d/%d: %s", idx + 1, len(samples), sample.image_id)

        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as e:
            logger.warning("Cannot open image %s: %s", sample.image_path, e)
            continue

        # Build perturbations (use subset for speed)
        perturbations = build_perturbation_suite(
            image,
            severity_levels=severity_levels,
            include_natural=True,
            include_frequency=True,
            num_bands=num_bands,
            seed=seed + idx,
        )
        if pert_subset and len(perturbations) > pert_subset:
            # Keep diverse subset: first N from different families
            perturbations = perturbations[:pert_subset]

        if not perturbations:
            continue

        sample_record: Dict[str, Any] = {"image_id": sample.image_id, "levels": {}}

        for level in GranularityLevel:
            level_data = sample.levels.get(level)
            if level_data is None:
                continue
            level_key = level.name

            prompt = level_data.question
            if level_data.options:
                opts_str = " ".join(
                    f"({k}) {v}" for k, v in sorted(level_data.options.items())
                )
                prompt = f"{prompt} Options: {opts_str}"

            # Extract clean pre/post fusion features
            try:
                clean_internals = adapter.extract_internals(
                    image, prompt,
                    extract_attention=False,
                    extract_pre_fusion=True,
                    extract_post_fusion=True,
                )
            except Exception as e:
                logger.warning("Clean extraction failed: %s", e)
                continue

            if clean_internals.pre_fusion_features is None or \
               clean_internals.post_fusion_features is None:
                continue

            clean_pre = clean_internals.pre_fusion_features.cpu().float().numpy()
            clean_post = clean_internals.post_fusion_features.cpu().float().numpy()

            level_pert_records = []

            for pr in perturbations:
                try:
                    pert_internals = adapter.extract_internals(
                        pr.perturbed_image, prompt,
                        extract_attention=False,
                        extract_pre_fusion=True,
                        extract_post_fusion=True,
                    )
                except Exception as e:
                    continue

                if pert_internals.pre_fusion_features is None or \
                   pert_internals.post_fusion_features is None:
                    continue

                pert_pre = pert_internals.pre_fusion_features.cpu().float().numpy()
                pert_post = pert_internals.post_fusion_features.cpu().float().numpy()

                # Band-decomposed drift
                pre_drift_bands = compute_band_drift(clean_pre, pert_pre, num_bands)
                post_drift_bands = compute_band_drift(clean_post, pert_post, num_bands)

                # Amplification ratio
                R_omega = compute_amplification_ratio(pre_drift_bands, post_drift_bands)

                # Scalar drifts
                pre_scalar = compute_scalar_drift(clean_pre, pert_pre)
                post_scalar = compute_scalar_drift(clean_post, pert_post)

                level_amplifications[level_key].append(R_omega)
                level_pre_drifts[level_key].append(pre_scalar)
                level_post_drifts[level_key].append(post_scalar)

                level_pert_records.append({
                    "perturbation": pr.name,
                    "pre_drift_scalar": pre_scalar,
                    "post_drift_scalar": post_scalar,
                    "amplification_ratio": R_omega.tolist(),
                    "mean_amplification": float(R_omega.mean()),
                })

            sample_record["levels"][level_key] = {
                "num_perturbations": len(level_pert_records),
                "perturbations": level_pert_records,
            }

        per_sample.append(sample_record)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info("Progress: %d/%d (%.2f samples/s)", idx + 1, len(samples),
                        (idx + 1) / elapsed)

    total_time = time.time() - t0
    logger.info("Drift analysis complete: %d samples in %.1fs", len(per_sample), total_time)

    # --- Aggregate ---
    agg: Dict[str, Any] = {"per_level": {}}
    level_order = ["L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE"]
    present_levels = [lk for lk in level_order if lk in level_amplifications]

    for lk in present_levels:
        R_list = level_amplifications[lk]
        if R_list:
            R_mean = np.mean(np.stack(R_list), axis=0)
        else:
            R_mean = np.zeros(num_bands)

        agg["per_level"][lk] = {
            "mean_amplification_ratio": R_mean.tolist(),
            "mean_pre_drift": float(np.mean(level_pre_drifts[lk])) if level_pre_drifts[lk] else 0.0,
            "mean_post_drift": float(np.mean(level_post_drifts[lk])) if level_post_drifts[lk] else 0.0,
            "overall_amplification": float(R_mean.mean()),
            "num_observations": len(R_list),
        }

    # --- Hypothesis tests ---
    tests: Dict[str, Any] = {}

    # H1: R(ω) correlates with W_t(ω) per level
    for lk in present_levels:
        R_mean = np.array(agg["per_level"][lk]["mean_amplification_ratio"])
        W_t = W_t_filters.get(lk)
        if W_t is not None and len(R_mean) == len(W_t):
            r, p = pearson_correlation(R_mean, W_t)
            tests[f"pearson_R_vs_Wt_{lk}"] = {
                "r": r,
                "p_value": p,
                "target": "r > 0.6",
                "passed": r > 0.6,
            }

    # H2: Post-fusion drift > Pre-fusion drift for fine-grained tasks
    if present_levels:
        amplification_by_level = []
        for lk in present_levels:
            amplification_by_level.append(
                agg["per_level"][lk]["overall_amplification"]
            )
        if len(amplification_by_level) >= 2:
            ranks = list(range(1, len(present_levels) + 1))
            rho, p = spearman_correlation(ranks, amplification_by_level)
            tests["spearman_amplification_vs_granularity"] = {
                "rho": rho,
                "p_value": p,
                "passed": rho > 0.6,
                "values": dict(zip(present_levels, amplification_by_level)),
            }

    # Summary
    r_wt_tests = [v.get("passed", False) for k, v in tests.items() if k.startswith("pearson_R_vs_Wt")]
    tests["hypothesis_supported"] = sum(r_wt_tests) > len(r_wt_tests) / 2 if r_wt_tests else False

    # --- Log ---
    logger.info("-" * 40)
    for lk in present_levels:
        stats = agg["per_level"][lk]
        logger.info(
            "  %s: pre_drift=%.4f  post_drift=%.4f  amplification=%.2f",
            lk, stats["mean_pre_drift"], stats["mean_post_drift"],
            stats["overall_amplification"],
        )
    logger.info("-" * 40)

    # --- Save ---
    save_json(agg, out_dir / "summary.json")
    save_json(tests, out_dir / "hypothesis_tests.json")
    save_json(per_sample, out_dir / "amplification.json")

    try:
        adapter.unload()
    except Exception:
        pass

    metrics = {
        "num_samples": len(per_sample),
        "total_time_s": total_time,
        **{
            f"{lk}_amplification": agg["per_level"][lk]["overall_amplification"]
            for lk in present_levels
        },
    }

    return ExperimentResult(
        experiment_id=3,
        experiment_name="exp3_fusion_drift",
        config=exp_cfg,
        metrics=metrics,
        per_sample=per_sample,
        hypothesis_tests=tests,
    )
