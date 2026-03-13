"""Experiment 2: Cross-Attention Frequency Analysis.

Extract cross-attention maps from the VLM for each granularity level and
decompose them into frequency bands via 2D FFT.  Measure how the effective
bandwidth G(t) of the attention filter varies with task granularity.

The hypothesis is that coarse tasks produce narrow (low-frequency) attention
filters, while fine-grained tasks produce broad (high-frequency) filters.
This is quantified by the effective bandwidth G(t) via inverse participation
ratio.

Outputs:
    exp2/summary.json              -- aggregate metrics and hypothesis tests
    exp2/filters/{id}_{level}.npy  -- W_t filters per sample per level
    exp2/power_spectra.json        -- per-sample power spectra
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..analysis.spectral import (
    compute_attention_power_spectrum_multi,
    compute_effective_bandwidth,
    compute_filter_W_t,
)
from ..analysis.statistics import (
    one_way_anova,
    spearman_correlation,
    bootstrap_ci,
)
from ..data.base import ExperimentResult, GranularityLevel
from ..data.loaders import load_multilevel_vqa_dataset
from ..models import get_adapter
from ..utils.device import select_device
from ..utils.io import save_json

logger = logging.getLogger(__name__)


def _extract_attention_for_prompt(
    adapter,
    image: Image.Image,
    prompt: str,
    layer_stride: int = 4,
    num_bands: int = 10,
) -> Optional[Dict[str, Any]]:
    """Extract attention maps and compute spectral analysis for one prompt.

    Returns dict with power spectrum, bandwidth, and W_t filter, or None
    on failure.
    """
    try:
        internals = adapter.extract_internals(
            image, prompt,
            extract_attention=True,
            extract_pre_fusion=False,
            extract_post_fusion=False,
            layer_stride=layer_stride,
        )
    except Exception as e:
        logger.warning("extract_internals failed: %s", e)
        return None

    if internals.cross_attention_weights is None or not internals.cross_attention_weights:
        logger.warning("No cross-attention weights returned")
        return None

    if internals.patch_grid is None:
        logger.warning("No patch grid shape returned")
        return None

    # Convert attention tensors to numpy
    attn_list = []
    for attn_tensor in internals.cross_attention_weights:
        if attn_tensor is not None:
            attn_list.append(attn_tensor.cpu().float().numpy())

    if not attn_list:
        return None

    # Compute power spectrum across all heads/layers
    mean_radial, per_head_radials = compute_attention_power_spectrum_multi(
        attn_list, internals.patch_grid, num_bands,
    )

    # Effective bandwidth
    bandwidth = compute_effective_bandwidth(mean_radial)

    # Normalized filter
    W_t = compute_filter_W_t(mean_radial)

    return {
        "radial_power": mean_radial,
        "bandwidth": bandwidth,
        "W_t": W_t,
        "patch_grid": internals.patch_grid,
        "num_layers_extracted": len(attn_list),
        "num_heads_total": sum(
            a.shape[0] if a.ndim == 3 else 1 for a in attn_list
        ),
    }


def run_exp2(
    cfg: dict,
    out_dir: Path,
    results_so_far: Dict[int, ExperimentResult],
) -> ExperimentResult:
    """Run Experiment 2: Cross-Attention Frequency Analysis.

    For each sample × granularity level, extract the cross-attention map,
    compute its 2D FFT power spectrum, and measure the effective bandwidth.
    Test whether bandwidth scales with task granularity.
    """
    logger.info("=" * 50)
    logger.info("Experiment 2: Cross-Attention Frequency Analysis")
    logger.info("=" * 50)

    exp_cfg = cfg.get("experiments", {}).get("exp2", {})
    max_samples = exp_cfg.get("max_samples", 200)
    layer_stride = exp_cfg.get("layer_stride", 4)
    num_bands = cfg.get("analysis", {}).get("num_bands", 10)
    seed = cfg.get("seed", 42)

    # --- Load model ---
    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("primary", "Qwen/Qwen3-VL-8B-Instruct")
    device = select_device(cfg.get("device", "auto"))
    quantization = model_cfg.get("quantization")

    logger.info("Loading model: %s (device=%s, quant=%s)", model_id, device, quantization)
    adapter = get_adapter(model_id)
    adapter.load(
        model_id=model_id,
        device=device,
        cache_dir=cfg.get("cache_dir"),
        quantization=quantization,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        device_map=model_cfg.get("device_map"),
        local_files_only=cfg.get("offline", False),
    )
    logger.info("Model loaded successfully")

    # --- Load dataset ---
    samples = load_multilevel_vqa_dataset(cfg, max_samples=max_samples)
    logger.info("Loaded %d samples for attention analysis", len(samples))

    if not samples:
        logger.error("No samples loaded.")
        return ExperimentResult(
            experiment_id=2,
            experiment_name="exp2_attention_frequency",
            config=exp_cfg,
            metrics={"error": "no_samples"},
        )

    # --- Create output dirs ---
    filters_dir = out_dir / "filters"
    filters_dir.mkdir(parents=True, exist_ok=True)

    # --- Extraction loop ---
    per_sample: List[Dict[str, Any]] = []
    # Collect bandwidths grouped by level
    level_bandwidths: Dict[str, List[float]] = defaultdict(list)
    level_radials: Dict[str, List[np.ndarray]] = defaultdict(list)
    t0 = time.time()

    for idx, sample in enumerate(samples):
        logger.info(
            "Processing sample %d/%d: %s", idx + 1, len(samples), sample.image_id,
        )

        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as e:
            logger.warning("Cannot open image %s: %s", sample.image_path, e)
            continue

        sample_record: Dict[str, Any] = {
            "image_id": sample.image_id,
            "levels": {},
        }

        for level in GranularityLevel:
            level_data = sample.levels.get(level)
            if level_data is None:
                continue

            level_key = level.name
            # Build prompt from question + options
            prompt = level_data.question
            if level_data.options:
                opts_str = " ".join(
                    f"({k}) {v}" for k, v in sorted(level_data.options.items())
                )
                prompt = f"{prompt} Options: {opts_str}"

            result = _extract_attention_for_prompt(
                adapter, image, prompt, layer_stride, num_bands,
            )

            if result is None:
                continue

            # Store per-sample result
            sample_record["levels"][level_key] = {
                "bandwidth": result["bandwidth"],
                "radial_power": result["radial_power"].tolist(),
                "W_t": result["W_t"].tolist(),
                "num_layers": result["num_layers_extracted"],
                "num_heads": result["num_heads_total"],
            }

            # Accumulate for hypothesis testing
            level_bandwidths[level_key].append(result["bandwidth"])
            level_radials[level_key].append(result["radial_power"])

            # Save W_t filter
            np.save(
                filters_dir / f"{sample.image_id}_{level_key}.npy",
                result["W_t"],
            )

        per_sample.append(sample_record)

        # Progress
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            logger.info(
                "Progress: %d/%d (%.2f samples/s)",
                idx + 1, len(samples), rate,
            )

    total_time = time.time() - t0
    logger.info(
        "Extraction complete: %d samples in %.1fs",
        len(per_sample), total_time,
    )

    # --- Aggregate ---
    agg: Dict[str, Any] = {"per_level": {}}
    level_order = ["L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE"]
    present_levels = [lk for lk in level_order if lk in level_bandwidths]

    for lk in present_levels:
        bws = level_bandwidths[lk]
        radials = level_radials[lk]
        mean_radial = np.mean(np.stack(radials), axis=0) if radials else np.zeros(num_bands)
        W_t_avg = compute_filter_W_t(mean_radial)

        agg["per_level"][lk] = {
            "mean_bandwidth": float(np.mean(bws)),
            "std_bandwidth": float(np.std(bws)),
            "median_bandwidth": float(np.median(bws)),
            "mean_radial_power": mean_radial.tolist(),
            "W_t_average": W_t_avg.tolist(),
            "num_samples": len(bws),
        }

    # Save average W_t per level (critical for Exp 3 and Exp 5)
    for lk in present_levels:
        W_t_avg = np.array(agg["per_level"][lk]["W_t_average"])
        np.save(filters_dir / f"average_{lk}.npy", W_t_avg)

    # --- Hypothesis tests ---
    tests: Dict[str, Any] = {}

    # H1: Bandwidth increases with granularity
    if len(present_levels) >= 2:
        ranks = list(range(1, len(present_levels) + 1))
        mean_bws = [np.mean(level_bandwidths[lk]) for lk in present_levels]

        rho, p = spearman_correlation(ranks, mean_bws)
        tests["spearman_bandwidth_vs_granularity"] = {
            "rho": rho,
            "p_value": p,
            "target": "rho > 0.8 (bandwidth increases with granularity)",
            "passed": rho > 0.8,
            "values": dict(zip(present_levels, mean_bws)),
        }

    # H2: ANOVA on bandwidths across levels
    groups = [level_bandwidths[lk] for lk in present_levels if level_bandwidths[lk]]
    if len(groups) >= 2:
        f_stat, p_anova = one_way_anova(*groups)
        tests["anova_bandwidth_across_levels"] = {
            "F_statistic": f_stat,
            "p_value": p_anova,
            "target": "F > 10, p < 0.001",
            "passed": f_stat > 10 and p_anova < 0.001,
        }

    # H3: Bootstrap CI on bandwidth per level
    tests["bootstrap_ci_bandwidth"] = {}
    for lk in present_levels:
        bws = level_bandwidths[lk]
        if len(bws) >= 5:
            point, lo, hi = bootstrap_ci(bws, n_boot=1000)
            tests["bootstrap_ci_bandwidth"][lk] = {
                "mean": point,
                "ci_95_lower": lo,
                "ci_95_upper": hi,
            }

    # Summary
    core_passed = tests.get("spearman_bandwidth_vs_granularity", {}).get("passed", False)
    tests["hypothesis_supported"] = core_passed

    # --- Log results ---
    logger.info("-" * 40)
    logger.info("Key results:")
    for lk in present_levels:
        stats = agg["per_level"][lk]
        logger.info(
            "  %s: G(t)=%.2f +/- %.2f  (n=%d)",
            lk, stats["mean_bandwidth"], stats["std_bandwidth"], stats["num_samples"],
        )
    sp = tests.get("spearman_bandwidth_vs_granularity", {})
    logger.info(
        "  Spearman(granularity, bandwidth): rho=%.3f, p=%.4f [%s]",
        sp.get("rho", 0), sp.get("p_value", 1),
        "PASS" if sp.get("passed") else "FAIL",
    )
    logger.info("-" * 40)

    # --- Save outputs ---
    save_json(agg, out_dir / "summary.json")
    save_json(tests, out_dir / "hypothesis_tests.json")
    save_json(
        [r for r in per_sample if r.get("levels")],
        out_dir / "power_spectra.json",
    )

    # --- Cleanup ---
    try:
        adapter.unload()
    except Exception:
        pass

    metrics = {
        "num_samples": len(per_sample),
        "total_time_s": total_time,
        **{
            f"{lk}_mean_bandwidth": agg["per_level"][lk]["mean_bandwidth"]
            for lk in present_levels
        },
        "spearman_rho": sp.get("rho", 0),
        "spearman_p": sp.get("p_value", 1),
    }

    return ExperimentResult(
        experiment_id=2,
        experiment_name="exp2_attention_frequency",
        config=exp_cfg,
        metrics=metrics,
        per_sample=per_sample,
        hypothesis_tests=tests,
    )
