"""Experiment 3: Controlled post-fusion response analysis.

This experiment treats pre-fusion drift as the task-agnostic controlled input
and post-fusion drift as the task-conditioned response:

    - pre-fusion: band-wise vision drift ΔV(ω)
    - post-fusion: scalar all-token drift ΔZ_all

The main hypothesis is that, after grouping perturbations with similar
pre-fusion drift profiles, the task-conditioned post-fusion response is
predicted by the overlap between the task filter W_t and the pre-fusion drift:

    overlap(t, p) = ∫ W_t(ω) · ΔV_p(ω) dω

Depends on: Experiment 2 outputs (W_t filters).

Outputs:
    exp3/summary.json           -- aggregate controlled-response summaries
    exp3/hypothesis_tests.json  -- overlap-vs-response correlations
    exp3/amplification.json     -- per-sample controlled-response data
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..analysis.continuous import (
    attach_complexity_residual,
    summarize_by_score,
    summarize_fixed_effects_trend,
    summarize_linear_trend,
    summarize_multivariate_regression,
)
from ..analysis.drift import (
    compute_band_drift,
    compute_scalar_drift,
    compute_scalar_drift_all_tokens,
)
from ..analysis.statistics import pearson_correlation, spearman_correlation
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
from ..perturbations import build_perturbation_suite, export_perturbation_suite_images
from ..utils.device import select_device
from ..utils.exp2_filters import load_exp2_filter_bank
from ..utils.io import save_json
from ..utils.layer_groups import LAYER_GROUP_ORDER

logger = logging.getLogger(__name__)
FILTER_ANALYSIS_ORDER = ("overall",) + LAYER_GROUP_ORDER
_EPS = 1e-10


def _resolve_perturbation_subset(value: Any) -> Optional[int]:
    """Resolve the Exp 3 perturbation cap.

    ``null`` / ``None`` in YAML means "use all perturbations".
    Positive integers keep only the first N perturbations per sample.
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "all", "null", "none"}:
            return None
        value = normalized
    subset = int(value)
    if subset <= 0:
        return None
    return subset


def _build_complexity_points(per_sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for record in per_sample:
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            perturbations = level_data.get("perturbations", [])
            if not perturbations:
                continue
            pre_values = [float(item.get("pre_drift_scalar", 0.0)) for item in perturbations]
            post_all_values = [float(item.get("post_drift_scalar_all", 0.0)) for item in perturbations]
            response_values = [float(item.get("response_amplification", 0.0)) for item in perturbations]
            point = {
                "image_id": image_id,
                "level": level_key,
                "question": level_data.get("question"),
                "question_type": level_data.get("question_type"),
                "complexity_score": float(level_data.get("complexity_score", 0.0) or 0.0),
                "question_complexity_score": float(
                    level_data.get("question_complexity_score", 0.0) or 0.0
                ),
                "prompt_complexity_score": float(
                    level_data.get("prompt_complexity_score", 0.0) or 0.0
                ),
                "option_hardness_score": float(
                    level_data.get("option_hardness_score", 0.0) or 0.0
                ),
                "mean_pre_drift_scalar": float(np.mean(pre_values)) if pre_values else 0.0,
                "mean_post_drift_all": float(np.mean(post_all_values)) if post_all_values else 0.0,
                "mean_response_amplification": float(np.mean(response_values)) if response_values else 0.0,
                "num_perturbations": len(perturbations),
            }
            points.append(point)
    attach_complexity_residual(points)
    return points


def _summarize_complexity(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not points:
        return {}
    complexity_values = [
        float(point.get("complexity_score", 0.0))
        for point in points
        if point.get("complexity_score") is not None
    ]
    return {
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
        "mean_post_drift_all_by_score": summarize_by_score(
            points,
            score_key="complexity_score",
            value_key="mean_post_drift_all",
        ),
        "mean_response_amplification_by_score": summarize_by_score(
            points,
            score_key="complexity_score",
            value_key="mean_response_amplification",
        ),
        "mean_post_drift_all_by_prompt_load": summarize_by_score(
            points,
            score_key="prompt_complexity_score",
            value_key="mean_post_drift_all",
        ),
        "mean_response_amplification_by_option_hardness": summarize_by_score(
            points,
            score_key="option_hardness_score",
            value_key="mean_response_amplification",
        ),
    }


def _spectral_overlap_score(pre_drift_bands: np.ndarray, W_t: np.ndarray) -> float:
    pre = np.asarray(pre_drift_bands, dtype=np.float64)
    weights = np.asarray(W_t, dtype=np.float64)
    usable = min(len(pre), len(weights))
    if usable == 0:
        return 0.0
    return float(np.dot(pre[:usable], weights[:usable]))


def _normalize_profile(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    total = float(arr.sum())
    if total <= _EPS:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / total


def _extract_vision_tokens_and_grid(adapter, image: Image.Image) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    tokens = adapter.get_vision_tokens(image)
    if tokens is None:
        return None, None
    tokens_np = tokens.cpu().float().numpy()
    patch_grid: Optional[Tuple[int, int]] = None
    if tokens_np.ndim == 4 and tokens_np.shape[0] == 1:
        patch_grid = (int(tokens_np.shape[1]), int(tokens_np.shape[2]))
    elif tokens_np.ndim == 3:
        patch_grid = (int(tokens_np.shape[0]), int(tokens_np.shape[1]))
    else:
        patch_grid = adapter.get_patch_grid_shape(image)
    return tokens_np, patch_grid


def _kmeans_profiles(
    profiles: np.ndarray,
    num_groups: int,
    seed: int,
    max_iter: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(profiles) == 0:
        return np.empty((0,), dtype=int), np.empty((0, 0), dtype=np.float64)
    if num_groups <= 1 or len(profiles) == 1:
        return np.zeros(len(profiles), dtype=int), profiles[:1].copy()

    k = min(num_groups, len(profiles))
    rng = np.random.default_rng(seed)
    centroid_indices = rng.choice(len(profiles), size=k, replace=False)
    centroids = profiles[centroid_indices].astype(np.float64, copy=True)
    labels = np.full(len(profiles), -1, dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(profiles[:, None, :] - centroids[None, :, :], axis=-1)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            members = profiles[labels == idx]
            if len(members) == 0:
                farthest_index = int(np.argmax(np.min(distances, axis=1)))
                centroids[idx] = profiles[farthest_index]
                labels[farthest_index] = idx
            else:
                centroids[idx] = members.mean(axis=0)

    centroid_scores = []
    for idx, centroid in enumerate(centroids):
        norm_centroid = _normalize_profile(centroid)
        score = float(np.dot(np.arange(len(norm_centroid), dtype=np.float64), norm_centroid))
        centroid_scores.append((score, idx))
    ordered = [idx for _, idx in sorted(centroid_scores)]
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(ordered)}
    remapped_labels = np.asarray([remap[int(label)] for label in labels], dtype=int)
    ordered_centroids = np.asarray([centroids[idx] for idx in ordered], dtype=np.float64)
    return remapped_labels, ordered_centroids


def _build_profile_groups(
    profile_entries: List[Dict[str, Any]],
    num_groups: int,
    seed: int,
) -> Tuple[Dict[Tuple[str, str], str], Dict[str, Dict[str, Any]]]:
    if not profile_entries:
        return {}, {}

    profile_matrix = np.asarray(
        [_normalize_profile(entry["pre_drift_bands"]) for entry in profile_entries],
        dtype=np.float64,
    )
    group_count = min(max(1, num_groups), len(profile_entries))
    labels, centroids = _kmeans_profiles(profile_matrix, group_count, seed=seed)

    assignment: Dict[Tuple[str, str], str] = {}
    summary: Dict[str, Dict[str, Any]] = {}
    grouped_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry, label in zip(profile_entries, labels):
        group_name = f"profile_{int(label)}"
        key = (str(entry["image_id"]), str(entry["perturbation"]))
        assignment[key] = group_name
        grouped_entries[group_name].append(entry)

    for group_name, entries in grouped_entries.items():
        centroid_index = int(group_name.split("_")[-1])
        centroid = centroids[centroid_index]
        mean_pre = np.mean(np.stack([np.asarray(entry["pre_drift_bands"], dtype=np.float64) for entry in entries]), axis=0)
        summary[group_name] = {
            "centroid_profile": _normalize_profile(centroid).tolist(),
            "mean_pre_drift_bands": mean_pre.tolist(),
            "mean_pre_drift_scalar": float(np.mean([float(entry["pre_drift_scalar"]) for entry in entries])),
            "num_unique_pairs": len(entries),
            "example_perturbations": [
                {
                    "image_id": str(entry["image_id"]),
                    "perturbation": str(entry["perturbation"]),
                }
                for entry in entries[:5]
            ],
        }
    return assignment, summary


def _controlled_correlation_payload(
    observations: List[Dict[str, Any]],
    analysis_group: str,
    *,
    level: Optional[str] = None,
) -> Dict[str, Any]:
    raw_x: List[float] = []
    raw_y: List[float] = []
    centered_x: List[float] = []
    centered_y: List[float] = []
    groups_used = 0
    by_profile: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for obs in observations:
        if level is not None and obs["level"] != level:
            continue
        overlap = obs["overlap_scores"].get(analysis_group)
        response = obs.get("post_drift_scalar_all")
        profile_group = obs.get("profile_group")
        if overlap is None or response is None or profile_group is None:
            continue
        overlap_f = float(overlap)
        response_f = float(response)
        raw_x.append(overlap_f)
        raw_y.append(response_f)
        by_profile[str(profile_group)].append((overlap_f, response_f))

    for pairs in by_profile.values():
        if len(pairs) < 2:
            continue
        groups_used += 1
        x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        centered_x.extend((x - x.mean()).tolist())
        centered_y.extend((y - y.mean()).tolist())

    raw_r, raw_p = pearson_correlation(raw_x, raw_y)
    controlled_r, controlled_p = pearson_correlation(centered_x, centered_y)
    return {
        "raw_r": raw_r,
        "raw_p_value": raw_p,
        "raw_n": len(raw_x),
        "controlled_r": controlled_r,
        "controlled_p_value": controlled_p,
        "controlled_n": len(centered_x),
        "num_profile_groups_used": groups_used,
        "target": "controlled_r > 0.6",
        "passed": controlled_r > 0.6,
    }


def run_exp3(
    cfg: dict,
    out_dir: Path,
    results_so_far: Dict[int, ExperimentResult],
) -> ExperimentResult:
    """Run Experiment 3: controlled post-fusion response analysis."""
    logger.info("=" * 50)
    logger.info("Experiment 3: Controlled Post-Fusion Response")
    logger.info("=" * 50)

    exp_cfg = cfg.get("experiments", {}).get("exp3", {})
    max_samples = exp_cfg.get("max_samples", 200)
    pert_subset = _resolve_perturbation_subset(exp_cfg.get("perturbation_subset", 10))
    num_bands = cfg.get("analysis", {}).get("num_bands", 10)
    suppress_dc = bool(cfg.get("analysis", {}).get("suppress_dc", True))
    post_fusion_layer_index = exp_cfg.get("post_fusion_layer_index")
    post_fusion_layer_fraction = exp_cfg.get("post_fusion_layer_fraction", 0.8)
    profile_group_count = exp_cfg.get("profile_group_count", 4)
    seed = cfg.get("seed", 42)

    base_out = Path(cfg.get("out_dir", "frequency_alignment_outputs"))
    W_t_average_by_group, W_t_per_sample_by_group = load_exp2_filter_bank(base_out / "exp2")
    logger.info(
        "Loaded W_t filters: %d overall averages, %d late per-sample",
        len(W_t_average_by_group["overall"]),
        len(W_t_per_sample_by_group["late"]),
    )
    if pert_subset is None:
        logger.info("Exp 3 perturbation subset: all perturbations per sample")
    else:
        logger.info("Exp 3 perturbation subset: first %d perturbations per sample", pert_subset)

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

    samples = load_multilevel_vqa_dataset(cfg, max_samples=max_samples)
    logger.info("Loaded %d samples", len(samples))
    if not samples:
        return ExperimentResult(
            experiment_id=3,
            experiment_name="exp3_fusion_drift",
            config=exp_cfg,
            metrics={"error": "no_samples"},
        )

    pert_cfg_global = cfg.get("perturbations", {})
    severity_levels = pert_cfg_global.get("severity_levels", [1, 2, 3])
    export_num_images = max(0, int(pert_cfg_global.get("export_num_images", 1) or 0))
    exported_examples = 0

    per_sample: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    profile_entries: List[Dict[str, Any]] = []

    level_pre_band_profiles: Dict[str, List[np.ndarray]] = defaultdict(list)
    level_pre_drifts: Dict[str, List[float]] = defaultdict(list)
    level_post_all_drifts: Dict[str, List[float]] = defaultdict(list)
    level_post_vision_drifts: Dict[str, List[float]] = defaultdict(list)
    level_response_amplifications: Dict[str, List[float]] = defaultdict(list)
    level_filter_values: Dict[str, Dict[str, List[np.ndarray]]] = {
        group_name: defaultdict(list) for group_name in FILTER_ANALYSIS_ORDER
    }
    level_overlap_scores: Dict[str, Dict[str, List[float]]] = {
        group_name: defaultdict(list) for group_name in FILTER_ANALYSIS_ORDER
    }
    level_post_layers: Dict[str, List[int]] = defaultdict(list)

    t0 = time.time()
    for idx, sample in enumerate(samples):
        logger.info("Processing sample %d/%d: %s", idx + 1, len(samples), sample.image_id)
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as exc:
            logger.warning("Cannot open image %s: %s", sample.image_path, exc)
            continue
        overlay_options, overlay_base_label = sample.reference_overlay_context()

        clean_pre, clean_patch_grid = _extract_vision_tokens_and_grid(adapter, image)
        if clean_pre is None:
            logger.warning("Skipping %s: could not extract clean pre-fusion vision tokens", sample.image_id)
            continue

        perturbations = build_perturbation_suite(
            image,
            severity_levels=severity_levels,
            include_natural=pert_cfg_global.get("include_natural", True),
            include_frequency=pert_cfg_global.get("include_frequency", True),
            num_bands=num_bands,
            suppress_dc=suppress_dc,
            natural_types=pert_cfg_global.get("natural_types"),
            frequency_types=pert_cfg_global.get("frequency_types"),
            severity_params=pert_cfg_global.get("severity_params"),
            seed=seed + idx,
            overlay_mode=pert_cfg_global.get("overlay_mode", "label_free"),
            overlay_count=int(pert_cfg_global.get("overlay_count", 3)),
            overlay_options=overlay_options,
            overlay_base_label=overlay_base_label,
            overlay_seed=seed + idx,
        )
        if pert_subset and len(perturbations) > pert_subset:
            perturbations = perturbations[:pert_subset]
        if not perturbations:
            continue

        if exported_examples < export_num_images:
            try:
                export_dir = out_dir / "perturbation_examples"
                export_perturbation_suite_images(
                    image,
                    perturbations,
                    export_dir,
                    sample.image_id,
                )
                exported_examples += 1
            except Exception as exc:
                logger.warning(
                    "Could not export perturbation images for %s: %s",
                    sample.image_id,
                    exc,
                )

        sample_record: Dict[str, Any] = {
            "image_id": sample.image_id,
            "pre_perturbations": {},
            "levels": {},
        }

        pre_stats_by_perturbation: Dict[str, Dict[str, Any]] = {}
        for pr in perturbations:
            pert_pre, pert_patch_grid = _extract_vision_tokens_and_grid(adapter, pr.perturbed_image)
            if pert_pre is None:
                continue
            pre_drift_bands = compute_band_drift(
                clean_pre,
                pert_pre,
                num_bands,
                clean_patch_grid,
                pert_patch_grid,
                suppress_dc=suppress_dc,
            )
            pre_drift_scalar = compute_scalar_drift(
                clean_pre,
                pert_pre,
                clean_patch_grid,
                pert_patch_grid,
            )
            pre_profile = _normalize_profile(pre_drift_bands)
            pre_stats = {
                "pre_drift_bands": pre_drift_bands,
                "pre_drift_scalar": pre_drift_scalar,
                "pre_drift_profile": pre_profile,
                "perturbed_patch_grid": pert_patch_grid,
            }
            pre_stats_by_perturbation[pr.name] = pre_stats
            sample_record["pre_perturbations"][pr.name] = {
                "pre_drift_scalar": pre_drift_scalar,
                "pre_drift_bands": pre_drift_bands.tolist(),
                "pre_drift_profile": pre_profile.tolist(),
            }
            profile_entries.append(
                {
                    "image_id": sample.image_id,
                    "perturbation": pr.name,
                    "pre_drift_bands": pre_drift_bands,
                    "pre_drift_scalar": pre_drift_scalar,
                }
            )

        if not pre_stats_by_perturbation:
            continue

        for level in GranularityLevel:
            level_data = sample.levels.get(level)
            if level_data is None:
                continue
            level_key = level.name
            complexity = ensure_level_complexity(level_data)

            group_filters: Dict[str, Optional[np.ndarray]] = {
                "overall": W_t_per_sample_by_group["overall"].get(
                    (sample.image_id, level_key),
                    W_t_average_by_group["overall"].get(level_key),
                )
            }
            for group_name in LAYER_GROUP_ORDER:
                group_filters[group_name] = W_t_per_sample_by_group[group_name].get(
                    (sample.image_id, level_key),
                    W_t_average_by_group[group_name].get(level_key),
                )
            if not any(filter_values is not None for filter_values in group_filters.values()):
                continue

            prompt = level_data.question
            if level_data.options:
                opts_str = " ".join(
                    f"({label}) {text}" for label, text in sorted(level_data.options.items())
                )
                prompt = f"{prompt} Options: {opts_str}"

            try:
                clean_internals = adapter.extract_internals(
                    image,
                    prompt,
                    extract_attention=False,
                    extract_pre_fusion=False,
                    extract_post_fusion=True,
                    post_fusion_layer_index=post_fusion_layer_index,
                    post_fusion_layer_fraction=post_fusion_layer_fraction,
                )
            except Exception as exc:
                logger.warning("Clean post-fusion extraction failed for %s/%s: %s", sample.image_id, level_key, exc)
                continue

            if clean_internals.post_fusion_all_features is None:
                continue

            clean_post_all = clean_internals.post_fusion_all_features.cpu().float().numpy()
            clean_post_vision = None
            if clean_internals.post_fusion_features is not None:
                clean_post_vision = clean_internals.post_fusion_features.cpu().float().numpy()
            if clean_internals.post_fusion_layer_index is not None:
                level_post_layers[level_key].append(clean_internals.post_fusion_layer_index)

            for group_name, filter_values in group_filters.items():
                if filter_values is not None:
                    level_filter_values[group_name][level_key].append(np.asarray(filter_values, dtype=np.float64))

            level_pert_records: List[Dict[str, Any]] = []
            for pr in perturbations:
                pre_stats = pre_stats_by_perturbation.get(pr.name)
                if pre_stats is None:
                    continue
                try:
                    pert_internals = adapter.extract_internals(
                        pr.perturbed_image,
                        prompt,
                        extract_attention=False,
                        extract_pre_fusion=False,
                        extract_post_fusion=True,
                        post_fusion_layer_index=post_fusion_layer_index,
                        post_fusion_layer_fraction=post_fusion_layer_fraction,
                    )
                except Exception:
                    continue

                if pert_internals.post_fusion_all_features is None:
                    continue

                pert_post_all = pert_internals.post_fusion_all_features.cpu().float().numpy()
                post_drift_scalar_all = compute_scalar_drift_all_tokens(
                    clean_post_all,
                    pert_post_all,
                    clean_vision_token_range=clean_internals.vision_token_range,
                    perturbed_vision_token_range=pert_internals.vision_token_range,
                    clean_patch_grid=clean_patch_grid,
                    perturbed_patch_grid=pre_stats.get("perturbed_patch_grid"),
                )

                post_drift_scalar_vision = None
                if clean_post_vision is not None and pert_internals.post_fusion_features is not None:
                    pert_post_vision = pert_internals.post_fusion_features.cpu().float().numpy()
                    post_drift_scalar_vision = compute_scalar_drift(
                        clean_post_vision,
                        pert_post_vision,
                        clean_patch_grid,
                        pre_stats.get("perturbed_patch_grid"),
                    )

                response_amplification = post_drift_scalar_all / (float(pre_stats["pre_drift_scalar"]) + _EPS)
                overlap_scores: Dict[str, float] = {}
                analysis_group_record: Dict[str, Any] = {}
                for group_name in FILTER_ANALYSIS_ORDER:
                    W_t = group_filters.get(group_name)
                    if W_t is None:
                        continue
                    overlap_score = _spectral_overlap_score(pre_stats["pre_drift_bands"], W_t)
                    overlap_scores[group_name] = overlap_score
                    level_overlap_scores[group_name][level_key].append(overlap_score)
                    analysis_group_record[group_name] = {
                        "spectral_overlap": overlap_score,
                        "W_t": np.asarray(W_t, dtype=np.float64).tolist(),
                    }

                level_pre_band_profiles[level_key].append(np.asarray(pre_stats["pre_drift_bands"], dtype=np.float64))
                level_pre_drifts[level_key].append(float(pre_stats["pre_drift_scalar"]))
                level_post_all_drifts[level_key].append(post_drift_scalar_all)
                if post_drift_scalar_vision is not None:
                    level_post_vision_drifts[level_key].append(post_drift_scalar_vision)
                level_response_amplifications[level_key].append(response_amplification)

                obs_record = {
                    "image_id": str(sample.image_id),
                    "level": level_key,
                    "perturbation": pr.name,
                    "pre_drift_scalar": float(pre_stats["pre_drift_scalar"]),
                    "pre_drift_bands": np.asarray(pre_stats["pre_drift_bands"], dtype=np.float64),
                    "pre_drift_profile": np.asarray(pre_stats["pre_drift_profile"], dtype=np.float64),
                    "post_drift_scalar_all": float(post_drift_scalar_all),
                    "post_drift_scalar_vision": (
                        float(post_drift_scalar_vision) if post_drift_scalar_vision is not None else None
                    ),
                    "response_amplification": float(response_amplification),
                    "overlap_scores": overlap_scores,
                }
                observations.append(obs_record)

                level_pert_records.append(
                    {
                        "perturbation": pr.name,
                        "pre_drift_scalar": float(pre_stats["pre_drift_scalar"]),
                        "pre_drift_bands": np.asarray(pre_stats["pre_drift_bands"], dtype=np.float64).tolist(),
                        "pre_drift_profile": np.asarray(pre_stats["pre_drift_profile"], dtype=np.float64).tolist(),
                        "post_drift_scalar_all": float(post_drift_scalar_all),
                        "post_drift_scalar_vision": (
                            float(post_drift_scalar_vision) if post_drift_scalar_vision is not None else None
                        ),
                        "response_amplification": float(response_amplification),
                        "analysis_groups": analysis_group_record,
                    }
                )

            if level_pert_records:
                sample_record["levels"][level_key] = {
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
                    "num_perturbations": len(level_pert_records),
                    "post_fusion_layer_index": clean_internals.post_fusion_layer_index,
                    "analysis_groups": {
                        group_name: {"W_t": np.asarray(filter_values, dtype=np.float64).tolist()}
                        for group_name, filter_values in group_filters.items()
                        if filter_values is not None
                    },
                    "perturbations": level_pert_records,
                }

        if sample_record["levels"]:
            per_sample.append(sample_record)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info(
                "Progress: %d/%d (%.2f samples/s)",
                idx + 1,
                len(samples),
                (idx + 1) / max(elapsed, 1e-6),
            )

    total_time = time.time() - t0
    logger.info("Controlled response analysis complete: %d samples in %.1fs", len(per_sample), total_time)

    if not observations:
        try:
            adapter.unload()
        except Exception:
            pass
        return ExperimentResult(
            experiment_id=3,
            experiment_name="exp3_fusion_drift",
            config=exp_cfg,
            metrics={"error": "no_observations"},
        )

    profile_assignment, profile_group_summary = _build_profile_groups(
        profile_entries,
        num_groups=profile_group_count,
        seed=seed,
    )

    for obs in observations:
        obs["profile_group"] = profile_assignment.get((obs["image_id"], obs["perturbation"]))

    for sample_record in per_sample:
        image_id = str(sample_record["image_id"])
        for perturbation, pre_record in sample_record.get("pre_perturbations", {}).items():
            pre_record["profile_group"] = profile_assignment.get((image_id, perturbation))
        for level_data in sample_record.get("levels", {}).values():
            for item in level_data.get("perturbations", []):
                item["profile_group"] = profile_assignment.get((image_id, str(item["perturbation"])))

    for profile_group, summary in profile_group_summary.items():
        group_obs = [obs for obs in observations if obs.get("profile_group") == profile_group]
        per_level_summary: Dict[str, Any] = {}
        for level_key in ALL_VQA_LEVEL_NAMES:
            level_obs = [obs for obs in group_obs if obs["level"] == level_key]
            if not level_obs:
                continue
            per_level_summary[level_key] = {
                "mean_post_drift_scalar_all": float(
                    np.mean([float(obs["post_drift_scalar_all"]) for obs in level_obs])
                ),
                "mean_response_amplification": float(
                    np.mean([float(obs["response_amplification"]) for obs in level_obs])
                ),
                "n": len(level_obs),
            }
        summary["per_level"] = per_level_summary

    agg: Dict[str, Any] = {
        "per_level": {},
        "analysis_groups": {
            "order": list(FILTER_ANALYSIS_ORDER),
            "primary_group": "late",
            "control_groups": [group_name for group_name in FILTER_ANALYSIS_ORDER if group_name != "late"],
        },
        "response_metric": {
            "name": "post_drift_scalar_all",
            "token_scope": "all_tokens",
            "diagnostic_token_scope": "vision_tokens",
        },
        "profile_grouping": {
            "method": "kmeans",
            "feature": "normalized_pre_drift_profile",
            "num_groups": len(profile_group_summary),
            "groups": profile_group_summary,
        },
        "post_fusion_layer_fraction": post_fusion_layer_fraction,
        "post_fusion_layer_index": post_fusion_layer_index,
    }

    complexity_points = _build_complexity_points(per_sample)
    agg["complexity_analysis"] = _summarize_complexity(complexity_points)

    level_order = list(ALL_VQA_LEVEL_NAMES)
    present_levels = [level_key for level_key in level_order if level_key in level_post_all_drifts]
    primary_present_levels = [level_key for level_key in PRIMARY_VQA_LEVEL_NAMES if level_key in level_post_all_drifts]

    for level_key in present_levels:
        mean_pre_bands = np.mean(np.stack(level_pre_band_profiles[level_key]), axis=0)
        agg["per_level"][level_key] = {
            "mean_pre_drift_bands": mean_pre_bands.tolist(),
            "mean_pre_drift": float(np.mean(level_pre_drifts[level_key])) if level_pre_drifts[level_key] else 0.0,
            "mean_post_drift_all": float(np.mean(level_post_all_drifts[level_key])) if level_post_all_drifts[level_key] else 0.0,
            "mean_post_drift_vision": (
                float(np.mean(level_post_vision_drifts[level_key])) if level_post_vision_drifts[level_key] else None
            ),
            "mean_response_amplification": (
                float(np.mean(level_response_amplifications[level_key]))
                if level_response_amplifications[level_key] else 0.0
            ),
            "num_observations": len(level_post_all_drifts[level_key]),
            "post_fusion_layer_index_mean": (
                float(np.mean(level_post_layers[level_key])) if level_post_layers[level_key] else None
            ),
            "analysis_groups": {},
        }
        level_obs = [obs for obs in observations if obs["level"] == level_key]
        for group_name in FILTER_ANALYSIS_ORDER:
            filters = level_filter_values[group_name].get(level_key, [])
            overlaps = level_overlap_scores[group_name].get(level_key, [])
            if not filters or not overlaps:
                continue
            filter_mean = np.mean(np.stack(filters), axis=0)
            corr_payload = _controlled_correlation_payload(
                observations,
                group_name,
                level=level_key,
            )
            agg["per_level"][level_key]["analysis_groups"][group_name] = {
                "mean_overlap_score": float(np.mean(overlaps)),
                "std_overlap_score": float(np.std(overlaps)),
                "W_t_average": filter_mean.tolist(),
                "num_observations": len(overlaps),
                "raw_overlap_to_post_drift_r": corr_payload["raw_r"],
                "controlled_overlap_to_post_drift_r": corr_payload["controlled_r"],
                "controlled_num_profile_groups_used": corr_payload["num_profile_groups_used"],
            }

    tests: Dict[str, Any] = {
        "analysis_groups": {
            "order": list(FILTER_ANALYSIS_ORDER),
            "primary_group": "late",
            "control_groups": [group_name for group_name in FILTER_ANALYSIS_ORDER if group_name != "late"],
        },
        "response_metric": "post_drift_scalar_all",
        "profile_grouping": {
            "method": "kmeans",
            "feature": "normalized_pre_drift_profile",
            "num_groups": len(profile_group_summary),
        },
        "pearson_post_response_vs_overlap_by_group": {},
        "pearson_post_response_vs_overlap_by_level": {},
    }

    for group_name in FILTER_ANALYSIS_ORDER:
        tests["pearson_post_response_vs_overlap_by_group"][group_name] = _controlled_correlation_payload(
            observations,
            group_name,
        )

    for level_key in present_levels:
        tests["pearson_post_response_vs_overlap_by_level"][level_key] = {}
        for group_name in FILTER_ANALYSIS_ORDER:
            tests["pearson_post_response_vs_overlap_by_level"][level_key][group_name] = (
                _controlled_correlation_payload(observations, group_name, level=level_key)
            )

    if primary_present_levels:
        post_values = [agg["per_level"][level_key]["mean_post_drift_all"] for level_key in primary_present_levels]
        rho, p = spearman_correlation(range(1, len(post_values) + 1), post_values)
        tests["spearman_post_drift_all_vs_granularity"] = {
            "rho": rho,
            "p_value": p,
            "passed": rho > 0.6,
            "values": dict(zip(primary_present_levels, post_values)),
        }

    for group_name in FILTER_ANALYSIS_ORDER:
        group_values = []
        group_levels = []
        for level_key in primary_present_levels:
            group_stats = agg["per_level"][level_key]["analysis_groups"].get(group_name)
            if group_stats is None:
                continue
            group_levels.append(level_key)
            group_values.append(group_stats["mean_overlap_score"])
        if len(group_values) >= 2:
            rho, p = spearman_correlation(range(1, len(group_values) + 1), group_values)
            tests[f"spearman_overlap_vs_granularity_{group_name}"] = {
                "rho": rho,
                "p_value": p,
                "passed": rho > 0.6,
                "values": dict(zip(group_levels, group_values)),
            }

    late_primary = tests["pearson_post_response_vs_overlap_by_group"].get("late", {})
    tests["primary_group"] = "late"
    tests["control_groups"] = [group_name for group_name in FILTER_ANALYSIS_ORDER if group_name != "late"]
    tests["hypothesis_supported"] = bool(late_primary.get("passed", False))
    if complexity_points:
        tests["continuous_complexity"] = {
            "mean_post_drift_all_vs_complexity": summarize_linear_trend(
                complexity_points,
                x_key="complexity_score",
                y_key="mean_post_drift_all",
            ),
            "mean_response_amplification_vs_complexity": summarize_linear_trend(
                complexity_points,
                x_key="complexity_score",
                y_key="mean_response_amplification",
            ),
            "mean_post_drift_all_vs_prompt_load": summarize_linear_trend(
                complexity_points,
                x_key="prompt_complexity_score",
                y_key="mean_post_drift_all",
            ),
            "mean_response_amplification_vs_option_hardness": summarize_linear_trend(
                complexity_points,
                x_key="option_hardness_score",
                y_key="mean_response_amplification",
            ),
            "mean_post_drift_all_vs_complexity_fixed_effects": summarize_fixed_effects_trend(
                complexity_points,
                group_key="image_id",
                x_key="complexity_score",
                y_key="mean_post_drift_all",
            ),
            "mean_response_amplification_vs_complexity_fixed_effects": summarize_fixed_effects_trend(
                complexity_points,
                group_key="image_id",
                x_key="complexity_score",
                y_key="mean_response_amplification",
            ),
            "horse_race_mean_post_drift_all": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_post_drift_all",
                x_keys=[
                    "complexity_score_residual",
                    "prompt_complexity_score",
                    "option_hardness_score",
                ],
            ),
            "horse_race_mean_post_drift_all_within_image": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_post_drift_all",
                x_keys=[
                    "complexity_score_residual",
                    "prompt_complexity_score",
                    "option_hardness_score",
                ],
                group_key="image_id",
                demean_by_group=True,
            ),
            "horse_race_mean_response_amplification": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_response_amplification",
                x_keys=[
                    "complexity_score_residual",
                    "prompt_complexity_score",
                    "option_hardness_score",
                ],
            ),
            "horse_race_mean_response_amplification_within_image": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_response_amplification",
                x_keys=[
                    "complexity_score_residual",
                    "prompt_complexity_score",
                    "option_hardness_score",
                ],
                group_key="image_id",
                demean_by_group=True,
            ),
        }

    logger.info("-" * 40)
    for level_key in present_levels:
        stats = agg["per_level"][level_key]
        logger.info(
            "  %s: pre_drift=%.4f  post_all=%.4f  response_amp=%.2f",
            level_key,
            stats["mean_pre_drift"],
            stats["mean_post_drift_all"],
            stats["mean_response_amplification"],
        )
        for group_name in FILTER_ANALYSIS_ORDER:
            group_stats = stats["analysis_groups"].get(group_name)
            if group_stats is None:
                continue
            logger.info(
                "    %s[%s]: overlap=%.4f  controlled_r=%.3f",
                level_key,
                group_name,
                group_stats["mean_overlap_score"],
                group_stats["controlled_overlap_to_post_drift_r"],
            )
    logger.info("-" * 40)

    save_json(agg, out_dir / "summary.json")
    save_json(tests, out_dir / "hypothesis_tests.json")
    save_json(per_sample, out_dir / "amplification.json")
    save_json(complexity_points, out_dir / "complexity_points.json")

    try:
        adapter.unload()
    except Exception:
        pass

    metrics: Dict[str, Any] = {
        "num_samples": len(per_sample),
        "num_observations": len(observations),
        "num_profile_groups": len(profile_group_summary),
        "total_time_s": total_time,
    }
    for level_key in present_levels:
        metrics[f"{level_key}_post_drift_all"] = agg["per_level"][level_key]["mean_post_drift_all"]
        metrics[f"{level_key}_response_amplification"] = agg["per_level"][level_key]["mean_response_amplification"]
    for group_name in FILTER_ANALYSIS_ORDER:
        group_test = tests["pearson_post_response_vs_overlap_by_group"].get(group_name)
        if group_test is not None:
            metrics[f"controlled_r_{group_name}"] = group_test["controlled_r"]

    return ExperimentResult(
        experiment_id=3,
        experiment_name="exp3_fusion_drift",
        config=exp_cfg,
        metrics=metrics,
        per_sample=per_sample,
        hypothesis_tests=tests,
    )
