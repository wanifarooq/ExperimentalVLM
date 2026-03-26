"""Experiment 6: Segmentation Granularity."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..analysis.statistics import monotonicity_test, spearman_correlation
from ..data.base import ExperimentResult, GranularityLevel, LevelData
from ..data.loaders import load_segmentation_dataset
from ..data.partimagenet import mask_to_box, segmentation_to_mask
from ..models.sam_adapters import SegmentationAdapter, mask_iou
from ..perturbations import build_perturbation_suite
from ..utils.io import save_json

logger = logging.getLogger(__name__)


def _level_gt_mask(level_data: LevelData, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    return segmentation_to_mask(level_data.segmentation, image_size)


def _transform_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int],
    transform_fn,
) -> np.ndarray:
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    transformed = transform_fn(pil_mask) if transform_fn is not None else pil_mask
    if transformed.size != target_size:
        transformed = transformed.resize(target_size, resample=Image.NEAREST)
    return np.array(transformed, dtype=np.uint8) > 127


def _best_mask_vs_gt(
    predictions: List[Tuple[np.ndarray, float]],
    gt_mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], float, float]:
    best_mask = None
    best_iou = 0.0
    best_score = 0.0
    for mask, score in predictions:
        iou = mask_iou(mask, gt_mask)
        if best_mask is None or iou > best_iou:
            best_mask = mask
            best_iou = iou
            best_score = float(score)
    return best_mask, best_iou, best_score


def _predict_level(
    model_name: str,
    adapter: SegmentationAdapter,
    image: Image.Image,
    level_data: LevelData,
    box_prompt: Optional[List[float]],
) -> List[Tuple[np.ndarray, float]]:
    if model_name == "sam3":
        return adapter.predict_with_text(image, level_data.text_prompt or level_data.question)
    if box_prompt is None:
        return []
    return adapter.predict_with_box(image, [tuple(box_prompt)])


def run_exp6(
    cfg: dict,
    out_dir: Path,
    results_so_far: Dict[int, ExperimentResult],
) -> ExperimentResult:
    """Run Experiment 6: Segmentation Granularity."""
    logger.info("=" * 50)
    logger.info("Experiment 6: Segmentation Granularity")
    logger.info("=" * 50)

    exp_cfg = cfg.get("experiments", {}).get("exp6", {})
    max_samples = exp_cfg.get("max_samples", 300)
    seed = cfg.get("seed", 42)
    num_bands = cfg.get("analysis", {}).get("num_bands", 10)
    suppress_dc = bool(cfg.get("analysis", {}).get("suppress_dc", True))

    samples = load_segmentation_dataset(
        cfg,
        max_samples=max_samples,
        dataset_override=exp_cfg.get("dataset"),
    )
    if not samples:
        return ExperimentResult(
            experiment_id=6,
            experiment_name="exp6_segmentation_granularity",
            config=exp_cfg,
            metrics={"error": "no_partimagenet_data"},
        )

    device = cfg.get("device", "auto")
    if device == "auto":
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    models_to_eval: Dict[str, SegmentationAdapter] = {}
    try:
        sam3 = SegmentationAdapter("sam3")
        sam3.load(model_type="sam3", device=device, **exp_cfg.get("sam3", {}))
        models_to_eval["sam3"] = sam3
    except Exception as exc:
        logger.warning("Could not load SAM3: %s", exc)

    try:
        sam2 = SegmentationAdapter("sam2")
        sam2.load(model_type="sam2", device=device, **exp_cfg.get("sam2", {}))
        models_to_eval["sam2"] = sam2
    except Exception as exc:
        logger.warning("Could not load SAM2: %s", exc)

    if not models_to_eval:
        return ExperimentResult(
            experiment_id=6,
            experiment_name="exp6_segmentation_granularity",
            config=exp_cfg,
            metrics={"error": "no_segmentation_models"},
        )

    pert_cfg = cfg.get("perturbations", {})
    severity_levels = pert_cfg.get("severity_levels", [1, 2, 3])

    per_sample: List[Dict[str, Any]] = []
    model_level_drops: Dict[str, Dict[str, List[float]]] = {
        model_name: defaultdict(list) for model_name in models_to_eval
    }
    model_level_clean: Dict[str, Dict[str, List[float]]] = {
        model_name: defaultdict(list) for model_name in models_to_eval
    }
    t0 = time.time()

    for idx, sample in enumerate(samples):
        logger.info("Processing sample %d/%d: %s", idx + 1, len(samples), sample.image_id)
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as exc:
            logger.warning("Cannot open image %s: %s", sample.image_path, exc)
            continue
        overlay_options, overlay_base_label = sample.reference_overlay_context()

        perturbations = build_perturbation_suite(
            image,
            severity_levels=severity_levels,
            include_natural=pert_cfg.get("include_natural", True),
            include_frequency=pert_cfg.get("include_frequency", True),
            num_bands=num_bands,
            suppress_dc=suppress_dc,
            natural_types=pert_cfg.get("natural_types"),
            frequency_types=pert_cfg.get("frequency_types"),
            severity_params=pert_cfg.get("severity_params"),
            seed=seed + idx,
            overlay_options=overlay_options,
            overlay_base_label=overlay_base_label,
            overlay_seed=seed + idx,
        )

        sample_record: Dict[str, Any] = {"image_id": sample.image_id, "models": {}}
        for model_name, adapter in models_to_eval.items():
            model_record: Dict[str, Any] = {"levels": {}}
            for level in (
                GranularityLevel.L1_COARSE,
                GranularityLevel.L2_MEDIUM,
                GranularityLevel.L3_FINE,
            ):
                level_data = sample.levels.get(level)
                if level_data is None:
                    continue
                level_key = level.name
                gt_mask = _level_gt_mask(level_data, image.size)
                if gt_mask is None:
                    continue
                gt_box = level_data.bbox or mask_to_box(gt_mask)
                if gt_box is None:
                    continue

                try:
                    clean_preds = _predict_level(model_name, adapter, image, level_data, gt_box)
                except Exception as exc:
                    logger.warning("%s clean prediction failed: %s", model_name, exc)
                    continue

                best_mask, clean_iou, best_score = _best_mask_vs_gt(clean_preds, gt_mask)
                if best_mask is None:
                    continue

                model_level_clean[model_name][level_key].append(clean_iou)
                pert_records = []
                for perturbation in perturbations[:10]:
                    pert_gt_mask = _transform_mask(
                        gt_mask,
                        perturbation.perturbed_image.size,
                        perturbation.mask_transform,
                    )
                    pert_box = mask_to_box(pert_gt_mask)
                    if pert_box is None:
                        continue
                    try:
                        pert_preds = _predict_level(
                            model_name,
                            adapter,
                            perturbation.perturbed_image,
                            level_data,
                            pert_box,
                        )
                    except Exception:
                        continue
                    _, pert_iou, pert_score = _best_mask_vs_gt(pert_preds, pert_gt_mask)
                    drop = float(clean_iou - pert_iou)
                    model_level_drops[model_name][level_key].append(drop)
                    pert_records.append(
                        {
                            "perturbation": perturbation.name,
                            "clean_iou": clean_iou,
                            "pert_iou": pert_iou,
                            "miou_drop": drop,
                            "clean_score": best_score,
                            "pert_score": pert_score,
                        }
                    )

                model_record["levels"][level_key] = {
                    "clean_miou": clean_iou,
                    "num_perturbations": len(pert_records),
                    "perturbations": pert_records,
                }

            sample_record["models"][model_name] = model_record

        per_sample.append(sample_record)

    total_time = time.time() - t0
    logger.info("Segmentation evaluation complete: %d samples in %.1fs", len(per_sample), total_time)

    agg: Dict[str, Any] = {"per_model": {}}
    level_order = ["L1_COARSE", "L2_MEDIUM", "L3_FINE"]
    for model_name in models_to_eval:
        model_agg: Dict[str, Any] = {}
        for level_key in level_order:
            drops = model_level_drops[model_name].get(level_key, [])
            clean_vals = model_level_clean[model_name].get(level_key, [])
            model_agg[level_key] = {
                "mean_drop": float(np.mean(drops)) if drops else 0.0,
                "std_drop": float(np.std(drops)) if drops else 0.0,
                "mean_clean_miou": float(np.mean(clean_vals)) if clean_vals else 0.0,
                "num_samples": len(drops),
            }
        agg["per_model"][model_name] = model_agg

    tests: Dict[str, Any] = {}
    for model_name in models_to_eval:
        present_levels = [lk for lk in level_order if model_level_drops[model_name].get(lk)]
        if len(present_levels) < 2:
            continue

        mean_drops = [float(np.mean(model_level_drops[model_name][lk])) for lk in present_levels]
        ranks = list(range(1, len(present_levels) + 1))
        is_mono, tau = monotonicity_test(mean_drops)
        rho, p = spearman_correlation(ranks, mean_drops)
        tests[f"monotonicity_{model_name}"] = {
            "is_monotonic": is_mono,
            "kendall_tau": tau,
            "values": dict(zip(present_levels, mean_drops)),
            "passed": tau > 0.6 if model_name == "sam3" else abs(tau) < 0.5,
        }
        tests[f"spearman_{model_name}"] = {
            "rho": rho,
            "p_value": p,
            "values": dict(zip(present_levels, mean_drops)),
            "passed": rho > 0.8 if model_name == "sam3" else abs(rho) < 0.5,
        }

    tests["hypothesis_supported"] = (
        tests.get("spearman_sam3", {}).get("passed", False)
        and tests.get("spearman_sam2", {}).get("passed", False)
    )

    save_json(agg, out_dir / "summary.json")
    save_json(tests, out_dir / "hypothesis_tests.json")

    jsonl_path = out_dir / "per_sample.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w") as handle:
        for record in per_sample:
            handle.write(json.dumps(record, default=str) + "\n")

    for adapter in models_to_eval.values():
        try:
            adapter.unload()
        except Exception:
            pass

    metrics = {
        "num_samples": len(per_sample),
        "total_time_s": total_time,
        "models_evaluated": list(models_to_eval.keys()),
    }
    return ExperimentResult(
        experiment_id=6,
        experiment_name="exp6_segmentation_granularity",
        config=exp_cfg,
        metrics=metrics,
        per_sample=per_sample,
        hypothesis_tests=tests,
    )
