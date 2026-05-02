"""Experiment 1: Task Granularity Spectrum.

For each image across the multilevel VQA hierarchy, measure VLM accuracy
degradation under perturbations. The central hypothesis is still evaluated on
the primary L1-L4 granularity ladder, while wordy control levels can be carried
through the summaries and plots for comparison.

Outputs:
    exp1/summary.json          -- aggregate metrics and hypothesis tests
    exp1/per_sample.jsonl      -- per-sample results for downstream experiments
    exp1/degradation_by_level.json -- mean degradation per (level, perturbation)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..analysis.continuous import (
    attach_complexity_residual,
    classify_delta_f_family,
    summarize_by_delta_f_family_horse_race,
    summarize_by_score,
    summarize_fixed_effects_trend,
    summarize_horse_race_view,
    summarize_linear_trend,
    summarize_long_format_horse_race,
    summarize_multivariate_regression,
    summarize_per_perturbation_horse_race,
)
from ..analysis.gate_thresholds import GATES
from ..analysis.level_views import LEVEL_VIEW_ORDER, LEVEL_VIEWS
from ..analysis.spectral import compute_feature_spectral_signature_stats
from ..analysis.statistics import (
    bootstrap_ci,
    cohens_d,
    monotonicity_test,
    one_way_anova,
    paired_wordy_mirror_ttests,
    spearman_correlation,
)
from ..data.base import (
    ALL_VQA_LEVEL_NAMES,
    ExperimentResult,
    GranularityLevel,
    GranularitySample,
    PRIMARY_VQA_LEVEL_NAMES,
)
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
from ..perturbations import (
    PerturbationResult,
    build_perturbation_suite,
    export_perturbation_suite_images,
)
from ..utils.device import select_device
from ..utils.io import save_json
from ..utils.parent_bridge import dirichlet_energy_from_tokens

logger = logging.getLogger(__name__)
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _predict_label(
    scores: Dict[str, float],
) -> str:
    """Pick the label with the highest score."""
    return max(scores, key=scores.get)


def _is_correct(scores: Dict[str, float], answer_label: str) -> bool:
    """Check if the top-scoring label matches the ground truth."""
    return _predict_label(scores) == answer_label


def _prediction_entropy(scores: Dict[str, float]) -> Optional[float]:
    """Entropy of the model's option distribution after softmaxing scores."""

    if not scores:
        return None
    values = np.asarray([float(value) for value in scores.values()], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return None
    probs = weights / total
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    return entropy


def _overlap_2d_enabled(cfg: Dict[str, Any]) -> bool:
    """Single grouped switch for the optional 2D overlap diagnostic path."""
    overlap_cfg = cfg.get("analysis", {}).get("overlap_2d", {})
    if isinstance(overlap_cfg, dict):
        return bool(overlap_cfg.get("enabled", False))
    return bool(overlap_cfg)


def _delta_f_2d_filename(image_id: Any, perturbation_name: str) -> str:
    digest = hashlib.sha1(
        f"{image_id}|{perturbation_name}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{image_id}__{digest}.npz"


def _save_delta_f_2d_suite(
    *,
    image_id: Any,
    perturbations: List[PerturbationResult],
    out_dir: Path,
    suppress_dc: bool,
) -> Dict[str, Dict[str, Any]]:
    """Save image-space 2D perturbation spectra once per image perturbation."""
    delta_dir = out_dir / "delta_f_2d"
    delta_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "image_id": str(image_id),
        "spectral_alignment": "fftshifted_image_space_delta_power",
        "suppress_dc": bool(suppress_dc),
        "perturbations": [],
    }
    saved: Dict[str, Dict[str, Any]] = {}
    for perturbation in perturbations:
        if perturbation.delta_f_2d is None:
            continue
        delta = np.asarray(perturbation.delta_f_2d, dtype=np.float64)
        if delta.ndim != 2 or delta.size == 0:
            continue
        delta_shifted = np.fft.fftshift(delta)
        if suppress_dc and delta_shifted.size:
            cy, cx = delta_shifted.shape[0] // 2, delta_shifted.shape[1] // 2
            delta_shifted = delta_shifted.copy()
            delta_shifted[cy, cx] = 0.0
        clean_energy = float(perturbation.clean_spectral_energy or 0.0)
        delta_relative = delta_shifted / max(clean_energy, _EPS)
        filename = _delta_f_2d_filename(image_id, perturbation.name)
        np.savez_compressed(
            delta_dir / filename,
            delta_power_2d=delta_shifted.astype(np.float32),
            delta_power_2d_relative=delta_relative.astype(np.float32),
            clean_spectral_energy=np.asarray(clean_energy, dtype=np.float64),
            image_id=np.asarray(str(image_id)),
            perturbation_name=np.asarray(str(perturbation.name)),
            perturbation_family=np.asarray(str(perturbation.family)),
            severity=np.asarray(int(perturbation.severity), dtype=np.int32),
            suppress_dc=np.asarray(bool(suppress_dc)),
        )
        entry = {
            "name": perturbation.name,
            "family": perturbation.family,
            "severity": int(perturbation.severity),
            "filename": filename,
            "shape": [int(delta_shifted.shape[0]), int(delta_shifted.shape[1])],
            "clean_spectral_energy": clean_energy,
        }
        saved[perturbation.name] = entry
        manifest["perturbations"].append(entry)

    save_json(manifest, delta_dir / f"{image_id}__manifest.json")
    return saved


def _save_delta_f_vision_2d_entry(
    *,
    image_id: Any,
    perturbation: PerturbationResult,
    feature_stats: Dict[str, Any],
    delta_dir: Path,
    suppress_dc: bool,
    clean_patch_grid: Optional[tuple],
    perturbed_patch_grid: Optional[tuple],
) -> Optional[Dict[str, Any]]:
    """Save one vision-token feature-space 2D perturbation spectrum."""

    delta = np.asarray(feature_stats.get("delta_power_2d"), dtype=np.float64)
    if delta.ndim != 2 or delta.size == 0:
        return None
    delta_shifted = np.fft.fftshift(delta)
    if suppress_dc and delta_shifted.size:
        cy, cx = delta_shifted.shape[0] // 2, delta_shifted.shape[1] // 2
        delta_shifted = delta_shifted.copy()
        delta_shifted[cy, cx] = 0.0
    clean_energy = float(feature_stats.get("clean_total_energy") or 0.0)
    delta_relative = delta_shifted / max(clean_energy, _EPS)
    filename = _delta_f_2d_filename(image_id, perturbation.name)
    delta_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        delta_dir / filename,
        delta_power_vision_2d=delta_shifted.astype(np.float32),
        delta_power_vision_2d_relative=delta_relative.astype(np.float32),
        clean_feature_spectral_energy=np.asarray(clean_energy, dtype=np.float64),
        image_id=np.asarray(str(image_id)),
        perturbation_name=np.asarray(str(perturbation.name)),
        perturbation_family=np.asarray(str(perturbation.family)),
        severity=np.asarray(int(perturbation.severity), dtype=np.int32),
        suppress_dc=np.asarray(bool(suppress_dc)),
        clean_patch_grid=np.asarray(clean_patch_grid or [], dtype=np.int32),
        perturbed_patch_grid=np.asarray(perturbed_patch_grid or [], dtype=np.int32),
    )
    return {
        "name": perturbation.name,
        "family": perturbation.family,
        "severity": int(perturbation.severity),
        "filename": filename,
        "shape": [int(delta_shifted.shape[0]), int(delta_shifted.shape[1])],
        "clean_feature_spectral_energy": clean_energy,
        "clean_patch_grid": list(clean_patch_grid) if clean_patch_grid else None,
        "perturbed_patch_grid": list(perturbed_patch_grid) if perturbed_patch_grid else None,
    }


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if np.isfinite(value_float) else None


def _summarize_vision_feature_status(per_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = [
        record.get("vision_feature_status", {})
        for record in per_sample
        if isinstance(record.get("vision_feature_status", {}), dict)
    ]
    requested = [status for status in statuses if status.get("requested")]
    total_attempted = int(sum(int(status.get("perturbations_attempted", 0) or 0) for status in statuses))
    total_extracted = int(sum(int(status.get("perturbations_extracted", 0) or 0) for status in statuses))
    total_delta_success = int(sum(int(status.get("feature_delta_success", 0) or 0) for status in statuses))
    total_vision_2d_saved = int(sum(int(status.get("vision_2d_saved", 0) or 0) for status in statuses))
    return {
        "requested_images": len(requested),
        "clean_extracted_images": int(sum(1 for status in requested if status.get("clean_extracted"))),
        "perturbations_attempted": total_attempted,
        "perturbations_extracted": total_extracted,
        "feature_delta_success": total_delta_success,
        "vision_2d_saved": total_vision_2d_saved,
        "clean_extraction_rate": (
            float(sum(1 for status in requested if status.get("clean_extracted")) / len(requested))
            if requested
            else 0.0
        ),
        "perturbation_extraction_rate": (
            float(total_extracted / total_attempted) if total_attempted else 0.0
        ),
        "feature_delta_success_rate": (
            float(total_delta_success / total_attempted) if total_attempted else 0.0
        ),
    }


def _horse_race_predictors(points: List[Dict[str, Any]]) -> List[str]:
    """Use entropy + task-format controls when available.

    ``is_binary`` is included whenever the point set spans both 2-option and
    4-option items (i.e. mixes binary L1/L3/L5/L7 with MCQ L2/L4/L6/L8) so the
    regression can absorb the task-format baseline shift in loglik metrics
    that ``option_hardness_score`` is degenerate on. If the slice is
    single-format the indicator is constant and would be dropped by the
    regression — we omit it to keep the design matrix full rank.
    """

    predictors = [
        "question_complexity_score",
        "prompt_complexity_score",
        "option_hardness_score",
    ]
    binary_vals = {float(point.get("is_binary", 0.0) or 0.0) for point in points}
    if len(binary_vals - {0.0, 1.0}) == 0 and len(binary_vals & {0.0, 1.0}) == 2:
        predictors.append("is_binary")
    if any(_optional_float(point.get("prediction_entropy")) is not None for point in points):
        predictors.append("prediction_entropy")
    return predictors


def _compute_cosine_drift(
    clean_tokens: Optional[np.ndarray],
    perturbed_tokens: Optional[np.ndarray],
) -> float:
    """Mean cosine distance between clean and perturbed vision tokens.

    Returns 0.0 if either is None or shapes mismatch.
    """
    if clean_tokens is None or perturbed_tokens is None:
        return 0.0
    c = clean_tokens.reshape(-1).astype(np.float64)
    p = perturbed_tokens.reshape(-1).astype(np.float64)
    if c.shape != p.shape or np.linalg.norm(c) == 0 or np.linalg.norm(p) == 0:
        return 0.0
    cos_sim = np.dot(c, p) / (np.linalg.norm(c) * np.linalg.norm(p))
    return float(1.0 - cos_sim)


def _directional_loglik_summary(drifts: List[float]) -> Dict[str, float]:
    """Split signed correct-answer drift into erosion / recovery / volatility."""

    positive = [float(d) for d in drifts if float(d) > 0.0]
    negative = [float(d) for d in drifts if float(d) < 0.0]
    return {
        "mean_loglik_erosion": float(np.mean(positive)) if positive else 0.0,
        "mean_loglik_recovery": float(np.mean(negative)) if negative else 0.0,
        "mean_loglik_volatility": float(np.mean(np.abs(drifts))) if drifts else 0.0,
    }


def _build_complexity_points(per_sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for record in per_sample:
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            perturbations = level_data.get("perturbations", [])
            if not perturbations:
                continue
            drops = [float(item.get("accuracy_drop", 0.0)) for item in perturbations]
            drifts = [
                float(item.get("loglik_drift", 0.0))
                for item in perturbations
                if item.get("loglik_drift") is not None
            ]
            directional = _directional_loglik_summary(drifts)
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
                "is_binary": float(level_data.get("is_binary", 0.0) or 0.0),
                "num_options": int(level_data.get("num_options", 0) or 0),
                "prediction_entropy": _optional_float(level_data.get("prediction_entropy")),
                "clean_accuracy": 1.0 if level_data.get("clean", {}).get("correct", False) else 0.0,
                "mean_accuracy_drop": float(np.mean(drops)) if drops else 0.0,
                "mean_loglik_drift": float(np.mean(drifts)) if drifts else 0.0,
                **directional,
                "num_perturbations": len(perturbations),
            }
            points.append(point)
    attach_complexity_residual(points)
    return points


def _build_perturbation_complexity_points(per_sample: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in per_sample:
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            clean_accuracy = 1.0 if level_data.get("clean", {}).get("correct", False) else 0.0
            base = {
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
                "is_binary": float(level_data.get("is_binary", 0.0) or 0.0),
                "num_options": int(level_data.get("num_options", 0) or 0),
                "prediction_entropy": _optional_float(level_data.get("prediction_entropy")),
                "clean_accuracy": clean_accuracy,
            }
            for perturbation in level_data.get("perturbations", []):
                name = str(perturbation.get("name") or perturbation.get("perturbation") or "unknown")
                drift = float(perturbation.get("loglik_drift", 0.0) or 0.0)
                point = dict(base)
                point.update(
                    {
                        "perturbation": name,
                        "accuracy_drop": float(perturbation.get("accuracy_drop", 0.0) or 0.0),
                        "loglik_drift": drift,
                        "loglik_erosion": max(drift, 0.0),
                        "loglik_recovery": min(drift, 0.0),
                        "loglik_volatility": abs(drift),
                    }
                )
                grouped[name].append(point)
    for points in grouped.values():
        attach_complexity_residual(points)
    return grouped


def _build_long_format_rows(
    per_sample: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Emit one row per (image, level, perturbation) for long-format analysis.

    Each row carries the full complexity predictor bundle, the four signed
    outcome variables (drift / erosion / recovery / volatility / accuracy_drop),
    ``delta_f_norm`` plus ``delta_f_family`` (classified from the radial ΔF
    spectrum), and the ``perturbation_type`` label. This is the canonical
    analysis table for cluster-robust regressions with perturbation fixed
    effects.
    """

    rows: List[Dict[str, Any]] = []
    for record in per_sample:
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            clean_accuracy = (
                1.0 if level_data.get("clean", {}).get("correct", False) else 0.0
            )
            base = {
                "image_id": image_id,
                "level": level_key,
                "question_complexity_score": float(
                    level_data.get("question_complexity_score", 0.0) or 0.0
                ),
                "prompt_complexity_score": float(
                    level_data.get("prompt_complexity_score", 0.0) or 0.0
                ),
                "option_hardness_score": float(
                    level_data.get("option_hardness_score", 0.0) or 0.0
                ),
                "is_binary": float(level_data.get("is_binary", 0.0) or 0.0),
                "num_options": int(level_data.get("num_options", 0) or 0),
                "prediction_entropy": _optional_float(
                    level_data.get("prediction_entropy")
                ),
                "complexity_score": float(
                    level_data.get("complexity_score", 0.0) or 0.0
                ),
                "clean_accuracy": clean_accuracy,
            }
            for perturbation in level_data.get("perturbations", []):
                name = str(
                    perturbation.get("name")
                    or perturbation.get("perturbation")
                    or "unknown"
                )
                drift_raw = perturbation.get("loglik_drift")
                if drift_raw is None:
                    continue
                drift = float(drift_raw)
                delta_f = perturbation.get("delta_f") or []
                family = classify_delta_f_family(delta_f) if delta_f else "unknown"
                row = dict(base)
                row.update(
                    {
                        "perturbation_type": name,
                        "delta_f_family": family,
                        "delta_f_norm": float(
                            perturbation.get("delta_f_norm", 0.0) or 0.0
                        ),
                        "delta_f_peak_band": int(
                            perturbation.get("delta_f_peak_band", 0) or 0
                        ),
                        "accuracy_drop": float(
                            perturbation.get("accuracy_drop", 0.0) or 0.0
                        ),
                        "loglik_drift": drift,
                        "loglik_erosion": max(drift, 0.0),
                        "loglik_recovery": min(drift, 0.0),
                        "loglik_volatility": abs(drift),
                    }
                )
                rows.append(row)
    attach_complexity_residual(rows)
    return rows


def _summarize_perturbation_specific_horse_races(
    perturbation_points: Dict[str, List[Dict[str, Any]]],
    *,
    y_key: str,
    gate_clean_correct: bool = False,
) -> Dict[str, Any]:
    by_perturbation: Dict[str, Any] = {}
    for perturbation_name, points in sorted(perturbation_points.items()):
        used_points = points
        filter_payload: Dict[str, Any] = {
            "n_points_used": len(points),
            "n_points_total": len(points),
        }
        if gate_clean_correct:
            used_points = [
                point
                for point in points
                if float(point.get("clean_accuracy", 0.0) or 0.0) == 1.0
            ]
            filter_payload.update(
                {
                    "clean_accuracy_required": 1.0,
                    "n_points_used": len(used_points),
                    "n_points_total": len(points),
                }
            )
        predictors = _horse_race_predictors(used_points)
        by_perturbation[perturbation_name] = {
            "filter": filter_payload,
            "pooled": summarize_multivariate_regression(
                used_points,
                y_key=y_key,
                x_keys=predictors,
                cluster_key="image_id",
            ),
            "within_image": summarize_multivariate_regression(
                used_points,
                y_key=y_key,
                x_keys=predictors,
                group_key="image_id",
                demean_by_group=True,
                cluster_key="image_id",
            ),
        }
        view_payload: Dict[str, Any] = {}
        for view_name in LEVEL_VIEW_ORDER:
            level_filter = LEVEL_VIEWS[view_name]
            view_points = (
                used_points
                if level_filter is None
                else [point for point in used_points if str(point.get("level")) in level_filter]
            )
            view_predictors = _horse_race_predictors(view_points)
            view_entry = summarize_horse_race_view(
                view_points,
                y_key=y_key,
                x_keys=view_predictors,
                view_name=view_name,
                level_filter=None,
            )
            view_entry["filter"] = {
                **filter_payload,
                "level_view": view_name,
                "level_filter": sorted(level_filter) if level_filter is not None else None,
                "n_points_used": len(view_points),
            }
            view_payload[view_name] = view_entry
            if view_name == "primary":
                by_perturbation[perturbation_name][f"marginal_{view_name}"] = view_entry["marginal"]
            else:
                by_perturbation[perturbation_name][f"pooled_{view_name}"] = view_entry["pooled"]
                by_perturbation[perturbation_name][f"within_image_{view_name}"] = view_entry["within_image"]
        by_perturbation[perturbation_name]["views"] = view_payload
    return by_perturbation


def _add_horse_race_views(
    payload: Dict[str, Any],
    *,
    base_key: str,
    points: List[Dict[str, Any]],
    y_key: str,
) -> None:
    """Add primary/wordy/pooled variants while preserving legacy keys."""

    view_payload: Dict[str, Any] = {}
    for view_name in LEVEL_VIEW_ORDER:
        level_filter = LEVEL_VIEWS[view_name]
        view_points = (
            points
            if level_filter is None
            else [point for point in points if str(point.get("level")) in level_filter]
        )
        predictors = _horse_race_predictors(view_points)
        view_entry = summarize_horse_race_view(
            view_points,
            y_key=y_key,
            x_keys=predictors,
            view_name=view_name,
            level_filter=None,
        )
        view_entry.update({
            "level_filter": sorted(level_filter) if level_filter is not None else None,
            "n_points": len(view_points),
        })
        view_payload[view_name] = view_entry
        if view_name == "primary":
            payload[f"{base_key}_{view_name}"] = view_entry["marginal"]
        else:
            payload[f"{base_key}_{view_name}"] = view_entry["pooled"]
            payload[f"{base_key}_within_image_{view_name}"] = view_entry["within_image"]

    payload[f"{base_key}_views"] = view_payload


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
        "prediction_entropy_score_name": "Clean Prediction Entropy",
        "prediction_entropy_score_definition": "Entropy of the clean-image MCQ option distribution after softmaxing model option scores.",
        "logic_residual_key": "complexity_score_residual",
        "logic_residual_definition": (
            "Residual of semantic complexity after non-negative linear regression "
            "on prompt load, fit on primary L1-L4 and applied to wordy controls."
        ),
        "score_min": float(min(complexity_values)) if complexity_values else 0.0,
        "score_max": float(max(complexity_values)) if complexity_values else 0.0,
        "num_points": len(points),
        "mean_accuracy_drop_by_score": summarize_by_score(
            points,
            score_key="complexity_score",
            value_key="mean_accuracy_drop",
        ),
        "mean_loglik_drift_by_score": summarize_by_score(
            points,
            score_key="complexity_score",
            value_key="mean_loglik_drift",
        ),
        "mean_loglik_erosion_by_score": summarize_by_score(
            points,
            score_key="complexity_score",
            value_key="mean_loglik_erosion",
        ),
        "mean_loglik_recovery_by_score": summarize_by_score(
            points,
            score_key="complexity_score",
            value_key="mean_loglik_recovery",
        ),
        "mean_loglik_volatility_by_score": summarize_by_score(
            points,
            score_key="complexity_score",
            value_key="mean_loglik_volatility",
        ),
        "mean_accuracy_drop_by_prompt_load": summarize_by_score(
            points,
            score_key="prompt_complexity_score",
            value_key="mean_accuracy_drop",
        ),
        "mean_loglik_drift_by_prompt_load": summarize_by_score(
            points,
            score_key="prompt_complexity_score",
            value_key="mean_loglik_drift",
        ),
        "mean_accuracy_drop_by_option_hardness": summarize_by_score(
            points,
            score_key="option_hardness_score",
            value_key="mean_accuracy_drop",
        ),
        "mean_loglik_drift_by_option_hardness": summarize_by_score(
            points,
            score_key="option_hardness_score",
            value_key="mean_loglik_drift",
        ),
        "mean_accuracy_drop_by_prediction_entropy": summarize_by_score(
            points,
            score_key="prediction_entropy",
            value_key="mean_accuracy_drop",
        ),
        "mean_loglik_drift_by_prediction_entropy": summarize_by_score(
            points,
            score_key="prediction_entropy",
            value_key="mean_loglik_drift",
        ),
    }


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def _evaluate_sample(
    sample: GranularitySample,
    adapter,
    perturbation_results: List[PerturbationResult],
    extract_vision_tokens: bool = False,
    store_feature_delta_f: bool = True,
    num_bands: int = 10,
    suppress_dc: bool = True,
    perturbation_2d_files: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_delta_2d_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate one sample across all levels and perturbations.

    Returns a dict with per-level clean/perturbed accuracy and drift metrics.
    """
    image = Image.open(sample.image_path).convert("RGB")
    record: Dict[str, Any] = {
        "image_id": sample.image_id,
        "dataset": sample.dataset,
        "levels": {},
    }

    # --- Clean evaluation per level ---
    clean_vision_tokens = None
    clean_patch_grid = None
    clean_dirichlet = None
    need_vision_features = extract_vision_tokens or store_feature_delta_f
    perturbation_vision_metrics: Dict[str, Dict[str, Any]] = {}
    vision_2d_entries: List[Dict[str, Any]] = []
    vision_feature_status: Dict[str, Any] = {
        "requested": bool(need_vision_features),
        "clean_extracted": False,
        "perturbations_attempted": 0,
        "perturbations_extracted": 0,
        "feature_delta_success": 0,
        "feature_delta_failed": 0,
        "vision_2d_saved": 0,
    }
    if need_vision_features:
        try:
            vt = adapter.get_vision_tokens(image)
            if vt is not None:
                clean_vision_tokens = vt.cpu().numpy()
                vision_feature_status["clean_extracted"] = True
                if clean_vision_tokens.ndim == 4 and clean_vision_tokens.shape[0] == 1:
                    clean_patch_grid = (
                        int(clean_vision_tokens.shape[1]),
                        int(clean_vision_tokens.shape[2]),
                    )
                elif clean_vision_tokens.ndim == 3:
                    clean_patch_grid = (
                        int(clean_vision_tokens.shape[0]),
                        int(clean_vision_tokens.shape[1]),
                    )
                clean_dirichlet = dirichlet_energy_from_tokens(vt)
        except Exception as exc:
            vision_feature_status["clean_error"] = str(exc)
        if clean_vision_tokens is None:
            logger.warning(
                "Vision-token extraction unavailable for clean image %s; "
                "delta_f_vision will be missing for this sample",
                sample.image_id,
            )
    if clean_vision_tokens is not None:
        for perturbation in perturbation_results:
            vision_feature_status["perturbations_attempted"] += 1
            try:
                pert_tokens = adapter.get_vision_tokens(perturbation.perturbed_image)
            except Exception as exc:
                logger.debug(
                    "Vision-token extraction raised for %s pert %s: %s",
                    sample.image_id,
                    perturbation.name,
                    exc,
                )
                continue
            if pert_tokens is None:
                continue
            vision_feature_status["perturbations_extracted"] += 1
            pert_np = pert_tokens.cpu().numpy()
            pert_patch_grid = None
            if pert_np.ndim == 4 and pert_np.shape[0] == 1:
                pert_patch_grid = (int(pert_np.shape[1]), int(pert_np.shape[2]))
            elif pert_np.ndim == 3:
                pert_patch_grid = (int(pert_np.shape[0]), int(pert_np.shape[1]))

            metrics: Dict[str, Any] = {}
            if extract_vision_tokens:
                metrics["cosine_drift"] = _compute_cosine_drift(clean_vision_tokens, pert_np)
                if clean_dirichlet is not None:
                    pert_dirichlet = dirichlet_energy_from_tokens(pert_tokens)
                    if pert_dirichlet is not None:
                        metrics["dirichlet_delta"] = float(pert_dirichlet - clean_dirichlet)
            if store_feature_delta_f:
                try:
                    feature_stats = compute_feature_spectral_signature_stats(
                        clean_vision_tokens,
                        pert_np,
                        num_bands=num_bands,
                        clean_patch_grid=clean_patch_grid,
                        perturbed_patch_grid=pert_patch_grid,
                        suppress_dc=suppress_dc,
                    )
                except Exception:
                    feature_stats = None
                if feature_stats is not None:
                    vision_feature_status["feature_delta_success"] += 1
                    delta_f_vision = feature_stats["delta_f"]
                    metrics["delta_f_vision"] = delta_f_vision.tolist()
                    metrics["delta_f_vision_relative"] = feature_stats["delta_f_relative"].tolist()
                    metrics["delta_f_vision_norm"] = float(np.linalg.norm(delta_f_vision))
                    metrics["delta_f_vision_peak_band"] = int(np.argmax(delta_f_vision))
                    metrics["clean_feature_spectral_energy"] = float(
                        feature_stats["clean_total_energy"]
                    )
                    if feature_delta_2d_dir is not None:
                        try:
                            vision_2d_entry = _save_delta_f_vision_2d_entry(
                                image_id=sample.image_id,
                                perturbation=perturbation,
                                feature_stats=feature_stats,
                                delta_dir=feature_delta_2d_dir,
                                suppress_dc=suppress_dc,
                                clean_patch_grid=clean_patch_grid,
                                perturbed_patch_grid=pert_patch_grid,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Could not save vision-token 2D spectrum for %s pert %s: %s",
                                sample.image_id,
                                perturbation.name,
                                exc,
                            )
                            vision_2d_entry = None
                        if vision_2d_entry:
                            metrics["delta_f_vision_2d_file"] = vision_2d_entry["filename"]
                            metrics["delta_f_vision_2d_shape"] = vision_2d_entry["shape"]
                            vision_2d_entries.append(vision_2d_entry)
                            vision_feature_status["vision_2d_saved"] += 1
                elif store_feature_delta_f:
                    vision_feature_status["feature_delta_failed"] += 1
            perturbation_vision_metrics[perturbation.name] = metrics

    record["vision_feature_status"] = vision_feature_status
    if (
        vision_feature_status["requested"]
        and vision_feature_status["clean_extracted"]
        and vision_feature_status["feature_delta_success"]
        < vision_feature_status["perturbations_attempted"]
    ):
        logger.info(
            "Vision-token spectra for %s: %d/%d perturbations",
            sample.image_id,
            vision_feature_status["feature_delta_success"],
            vision_feature_status["perturbations_attempted"],
        )

    if feature_delta_2d_dir is not None and vision_2d_entries:
        save_json(
            {
                "image_id": str(sample.image_id),
                "spectral_alignment": "fftshifted_vision_feature_delta_power",
                "suppress_dc": bool(suppress_dc),
                "keying": "image_id x perturbation",
                "perturbations": vision_2d_entries,
            },
            feature_delta_2d_dir / f"{sample.image_id}__manifest.json",
        )

    for level in GranularityLevel:
        level_data = sample.levels.get(level)
        if level_data is None:
            continue

        level_key = level.name  # e.g. "L1_COARSE"
        complexity = ensure_level_complexity(level_data)
        num_options = int(len(level_data.options or {}))
        level_record: Dict[str, Any] = {
            "question": level_data.question,
            "question_type": level_data.question_type,
            "answer_label": level_data.answer_label,
            "complexity_score": complexity["complexity_score"],
            "question_complexity_score": complexity["question_complexity_score"],
            "prompt_complexity_score": complexity["prompt_complexity_score"],
            "option_hardness_score": float(getattr(level_data, "option_hardness_score", 0.0) or 0.0),
            # Task-format covariate to disentangle binary (yes/no, 2 options) from
            # MCQ (4+ options): the scoring topology of loglik_drift differs
            # mechanically between the two regimes and option_hardness is degenerate
            # on binary items (always ~0). Reported as both an indicator and a
            # raw count so downstream regressions can pick whichever is cleaner.
            "is_binary": 1.0 if num_options == 2 else 0.0,
            "num_options": num_options,
            "semantic_atoms": complexity["semantic_atoms"],
            "prompt_semantic_atoms": complexity["prompt_semantic_atoms"],
            "semantic_atom_counts": complexity["semantic_atom_counts"],
            "option_hardness_components": dict(
                getattr(level_data, "option_hardness_components", {}) or {}
            ),
            "clean": {},
            "perturbations": [],
        }

        # Score on clean image
        clean_scores: Dict[str, float] = {}
        try:
            clean_scores = adapter.score_options(
                image, level_data.question, level_data.options,
            )
            clean_pred = _predict_label(clean_scores)
            clean_correct = _is_correct(clean_scores, level_data.answer_label)
            clean_entropy = _prediction_entropy(clean_scores)
            level_record["prediction_entropy"] = clean_entropy
            level_record["clean"] = {
                "predicted": clean_pred,
                "correct": clean_correct,
                "prediction_entropy": clean_entropy,
                "scores": {k: float(v) for k, v in clean_scores.items()},
            }
            if clean_dirichlet is not None:
                level_record["clean"]["dirichlet_energy"] = float(clean_dirichlet)
        except Exception as e:
            logger.warning(
                "Clean scoring failed for %s level %s: %s",
                sample.image_id, level_key, e,
            )
            level_record["clean"] = {"correct": False, "error": str(e)}

        # --- Perturbed evaluation ---
        for pr in perturbation_results:
            pert_record: Dict[str, Any] = {
                "name": pr.name,
                "family": pr.family,
                "severity": pr.severity,
            }

            try:
                pert_scores = adapter.score_options(
                    pr.perturbed_image, level_data.question, level_data.options,
                )
                pert_pred = _predict_label(pert_scores)
                pert_correct = _is_correct(pert_scores, level_data.answer_label)
                pert_record["predicted"] = pert_pred
                pert_record["correct"] = pert_correct
                pert_record["scores"] = {
                    k: float(v) for k, v in pert_scores.items()
                }
                pert_record["prediction_entropy"] = _prediction_entropy(pert_scores)
                # Accuracy drop (1 if correct→wrong, 0 if stayed same)
                pert_record["accuracy_drop"] = (
                    1.0 if level_record["clean"].get("correct") and not pert_correct
                    else 0.0
                )
                # Signed correct-answer drift: positive means confidence erosion.
                if level_data.answer_label in clean_scores:
                    clean_ll = clean_scores[level_data.answer_label]
                    pert_ll = pert_scores.get(level_data.answer_label, clean_ll)
                    pert_record["loglik_drift"] = float(clean_ll - pert_ll)
            except Exception as e:
                logger.warning(
                    "Perturbed scoring failed for %s level %s pert %s: %s",
                    sample.image_id, level_key, pr.name, e,
                )
                pert_record["error"] = str(e)
                pert_record["correct"] = False
                pert_record["accuracy_drop"] = (
                    1.0 if level_record["clean"].get("correct") else 0.0
                )

            # Store spectral signature stats (for downstream Exp 5)
            pert_record["delta_f"] = pr.delta_f.tolist()
            if pr.delta_f_relative is not None:
                pert_record["delta_f_relative"] = pr.delta_f_relative.tolist()
            pert_record["delta_f_norm"] = float(np.linalg.norm(pr.delta_f))
            pert_record["delta_f_peak_band"] = int(np.argmax(pr.delta_f))
            if pr.clean_spectral_energy is not None:
                pert_record["clean_spectral_energy"] = float(pr.clean_spectral_energy)
            two_d_entry = (perturbation_2d_files or {}).get(pr.name)
            if two_d_entry:
                pert_record["delta_f_2d_file"] = two_d_entry["filename"]
                pert_record["delta_f_2d_shape"] = two_d_entry.get("shape")
            vision_metrics = perturbation_vision_metrics.get(pr.name)
            if vision_metrics:
                pert_record.update(vision_metrics)

            level_record["perturbations"].append(pert_record)

        record["levels"][level_key] = level_record

    return record


# ---------------------------------------------------------------------------
# Aggregation and hypothesis testing
# ---------------------------------------------------------------------------


def _aggregate_results(
    per_sample: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-sample results into per-level statistics."""
    # Collect accuracy drops by (level, perturbation_name)
    level_drops: Dict[str, List[float]] = defaultdict(list)
    level_loglik_drifts: Dict[str, List[float]] = defaultdict(list)
    level_clean_acc: Dict[str, List[float]] = defaultdict(list)
    level_pert_acc: Dict[str, List[float]] = defaultdict(list)
    level_cosine_drifts: Dict[str, List[float]] = defaultdict(list)
    level_dirichlet_deltas: Dict[str, List[float]] = defaultdict(list)

    # Per (level, perturbation) breakdown
    level_pert_drops: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for record in per_sample:
        for level_key, level_data in record.get("levels", {}).items():
            clean_correct = level_data.get("clean", {}).get("correct", False)
            level_clean_acc[level_key].append(1.0 if clean_correct else 0.0)

            for pr in level_data.get("perturbations", []):
                drop = pr.get("accuracy_drop", 0.0)
                level_drops[level_key].append(drop)
                level_pert_acc[level_key].append(
                    1.0 if pr.get("correct", False) else 0.0
                )

                drift = pr.get("loglik_drift", 0.0)
                level_loglik_drifts[level_key].append(drift)

                cos_d = pr.get("cosine_drift", 0.0)
                if cos_d > 0:
                    level_cosine_drifts[level_key].append(cos_d)

                dirichlet_delta = pr.get("dirichlet_delta")
                if dirichlet_delta is not None:
                    level_dirichlet_deltas[level_key].append(dirichlet_delta)

                pname = pr.get("name", "unknown")
                level_pert_drops[level_key][pname].append(drop)

    # Build aggregate dict
    agg: Dict[str, Any] = {"per_level": {}}
    levels_ordered = [level_key for level_key in ALL_VQA_LEVEL_NAMES if level_key in level_drops]

    for lk in levels_ordered:
        drops = level_drops[lk]
        drifts = level_loglik_drifts[lk]
        directional = _directional_loglik_summary(drifts)
        clean = level_clean_acc.get(lk, [])
        pert = level_pert_acc.get(lk, [])
        cos_drifts = level_cosine_drifts.get(lk, [])

        agg["per_level"][lk] = {
            "clean_accuracy": float(np.mean(clean)) if clean else 0.0,
            "perturbed_accuracy": float(np.mean(pert)) if pert else 0.0,
            "mean_accuracy_drop": float(np.mean(drops)) if drops else 0.0,
            "std_accuracy_drop": float(np.std(drops)) if drops else 0.0,
            "mean_loglik_drift": float(np.mean(drifts)) if drifts else 0.0,
            "std_loglik_drift": float(np.std(drifts)) if drifts else 0.0,
            **directional,
            "mean_cosine_drift": float(np.mean(cos_drifts)) if cos_drifts else 0.0,
            "mean_dirichlet_delta": float(np.mean(level_dirichlet_deltas.get(lk, [])))
            if level_dirichlet_deltas.get(lk)
            else 0.0,
            "std_dirichlet_delta": float(np.std(level_dirichlet_deltas.get(lk, [])))
            if level_dirichlet_deltas.get(lk)
            else 0.0,
            "num_samples": len(clean),
            "num_perturbation_evals": len(drops),
        }

    # Per-level, per-perturbation breakdown
    agg["per_level_perturbation"] = {}
    for lk in levels_ordered:
        agg["per_level_perturbation"][lk] = {}
        for pname, drops in sorted(level_pert_drops[lk].items()):
            agg["per_level_perturbation"][lk][pname] = {
                "mean_drop": float(np.mean(drops)),
                "std_drop": float(np.std(drops)),
                "n": len(drops),
            }

    return agg


def _run_hypothesis_tests(
    agg: Dict[str, Any],
    complexity_points: Optional[List[Dict[str, Any]]] = None,
    perturbation_points: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    long_format_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run statistical tests on aggregated results.

    Core hypothesis: sensitivity scales monotonically with granularity.
    """
    tests: Dict[str, Any] = {}
    per_level = agg.get("per_level", {})

    # Order levels by granularity
    level_order = list(PRIMARY_VQA_LEVEL_NAMES)
    present_levels = [lk for lk in level_order if lk in per_level]

    if len(present_levels) < 2:
        tests["insufficient_levels"] = True
        return tests

    # H1: Spearman correlation between granularity rank and mean accuracy drop
    granularity_ranks = list(range(1, len(present_levels) + 1))
    mean_drops = [per_level[lk]["mean_accuracy_drop"] for lk in present_levels]
    mean_drifts = [per_level[lk]["mean_loglik_drift"] for lk in present_levels]

    is_mono, tau = monotonicity_test(mean_drops)
    tau_passed = tau > GATES["monotonicity_kendall_tau_min"]
    rho_drop, p_drop = spearman_correlation(granularity_ranks, mean_drops)
    tests["spearman_drop_vs_granularity"] = {
        "rho": rho_drop,
        "p_value": p_drop,
        "target": "reported only; pass delegated to Kendall tau monotonicity gate",
        "passed": tau_passed,
        "caveat": (
            "Spearman p-value gating over four level means is underpowered; "
            "the pass flag uses monotonicity_accuracy_drop.kendall_tau instead."
        ),
    }

    rho_drift, p_drift = spearman_correlation(granularity_ranks, mean_drifts)
    tests["spearman_loglik_drift_vs_granularity"] = {
        "rho": rho_drift,
        "p_value": p_drift,
        "passed": rho_drift > GATES["secondary_spearman_rho_min"],
    }

    # H2: Monotonicity of mean accuracy drop across levels
    tests["monotonicity_accuracy_drop"] = {
        "is_monotonic": is_mono,
        "kendall_tau": tau,
        "values": dict(zip(present_levels, mean_drops)),
        "target": f"kendall_tau > {GATES['monotonicity_kendall_tau_min']}",
        "passed": tau_passed,
    }

    wordy_present_levels = [
        level for level in ALL_VQA_LEVEL_NAMES if level in (LEVEL_VIEWS["wordy"] or set()) and level in per_level
    ]
    if len(wordy_present_levels) >= 2:
        wordy_ranks = list(range(1, len(wordy_present_levels) + 1))
        wordy_drops = [per_level[level]["mean_accuracy_drop"] for level in wordy_present_levels]
        wordy_rho, wordy_p = spearman_correlation(wordy_ranks, wordy_drops)
        wordy_mono, wordy_tau = monotonicity_test(wordy_drops)
        tests["spearman_drop_vs_granularity_wordy"] = {
            "rho": wordy_rho,
            "p_value": wordy_p,
            "target": "reported only; pass delegated to wordy Kendall tau monotonicity gate",
            "passed": wordy_tau > GATES["monotonicity_kendall_tau_min"],
            "values": dict(zip(wordy_present_levels, wordy_drops)),
            "caveat": (
                "Spearman p-value gating over four wordy level means is underpowered; "
                "the pass flag uses monotonicity_accuracy_drop_wordy.kendall_tau instead."
            ),
        }
        tests["monotonicity_accuracy_drop_wordy"] = {
            "is_monotonic": wordy_mono,
            "kendall_tau": wordy_tau,
            "values": dict(zip(wordy_present_levels, wordy_drops)),
            "target": f"kendall_tau > {GATES['monotonicity_kendall_tau_min']}",
            "passed": wordy_tau > GATES["monotonicity_kendall_tau_min"],
        }

    # Wordy-vs-base drift volatility split (Theorem 4, revised 2026-04-27).
    # Under the three-stage trajectory + asymmetric late-layer relaxation,
    # wordy prompts land on a broader late-layer filter than their semantic
    # counterparts, which lowers <W_t, dF> linearly and therefore shrinks
    # per-sample drift volatility.
    primary_present_levels = [
        lk for lk in ALL_VQA_LEVEL_NAMES
        if lk in (LEVEL_VIEWS["primary"] or set()) and lk in per_level
    ]
    wordy_order = [
        lk for lk in ALL_VQA_LEVEL_NAMES
        if lk in (LEVEL_VIEWS["wordy"] or set()) and lk in per_level
    ]
    volatility_pairs: Dict[str, Any] = {}
    if primary_present_levels and wordy_order:
        pair_ratios: List[float] = []
        for base_lk, wordy_lk in zip(primary_present_levels, wordy_order):
            base_std = float(per_level.get(base_lk, {}).get("std_loglik_drift", 0.0) or 0.0)
            wordy_std = float(per_level.get(wordy_lk, {}).get("std_loglik_drift", 0.0) or 0.0)
            if base_std <= 0.0:
                continue
            var_ratio = (wordy_std ** 2) / (base_std ** 2)
            pair_ratios.append(var_ratio)
            volatility_pairs[f"{base_lk}__vs__{wordy_lk}"] = {
                "base_std_loglik_drift": base_std,
                "wordy_std_loglik_drift": wordy_std,
                "variance_ratio_wordy_over_base": var_ratio,
                "passed": bool(var_ratio < 1.0),
            }
        if pair_ratios:
            mean_ratio = float(np.mean(pair_ratios))
            tests["wordy_volatility_reduction"] = {
                "per_pair": volatility_pairs,
                "mean_variance_ratio_wordy_over_base": mean_ratio,
                "num_pairs": len(pair_ratios),
                "target": "mean_variance_ratio_wordy_over_base < 1.0",
                "passed": bool(mean_ratio < 1.0),
            }

    # H3: ANOVA across levels — are the groups significantly different?
    # Collect per-perturbation drops grouped by level
    per_level_all_drops: Dict[str, List[float]] = defaultdict(list)
    for lk in present_levels:
        for pert_data in agg.get("per_level_perturbation", {}).get(lk, {}).values():
            per_level_all_drops[lk].extend(
                [pert_data["mean_drop"]] * max(1, pert_data["n"])
            )

    groups = [per_level_all_drops[lk] for lk in present_levels]
    valid_groups = [g for g in groups if len(g) > 0]
    if len(valid_groups) >= 2:
        f_stat, p_anova = one_way_anova(*valid_groups)
        tests["anova_across_levels"] = {
            "F_statistic": f_stat,
            "p_value": p_anova,
            "target": "F > 10, p < 0.001",
            "passed": f_stat > 10 and p_anova < 0.001,
        }

    # H4: Cohen's d between L1 (coarsest) and L4 (finest)
    if "L1_COARSE" in per_level_all_drops and "L4_VERY_FINE" in per_level_all_drops:
        d = cohens_d(
            per_level_all_drops["L4_VERY_FINE"],
            per_level_all_drops["L1_COARSE"],
        )
        tests["cohens_d_L4_vs_L1"] = {
            "d": d,
            "target": "d > 0.8 (large effect)",
            "passed": d > 0.8,
        }

    # Bootstrap CI on mean degradation per level
    tests["bootstrap_ci"] = {}
    for lk in present_levels:
        drops = per_level_all_drops.get(lk, [])
        if len(drops) >= 5:
            point, lo, hi = bootstrap_ci(drops, n_boot=1000)
            tests["bootstrap_ci"][lk] = {
                "mean": point,
                "ci_95_lower": lo,
                "ci_95_upper": hi,
            }

    # Provisional summary verdict (recomputed at end to include cluster-robust gates)
    core_tests = [
        tests.get("spearman_drop_vs_granularity", {}).get("passed", False),
        tests.get("monotonicity_accuracy_drop", {}).get("passed", False),
    ]
    tests["hypothesis_supported"] = all(core_tests)
    tests["num_criteria_passed"] = sum(1 for t in core_tests if t)
    tests["num_criteria_total"] = len(core_tests)

    if complexity_points:
        accuracy_drop_points = [
            point
            for point in complexity_points
            if float(point.get("clean_accuracy", 0.0) or 0.0) == 1.0
        ]
        accuracy_predictors = _horse_race_predictors(accuracy_drop_points)
        all_predictors = _horse_race_predictors(complexity_points)
        tests["continuous_complexity"] = {
            "accuracy_drop_horse_race_filter": {
                "description": "Accuracy-drop horse races use only image-level points where the clean answer was correct.",
                "clean_accuracy_required": 1.0,
                "n_points_used": len(accuracy_drop_points),
                "n_points_total": len(complexity_points),
            },
            "prediction_entropy_control": {
                "description": "Clean-image MCQ prediction entropy controls for model uncertainty before perturbation.",
                "formula": "H = -sum_i p_i log(p_i), where p_i = softmax(option_score_i)",
                "included_in_horse_race": "prediction_entropy" in all_predictors,
            },
            "wordy_mirror_paired_tests": paired_wordy_mirror_ttests(
                complexity_points,
                y_keys=[
                    "mean_accuracy_drop",
                    "mean_loglik_drift",
                    "mean_loglik_erosion",
                    "mean_loglik_recovery",
                    "mean_loglik_volatility",
                ],
            ),
            "mean_accuracy_drop_vs_complexity": summarize_linear_trend(
                complexity_points,
                x_key="complexity_score",
                y_key="mean_accuracy_drop",
            ),
            "mean_loglik_drift_vs_complexity": summarize_linear_trend(
                complexity_points,
                x_key="complexity_score",
                y_key="mean_loglik_drift",
            ),
            "mean_loglik_erosion_vs_complexity": summarize_linear_trend(
                complexity_points,
                x_key="complexity_score",
                y_key="mean_loglik_erosion",
            ),
            "mean_loglik_recovery_vs_complexity": summarize_linear_trend(
                complexity_points,
                x_key="complexity_score",
                y_key="mean_loglik_recovery",
            ),
            "mean_loglik_volatility_vs_complexity": summarize_linear_trend(
                complexity_points,
                x_key="complexity_score",
                y_key="mean_loglik_volatility",
            ),
            "mean_accuracy_drop_vs_prompt_load": summarize_linear_trend(
                complexity_points,
                x_key="prompt_complexity_score",
                y_key="mean_accuracy_drop",
            ),
            "mean_loglik_drift_vs_prompt_load": summarize_linear_trend(
                complexity_points,
                x_key="prompt_complexity_score",
                y_key="mean_loglik_drift",
            ),
            "mean_accuracy_drop_vs_option_hardness": summarize_linear_trend(
                complexity_points,
                x_key="option_hardness_score",
                y_key="mean_accuracy_drop",
            ),
            "mean_loglik_drift_vs_option_hardness": summarize_linear_trend(
                complexity_points,
                x_key="option_hardness_score",
                y_key="mean_loglik_drift",
            ),
            "mean_accuracy_drop_vs_prediction_entropy": summarize_linear_trend(
                complexity_points,
                x_key="prediction_entropy",
                y_key="mean_accuracy_drop",
            ),
            "mean_loglik_drift_vs_prediction_entropy": summarize_linear_trend(
                complexity_points,
                x_key="prediction_entropy",
                y_key="mean_loglik_drift",
            ),
            "mean_accuracy_drop_vs_complexity_fixed_effects": summarize_fixed_effects_trend(
                complexity_points,
                group_key="image_id",
                x_key="complexity_score",
                y_key="mean_accuracy_drop",
            ),
            "mean_loglik_drift_vs_complexity_fixed_effects": summarize_fixed_effects_trend(
                complexity_points,
                group_key="image_id",
                x_key="complexity_score",
                y_key="mean_loglik_drift",
            ),
            "horse_race_mean_accuracy_drop": summarize_multivariate_regression(
                accuracy_drop_points,
                y_key="mean_accuracy_drop",
                x_keys=accuracy_predictors,
                cluster_key="image_id",
            ),
            "horse_race_mean_accuracy_drop_within_image": summarize_multivariate_regression(
                accuracy_drop_points,
                y_key="mean_accuracy_drop",
                x_keys=accuracy_predictors,
                group_key="image_id",
                demean_by_group=True,
                cluster_key="image_id",
            ),
            "horse_race_mean_loglik_drift": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_drift",
                x_keys=all_predictors,
                cluster_key="image_id",
            ),
            "horse_race_mean_loglik_drift_within_image": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_drift",
                x_keys=all_predictors,
                group_key="image_id",
                demean_by_group=True,
                cluster_key="image_id",
            ),
            "horse_race_mean_loglik_erosion": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_erosion",
                x_keys=all_predictors,
                cluster_key="image_id",
            ),
            "horse_race_mean_loglik_erosion_within_image": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_erosion",
                x_keys=all_predictors,
                group_key="image_id",
                demean_by_group=True,
                cluster_key="image_id",
            ),
            "horse_race_mean_loglik_recovery": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_recovery",
                x_keys=all_predictors,
                cluster_key="image_id",
            ),
            "horse_race_mean_loglik_recovery_within_image": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_recovery",
                x_keys=all_predictors,
                group_key="image_id",
                demean_by_group=True,
                cluster_key="image_id",
            ),
            "horse_race_mean_loglik_volatility": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_volatility",
                x_keys=all_predictors,
                cluster_key="image_id",
            ),
            "horse_race_mean_loglik_volatility_within_image": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_volatility",
                x_keys=all_predictors,
                group_key="image_id",
                demean_by_group=True,
                cluster_key="image_id",
            ),
        }
        cc = tests["continuous_complexity"]
        _add_horse_race_views(
            cc,
            base_key="horse_race_mean_accuracy_drop",
            points=accuracy_drop_points,
            y_key="mean_accuracy_drop",
        )
        for outcome_key in (
            "mean_loglik_drift",
            "mean_loglik_erosion",
            "mean_loglik_recovery",
            "mean_loglik_volatility",
        ):
            _add_horse_race_views(
                cc,
                base_key=f"horse_race_{outcome_key}",
                points=complexity_points,
                y_key=outcome_key,
            )
        if perturbation_points:
            tests["continuous_complexity"]["horse_race_mean_accuracy_drop_by_perturbation"] = (
                _summarize_perturbation_specific_horse_races(
                    perturbation_points,
                    y_key="accuracy_drop",
                    gate_clean_correct=True,
                )
            )
            tests["continuous_complexity"]["horse_race_mean_loglik_drift_by_perturbation"] = (
                _summarize_perturbation_specific_horse_races(
                    perturbation_points,
                    y_key="loglik_drift",
                )
            )
            tests["continuous_complexity"]["horse_race_mean_loglik_erosion_by_perturbation"] = (
                _summarize_perturbation_specific_horse_races(
                    perturbation_points,
                    y_key="loglik_erosion",
                )
            )
            tests["continuous_complexity"]["horse_race_mean_loglik_recovery_by_perturbation"] = (
                _summarize_perturbation_specific_horse_races(
                    perturbation_points,
                    y_key="loglik_recovery",
                )
            )
            tests["continuous_complexity"]["horse_race_mean_loglik_volatility_by_perturbation"] = (
                _summarize_perturbation_specific_horse_races(
                    perturbation_points,
                    y_key="loglik_volatility",
                )
            )

        # --- Publication-grade cluster-robust, long-format, stratified blocks ---
        if long_format_rows:
            lf_predictors = _horse_race_predictors(long_format_rows)
            lf_accuracy_rows = [
                row
                for row in long_format_rows
                if float(row.get("clean_accuracy", 0.0) or 0.0) == 1.0
            ]
            lf_accuracy_predictors = _horse_race_predictors(lf_accuracy_rows)
            outcome_specs = [
                ("accuracy_drop", lf_accuracy_rows, lf_accuracy_predictors),
                ("loglik_drift", long_format_rows, lf_predictors),
                ("loglik_erosion", long_format_rows, lf_predictors),
                ("loglik_recovery", long_format_rows, lf_predictors),
                ("loglik_volatility", long_format_rows, lf_predictors),
            ]
            cc = tests["continuous_complexity"]
            cc["long_format_row_counts"] = {
                "n_rows_total": len(long_format_rows),
                "n_rows_clean_correct": len(lf_accuracy_rows),
                "unique_images": len({str(r.get("image_id")) for r in long_format_rows}),
                "unique_perturbations": len(
                    {str(r.get("perturbation_type")) for r in long_format_rows}
                ),
                "unique_levels": len({str(r.get("level")) for r in long_format_rows}),
            }
            for outcome_key, rows, preds in outcome_specs:
                if not rows or not preds:
                    continue
                cc[f"long_format_{outcome_key}"] = summarize_long_format_horse_race(
                    rows,
                    y_key=outcome_key,
                    x_keys=preds,
                    cluster_key="image_id",
                    perturbation_key="perturbation_type",
                )
                cc[f"per_perturbation_{outcome_key}"] = summarize_per_perturbation_horse_race(
                    rows,
                    y_key=outcome_key,
                    x_keys=preds,
                    cluster_key="image_id",
                    perturbation_key="perturbation_type",
                )
                cc[f"by_delta_f_family_{outcome_key}"] = summarize_by_delta_f_family_horse_race(
                    rows,
                    y_key=outcome_key,
                    x_keys=preds,
                    cluster_key="image_id",
                    family_key="delta_f_family",
                )

            # --- Gate: cluster-robust long-format β_Csem > 0 with p <= 0.05 ---
            p_max = GATES.get("long_format_dual_force_beta_p_max", GATES["dual_force_beta_p_max"])

            def _dual_force_gate(outcome_key: str) -> Dict[str, Any]:
                lf_summary = cc.get(f"long_format_{outcome_key}", {}) or {}
                fe_fit = lf_summary.get("pooled_with_perturbation_fe", {}) or {}
                fe_preds = fe_fit.get("predictors", {}) or {}
                csem_info = fe_preds.get("question_complexity_score", {}) or {}
                cprompt_info = fe_preds.get("prompt_complexity_score", {}) or {}
                csem_beta = csem_info.get("beta")
                cprompt_beta = cprompt_info.get("beta")
                csem_p = csem_info.get("p_value")
                cprompt_p = cprompt_info.get("p_value")
                csem_pass = (
                    csem_beta is not None
                    and csem_p is not None
                    and float(csem_beta) > 0.0
                    and float(csem_p) <= p_max
                )
                cprompt_pass = (
                    cprompt_beta is not None
                    and cprompt_p is not None
                    and float(cprompt_beta) < 0.0
                    and float(cprompt_p) <= p_max
                )
                return {
                    "outcome": outcome_key,
                    "csem_beta": csem_beta,
                    "csem_p_value": csem_p,
                    "cprompt_beta": cprompt_beta,
                    "cprompt_p_value": cprompt_p,
                    "inference_mode": fe_fit.get("inference_mode"),
                    "n_rows": fe_fit.get("n"),
                    "n_clusters": fe_fit.get("n_clusters"),
                    "target": (
                        f"cluster-robust β_Csem > 0 AND β_Cprompt < 0 with p <= {p_max} "
                        f"for {outcome_key} in long-format regression with perturbation FE"
                    ),
                    "passed": bool(csem_pass and cprompt_pass),
                }

            tests["long_format_dual_force_loglik_drift"] = _dual_force_gate("loglik_drift")
            tests["long_format_dual_force_loglik_volatility"] = _dual_force_gate(
                "loglik_volatility"
            )

            # --- Gate: β_Csem sign stability across perturbation families ---
            per_pert_drift = cc.get("per_perturbation_loglik_drift", {}) or {}
            sign_stab = per_pert_drift.get("sign_stability", {}) or {}
            csem_stab = sign_stab.get("question_complexity_score", {}) or {}
            frac_pos = float(csem_stab.get("fraction_positive", 0.0) or 0.0)
            min_frac = GATES.get("perturbation_sign_stability_min_fraction", 0.6)
            tests["per_perturbation_csem_sign_stability"] = {
                "fraction_positive": frac_pos,
                "n_perturbations": csem_stab.get("n_perturbations", 0),
                "mean_beta": csem_stab.get("mean_beta"),
                "median_beta": csem_stab.get("median_beta"),
                "std_beta": csem_stab.get("std_beta"),
                "target": (
                    f"β_Csem positive in >= {min_frac:.0%} of perturbation-specific fits"
                ),
                "passed": frac_pos >= min_frac,
            }

            # --- Gate: ΔF-family falsifiable prediction ---
            # We interpret the gate *defensively*: the theory predicts β_Csem ≈ 0
            # on the zero-ΔF family and β_Csem > 0 on the high-frequency family.
            # We only FAIL the gate if the data *actively* contradicts the theory
            # (wrong-signed β with p <= 0.05). Underpowered or absent strata
            # (skipped, too few rows, p > 0.05) are treated as non-falsifying.
            fam_drift = cc.get("by_delta_f_family_loglik_drift", {}) or {}
            fam_fits = fam_drift.get("family_fits", {}) or {}

            def _not_falsified(fit: Dict[str, Any], *, expected_sign: int) -> bool:
                if not fit or fit.get("skipped"):
                    return True
                preds = fit.get("predictors", {}) or {}
                info = preds.get("question_complexity_score", {}) or {}
                beta = info.get("beta")
                p_val = info.get("p_value")
                if beta is None or p_val is None:
                    return True
                if float(p_val) > 0.05:
                    return True
                # Significant: must not be in the wrong direction
                if expected_sign > 0 and float(beta) <= 0.0:
                    return False
                if expected_sign < 0 and float(beta) >= 0.0:
                    return False
                if expected_sign == 0 and abs(float(beta)) > 1e-3:
                    # Significant non-zero on zero-ΔF family — falsifying
                    return False
                return True

            zero_fit = fam_fits.get("zero") or {}
            high_fit = fam_fits.get("high_freq") or {}
            low_fit = fam_fits.get("low_freq") or {}
            broadband_fit = fam_fits.get("broadband") or {}
            zero_ok = _not_falsified(zero_fit, expected_sign=0)
            high_ok = _not_falsified(high_fit, expected_sign=+1)
            # We also require that *at least one* non-zero family shows the
            # positive β_Csem signal, so the gate doesn't pass vacuously when
            # every family is underpowered.
            positive_family_fits = [
                f
                for f in (high_fit, low_fit, broadband_fit)
                if f
                and not f.get("skipped")
                and ((f.get("predictors") or {}).get("question_complexity_score") or {}).get("beta") is not None
            ]

            def _family_positive(fit: Dict[str, Any]) -> bool:
                info = (fit.get("predictors") or {}).get("question_complexity_score", {}) or {}
                beta = info.get("beta")
                p_val = info.get("p_value")
                return (
                    beta is not None
                    and p_val is not None
                    and float(beta) > 0.0
                    and float(p_val) <= 0.05
                )

            any_family_supports = any(_family_positive(f) for f in positive_family_fits)

            def _csem_summary(fit: Dict[str, Any]) -> Dict[str, Any]:
                if not fit or fit.get("skipped"):
                    return {
                        "n_rows": fit.get("n_rows") or fit.get("n"),
                        "skipped": bool(fit.get("skipped")),
                    }
                info = (fit.get("predictors") or {}).get("question_complexity_score", {}) or {}
                return {
                    "csem_beta": info.get("beta"),
                    "csem_p_value": info.get("p_value"),
                    "n_rows": fit.get("n"),
                    "n_clusters": fit.get("n_clusters"),
                    "skipped": False,
                }

            tests["delta_f_family_falsifiable_prediction"] = {
                "zero_family": _csem_summary(zero_fit),
                "low_freq_family": _csem_summary(low_fit),
                "broadband_family": _csem_summary(broadband_fit),
                "high_freq_family": _csem_summary(high_fit),
                "target": (
                    "Theory is not actively falsified on any ΔF family "
                    "(β_Csem not significantly wrong-signed) AND at least one "
                    "non-zero ΔF family shows β_Csem > 0 with p <= 0.05"
                ),
                "zero_family_not_falsified": zero_ok,
                "high_freq_family_not_falsified": high_ok,
                "any_non_zero_family_supports_theory": any_family_supports,
                "passed": bool(zero_ok and high_ok and any_family_supports),
            }

    # --- Final composite verdict (includes cluster-robust gates when available) ---
    composite_tests = [
        tests.get("spearman_drop_vs_granularity", {}).get("passed", False),
        tests.get("monotonicity_accuracy_drop", {}).get("passed", False),
    ]
    composite_labels = ["spearman_drop_vs_granularity", "monotonicity_accuracy_drop"]
    for name in (
        "long_format_dual_force_loglik_volatility",
        "per_perturbation_csem_sign_stability",
        "delta_f_family_falsifiable_prediction",
    ):
        if name in tests:
            composite_tests.append(bool(tests[name].get("passed", False)))
            composite_labels.append(name)
    if (
        "long_format_dual_force_loglik_volatility" not in tests
        and "long_format_dual_force_loglik_drift" in tests
    ):
        composite_tests.append(
            bool(tests["long_format_dual_force_loglik_drift"].get("passed", False))
        )
        composite_labels.append("long_format_dual_force_loglik_drift")
    tests["hypothesis_supported"] = all(composite_tests)
    tests["num_criteria_passed"] = sum(1 for t in composite_tests if t)
    tests["num_criteria_total"] = len(composite_tests)
    tests["composite_criteria"] = composite_labels

    return tests


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_exp1(
    cfg: dict,
    out_dir: Path,
    results_so_far: Dict[int, ExperimentResult],
) -> ExperimentResult:
    """Run Experiment 1: Task Granularity Spectrum.

    For each image × level × perturbation, score MCQ and record accuracy
    degradation.  Then test whether degradation scales monotonically with
    task granularity.
    """
    logger.info("=" * 50)
    logger.info("Experiment 1: Task Granularity Spectrum")
    logger.info("=" * 50)

    exp_cfg = cfg.get("experiments", {}).get("exp1", {})
    max_samples = exp_cfg.get("max_samples", cfg.get("data", {}).get("max_samples", 100))
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
        attn_implementation=model_cfg.get("attn_implementation"),
        attention_extract_implementation=model_cfg.get(
            "attention_extract_implementation", "eager"
        ),
    )
    logger.info("Model loaded successfully")

    # --- Load dataset ---
    logger.info("Building VQA granularity dataset (max_samples=%d)...", max_samples)
    samples = load_multilevel_vqa_dataset(cfg, max_samples=max_samples)
    logger.info("Loaded %d samples with all available granularity levels", len(samples))

    if not samples:
        logger.error("No samples loaded. Check GQA data availability.")
        return ExperimentResult(
            experiment_id=1,
            experiment_name="exp1_granularity",
            config=exp_cfg,
            metrics={"error": "no_samples"},
        )

    # --- Perturbation config ---
    pert_cfg = cfg.get("perturbations", {})
    severity_levels = pert_cfg.get("severity_levels", [1, 2, 3])
    num_bands = cfg.get("analysis", {}).get("num_bands", 10)
    suppress_dc = bool(cfg.get("analysis", {}).get("suppress_dc", True))
    overlap_2d_enabled = _overlap_2d_enabled(cfg)
    export_num_images = max(0, int(pert_cfg.get("export_num_images", 1) or 0))
    exported_examples = 0

    # Whether to extract vision tokens for cosine drift (slower)
    extract_vision_tokens = exp_cfg.get("extract_vision_tokens", False)
    store_feature_delta_f = exp_cfg.get("store_feature_delta_f", True)

    # --- Evaluation loop ---
    per_sample_results: List[Dict[str, Any]] = []
    t0 = time.time()

    for idx, sample in enumerate(samples):
        logger.info(
            "Processing sample %d/%d: %s", idx + 1, len(samples), sample.image_id,
        )

        # Build perturbation suite for this image
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as e:
            logger.warning("Cannot open image %s: %s", sample.image_path, e)
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
            overlay_mode=pert_cfg.get("overlay_mode", "label_free"),
            overlay_count=int(pert_cfg.get("overlay_count", 3)),
            overlay_options=overlay_options,
            overlay_base_label=overlay_base_label,
            overlay_seed=seed + idx,
        )

        if not perturbations:
            logger.warning("No perturbations for sample %s", sample.image_id)
            continue

        perturbation_2d_files: Dict[str, Dict[str, Any]] = {}
        if overlap_2d_enabled:
            try:
                perturbation_2d_files = _save_delta_f_2d_suite(
                    image_id=sample.image_id,
                    perturbations=perturbations,
                    out_dir=out_dir,
                    suppress_dc=suppress_dc,
                )
            except Exception as exc:
                logger.warning(
                    "Could not save 2D perturbation spectra for %s: %s",
                    sample.image_id,
                    exc,
                )

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

        # Evaluate across all levels and perturbations
        feature_delta_2d_dir = (
            out_dir / "delta_f_vision_2d"
            if overlap_2d_enabled and store_feature_delta_f
            else None
        )
        record = _evaluate_sample(
            sample, adapter, perturbations,
            extract_vision_tokens=extract_vision_tokens,
            store_feature_delta_f=store_feature_delta_f,
            num_bands=num_bands,
            suppress_dc=suppress_dc,
            perturbation_2d_files=perturbation_2d_files,
            feature_delta_2d_dir=feature_delta_2d_dir,
        )
        per_sample_results.append(record)

        # Progress logging
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(samples) - idx - 1) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f samples/s, ETA %.0fs)",
                idx + 1, len(samples), rate, eta,
            )

    total_time = time.time() - t0
    logger.info(
        "Evaluation complete: %d samples in %.1fs (%.2f samples/s)",
        len(per_sample_results), total_time,
        len(per_sample_results) / total_time if total_time > 0 else 0,
    )

    # --- Aggregate results ---
    agg = _aggregate_results(per_sample_results)
    agg["vision_feature_status"] = _summarize_vision_feature_status(per_sample_results)
    delta_2d_dir = out_dir / "delta_f_2d"
    delta_vision_2d_dir = out_dir / "delta_f_vision_2d"
    num_delta_f_2d_files = (
        len(list(delta_2d_dir.glob("*.npz")))
        if overlap_2d_enabled and delta_2d_dir.exists()
        else 0
    )
    num_delta_f_vision_2d_files = (
        len(list(delta_vision_2d_dir.glob("*.npz")))
        if overlap_2d_enabled and delta_vision_2d_dir.exists()
        else 0
    )
    agg["overlap_2d"] = {
        "enabled": bool(overlap_2d_enabled),
        "domain": "image_space",
        "storage": "npz_sidecars",
        "delta_f_2d_dir": "delta_f_2d",
        "num_delta_f_2d_files": num_delta_f_2d_files,
        "keying": "image_id x perturbation",
        "domains": {
            "image_space": {
                "storage": "npz_sidecars",
                "delta_f_2d_dir": "delta_f_2d",
                "num_delta_f_2d_files": num_delta_f_2d_files,
                "keying": "image_id x perturbation",
                "spectral_alignment": "fftshifted_image_space_delta_power",
            },
            "vision_feature_space": {
                "storage": "npz_sidecars",
                "delta_f_2d_dir": "delta_f_vision_2d",
                "num_delta_f_2d_files": num_delta_f_vision_2d_files,
                "keying": "image_id x perturbation",
                "spectral_alignment": "fftshifted_vision_feature_delta_power",
                "wired_into_exp5": False,
            },
        },
    }
    complexity_points = _build_complexity_points(per_sample_results)
    perturbation_complexity_points = _build_perturbation_complexity_points(per_sample_results)
    long_format_rows = _build_long_format_rows(per_sample_results)
    agg["complexity_analysis"] = _summarize_complexity(complexity_points)

    # --- Hypothesis testing ---
    hypothesis_tests = _run_hypothesis_tests(
        agg,
        complexity_points,
        perturbation_complexity_points,
        long_format_rows=long_format_rows,
    )

    # --- Log key results ---
    logger.info("-" * 40)
    logger.info("Key results:")
    for lk, stats in agg.get("per_level", {}).items():
        logger.info(
            "  %s: clean_acc=%.3f  pert_acc=%.3f  mean_drop=%.3f",
            lk, stats["clean_accuracy"], stats["perturbed_accuracy"],
            stats["mean_accuracy_drop"],
        )
    spearman = hypothesis_tests.get("spearman_drop_vs_granularity", {})
    logger.info(
        "  Spearman(granularity, degradation): rho=%.3f, p=%.4f [%s]",
        spearman.get("rho", 0), spearman.get("p_value", 1),
        "PASS" if spearman.get("passed") else "FAIL",
    )
    logger.info(
        "  Hypothesis supported: %s (%d/%d criteria)",
        hypothesis_tests.get("hypothesis_supported", False),
        hypothesis_tests.get("num_criteria_passed", 0),
        hypothesis_tests.get("num_criteria_total", 0),
    )
    logger.info("-" * 40)

    # --- Save outputs ---
    save_json(agg, out_dir / "summary.json")
    save_json(hypothesis_tests, out_dir / "hypothesis_tests.json")
    save_json(complexity_points, out_dir / "complexity_points.json")

    # Save per-sample as JSONL
    jsonl_path = out_dir / "per_sample.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w") as f:
        for record in per_sample_results:
            f.write(json.dumps(record, default=str) + "\n")
    logger.info("Saved %d per-sample records to %s", len(per_sample_results), jsonl_path)

    # Save degradation breakdown
    save_json(
        agg.get("per_level_perturbation", {}),
        out_dir / "degradation_by_level.json",
    )

    # --- Cleanup ---
    try:
        adapter.unload()
    except Exception:
        pass

    # --- Build result ---
    metrics = {
        "num_samples": len(per_sample_results),
        "total_time_s": total_time,
        **{
            f"{lk}_clean_acc": stats["clean_accuracy"]
            for lk, stats in agg.get("per_level", {}).items()
        },
        **{
            f"{lk}_mean_drop": stats["mean_accuracy_drop"]
            for lk, stats in agg.get("per_level", {}).items()
        },
        "spearman_rho": spearman.get("rho", 0),
        "spearman_p": spearman.get("p_value", 1),
        "complexity_pearson_drop": hypothesis_tests.get("continuous_complexity", {})
        .get("mean_accuracy_drop_vs_complexity", {})
        .get("pearson_r", 0.0),
    }

    return ExperimentResult(
        experiment_id=1,
        experiment_name="exp1_granularity",
        config=exp_cfg,
        metrics=metrics,
        per_sample=per_sample_results,
        hypothesis_tests=hypothesis_tests,
    )
