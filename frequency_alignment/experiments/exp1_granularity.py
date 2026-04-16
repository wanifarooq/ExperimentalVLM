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
    summarize_by_score,
    summarize_fixed_effects_trend,
    summarize_linear_trend,
    summarize_multivariate_regression,
)
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


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if np.isfinite(value_float) else None


def _horse_race_predictors(points: List[Dict[str, Any]]) -> List[str]:
    """Use entropy control when available, while preserving old-result compatibility."""

    predictors = [
        "complexity_score_residual",
        "prompt_complexity_score",
        "option_hardness_score",
    ]
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
            ),
            "within_image": summarize_multivariate_regression(
                used_points,
                y_key=y_key,
                x_keys=predictors,
                group_key="image_id",
                demean_by_group=True,
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
            view_payload[view_name] = {
                "filter": {
                    **filter_payload,
                    "level_view": view_name,
                    "level_filter": sorted(level_filter) if level_filter is not None else None,
                    "n_points_used": len(view_points),
                },
                "pooled": summarize_multivariate_regression(
                    view_points,
                    y_key=y_key,
                    x_keys=view_predictors,
                    level_filter=level_filter,
                ),
                "within_image": summarize_multivariate_regression(
                    view_points,
                    y_key=y_key,
                    x_keys=view_predictors,
                    group_key="image_id",
                    demean_by_group=True,
                    level_filter=level_filter,
                ),
            }
            by_perturbation[perturbation_name][f"pooled_{view_name}"] = view_payload[view_name]["pooled"]
            by_perturbation[perturbation_name][f"within_image_{view_name}"] = view_payload[view_name]["within_image"]
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
        pooled = summarize_multivariate_regression(
            view_points,
            y_key=y_key,
            x_keys=predictors,
            level_filter=level_filter,
        )
        within = summarize_multivariate_regression(
            view_points,
            y_key=y_key,
            x_keys=predictors,
            group_key="image_id",
            demean_by_group=True,
            level_filter=level_filter,
        )
        view_payload[view_name] = {
            "level_filter": sorted(level_filter) if level_filter is not None else None,
            "n_points": len(view_points),
            "pooled": pooled,
            "within_image": within,
        }
        payload[f"{base_key}_{view_name}"] = pooled
        payload[f"{base_key}_within_image_{view_name}"] = within

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
    if need_vision_features:
        try:
            vt = adapter.get_vision_tokens(image)
            if vt is not None:
                clean_vision_tokens = vt.cpu().numpy()
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
        except Exception:
            pass
    if clean_vision_tokens is not None:
        for perturbation in perturbation_results:
            try:
                pert_tokens = adapter.get_vision_tokens(perturbation.perturbed_image)
            except Exception:
                continue
            if pert_tokens is None:
                continue
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
                    delta_f_vision = feature_stats["delta_f"]
                    metrics["delta_f_vision"] = delta_f_vision.tolist()
                    metrics["delta_f_vision_relative"] = feature_stats["delta_f_relative"].tolist()
                    metrics["delta_f_vision_norm"] = float(np.linalg.norm(delta_f_vision))
                    metrics["delta_f_vision_peak_band"] = int(np.argmax(delta_f_vision))
                    metrics["clean_feature_spectral_energy"] = float(
                        feature_stats["clean_total_energy"]
                    )
            perturbation_vision_metrics[perturbation.name] = metrics

    for level in GranularityLevel:
        level_data = sample.levels.get(level)
        if level_data is None:
            continue

        level_key = level.name  # e.g. "L1_COARSE"
        complexity = ensure_level_complexity(level_data)
        level_record: Dict[str, Any] = {
            "question": level_data.question,
            "question_type": level_data.question_type,
            "answer_label": level_data.answer_label,
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

    rho_drop, p_drop = spearman_correlation(granularity_ranks, mean_drops)
    tests["spearman_drop_vs_granularity"] = {
        "rho": rho_drop,
        "p_value": p_drop,
        "target": "rho > 0.8",
        "passed": rho_drop > 0.8 and p_drop < 0.05,
    }

    rho_drift, p_drift = spearman_correlation(granularity_ranks, mean_drifts)
    tests["spearman_loglik_drift_vs_granularity"] = {
        "rho": rho_drift,
        "p_value": p_drift,
        "passed": rho_drift > 0.6,
    }

    # H2: Monotonicity of mean accuracy drop across levels
    is_mono, tau = monotonicity_test(mean_drops)
    tests["monotonicity_accuracy_drop"] = {
        "is_monotonic": is_mono,
        "kendall_tau": tau,
        "values": dict(zip(present_levels, mean_drops)),
        "passed": tau > 0.6,
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
            "target": "secondary wordy-ladder monotonicity only; no pooled monotonicity is run",
            "passed": wordy_rho > 0.6,
            "values": dict(zip(wordy_present_levels, wordy_drops)),
        }
        tests["monotonicity_accuracy_drop_wordy"] = {
            "is_monotonic": wordy_mono,
            "kendall_tau": wordy_tau,
            "values": dict(zip(wordy_present_levels, wordy_drops)),
            "passed": wordy_tau > 0.6,
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

    # Summary verdict
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
            ),
            "horse_race_mean_accuracy_drop_within_image": summarize_multivariate_regression(
                accuracy_drop_points,
                y_key="mean_accuracy_drop",
                x_keys=accuracy_predictors,
                group_key="image_id",
                demean_by_group=True,
            ),
            "horse_race_mean_loglik_drift": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_drift",
                x_keys=all_predictors,
            ),
            "horse_race_mean_loglik_drift_within_image": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_drift",
                x_keys=all_predictors,
                group_key="image_id",
                demean_by_group=True,
            ),
            "horse_race_mean_loglik_erosion": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_erosion",
                x_keys=all_predictors,
            ),
            "horse_race_mean_loglik_erosion_within_image": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_erosion",
                x_keys=all_predictors,
                group_key="image_id",
                demean_by_group=True,
            ),
            "horse_race_mean_loglik_recovery": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_recovery",
                x_keys=all_predictors,
            ),
            "horse_race_mean_loglik_recovery_within_image": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_recovery",
                x_keys=all_predictors,
                group_key="image_id",
                demean_by_group=True,
            ),
            "horse_race_mean_loglik_volatility": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_volatility",
                x_keys=all_predictors,
            ),
            "horse_race_mean_loglik_volatility_within_image": summarize_multivariate_regression(
                complexity_points,
                y_key="mean_loglik_volatility",
                x_keys=all_predictors,
                group_key="image_id",
                demean_by_group=True,
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
        record = _evaluate_sample(
            sample, adapter, perturbations,
            extract_vision_tokens=extract_vision_tokens,
            store_feature_delta_f=store_feature_delta_f,
            num_bands=num_bands,
            suppress_dc=suppress_dc,
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
    complexity_points = _build_complexity_points(per_sample_results)
    perturbation_complexity_points = _build_perturbation_complexity_points(per_sample_results)
    agg["complexity_analysis"] = _summarize_complexity(complexity_points)

    # --- Hypothesis testing ---
    hypothesis_tests = _run_hypothesis_tests(agg, complexity_points, perturbation_complexity_points)

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
