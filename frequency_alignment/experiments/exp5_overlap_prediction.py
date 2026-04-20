"""Experiment 5: Spectral Overlap Prediction."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..analysis.continuous import (
    summarize_fixed_effects_trend,
    summarize_horse_race_view,
    summarize_linear_trend,
    summarize_multivariate_regression,
)
from ..analysis.gate_thresholds import GATES
from ..analysis.level_views import LEVEL_VIEW_ORDER, LEVEL_VIEWS
from ..analysis.spectral import compute_spectral_overlap, compute_spectral_overlap_linear
from ..analysis.statistics import bootstrap_ci, pearson_correlation, spearman_correlation
from ..data.base import ALL_VQA_LEVEL_NAMES, ExperimentResult
from ..utils.exp2_filters import load_exp2_filter_bank
from ..utils.io import save_json
from ..utils.layer_groups import LAYER_GROUP_ORDER

logger = logging.getLogger(__name__)

FILTER_ANALYSIS_ORDER = ("overall",) + LAYER_GROUP_ORDER
PRIMARY_TARGET = "loglik_volatility"
TARGET_SPECS: Dict[str, Dict[str, Any]] = {
    "accuracy_drop": {
        "grouped_key": "accuracy_drop",
        "sample_key": "accuracy_drop",
        "label": "Accuracy Drop",
        "supports_sample": True,
    },
    "loglik_erosion": {
        "grouped_key": "loglik_erosion",
        "sample_key": "loglik_erosion",
        "label": "Correct-Answer Log-Likelihood Erosion",
        "supports_sample": True,
    },
    "loglik_volatility": {
        "grouped_key": "loglik_volatility",
        "sample_key": "loglik_volatility",
        "label": "Correct-Answer Log-Likelihood Volatility",
        "supports_sample": True,
    },
    "net_drop": {
        "grouped_key": "net_drop",
        "sample_key": "net_change",
        "label": "Net Accuracy Drop (CI - IC)",
        "supports_sample": True,
    },
    "relative_accuracy_drop": {
        "grouped_key": "relative_accuracy_drop",
        "sample_key": None,
        "label": "Relative Accuracy Drop",
        "supports_sample": False,
    },
}
TARGET_ORDER = tuple(TARGET_SPECS.keys())
_EPS = 1e-8


def _build_source_specs(primary_normalization: str) -> List[Dict[str, str]]:
    primary_normalization = (primary_normalization or "relative").lower()
    if primary_normalization not in {"raw", "relative"}:
        primary_normalization = "relative"

    specs: List[Dict[str, str]] = []
    for domain in ("image_space", "vision_feature_space"):
        if domain == "image_space":
            raw_key = "delta_f"
            rel_key = "delta_f_relative"
        else:
            raw_key = "delta_f_vision"
            rel_key = "delta_f_vision_relative"

        primary_key = rel_key if primary_normalization == "relative" else raw_key
        specs.append(
            {
                "name": domain,
                "domain": domain,
                "normalization": primary_normalization,
                "delta_key": primary_key,
            }
        )
        for normalization, delta_key in (("raw", raw_key), ("relative", rel_key)):
            if normalization == primary_normalization:
                continue
            specs.append(
                {
                    "name": f"{domain}_{normalization}",
                    "domain": domain,
                    "normalization": normalization,
                    "delta_key": delta_key,
                }
            )
    return specs


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_overlap_pairs(
    exp1_out_dir: Path,
    average_filters: Dict[str, np.ndarray],
    sample_filters: Dict[Tuple[str, str], np.ndarray],
    delta_key: str = "delta_f",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    exp1_records = _load_jsonl(exp1_out_dir / "per_sample.jsonl")

    sample_pairs: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "predicted": [],
            "predicted_quadratic": [],
            "predicted_linear": [],
            "complexity_score": [],
            "question_complexity_score": [],
            "prompt_complexity_score": [],
            "option_hardness_score": [],
            "accuracy_drop": [],
            "loglik_drift": [],
            "loglik_erosion": [],
            "loglik_recovery": [],
            "loglik_volatility": [],
            "net_change": [],
            "clean_correct": [],
            "perturbed_correct": [],
            "ci": [],
            "ic": [],
        }
    )

    for record in exp1_records:
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            W_t = sample_filters.get((image_id, level_key), average_filters.get(level_key))
            if W_t is None:
                continue
            complexity_score = float(level_data.get("complexity_score", 0.0) or 0.0)
            question_complexity_score = float(
                level_data.get("question_complexity_score", complexity_score) or 0.0
            )
            prompt_complexity_score = float(level_data.get("prompt_complexity_score", 0.0) or 0.0)
            option_hardness_score = float(level_data.get("option_hardness_score", 0.0) or 0.0)

            clean_correct = float(bool(level_data.get("clean", {}).get("correct", False)))
            for perturbation in level_data.get("perturbations", []):
                delta_f = np.asarray(perturbation.get(delta_key) or [], dtype=np.float64)
                if delta_f.size == 0:
                    continue

                perturbed_correct = float(bool(perturbation.get("correct", False)))
                ci = 1.0 if clean_correct > 0.5 and perturbed_correct < 0.5 else 0.0
                ic = 1.0 if clean_correct < 0.5 and perturbed_correct > 0.5 else 0.0
                net_change = ci - ic
                accuracy_drop = float(perturbation.get("accuracy_drop", ci))
                loglik_drift = float(perturbation.get("loglik_drift", 0.0))
                loglik_erosion = max(loglik_drift, 0.0)
                loglik_recovery = min(loglik_drift, 0.0)
                loglik_volatility = abs(loglik_drift)
                predicted_quadratic = compute_spectral_overlap(W_t, delta_f)
                predicted_linear = compute_spectral_overlap_linear(W_t, delta_f)

                pair = {
                    "image_id": image_id,
                    "level": level_key,
                    "perturbation": perturbation.get("name", "unknown"),
                    "predicted": predicted_quadratic,
                    "predicted_quadratic": predicted_quadratic,
                    "predicted_linear": predicted_linear,
                    "actual": accuracy_drop,
                    "accuracy_drop": accuracy_drop,
                    "loglik_erosion": loglik_erosion,
                    "loglik_drift": loglik_drift,
                    "loglik_recovery": loglik_recovery,
                    "loglik_volatility": loglik_volatility,
                    "net_change": net_change,
                    "clean_correct": clean_correct,
                    "perturbed_correct": perturbed_correct,
                    "ci": ci,
                    "ic": ic,
                    "complexity_score": complexity_score,
                    "question_complexity_score": question_complexity_score,
                    "prompt_complexity_score": prompt_complexity_score,
                    "option_hardness_score": option_hardness_score,
                }
                sample_pairs.append(pair)

                key = (level_key, pair["perturbation"])
                grouped[key]["predicted"].append(predicted_quadratic)
                grouped[key]["predicted_quadratic"].append(predicted_quadratic)
                grouped[key]["predicted_linear"].append(predicted_linear)
                grouped[key]["complexity_score"].append(complexity_score)
                grouped[key]["question_complexity_score"].append(question_complexity_score)
                grouped[key]["prompt_complexity_score"].append(prompt_complexity_score)
                grouped[key]["option_hardness_score"].append(option_hardness_score)
                grouped[key]["accuracy_drop"].append(accuracy_drop)
                grouped[key]["loglik_drift"].append(loglik_drift)
                grouped[key]["loglik_erosion"].append(loglik_erosion)
                grouped[key]["loglik_recovery"].append(loglik_recovery)
                grouped[key]["loglik_volatility"].append(loglik_volatility)
                grouped[key]["net_change"].append(net_change)
                grouped[key]["clean_correct"].append(clean_correct)
                grouped[key]["perturbed_correct"].append(perturbed_correct)
                grouped[key]["ci"].append(ci)
                grouped[key]["ic"].append(ic)

    grouped_pairs: List[Dict[str, Any]] = []
    for (level_key, perturbation), values in sorted(grouped.items()):
        clean_accuracy = float(np.mean(values["clean_correct"]))
        perturbed_accuracy = float(np.mean(values["perturbed_correct"]))
        net_drop = float(np.mean(values["net_change"]))
        relative_accuracy_drop = None
        if clean_accuracy > _EPS:
            relative_accuracy_drop = float(net_drop / clean_accuracy)

        grouped_pairs.append(
            {
                "label": f"{level_key}|{perturbation}",
                "level": level_key,
                "perturbation": perturbation,
                "predicted": float(np.mean(values["predicted"])),
                "predicted_quadratic": float(np.mean(values["predicted_quadratic"])),
                "predicted_linear": float(np.mean(values["predicted_linear"])),
                "complexity_score": float(np.mean(values["complexity_score"])),
                "question_complexity_score": float(np.mean(values["question_complexity_score"])),
                "prompt_complexity_score": float(np.mean(values["prompt_complexity_score"])),
                "option_hardness_score": float(np.mean(values["option_hardness_score"])),
                "actual": float(np.mean(values["accuracy_drop"])),
                "accuracy_drop": float(np.mean(values["accuracy_drop"])),
                "loglik_erosion": float(np.mean(values["loglik_erosion"])),
                "loglik_drift": float(np.mean(values["loglik_drift"])),
                "mean_loglik_drift": float(np.mean(values["loglik_drift"])),
                "loglik_recovery": float(np.mean(values["loglik_recovery"])),
                "mean_loglik_recovery": float(np.mean(values["loglik_recovery"])),
                "loglik_volatility": float(np.mean(values["loglik_volatility"])),
                "mean_loglik_volatility": float(np.mean(values["loglik_volatility"])),
                "net_drop": net_drop,
                "clean_accuracy": clean_accuracy,
                "perturbed_accuracy": perturbed_accuracy,
                "relative_accuracy_drop": relative_accuracy_drop,
                "ci_count": int(round(sum(values["ci"]))),
                "ic_count": int(round(sum(values["ic"]))),
                "n": len(values["accuracy_drop"]),
            }
        )

    return sample_pairs, grouped_pairs


def _materialize_target_pairs(
    pairs: List[Dict[str, Any]],
    actual_key: Optional[str],
) -> List[Dict[str, Any]]:
    if actual_key is None:
        return []

    target_pairs: List[Dict[str, Any]] = []
    for pair in pairs:
        actual = pair.get(actual_key)
        if actual is None:
            continue
        actual_float = float(actual)
        if not np.isfinite(actual_float):
            continue
        target_pair = dict(pair)
        target_pair["actual"] = actual_float
        target_pair["actual_key"] = actual_key
        target_pairs.append(target_pair)
    return target_pairs


def _correlate(
    pairs: List[Dict[str, Any]],
    actual_key: str = "actual",
) -> Tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray([pair["predicted"] for pair in pairs], dtype=np.float64)
    actual = np.asarray([pair[actual_key] for pair in pairs], dtype=np.float64)
    return predicted, actual


def _overlap_variant_correlations(
    grouped_pairs: List[Dict[str, Any]],
    sample_pairs: List[Dict[str, Any]],
    *,
    actual_key: str = "actual",
) -> Dict[str, Dict[str, Any]]:
    variants = {
        "quadratic": "predicted_quadratic",
        "linear": "predicted_linear",
    }
    payload: Dict[str, Dict[str, Any]] = {}
    for variant_name, predicted_key in variants.items():
        variant_payload: Dict[str, Any] = {
            "predicted_key": predicted_key,
            "grouped": {
                "aggregation_level": "grouped_perturbation_family",
                "n": 0,
                "pearson_r": None,
                "pearson_p_value": None,
                "spearman_rho": None,
                "spearman_p_value": None,
            },
            "sample": {
                "aggregation_level": "sample",
                "n": 0,
                "pearson_r": None,
                "pearson_p_value": None,
                "spearman_rho": None,
                "spearman_p_value": None,
            },
        }
        for level_name, rows in (
            ("grouped", grouped_pairs),
            ("sample", sample_pairs),
        ):
            usable = [
                row
                for row in rows
                if row.get(predicted_key) is not None and row.get(actual_key) is not None
            ]
            variant_payload[level_name]["n"] = len(usable)
            if len(usable) < 3:
                continue
            pred = np.asarray([row[predicted_key] for row in usable], dtype=np.float64)
            actual = np.asarray([row[actual_key] for row in usable], dtype=np.float64)
            r, p = pearson_correlation(pred, actual)
            rho, rho_p = spearman_correlation(pred, actual)
            variant_payload[level_name].update(
                {
                    "pearson_r": r,
                    "pearson_p_value": p,
                    "spearman_rho": rho,
                    "spearman_p_value": rho_p,
                }
            )
        payload[variant_name] = variant_payload
    return payload


def _filter_pairs_by_level_view(
    pairs: List[Dict[str, Any]],
    view_name: str,
) -> List[Dict[str, Any]]:
    level_filter = LEVEL_VIEWS[view_name]
    if level_filter is None:
        return list(pairs)
    return [pair for pair in pairs if str(pair.get("level")) in level_filter]


def _grouped_view_correlation_summary(
    pairs: List[Dict[str, Any]],
    view_name: str,
) -> Dict[str, Any]:
    view_pairs = _filter_pairs_by_level_view(pairs, view_name)
    if len(view_pairs) < 3:
        return {
            "view": view_name,
            "level_filter": sorted(LEVEL_VIEWS[view_name]) if LEVEL_VIEWS[view_name] is not None else None,
            "n_grouped_pairs": len(view_pairs),
            "pearson_r_grouped": 0.0,
            "pearson_p_grouped": 1.0,
            "spearman_rho_grouped": 0.0,
            "spearman_p_grouped": 1.0,
            "passed": False,
            "target": f"r > {GATES['overlap_law_r_min']} with n >= {GATES['overlap_law_min_n']}",
        }
    pred_arr, actual_arr = _correlate(view_pairs)
    r_pearson, p_pearson = pearson_correlation(pred_arr, actual_arr)
    rho_spearman, p_spearman = spearman_correlation(pred_arr, actual_arr)
    return {
        "view": view_name,
        "level_filter": sorted(LEVEL_VIEWS[view_name]) if LEVEL_VIEWS[view_name] is not None else None,
        "n_grouped_pairs": len(view_pairs),
        "pearson_r_grouped": r_pearson,
        "pearson_p_grouped": p_pearson,
        "spearman_rho_grouped": rho_spearman,
        "spearman_p_grouped": p_spearman,
        "target": f"r > {GATES['overlap_law_r_min']} with n >= {GATES['overlap_law_min_n']}",
        "passed": r_pearson > GATES["overlap_law_r_min"] and len(view_pairs) >= GATES["overlap_law_min_n"],
    }


def _zscore_array(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr, 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= _EPS:
        return arr - mean, mean, std
    return (arr - mean) / std, mean, std


def _attach_standardized_bridge(
    pairs: List[Dict[str, Any]],
    *,
    actual_key: str = "actual",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not pairs:
        return pairs, {
            "n": 0,
            "x_transform": "zscore(log1p(predicted))",
            "y_transform": f"zscore({actual_key})",
        }

    pred_arr, actual_arr = _correlate(pairs, actual_key=actual_key)
    pred_log = np.log1p(np.clip(pred_arr, 0.0, None))
    pred_z, pred_log_mean, pred_log_std = _zscore_array(pred_log)
    actual_z, actual_mean, actual_std = _zscore_array(actual_arr)

    if len(pairs) >= 2 and np.unique(pred_z).size > 1:
        slope, intercept = np.polyfit(pred_z, actual_z, 1)
        pearson_r, pearson_p = pearson_correlation(pred_z, actual_z)
        spearman_rho, spearman_p = spearman_correlation(pred_z, actual_z)
    else:
        slope = 0.0
        intercept = float(np.mean(actual_z)) if len(actual_z) else 0.0
        pearson_r, pearson_p = 0.0, 1.0
        spearman_rho, spearman_p = 0.0, 1.0

    annotated: List[Dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        pair_copy = dict(pair)
        pair_copy["predicted_log1p"] = float(pred_log[idx])
        pair_copy["predicted_bridge_z"] = float(pred_z[idx])
        pair_copy["actual_bridge_z"] = float(actual_z[idx])
        annotated.append(pair_copy)

    summary = {
        "n": len(pairs),
        "x_transform": "zscore(log1p(predicted))",
        "y_transform": f"zscore({actual_key})",
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
        "slope": float(slope),
        "intercept": float(intercept),
        "predicted_log_mean": float(pred_log_mean),
        "predicted_log_std": float(pred_log_std),
        "actual_mean": float(actual_mean),
        "actual_std": float(actual_std),
    }
    return annotated, summary


def _attach_prediction_error(
    pairs: List[Dict[str, Any]],
    *,
    actual_key: str = "actual",
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if not pairs:
        return pairs, {"slope": 0.0, "intercept": 0.0}

    pred_arr, actual_arr = _correlate(pairs, actual_key=actual_key)
    if len(pairs) >= 2 and np.unique(pred_arr).size > 1:
        slope, intercept = np.polyfit(pred_arr, actual_arr, 1)
    else:
        slope = 0.0
        intercept = float(np.mean(actual_arr)) if len(actual_arr) else 0.0

    annotated: List[Dict[str, Any]] = []
    for pair in pairs:
        pair_copy = dict(pair)
        calibrated_prediction = float(intercept + slope * float(pair["predicted"]))
        prediction_residual = float(pair_copy[actual_key] - calibrated_prediction)
        pair_copy["calibrated_prediction"] = calibrated_prediction
        pair_copy["prediction_residual"] = prediction_residual
        pair_copy["absolute_prediction_error"] = abs(prediction_residual)
        annotated.append(pair_copy)
    return annotated, {"slope": float(slope), "intercept": float(intercept)}


def _summarize_prediction_factor_horse_race(
    pairs: List[Dict[str, Any]],
    *,
    y_key: str = "accuracy_drop",
) -> Dict[str, Any]:
    """Horse race: spectral overlap vs language/MCQ controls."""

    rows: List[Dict[str, Any]] = []
    raw_predicted: List[float] = []
    candidates: List[Dict[str, Any]] = []
    for pair in pairs:
        y_val = pair.get(y_key)
        predicted = pair.get("predicted")
        question_complexity = pair.get("question_complexity_score", pair.get("complexity_score"))
        prompt = pair.get("prompt_complexity_score")
        hardness = pair.get("option_hardness_score")
        if (
            y_val is None
            or predicted is None
            or question_complexity is None
            or prompt is None
            or hardness is None
        ):
            continue
        try:
            y_float = float(y_val)
            pred_float = float(predicted)
            question_complexity_float = float(question_complexity)
            prompt_float = float(prompt)
            hardness_float = float(hardness)
        except (TypeError, ValueError):
            continue
        if not all(
            np.isfinite(v)
            for v in (y_float, pred_float, question_complexity_float, prompt_float, hardness_float)
        ):
            continue
        candidates.append(
            {
                "image_id": pair.get("image_id"),
                "level": pair.get("level"),
                y_key: y_float,
                "predicted_overlap_log1p_z": pred_float,
                "question_complexity_score": question_complexity_float,
                "prompt_complexity_score": prompt_float,
                "option_hardness_score": hardness_float,
            }
        )
        raw_predicted.append(pred_float)

    if candidates:
        pred_log = np.log1p(np.clip(np.asarray(raw_predicted, dtype=np.float64), 0.0, None))
        pred_z, pred_mean, pred_std = _zscore_array(pred_log)
        for row, z_value in zip(candidates, pred_z):
            row["predicted_overlap_log1p_z"] = float(z_value)
            rows.append(row)
    else:
        pred_mean = pred_std = 0.0

    regression = summarize_multivariate_regression(
        rows,
        y_key=y_key,
        x_keys=[
            "predicted_overlap_log1p_z",
            "question_complexity_score",
            "prompt_complexity_score",
            "option_hardness_score",
        ],
    )
    view_payload: Dict[str, Any] = {}
    predictors = [
        "predicted_overlap_log1p_z",
        "question_complexity_score",
        "prompt_complexity_score",
        "option_hardness_score",
    ]
    for view_name in LEVEL_VIEW_ORDER:
        level_filter = LEVEL_VIEWS[view_name]
        view_payload[view_name] = summarize_horse_race_view(
            rows,
            y_key=y_key,
            x_keys=predictors,
            view_name=view_name,
            level_filter=level_filter,
        )
    return {
        "description": "Multivariate horse race for observed target: spectral overlap vs raw semantic complexity, prompt load, and option hardness.",
        "outcome": y_key,
        "predictors": predictors,
        "prediction_transform": "zscore(log1p(predicted_overlap))",
        "n_candidate_rows": len(rows),
        "predicted_log1p_mean": float(pred_mean),
        "predicted_log1p_std": float(pred_std),
        "regression": regression,
        "views": view_payload,
    }


def _summarize_target(
    sample_pairs: List[Dict[str, Any]],
    grouped_pairs: List[Dict[str, Any]],
    target_name: str,
    target_spec: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    grouped_target_pairs = _materialize_target_pairs(grouped_pairs, target_spec.get("grouped_key"))
    if len(grouped_target_pairs) < 3:
        return None
    grouped_target_pairs, grouped_bridge = _attach_standardized_bridge(grouped_target_pairs)
    grouped_target_pairs, grouped_error_model = _attach_prediction_error(grouped_target_pairs)

    pred_arr, actual_arr = _correlate(grouped_target_pairs)
    r_pearson, p_pearson = pearson_correlation(pred_arr, actual_arr)
    rho_spearman, p_spearman = spearman_correlation(pred_arr, actual_arr)
    grouped_view_summaries = {
        view_name: _grouped_view_correlation_summary(grouped_target_pairs, view_name)
        for view_name in LEVEL_VIEW_ORDER
    }

    sample_target_pairs: List[Dict[str, Any]] = []
    sample_error_model = None
    sample_r = None
    sample_r_p = None
    sample_bridge = None
    supports_sample = bool(target_spec.get("supports_sample")) and target_spec.get("sample_key") is not None
    if supports_sample:
        sample_target_pairs = _materialize_target_pairs(sample_pairs, target_spec["sample_key"])
        if len(sample_target_pairs) >= 3:
            sample_target_pairs, sample_bridge = _attach_standardized_bridge(sample_target_pairs)
            sample_target_pairs, sample_error_model = _attach_prediction_error(sample_target_pairs)
            sample_pred_arr, sample_actual_arr = _correlate(sample_target_pairs)
            sample_r, sample_r_p = pearson_correlation(sample_pred_arr, sample_actual_arr)

    pearson_ci = None
    if len(grouped_target_pairs) >= 10:

        def _pearson_stat(indices):
            idx = indices.astype(int)
            return np.corrcoef(pred_arr[idx], actual_arr[idx])[0, 1]

        point, lo, hi = bootstrap_ci(np.arange(len(grouped_target_pairs)), statistic=_pearson_stat)
        pearson_ci = {"mean": point, "ci_95_lower": lo, "ci_95_upper": hi}

    tests: Dict[str, Any] = {
        "target": target_name,
        "label": target_spec["label"],
        "pearson_predicted_vs_actual_grouped": {
            "r": r_pearson,
            "p_value": p_pearson,
            "target": f"r > {GATES['overlap_law_r_min']} with n >= {GATES['overlap_law_min_n']}",
            "passed": r_pearson > GATES["overlap_law_r_min"]
            and len(grouped_target_pairs) >= GATES["overlap_law_min_n"],
            "min_n": GATES["overlap_law_min_n"],
        },
        "spearman_predicted_vs_actual_grouped": {
            "rho": rho_spearman,
            "p_value": p_spearman,
            "passed": rho_spearman > 0.6,
        },
        "pearson_predicted_vs_actual_grouped_views": grouped_view_summaries,
    }
    for view_name, view_summary in grouped_view_summaries.items():
        tests[f"pearson_predicted_vs_actual_grouped_{view_name}"] = {
            "r": view_summary["pearson_r_grouped"],
            "p_value": view_summary["pearson_p_grouped"],
            "n": view_summary["n_grouped_pairs"],
            "target": f"r > {GATES['overlap_law_r_min']} with n >= {GATES['overlap_law_min_n']}",
            "passed": view_summary["passed"],
            "level_filter": view_summary["level_filter"],
            "min_n": GATES["overlap_law_min_n"],
        }
    if sample_r is not None:
        tests["pearson_predicted_vs_actual_sample"] = {
            "r": sample_r,
            "p_value": sample_r_p,
            "passed": sample_r > 0.3,
        }
    else:
        tests["pearson_predicted_vs_actual_sample"] = {
            "r": None,
            "p_value": None,
            "passed": False,
            "not_applicable": not supports_sample,
        }
    if pearson_ci is not None:
        tests["pearson_grouped_bootstrap_ci"] = pearson_ci
    aggregation_note = (
        "Points are perturbation families averaged within (image_id, level); "
        "sample-level r available separately."
    )
    tests["pearson_predicted_vs_actual_grouped"]["aggregation_level"] = (
        "grouped_perturbation_family"
    )
    tests["pearson_predicted_vs_actual_grouped"]["aggregation_note"] = aggregation_note
    tests["pearson_predicted_vs_actual_sample"]["aggregation_level"] = "sample"
    tests["pearson_predicted_vs_actual_sample"]["aggregation_note"] = aggregation_note
    tests["comparable_bridge_grouped"] = grouped_bridge
    tests["comparable_bridge_sample"] = (
        sample_bridge
        if sample_bridge is not None
        else {
            "n": len(sample_target_pairs),
            "x_transform": "zscore(log1p(predicted))",
            "y_transform": "zscore(actual)",
            "not_applicable": not supports_sample,
        }
    )
    if len(sample_target_pairs) >= 3:
        tests["continuous_complexity"] = {
            "predicted_vs_complexity": summarize_linear_trend(
                sample_target_pairs,
                x_key="complexity_score",
                y_key="predicted",
            ),
            "prediction_error_vs_complexity": summarize_linear_trend(
                sample_target_pairs,
                x_key="complexity_score",
                y_key="absolute_prediction_error",
            ),
            "predicted_vs_complexity_fixed_effects": summarize_fixed_effects_trend(
                sample_target_pairs,
                group_key="image_id",
                x_key="complexity_score",
                y_key="predicted",
            ),
            "prediction_error_vs_complexity_fixed_effects": summarize_fixed_effects_trend(
                sample_target_pairs,
                group_key="image_id",
                x_key="complexity_score",
                y_key="absolute_prediction_error",
            ),
        }
    tests["hypothesis_supported"] = tests["pearson_predicted_vs_actual_grouped"]["passed"]

    per_level_corr: Dict[str, Any] = {}
    present_level_keys = {str(pair["level"]) for pair in grouped_target_pairs if pair.get("level")}
    for level_key in [level for level in ALL_VQA_LEVEL_NAMES if level in present_level_keys]:
        level_pairs = [pair for pair in grouped_target_pairs if pair["level"] == level_key]
        if len(level_pairs) < 3:
            continue
        level_pred, level_actual = _correlate(level_pairs)
        r, p = pearson_correlation(level_pred, level_actual)
        per_level_corr[level_key] = {"pearson_r": r, "p_value": p, "n": len(level_pairs)}

    overlap_law_variants = _overlap_variant_correlations(
        grouped_target_pairs,
        sample_target_pairs,
        actual_key="actual",
    )
    for pair in grouped_target_pairs:
        pair["aggregation_level"] = "grouped_perturbation_family"
        pair["aggregation_note"] = aggregation_note
        pair["pearson_r_grouped_context"] = r_pearson
        pair["n_grouped_context"] = len(grouped_target_pairs)
        pair["pearson_r_sample_context"] = sample_r
        pair["n_sample_context"] = len(sample_target_pairs)
    for pair in sample_target_pairs:
        pair["aggregation_level"] = "sample"
        pair["aggregation_note"] = aggregation_note
        pair["pearson_r_grouped_context"] = r_pearson
        pair["n_grouped_context"] = len(grouped_target_pairs)
        pair["pearson_r_sample_context"] = sample_r
        pair["n_sample_context"] = len(sample_target_pairs)
    summary = {
        "target": target_name,
        "label": target_spec["label"],
        "aggregation_level": "grouped_perturbation_family",
        "aggregation_note": aggregation_note,
        "n_grouped_pairs": len(grouped_target_pairs),
        "n_sample_pairs": len(sample_target_pairs),
        "pearson_r_grouped": r_pearson,
        "spearman_rho_grouped": rho_spearman,
        "pearson_r_sample": sample_r,
        "per_level_correlation": per_level_corr,
        "grouped_views": grouped_view_summaries,
        "grouped_primary": grouped_view_summaries["primary"],
        "grouped_wordy": grouped_view_summaries["wordy"],
        "grouped_pooled": grouped_view_summaries["pooled"],
        "pearson_r_grouped_primary": grouped_view_summaries["primary"]["pearson_r_grouped"],
        "pearson_r_grouped_wordy": grouped_view_summaries["wordy"]["pearson_r_grouped"],
        "pearson_r_grouped_pooled": grouped_view_summaries["pooled"]["pearson_r_grouped"],
        "sample_metric_available": sample_r is not None,
        "comparable_bridge_grouped": grouped_bridge,
        "comparable_bridge_sample": sample_bridge,
        "prediction_error_model_grouped": grouped_error_model,
        "prediction_error_model_sample": sample_error_model,
        "overlap_law_variants": overlap_law_variants,
    }
    tests["overlap_law_variants"] = overlap_law_variants
    metrics = {
        "n_grouped_pairs": len(grouped_target_pairs),
        "n_sample_pairs": len(sample_target_pairs),
        "pearson_r_grouped": r_pearson,
        "pearson_r_grouped_primary": grouped_view_summaries["primary"]["pearson_r_grouped"],
        "pearson_r_grouped_wordy": grouped_view_summaries["wordy"]["pearson_r_grouped"],
        "pearson_r_grouped_pooled": grouped_view_summaries["pooled"]["pearson_r_grouped"],
        "pearson_p_grouped": p_pearson,
        "spearman_rho_grouped": rho_spearman,
        "spearman_p_grouped": p_spearman,
        "pearson_r_sample": sample_r,
        "pearson_p_sample": sample_r_p,
    }
    return summary, tests, metrics, sample_target_pairs, grouped_target_pairs


def run_exp5(
    cfg: dict,
    out_dir: Path,
    results_so_far: Dict[int, ExperimentResult],
) -> ExperimentResult:
    """Run Experiment 5: Spectral Overlap Prediction."""
    logger.info("=" * 50)
    logger.info("Experiment 5: Spectral Overlap Prediction")
    logger.info("=" * 50)

    exp_cfg = cfg.get("experiments", {}).get("exp5", {})
    base_out = Path(cfg.get("out_dir", "frequency_alignment_outputs"))
    exp1_dir = base_out / "exp1"
    exp2_dir = base_out / "exp2"
    primary_group = exp_cfg.get("primary_layer_group", "late")
    primary_normalization = exp_cfg.get("primary_delta_normalization", "relative")
    control_groups = [group_name for group_name in FILTER_ANALYSIS_ORDER if group_name != primary_group]
    source_specs = _build_source_specs(primary_normalization)

    average_filters_by_group, sample_filters_by_group = load_exp2_filter_bank(exp2_dir)
    average_filters_by_group_l2, sample_filters_by_group_l2 = load_exp2_filter_bank(
        exp2_dir,
        norm="l2",
    )
    analyses: Dict[str, Dict[str, Dict[str, Any]]] = {}
    analyses_l2: Dict[str, Dict[str, Dict[str, Any]]] = {}
    source_metadata: Dict[str, Dict[str, str]] = {}
    for source_spec in source_specs:
        source_name = source_spec["name"]
        delta_key = source_spec["delta_key"]
        source_metadata[source_name] = {
            "domain": source_spec["domain"],
            "normalization": source_spec["normalization"],
        }
        source_analyses: Dict[str, Dict[str, Any]] = {}
        for group_name in FILTER_ANALYSIS_ORDER:
            sample_pairs, grouped_pairs = _build_overlap_pairs(
                exp1_dir,
                average_filters_by_group[group_name],
                sample_filters_by_group[group_name],
                delta_key,
            )
            if len(grouped_pairs) < 3:
                logger.warning(
                    "Too few grouped pairs (%d) for correlation in %s/%s",
                    len(grouped_pairs),
                    source_name,
                    group_name,
                )
                continue

            target_analyses: Dict[str, Dict[str, Any]] = {}
            for target_name in TARGET_ORDER:
                summarized = _summarize_target(
                    sample_pairs,
                    grouped_pairs,
                    target_name,
                    TARGET_SPECS[target_name],
                )
                if summarized is None:
                    continue
                summary, tests, metrics, sample_target_pairs, grouped_target_pairs = summarized
                target_analyses[target_name] = {
                    "summary": summary,
                    "tests": tests,
                    "metrics": metrics,
                    "sample_pairs": sample_target_pairs,
                    "grouped_pairs": grouped_target_pairs,
                }
            if not target_analyses:
                continue

            prediction_factor_horse_race = {
                "grouped": _summarize_prediction_factor_horse_race(
                    grouped_pairs,
                    y_key="accuracy_drop",
                ),
                "sample": _summarize_prediction_factor_horse_race(
                    sample_pairs,
                    y_key="accuracy_drop",
                ),
            }

            source_analyses[group_name] = {
                "raw_sample_pairs": sample_pairs,
                "raw_grouped_pairs": grouped_pairs,
                "target_analyses": target_analyses,
                "prediction_factor_horse_race": prediction_factor_horse_race,
            }
        source_analyses_l2: Dict[str, Dict[str, Any]] = {}
        for group_name in FILTER_ANALYSIS_ORDER:
            if not average_filters_by_group_l2.get(group_name):
                continue
            sample_pairs_l2, grouped_pairs_l2 = _build_overlap_pairs(
                exp1_dir,
                average_filters_by_group_l2[group_name],
                sample_filters_by_group_l2[group_name],
                delta_key,
            )
            summarized_l2 = _summarize_target(
                sample_pairs_l2,
                grouped_pairs_l2,
                PRIMARY_TARGET,
                TARGET_SPECS[PRIMARY_TARGET],
            )
            if summarized_l2 is None:
                continue
            summary_l2, tests_l2, metrics_l2, _, _ = summarized_l2
            source_analyses_l2[group_name] = {
                "summary": summary_l2,
                "tests": tests_l2,
                "metrics": metrics_l2,
                "normalization": "l2",
            }
        if source_analyses:
            analyses[source_name] = source_analyses
        if source_analyses_l2:
            analyses_l2[source_name] = source_analyses_l2

    if not analyses:
        return ExperimentResult(
            experiment_id=5,
            experiment_name="exp5_overlap_prediction",
            config=exp_cfg,
            metrics={"error": "too_few_pairs", "n_pairs": 0},
        )

    logger.info("-" * 40)
    primary_target_label = TARGET_SPECS.get(PRIMARY_TARGET, {}).get("label", PRIMARY_TARGET)
    for source_spec in source_specs:
        source_name = source_spec["name"]
        source_analyses = analyses.get(source_name)
        if source_analyses is None:
            continue
        for group_name in FILTER_ANALYSIS_ORDER:
            analysis = source_analyses.get(group_name)
            if analysis is None:
                continue
            acc_summary = analysis["target_analyses"].get(PRIMARY_TARGET, {}).get("summary")
            if acc_summary is not None:
                logger.info(
                    "  %s/%s grouped Pearson r (%s) = %.3f",
                    source_name,
                    group_name,
                    primary_target_label,
                    acc_summary["pearson_r_grouped"],
                )
            loglik_summary = analysis["target_analyses"].get("loglik_erosion", {}).get("summary")
            if loglik_summary is not None and group_name == primary_group:
                logger.info(
                    "  %s/%s grouped Pearson r (loglik erosion) = %.3f",
                    source_name,
                    group_name,
                    loglik_summary["pearson_r_grouped"],
                )
        primary_analysis = source_analyses.get(primary_group)
        if primary_analysis is not None:
            acc_summary = primary_analysis["target_analyses"].get(PRIMARY_TARGET, {}).get("summary")
            if acc_summary is not None and acc_summary["pearson_r_sample"] is not None:
                logger.info(
                    "  %s/%s sample Pearson r (%s) = %.3f",
                    source_name,
                    primary_group,
                    primary_target_label,
                    acc_summary["pearson_r_sample"],
                )
    logger.info("-" * 40)

    summary_payload: Dict[str, Any] = {
        "analysis_groups": list(FILTER_ANALYSIS_ORDER),
        "targets": list(TARGET_ORDER),
        "primary_group": primary_group,
        "primary_target": PRIMARY_TARGET,
        "primary_delta_normalization": primary_normalization,
        "control_groups": control_groups,
        "source_metadata": source_metadata,
        "normalization_variants": {"l2": {}},
    }
    tests_payload: Dict[str, Any] = {
        "analysis_groups": list(FILTER_ANALYSIS_ORDER),
        "targets": list(TARGET_ORDER),
        "primary_group": primary_group,
        "primary_target": PRIMARY_TARGET,
        "primary_delta_normalization": primary_normalization,
        "control_groups": control_groups,
        "source_metadata": source_metadata,
        "normalization_variants": {"l2": {}},
    }
    metrics: Dict[str, Any] = {}
    scatter_payload: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    sample_scatter_payload: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    scatter_payload_by_target: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    sample_scatter_payload_by_target: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}

    for source_name, source_analyses in analyses.items():
        source_summary: Dict[str, Any] = {
            "analysis_groups": list(FILTER_ANALYSIS_ORDER),
            "targets": list(TARGET_ORDER),
            "primary_group": primary_group,
            "primary_target": PRIMARY_TARGET,
            "domain": source_metadata[source_name]["domain"],
            "delta_normalization": source_metadata[source_name]["normalization"],
            "control_groups": control_groups,
            "group_summaries": {},
            "group_target_summaries": {},
            "grouped_pearson_by_group": {},
            "grouped_pearson_views_by_group": {},
            "target_grouped_pearson_by_group": {target_name: {} for target_name in TARGET_ORDER},
            "target_grouped_pearson_views_by_group": {target_name: {} for target_name in TARGET_ORDER},
            "prediction_factor_horse_races": {},
        }
        source_tests: Dict[str, Any] = {
            "analysis_groups": list(FILTER_ANALYSIS_ORDER),
            "targets": list(TARGET_ORDER),
            "primary_group": primary_group,
            "primary_target": PRIMARY_TARGET,
            "domain": source_metadata[source_name]["domain"],
            "delta_normalization": source_metadata[source_name]["normalization"],
            "control_groups": control_groups,
            "group_tests": {},
            "group_target_tests": {},
            "prediction_factor_horse_races": {},
        }
        scatter_payload[source_name] = {}
        sample_scatter_payload[source_name] = {}
        scatter_payload_by_target[source_name] = {}
        sample_scatter_payload_by_target[source_name] = {}

        for group_name in FILTER_ANALYSIS_ORDER:
            analysis = source_analyses.get(group_name)
            if analysis is None:
                continue

            group_target_summaries: Dict[str, Any] = {}
            group_target_tests: Dict[str, Any] = {}
            scatter_payload_by_target[source_name][group_name] = {}
            sample_scatter_payload_by_target[source_name][group_name] = {}

            for target_name in TARGET_ORDER:
                target_analysis = analysis["target_analyses"].get(target_name)
                if target_analysis is None:
                    continue
                group_target_summaries[target_name] = target_analysis["summary"]
                group_target_tests[target_name] = target_analysis["tests"]
                scatter_payload_by_target[source_name][group_name][target_name] = target_analysis["grouped_pairs"]
                if target_analysis["sample_pairs"]:
                    sample_scatter_payload_by_target[source_name][group_name][target_name] = target_analysis["sample_pairs"]
                for metric_name, value in target_analysis["metrics"].items():
                    metrics[f"{metric_name}_{target_name}_{source_name}_{group_name}"] = value
                source_summary["target_grouped_pearson_by_group"][target_name][group_name] = (
                    target_analysis["summary"]["pearson_r_grouped"]
                )
                source_summary["target_grouped_pearson_views_by_group"][target_name][group_name] = (
                    target_analysis["summary"].get("grouped_views", {})
                )

            accuracy_analysis = analysis["target_analyses"].get(PRIMARY_TARGET)
            if accuracy_analysis is None:
                continue

            source_summary["group_summaries"][group_name] = accuracy_analysis["summary"]
            source_summary["group_target_summaries"][group_name] = group_target_summaries
            source_summary["grouped_pearson_by_group"][group_name] = accuracy_analysis["summary"]["pearson_r_grouped"]
            source_summary["grouped_pearson_views_by_group"][group_name] = accuracy_analysis["summary"].get("grouped_views", {})
            source_summary["prediction_factor_horse_races"][group_name] = analysis.get(
                "prediction_factor_horse_race",
                {},
            )

            source_tests["group_tests"][group_name] = accuracy_analysis["tests"]
            source_tests["group_target_tests"][group_name] = group_target_tests
            source_tests["prediction_factor_horse_races"][group_name] = analysis.get(
                "prediction_factor_horse_race",
                {},
            )

            scatter_payload[source_name][group_name] = accuracy_analysis["grouped_pairs"]
            sample_scatter_payload[source_name][group_name] = accuracy_analysis["sample_pairs"]

        primary_group_analysis = source_analyses.get(primary_group) or next(iter(source_analyses.values()))
        primary_target_analyses = primary_group_analysis["target_analyses"]
        primary_accuracy = primary_target_analyses.get(PRIMARY_TARGET) or next(iter(primary_target_analyses.values()))

        source_summary["primary_group_summary"] = primary_accuracy["summary"]
        source_summary["primary_prediction_factor_horse_race"] = primary_group_analysis.get(
            "prediction_factor_horse_race",
            {},
        )
        source_summary["primary_group_target_summaries"] = {
            target_name: analysis["summary"]
            for target_name, analysis in primary_target_analyses.items()
        }
        source_tests["primary_group_tests"] = primary_accuracy["tests"]
        source_tests["primary_prediction_factor_horse_race"] = primary_group_analysis.get(
            "prediction_factor_horse_race",
            {},
        )
        source_tests["primary_group_target_tests"] = {
            target_name: analysis["tests"]
            for target_name, analysis in primary_target_analyses.items()
        }
        source_tests["hypothesis_supported"] = primary_accuracy["tests"].get("hypothesis_supported", False)

        summary_payload[source_name] = source_summary
        tests_payload[source_name] = source_tests

    for source_name, source_analyses_l2 in analyses_l2.items():
        summary_payload["normalization_variants"]["l2"][source_name] = {}
        tests_payload["normalization_variants"]["l2"][source_name] = {}
        for group_name, analysis_l2 in source_analyses_l2.items():
            summary_payload["normalization_variants"]["l2"][source_name][group_name] = {
                "summary": analysis_l2["summary"],
                "metrics": analysis_l2["metrics"],
                "normalization": "l2",
            }
            tests_payload["normalization_variants"]["l2"][source_name][group_name] = analysis_l2["tests"]

    image_primary = analyses.get("image_space", {}).get(primary_group)
    if image_primary is None and analyses.get("image_space"):
        image_primary = next(iter(analyses["image_space"].values()))
    if image_primary is None:
        first_source = next(iter(analyses.values()))
        primary_group_analysis = first_source.get(primary_group) or next(iter(first_source.values()))
    else:
        primary_group_analysis = image_primary

    primary_target_analysis = (
        primary_group_analysis["target_analyses"].get(PRIMARY_TARGET)
        or next(iter(primary_group_analysis["target_analyses"].values()))
    )
    summary_payload.update(primary_target_analysis["summary"])
    summary_payload["prediction_factor_horse_race"] = primary_group_analysis.get(
        "prediction_factor_horse_race",
        {},
    )
    tests_payload.update(primary_target_analysis["tests"])
    tests_payload["prediction_factor_horse_race"] = primary_group_analysis.get(
        "prediction_factor_horse_race",
        {},
    )
    tests_payload["hypothesis_supported"] = any(
        analyses[source_name].get(primary_group, next(iter(analyses[source_name].values())))["target_analyses"]
        .get(
            PRIMARY_TARGET,
            next(
                iter(
                    analyses[source_name]
                    .get(primary_group, next(iter(analyses[source_name].values())))["target_analyses"]
                    .values()
                )
            ),
        )["tests"].get("hypothesis_supported", False)
        for source_name in ("image_space", "vision_feature_space")
        if analyses.get(source_name)
    )

    save_json(summary_payload, out_dir / "summary.json")
    save_json(tests_payload, out_dir / "hypothesis_tests.json")
    save_json(scatter_payload, out_dir / "scatter_data_by_group.json")
    save_json(sample_scatter_payload, out_dir / "sample_scatter_data_by_group.json")
    save_json(scatter_payload_by_target, out_dir / "scatter_data_by_group_and_target.json")
    save_json(sample_scatter_payload_by_target, out_dir / "sample_scatter_data_by_group_and_target.json")
    if analyses.get("image_space", {}).get(primary_group) is not None:
        primary_accuracy = analyses["image_space"][primary_group]["target_analyses"].get(PRIMARY_TARGET)
        if primary_accuracy is not None:
            save_json(primary_accuracy["grouped_pairs"], out_dir / "scatter_data.json")
            save_json(primary_accuracy["sample_pairs"], out_dir / "sample_scatter_data.json")
    if analyses.get("vision_feature_space", {}).get(primary_group) is not None:
        primary_accuracy = analyses["vision_feature_space"][primary_group]["target_analyses"].get(PRIMARY_TARGET)
        if primary_accuracy is not None:
            save_json(primary_accuracy["grouped_pairs"], out_dir / "vision_scatter_data.json")
            save_json(primary_accuracy["sample_pairs"], out_dir / "vision_sample_scatter_data.json")

    return ExperimentResult(
        experiment_id=5,
        experiment_name="exp5_overlap_prediction",
        config=exp_cfg,
        metrics=metrics,
        hypothesis_tests=tests_payload,
    )
