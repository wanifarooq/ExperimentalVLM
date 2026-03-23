"""Experiment 5: Spectral Overlap Prediction."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..analysis.spectral import compute_spectral_overlap
from ..analysis.statistics import bootstrap_ci, pearson_correlation, spearman_correlation
from ..data.base import ExperimentResult
from ..utils.io import load_json, save_json

logger = logging.getLogger(__name__)


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


def _load_average_filters(exp2_out_dir: Path) -> Dict[str, np.ndarray]:
    filters_dir = exp2_out_dir / "filters"
    filters: Dict[str, np.ndarray] = {}
    for level_key in ["L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE"]:
        path = filters_dir / f"average_{level_key}.npy"
        if path.exists():
            filters[level_key] = np.load(path)
    return filters


def _load_sample_filters(exp2_out_dir: Path) -> Dict[Tuple[str, str], np.ndarray]:
    power_path = exp2_out_dir / "power_spectra.json"
    if not power_path.exists():
        return {}

    filters: Dict[Tuple[str, str], np.ndarray] = {}
    for record in load_json(power_path):
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            w_t = level_data.get("W_t")
            if w_t:
                filters[(image_id, level_key)] = np.asarray(w_t, dtype=np.float64)
    return filters


def _build_overlap_pairs(
    exp1_out_dir: Path,
    exp2_out_dir: Path,
    delta_key: str = "delta_f",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    exp1_records = _load_jsonl(exp1_out_dir / "per_sample.jsonl")
    sample_filters = _load_sample_filters(exp2_out_dir)
    average_filters = _load_average_filters(exp2_out_dir)

    sample_pairs: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"predicted": [], "actual": [], "loglik_drift": []}
    )

    for record in exp1_records:
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            W_t = sample_filters.get((image_id, level_key), average_filters.get(level_key))
            if W_t is None:
                continue

            for perturbation in level_data.get("perturbations", []):
                delta_f = np.asarray(perturbation.get(delta_key) or [], dtype=np.float64)
                if delta_f.size == 0:
                    continue

                predicted = compute_spectral_overlap(W_t, delta_f)
                actual = float(perturbation.get("accuracy_drop", 0.0))
                loglik_drift = float(perturbation.get("loglik_drift", 0.0))
                pair = {
                    "image_id": image_id,
                    "level": level_key,
                    "perturbation": perturbation.get("name", "unknown"),
                    "predicted": predicted,
                    "actual": actual,
                    "loglik_drift": loglik_drift,
                }
                sample_pairs.append(pair)

                key = (level_key, pair["perturbation"])
                grouped[key]["predicted"].append(predicted)
                grouped[key]["actual"].append(actual)
                grouped[key]["loglik_drift"].append(loglik_drift)

    grouped_pairs: List[Dict[str, Any]] = []
    for (level_key, perturbation), values in sorted(grouped.items()):
        grouped_pairs.append(
            {
                "label": f"{level_key}|{perturbation}",
                "level": level_key,
                "perturbation": perturbation,
                "predicted": float(np.mean(values["predicted"])),
                "actual": float(np.mean(values["actual"])),
                "mean_loglik_drift": float(np.mean(values["loglik_drift"])),
                "n": len(values["actual"]),
            }
        )

    return sample_pairs, grouped_pairs


def _summarize_overlap(
    sample_pairs: List[Dict[str, Any]],
    grouped_pairs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    pred_arr, actual_arr = _correlate(grouped_pairs, "actual")
    r_pearson, p_pearson = pearson_correlation(pred_arr, actual_arr)
    rho_spearman, p_spearman = spearman_correlation(pred_arr, actual_arr)

    sample_pred_arr, sample_actual_arr = _correlate(sample_pairs, "actual")
    sample_r, sample_r_p = pearson_correlation(sample_pred_arr, sample_actual_arr)

    pearson_ci = None
    if len(grouped_pairs) >= 10:
        def _pearson_stat(indices):
            idx = indices.astype(int)
            return np.corrcoef(pred_arr[idx], actual_arr[idx])[0, 1]

        point, lo, hi = bootstrap_ci(np.arange(len(grouped_pairs)), statistic=_pearson_stat)
        pearson_ci = {"mean": point, "ci_95_lower": lo, "ci_95_upper": hi}

    tests: Dict[str, Any] = {
        "pearson_predicted_vs_actual_grouped": {
            "r": r_pearson,
            "p_value": p_pearson,
            "target": "r > 0.7",
            "passed": r_pearson > 0.7,
        },
        "spearman_predicted_vs_actual_grouped": {
            "rho": rho_spearman,
            "p_value": p_spearman,
            "passed": rho_spearman > 0.6,
        },
        "pearson_predicted_vs_actual_sample": {
            "r": sample_r,
            "p_value": sample_r_p,
            "passed": sample_r > 0.3,
        },
    }
    if pearson_ci is not None:
        tests["pearson_grouped_bootstrap_ci"] = pearson_ci
    tests["hypothesis_supported"] = tests["pearson_predicted_vs_actual_grouped"]["passed"]

    per_level_corr: Dict[str, Any] = {}
    for level_key in sorted({pair["level"] for pair in grouped_pairs}):
        level_pairs = [pair for pair in grouped_pairs if pair["level"] == level_key]
        if len(level_pairs) < 3:
            continue
        level_pred, level_actual = _correlate(level_pairs, "actual")
        r, p = pearson_correlation(level_pred, level_actual)
        per_level_corr[level_key] = {"pearson_r": r, "p_value": p, "n": len(level_pairs)}

    summary = {
        "n_grouped_pairs": len(grouped_pairs),
        "n_sample_pairs": len(sample_pairs),
        "pearson_r_grouped": r_pearson,
        "spearman_rho_grouped": rho_spearman,
        "pearson_r_sample": sample_r,
        "per_level_correlation": per_level_corr,
    }
    metrics = {
        "n_grouped_pairs": len(grouped_pairs),
        "n_sample_pairs": len(sample_pairs),
        "pearson_r_grouped": r_pearson,
        "pearson_p_grouped": p_pearson,
        "spearman_rho_grouped": rho_spearman,
        "spearman_p_grouped": p_spearman,
        "pearson_r_sample": sample_r,
        "pearson_p_sample": sample_r_p,
    }
    return summary, tests, metrics


def _correlate(pairs: List[Dict[str, Any]], actual_key: str) -> Tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray([pair["predicted"] for pair in pairs], dtype=np.float64)
    actual = np.asarray([pair[actual_key] for pair in pairs], dtype=np.float64)
    return predicted, actual


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

    overlap_sources = {
        "image_space": _build_overlap_pairs(exp1_dir, exp2_dir, "delta_f"),
        "vision_feature_space": _build_overlap_pairs(exp1_dir, exp2_dir, "delta_f_vision"),
    }
    analyses: Dict[str, Dict[str, Any]] = {}
    for source_name, (sample_pairs, grouped_pairs) in overlap_sources.items():
        if len(grouped_pairs) < 3:
            logger.warning(
                "Too few grouped pairs (%d) for correlation in %s",
                len(grouped_pairs),
                source_name,
            )
            continue
        summary, tests, metrics = _summarize_overlap(sample_pairs, grouped_pairs)
        analyses[source_name] = {
            "summary": summary,
            "tests": tests,
            "metrics": metrics,
            "sample_pairs": sample_pairs,
            "grouped_pairs": grouped_pairs,
        }

    if not analyses:
        return ExperimentResult(
            experiment_id=5,
            experiment_name="exp5_overlap_prediction",
            config=exp_cfg,
            metrics={"error": "too_few_pairs", "n_pairs": 0},
        )
    image_analysis = analyses.get("image_space")
    vision_analysis = analyses.get("vision_feature_space")

    logger.info("-" * 40)
    if image_analysis is not None:
        logger.info(
            "  Image-space grouped Pearson r = %.3f [%s]",
            image_analysis["summary"]["pearson_r_grouped"],
            "PASS" if image_analysis["tests"]["pearson_predicted_vs_actual_grouped"]["passed"] else "FAIL",
        )
        logger.info(
            "  Image-space sample Pearson r = %.3f",
            image_analysis["summary"]["pearson_r_sample"],
        )
    if vision_analysis is not None:
        logger.info(
            "  Vision-feature grouped Pearson r = %.3f [%s]",
            vision_analysis["summary"]["pearson_r_grouped"],
            "PASS" if vision_analysis["tests"]["pearson_predicted_vs_actual_grouped"]["passed"] else "FAIL",
        )
        logger.info(
            "  Vision-feature sample Pearson r = %.3f",
            vision_analysis["summary"]["pearson_r_sample"],
        )
    logger.info("-" * 40)

    summary_payload: Dict[str, Any] = {}
    tests_payload: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    for source_name, analysis in analyses.items():
        summary_payload[source_name] = analysis["summary"]
        tests_payload[source_name] = analysis["tests"]
        for metric_name, value in analysis["metrics"].items():
            metrics[f"{metric_name}_{source_name}"] = value

    primary = image_analysis or next(iter(analyses.values()))
    summary_payload.update(primary["summary"])
    tests_payload.update(primary["tests"])
    tests_payload["hypothesis_supported"] = any(
        analysis["tests"].get("hypothesis_supported", False)
        for analysis in analyses.values()
    )

    save_json(summary_payload, out_dir / "summary.json")
    save_json(tests_payload, out_dir / "hypothesis_tests.json")
    if image_analysis is not None:
        save_json(image_analysis["grouped_pairs"], out_dir / "scatter_data.json")
        save_json(image_analysis["sample_pairs"], out_dir / "sample_scatter_data.json")
    if vision_analysis is not None:
        save_json(vision_analysis["grouped_pairs"], out_dir / "vision_scatter_data.json")
        save_json(vision_analysis["sample_pairs"], out_dir / "vision_sample_scatter_data.json")

    return ExperimentResult(
        experiment_id=5,
        experiment_name="exp5_overlap_prediction",
        config=exp_cfg,
        metrics=metrics,
        hypothesis_tests=tests,
    )
