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
                delta_f = np.asarray(perturbation.get("delta_f") or [], dtype=np.float64)
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

    sample_pairs, grouped_pairs = _build_overlap_pairs(exp1_dir, exp2_dir)
    if len(grouped_pairs) < 3:
        logger.warning("Too few grouped pairs (%d) for correlation", len(grouped_pairs))
        return ExperimentResult(
            experiment_id=5,
            experiment_name="exp5_overlap_prediction",
            config=exp_cfg,
            metrics={"error": "too_few_pairs", "n_pairs": len(grouped_pairs)},
        )

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

    logger.info("-" * 40)
    logger.info(
        "  Grouped Pearson r = %.3f (p = %.4f) [%s]",
        r_pearson,
        p_pearson,
        "PASS" if r_pearson > 0.7 else "FAIL",
    )
    logger.info("  Grouped Spearman rho = %.3f (p = %.4f)", rho_spearman, p_spearman)
    logger.info("  Sample Pearson r = %.3f (p = %.4f)", sample_r, sample_r_p)
    logger.info("-" * 40)

    save_json(
        {
            "n_grouped_pairs": len(grouped_pairs),
            "n_sample_pairs": len(sample_pairs),
            "pearson_r_grouped": r_pearson,
            "spearman_rho_grouped": rho_spearman,
            "pearson_r_sample": sample_r,
            "per_level_correlation": per_level_corr,
        },
        out_dir / "summary.json",
    )
    save_json(tests, out_dir / "hypothesis_tests.json")
    save_json(grouped_pairs, out_dir / "scatter_data.json")
    save_json(sample_pairs, out_dir / "sample_scatter_data.json")

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

    return ExperimentResult(
        experiment_id=5,
        experiment_name="exp5_overlap_prediction",
        config=exp_cfg,
        metrics=metrics,
        hypothesis_tests=tests,
    )
