"""Experiment 5: Spectral Overlap Prediction."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..analysis.spectral import compute_spectral_overlap
from ..analysis.statistics import bootstrap_ci, pearson_correlation, spearman_correlation
from ..data.base import ExperimentResult
from ..utils.exp2_filters import load_exp2_filter_bank
from ..utils.io import save_json
from ..utils.layer_groups import LAYER_GROUP_ORDER

logger = logging.getLogger(__name__)

FILTER_ANALYSIS_ORDER = ("overall",) + LAYER_GROUP_ORDER
PRIMARY_TARGET = "accuracy_drop"
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
    "net_drop": {
        "grouped_key": "net_drop",
        "sample_key": "net_change",
        "label": "Net Accuracy Change",
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
            "accuracy_drop": [],
            "loglik_erosion": [],
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
                loglik_erosion = float(perturbation.get("loglik_drift", 0.0))
                predicted = compute_spectral_overlap(W_t, delta_f)

                pair = {
                    "image_id": image_id,
                    "level": level_key,
                    "perturbation": perturbation.get("name", "unknown"),
                    "predicted": predicted,
                    "actual": accuracy_drop,
                    "accuracy_drop": accuracy_drop,
                    "loglik_erosion": loglik_erosion,
                    "loglik_drift": loglik_erosion,
                    "net_change": net_change,
                    "clean_correct": clean_correct,
                    "perturbed_correct": perturbed_correct,
                    "ci": ci,
                    "ic": ic,
                }
                sample_pairs.append(pair)

                key = (level_key, pair["perturbation"])
                grouped[key]["predicted"].append(predicted)
                grouped[key]["accuracy_drop"].append(accuracy_drop)
                grouped[key]["loglik_erosion"].append(loglik_erosion)
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
                "actual": float(np.mean(values["accuracy_drop"])),
                "accuracy_drop": float(np.mean(values["accuracy_drop"])),
                "loglik_erosion": float(np.mean(values["loglik_erosion"])),
                "loglik_drift": float(np.mean(values["loglik_erosion"])),
                "mean_loglik_drift": float(np.mean(values["loglik_erosion"])),
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


def _summarize_target(
    sample_pairs: List[Dict[str, Any]],
    grouped_pairs: List[Dict[str, Any]],
    target_name: str,
    target_spec: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    grouped_target_pairs = _materialize_target_pairs(grouped_pairs, target_spec.get("grouped_key"))
    if len(grouped_target_pairs) < 3:
        return None

    pred_arr, actual_arr = _correlate(grouped_target_pairs)
    r_pearson, p_pearson = pearson_correlation(pred_arr, actual_arr)
    rho_spearman, p_spearman = spearman_correlation(pred_arr, actual_arr)

    sample_target_pairs: List[Dict[str, Any]] = []
    sample_r = None
    sample_r_p = None
    supports_sample = bool(target_spec.get("supports_sample")) and target_spec.get("sample_key") is not None
    if supports_sample:
        sample_target_pairs = _materialize_target_pairs(sample_pairs, target_spec["sample_key"])
        if len(sample_target_pairs) >= 3:
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
            "target": "r > 0.7",
            "passed": r_pearson > 0.7,
        },
        "spearman_predicted_vs_actual_grouped": {
            "rho": rho_spearman,
            "p_value": p_spearman,
            "passed": rho_spearman > 0.6,
        },
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
    tests["hypothesis_supported"] = tests["pearson_predicted_vs_actual_grouped"]["passed"]

    per_level_corr: Dict[str, Any] = {}
    for level_key in sorted({pair["level"] for pair in grouped_target_pairs}):
        level_pairs = [pair for pair in grouped_target_pairs if pair["level"] == level_key]
        if len(level_pairs) < 3:
            continue
        level_pred, level_actual = _correlate(level_pairs)
        r, p = pearson_correlation(level_pred, level_actual)
        per_level_corr[level_key] = {"pearson_r": r, "p_value": p, "n": len(level_pairs)}

    summary = {
        "target": target_name,
        "label": target_spec["label"],
        "n_grouped_pairs": len(grouped_target_pairs),
        "n_sample_pairs": len(sample_target_pairs),
        "pearson_r_grouped": r_pearson,
        "spearman_rho_grouped": rho_spearman,
        "pearson_r_sample": sample_r,
        "per_level_correlation": per_level_corr,
        "sample_metric_available": sample_r is not None,
    }
    metrics = {
        "n_grouped_pairs": len(grouped_target_pairs),
        "n_sample_pairs": len(sample_target_pairs),
        "pearson_r_grouped": r_pearson,
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
    analyses: Dict[str, Dict[str, Dict[str, Any]]] = {}
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

            source_analyses[group_name] = {
                "raw_sample_pairs": sample_pairs,
                "raw_grouped_pairs": grouped_pairs,
                "target_analyses": target_analyses,
            }
        if source_analyses:
            analyses[source_name] = source_analyses

    if not analyses:
        return ExperimentResult(
            experiment_id=5,
            experiment_name="exp5_overlap_prediction",
            config=exp_cfg,
            metrics={"error": "too_few_pairs", "n_pairs": 0},
        )

    logger.info("-" * 40)
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
                    "  %s/%s grouped Pearson r (accuracy) = %.3f",
                    source_name,
                    group_name,
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
                    "  %s/%s sample Pearson r (accuracy) = %.3f",
                    source_name,
                    primary_group,
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
    }
    tests_payload: Dict[str, Any] = {
        "analysis_groups": list(FILTER_ANALYSIS_ORDER),
        "targets": list(TARGET_ORDER),
        "primary_group": primary_group,
        "primary_target": PRIMARY_TARGET,
        "primary_delta_normalization": primary_normalization,
        "control_groups": control_groups,
        "source_metadata": source_metadata,
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
            "target_grouped_pearson_by_group": {target_name: {} for target_name in TARGET_ORDER},
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

            accuracy_analysis = analysis["target_analyses"].get(PRIMARY_TARGET)
            if accuracy_analysis is None:
                continue

            source_summary["group_summaries"][group_name] = accuracy_analysis["summary"]
            source_summary["group_target_summaries"][group_name] = group_target_summaries
            source_summary["grouped_pearson_by_group"][group_name] = accuracy_analysis["summary"]["pearson_r_grouped"]

            source_tests["group_tests"][group_name] = accuracy_analysis["tests"]
            source_tests["group_target_tests"][group_name] = group_target_tests

            scatter_payload[source_name][group_name] = accuracy_analysis["grouped_pairs"]
            sample_scatter_payload[source_name][group_name] = accuracy_analysis["sample_pairs"]

        primary_group_analysis = source_analyses.get(primary_group) or next(iter(source_analyses.values()))
        primary_target_analyses = primary_group_analysis["target_analyses"]
        primary_accuracy = primary_target_analyses.get(PRIMARY_TARGET) or next(iter(primary_target_analyses.values()))

        source_summary["primary_group_summary"] = primary_accuracy["summary"]
        source_summary["primary_group_target_summaries"] = {
            target_name: analysis["summary"]
            for target_name, analysis in primary_target_analyses.items()
        }
        source_tests["primary_group_tests"] = primary_accuracy["tests"]
        source_tests["primary_group_target_tests"] = {
            target_name: analysis["tests"]
            for target_name, analysis in primary_target_analyses.items()
        }
        source_tests["hypothesis_supported"] = primary_accuracy["tests"].get("hypothesis_supported", False)

        summary_payload[source_name] = source_summary
        tests_payload[source_name] = source_tests

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
    tests_payload.update(primary_target_analysis["tests"])
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
