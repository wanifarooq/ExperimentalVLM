#!/usr/bin/env python3
"""Memory-safe Exp5 reaggregation for existing Exp1/Exp2 outputs.

This is intentionally a sidecar runner: it reuses the production Exp5 math but
does not materialize every source/group/target scatter payload at once.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frequency_alignment.analysis.spectral import (  # noqa: E402
    compute_overlap_integral,
    compute_spectral_overlap,
)
from frequency_alignment.data.base import ALL_VQA_LEVEL_NAMES  # noqa: E402
from frequency_alignment.experiments.exp5_overlap_prediction import (  # noqa: E402
    FILTER_ANALYSIS_ORDER,
    PRIMARY_OVERLAP_VARIANT,
    PRIMARY_TARGET,
    TARGET_ORDER,
    TARGET_SPECS,
    _build_source_specs,
    _perturbation_family_name,
    _summarize_prediction_factor_horse_race,
    _summarize_target,
)
from frequency_alignment.utils.io import save_json  # noqa: E402

LOGGER = logging.getLogger("run_exp5_streaming")
_EPS = 1e-8


def _iter_json_array(path: Path, *, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Yield objects from a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    in_array = False
    with path.open("r") as handle:
        while True:
            chunk = handle.read(chunk_size)
            eof = chunk == ""
            buffer += chunk

            if not in_array:
                idx = 0
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
                if idx < len(buffer):
                    if buffer[idx] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    in_array = True
                    buffer = buffer[idx + 1 :]
                elif eof:
                    return
                else:
                    continue

            while True:
                idx = 0
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
                if idx < len(buffer) and buffer[idx] == ",":
                    idx += 1
                    while idx < len(buffer) and buffer[idx].isspace():
                        idx += 1
                if idx < len(buffer) and buffer[idx] == "]":
                    return
                if idx >= len(buffer):
                    buffer = ""
                    break
                try:
                    obj, end = decoder.raw_decode(buffer, idx)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    buffer = buffer[idx:]
                    break
                yield obj
                buffer = buffer[end:]

            if eof:
                return


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_config(output_dir: Path) -> dict:
    config_path = output_dir / "config_snapshot.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")
    with config_path.open("r") as handle:
        cfg = json.load(handle)
    cfg["out_dir"] = str(output_dir)
    return cfg


def _load_average_filters(exp2_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    average_filters: Dict[str, Dict[str, np.ndarray]] = {group: {} for group in FILTER_ANALYSIS_ORDER}
    filters_dir = exp2_dir / "filters"
    for level_key in ALL_VQA_LEVEL_NAMES:
        overall_path = filters_dir / f"average_{level_key}.npy"
        if overall_path.exists():
            average_filters["overall"][level_key] = np.load(overall_path)
        for group_name in FILTER_ANALYSIS_ORDER:
            if group_name == "overall":
                continue
            group_path = filters_dir / f"average_{level_key}_{group_name}.npy"
            if group_path.exists():
                average_filters[group_name][level_key] = np.load(group_path)
    return average_filters


def _load_sample_filters_streaming(
    exp2_dir: Path,
) -> Dict[str, Dict[Tuple[str, str], np.ndarray]]:
    sample_filters: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {
        group: {} for group in FILTER_ANALYSIS_ORDER
    }
    power_path = exp2_dir / "power_spectra.json"
    if not power_path.exists():
        LOGGER.warning("Missing %s; Exp5 will fall back to average filters only", power_path)
        return sample_filters

    for idx, record in enumerate(_iter_json_array(power_path), start=1):
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            w_t = level_data.get("W_t")
            if w_t:
                sample_filters["overall"][(image_id, level_key)] = np.asarray(
                    w_t,
                    dtype=np.float64,
                )
            layer_groups = level_data.get("layer_groups", {})
            for group_name in FILTER_ANALYSIS_ORDER:
                if group_name == "overall":
                    continue
                group_w_t = layer_groups.get(group_name, {}).get("W_t")
                if group_w_t:
                    sample_filters[group_name][(image_id, level_key)] = np.asarray(
                        group_w_t,
                        dtype=np.float64,
                    )
        if idx % 100 == 0:
            LOGGER.info("Loaded compact Exp2 filters for %d samples", idx)
    return sample_filters


def _empty_group_bucket() -> Dict[str, list]:
    return {
        "predicted": [],
        "predicted_quadratic": [],
        "predicted_first_order": [],
        "complexity_score": [],
        "question_complexity_score": [],
        "prompt_complexity_score": [],
        "option_hardness_score": [],
        "is_binary": [],
        "num_options": [],
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


def _build_overlap_pairs_streaming(
    exp1_dir: Path,
    average_filters: Dict[str, np.ndarray],
    sample_filters: Dict[Tuple[str, str], np.ndarray],
    delta_key: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build one source/group pair set from Exp1 JSONL without loading all Exp1 first."""
    sample_pairs: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, str], Dict[str, list]] = defaultdict(_empty_group_bucket)
    per_sample_path = exp1_dir / "per_sample.jsonl"

    for record_idx, record in enumerate(_iter_jsonl(per_sample_path), start=1):
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
            is_binary_flag = float(level_data.get("is_binary", 0.0) or 0.0)
            num_options = int(level_data.get("num_options", 0) or 0)
            clean_correct = float(bool(level_data.get("clean", {}).get("correct", False)))

            for perturbation in level_data.get("perturbations", []):
                perturbation_name = perturbation.get("name", "unknown")
                perturbation_family = _perturbation_family_name(perturbation_name)
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
                predicted_first_order = compute_overlap_integral(W_t, delta_f)

                pair = {
                    "image_id": image_id,
                    "level": level_key,
                    "perturbation": perturbation_name,
                    "perturbation_family": perturbation_family,
                    "predicted": predicted_first_order,
                    "predicted_quadratic": predicted_quadratic,
                    "predicted_first_order": predicted_first_order,
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
                    "is_binary": is_binary_flag,
                    "num_options": num_options,
                }
                sample_pairs.append(pair)

                key = (image_id, level_key, perturbation_family)
                bucket = grouped[key]
                bucket["predicted"].append(predicted_first_order)
                bucket["predicted_quadratic"].append(predicted_quadratic)
                bucket["predicted_first_order"].append(predicted_first_order)
                bucket["complexity_score"].append(complexity_score)
                bucket["question_complexity_score"].append(question_complexity_score)
                bucket["prompt_complexity_score"].append(prompt_complexity_score)
                bucket["option_hardness_score"].append(option_hardness_score)
                bucket["is_binary"].append(is_binary_flag)
                bucket["num_options"].append(num_options)
                bucket["accuracy_drop"].append(accuracy_drop)
                bucket["loglik_drift"].append(loglik_drift)
                bucket["loglik_erosion"].append(loglik_erosion)
                bucket["loglik_recovery"].append(loglik_recovery)
                bucket["loglik_volatility"].append(loglik_volatility)
                bucket["net_change"].append(net_change)
                bucket["clean_correct"].append(clean_correct)
                bucket["perturbed_correct"].append(perturbed_correct)
                bucket["ci"].append(ci)
                bucket["ic"].append(ic)

        if record_idx % 100 == 0:
            LOGGER.info("  processed %d Exp1 samples for delta=%s", record_idx, delta_key)

    grouped_pairs: List[Dict[str, Any]] = []
    for (image_id, level_key, perturbation_family), values in sorted(grouped.items()):
        clean_accuracy = float(np.mean(values["clean_correct"]))
        perturbed_accuracy = float(np.mean(values["perturbed_correct"]))
        net_drop = float(np.mean(values["net_change"]))
        relative_accuracy_drop = None
        if clean_accuracy > _EPS:
            relative_accuracy_drop = float(net_drop / clean_accuracy)

        grouped_pairs.append(
            {
                "label": f"{image_id}|{level_key}|{perturbation_family}",
                "image_id": image_id,
                "level": level_key,
                "perturbation": perturbation_family,
                "perturbation_family": perturbation_family,
                "predicted": float(np.mean(values["predicted"])),
                "predicted_quadratic": float(np.mean(values["predicted_quadratic"])),
                "predicted_first_order": float(np.mean(values["predicted_first_order"])),
                "predicted_2d_first_order": None,
                "n_2d_overlap": 0,
                "complexity_score": float(np.mean(values["complexity_score"])),
                "question_complexity_score": float(np.mean(values["question_complexity_score"])),
                "prompt_complexity_score": float(np.mean(values["prompt_complexity_score"])),
                "option_hardness_score": float(np.mean(values["option_hardness_score"])),
                "is_binary": float(np.mean(values["is_binary"])) if values["is_binary"] else 0.0,
                "num_options": int(round(float(np.mean(values["num_options"])))) if values["num_options"] else 0,
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


def _build_first_order_table(grouped_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    levels_seen: List[str] = []
    perts_seen: List[str] = []
    cell_values: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in grouped_pairs:
        level_key = str(row.get("level"))
        pert_name = str(row.get("perturbation"))
        if level_key not in levels_seen:
            levels_seen.append(level_key)
        if pert_name not in perts_seen:
            perts_seen.append(pert_name)
        cell_values[(level_key, pert_name)] = {
            "first_order": float(row.get("predicted_first_order", 0.0) or 0.0),
            "quadratic": float(row.get("predicted_quadratic", 0.0) or 0.0),
            "mean_loglik_drift": float(row.get("mean_loglik_drift", 0.0) or 0.0),
            "mean_loglik_volatility": float(row.get("mean_loglik_volatility", 0.0) or 0.0),
            "mean_accuracy_drop": float(row.get("accuracy_drop", 0.0) or 0.0),
            "n": int(row.get("n", 0) or 0),
        }

    ordered_levels = [level for level in ALL_VQA_LEVEL_NAMES if level in levels_seen]
    ordered_perts = list(perts_seen)
    return {
        "levels": ordered_levels,
        "perturbations": ordered_perts,
        "matrix_first_order": [
            [cell_values.get((level, pert), {}).get("first_order", None) for level in ordered_levels]
            for pert in ordered_perts
        ],
        "cells": {
            f"{level}|{pert}": cell_values.get((level, pert), {})
            for level in ordered_levels
            for pert in ordered_perts
            if (level, pert) in cell_values
        },
    }


def run_streaming_exp5(output_dir: Path) -> None:
    cfg = _load_config(output_dir)
    exp5_dir = output_dir / "exp5"
    exp1_dir = output_dir / "exp1"
    exp2_dir = output_dir / "exp2"
    exp5_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = cfg.get("experiments", {}).get("exp5", {})
    primary_group = exp_cfg.get("primary_layer_group", "late")
    if primary_group not in FILTER_ANALYSIS_ORDER:
        LOGGER.warning("Unknown primary group %s; using late", primary_group)
        primary_group = "late"
    primary_normalization = exp_cfg.get("primary_delta_normalization", "relative")
    source_specs = _build_source_specs(primary_normalization)
    control_groups = [group_name for group_name in FILTER_ANALYSIS_ORDER if group_name != primary_group]

    LOGGER.info("Loading compact Exp2 filters from %s", exp2_dir)
    average_filters_by_group = _load_average_filters(exp2_dir)
    sample_filters_by_group = _load_sample_filters_streaming(exp2_dir)

    source_metadata = {
        spec["name"]: {
            "domain": spec["domain"],
            "normalization": spec["normalization"],
        }
        for spec in source_specs
    }
    summary_payload: Dict[str, Any] = {
        "analysis_groups": list(FILTER_ANALYSIS_ORDER),
        "targets": list(TARGET_ORDER),
        "primary_group": primary_group,
        "primary_target": PRIMARY_TARGET,
        "primary_overlap_variant": PRIMARY_OVERLAP_VARIANT,
        "primary_predicted_key": "predicted_first_order",
        "primary_delta_normalization": primary_normalization,
        "control_groups": control_groups,
        "source_metadata": source_metadata,
        "normalization_variants": {"l2": {}},
        "overlap_2d": {
            "enabled": False,
            "scope": "not_computed_by_streaming_exp5",
            "note": "This sidecar runner intentionally skips optional 2D overlap.",
        },
        "streaming_exp5": {
            "enabled": True,
            "omitted_large_scatter_payloads": True,
            "reason": "Avoid OOM by processing one source/group at a time.",
        },
    }
    tests_payload: Dict[str, Any] = {
        "analysis_groups": list(FILTER_ANALYSIS_ORDER),
        "targets": list(TARGET_ORDER),
        "primary_group": primary_group,
        "primary_target": PRIMARY_TARGET,
        "primary_overlap_variant": PRIMARY_OVERLAP_VARIANT,
        "primary_predicted_key": "predicted_first_order",
        "primary_delta_normalization": primary_normalization,
        "control_groups": control_groups,
        "source_metadata": source_metadata,
        "normalization_variants": {"l2": {}},
        "overlap_2d": summary_payload["overlap_2d"],
        "streaming_exp5": summary_payload["streaming_exp5"],
    }
    metrics: Dict[str, Any] = {}
    first_order_tables: Dict[str, Any] = {}
    primary_summary: Optional[Dict[str, Any]] = None
    primary_tests: Optional[Dict[str, Any]] = None
    primary_horse_race: Optional[Dict[str, Any]] = None

    for source_spec in source_specs:
        source_name = source_spec["name"]
        delta_key = source_spec["delta_key"]
        LOGGER.info("Processing source=%s delta_key=%s", source_name, delta_key)
        source_summary: Dict[str, Any] = {
            "analysis_groups": list(FILTER_ANALYSIS_ORDER),
            "targets": list(TARGET_ORDER),
            "primary_group": primary_group,
            "primary_target": PRIMARY_TARGET,
            "primary_overlap_variant": PRIMARY_OVERLAP_VARIANT,
            "primary_predicted_key": "predicted_first_order",
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
            "primary_overlap_variant": PRIMARY_OVERLAP_VARIANT,
            "primary_predicted_key": "predicted_first_order",
            "domain": source_metadata[source_name]["domain"],
            "delta_normalization": source_metadata[source_name]["normalization"],
            "control_groups": control_groups,
            "group_tests": {},
            "group_target_tests": {},
            "prediction_factor_horse_races": {},
        }

        for group_name in FILTER_ANALYSIS_ORDER:
            LOGGER.info("Building pairs for %s/%s", source_name, group_name)
            sample_pairs, grouped_pairs = _build_overlap_pairs_streaming(
                exp1_dir,
                average_filters_by_group.get(group_name, {}),
                sample_filters_by_group.get(group_name, {}),
                delta_key,
            )
            if len(grouped_pairs) < 3:
                LOGGER.warning("Too few grouped pairs for %s/%s", source_name, group_name)
                continue

            group_target_summaries: Dict[str, Any] = {}
            group_target_tests: Dict[str, Any] = {}
            for target_name in TARGET_ORDER:
                summarized = _summarize_target(
                    sample_pairs,
                    grouped_pairs,
                    target_name,
                    TARGET_SPECS[target_name],
                )
                if summarized is None:
                    continue
                summary, tests, target_metrics, _, _ = summarized
                group_target_summaries[target_name] = summary
                group_target_tests[target_name] = tests
                for metric_name, value in target_metrics.items():
                    metrics[f"{metric_name}_{target_name}_{source_name}_{group_name}"] = value
                source_summary["target_grouped_pearson_by_group"][target_name][group_name] = (
                    summary["pearson_r_grouped"]
                )
                source_summary["target_grouped_pearson_views_by_group"][target_name][group_name] = (
                    summary.get("grouped_views", {})
                )

            primary_target_summary = group_target_summaries.get(PRIMARY_TARGET)
            primary_target_tests = group_target_tests.get(PRIMARY_TARGET)
            if primary_target_summary is not None:
                source_summary["group_summaries"][group_name] = primary_target_summary
                source_summary["group_target_summaries"][group_name] = group_target_summaries
                source_summary["grouped_pearson_by_group"][group_name] = primary_target_summary[
                    "pearson_r_grouped"
                ]
                source_summary["grouped_pearson_views_by_group"][group_name] = (
                    primary_target_summary.get("grouped_views", {})
                )
                source_tests["group_tests"][group_name] = primary_target_tests
                source_tests["group_target_tests"][group_name] = group_target_tests

            horse_race = {
                "grouped": _summarize_prediction_factor_horse_race(
                    grouped_pairs,
                    y_key="accuracy_drop",
                ),
                "sample": _summarize_prediction_factor_horse_race(
                    sample_pairs,
                    y_key="accuracy_drop",
                ),
            }
            source_summary["prediction_factor_horse_races"][group_name] = horse_race
            source_tests["prediction_factor_horse_races"][group_name] = horse_race

            if group_name == primary_group:
                source_summary["primary_group_summary"] = primary_target_summary
                source_summary["primary_prediction_factor_horse_race"] = horse_race
                source_summary["primary_group_target_summaries"] = group_target_summaries
                source_tests["primary_group_tests"] = primary_target_tests
                source_tests["primary_prediction_factor_horse_race"] = horse_race
                source_tests["primary_group_target_tests"] = group_target_tests
                source_tests["hypothesis_supported"] = (
                    primary_target_tests or {}
                ).get("hypothesis_supported", False)
                first_order_tables[source_name] = _build_first_order_table(grouped_pairs)

                if source_name == "image_space" and primary_target_summary is not None:
                    primary_summary = primary_target_summary
                    primary_tests = primary_target_tests
                    primary_horse_race = horse_race

            LOGGER.info(
                "%s/%s: grouped=%d sample=%d primary_r=%s",
                source_name,
                group_name,
                len(grouped_pairs),
                len(sample_pairs),
                (
                    f"{primary_target_summary['pearson_r_grouped']:.4f}"
                    if primary_target_summary is not None
                    else "NA"
                ),
            )
            del sample_pairs, grouped_pairs
            gc.collect()

        summary_payload[source_name] = source_summary
        tests_payload[source_name] = source_tests

    if first_order_tables:
        summary_payload["first_order_overlap_tables"] = first_order_tables

    if primary_summary is not None:
        summary_payload.update(primary_summary)
        summary_payload["prediction_factor_horse_race"] = primary_horse_race or {}
    if primary_tests is not None:
        tests_payload.update(primary_tests)
        tests_payload["prediction_factor_horse_race"] = primary_horse_race or {}
        tests_payload["hypothesis_supported"] = any(
            (tests_payload.get(source_name, {}).get("primary_group_tests") or {}).get(
                "hypothesis_supported",
                False,
            )
            for source_name in ("image_space", "vision_feature_space")
        )

    save_json(summary_payload, exp5_dir / "summary.json")
    save_json(tests_payload, exp5_dir / "hypothesis_tests.json")
    save_json(metrics, exp5_dir / "metrics.json")
    save_json(
        {
            "runner": "scripts/run_exp5_streaming.py",
            "output_dir": str(output_dir),
            "exp1": str(exp1_dir),
            "exp2": str(exp2_dir),
            "exp5": str(exp5_dir),
            "large_scatter_payloads_written": False,
        },
        exp5_dir / "streaming_manifest.json",
    )
    LOGGER.info("Wrote %s", exp5_dir / "summary.json")
    LOGGER.info("Wrote %s", exp5_dir / "hypothesis_tests.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Existing experiment output directory containing exp1/ and exp2/.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_streaming_exp5(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
