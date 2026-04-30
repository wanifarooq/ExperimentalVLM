#!/usr/bin/env python3
"""Offline Exp5 overlap-law diagnostics on existing output directories.

This script does not run model inference and does not modify the main
frequency_alignment pipeline. It reads Exp1/Exp5 artifacts, restricts analysis
to a fixed number of images, and reports stricter vs coarser aggregation views.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PREDICTORS = ("predicted_first_order", "predicted_linear", "predicted_quadratic")
TARGETS = ("loglik_volatility", "loglik_erosion", "accuracy_drop")
DEFAULT_DIRS = (
    "frequency_alignment_outputs_local_200samples_20260428_125813",
    "frequency_alignment_outputs_local_200samples_20260428_131910",
)
EPS = 1e-12


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, scipy-free."""
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    sorted_vals = arr[order]
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _corr(x_values: Sequence[Any], y_values: Sequence[Any], *, spearman: bool = False) -> Dict[str, Any]:
    pairs = [
        (_safe_float(x, float("nan")), _safe_float(y, float("nan")))
        for x, y in zip(x_values, y_values)
    ]
    pairs = [(x, y) for x, y in pairs if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return {"r": None, "n": len(pairs)}
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if spearman:
        x = _rankdata(x)
        y = _rankdata(y)
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return {"r": None, "n": int(len(x)), "degenerate": True}
    return {"r": float(np.corrcoef(x, y)[0, 1]), "n": int(len(x))}


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr))
    if std <= EPS:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def _ols_residual(y: Sequence[Any], controls: Sequence[Sequence[Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    y_arr = np.asarray([_safe_float(v, float("nan")) for v in y], dtype=np.float64)
    columns = [np.ones(len(y_arr), dtype=np.float64)]
    used_controls = 0
    for control in controls:
        col = np.asarray([_safe_float(v, float("nan")) for v in control], dtype=np.float64)
        if len(col) != len(y_arr):
            continue
        if np.any(np.isnan(col)) or float(np.std(col)) <= EPS:
            continue
        columns.append(_zscore(col))
        used_controls += 1
    mask = ~np.isnan(y_arr)
    x = np.column_stack(columns)[mask]
    yy = y_arr[mask]
    if len(yy) < x.shape[1] + 2:
        return np.asarray([], dtype=np.float64), {"skipped": "too_few_rows", "n": int(len(yy))}
    beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    resid = yy - x @ beta
    return resid, {"n": int(len(yy)), "num_controls_used": int(used_controls)}


def _standardized_ols(
    rows: List[Dict[str, Any]],
    *,
    target_key: str,
    predicted_key: str,
    controls: Sequence[str],
    family_fixed_effects: bool = False,
) -> Dict[str, Any]:
    if len(rows) < 5:
        return {"skipped": "too_few_rows", "n": len(rows)}

    y = np.asarray([_safe_float(row.get(target_key), float("nan")) for row in rows], dtype=np.float64)
    pred_raw = np.asarray(
        [math.log1p(max(0.0, _safe_float(row.get(predicted_key), 0.0))) for row in rows],
        dtype=np.float64,
    )
    columns = [np.ones(len(rows), dtype=np.float64), _zscore(pred_raw)]
    names = ["intercept", f"z_log1p_{predicted_key}"]

    for key in controls:
        col = np.asarray([_safe_float(row.get(key), float("nan")) for row in rows], dtype=np.float64)
        if np.any(np.isnan(col)) or float(np.std(col)) <= EPS:
            continue
        columns.append(_zscore(col))
        names.append(key)

    if family_fixed_effects:
        families = sorted({str(row.get("perturbation_family", "unknown")) for row in rows})
        for family in families[1:]:
            columns.append(
                np.asarray(
                    [1.0 if str(row.get("perturbation_family", "unknown")) == family else 0.0 for row in rows],
                    dtype=np.float64,
                )
            )
            names.append(f"FE:{family}")

    x_all = np.column_stack(columns)
    mask = ~np.isnan(y)
    x = x_all[mask]
    yy = _zscore(y[mask])
    if len(yy) < x.shape[1] + 2:
        return {"skipped": "too_few_rows_or_too_many_fe", "n": int(len(yy)), "p": int(x.shape[1])}
    beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    fitted = x @ beta
    ss_res = float(np.sum((yy - fitted) ** 2))
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    return {
        "n": int(len(yy)),
        "p": int(x.shape[1]),
        "target": target_key,
        "predicted": predicted_key,
        "family_fixed_effects": bool(family_fixed_effects),
        "predicted_beta_std": float(beta[names.index(f"z_log1p_{predicted_key}")]),
        "r2": float(1.0 - ss_res / max(ss_tot, EPS)),
        "controls_used": [name for name in names if name not in {"intercept", f"z_log1p_{predicted_key}"} and not name.startswith("FE:")],
        "num_family_fixed_effects": sum(1 for name in names if name.startswith("FE:")),
    }


def _group_rows(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    meta: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    numeric_keys = set(PREDICTORS) | set(TARGETS) | {
        "clean_correct",
        "clean_answer_prob",
        "clean_margin",
        "clean_prediction_entropy",
        "perturbation_prediction_entropy",
        "is_binary",
        "option_hardness_score",
        "prompt_complexity_score",
        "question_complexity_score",
    }
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        meta.setdefault(key, {k: row.get(k) for k in keys})
        for numeric_key in numeric_keys:
            if numeric_key in row:
                buckets[key][numeric_key].append(_safe_float(row.get(numeric_key), 0.0))
    grouped: List[Dict[str, Any]] = []
    for key, values in buckets.items():
        out = dict(meta[key])
        out["n"] = len(next(iter(values.values()))) if values else 0
        for numeric_key, vals in values.items():
            out[numeric_key] = float(np.mean(vals)) if vals else 0.0
        grouped.append(out)
    return grouped


def _softmax_from_loglik(scores: Dict[str, Any]) -> Dict[str, float]:
    if not scores:
        return {}
    labels = list(scores.keys())
    vals = np.asarray([_safe_float(scores[label], -1e9) for label in labels], dtype=np.float64)
    vals = vals - float(np.max(vals))
    exp_vals = np.exp(vals)
    probs = exp_vals / max(float(np.sum(exp_vals)), EPS)
    return {label: float(prob) for label, prob in zip(labels, probs)}


def _build_exp1_control_maps(exp1_dir: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    level_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    perturbation_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    per_sample_path = exp1_dir / "per_sample.jsonl"
    if not per_sample_path.exists():
        return level_map, perturbation_map
    for record in _load_jsonl(per_sample_path):
        image_id = str(record.get("image_id"))
        for level, level_data in (record.get("levels") or {}).items():
            clean = level_data.get("clean", {}) or {}
            scores = clean.get("scores", {}) or {}
            probs = _softmax_from_loglik(scores)
            answer = str(level_data.get("answer_label", ""))
            sorted_scores = sorted((_safe_float(v, -1e9) for v in scores.values()), reverse=True)
            margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0
            level_map[(image_id, str(level))] = {
                "clean_prediction_entropy": _safe_float(
                    clean.get("prediction_entropy", level_data.get("prediction_entropy")), 0.0
                ),
                "clean_answer_prob": probs.get(answer, 0.0),
                "clean_answer_loglik": _safe_float(scores.get(answer), 0.0),
                "clean_margin": float(margin),
            }
            for perturbation in level_data.get("perturbations", []) or []:
                perturbation_map[(image_id, str(level), str(perturbation.get("name")))] = {
                    "perturbation_prediction_entropy": _safe_float(
                        perturbation.get("prediction_entropy"), 0.0
                    ),
                    "severity": perturbation.get("severity"),
                }
    return level_map, perturbation_map


def _load_exp5_sample_rows(output_dir: Path, *, max_images: int) -> List[Dict[str, Any]]:
    sample_path = output_dir / "exp5" / "sample_scatter_data.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {sample_path}")
    rows = _load_json(sample_path)
    image_ids = sorted({str(row.get("image_id")) for row in rows})
    keep_images = set(image_ids[:max_images])
    rows = [dict(row) for row in rows if str(row.get("image_id")) in keep_images]

    level_controls, perturbation_controls = _build_exp1_control_maps(output_dir / "exp1")
    for row in rows:
        image_id = str(row.get("image_id"))
        level = str(row.get("level"))
        perturbation = str(row.get("perturbation"))
        row.update(level_controls.get((image_id, level), {}))
        row.update(perturbation_controls.get((image_id, level, perturbation), {}))
    return rows


def _correlation_table(rows: List[Dict[str, Any]], *, targets: Sequence[str] = TARGETS) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for target in targets:
        out[target] = {}
        for pred in PREDICTORS:
            out[target][pred] = {
                "pearson": _corr([row.get(pred) for row in rows], [row.get(target) for row in rows]),
                "spearman": _corr(
                    [row.get(pred) for row in rows],
                    [row.get(target) for row in rows],
                    spearman=True,
                ),
            }
    return out


def _within_family(rows: List[Dict[str, Any]], *, target: str, pred: str) -> Dict[str, Any]:
    out = {}
    for family in sorted({str(row.get("perturbation_family", "unknown")) for row in rows}):
        fam_rows = [row for row in rows if str(row.get("perturbation_family", "unknown")) == family]
        if len(fam_rows) < 10:
            continue
        out[family] = {
            "n": len(fam_rows),
            "pearson": _corr([row.get(pred) for row in fam_rows], [row.get(target) for row in fam_rows]),
            "spearman": _corr(
                [row.get(pred) for row in fam_rows],
                [row.get(target) for row in fam_rows],
                spearman=True,
            ),
        }
    return out


def _residualized_checks(rows: List[Dict[str, Any]], *, target: str, pred: str) -> Dict[str, Any]:
    controls = [
        "clean_answer_prob",
        "clean_margin",
        "clean_prediction_entropy",
        "perturbation_prediction_entropy",
        "is_binary",
        "option_hardness_score",
        "prompt_complexity_score",
    ]
    residuals, info = _ols_residual(
        [row.get(target) for row in rows],
        [[row.get(control) for row in rows] for control in controls],
    )
    if residuals.size == 0:
        return info
    return {
        **info,
        "target": target,
        "predicted": pred,
        "controls_requested": controls,
        "pearson_predicted_vs_target_residual": _corr(
            [row.get(pred) for row in rows[: len(residuals)]],
            residuals,
        ),
        "spearman_predicted_vs_target_residual": _corr(
            [row.get(pred) for row in rows[: len(residuals)]],
            residuals,
            spearman=True,
        ),
    }


def analyze_output_dir(output_dir: Path, *, max_images: int) -> Dict[str, Any]:
    cfg_path = output_dir / "config_snapshot.json"
    cfg = _load_json(cfg_path) if cfg_path.exists() else {}
    rows = _load_exp5_sample_rows(output_dir, max_images=max_images)
    grouped_image_level_family = _group_rows(rows, ("image_id", "level", "perturbation_family"))
    grouped_level_family = _group_rows(rows, ("level", "perturbation_family"))
    grouped_level_exact = _group_rows(rows, ("level", "perturbation_family", "perturbation"))

    controls = [
        "clean_answer_prob",
        "clean_margin",
        "clean_prediction_entropy",
        "perturbation_prediction_entropy",
        "is_binary",
        "option_hardness_score",
        "prompt_complexity_score",
    ]

    return {
        "output_dir": str(output_dir),
        "suppress_dc": cfg.get("analysis", {}).get("suppress_dc"),
        "num_bands": cfg.get("analysis", {}).get("num_bands"),
        "num_bands_resolution": cfg.get("analysis", {}).get("num_bands_resolution"),
        "max_images_requested": max_images,
        "num_images_used": len({str(row.get("image_id")) for row in rows}),
        "num_sample_rows": len(rows),
        "num_image_level_family_rows": len(grouped_image_level_family),
        "num_level_family_rows": len(grouped_level_family),
        "num_level_exact_perturbation_rows": len(grouped_level_exact),
        "sample_exact_correlations": _correlation_table(rows),
        "image_level_family_correlations": _correlation_table(grouped_image_level_family),
        "level_family_macro_correlations": _correlation_table(grouped_level_family),
        "level_exact_perturbation_macro_correlations": _correlation_table(grouped_level_exact),
        "within_family_loglik_volatility_first_order": _within_family(
            rows,
            target="loglik_volatility",
            pred="predicted_first_order",
        ),
        "within_family_loglik_erosion_first_order": _within_family(
            rows,
            target="loglik_erosion",
            pred="predicted_first_order",
        ),
        "residualized_loglik_volatility_first_order": _residualized_checks(
            rows,
            target="loglik_volatility",
            pred="predicted_first_order",
        ),
        "family_fe_horse_race_loglik_volatility": {
            pred: _standardized_ols(
                rows,
                target_key="loglik_volatility",
                predicted_key=pred,
                controls=controls,
                family_fixed_effects=True,
            )
            for pred in PREDICTORS
        },
        "no_family_fe_horse_race_loglik_volatility": {
            pred: _standardized_ols(
                rows,
                target_key="loglik_volatility",
                predicted_key=pred,
                controls=controls,
                family_fixed_effects=False,
            )
            for pred in PREDICTORS
        },
    }


def _print_summary(result: Dict[str, Any]) -> None:
    print(f"\n=== {result['output_dir']} ===")
    print(
        f"suppress_dc={result.get('suppress_dc')} "
        f"images={result['num_images_used']} sample_rows={result['num_sample_rows']}"
    )
    for block_name in (
        "sample_exact_correlations",
        "image_level_family_correlations",
        "level_family_macro_correlations",
        "level_exact_perturbation_macro_correlations",
    ):
        block = result[block_name]["loglik_volatility"]["predicted_first_order"]
        pearson = block["pearson"]
        spearman = block["spearman"]
        print(
            f"{block_name}: first_order vs volatility "
            f"Pearson={pearson.get('r')} n={pearson.get('n')} "
            f"Spearman={spearman.get('r')} n={spearman.get('n')}"
        )
    resid = result["residualized_loglik_volatility_first_order"]
    print(
        "residualized volatility: "
        f"Pearson={resid.get('pearson_predicted_vs_target_residual', {}).get('r')} "
        f"n={resid.get('pearson_predicted_vs_target_residual', {}).get('n')}"
    )
    fe = result["family_fe_horse_race_loglik_volatility"]["predicted_first_order"]
    print(
        "family-FE horse race: "
        f"beta_std={fe.get('predicted_beta_std')} r2={fe.get('r2')} "
        f"n={fe.get('n')} controls={fe.get('controls_used')}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        default=[d for d in DEFAULT_DIRS if Path(d).exists()],
        help="Existing frequency_alignment output directories to inspect.",
    )
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("exp5_overlap_diagnostics_50.json"),
        help="Where to write the diagnostic JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output_dirs:
        raise SystemExit("No output directories provided/found.")
    results = [
        analyze_output_dir(Path(output_dir), max_images=args.max_images)
        for output_dir in args.output_dirs
    ]
    payload = {
        "description": (
            "Offline Exp5 overlap-law diagnostics. No model inference. "
            "Rows are restricted to the first N image IDs per output directory."
        ),
        "max_images": args.max_images,
        "predictors": list(PREDICTORS),
        "targets": list(TARGETS),
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    for result in results:
        _print_summary(result)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
