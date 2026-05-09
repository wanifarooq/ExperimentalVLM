#!/usr/bin/env python3
"""Sample-level Exp 5 overlap diagnostics.

This script is intentionally outside the production pipeline. It recomputes
radial and 2D overlap scores for individual perturbation rows:

    image_id x level x perturbation_name x severity

and reports correlations both pooled over all levels and separately within each
level. No perturbation-family averaging is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


LEVEL_ORDER = [
    "L1_COARSE",
    "L2_MEDIUM",
    "L3_FINE",
    "L4_VERY_FINE",
    "L5_WORDY_SIMPLETON",
    "L6_WORDY_MEDIUM",
    "L7_WORDY_FINE",
    "L8_WORDY_VERY_FINE",
]
GROUP_ORDER = ["overall", "early", "mid", "late", "last_1", "last_2", "last_3", "last_4"]
DOMAIN_SPECS = {
    "image_space": {
        "delta_radial_raw": "delta_f",
        "delta_radial_relative": "delta_f_relative",
        "delta_2d_dir": "delta_f_2d",
        "delta_2d_file": "delta_f_2d_file",
        "delta_2d_raw": "delta_power_2d",
        "delta_2d_relative": "delta_power_2d_relative",
    },
    "vision_feature_space": {
        "delta_radial_raw": "delta_f_vision",
        "delta_radial_relative": "delta_f_vision_relative",
        "delta_2d_dir": "delta_f_vision_2d",
        "delta_2d_file": "delta_f_vision_2d_file",
        "delta_2d_raw": "delta_power_vision_2d",
        "delta_2d_relative": "delta_power_vision_2d_relative",
    },
}
VARIANT_ORDER = [
    "radial_first_order",
    "two_d_first_order",
    "two_d_linear",
    "two_d_quadratic",
    "two_d_cosine",
]
EPS = 1e-12


def _parse_csv(value: Optional[str], default: Sequence[str]) -> List[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _target_value(level_data: Dict[str, Any], perturbation: Dict[str, Any], target_key: str) -> Optional[float]:
    if target_key == "loglik_volatility":
        drift = _as_float(perturbation.get("loglik_drift"))
        return abs(drift) if drift is not None else None
    if target_key == "loglik_erosion":
        drift = _as_float(perturbation.get("loglik_drift"))
        return max(drift, 0.0) if drift is not None else None
    if target_key == "loglik_recovery":
        drift = _as_float(perturbation.get("loglik_drift"))
        return min(drift, 0.0) if drift is not None else None
    if target_key == "loglik_drift":
        return _as_float(perturbation.get("loglik_drift"))
    if target_key == "accuracy_drop":
        value = _as_float(perturbation.get("accuracy_drop"))
        if value is not None:
            return value
        clean_correct = bool(level_data.get("clean", {}).get("correct", False))
        perturbed_correct = bool(perturbation.get("correct", False))
        return 1.0 if clean_correct and not perturbed_correct else 0.0
    if target_key in {"net_drop", "net_change"}:
        clean_correct = bool(level_data.get("clean", {}).get("correct", False))
        perturbed_correct = bool(perturbation.get("correct", False))
        ci = 1.0 if clean_correct and not perturbed_correct else 0.0
        ic = 1.0 if (not clean_correct) and perturbed_correct else 0.0
        return ci - ic
    return _as_float(perturbation.get(target_key))


def _rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    sorted_values = arr[order]
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3 or np.std(x_arr) <= EPS or np.std(y_arr) <= EPS:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3:
        return None
    return _pearson(_rankdata(x_arr), _rankdata(y_arr))


def _load_npz_array(path: Path, key: str, cache: Dict[Tuple[str, str], Optional[np.ndarray]]) -> Optional[np.ndarray]:
    cache_key = (str(path), key)
    if cache_key in cache:
        return cache[cache_key]
    if not path.exists():
        cache[cache_key] = None
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            value = np.asarray(data[key], dtype=np.float64)
    except Exception:
        value = None
    cache[cache_key] = value
    return value


def _resize_power_preserve_sum(power_2d: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(power_2d, dtype=np.float64)
    if arr.shape == target_shape:
        return arr
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros(target_shape, dtype=np.float64)
    source_sum = float(np.sum(arr))
    image = Image.fromarray(arr.astype(np.float32))
    resized = np.asarray(
        image.resize((int(target_shape[1]), int(target_shape[0])), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    resized_sum = float(np.sum(resized))
    if resized_sum > EPS:
        resized *= source_sum / resized_sum
    return resized


def _load_aligned_npz_array(
    path: Path,
    key: str,
    target_shape: Tuple[int, int],
    npz_cache: Dict[Tuple[str, str], Optional[np.ndarray]],
    aligned_cache: Dict[Tuple[str, str, Tuple[int, int]], Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    cache_key = (str(path), key, target_shape)
    if cache_key in aligned_cache:
        return aligned_cache[cache_key]
    arr = _load_npz_array(path, key, npz_cache)
    if arr is None:
        aligned_cache[cache_key] = None
        return None
    aligned = _resize_power_preserve_sum(arr, target_shape)
    aligned_cache[cache_key] = aligned
    return aligned


def _load_filter_bank(run_dir: Path, groups: Sequence[str]) -> Dict[Tuple[str, str, str], np.ndarray]:
    records = _load_json(run_dir / "exp2" / "power_spectra.json")
    bank: Dict[Tuple[str, str, str], np.ndarray] = {}
    for record in records:
        image_id = str(record.get("image_id"))
        for level_key, level_data in (record.get("levels") or {}).items():
            if "overall" in groups and level_data.get("W_t"):
                bank[(image_id, level_key, "overall")] = np.asarray(level_data["W_t"], dtype=np.float64)
            for group_name in groups:
                if group_name == "overall":
                    continue
                group_data = (level_data.get("layer_groups") or {}).get(group_name) or {}
                if group_data.get("W_t"):
                    bank[(image_id, level_key, group_name)] = np.asarray(group_data["W_t"], dtype=np.float64)
    return bank


def _first_order(w: np.ndarray, d: np.ndarray) -> Optional[float]:
    w_arr = np.asarray(w, dtype=np.float64).reshape(-1)
    d_arr = np.asarray(d, dtype=np.float64).reshape(-1)
    usable = min(w_arr.size, d_arr.size)
    if usable <= 0:
        return None
    return float(np.sum(w_arr[:usable] * d_arr[:usable]))


def _two_d_scores(w_2d: np.ndarray, d_2d: np.ndarray) -> Dict[str, float]:
    w = np.asarray(w_2d, dtype=np.float64)
    d = np.asarray(d_2d, dtype=np.float64)
    if w.ndim != 2 or d.ndim != 2 or w.size == 0 or d.size == 0:
        return {}
    if w.shape != d.shape:
        d = _resize_power_preserve_sum(d, (int(w.shape[0]), int(w.shape[1])))
    dot = float(np.sum(w * d))
    denom = float(np.linalg.norm(w) * np.linalg.norm(d))
    return {
        "two_d_first_order": dot,
        "two_d_linear": float(np.sum(w * d ** 2)),
        "two_d_quadratic": float(np.sum((w ** 2) * (d ** 2))),
        "two_d_cosine": float(dot / denom) if denom > EPS else 0.0,
    }


def _accumulate(
    buckets: Dict[Tuple[str, str, str, str, str], Dict[str, List[float]]],
    *,
    aggregation_level: str,
    domain: str,
    group_name: str,
    variant: str,
    level_key: str,
    score: Optional[float],
    target: Optional[float],
) -> None:
    if score is None or target is None:
        return
    if not math.isfinite(score) or not math.isfinite(target):
        return
    for level_value in ("ALL", level_key):
        key = (aggregation_level if level_value == "ALL" else "sample_by_level", domain, group_name, variant, level_value)
        buckets[key]["score"].append(float(score))
        buckets[key]["target"].append(float(target))


def _compute_sample_summaries(
    run_dir: Path,
    *,
    domains: Sequence[str],
    groups: Sequence[str],
    variants: Sequence[str],
    target_key: str,
    normalization: str,
) -> List[Dict[str, Any]]:
    exp1_path = run_dir / "exp1" / "per_sample.jsonl"
    filters_2d_dir = run_dir / "exp2" / "filters_2d"
    radial_filters = _load_filter_bank(run_dir, groups)
    npz_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    aligned_cache: Dict[Tuple[str, str, Tuple[int, int]], Optional[np.ndarray]] = {}
    buckets: Dict[Tuple[str, str, str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for record in _iter_jsonl(exp1_path):
        image_id = str(record.get("image_id"))
        for level_key, level_data in (record.get("levels") or {}).items():
            if level_key not in LEVEL_ORDER:
                continue
            for perturbation in level_data.get("perturbations", []):
                target = _target_value(level_data, perturbation, target_key)
                if target is None:
                    continue
                for domain in domains:
                    spec = DOMAIN_SPECS[domain]
                    radial_delta_key = str(spec[f"delta_radial_{normalization}"])
                    radial_delta = perturbation.get(radial_delta_key)
                    delta_2d_file = perturbation.get(str(spec["delta_2d_file"]))
                    delta_2d_dir = run_dir / "exp1" / str(spec["delta_2d_dir"])
                    delta_2d_key = str(spec[f"delta_2d_{normalization}"])
                    delta_2d_path = delta_2d_dir / str(delta_2d_file) if delta_2d_file else None
                    for group_name in groups:
                        if "radial_first_order" in variants and radial_delta:
                            radial_w = radial_filters.get((image_id, level_key, group_name))
                            if radial_w is not None:
                                _accumulate(
                                    buckets,
                                    aggregation_level="sample_all_levels",
                                    domain=domain,
                                    group_name=group_name,
                                    variant="radial_first_order",
                                    level_key=level_key,
                                    score=_first_order(radial_w, np.asarray(radial_delta, dtype=np.float64)),
                                    target=target,
                                )

                        requested_2d = [variant for variant in variants if variant.startswith("two_d_")]
                        if not requested_2d or delta_2d_path is None:
                            continue
                        w_2d = _load_npz_array(
                            filters_2d_dir / f"{image_id}_{level_key}_{group_name}.npz",
                            "W_t_2d",
                            npz_cache,
                        )
                        if w_2d is None:
                            continue
                        d_2d = _load_aligned_npz_array(
                            delta_2d_path,
                            delta_2d_key,
                            (int(w_2d.shape[0]), int(w_2d.shape[1])),
                            npz_cache,
                            aligned_cache,
                        )
                        if d_2d is None:
                            continue
                        two_d = _two_d_scores(w_2d, d_2d)
                        for variant in requested_2d:
                            _accumulate(
                                buckets,
                                aggregation_level="sample_all_levels",
                                domain=domain,
                                group_name=group_name,
                                variant=variant,
                                level_key=level_key,
                                score=two_d.get(variant),
                                target=target,
                            )

    rows: List[Dict[str, Any]] = []
    for key, values in sorted(buckets.items()):
        aggregation_level, domain, group_name, variant, level_key = key
        scores = values["score"]
        targets = values["target"]
        rows.append(
            {
                "aggregation_level": aggregation_level,
                "matched_unit": "image_id x level x perturbation_name x severity",
                "domain": domain,
                "group": group_name,
                "variant": variant,
                "target": target_key,
                "normalization": normalization,
                "level": "" if level_key == "ALL" else level_key,
                "n": len(scores),
                "pearson_r": _pearson(scores, targets),
                "spearman_rho": _spearman(scores, targets),
                "score_mean": float(np.mean(scores)) if scores else None,
                "score_std": float(np.std(scores)) if scores else None,
                "target_mean": float(np.mean(targets)) if targets else None,
                "target_std": float(np.std(targets)) if targets else None,
            }
        )
    return rows


def _write_outputs(out_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "overlap_sample_level_summary.json").open("w") as handle:
        json.dump(
            {
                "notes": [
                    "Rows are individual perturbation instances: image_id x level x perturbation_name x severity.",
                    "sample_all_levels pools all levels; sample_by_level reports the same sample rows separately inside each level.",
                    "No perturbation-family averaging is applied.",
                ],
                "correlations": list(rows),
            },
            handle,
            indent=2,
        )
    fieldnames = [
        "aggregation_level",
        "matched_unit",
        "domain",
        "group",
        "variant",
        "target",
        "normalization",
        "level",
        "n",
        "pearson_r",
        "spearman_rho",
        "score_mean",
        "score_std",
        "target_mean",
        "target_std",
    ]
    with (out_dir / "overlap_sample_level_correlations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _plot_levelwise(
    out_dir: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    target_key: str,
    domain: str,
    group_name: str,
    variant: str,
) -> None:
    selected = [
        row for row in rows
        if row["aggregation_level"] == "sample_by_level"
        and row["domain"] == domain
        and row["group"] == group_name
        and row["variant"] == variant
        and row["pearson_r"] is not None
    ]
    if not selected:
        return
    by_level = {row["level"]: row for row in selected}
    levels = [level for level in LEVEL_ORDER if level in by_level]
    values = [float(by_level[level]["pearson_r"]) for level in levels]
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    colors = ["#2563eb" if level.startswith(("L1", "L2", "L3", "L4")) else "#16a34a" for level in levels]
    ax.bar(np.arange(len(levels)), values, color=colors, alpha=0.86)
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([level.replace("_", " ") for level in levels], rotation=25, ha="right")
    ax.set_ylabel(f"Pearson r vs {target_key}")
    ax.set_title(f"Sample-level overlap correlation within each level: {domain}, {group_name}, {variant}")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Rows are image_id x level x perturbation_name x severity; no perturbation-family averaging.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out_dir / f"overlap_sample_level_{domain}_{group_name}_{variant}_{target_key}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute sample-level overlap correlations over image x level x perturbation x severity rows."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to <output-dir>/plots_overlap_sample_level.",
    )
    parser.add_argument(
        "--domains",
        default="image_space,vision_feature_space",
        help="Comma-separated domains.",
    )
    parser.add_argument(
        "--groups",
        default=",".join(GROUP_ORDER),
        help="Comma-separated attention groups.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_ORDER),
        help="Comma-separated overlap variants.",
    )
    parser.add_argument("--target", default="loglik_volatility")
    parser.add_argument("--normalization", default="relative", choices=["relative", "raw"])
    parser.add_argument("--plot-domain", default="image_space", choices=sorted(DOMAIN_SPECS))
    parser.add_argument("--plot-group", default="last_2")
    parser.add_argument("--plot-variant", default="two_d_first_order")
    args = parser.parse_args()

    domains = _parse_csv(args.domains, list(DOMAIN_SPECS))
    groups = _parse_csv(args.groups, GROUP_ORDER)
    variants = _parse_csv(args.variants, VARIANT_ORDER)
    unknown_domains = sorted(set(domains) - set(DOMAIN_SPECS))
    unknown_variants = sorted(set(variants) - set(VARIANT_ORDER))
    if unknown_domains:
        raise ValueError(f"Unknown domains: {unknown_domains}")
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")

    out_dir = args.plots_dir or (args.output_dir / "plots_overlap_sample_level")
    rows = _compute_sample_summaries(
        args.output_dir,
        domains=domains,
        groups=groups,
        variants=variants,
        target_key=args.target,
        normalization=args.normalization,
    )
    _write_outputs(out_dir, rows)
    _plot_levelwise(
        out_dir,
        rows,
        target_key=args.target,
        domain=args.plot_domain,
        group_name=args.plot_group,
        variant=args.plot_variant,
    )

    print(f"Wrote sample-level overlap diagnostics to {out_dir}")
    print(f"CSV: {out_dir / 'overlap_sample_level_correlations.csv'}")
    print("Top sample_by_level Pearson r:")
    level_rows = [row for row in rows if row["aggregation_level"] == "sample_by_level" and row["pearson_r"] is not None]
    for row in sorted(level_rows, key=lambda item: float(item["pearson_r"]), reverse=True)[:16]:
        print(
            f"  {row['domain']} {row['group']} {row['variant']} {row['level']}: "
            f"r={float(row['pearson_r']):.4f} rho={float(row['spearman_rho']):.4f} n={row['n']}"
        )


if __name__ == "__main__":
    main()
