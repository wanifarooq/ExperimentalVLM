#!/usr/bin/env python3
"""Instance-local broadband-centering diagnostics for Exp 5 overlap laws.

This script is intentionally outside the production pipeline. It recomputes
first-order radial and 2D overlap scores from saved Exp 1/Exp 2 artifacts, then
tests whether removing the per-instance broadband floor improves correlation
with observed model degradation.
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
CENTERING_ORDER = ["raw", "delta_centered", "filter_centered", "both_centered"]
SPACE_ORDER = ["radial_first_order", "two_d_first_order"]
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


def _safe_name(value: Any) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned.strip("_") or "unknown"


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _perturbation_family_name(name: Any) -> str:
    text = str(name or "unknown")
    text = text.split("|sev", 1)[0]
    text = text.split("(", 1)[0]
    return text.strip() or "unknown"


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
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
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
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3:
        return None
    return _pearson(_rankdata(x_arr), _rankdata(y_arr))


def _centered_vectors(w: np.ndarray, d: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    w_arr = np.asarray(w, dtype=np.float64).reshape(-1)
    d_arr = np.asarray(d, dtype=np.float64).reshape(-1)
    usable = min(w_arr.size, d_arr.size)
    w_arr = w_arr[:usable]
    d_arr = d_arr[:usable]
    if mode in {"filter_centered", "both_centered"}:
        w_arr = w_arr - float(np.mean(w_arr))
    if mode in {"delta_centered", "both_centered"}:
        d_arr = d_arr - float(np.mean(d_arr))
    return w_arr, d_arr


def _first_order_overlap(w: np.ndarray, d: np.ndarray, mode: str) -> Optional[float]:
    w_arr, d_arr = _centered_vectors(w, d, mode)
    if w_arr.size == 0 or d_arr.size == 0:
        return None
    return float(np.sum(w_arr * d_arr))


def _first_order_overlap_all(
    w: np.ndarray,
    d: np.ndarray,
    modes: Sequence[str],
) -> Dict[str, Optional[float]]:
    return {mode: _first_order_overlap(w, d, mode) for mode in modes}


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


def _load_filter_bank(run_dir: Path, groups: Sequence[str]) -> Dict[Tuple[str, str, str], np.ndarray]:
    power_path = run_dir / "exp2" / "power_spectra.json"
    if not power_path.exists():
        raise FileNotFoundError(f"Missing {power_path}")
    records = _load_json(power_path)
    bank: Dict[Tuple[str, str, str], np.ndarray] = {}
    for record in records:
        image_id = str(record.get("image_id"))
        for level_key, level_data in (record.get("levels") or {}).items():
            if "overall" in groups and level_data.get("W_t"):
                bank[(image_id, level_key, "overall")] = np.asarray(level_data["W_t"], dtype=np.float64)
            layer_groups = level_data.get("layer_groups") or {}
            for group_name in groups:
                if group_name == "overall":
                    continue
                group_data = layer_groups.get(group_name) or {}
                if group_data.get("W_t"):
                    bank[(image_id, level_key, group_name)] = np.asarray(group_data["W_t"], dtype=np.float64)
    return bank


def _accumulate(
    bucket: Dict[str, List[float]],
    score: Optional[float],
    target: Optional[float],
) -> None:
    if score is None or target is None:
        return
    if not math.isfinite(score) or not math.isfinite(target):
        return
    bucket["score"].append(float(score))
    bucket["target"].append(float(target))


def _compute_rows(
    run_dir: Path,
    *,
    domains: Sequence[str],
    groups: Sequence[str],
    centering_modes: Sequence[str],
    target_key: str,
    normalization: str,
) -> Dict[str, Dict[Tuple[str, ...], Dict[str, List[float]]]]:
    exp1_path = run_dir / "exp1" / "per_sample.jsonl"
    filters_2d_dir = run_dir / "exp2" / "filters_2d"
    if not exp1_path.exists():
        raise FileNotFoundError(f"Missing {exp1_path}")
    if not filters_2d_dir.exists():
        raise FileNotFoundError(f"Missing {filters_2d_dir}")

    radial_filters = _load_filter_bank(run_dir, groups)
    npz_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    aligned_cache: Dict[Tuple[str, str, Tuple[int, int]], Optional[np.ndarray]] = {}
    sample_rows: Dict[Tuple[str, ...], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    grouped_cells: Dict[Tuple[str, ...], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for record in _iter_jsonl(exp1_path):
        image_id = str(record.get("image_id"))
        for level_key, level_data in (record.get("levels") or {}).items():
            for perturbation in level_data.get("perturbations", []):
                target = _target_value(level_data, perturbation, target_key)
                if target is None:
                    continue
                perturbation_name = str(perturbation.get("name", "unknown"))
                family = _perturbation_family_name(perturbation_name)
                for domain in domains:
                    spec = DOMAIN_SPECS[domain]
                    radial_delta_key = str(spec[f"delta_radial_{normalization}"])
                    radial_delta = perturbation.get(radial_delta_key)
                    delta_2d_file = perturbation.get(str(spec["delta_2d_file"]))
                    delta_2d_dir = run_dir / "exp1" / str(spec["delta_2d_dir"])
                    delta_2d_key = str(spec[f"delta_2d_{normalization}"])
                    delta_2d_path = delta_2d_dir / str(delta_2d_file) if delta_2d_file else None

                    for group_name in groups:
                        radial_w = radial_filters.get((image_id, level_key, group_name))
                        w_2d = _load_npz_array(
                            filters_2d_dir / f"{image_id}_{level_key}_{group_name}.npz",
                            "W_t_2d",
                            npz_cache,
                        )
                        if radial_w is None and w_2d is None:
                            continue

                        if radial_w is not None and radial_delta:
                            radial_scores = _first_order_overlap_all(
                                radial_w,
                                np.asarray(radial_delta, dtype=np.float64),
                                centering_modes,
                            )
                            for centering, score in radial_scores.items():
                                key = (domain, group_name, "radial_first_order", centering)
                                _accumulate(sample_rows[key], score, target)
                                cell_key = key + (image_id, level_key, family)
                                _accumulate(grouped_cells[cell_key], score, target)

                        if w_2d is not None and delta_2d_path is not None:
                            aligned_delta = _load_aligned_npz_array(
                                delta_2d_path,
                                delta_2d_key,
                                (int(w_2d.shape[0]), int(w_2d.shape[1])),
                                npz_cache,
                                aligned_cache,
                            )
                            if aligned_delta is not None:
                                two_d_scores = _first_order_overlap_all(
                                    w_2d,
                                    aligned_delta,
                                    centering_modes,
                                )
                            else:
                                two_d_scores = {}
                            for centering, score in two_d_scores.items():
                                key = (domain, group_name, "two_d_first_order", centering)
                                _accumulate(sample_rows[key], score, target)
                                cell_key = key + (image_id, level_key, family)
                                _accumulate(grouped_cells[cell_key], score, target)

    grouped_rows: Dict[Tuple[str, ...], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for cell_key, values in grouped_cells.items():
        if not values["score"] or not values["target"]:
            continue
        key = cell_key[:4]
        grouped_rows[key]["score"].append(float(np.mean(values["score"])))
        grouped_rows[key]["target"].append(float(np.mean(values["target"])))
    return {"sample": sample_rows, "grouped_image_level_family": grouped_rows}


def _summarize_rows(
    rows_by_level: Dict[str, Dict[Tuple[str, ...], Dict[str, List[float]]]],
    *,
    target_key: str,
    normalization: str,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for aggregation_level, rows in rows_by_level.items():
        for key, values in sorted(rows.items()):
            domain, group_name, overlap_space, centering = key
            scores = values["score"]
            targets = values["target"]
            summaries.append(
                {
                    "aggregation_level": aggregation_level,
                    "domain": domain,
                    "group": group_name,
                    "overlap_space": overlap_space,
                    "centering": centering,
                    "target": target_key,
                    "normalization": normalization,
                    "n": len(scores),
                    "pearson_r": _pearson(scores, targets),
                    "spearman_rho": _spearman(scores, targets),
                    "score_mean": float(np.mean(scores)) if scores else None,
                    "score_std": float(np.std(scores)) if scores else None,
                    "target_mean": float(np.mean(targets)) if targets else None,
                    "target_std": float(np.std(targets)) if targets else None,
                }
            )
    return summaries


def _write_outputs(out_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "overlap_broadband_centering_summary.json").open("w") as handle:
        json.dump(
            {
                "notes": [
                    "Centering is instance-local: means are computed separately for each W_t and DeltaF row.",
                    "For a dot product, delta_centered, filter_centered, and both_centered are algebraically identical when centering by each vector's own mean.",
                    "Grouped rows average per-row scores within image_id x level x perturbation_family cells.",
                ],
                "correlations": list(summaries),
            },
            handle,
            indent=2,
        )
    with (out_dir / "overlap_broadband_centering_correlations.csv").open("w", newline="") as handle:
        fieldnames = [
            "aggregation_level",
            "domain",
            "group",
            "overlap_space",
            "centering",
            "target",
            "normalization",
            "n",
            "pearson_r",
            "spearman_rho",
            "score_mean",
            "score_std",
            "target_mean",
            "target_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _plot_grouped_bars(out_dir: Path, summaries: Sequence[Dict[str, Any]], target_key: str) -> None:
    rows = [
        row
        for row in summaries
        if row["aggregation_level"] == "grouped_image_level_family"
        and row["group"] == "late"
        and row["pearson_r"] is not None
    ]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True)
    for ax, domain in zip(axes, ["image_space", "vision_feature_space"]):
        domain_rows = [row for row in rows if row["domain"] == domain]
        labels: List[str] = []
        values: List[float] = []
        colors: List[str] = []
        for overlap_space in SPACE_ORDER:
            for centering in CENTERING_ORDER:
                match = next(
                    (
                        row
                        for row in domain_rows
                        if row["overlap_space"] == overlap_space and row["centering"] == centering
                    ),
                    None,
                )
                if match is None:
                    continue
                labels.append(f"{overlap_space.replace('_first_order', '')}\n{centering}")
                values.append(float(match["pearson_r"]))
                colors.append("#2563eb" if overlap_space == "radial_first_order" else "#16a34a")
        x = np.arange(len(values), dtype=np.float64)
        ax.bar(x, values, color=colors, alpha=0.84)
        ax.axhline(0.0, color="#111827", linewidth=0.9)
        ax.set_title(domain.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(f"Pearson r vs {target_key}")
    fig.suptitle("Late-group overlap law after instance-local broadband centering", fontsize=14)
    fig.text(
        0.01,
        0.01,
        "Centering is per row only: W_t - mean(W_t), DeltaF - mean(DeltaF). "
        "Grouped rows are image_id x level x perturbation_family means.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_dir / "overlap_broadband_centering_late_grouped_pearson.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose instance-local broadband-centering variants for Exp5 overlap laws."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Frequency-alignment run output directory containing exp1/ and exp2/.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory. Defaults to "
            "<output-dir>/plots_overlap_broadband_centering."
        ),
    )
    parser.add_argument(
        "--domains",
        default="image_space,vision_feature_space",
        help="Comma-separated domains: image_space,vision_feature_space.",
    )
    parser.add_argument(
        "--groups",
        default=",".join(GROUP_ORDER),
        help="Comma-separated attention groups.",
    )
    parser.add_argument(
        "--centering",
        default="raw,delta_centered,filter_centered,both_centered",
        help="Comma-separated centering modes.",
    )
    parser.add_argument(
        "--target",
        default="loglik_volatility",
        help="Observed target, e.g. loglik_volatility or accuracy_drop.",
    )
    parser.add_argument(
        "--normalization",
        default="relative",
        choices=["relative", "raw"],
        help="Use raw or energy-relative perturbation spectra.",
    )
    args = parser.parse_args()

    run_dir = args.output_dir
    out_dir = args.plots_dir or (run_dir / "plots_overlap_broadband_centering")
    domains = _parse_csv(args.domains, list(DOMAIN_SPECS))
    groups = _parse_csv(args.groups, GROUP_ORDER)
    centering_modes = _parse_csv(args.centering, CENTERING_ORDER)
    unknown_domains = sorted(set(domains) - set(DOMAIN_SPECS))
    unknown_centering = sorted(set(centering_modes) - set(CENTERING_ORDER))
    if unknown_domains:
        raise ValueError(f"Unknown domains: {unknown_domains}")
    if unknown_centering:
        raise ValueError(f"Unknown centering modes: {unknown_centering}")

    rows_by_level = _compute_rows(
        run_dir,
        domains=domains,
        groups=groups,
        centering_modes=centering_modes,
        target_key=args.target,
        normalization=args.normalization,
    )
    summaries = _summarize_rows(
        rows_by_level,
        target_key=args.target,
        normalization=args.normalization,
    )
    _write_outputs(out_dir, summaries)
    _plot_grouped_bars(out_dir, summaries, args.target)

    print(f"Wrote broadband-centering diagnostics to {out_dir}")
    print(f"CSV: {out_dir / 'overlap_broadband_centering_correlations.csv'}")
    print("Late grouped Pearson r:")
    for row in summaries:
        if row["aggregation_level"] != "grouped_image_level_family" or row["group"] != "late":
            continue
        pearson = row["pearson_r"]
        pearson_text = "NA" if pearson is None else f"{pearson:.4f}"
        print(
            f"  {row['domain']} {row['overlap_space']} {row['centering']}: "
            f"r={pearson_text} n={row['n']}"
        )


if __name__ == "__main__":
    main()
