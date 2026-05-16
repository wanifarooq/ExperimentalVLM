#!/usr/bin/env python3
"""Offline ΔF-signature × W_t-signature volatility matrix.

Reads saved Exp1/Exp2 outputs and tests the interaction:

    perturbation spectrum signature × query filter signature -> loglik volatility

No model inference is run. The default W_t group is ``last_2`` because that is
the current strongest depth probe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SIGNATURE_ORDER = ["low", "mid", "high", "broadband"]
SIGNATURE_COLORS = {
    "low": "#3182bd",
    "mid": "#31a354",
    "high": "#de2d26",
    "broadband": "#8c6bb1",
}
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
EPS = 1e-12


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _as_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_name(value: Any) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned.strip("_") or "unknown"


def _designed_name(name: Any) -> str:
    text = str(name or "unknown")
    text = text.split("|sev", 1)[0]
    text = text.split("(", 1)[0]
    return text.strip() or "unknown"


def _spectral_features(values: Sequence[float], *, entropy_threshold: float = 0.86, dominance_margin: float = 0.12) -> Dict[str, Any]:
    """Return low/mid/high fractions, entropy, centroid, and hard signature.

    ``broadband`` is assigned when the normalized entropy is high and no
    low/mid/high third dominates by ``dominance_margin``. Otherwise the label is
    the dominant third. This is more robust than centroid-only labels.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.abs(arr[np.isfinite(arr)])
    if arr.size == 0 or float(np.sum(arr)) <= EPS:
        return {
            "signature": "broadband",
            "low_mass": 0.0,
            "mid_mass": 0.0,
            "high_mass": 0.0,
            "entropy": 0.0,
            "centroid": None,
            "num_bins": int(arr.size),
        }
    total = float(np.sum(arr))
    probs = arr / total
    num_bins = int(arr.size)
    thirds = np.array_split(np.arange(num_bins), 3)
    low_mass = float(np.sum(probs[thirds[0]])) if thirds[0].size else 0.0
    mid_mass = float(np.sum(probs[thirds[1]])) if thirds[1].size else 0.0
    high_mass = float(np.sum(probs[thirds[2]])) if thirds[2].size else 0.0
    masses = {"low": low_mass, "mid": mid_mass, "high": high_mass}
    ordered = sorted(masses.items(), key=lambda item: item[1], reverse=True)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, EPS, 1.0))))
    norm_entropy = entropy / math.log(num_bins) if num_bins > 1 else 0.0
    centroid = float(np.sum(probs * np.arange(num_bins)))
    if norm_entropy >= entropy_threshold and (ordered[0][1] - ordered[1][1]) < dominance_margin:
        signature = "broadband"
    else:
        signature = ordered[0][0]
    return {
        "signature": signature,
        "low_mass": low_mass,
        "mid_mass": mid_mass,
        "high_mass": high_mass,
        "entropy": norm_entropy,
        "centroid": centroid,
        "num_bins": num_bins,
    }


def _spectrum_values(perturbation: Dict[str, Any], spectrum: str) -> Optional[Sequence[float]]:
    if spectrum == "image":
        return perturbation.get("delta_f_relative") or perturbation.get("delta_f")
    if spectrum == "vision":
        return perturbation.get("delta_f_vision_relative") or perturbation.get("delta_f_vision")
    raise ValueError(f"Unknown spectrum {spectrum!r}")


def _rankdata(values: Sequence[float]) -> np.ndarray:
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


def _first_order_overlap(w_t: Sequence[float], delta_f: Sequence[float]) -> Optional[float]:
    w = np.asarray(w_t, dtype=np.float64).reshape(-1)
    d = np.asarray(delta_f, dtype=np.float64).reshape(-1)
    usable = min(w.size, d.size)
    if usable <= 0:
        return None
    value = float(np.sum(w[:usable] * d[:usable]))
    return value if math.isfinite(value) else None


def _load_wt_bank(run_dir: Path, *, group: str, question_only: bool) -> Dict[Tuple[str, str], Dict[str, Any]]:
    path = run_dir / "exp2" / "power_spectra.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    records = json.load(path.open())
    bank: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        image_id = str(record.get("image_id"))
        for level, level_data in (record.get("levels") or {}).items():
            source: Optional[Dict[str, Any]]
            if question_only:
                qo = level_data.get("question_only_filter") or {}
                source = qo if group == "overall" else (qo.get("layer_groups") or {}).get(group)
            else:
                source = level_data if group == "overall" else (level_data.get("layer_groups") or {}).get(group)
            if not source:
                continue
            w_t = source.get("W_t")
            if not w_t:
                continue
            features = _spectral_features(w_t)
            bank[(image_id, str(level))] = {
                "W_t": w_t,
                "wt_signature": features["signature"],
                "wt_low_mass": features["low_mass"],
                "wt_mid_mass": features["mid_mass"],
                "wt_high_mass": features["high_mass"],
                "wt_entropy": features["entropy"],
                "wt_centroid": features["centroid"],
                "wt_num_bins": features["num_bins"],
            }
    return bank


def _load_rows(
    run_dir: Path,
    *,
    spectrum: str,
    wt_group: str,
    question_only_wt: bool,
) -> List[Dict[str, Any]]:
    exp1_path = run_dir / "exp1" / "per_sample.jsonl"
    if not exp1_path.exists():
        raise FileNotFoundError(f"Missing {exp1_path}")
    wt_bank = _load_wt_bank(run_dir, group=wt_group, question_only=question_only_wt)
    rows: List[Dict[str, Any]] = []
    for record in _iter_jsonl(exp1_path):
        image_id = str(record.get("image_id"))
        for level, level_data in (record.get("levels") or {}).items():
            wt = wt_bank.get((image_id, str(level)))
            if wt is None:
                continue
            for perturbation in level_data.get("perturbations", []):
                drift = _as_float(perturbation.get("loglik_drift"))
                delta = _spectrum_values(perturbation, spectrum)
                if drift is None or not delta:
                    continue
                delta_features = _spectral_features(delta)
                predicted_overlap = _first_order_overlap(wt["W_t"], delta)
                name = str(perturbation.get("name", "unknown"))
                rows.append(
                    {
                        "image_id": image_id,
                        "level": str(level),
                        "designed_perturbation": _designed_name(name),
                        "perturbation_name": name,
                        "severity": str(perturbation.get("severity", "unknown")),
                        "delta_signature": delta_features["signature"],
                        "delta_low_mass": delta_features["low_mass"],
                        "delta_mid_mass": delta_features["mid_mass"],
                        "delta_high_mass": delta_features["high_mass"],
                        "delta_entropy": delta_features["entropy"],
                        "delta_centroid": delta_features["centroid"],
                        "wt_signature": wt["wt_signature"],
                        "wt_low_mass": wt["wt_low_mass"],
                        "wt_mid_mass": wt["wt_mid_mass"],
                        "wt_high_mass": wt["wt_high_mass"],
                        "wt_entropy": wt["wt_entropy"],
                        "wt_centroid": wt["wt_centroid"],
                        "predicted_first_order_overlap": predicted_overlap,
                        "loglik_drift": float(drift),
                        "loglik_volatility": abs(float(drift)),
                    }
                )
    return rows


def _mean_ci(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean_loglik_volatility": None,
            "median_loglik_volatility": None,
            "std_loglik_volatility": None,
            "ci95_mean_loglik_volatility": None,
        }
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean_loglik_volatility": float(np.mean(arr)),
        "median_loglik_volatility": float(np.median(arr)),
        "std_loglik_volatility": std,
        "ci95_mean_loglik_volatility": float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
    }


def _correlation_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    x = [
        float(row["predicted_first_order_overlap"])
        for row in rows
        if row.get("predicted_first_order_overlap") is not None
    ]
    y = [
        float(row["loglik_volatility"])
        for row in rows
        if row.get("predicted_first_order_overlap") is not None
    ]
    return {
        "pearson_r_overlap_vs_volatility": _pearson(x, y),
        "spearman_rho_overlap_vs_volatility": _spearman(x, y),
    }


def _summarize(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key, "unknown")) for key in keys)].append(row)
    out: List[Dict[str, Any]] = []
    for key_values, bucket_rows in sorted(buckets.items()):
        item = {key: value for key, value in zip(keys, key_values)}
        item.update(_mean_ci([float(row["loglik_volatility"]) for row in bucket_rows]))
        item.update(_correlation_summary(bucket_rows))
        out.append(item)
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _matrix_plot(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    title: str,
    value_key: str,
    cbar_label: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    lookup = {(row["delta_signature"], row["wt_signature"]): row for row in rows}
    matrix = np.full((len(SIGNATURE_ORDER), len(SIGNATURE_ORDER)), np.nan, dtype=np.float64)
    counts = np.zeros_like(matrix)
    for i, delta_sig in enumerate(SIGNATURE_ORDER):
        for j, wt_sig in enumerate(SIGNATURE_ORDER):
            row = lookup.get((delta_sig, wt_sig))
            if row and row.get(value_key) is not None:
                matrix[i, j] = float(row[value_key])
                counts[i, j] = int(row["n"])
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(SIGNATURE_ORDER)))
    ax.set_xticklabels(SIGNATURE_ORDER)
    ax.set_yticks(np.arange(len(SIGNATURE_ORDER)))
    ax.set_yticklabels(SIGNATURE_ORDER)
    ax.set_xlabel("Query W_t signature")
    ax.set_ylabel("Perturbation ΔF signature")
    ax.set_title(title)
    max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    max_value = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    for i in range(len(SIGNATURE_ORDER)):
        for j in range(len(SIGNATURE_ORDER)):
            if np.isfinite(matrix[i, j]):
                if value_key.endswith("volatility"):
                    color = "white" if matrix[i, j] > 0.45 * max_value else "#111827"
                else:
                    color = "white" if abs(matrix[i, j]) > 0.42 * max_abs else "#111827"
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}\nn={int(counts[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.text(
        0.01,
        0.01,
        "Signatures use low/mid/high mass fractions plus normalized entropy. Correlations use W_t·ΔF vs abs(loglik_drift) inside each cell.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _signature_distribution_plot(rows: Sequence[Dict[str, Any]], out_path: Path, *, key: str, title: str) -> None:
    counts = {sig: 0 for sig in SIGNATURE_ORDER}
    for row in rows:
        sig = str(row.get(key, "unknown"))
        counts[sig] = counts.get(sig, 0) + 1
    labels = [sig for sig in SIGNATURE_ORDER if counts.get(sig, 0) > 0]
    values = [counts[sig] for sig in labels]
    colors = [SIGNATURE_COLORS.get(sig, "#4b5563") for sig in labels]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(labels, values, color=colors, alpha=0.88)
    ax.set_ylabel("Rows")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ΔF-signature × W_t-signature volatility matrix from saved Exp1/Exp2 outputs."
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Run output directory.")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to <output-dir>/plots_delta_wt_signature_matrix/<spectrum>_<group>.",
    )
    parser.add_argument("--spectrum", default="image", choices=["image", "vision"])
    parser.add_argument("--wt-group", default="last_2")
    parser.add_argument(
        "--question-only-wt",
        action="store_true",
        help="Use Exp2 question-only W_t filters instead of option-conditioned W_t.",
    )
    args = parser.parse_args()

    run_dir = args.output_dir
    mode = f"{args.spectrum}_{args.wt_group}" + ("_question_only" if args.question_only_wt else "")
    out_dir = args.plots_dir or (run_dir / "plots_delta_wt_signature_matrix" / mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(
        run_dir,
        spectrum=args.spectrum,
        wt_group=args.wt_group,
        question_only_wt=args.question_only_wt,
    )
    if not rows:
        raise SystemExit("No usable rows found.")

    matrix_rows = _summarize(rows, ["delta_signature", "wt_signature"])
    level_matrix_rows = _summarize(rows, ["level", "delta_signature", "wt_signature"])
    designed_matrix_rows = _summarize(rows, ["designed_perturbation", "delta_signature", "wt_signature"])
    delta_summary = _summarize(rows, ["delta_signature"])
    wt_summary = _summarize(rows, ["wt_signature"])

    row_fields = [
        "image_id",
        "level",
        "designed_perturbation",
        "perturbation_name",
        "severity",
        "delta_signature",
        "delta_low_mass",
        "delta_mid_mass",
        "delta_high_mass",
        "delta_entropy",
        "delta_centroid",
        "wt_signature",
        "wt_low_mass",
        "wt_mid_mass",
        "wt_high_mass",
        "wt_entropy",
        "wt_centroid",
        "loglik_drift",
        "loglik_volatility",
        "predicted_first_order_overlap",
    ]
    summary_fields = [
        "n",
        "mean_loglik_volatility",
        "median_loglik_volatility",
        "std_loglik_volatility",
        "ci95_mean_loglik_volatility",
        "pearson_r_overlap_vs_volatility",
        "spearman_rho_overlap_vs_volatility",
    ]
    _write_csv(out_dir / "delta_wt_signature_rows.csv", rows, row_fields)
    _write_csv(
        out_dir / "delta_wt_signature_matrix.csv",
        matrix_rows,
        ["delta_signature", "wt_signature", *summary_fields],
    )
    _write_csv(
        out_dir / "level_delta_wt_signature_matrix.csv",
        level_matrix_rows,
        ["level", "delta_signature", "wt_signature", *summary_fields],
    )
    _write_csv(
        out_dir / "designed_delta_wt_signature_matrix.csv",
        designed_matrix_rows,
        ["designed_perturbation", "delta_signature", "wt_signature", *summary_fields],
    )
    _write_csv(out_dir / "delta_signature_summary.csv", delta_summary, ["delta_signature", *summary_fields])
    _write_csv(out_dir / "wt_signature_summary.csv", wt_summary, ["wt_signature", *summary_fields])

    wt_label = f"{args.wt_group} W_t" + (" question-only" if args.question_only_wt else "")
    _matrix_plot(
        matrix_rows,
        out_dir / "delta_wt_signature_volatility_matrix.png",
        title=f"ΔF signature × {wt_label} signature vs volatility ({args.spectrum} ΔF)",
        value_key="mean_loglik_volatility",
        cbar_label="Mean log-likelihood volatility",
        cmap="magma",
    )
    _matrix_plot(
        matrix_rows,
        out_dir / "delta_wt_signature_pearson_matrix.png",
        title=f"Within-cell Pearson r: overlap vs volatility ({args.spectrum} ΔF, {wt_label})",
        value_key="pearson_r_overlap_vs_volatility",
        cbar_label="Pearson r(W_t·ΔF, volatility)",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    _matrix_plot(
        matrix_rows,
        out_dir / "delta_wt_signature_spearman_matrix.png",
        title=f"Within-cell Spearman rho: overlap vs volatility ({args.spectrum} ΔF, {wt_label})",
        value_key="spearman_rho_overlap_vs_volatility",
        cbar_label="Spearman rho(W_t·ΔF, volatility)",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    _signature_distribution_plot(
        rows,
        out_dir / "delta_signature_distribution.png",
        key="delta_signature",
        title=f"Perturbation ΔF signature distribution ({args.spectrum})",
    )
    _signature_distribution_plot(
        rows,
        out_dir / "wt_signature_distribution.png",
        key="wt_signature",
        title=f"Query {wt_label} signature distribution",
    )

    manifest = {
        "run_dir": str(run_dir),
        "spectrum": args.spectrum,
        "wt_group": args.wt_group,
        "question_only_wt": bool(args.question_only_wt),
        "n_rows": len(rows),
        "outputs": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
        "signature_method": {
            "mass_bins": "np.array_split(radial vector, 3) -> low/mid/high",
            "broadband_rule": "normalized_entropy>=0.86 and top_mass-second_mass<0.12",
            "otherwise": "dominant low/mid/high mass",
        },
    }
    with (out_dir / "plot_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote ΔF × W_t signature matrix to {out_dir}")
    print(f"Rows: {len(rows)}  Spectrum: {args.spectrum}  W_t group: {args.wt_group}  question_only={args.question_only_wt}")
    print(f"Manifest: {out_dir / 'plot_manifest.json'}")


if __name__ == "__main__":
    main()
