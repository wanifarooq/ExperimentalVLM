#!/usr/bin/env python3
"""Plot log-likelihood volatility by perturbation type and measured ΔF signature.

This is an offline diagnostic script. It reads ``exp1/per_sample.jsonl`` from an
existing run and writes plots/CSVs without rerunning model inference.
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

from frequency_alignment.analysis.continuous import classify_delta_f_family


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
SIGNATURE_ORDER = ["zero", "low_freq", "broadband", "high_freq"]
SIGNATURE_COLORS = {
    "zero": "#9ca3af",
    "low_freq": "#3182bd",
    "broadband": "#8c6bb1",
    "high_freq": "#de2d26",
}


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


def _spectrum_values(perturbation: Dict[str, Any], spectrum: str) -> Optional[Sequence[float]]:
    if spectrum == "image":
        return perturbation.get("delta_f_relative") or perturbation.get("delta_f")
    if spectrum == "vision":
        return perturbation.get("delta_f_vision_relative") or perturbation.get("delta_f_vision")
    raise ValueError(f"Unknown spectrum {spectrum!r}")


def _load_rows(run_dir: Path, *, spectrum: str) -> List[Dict[str, Any]]:
    exp1_path = run_dir / "exp1" / "per_sample.jsonl"
    if not exp1_path.exists():
        raise FileNotFoundError(f"Missing {exp1_path}")
    rows: List[Dict[str, Any]] = []
    for record in _iter_jsonl(exp1_path):
        image_id = str(record.get("image_id"))
        for level, level_data in (record.get("levels") or {}).items():
            for perturbation in level_data.get("perturbations", []):
                drift = _as_float(perturbation.get("loglik_drift"))
                spectrum_values = _spectrum_values(perturbation, spectrum)
                if drift is None or not spectrum_values:
                    continue
                name = str(perturbation.get("name", "unknown"))
                designed = _designed_name(name)
                severity = perturbation.get("severity", "unknown")
                signature = classify_delta_f_family(spectrum_values)
                rows.append(
                    {
                        "image_id": image_id,
                        "level": str(level),
                        "designed_perturbation": designed,
                        "perturbation_name": name,
                        "severity": str(severity),
                        "actual_signature": signature,
                        "loglik_drift": float(drift),
                        "loglik_volatility": abs(float(drift)),
                        "delta_f_peak_band": perturbation.get(
                            "delta_f_vision_peak_band" if spectrum == "vision" else "delta_f_peak_band"
                        ),
                    }
                )
    return rows


def _mean_ci(values: Sequence[float]) -> Tuple[float, float, float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, median, std, ci95, int(arr.size)


def _summarize(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    for row in rows:
        bucket_key = tuple(str(row.get(key, "unknown")) for key in keys)
        buckets[bucket_key].append(float(row["loglik_volatility"]))
    summaries: List[Dict[str, Any]] = []
    for bucket_key, values in sorted(buckets.items()):
        mean, median, std, ci95, n = _mean_ci(values)
        item = {key: value for key, value in zip(keys, bucket_key)}
        item.update(
            {
                "n": n,
                "mean_loglik_volatility": mean,
                "median_loglik_volatility": median,
                "std_loglik_volatility": std,
                "ci95_mean_loglik_volatility": ci95,
            }
        )
        summaries.append(item)
    return summaries


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _bar_plot(
    rows: Sequence[Dict[str, Any]],
    *,
    label_key: str,
    title: str,
    out_path: Path,
    top_n: Optional[int] = None,
    color_key: Optional[str] = None,
    horizontal: bool = True,
) -> None:
    plot_rows = sorted(rows, key=lambda r: float(r["mean_loglik_volatility"]), reverse=True)
    if top_n:
        plot_rows = plot_rows[:top_n]
    if not plot_rows:
        return
    labels = [str(row[label_key]) for row in plot_rows]
    means = np.asarray([float(row["mean_loglik_volatility"]) for row in plot_rows], dtype=np.float64)
    errors = np.asarray([float(row["ci95_mean_loglik_volatility"]) for row in plot_rows], dtype=np.float64)
    colors = (
        [SIGNATURE_COLORS.get(str(row.get(color_key)), "#4b5563") for row in plot_rows]
        if color_key
        else ["#2563eb"] * len(plot_rows)
    )
    height = max(5.0, 0.32 * len(plot_rows) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, height if horizontal else 6.0))
    if horizontal:
        y = np.arange(len(plot_rows), dtype=np.float64)
        ax.barh(y, means, xerr=errors, color=colors, alpha=0.86, ecolor="#111827", capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Mean correct-answer log-likelihood volatility")
    else:
        x = np.arange(len(plot_rows), dtype=np.float64)
        ax.bar(x, means, yerr=errors, color=colors, alpha=0.86, ecolor="#111827", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Mean correct-answer log-likelihood volatility")
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Bars show mean abs(clean_loglik - perturbed_loglik); error bars are normal 95% CI of the mean.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _heatmap(
    rows: Sequence[Dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    title: str,
    out_path: Path,
    row_order: Optional[Sequence[str]] = None,
    col_order: Optional[Sequence[str]] = None,
) -> None:
    if not rows:
        return
    row_labels = list(row_order) if row_order else sorted({str(row[row_key]) for row in rows})
    col_labels = list(col_order) if col_order else sorted({str(row[col_key]) for row in rows})
    lookup = {(str(row[row_key]), str(row[col_key])): row for row in rows}
    matrix = np.full((len(row_labels), len(col_labels)), np.nan, dtype=np.float64)
    counts = np.zeros_like(matrix)
    for i, r_label in enumerate(row_labels):
        for j, c_label in enumerate(col_labels):
            item = lookup.get((r_label, c_label))
            if item:
                matrix[i, j] = float(item[value_key])
                counts[i, j] = int(item["n"])
    height = max(4.8, 0.35 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, height))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}\nn={int(counts[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] > np.nanmax(matrix) * 0.45 else "#111827",
                )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean log-likelihood volatility")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot perturbation volatility by designed perturbation and measured ΔF signature."
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Run output directory.")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to <output-dir>/plots_perturbation_signature_volatility/<spectrum>.",
    )
    parser.add_argument(
        "--spectrum",
        default="image",
        choices=["image", "vision"],
        help="Which radial ΔF vector to classify: image uses delta_f; vision uses delta_f_vision.",
    )
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    run_dir = args.output_dir
    out_dir = args.plots_dir or (run_dir / "plots_perturbation_signature_volatility" / args.spectrum)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(run_dir, spectrum=args.spectrum)
    if not rows:
        raise SystemExit(f"No usable rows found for spectrum={args.spectrum!r}")

    designed = _summarize(rows, ["designed_perturbation"])
    designed_severity = _summarize(rows, ["designed_perturbation", "severity"])
    signature = _summarize(rows, ["actual_signature"])
    designed_signature = _summarize(rows, ["designed_perturbation", "actual_signature"])
    level_signature = _summarize(rows, ["level", "actual_signature"])

    _write_csv(
        out_dir / "perturbation_signature_rows.csv",
        rows,
        [
            "image_id",
            "level",
            "designed_perturbation",
            "perturbation_name",
            "severity",
            "actual_signature",
            "loglik_drift",
            "loglik_volatility",
            "delta_f_peak_band",
        ],
    )
    summary_fields = [
        "n",
        "mean_loglik_volatility",
        "median_loglik_volatility",
        "std_loglik_volatility",
        "ci95_mean_loglik_volatility",
    ]
    _write_csv(
        out_dir / "volatility_by_designed_perturbation.csv",
        designed,
        ["designed_perturbation", *summary_fields],
    )
    _write_csv(
        out_dir / "volatility_by_designed_perturbation_severity.csv",
        designed_severity,
        ["designed_perturbation", "severity", *summary_fields],
    )
    _write_csv(
        out_dir / "volatility_by_actual_signature.csv",
        signature,
        ["actual_signature", *summary_fields],
    )
    _write_csv(
        out_dir / "volatility_by_designed_and_actual_signature.csv",
        designed_signature,
        ["designed_perturbation", "actual_signature", *summary_fields],
    )
    _write_csv(
        out_dir / "volatility_by_level_and_actual_signature.csv",
        level_signature,
        ["level", "actual_signature", *summary_fields],
    )

    signature_ordered = sorted(
        signature,
        key=lambda row: SIGNATURE_ORDER.index(str(row["actual_signature"]))
        if str(row["actual_signature"]) in SIGNATURE_ORDER
        else 999,
    )
    _bar_plot(
        signature_ordered,
        label_key="actual_signature",
        title=f"Volatility by measured ΔF signature ({args.spectrum} spectrum)",
        out_path=out_dir / "volatility_by_actual_signature.png",
        color_key="actual_signature",
        horizontal=False,
    )
    _bar_plot(
        designed,
        label_key="designed_perturbation",
        title=f"Volatility by designed perturbation type ({args.spectrum} spectrum)",
        out_path=out_dir / "volatility_by_designed_perturbation.png",
        top_n=args.top_n,
    )
    freq_rows = [
        row
        for row in designed_severity
        if any(
            token in str(row["designed_perturbation"]).lower()
            for token in ("lowband", "highband", "allband", "lowpass", "highpass")
        )
    ]
    if freq_rows:
        for row in freq_rows:
            row["label"] = f"{row['designed_perturbation']} sev{row['severity']}"
        _bar_plot(
            freq_rows,
            label_key="label",
            title=f"Frequency perturbation volatility by severity ({args.spectrum} spectrum)",
            out_path=out_dir / "volatility_by_frequency_perturbation_severity.png",
            top_n=None,
        )
    _heatmap(
        designed_signature,
        row_key="designed_perturbation",
        col_key="actual_signature",
        value_key="mean_loglik_volatility",
        title=f"Designed perturbation vs measured ΔF signature ({args.spectrum} spectrum)",
        out_path=out_dir / "volatility_designed_vs_actual_signature.png",
        col_order=SIGNATURE_ORDER,
    )
    _heatmap(
        level_signature,
        row_key="level",
        col_key="actual_signature",
        value_key="mean_loglik_volatility",
        title=f"Level vs measured ΔF signature ({args.spectrum} spectrum)",
        out_path=out_dir / "volatility_level_vs_actual_signature.png",
        row_order=LEVEL_ORDER,
        col_order=SIGNATURE_ORDER,
    )

    manifest = {
        "run_dir": str(run_dir),
        "spectrum": args.spectrum,
        "n_rows": len(rows),
        "outputs": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
        "notes": [
            "designed_perturbation is parsed from the perturbation name before parameters/severity.",
            "actual_signature is classified from the saved radial ΔF vector using classify_delta_f_family.",
            "loglik_volatility is abs(loglik_drift), i.e. correct-answer confidence movement regardless of sign.",
        ],
    }
    with (out_dir / "plot_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote perturbation-signature volatility plots to {out_dir}")
    print(f"Rows: {len(rows)}  Spectrum: {args.spectrum}")
    print(f"Manifest: {out_dir / 'plot_manifest.json'}")


if __name__ == "__main__":
    main()
