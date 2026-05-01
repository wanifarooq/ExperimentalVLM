#!/usr/bin/env python3
"""Visualize saved Exp 2 2D task-frequency filters for selected samples.

This is an experimental inspection script. It reads ``exp2/filters_2d/*.npz``
and ``exp2/power_spectra.json`` from an existing run and writes one plot per
sample × level × selected group into the experimental plots folder.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
GROUP_ORDER = ["overall", "early", "mid", "late"]
EPS = 1e-12


def _load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def _safe_name(value: Any) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned.strip("_") or "unknown"


def _parse_csv(value: Optional[str], default: Sequence[str]) -> List[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _short_question(text: Any, max_len: int = 92) -> str:
    question = " ".join(str(text or "").split())
    if len(question) <= max_len:
        return question
    return question[: max_len - 3].rstrip() + "..."


def _load_filter(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {
                "W_t_2d": np.asarray(data["W_t_2d"], dtype=np.float64),
                "power_2d": np.asarray(data["power_2d"], dtype=np.float64),
                "patch_grid": data["patch_grid"].tolist() if "patch_grid" in data else None,
                "layer_indices": data["layer_indices"].tolist() if "layer_indices" in data else [],
                "fft_window": str(data["fft_window"]) if "fft_window" in data else "",
                "suppress_dc": bool(data["suppress_dc"]) if "suppress_dc" in data else None,
            }
    except Exception:
        return None


def _spectral_centroid_2d(W_t_2d: np.ndarray) -> float:
    w = np.asarray(W_t_2d, dtype=np.float64)
    if w.ndim != 2 or w.size == 0:
        return float("nan")
    total = float(np.sum(w))
    if total <= EPS:
        return float("nan")
    h, width = w.shape
    cy, cx = h // 2, width // 2
    y, x = np.ogrid[:h, :width]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_radius = float(np.max(radius))
    if max_radius <= EPS:
        return 0.0
    return float(np.sum((radius / max_radius) * w) / total)


def _radial_profile_from_2d(W_t_2d: np.ndarray, num_bins: int = 12) -> np.ndarray:
    w = np.asarray(W_t_2d, dtype=np.float64)
    if w.ndim != 2 or w.size == 0:
        return np.zeros(0, dtype=np.float64)
    h, width = w.shape
    cy, cx = h // 2, width // 2
    y, x = np.ogrid[:h, :width]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_radius = float(np.max(radius))
    if max_radius <= EPS:
        return np.asarray([float(np.sum(w))], dtype=np.float64)
    edges = np.linspace(0.0, max_radius, int(num_bins) + 1)
    profile = np.zeros(int(num_bins), dtype=np.float64)
    for idx in range(int(num_bins)):
        if idx == int(num_bins) - 1:
            mask = (radius >= edges[idx]) & (radius <= edges[idx + 1])
        else:
            mask = (radius >= edges[idx]) & (radius < edges[idx + 1])
        if np.any(mask):
            profile[idx] = float(np.sum(w[mask]))
    return profile


def _set_metadata(fig: plt.Figure, lines: Sequence[str]) -> None:
    fig.text(
        0.01,
        0.01,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#374151",
    )


def _plot_filter(
    *,
    sample_id: str,
    level_key: str,
    group_name: str,
    level_data: Dict[str, Any],
    filter_payload: Dict[str, Any],
    out_path: Path,
    radial_bins: int,
) -> None:
    W_t_2d = np.asarray(filter_payload["W_t_2d"], dtype=np.float64)
    radial = _radial_profile_from_2d(W_t_2d, num_bins=radial_bins)
    centroid = _spectral_centroid_2d(W_t_2d)
    patch_grid = filter_payload.get("patch_grid")
    layer_indices = filter_payload.get("layer_indices") or []
    question = _short_question(level_data.get("question"))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.05]},
    )
    ax0, ax1 = axes
    image = ax0.imshow(W_t_2d, cmap="magma", origin="lower", interpolation="nearest")
    ax0.set_title(f"{level_key.replace('_', ' ')} | {group_name}")
    ax0.set_xlabel("FFT x")
    ax0.set_ylabel("FFT y")
    fig.colorbar(image, ax=ax0, fraction=0.047, pad=0.04, label="W_t(u,v)")

    x = np.linspace(0.0, 1.0, len(radial), endpoint=True) if len(radial) else []
    ax1.plot(x, radial, marker="o", linewidth=2.0, color="#2563eb")
    ax1.fill_between(x, radial, color="#93c5fd", alpha=0.35)
    ax1.set_title("Radial summary of this 2D filter")
    ax1.set_xlabel("Normalized radial frequency")
    ax1.set_ylabel("Summed W_t mass")
    ax1.grid(alpha=0.25, linewidth=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.suptitle(f"Sample {sample_id}: {question}", fontsize=12.5)
    _set_metadata(
        fig,
        [
            "Experiment=2",
            "View=sample-level 2D task-frequency filter",
            f"Sample={sample_id}",
            f"Level={level_key}",
            f"Group={group_name}",
            f"PatchGrid={patch_grid}",
            f"LayerIndices={layer_indices}",
            f"W_t_sum={float(np.sum(W_t_2d)):.6f}",
            f"SpectralCentroid2D={centroid:.4f}" if np.isfinite(centroid) else "SpectralCentroid2D=NA",
        ],
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.93))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _select_records(records: Sequence[Dict[str, Any]], sample_ids: Sequence[str], max_samples: int) -> List[Dict[str, Any]]:
    if sample_ids:
        wanted = {str(item) for item in sample_ids}
        return [record for record in records if str(record.get("image_id")) in wanted]
    return list(records[: max(0, int(max_samples))])


def _write_index(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    with (out_dir / "exp2_2d_filter_sample_index.json").open("w") as handle:
        json.dump(list(rows), handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot saved Exp2 2D frequency filters for selected samples and levels."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Frequency-alignment run output directory containing exp2/.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory. Defaults to "
            "<output-dir>/plots_overlap_2d_experimental/exp2_2d_filter_samples."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Optional comma-separated image IDs. Overrides --num-samples.",
    )
    parser.add_argument(
        "--levels",
        default=",".join(LEVEL_ORDER),
        help="Comma-separated levels to plot.",
    )
    parser.add_argument(
        "--groups",
        default="late",
        help="Comma-separated groups to plot. Default late gives 10×8=80 plots.",
    )
    parser.add_argument("--radial-bins", type=int, default=12)
    args = parser.parse_args()

    run_dir = args.output_dir
    exp2_dir = run_dir / "exp2"
    filters_dir = exp2_dir / "filters_2d"
    power_path = exp2_dir / "power_spectra.json"
    if not power_path.exists():
        raise FileNotFoundError(f"Missing {power_path}")
    if not filters_dir.exists():
        raise FileNotFoundError(f"Missing {filters_dir}")

    plots_dir = args.plots_dir or (
        run_dir / "plots_overlap_2d_experimental" / "exp2_2d_filter_samples"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = _load_json(power_path)
    selected = _select_records(
        records,
        _parse_csv(args.sample_ids, []),
        args.num_samples,
    )
    levels = _parse_csv(args.levels, LEVEL_ORDER)
    groups = _parse_csv(args.groups, ["late"])

    index_rows: List[Dict[str, Any]] = []
    created = 0
    missing = 0
    for record in selected:
        sample_id = str(record.get("image_id"))
        for level_key in levels:
            level_data = (record.get("levels") or {}).get(level_key)
            if not level_data:
                continue
            for group_name in groups:
                filter_path = filters_dir / f"{sample_id}_{level_key}_{group_name}.npz"
                payload = _load_filter(filter_path)
                if payload is None:
                    missing += 1
                    continue
                out_name = (
                    f"exp2_2d_filter_sample_{_safe_name(sample_id)}_"
                    f"{_safe_name(level_key)}_{_safe_name(group_name)}.png"
                )
                out_path = plots_dir / out_name
                _plot_filter(
                    sample_id=sample_id,
                    level_key=level_key,
                    group_name=group_name,
                    level_data=level_data,
                    filter_payload=payload,
                    out_path=out_path,
                    radial_bins=args.radial_bins,
                )
                created += 1
                index_rows.append(
                    {
                        "sample_id": sample_id,
                        "level": level_key,
                        "group": group_name,
                        "question": level_data.get("question"),
                        "plot": str(out_path),
                        "filter_file": str(filter_path),
                        "W_t_sum": float(np.sum(payload["W_t_2d"])),
                        "spectral_centroid_2d": _spectral_centroid_2d(payload["W_t_2d"]),
                    }
                )

    _write_index(index_rows, plots_dir)
    print(f"Wrote {created} Exp2 2D filter plots to {plots_dir}")
    if missing:
        print(f"Missing filters skipped: {missing}")
    print(f"Index JSON: {plots_dir / 'exp2_2d_filter_sample_index.json'}")


if __name__ == "__main__":
    main()
