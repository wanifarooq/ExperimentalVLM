#!/usr/bin/env python3
"""Export a compact teaser package for one Exp 1 sample.

This script is offline-only. It reads an existing run directory and writes CSV,
Markdown, and PNG summaries without rerunning model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frequency_alignment.perturbations import build_perturbation_suite, export_perturbation_suite_images


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


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_") or "unknown"


def _find_sample(run_dir: Path, sample_id: Optional[str]) -> Dict[str, Any]:
    per_sample_path = run_dir / "exp1" / "per_sample.jsonl"
    if not per_sample_path.exists():
        raise FileNotFoundError(f"Missing {per_sample_path}")

    if sample_id is None:
        example_root = run_dir / "exp1" / "perturbation_examples"
        candidates = sorted(path.name for path in example_root.iterdir() if path.is_dir()) if example_root.exists() else []
        sample_id = candidates[0] if candidates else None

    first_record: Optional[Dict[str, Any]] = None
    for record in _iter_jsonl(per_sample_path):
        if first_record is None:
            first_record = record
        if sample_id is None or str(record.get("image_id")) == str(sample_id):
            return record
    if sample_id is not None:
        raise ValueError(f"Sample {sample_id!r} not found in {per_sample_path}")
    if first_record is None:
        raise ValueError(f"No rows found in {per_sample_path}")
    return first_record


def _source_image_path(run_dir: Path, image_id: str) -> Optional[Path]:
    config_path = run_dir / "config_snapshot.json"
    cache_dir = ROOT / ".hf_cache"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            cache_dir = (ROOT / str(cfg.get("cache_dir", ".hf_cache"))).resolve()
        except json.JSONDecodeError:
            pass
    candidates = [
        cache_dir / "gqa" / "images" / f"{image_id}.jpg",
        cache_dir / "gqa" / "images" / f"{image_id}.png",
        ROOT / ".hf_cache" / "gqa" / "images" / f"{image_id}.jpg",
        ROOT / ".hf_cache" / "gqa" / "images" / f"{image_id}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _generate_example_manifest(run_dir: Path, image_id: str, out_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot generate perturbation images without {config_path}")
    cfg = json.loads(config_path.read_text())
    image_path = _source_image_path(run_dir, image_id)
    if image_path is None:
        raise FileNotFoundError(f"Cannot find source GQA image for sample {image_id}")

    pert_cfg = cfg.get("perturbations", {})
    analysis_cfg = cfg.get("analysis", {})
    with Image.open(image_path) as image:
        clean = image.convert("RGB")
        perturbations = build_perturbation_suite(
            clean,
            severity_levels=list(pert_cfg.get("severity_levels", [1, 2, 3])),
            include_natural=bool(pert_cfg.get("include_natural", True)),
            include_frequency=bool(pert_cfg.get("include_frequency", True)),
            num_bands=int(analysis_cfg.get("num_bands", 10)),
            suppress_dc=bool(analysis_cfg.get("suppress_dc", True)),
            natural_types=list(pert_cfg.get("natural_types", [])),
            frequency_types=list(pert_cfg.get("frequency_types", [])),
            severity_params={int(k): v for k, v in (pert_cfg.get("severity_params") or {}).items()},
            seed=int(cfg.get("seed", 42)),
            overlay_mode=str(pert_cfg.get("overlay_mode", "label_free")),
            overlay_count=int(pert_cfg.get("overlay_count", 3)),
            overlay_seed=int(cfg.get("seed", 42)),
        )
        example_root = out_dir / "generated_perturbation_examples"
        example_dir = export_perturbation_suite_images(clean, perturbations, example_root, image_id)
    manifest = json.loads((example_dir / "manifest.json").read_text())
    return example_dir, manifest


def _load_or_generate_example_manifest(
    run_dir: Path,
    image_id: str,
    out_dir: Path,
) -> Tuple[Path, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    example_dir = run_dir / "exp1" / "perturbation_examples" / image_id
    manifest_path = example_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        example_dir, manifest = _generate_example_manifest(run_dir, image_id, out_dir)
    by_name = {str(item.get("name")): item for item in manifest.get("perturbations", [])}
    return example_dir, manifest, by_name


def _level_summary(level: str, level_data: Dict[str, Any]) -> Dict[str, Any]:
    perturbations = list(level_data.get("perturbations", []))
    volatilities = np.asarray([abs(_safe_float(item.get("loglik_drift"))) for item in perturbations], dtype=np.float64)
    drops = np.asarray([_safe_float(item.get("accuracy_drop")) for item in perturbations], dtype=np.float64)
    worst = max(perturbations, key=lambda item: abs(_safe_float(item.get("loglik_drift"))), default={})
    drop_items = [item for item in perturbations if _safe_float(item.get("accuracy_drop")) > 0]
    return {
        "level": level,
        "question": level_data.get("question", ""),
        "question_type": level_data.get("question_type", ""),
        "answer_label": level_data.get("answer_label", ""),
        "clean_predicted": (level_data.get("clean") or {}).get("predicted", ""),
        "clean_correct": bool((level_data.get("clean") or {}).get("correct", False)),
        "complexity_score": _safe_float(level_data.get("complexity_score"), float("nan")),
        "question_complexity_score": _safe_float(level_data.get("question_complexity_score"), float("nan")),
        "prompt_complexity_score": _safe_float(level_data.get("prompt_complexity_score"), float("nan")),
        "option_hardness_score": _safe_float(level_data.get("option_hardness_score"), float("nan")),
        "prediction_entropy": _safe_float(level_data.get("prediction_entropy"), float("nan")),
        "num_perturbations": len(perturbations),
        "accuracy_drops": int(np.sum(drops > 0)) if drops.size else 0,
        "mean_accuracy_drop": float(np.mean(drops)) if drops.size else 0.0,
        "mean_loglik_volatility": float(np.mean(volatilities)) if volatilities.size else 0.0,
        "median_loglik_volatility": float(np.median(volatilities)) if volatilities.size else 0.0,
        "max_loglik_volatility": float(np.max(volatilities)) if volatilities.size else 0.0,
        "worst_perturbation": worst.get("name", ""),
        "worst_family": worst.get("family", ""),
        "worst_severity": worst.get("severity", ""),
        "worst_predicted": worst.get("predicted", ""),
        "worst_correct": bool(worst.get("correct", False)) if worst else False,
        "worst_loglik_drift": _safe_float(worst.get("loglik_drift"), 0.0) if worst else 0.0,
        "drop_perturbations": "; ".join(str(item.get("name", "")) for item in drop_items),
    }


def _flatten_perturbations(record: Dict[str, Any], example_by_name: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    image_id = str(record.get("image_id"))
    for level in LEVEL_ORDER:
        level_data = (record.get("levels") or {}).get(level)
        if not level_data:
            continue
        clean = level_data.get("clean") or {}
        for idx, perturbation in enumerate(level_data.get("perturbations", [])):
            name = str(perturbation.get("name", ""))
            example = example_by_name.get(name, {})
            drift = _safe_float(perturbation.get("loglik_drift"))
            rows.append(
                {
                    "image_id": image_id,
                    "level": level,
                    "question": level_data.get("question", ""),
                    "answer_label": level_data.get("answer_label", ""),
                    "clean_predicted": clean.get("predicted", ""),
                    "clean_correct": bool(clean.get("correct", False)),
                    "perturbation_index": idx,
                    "perturbation_name": name,
                    "family": perturbation.get("family", ""),
                    "severity": perturbation.get("severity", ""),
                    "perturbed_predicted": perturbation.get("predicted", ""),
                    "perturbed_correct": bool(perturbation.get("correct", False)),
                    "accuracy_drop": _safe_float(perturbation.get("accuracy_drop")),
                    "loglik_drift": drift,
                    "loglik_volatility": abs(drift),
                    "prediction_entropy": _safe_float(perturbation.get("prediction_entropy"), float("nan")),
                    "delta_f_peak_band": perturbation.get("delta_f_peak_band", ""),
                    "delta_f_norm": _safe_float(perturbation.get("delta_f_norm"), float("nan")),
                    "delta_f_vision_peak_band": perturbation.get("delta_f_vision_peak_band", ""),
                    "delta_f_vision_norm": _safe_float(perturbation.get("delta_f_vision_norm"), float("nan")),
                    "delta_f_2d_file": perturbation.get("delta_f_2d_file", ""),
                    "delta_f_vision_2d_file": perturbation.get("delta_f_vision_2d_file", ""),
                    "perturbation_png": example.get("filename", ""),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "image_id",
        "level",
        "question",
        "answer_label",
        "clean_predicted",
        "clean_correct",
        "perturbation_index",
        "perturbation_name",
        "family",
        "severity",
        "perturbed_predicted",
        "perturbed_correct",
        "accuracy_drop",
        "loglik_drift",
        "loglik_volatility",
        "mean_loglik_volatility",
        "median_loglik_volatility",
        "worst_perturbation",
    ]
    ordered = [field for field in preferred if field in fieldnames] + [field for field in fieldnames if field not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ordered})


def _copy_selected_images(
    example_dir: Path,
    manifest: Dict[str, Any],
    level_summaries: Sequence[Dict[str, Any]],
    out_dir: Path,
) -> List[Tuple[str, Path]]:
    selected_names = [
        "HighPassKeep(0.38)|sev3",
        "JPEG(70)|sev1",
        "Occlusion(0.08)|sev1",
        "GaussianBlur(3.0)|sev3",
        "AllBandNoise(0.38)|sev3",
        "LowPassKeep(0.38)|sev3",
    ]
    for summary in level_summaries:
        if summary.get("worst_perturbation"):
            selected_names.append(str(summary["worst_perturbation"]))
    manifest_by_name = {str(item.get("name")): item for item in manifest.get("perturbations", [])}
    image_out = out_dir / "selected_images"
    image_out.mkdir(parents=True, exist_ok=True)
    copied: List[Tuple[str, Path]] = []
    clean_src = example_dir / str(manifest.get("clean_image", "clean.png"))
    if clean_src.exists():
        clean_dst = image_out / "clean.png"
        shutil.copy2(clean_src, clean_dst)
        copied.append(("Clean", clean_dst))
    seen = set()
    for name in selected_names:
        if name in seen:
            continue
        seen.add(name)
        item = manifest_by_name.get(name)
        if not item:
            continue
        src = example_dir / str(item.get("filename"))
        if not src.exists():
            continue
        dst = image_out / f"{len(copied):02d}_{_safe_name(name)}.png"
        shutil.copy2(src, dst)
        copied.append((name, dst))
    return copied


def _plot_selected_images(images: Sequence[Tuple[str, Path]], out_path: Path) -> None:
    if not images:
        return
    cols = min(4, len(images))
    rows = int(math.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (title, path) in zip(axes_arr, images):
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes_arr[len(images) :]:
        ax.axis("off")
    fig.suptitle("Teaser sample: clean image and selected perturbations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_level_table(level_summaries: Sequence[Dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 10.5))
    ax.axis("off")
    y = 0.98
    ax.text(0.0, y, "All query levels and outcomes for teaser sample", fontsize=16, fontweight="bold", va="top")
    y -= 0.055
    for summary in level_summaries:
        question = "\n".join(textwrap.wrap(str(summary["question"]), width=92))
        header = (
            f"{summary['level']} | answer={summary['answer_label']} | clean={summary['clean_predicted']} "
            f"({'correct' if summary['clean_correct'] else 'wrong'}) | "
            f"drops={summary['accuracy_drops']}/{summary['num_perturbations']} | "
            f"mean volatility={summary['mean_loglik_volatility']:.3f}"
        )
        worst = (
            f"Worst: {summary['worst_perturbation']} -> pred={summary['worst_predicted']} "
            f"({'correct' if summary['worst_correct'] else 'wrong'}), "
            f"drift={summary['worst_loglik_drift']:.3f}"
        )
        ax.text(0.0, y, header, fontsize=10.5, fontweight="bold", va="top", family="monospace")
        y -= 0.032
        ax.text(0.018, y, question, fontsize=10.2, va="top")
        y -= 0.03 * (question.count("\n") + 1)
        ax.text(0.018, y, worst, fontsize=9.6, va="top", color="#374151")
        y -= 0.052
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_heatmaps(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    perturbations = sorted({int(row["perturbation_index"]): str(row["perturbation_name"]) for row in rows}.items())
    levels = [level for level in LEVEL_ORDER if any(row["level"] == level for row in rows)]
    col_index = {idx: i for i, (idx, _) in enumerate(perturbations)}
    row_index = {level: i for i, level in enumerate(levels)}
    volatility = np.full((len(levels), len(perturbations)), np.nan, dtype=np.float64)
    drops = np.full_like(volatility, np.nan)
    for row in rows:
        volatility[row_index[row["level"]], col_index[int(row["perturbation_index"])]] = _safe_float(row["loglik_volatility"], np.nan)
        drops[row_index[row["level"]], col_index[int(row["perturbation_index"])]] = _safe_float(row["accuracy_drop"], np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(22, 8.5), sharex=True)
    im0 = axes[0].imshow(volatility, aspect="auto", cmap="magma")
    axes[0].set_title("Correct-answer log-likelihood volatility by query level and perturbation")
    axes[0].set_yticks(np.arange(len(levels)))
    axes[0].set_yticklabels(levels)
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)

    im1 = axes[1].imshow(drops, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Accuracy drops by query level and perturbation")
    axes[1].set_yticks(np.arange(len(levels)))
    axes[1].set_yticklabels(levels)
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01)

    tick_positions = list(range(0, len(perturbations), 3))
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels([f"{perturbations[i][0]}:{perturbations[i][1]}" for i in tick_positions], rotation=70, ha="right", fontsize=7)
    axes[1].set_xlabel("Perturbation index:name")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_markdown(
    path: Path,
    *,
    run_dir: Path,
    image_id: str,
    level_summaries: Sequence[Dict[str, Any]],
    all_rows: Sequence[Dict[str, Any]],
) -> None:
    lines = [
        f"# Teaser Sample {image_id}",
        "",
        f"Source run: `{run_dir}`",
        "",
        "This sample was selected from the saved run because it has clean-correct primary queries, visible perturbation failures, and a useful contrast between terse and wordy query behavior. The tables below use the saved model outputs; any perturbation PNGs are only reconstructed for visualization.",
        "",
        "## Level Summary",
        "",
        "| Level | Question | Answer | Clean | Drops | Mean Volatility | Worst Perturbation | Worst Result |",
        "|---|---|---:|---|---:|---:|---|---|",
    ]
    for summary in level_summaries:
        clean = f"{summary['clean_predicted']} ({'correct' if summary['clean_correct'] else 'wrong'})"
        drops = f"{summary['accuracy_drops']}/{summary['num_perturbations']}"
        worst = str(summary["worst_perturbation"]).replace("|", "\\|")
        worst_result = (
            f"{summary['worst_predicted']} ({'correct' if summary['worst_correct'] else 'wrong'}), "
            f"drift={summary['worst_loglik_drift']:.3f}"
        )
        question = str(summary["question"]).replace("|", "\\|")
        lines.append(
            f"| {summary['level']} | {question} | {summary['answer_label']} | {clean} | {drops} | "
            f"{summary['mean_loglik_volatility']:.3f} | {worst} | {worst_result} |"
        )
    lines.extend(
        [
            "",
            "## Useful Files",
            "",
            "- `level_summary.csv`: one row per query level.",
            "- `all_perturbation_results.csv`: all level x perturbation outcomes.",
            "- `selected_perturbations.png`: clean image plus representative perturbation images.",
            "- `query_results_table.png`: all eight queries and key outcomes.",
            "- `perturbation_heatmaps.png`: volatility and accuracy-drop heatmaps across all 78 perturbations.",
            "",
            "## Strongest Accuracy Drops",
            "",
            "| Level | Perturbation | Prediction | Drift |",
            "|---|---|---|---:|",
        ]
    )
    drop_rows = [row for row in all_rows if _safe_float(row.get("accuracy_drop")) > 0]
    drop_rows.sort(key=lambda row: _safe_float(row.get("loglik_volatility")), reverse=True)
    for row in drop_rows[:20]:
        lines.append(
            f"| {row['level']} | {str(row['perturbation_name']).replace('|', '\\\\|')} | "
            f"{row['perturbed_predicted']} ({'correct' if row['perturbed_correct'] else 'wrong'}) | "
            f"{_safe_float(row['loglik_drift']):.3f} |"
        )
    if not drop_rows:
        lines.append("| none | none | none | 0.000 |")
    path.write_text("\n".join(lines) + "\n")


def export_teaser(run_dir: Path, sample_id: Optional[str], out_dir: Optional[Path]) -> Path:
    record = _find_sample(run_dir, sample_id)
    image_id = str(record.get("image_id"))
    out_dir = out_dir or (run_dir / f"teaser_sample_{image_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    example_dir, manifest, example_by_name = _load_or_generate_example_manifest(run_dir, image_id, out_dir)

    level_summaries = [
        _level_summary(level, record["levels"][level])
        for level in LEVEL_ORDER
        if level in (record.get("levels") or {})
    ]
    all_rows = _flatten_perturbations(record, example_by_name)

    _write_csv(out_dir / "level_summary.csv", level_summaries)
    _write_csv(out_dir / "all_perturbation_results.csv", all_rows)
    _write_markdown(out_dir / "teaser_summary.md", run_dir=run_dir, image_id=image_id, level_summaries=level_summaries, all_rows=all_rows)

    selected = _copy_selected_images(example_dir, manifest, level_summaries, out_dir)
    _plot_selected_images(selected, out_dir / "selected_perturbations.png")
    _plot_level_table(level_summaries, out_dir / "query_results_table.png")
    _plot_heatmaps(all_rows, out_dir / "perturbation_heatmaps.png")

    manifest_out = {
        "image_id": image_id,
        "source_run": str(run_dir),
        "source_example_dir": str(example_dir),
        "num_levels": len(level_summaries),
        "num_level_perturbation_rows": len(all_rows),
        "outputs": [
            "teaser_summary.md",
            "level_summary.csv",
            "all_perturbation_results.csv",
            "selected_perturbations.png",
            "query_results_table.png",
            "perturbation_heatmaps.png",
            "selected_images/",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Existing frequency_alignment output directory.")
    parser.add_argument("--sample-id", default=None, help="Image/sample id to export. Defaults to first sample with saved perturbation images.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to <run_dir>/teaser_sample_<id>.")
    args = parser.parse_args()
    out_dir = export_teaser(args.run_dir, args.sample_id, args.out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
