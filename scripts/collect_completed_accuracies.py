#!/usr/bin/env python3
"""Collect Exp1 accuracy metrics from completed frequency-alignment runs.

The script scans for ``exp1/summary.json`` files and writes two CSV tables:

* ``accuracy_by_level.csv``: clean/perturbed accuracy and mean drop per level.
* ``accuracy_by_perturbation.csv``: mean drop per level x perturbation.

No model inference is run and no experiment outputs are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _run_dir_from_summary(summary_path: Path) -> Path:
    # .../<run>/exp1/summary.json
    return summary_path.parent.parent


def _config_snapshot(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "config_snapshot.json"
    return _load_json(path) if path.exists() else {}


def _value_from_config(cfg: Dict[str, Any], *keys: str) -> Any:
    current: Any = cfg
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _infer_model(run_dir: Path, cfg: Dict[str, Any]) -> str:
    model = _value_from_config(cfg, "model", "primary")
    if model:
        return str(model)
    name = str(run_dir).lower()
    if "llava" in name:
        return "llava"
    if "qwen" in name:
        return "qwen"
    return ""


def _infer_dataset(run_dir: Path, cfg: Dict[str, Any]) -> str:
    dataset = _value_from_config(cfg, "data", "dataset")
    if dataset:
        return str(dataset)
    name = str(run_dir).lower()
    if "clevr" in name:
        return "clevr"
    if "gqa" in name or "gpa" in name:
        return "gqa"
    return ""


def _sort_level(level: str) -> int:
    try:
        return LEVEL_ORDER.index(level)
    except ValueError:
        return len(LEVEL_ORDER)


def iter_summaries(roots: Iterable[Path], include_sanity: bool) -> List[Path]:
    summaries: List[Path] = []
    seen = set()
    for root in roots:
        for path in root.rglob("exp1/summary.json"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            if not include_sanity and "sanity" in str(path).lower():
                continue
            seen.add(resolved)
            summaries.append(path)
    return sorted(summaries, key=lambda p: str(p))


def collect_level_rows(summary_path: Path) -> List[Dict[str, Any]]:
    run_dir = _run_dir_from_summary(summary_path)
    summary = _load_json(summary_path)
    cfg = _config_snapshot(run_dir)
    model = _infer_model(run_dir, cfg)
    dataset = _infer_dataset(run_dir, cfg)
    per_level = summary.get("per_level") or {}
    rows: List[Dict[str, Any]] = []
    for level, metrics in sorted(per_level.items(), key=lambda item: _sort_level(item[0])):
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "run_dir": str(run_dir),
                "model": model,
                "dataset": dataset,
                "level": level,
                "clean_accuracy": metrics.get("clean_accuracy"),
                "perturbed_accuracy": metrics.get("perturbed_accuracy"),
                "mean_accuracy_drop": metrics.get("mean_accuracy_drop"),
                "std_accuracy_drop": metrics.get("std_accuracy_drop"),
                "num_samples": metrics.get("num_samples"),
                "num_perturbation_evals": metrics.get("num_perturbation_evals"),
            }
        )
    return rows


def collect_perturbation_rows(summary_path: Path) -> List[Dict[str, Any]]:
    run_dir = _run_dir_from_summary(summary_path)
    summary = _load_json(summary_path)
    cfg = _config_snapshot(run_dir)
    model = _infer_model(run_dir, cfg)
    dataset = _infer_dataset(run_dir, cfg)
    per_level_perturbation = summary.get("per_level_perturbation") or {}
    rows: List[Dict[str, Any]] = []
    if not isinstance(per_level_perturbation, dict):
        return rows
    for level, perturbations in sorted(
        per_level_perturbation.items(),
        key=lambda item: _sort_level(item[0]),
    ):
        if not isinstance(perturbations, dict):
            continue
        for perturbation, metrics in sorted(perturbations.items()):
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "model": model,
                    "dataset": dataset,
                    "level": level,
                    "perturbation": perturbation,
                    "mean_accuracy_drop": metrics.get("mean_drop"),
                    "std_accuracy_drop": metrics.get("std_drop"),
                    "n": metrics.get("n"),
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_run_overview(level_rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in level_rows:
        grouped.setdefault(str(row["run_dir"]), []).append(row)
    for run_dir, rows in sorted(grouped.items()):
        clean_values = [
            float(row["clean_accuracy"])
            for row in rows
            if row.get("clean_accuracy") is not None
        ]
        pert_values = [
            float(row["perturbed_accuracy"])
            for row in rows
            if row.get("perturbed_accuracy") is not None
        ]
        n_values = [row.get("num_samples") for row in rows if row.get("num_samples") is not None]
        model = rows[0].get("model", "")
        dataset = rows[0].get("dataset", "")
        clean = sum(clean_values) / len(clean_values) if clean_values else 0.0
        pert = sum(pert_values) / len(pert_values) if pert_values else 0.0
        n = max(n_values) if n_values else ""
        print(
            f"{run_dir} | dataset={dataset} | model={model} | "
            f"levels={len(rows)} | n={n} | mean_clean={clean:.4f} | mean_perturbed={pert:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path(".")],
        help="Root directories to scan. Defaults to current directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("accuracy_reports"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--include-sanity",
        action="store_true",
        help="Include sanity-check output directories. Defaults to excluded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = iter_summaries(args.roots, include_sanity=args.include_sanity)
    level_rows: List[Dict[str, Any]] = []
    perturbation_rows: List[Dict[str, Any]] = []
    for summary_path in summaries:
        level_rows.extend(collect_level_rows(summary_path))
        perturbation_rows.extend(collect_perturbation_rows(summary_path))

    level_fields = [
        "run_dir",
        "model",
        "dataset",
        "level",
        "clean_accuracy",
        "perturbed_accuracy",
        "mean_accuracy_drop",
        "std_accuracy_drop",
        "num_samples",
        "num_perturbation_evals",
    ]
    perturbation_fields = [
        "run_dir",
        "model",
        "dataset",
        "level",
        "perturbation",
        "mean_accuracy_drop",
        "std_accuracy_drop",
        "n",
    ]
    write_csv(args.out_dir / "accuracy_by_level.csv", level_rows, level_fields)
    write_csv(
        args.out_dir / "accuracy_by_perturbation.csv",
        perturbation_rows,
        perturbation_fields,
    )

    print(f"Found {len(summaries)} completed Exp1 summaries.")
    print(f"Wrote {len(level_rows)} level rows to {args.out_dir / 'accuracy_by_level.csv'}")
    print(
        f"Wrote {len(perturbation_rows)} perturbation rows to "
        f"{args.out_dir / 'accuracy_by_perturbation.csv'}"
    )
    print_run_overview(level_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
