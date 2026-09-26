#!/usr/bin/env python3
"""Analysis-only prompt-template sensitivity check for GQA/Qwen2B.

This script intentionally does not modify the production dataset builders or
experiment code. It reuses the Exp1 scoring path on a small sample to compare
alternative wordy wrappers against the current wordy mirror prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frequency_alignment.data.base import GranularityLevel, LevelData
from frequency_alignment.data.complexity import prompt_content_word_count
from frequency_alignment.data.loaders import load_multilevel_vqa_dataset
from frequency_alignment.models import get_adapter
from frequency_alignment.perturbations import build_perturbation_suite
from frequency_alignment.utils.device import select_device


LOGGER = logging.getLogger("prompt_template_analysis")

PRIMARY_TO_WORDY = {
    GranularityLevel.L1_COARSE: GranularityLevel.L5_WORDY_SIMPLETON,
    GranularityLevel.L2_MEDIUM: GranularityLevel.L6_WORDY_MEDIUM,
    GranularityLevel.L3_FINE: GranularityLevel.L7_WORDY_FINE,
    GranularityLevel.L4_VERY_FINE: GranularityLevel.L8_WORDY_VERY_FINE,
}

MEANINGFUL_FILLER_WORDS = (
    "carefully inspect visible objects attributes spatial relations scene evidence "
    "before answering focus on grounded visual details compare relevant regions "
    "avoid assumptions use observable information"
).split()

RANDOM_NEUTRAL_PHRASES = [
    "For this visual checkpoint, continue with the neutral inspection step.",
    "Before the final response, maintain the same image context and proceed.",
    "This wording adds no new scene facts; treat it only as instruction padding.",
    "Keep the visual instance fixed while answering the following item.",
    "Move through the prompt carefully and rely only on the displayed image.",
]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _safe_float(value: Any) -> Optional[float]:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def _predict_label(scores: Dict[str, float]) -> Optional[str]:
    if not scores:
        return None
    return max(scores, key=scores.get)


def _prediction_entropy(scores: Dict[str, float]) -> Optional[float]:
    if not scores:
        return None
    values = np.asarray([float(v) for v in scores.values()], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return None
    probs = weights / total
    return -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def _score_margin(scores: Dict[str, float]) -> Optional[float]:
    vals = sorted((float(v) for v in scores.values() if math.isfinite(float(v))), reverse=True)
    if len(vals) < 2:
        return None
    return float(vals[0] - vals[1])


def _repeat_words_to_count(words: List[str], count: int) -> str:
    if count <= 0 or not words:
        return ""
    out = [words[i % len(words)] for i in range(count)]
    return " ".join(out)


def _content_count(text: str) -> int:
    return int(prompt_content_word_count(text))


def _meaningful_same_content_count(question: str, target_content_words: int) -> str:
    """Add meaningful visual-analysis wording until content count matches target."""

    question_count = _content_count(question)
    extra_needed = max(0, int(target_content_words) - question_count)
    prefix = _repeat_words_to_count(MEANINGFUL_FILLER_WORDS, extra_needed)
    if not prefix:
        return question
    return f"{prefix}. {question}"


def _random_neutral_template(question: str, *, seed: int) -> str:
    rng = random.Random(seed)
    phrases = rng.sample(RANDOM_NEUTRAL_PHRASES, k=min(3, len(RANDOM_NEUTRAL_PHRASES)))
    return " ".join(phrases + [question])


def _structured_template(question: str) -> str:
    return (
        "Visual evidence check. Read the item, inspect the image, and answer with "
        f"the most appropriate option. Question: {question}"
    )


def _template_variants(
    *,
    base_level: LevelData,
    wordy_level: Optional[LevelData],
    seed: int,
) -> List[Dict[str, Any]]:
    current_wordy_question = wordy_level.question if wordy_level is not None else base_level.question
    target_count = max(_content_count(current_wordy_question), _content_count(base_level.question))
    return [
        {
            "template": "original_primary",
            "description": "Original terse primary-level GQA question.",
            "question": base_level.question,
            "source_level": base_level.level.name,
        },
        {
            "template": "current_wordy_mirror",
            "description": "Current production wordy mirror prompt for the same semantic level.",
            "question": current_wordy_question,
            "source_level": wordy_level.level.name if wordy_level is not None else base_level.level.name,
        },
        {
            "template": "random_neutral_wrapper",
            "description": "Seeded neutral wrapper with semantically empty instruction text.",
            "question": _random_neutral_template(base_level.question, seed=seed),
            "source_level": base_level.level.name,
        },
        {
            "template": "meaningful_same_content_count",
            "description": "Visual-analysis wording padded to the current wordy mirror content-word count.",
            "question": _meaningful_same_content_count(base_level.question, target_count),
            "source_level": base_level.level.name,
        },
        {
            "template": "structured_instruction",
            "description": "Short structured instruction wrapper around the same question.",
            "question": _structured_template(base_level.question),
            "source_level": base_level.level.name,
        },
    ]


def _directional_summary(drifts: Iterable[float]) -> Dict[str, float]:
    values = [float(v) for v in drifts if math.isfinite(float(v))]
    positive = [v for v in values if v > 0.0]
    negative = [v for v in values if v < 0.0]
    return {
        "mean_loglik_drift": float(np.mean(values)) if values else 0.0,
        "mean_loglik_erosion": float(np.mean(positive)) if positive else 0.0,
        "mean_loglik_recovery": float(np.mean(negative)) if negative else 0.0,
        "mean_loglik_volatility": float(np.mean(np.abs(values))) if values else 0.0,
    }


def _aggregate_rows(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(row)

    output: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: item[0]):
        drifts = [
            float(item["loglik_drift"])
            for item in items
            if _safe_float(item.get("loglik_drift")) is not None
        ]
        clean_keys = {(item["image_id"], item["level"], item["template"]) for item in items}
        clean_correct_by_key: Dict[Tuple[Any, ...], float] = {}
        clean_word_count_by_key: Dict[Tuple[Any, ...], float] = {}
        for item in items:
            clean_key = (item["image_id"], item["level"], item["template"])
            clean_correct_by_key[clean_key] = float(item.get("clean_correct", 0.0) or 0.0)
            clean_word_count_by_key[clean_key] = float(item.get("prompt_content_words", 0.0) or 0.0)

        record = {group_keys[i]: key[i] for i in range(len(group_keys))}
        record.update(
            {
                "n_samples": len(clean_keys),
                "n_perturbation_rows": len(items),
                "clean_accuracy": float(np.mean(list(clean_correct_by_key.values())))
                if clean_correct_by_key
                else 0.0,
                "perturbed_accuracy": float(
                    np.mean([float(item.get("pert_correct", 0.0) or 0.0) for item in items])
                )
                if items
                else 0.0,
                "mean_accuracy_drop": float(
                    np.mean([float(item.get("accuracy_drop", 0.0) or 0.0) for item in items])
                )
                if items
                else 0.0,
                "mean_prompt_content_words": float(np.mean(list(clean_word_count_by_key.values())))
                if clean_word_count_by_key
                else 0.0,
            }
        )
        record.update(_directional_summary(drifts))
        output.append(record)
    return output


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_summary_plots(summary_by_level: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional.
        LOGGER.warning("Skipping plots because matplotlib is unavailable: %s", exc)
        return []

    paths: List[str] = []
    metrics = [
        ("mean_loglik_volatility", "Mean Correct-Answer Log-Likelihood Volatility"),
        ("mean_accuracy_drop", "Mean Accuracy Drop"),
    ]
    levels = [level.name for level in PRIMARY_TO_WORDY.keys()]
    templates = [
        "original_primary",
        "current_wordy_mirror",
        "random_neutral_wrapper",
        "meaningful_same_content_count",
        "structured_instruction",
    ]
    by_key = {
        (row.get("level"), row.get("template")): row
        for row in summary_by_level
    }
    x = np.arange(len(levels), dtype=np.float64)
    width = 0.15

    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(12, 5.8))
        for idx, template in enumerate(templates):
            values = [
                float(by_key.get((level, template), {}).get(metric, 0.0) or 0.0)
                for level in levels
            ]
            ax.bar(x + (idx - 2) * width, values, width=width, label=template)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"GQA Qwen2B prompt-template sensitivity: {ylabel}")
        ax.legend(fontsize=8, ncols=2)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = out_dir / f"prompt_template_{metric}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _prepare_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = deepcopy(_load_yaml(Path(args.config)))
    cfg.setdefault("model", {})["primary"] = args.model_id
    cfg.setdefault("data", {})["dataset"] = "gqa"
    cfg.setdefault("data", {})["max_samples"] = int(args.max_samples)
    # Keep this analysis focused on option scoring, not feature extraction.
    cfg.setdefault("experiments", {}).setdefault("exp1", {})["extract_vision_tokens"] = False
    cfg["experiments"]["exp1"]["store_feature_delta_f"] = False
    return cfg


def _score_template(
    *,
    adapter: Any,
    image: Image.Image,
    question: str,
    options: Dict[str, str],
    answer_label: Optional[str],
) -> Dict[str, Any]:
    scores = adapter.score_options(image, question, options)
    predicted = _predict_label(scores)
    correct = bool(answer_label is not None and predicted == answer_label)
    return {
        "scores": {k: float(v) for k, v in scores.items()},
        "predicted": predicted,
        "correct": correct,
        "entropy": _prediction_entropy(scores),
        "margin": _score_margin(scores),
        "correct_answer_loglik": (
            float(scores[answer_label])
            if answer_label is not None and answer_label in scores
            else None
        ),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _prepare_cfg(args)
    seed = int(cfg.get("seed", 42))
    model_cfg = cfg.get("model", {})
    model_id = str(model_cfg.get("primary") or args.model_id)
    device = select_device(str(cfg.get("device", "auto")))

    LOGGER.info("Loading %s on %s", model_id, device)
    adapter = get_adapter(model_id)
    adapter.load(
        model_id=model_id,
        device=device,
        cache_dir=cfg.get("cache_dir"),
        quantization=model_cfg.get("quantization"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        device_map=model_cfg.get("device_map"),
        local_files_only=bool(cfg.get("offline", False)),
        attn_implementation=model_cfg.get("attn_implementation"),
        attention_extract_implementation=model_cfg.get(
            "attention_extract_implementation", "eager"
        ),
    )

    request_n = int(args.max_samples)
    if request_n % 2 == 1:
        request_n += 1
    LOGGER.info("Loading GQA samples: requested=%d, loader_cap=%d", args.max_samples, request_n)
    samples = load_multilevel_vqa_dataset(cfg, max_samples=request_n)[: int(args.max_samples)]
    if not samples:
        raise RuntimeError("No GQA samples loaded.")

    pert_cfg = cfg.get("perturbations", {})
    analysis_cfg = cfg.get("analysis", {})
    rows: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []
    template_definitions: Dict[str, str] = {}
    t0 = time.time()

    try:
        for sample_idx, sample in enumerate(samples):
            LOGGER.info("Sample %d/%d image_id=%s", sample_idx + 1, len(samples), sample.image_id)
            image = Image.open(sample.image_path).convert("RGB")
            overlay_options, overlay_base_label = sample.reference_overlay_context()
            perturbations = build_perturbation_suite(
                image,
                severity_levels=list(pert_cfg.get("severity_levels", [1])),
                include_natural=bool(pert_cfg.get("include_natural", True)),
                include_frequency=bool(pert_cfg.get("include_frequency", True)),
                num_bands=int(analysis_cfg.get("num_bands", 10) or 10),
                suppress_dc=bool(analysis_cfg.get("suppress_dc", True)),
                natural_types=pert_cfg.get("natural_types"),
                frequency_types=pert_cfg.get("frequency_types"),
                severity_params=pert_cfg.get("severity_params"),
                seed=seed + sample_idx,
                overlay_mode=str(pert_cfg.get("overlay_mode", "label_free")),
                overlay_count=int(pert_cfg.get("overlay_count", 3) or 0),
                overlay_options=overlay_options,
                overlay_base_label=overlay_base_label,
                overlay_seed=seed + sample_idx,
            )
            if args.max_perturbations > 0:
                perturbations = perturbations[: int(args.max_perturbations)]
            LOGGER.info("  perturbations=%d", len(perturbations))

            for base_level, wordy_level_key in PRIMARY_TO_WORDY.items():
                base_data = sample.levels.get(base_level)
                if base_data is None:
                    continue
                wordy_data = sample.levels.get(wordy_level_key)
                variants = _template_variants(
                    base_level=base_data,
                    wordy_level=wordy_data,
                    seed=seed + sample_idx * 100 + int(base_level),
                )

                for variant in variants:
                    template_name = str(variant["template"])
                    template_definitions[template_name] = str(variant["description"])
                    question = str(variant["question"])
                    options = dict(base_data.options or {})
                    answer_label = base_data.answer_label
                    clean = _score_template(
                        adapter=adapter,
                        image=image,
                        question=question,
                        options=options,
                        answer_label=answer_label,
                    )
                    prompt_words = _content_count(question)
                    clean_row = {
                        "image_id": sample.image_id,
                        "level": base_level.name,
                        "wordy_level": wordy_level_key.name,
                        "template": template_name,
                        "source_level": variant.get("source_level"),
                        "question": question,
                        "prompt_content_words": prompt_words,
                        "base_prompt_content_words": _content_count(base_data.question),
                        "current_wordy_prompt_content_words": (
                            _content_count(wordy_data.question) if wordy_data is not None else None
                        ),
                        "num_options": len(options),
                        "answer_label": answer_label,
                        "answer_text": options.get(answer_label) if answer_label else None,
                        "clean_pred": clean["predicted"],
                        "clean_correct": 1.0 if clean["correct"] else 0.0,
                        "clean_entropy": clean["entropy"],
                        "clean_margin": clean["margin"],
                        "clean_correct_answer_loglik": clean["correct_answer_loglik"],
                    }
                    clean_rows.append(clean_row)

                    for perturbation in perturbations:
                        pert = _score_template(
                            adapter=adapter,
                            image=perturbation.perturbed_image,
                            question=question,
                            options=options,
                            answer_label=answer_label,
                        )
                        clean_ll = _safe_float(clean.get("correct_answer_loglik"))
                        pert_ll = _safe_float(pert.get("correct_answer_loglik"))
                        drift = (
                            float(clean_ll - pert_ll)
                            if clean_ll is not None and pert_ll is not None
                            else None
                        )
                        rows.append(
                            {
                                **clean_row,
                                "perturbation": perturbation.name,
                                "perturbation_family": perturbation.family,
                                "severity": int(perturbation.severity),
                                "pert_pred": pert["predicted"],
                                "pert_correct": 1.0 if pert["correct"] else 0.0,
                                "pert_entropy": pert["entropy"],
                                "pert_margin": pert["margin"],
                                "pert_correct_answer_loglik": pert["correct_answer_loglik"],
                                "accuracy_drop": (
                                    1.0 if clean["correct"] and not pert["correct"] else 0.0
                                ),
                                "loglik_drift": drift,
                                "loglik_erosion": max(drift, 0.0) if drift is not None else None,
                                "loglik_recovery": min(drift, 0.0) if drift is not None else None,
                                "loglik_volatility": abs(drift) if drift is not None else None,
                            }
                        )
    finally:
        adapter.unload()

    summary_by_level = _aggregate_rows(rows, ["template", "level"])
    summary_overall = _aggregate_rows(rows, ["template"])
    summary_by_perturbation_family = _aggregate_rows(rows, ["template", "level", "perturbation_family"])

    _write_csv(clean_rows, out_dir / "prompt_template_clean_scores.csv")
    _write_csv(rows, out_dir / "prompt_template_perturbation_rows.csv")
    _write_csv(summary_by_level, out_dir / "prompt_template_summary_by_level.csv")
    _write_csv(summary_overall, out_dir / "prompt_template_summary_overall.csv")
    _write_csv(
        summary_by_perturbation_family,
        out_dir / "prompt_template_summary_by_level_family.csv",
    )
    plot_paths = _make_summary_plots(summary_by_level, out_dir)

    manifest = {
        "status": "complete",
        "analysis_only": True,
        "main_pipeline_modified": False,
        "model_id": model_id,
        "dataset": "gqa",
        "num_samples": len(samples),
        "image_ids": [str(sample.image_id) for sample in samples],
        "levels": [level.name for level in PRIMARY_TO_WORDY.keys()],
        "templates": template_definitions,
        "num_clean_rows": len(clean_rows),
        "num_perturbation_rows": len(rows),
        "elapsed_seconds": time.time() - t0,
        "outputs": {
            "clean_scores": "prompt_template_clean_scores.csv",
            "perturbation_rows": "prompt_template_perturbation_rows.csv",
            "summary_by_level": "prompt_template_summary_by_level.csv",
            "summary_overall": "prompt_template_summary_overall.csv",
            "summary_by_level_family": "prompt_template_summary_by_level_family.csv",
            "plots": [str(Path(path).name) for path in plot_paths],
        },
        "config_used": {
            "config_path": str(args.config),
            "max_samples": int(args.max_samples),
            "max_perturbations": int(args.max_perturbations),
            "include_natural": bool(pert_cfg.get("include_natural", True)),
            "include_frequency": bool(pert_cfg.get("include_frequency", True)),
            "severity_levels": list(pert_cfg.get("severity_levels", [1])),
            "natural_types": pert_cfg.get("natural_types"),
            "frequency_types": pert_cfg.get("frequency_types"),
        },
    }
    _write_json(manifest, out_dir / "manifest.json")
    LOGGER.info("Wrote outputs to %s", out_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="frequency_alignment/configs/local_test.yaml",
        help="Base config to reuse for data/perturbations.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Model ID for the analysis run.",
    )
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument(
        "--max-perturbations",
        type=int,
        default=0,
        help="Optional cap per image. Use 0 to run all configured perturbations.",
    )
    parser.add_argument(
        "--out-dir",
        default="prompt_template_analysis_qwen2b_gqa",
        help="Output directory for CSV/JSON/plots.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args)


if __name__ == "__main__":
    main()
