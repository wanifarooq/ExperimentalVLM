from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from ..data.loaders import load_multilevel_vqa_dataset
from ..models import get_adapter
from .device import select_device

logger = logging.getLogger(__name__)

_SPECTRAL_EXPERIMENTS = {1, 2, 3, 5}


def _round_band_value(value: float, mode: str) -> int:
    rounded_mode = str(mode or "ceil").strip().lower()
    if rounded_mode == "ceil":
        return int(math.ceil(value))
    if rounded_mode == "floor":
        return int(math.floor(value))
    return int(round(value))


def _clip_band_value(value: int, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        maximum = minimum
    return max(minimum, min(int(value), maximum))


def recommend_num_bands_from_token_count(
    token_count: float,
    *,
    coefficient: float = 0.6,
    minimum: int = 8,
    maximum: int = 20,
    rounding: str = "ceil",
) -> int:
    token_count = max(1.0, float(token_count))
    raw_value = float(coefficient) * math.sqrt(token_count)
    rounded = _round_band_value(raw_value, rounding)
    return _clip_band_value(rounded, int(minimum), int(maximum))


def _probe_patch_grids(
    cfg: Dict[str, Any],
    *,
    max_probe_samples: int,
) -> Tuple[List[Tuple[int, int]], List[str], List[Dict[str, Any]]]:
    samples = load_multilevel_vqa_dataset(cfg, max_samples=max_probe_samples)
    if not samples:
        return [], [], []

    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("primary", "Qwen/Qwen3-VL-8B-Instruct")
    device = select_device(cfg.get("device", "auto"))
    quantization = model_cfg.get("quantization")

    adapter = get_adapter(model_id)
    patch_grids: List[Tuple[int, int]] = []
    sample_ids: List[str] = []
    failures: List[Dict[str, Any]] = []

    try:
        adapter.load(
            model_id=model_id,
            device=device,
            cache_dir=cfg.get("cache_dir"),
            quantization=quantization,
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            device_map=model_cfg.get("device_map"),
            local_files_only=cfg.get("offline", False),
            attn_implementation=model_cfg.get("attn_implementation"),
            attention_extract_implementation=model_cfg.get(
                "attention_extract_implementation", "eager"
            ),
        )

        for sample in samples:
            try:
                with Image.open(sample.image_path) as image_handle:
                    image = image_handle.convert("RGB")
            except Exception as exc:
                logger.warning("Skipping band probe image %s: %s", sample.image_path, exc)
                continue
            try:
                patch_grid = adapter.get_patch_grid_shape(image)
            except Exception as exc:
                logger.warning("Could not infer patch grid for %s: %s", sample.image_id, exc)
                failures.append({"image_id": str(sample.image_id), "reason": str(exc)})
                continue

            h, w = patch_grid or (0, 0)
            if h <= 0 or w <= 0:
                logger.warning(
                    "Patch-grid probe returned an empty grid for %s (model=%s)",
                    sample.image_id,
                    model_id,
                )
                failures.append(
                    {
                        "image_id": str(sample.image_id),
                        "reason": f"empty_patch_grid:{patch_grid}",
                    }
                )
                continue
            patch_grids.append((int(h), int(w)))
            sample_ids.append(str(sample.image_id))
    finally:
        try:
            adapter.unload()
        except Exception:
            logger.debug("Adapter unload after band probe failed", exc_info=True)

    return patch_grids, sample_ids, failures


def resolve_num_bands_config(
    cfg: Dict[str, Any],
    exp_ids: Sequence[int],
) -> Dict[str, Any]:
    analysis_cfg = cfg.setdefault("analysis", {})
    mode = str(analysis_cfg.get("num_bands_mode", "fixed")).strip().lower()
    fallback_num_bands = max(1, int(analysis_cfg.get("num_bands", 12)))

    resolution_info: Dict[str, Any] = {
        "mode": mode,
        "requested_num_bands": fallback_num_bands,
        "resolved_num_bands": fallback_num_bands,
        "status": "fixed" if mode != "auto" else "pending",
    }

    if mode != "auto":
        analysis_cfg["num_bands_resolution"] = resolution_info
        return cfg

    if not any(exp_id in _SPECTRAL_EXPERIMENTS for exp_id in exp_ids):
        resolution_info["status"] = "skipped_no_spectral_experiment"
        analysis_cfg["num_bands_resolution"] = resolution_info
        return cfg

    auto_cfg = analysis_cfg.get("num_bands_auto", {}) if isinstance(analysis_cfg.get("num_bands_auto"), dict) else {}
    probe_samples = max(1, int(auto_cfg.get("probe_samples", 8)))
    coefficient = float(auto_cfg.get("coefficient", 0.6))
    minimum = max(1, int(auto_cfg.get("min_bands", 8)))
    maximum = max(minimum, int(auto_cfg.get("max_bands", 20)))
    rounding = str(auto_cfg.get("rounding", "ceil"))
    statistic = str(auto_cfg.get("statistic", "median")).strip().lower()

    try:
        patch_grids, sample_ids, probe_failures = _probe_patch_grids(
            cfg,
            max_probe_samples=probe_samples,
        )
    except Exception as exc:
        logger.warning("Auto num_bands probe failed; using fallback=%d: %s", fallback_num_bands, exc)
        resolution_info.update(
            {
                "status": "fallback_probe_failed",
                "error": str(exc),
            }
        )
        analysis_cfg["num_bands_resolution"] = resolution_info
        return cfg

    if not patch_grids:
        resolution_info["status"] = "fallback_no_patch_grids"
        resolution_info["probe_failures"] = probe_failures
        logger.warning(
            "Auto num_bands probe found no valid patch grids for model=%s; using fallback=%d",
            cfg.get("model", {}).get("primary"),
            fallback_num_bands,
        )
        analysis_cfg["num_bands_resolution"] = resolution_info
        return cfg

    token_counts = [int(h * w) for h, w in patch_grids]
    if statistic == "mean":
        representative_tokens = float(sum(token_counts)) / max(1, len(token_counts))
    else:
        sorted_counts = sorted(token_counts)
        mid = len(sorted_counts) // 2
        if len(sorted_counts) % 2 == 0:
            representative_tokens = 0.5 * (sorted_counts[mid - 1] + sorted_counts[mid])
        else:
            representative_tokens = float(sorted_counts[mid])

    resolved_num_bands = recommend_num_bands_from_token_count(
        representative_tokens,
        coefficient=coefficient,
        minimum=minimum,
        maximum=maximum,
        rounding=rounding,
    )

    analysis_cfg["num_bands"] = int(resolved_num_bands)
    resolution_info.update(
        {
            "status": "resolved_from_patch_grid",
            "resolved_num_bands": int(resolved_num_bands),
            "coefficient": coefficient,
            "rounding": rounding,
            "statistic": statistic,
            "min_bands": minimum,
            "max_bands": maximum,
            "probe_samples": probe_samples,
            "probed_patch_grids": [[int(h), int(w)] for h, w in patch_grids],
            "probed_sample_ids": sample_ids,
            "probe_failures": probe_failures,
            "token_counts": token_counts,
            "representative_token_count": representative_tokens,
        }
    )
    analysis_cfg["num_bands_resolution"] = resolution_info
    return cfg
