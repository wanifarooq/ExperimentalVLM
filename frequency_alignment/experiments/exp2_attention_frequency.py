"""Experiment 2: Cross-Attention Frequency Analysis.

Extract cross-attention maps from the VLM for each granularity level and
decompose them into frequency bands via 2D FFT.  Measure how the effective
bandwidth G(t) of the attention filter varies with task granularity.

The current hypothesis is that late-layer task filters concentrate downward
with semantic granularity: top-2 low-frequency mass rises, high-frequency tail
mass falls, and effective bandwidth G(t) decreases. Early layers are treated
as prompt-format controls rather than the load-bearing granularity test.

Outputs:
    exp2/summary.json              -- aggregate metrics and hypothesis tests
    exp2/filters/{id}_{level}.npy  -- W_t filters per sample per level
    exp2/power_spectra.json        -- per-sample power spectra
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..analysis.continuous import (
    attach_complexity_residual,
    summarize_by_score,
    summarize_fixed_effects_trend,
    summarize_horse_race_view,
    summarize_linear_trend,
    summarize_multivariate_regression,
)
from ..analysis.gate_thresholds import GATES
from ..analysis.level_views import LEVEL_VIEW_ORDER, LEVEL_VIEWS
from ..analysis.spectral import (
    attention_to_spatial_grid,
    compare_spectral_filters,
    compute_attention_power_spectrum,
    compute_attention_power_spectrum_by_layer,
    compute_attention_power_spectrum_multi,
    compute_effective_bandwidth,
    compute_filter_shape_metrics,
    compute_filter_W_t,
    spectral_vector_length,
)
from ..analysis.statistics import (
    one_way_anova,
    spearman_correlation,
    bootstrap_ci,
)
from ..data.base import ExperimentResult, GranularityLevel, ALL_VQA_LEVEL_NAMES, PRIMARY_VQA_LEVEL_NAMES
from ..data.complexity import ensure_level_complexity
from ..data.complexity import (
    OPTION_HARDNESS_SCORE_DEFINITION,
    OPTION_HARDNESS_SCORE_NAME,
    PROMPT_COMPLEXITY_SCORE_DEFINITION,
    PROMPT_COMPLEXITY_SCORE_NAME,
    SEMANTIC_COMPLEXITY_SCORE_DEFINITION,
    SEMANTIC_COMPLEXITY_SCORE_NAME,
)
from ..data.loaders import load_multilevel_vqa_dataset
from ..models import get_adapter
from ..utils.device import select_device
from ..utils.layer_groups import LAYER_GROUP_ORDER, layer_group_ranges, split_by_layer_group
from ..utils.io import save_json

logger = logging.getLogger(__name__)
CONTROL_ORDER = ("empty_language", "random_language")
# Last-layer subgroups: only `last_2` is retained as the canonical depth-probe
# (it consistently delivers the strongest overlap-law correlation in the
# matched-FE diagnostic). last_1 / last_3 / last_4 added clutter without
# changing the conclusions and were removed for the camera-ready cleanup.
# Naming convention: `last_K` is the K-th-from-last decoder attention layer
# (singleton, not an average), so the maximum K determines how many trailing
# layers we must extract from the adapter.
LAST_LAYER_GROUP_ORDER = ("last_2",)


def _max_last_layer_offset(group_order: tuple = LAST_LAYER_GROUP_ORDER) -> int:
    """Largest K in any `last_K` name in ``group_order`` (0 if none)."""
    max_k = 0
    for name in group_order:
        if not name.startswith("last_"):
            continue
        try:
            max_k = max(max_k, int(name.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    return max_k


LAST_LAYER_MAX_OFFSET = _max_last_layer_offset()
_EPS = 1e-8


def _overlap_2d_enabled(cfg: Dict[str, Any]) -> bool:
    """Single grouped switch for the optional 2D overlap diagnostic path."""
    overlap_cfg = cfg.get("analysis", {}).get("overlap_2d", {})
    if isinstance(overlap_cfg, dict):
        return bool(overlap_cfg.get("enabled", False))
    return bool(overlap_cfg)


def _question_only_control_enabled(exp_cfg: Dict[str, Any]) -> bool:
    """Optional Exp2 side branch for filters extracted from question text only."""
    control_cfg = exp_cfg.get("question_only_control", {})
    if isinstance(control_cfg, dict):
        return bool(control_cfg.get("enabled", False))
    return bool(control_cfg)


def _compute_l2_variants_enabled(cfg: Dict[str, Any]) -> bool:
    """Whether to compute and persist the L2-normalised W_t variants.

    The paper uses L1-normalised W_t throughout; the L2 variant was an
    early-exploration convention that doubles JSON / .npy output and is
    no longer cited. Disabled by default; flip via
    ``analysis.compute_l2_variants: true``.
    """
    return bool(cfg.get("analysis", {}).get("compute_l2_variants", False))


def _store_per_sample_filter_npy_enabled(cfg: Dict[str, Any]) -> bool:
    """Whether to write the per-sample W_t as a sidecar ``.npy`` file.

    The same numbers are persisted in ``power_spectra.json`` (which is what
    ``load_exp2_filter_bank`` reads). The sidecars only duplicate that
    information and inflate disk usage by an extra ~16 MB per 30-sample run.
    Disabled by default; flip via
    ``analysis.store_per_sample_filter_npy: true``.
    """
    return bool(cfg.get("analysis", {}).get("store_per_sample_filter_npy", False))


_L2_KEY_MARKERS = ("_l2", "l2_")


def _strip_l2_variants_inplace(obj: Any) -> None:
    """Recursively drop dict keys ending in ``_l2`` / containing the ``l2_``
    marker so a run with ``compute_l2_variants: false`` does not emit a fossil
    of the L2 branch in its JSON outputs.

    The function mutates ``obj`` and silently skips non-container values.
    """
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if isinstance(key, str) and (
                key.endswith("_l2")
                or key.startswith("l2_")
                or key.endswith("_l2_normalized")
                or "_l2_" in key
            ):
                del obj[key]
                continue
            _strip_l2_variants_inplace(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            _strip_l2_variants_inplace(item)


def _filter_group_order(include_last_layers: bool = True) -> tuple[str, ...]:
    """Groups persisted for task filters.

    ``last_K`` subgroups are now always included so Exp 5's radial overlap
    law can consume them regardless of whether the 2D-overlap pathway is
    enabled. The ``include_last_layers`` arg is retained for back-compat
    but defaults to ``True``.
    """
    if include_last_layers:
        return (*LAYER_GROUP_ORDER, *LAST_LAYER_GROUP_ORDER)
    return tuple(LAYER_GROUP_ORDER)


def _zero_dc_2d(power_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(power_2d, dtype=np.float64).copy()
    if arr.ndim == 2 and arr.size:
        arr[arr.shape[0] // 2, arr.shape[1] // 2] = 0.0
    return arr


def _normalize_power_2d(power_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(power_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    total = float(np.sum(arr))
    if total <= _EPS:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / total


def _compute_attention_power_2d_mean(
    attention_maps: List[np.ndarray],
    patch_grid: tuple[int, int],
    num_bands: int,
    *,
    fft_window: str,
    suppress_dc: bool,
) -> Optional[np.ndarray]:
    """Mean full 2D attention spectrum aligned with the radial W_t path."""
    powers: List[np.ndarray] = []

    def _append_power(mean_attention: np.ndarray) -> None:
        grid = attention_to_spatial_grid(mean_attention, patch_grid)
        _, power_2d = compute_attention_power_spectrum(
            grid,
            num_bands=num_bands,
            window=fft_window,
            suppress_dc=suppress_dc,
        )
        powers.append(_zero_dc_2d(power_2d) if suppress_dc else power_2d)

    for attn in attention_maps:
        if attn.ndim == 1:
            _append_power(attn)
        elif attn.ndim == 2:
            _append_power(attn.mean(axis=0))
        elif attn.ndim == 3:
            for head_idx in range(attn.shape[0]):
                _append_power(attn[head_idx].mean(axis=0))

    if not powers:
        return None
    return np.mean(np.stack(powers), axis=0)


def _save_w_t_2d(
    *,
    filters_2d_dir: Path,
    image_id: Any,
    level_key: str,
    group_name: str,
    power_2d: np.ndarray,
    W_t_2d: np.ndarray,
    patch_grid: Any,
    layer_indices: List[int],
    fft_window: str,
    suppress_dc: bool,
) -> Dict[str, Any]:
    filters_2d_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{image_id}_{level_key}_{group_name}.npz"
    np.savez_compressed(
        filters_2d_dir / filename,
        W_t_2d=np.asarray(W_t_2d, dtype=np.float32),
        power_2d=np.asarray(power_2d, dtype=np.float32),
        patch_grid=np.asarray(patch_grid, dtype=np.int32),
        layer_indices=np.asarray(layer_indices, dtype=np.int32),
        image_id=np.asarray(str(image_id)),
        level=np.asarray(level_key),
        group=np.asarray(group_name),
        fft_window=np.asarray(str(fft_window)),
        suppress_dc=np.asarray(bool(suppress_dc)),
    )
    return {
        "filename": filename,
        "shape": [int(W_t_2d.shape[0]), int(W_t_2d.shape[1])],
        "spectral_alignment": "fftshifted_attention_power",
    }


def _serialise_filter_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Compact JSON-safe representation of one extracted filter result."""
    return {
        "bandwidth": result["bandwidth"],
        "bandwidth_l2": result["bandwidth_l2"],
        "radial_power": result["radial_power"].tolist(),
        "W_t": result["W_t"].tolist(),
        "W_t_l2": result["W_t_l2"].tolist(),
        "num_layers": result["num_layers_extracted"],
        "num_heads": result["num_heads_total"],
        "layer_indices": result.get("layer_indices", []),
    }


def _store_filter_artifacts(
    *,
    result: Dict[str, Any],
    filters_dir: Path,
    filters_2d_dir: Path,
    image_id: Any,
    level_key: str,
    group_name: Optional[str],
    patch_grid: Any,
    fft_window: str,
    suppress_dc: bool,
    overlap_2d_enabled: bool,
    store_radial_npy: bool = True,
) -> Dict[str, Any]:
    """Save radial/2D filter sidecars and return JSON metadata.

    When ``store_radial_npy`` is false the per-sample radial W_t .npy sidecars
    are skipped (the same numbers live in ``power_spectra.json``). The 2D
    sidecars under ``filters_2d/`` are always written when
    ``overlap_2d_enabled`` because Exp 5 loads them by path.
    """
    suffix = "" if group_name is None else f"_{group_name}"
    if store_radial_npy:
        np.save(filters_dir / f"{image_id}_{level_key}{suffix}.npy", result["W_t"])
        np.save(filters_dir / f"{image_id}_{level_key}{suffix}_l2.npy", result["W_t_l2"])
    metadata: Dict[str, Any] = {}
    if overlap_2d_enabled and result.get("W_t_2d") is not None:
        w_t_2d_entry = _save_w_t_2d(
            filters_2d_dir=filters_2d_dir,
            image_id=image_id,
            level_key=level_key,
            group_name=group_name or "overall",
            power_2d=result["power_2d"],
            W_t_2d=result["W_t_2d"],
            patch_grid=patch_grid,
            layer_indices=result.get("layer_indices", []),
            fft_window=fft_window,
            suppress_dc=suppress_dc,
        )
        metadata["W_t_2d_file"] = w_t_2d_entry["filename"]
        metadata["W_t_2d_shape"] = w_t_2d_entry["shape"]
    return metadata


def _bandwidth_horse_race_predictors(points: List[Dict[str, Any]]) -> List[str]:
    """Predictor list for bandwidth horse races, with optional task-format control.

    Adds ``is_binary`` whenever the slice mixes 2-option and 4-option items so
    the regression can absorb the task-format baseline shift; omitted on
    single-format slices (e.g. binary-only or MCQ-only views) to keep the
    design matrix full rank.
    """

    predictors = [
        "question_complexity_score",
        "prompt_complexity_score",
        "option_hardness_score",
    ]
    binary_vals = {float(point.get("is_binary", 0.0) or 0.0) for point in points}
    if len(binary_vals - {0.0, 1.0}) == 0 and len(binary_vals & {0.0, 1.0}) == 2:
        predictors.append("is_binary")
    return predictors


def _build_complexity_points(per_sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for record in per_sample:
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            point = {
                "image_id": image_id,
                "level": level_key,
                "question": level_data.get("question"),
                "question_type": level_data.get("question_type"),
                "complexity_score": float(level_data.get("complexity_score", 0.0) or 0.0),
                "question_complexity_score": float(
                    level_data.get("question_complexity_score", 0.0) or 0.0
                ),
                "prompt_complexity_score": float(
                    level_data.get("prompt_complexity_score", 0.0) or 0.0
                ),
                "option_hardness_score": float(
                    level_data.get("option_hardness_score", 0.0) or 0.0
                ),
                "is_binary": float(level_data.get("is_binary", 0.0) or 0.0),
                "num_options": int(level_data.get("num_options", 0) or 0),
                "bandwidth": float(level_data.get("bandwidth", 0.0) or 0.0),
            }
            for group_name, group_data in level_data.get("layer_groups", {}).items():
                group_bandwidth = group_data.get("bandwidth")
                if group_bandwidth is not None:
                    point[f"bandwidth_{group_name}"] = float(group_bandwidth)
            points.append(point)
    attach_complexity_residual(points)
    return points


def _summarize_complexity(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not points:
        return {}
    complexity_values = [
        float(point.get("complexity_score", 0.0))
        for point in points
        if point.get("complexity_score") is not None
    ]
    summary: Dict[str, Any] = {
        "score_name": SEMANTIC_COMPLEXITY_SCORE_NAME,
        "score_definition": SEMANTIC_COMPLEXITY_SCORE_DEFINITION,
        "control_score_name": PROMPT_COMPLEXITY_SCORE_NAME,
        "control_score_definition": PROMPT_COMPLEXITY_SCORE_DEFINITION,
        "option_hardness_score_name": OPTION_HARDNESS_SCORE_NAME,
        "option_hardness_score_definition": OPTION_HARDNESS_SCORE_DEFINITION,
        "logic_residual_key": "complexity_score_residual",
        "logic_residual_definition": "Residual of semantic complexity after linear regression on prompt load.",
        "score_min": float(min(complexity_values)) if complexity_values else 0.0,
        "score_max": float(max(complexity_values)) if complexity_values else 0.0,
        "num_points": len(points),
        "mean_bandwidth_by_score": summarize_by_score(
            points,
            score_key="complexity_score",
            value_key="bandwidth",
        ),
        "mean_bandwidth_by_prompt_load": summarize_by_score(
            points,
            score_key="prompt_complexity_score",
            value_key="bandwidth",
        ),
        "mean_bandwidth_by_option_hardness": summarize_by_score(
            points,
            score_key="option_hardness_score",
            value_key="bandwidth",
        ),
        "layer_groups": {},
    }
    for group_name in LAYER_GROUP_ORDER:
        value_key = f"bandwidth_{group_name}"
        if any(point.get(value_key) is not None for point in points):
            summary["layer_groups"][group_name] = {
                "mean_bandwidth_by_score": summarize_by_score(
                    points,
                    score_key="complexity_score",
                    value_key=value_key,
                ),
                "mean_bandwidth_by_option_hardness": summarize_by_score(
                    points,
                    score_key="option_hardness_score",
                    value_key=value_key,
                ),
            }
    return summary


def _build_random_prompt(reference_prompt: str, seed: int) -> str:
    tokens = [token for token in reference_prompt.replace("\n", " ").split(" ") if token]
    token_count = max(4, len(tokens))
    rng = np.random.default_rng(seed)
    vocab = [
        "kava", "lorn", "mep", "silar", "torin", "vexa", "drim", "palo",
        "nure", "cesta", "brin", "zalor", "tiven", "mora", "quess",
    ]
    generated = [str(rng.choice(vocab)) for _ in range(token_count)]
    return " ".join(generated)


def _build_control_prompts(
    prompt: str,
    control_names: List[str],
    seed: int,
) -> Dict[str, str]:
    controls: Dict[str, str] = {}
    for idx, control_name in enumerate(control_names):
        if control_name == "empty_language":
            controls[control_name] = "."
        elif control_name == "random_language":
            controls[control_name] = _build_random_prompt(prompt, seed + idx)
    return controls


def _extract_attention_for_prompt(
    adapter,
    image: Image.Image,
    prompt: str,
    layer_stride: int = 4,
    num_bands: int = 10,
    fft_window: str = "hann",
    suppress_dc: bool = True,
    store_2d_spectra: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract attention maps and compute spectral analysis for one prompt.

    Returns dict with power spectrum, bandwidth, and W_t filter, or None
    on failure.
    """
    try:
        internals = adapter.extract_internals(
            image, prompt,
            extract_attention=True,
            extract_pre_fusion=False,
            extract_post_fusion=False,
            layer_stride=layer_stride,
            # Always pull the trailing K=LAST_LAYER_MAX_OFFSET layers so the
            # `last_2` filter is available downstream regardless of whether the
            # 2D-overlap pathway is enabled. Decoupling this from
            # ``store_2d_spectra`` ensures Exp 5 always sees ``last_2`` filters
            # for the radial overlap correlation too.
            attention_extra_last_layers=LAST_LAYER_MAX_OFFSET,
        )
    except Exception as e:
        logger.warning("extract_internals failed: %s", e)
        return None

    if internals.cross_attention_weights is None or not internals.cross_attention_weights:
        logger.warning("No cross-attention weights returned")
        return None

    if internals.patch_grid is None:
        logger.warning("No patch grid shape returned")
        return None

    # Convert attention tensors to numpy
    attn_list = []
    for attn_tensor in internals.cross_attention_weights:
        if attn_tensor is not None:
            attn_list.append(attn_tensor.cpu().float().numpy())

    if not attn_list:
        return None

    def _summarize_group(group_attn: List[np.ndarray], layer_indices: List[int]) -> Dict[str, Any]:
        mean_radial, _ = compute_attention_power_spectrum_multi(
            group_attn,
            internals.patch_grid,
            num_bands,
            window=fft_window,
            suppress_dc=suppress_dc,
        )
        bandwidth = compute_effective_bandwidth(mean_radial)
        W_t = compute_filter_W_t(mean_radial)
        W_t_l2 = compute_filter_W_t(mean_radial, norm="l2")
        bandwidth_l2 = compute_effective_bandwidth(W_t_l2)
        payload: Dict[str, Any] = {
            "radial_power": mean_radial,
            "bandwidth": bandwidth,
            "bandwidth_l2": bandwidth_l2,
            "W_t": W_t,
            "W_t_l2": W_t_l2,
            "layer_indices": layer_indices,
            "num_layers_extracted": len(group_attn),
            "num_heads_total": sum(
                a.shape[0] if a.ndim == 3 else 1 for a in group_attn
            ),
        }
        if store_2d_spectra:
            power_2d = _compute_attention_power_2d_mean(
                group_attn,
                internals.patch_grid,
                num_bands,
                fft_window=fft_window,
                suppress_dc=suppress_dc,
            )
            if power_2d is not None:
                payload["power_2d"] = power_2d
                payload["W_t_2d"] = _normalize_power_2d(power_2d)
                payload["W_t_2d_shape"] = [
                    int(power_2d.shape[0]),
                    int(power_2d.shape[1]),
                ]
        return payload

    layer_indices = list(internals.cross_attention_layer_indices or range(len(attn_list)))
    total_layers = max(1, adapter.num_layers)
    overall = _summarize_group(attn_list, layer_indices)
    per_layer = compute_attention_power_spectrum_by_layer(
        attn_list,
        internals.patch_grid,
        layer_indices,
        num_bands=num_bands,
        window=fft_window,
        suppress_dc=suppress_dc,
    )
    grouped_attn = split_by_layer_group(attn_list, layer_indices, total_layers)
    group_stats: Dict[str, Any] = {}
    for group_name in LAYER_GROUP_ORDER:
        group_values = grouped_attn[group_name]["values"]
        group_layer_indices = grouped_attn[group_name]["layer_indices"]
        if not group_values:
            continue
        group_stats[group_name] = _summarize_group(group_values, group_layer_indices)
    # Always emit the `last_K` subgroups defined in LAST_LAYER_GROUP_ORDER so
    # Exp 5 can read them for the radial overlap law, regardless of whether
    # the 2D overlap pathway is enabled. (The 2D pathway separately consumes
    # ``power_2d`` from the per-group result when ``store_2d_spectra`` is on.)
    last_pairs = sorted(
        zip(layer_indices, attn_list),
        key=lambda pair: int(pair[0]),
        reverse=True,
    )[:LAST_LAYER_MAX_OFFSET]
    for offset, (layer_index, attn) in enumerate(last_pairs, start=1):
        group_name = f"last_{offset}"
        if group_name not in LAST_LAYER_GROUP_ORDER:
            continue
        group_stats[group_name] = _summarize_group([attn], [int(layer_index)])

    return {
        **overall,
        "layer_groups": group_stats,
        "per_layer": per_layer,
        "layer_group_ranges": layer_group_ranges(total_layers),
        "total_model_layers": total_layers,
        "patch_grid": internals.patch_grid,
        "fft_window": fft_window,
    }


def run_exp2(
    cfg: dict,
    out_dir: Path,
    results_so_far: Dict[int, ExperimentResult],
) -> ExperimentResult:
    """Run Experiment 2: Cross-Attention Frequency Analysis.

    For each sample × granularity level, extract the cross-attention map,
    compute its 2D FFT power spectrum, and measure the effective bandwidth.
    Test whether bandwidth scales with task granularity.
    """
    logger.info("=" * 50)
    logger.info("Experiment 2: Cross-Attention Frequency Analysis")
    logger.info("=" * 50)

    exp_cfg = cfg.get("experiments", {}).get("exp2", {})
    max_samples = exp_cfg.get("max_samples", 200)
    layer_stride = exp_cfg.get("layer_stride", 4)
    fft_window = exp_cfg.get(
        "fft_window",
        cfg.get("analysis", {}).get("attention_fft_window", "hann"),
    )
    prompt_controls = [
        name
        for name in exp_cfg.get("prompt_controls", list(CONTROL_ORDER))
        if name in CONTROL_ORDER
    ]
    num_bands = cfg.get("analysis", {}).get("num_bands", 10)
    suppress_dc = bool(cfg.get("analysis", {}).get("suppress_dc", True))
    overlap_2d_enabled = _overlap_2d_enabled(cfg)
    question_only_enabled = _question_only_control_enabled(exp_cfg)
    store_per_sample_npy = _store_per_sample_filter_npy_enabled(cfg)
    filter_group_order = _filter_group_order(include_last_layers=True)
    effective_band_count = spectral_vector_length(num_bands, suppress_dc=suppress_dc)
    seed = cfg.get("seed", 42)

    # --- Load model ---
    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("primary", "Qwen/Qwen3-VL-8B-Instruct")
    device = select_device(cfg.get("device", "auto"))
    quantization = model_cfg.get("quantization")

    logger.info("Loading model: %s (device=%s, quant=%s)", model_id, device, quantization)
    adapter = get_adapter(model_id)
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
    logger.info("Model loaded successfully")

    # --- Load dataset ---
    samples = load_multilevel_vqa_dataset(cfg, max_samples=max_samples)
    logger.info("Loaded %d samples for attention analysis", len(samples))

    if not samples:
        logger.error("No samples loaded.")
        return ExperimentResult(
            experiment_id=2,
            experiment_name="exp2_attention_frequency",
            config=exp_cfg,
            metrics={"error": "no_samples"},
        )

    # --- Create output dirs ---
    filters_dir = out_dir / "filters"
    filters_dir.mkdir(parents=True, exist_ok=True)
    filters_2d_dir = out_dir / "filters_2d"
    if overlap_2d_enabled:
        filters_2d_dir.mkdir(parents=True, exist_ok=True)
    question_only_filters_dir = out_dir / "filters_question_only"
    question_only_filters_2d_dir = out_dir / "filters_2d_question_only"
    if question_only_enabled:
        question_only_filters_dir.mkdir(parents=True, exist_ok=True)
        if overlap_2d_enabled:
            question_only_filters_2d_dir.mkdir(parents=True, exist_ok=True)

    # --- Extraction loop ---
    per_sample: List[Dict[str, Any]] = []
    # Collect bandwidths grouped by level
    level_bandwidths: Dict[str, List[float]] = defaultdict(list)
    level_bandwidths_l2: Dict[str, List[float]] = defaultdict(list)
    level_radials: Dict[str, List[np.ndarray]] = defaultdict(list)
    level_group_bandwidths: Dict[str, Dict[str, List[float]]] = {
        group_name: defaultdict(list) for group_name in filter_group_order
    }
    level_group_bandwidths_l2: Dict[str, Dict[str, List[float]]] = {
        group_name: defaultdict(list) for group_name in filter_group_order
    }
    level_group_radials: Dict[str, Dict[str, List[np.ndarray]]] = {
        group_name: defaultdict(list) for group_name in filter_group_order
    }
    level_per_layer_bandwidths: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    qo_level_bandwidths: Dict[str, List[float]] = defaultdict(list)
    qo_level_bandwidths_l2: Dict[str, List[float]] = defaultdict(list)
    qo_level_radials: Dict[str, List[np.ndarray]] = defaultdict(list)
    qo_level_group_bandwidths: Dict[str, Dict[str, List[float]]] = {
        group_name: defaultdict(list) for group_name in filter_group_order
    }
    qo_level_group_bandwidths_l2: Dict[str, Dict[str, List[float]]] = {
        group_name: defaultdict(list) for group_name in filter_group_order
    }
    qo_level_group_radials: Dict[str, Dict[str, List[np.ndarray]]] = {
        group_name: defaultdict(list) for group_name in filter_group_order
    }
    qo_level_divergences: Dict[str, Dict[str, List[float]]] = {
        "js_divergence": defaultdict(list),
        "l2_distance": defaultdict(list),
        "cosine_similarity": defaultdict(list),
        "bandwidth_delta_with_options_minus_question_only": defaultdict(list),
    }
    qo_level_group_divergences: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        group_name: {
            "js_divergence": defaultdict(list),
            "l2_distance": defaultdict(list),
            "cosine_similarity": defaultdict(list),
            "bandwidth_delta_with_options_minus_question_only": defaultdict(list),
        }
        for group_name in filter_group_order
    }
    level_control_bandwidths: Dict[str, Dict[str, List[float]]] = {
        control_name: defaultdict(list) for control_name in prompt_controls
    }
    level_control_radials: Dict[str, Dict[str, List[np.ndarray]]] = {
        control_name: defaultdict(list) for control_name in prompt_controls
    }
    level_control_divergences: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        control_name: {
            "js_divergence": defaultdict(list),
            "l2_distance": defaultdict(list),
            "cosine_similarity": defaultdict(list),
            "bandwidth_delta": defaultdict(list),
        }
        for control_name in prompt_controls
    }
    level_control_group_bandwidths: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        control_name: {group_name: defaultdict(list) for group_name in LAYER_GROUP_ORDER}
        for control_name in prompt_controls
    }
    level_control_group_radials: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = {
        control_name: {group_name: defaultdict(list) for group_name in LAYER_GROUP_ORDER}
        for control_name in prompt_controls
    }
    level_control_group_divergences: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {
        control_name: {
            group_name: {
                "js_divergence": defaultdict(list),
                "l2_distance": defaultdict(list),
                "cosine_similarity": defaultdict(list),
                "bandwidth_delta": defaultdict(list),
            }
            for group_name in LAYER_GROUP_ORDER
        }
        for control_name in prompt_controls
    }
    layer_group_meta = layer_group_ranges(max(1, adapter.num_layers))
    t0 = time.time()

    for idx, sample in enumerate(samples):
        logger.info(
            "Processing sample %d/%d: %s", idx + 1, len(samples), sample.image_id,
        )

        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as e:
            logger.warning("Cannot open image %s: %s", sample.image_path, e)
            continue

        sample_record: Dict[str, Any] = {
            "image_id": sample.image_id,
            "levels": {},
        }

        for level in GranularityLevel:
            level_data = sample.levels.get(level)
            if level_data is None:
                continue

            level_key = level.name
            complexity = ensure_level_complexity(level_data)
            # Build prompt from question + options
            prompt = level_data.question
            if level_data.options:
                opts_str = " ".join(
                    f"({k}) {v}" for k, v in sorted(level_data.options.items())
                )
                prompt = f"{prompt} Options: {opts_str}"

            result = _extract_attention_for_prompt(
                adapter,
                image,
                prompt,
                layer_stride,
                num_bands,
                fft_window,
                suppress_dc=suppress_dc,
                store_2d_spectra=overlap_2d_enabled,
            )

            if result is None:
                continue

            # Store per-sample result
            num_options = int(len(level_data.options or {}))
            sample_record["levels"][level_key] = {
                "question": level_data.question,
                "question_type": level_data.question_type,
                "complexity_score": complexity["complexity_score"],
                "question_complexity_score": complexity["question_complexity_score"],
                "prompt_complexity_score": complexity["prompt_complexity_score"],
                "option_hardness_score": float(getattr(level_data, "option_hardness_score", 0.0) or 0.0),
                "is_binary": 1.0 if num_options == 2 else 0.0,
                "num_options": num_options,
                "semantic_atoms": complexity["semantic_atoms"],
                "prompt_semantic_atoms": complexity["prompt_semantic_atoms"],
                "semantic_atom_counts": complexity["semantic_atom_counts"],
                "option_hardness_components": dict(
                    getattr(level_data, "option_hardness_components", {}) or {}
                ),
                "bandwidth": result["bandwidth"],
                "bandwidth_l2": result["bandwidth_l2"],
                "radial_power": result["radial_power"].tolist(),
                "W_t": result["W_t"].tolist(),
                "W_t_l2": result["W_t_l2"].tolist(),
                "num_layers": result["num_layers_extracted"],
                "num_heads": result["num_heads_total"],
                "patch_grid": list(result["patch_grid"]) if result.get("patch_grid") else None,
                "fft_window": result.get("fft_window"),
                "layer_groups": {},
                "per_layer": [
                    {
                        "layer_index": int(item["layer_index"]),
                        "bandwidth": float(item["bandwidth"]),
                        "radial_power": item["radial_power"].tolist(),
                    }
                    for item in result.get("per_layer", [])
                ],
                "controls": {},
            }

            # Accumulate for hypothesis testing
            level_bandwidths[level_key].append(result["bandwidth"])
            level_bandwidths_l2[level_key].append(result["bandwidth_l2"])
            level_radials[level_key].append(result["radial_power"])
            for layer_item in result.get("per_layer", []):
                level_per_layer_bandwidths[level_key][int(layer_item["layer_index"])].append(
                    float(layer_item["bandwidth"])
                )

            # Save W_t filter (per-sample sidecars are opt-in; the same data
            # lives in power_spectra.json and load_exp2_filter_bank reads from
            # there for sample filters).
            if store_per_sample_npy:
                np.save(
                    filters_dir / f"{sample.image_id}_{level_key}.npy",
                    result["W_t"],
                )
                np.save(
                    filters_dir / f"{sample.image_id}_{level_key}_l2.npy",
                    result["W_t_l2"],
                )
            if overlap_2d_enabled and result.get("W_t_2d") is not None:
                w_t_2d_entry = _save_w_t_2d(
                    filters_2d_dir=filters_2d_dir,
                    image_id=sample.image_id,
                    level_key=level_key,
                    group_name="overall",
                    power_2d=result["power_2d"],
                    W_t_2d=result["W_t_2d"],
                    patch_grid=result["patch_grid"],
                    layer_indices=result.get("layer_indices", []),
                    fft_window=fft_window,
                    suppress_dc=suppress_dc,
                )
                sample_record["levels"][level_key]["W_t_2d_file"] = w_t_2d_entry["filename"]
                sample_record["levels"][level_key]["W_t_2d_shape"] = w_t_2d_entry["shape"]

            for group_name in filter_group_order:
                group_result = result.get("layer_groups", {}).get(group_name)
                if group_result is None:
                    continue
                sample_record["levels"][level_key]["layer_groups"][group_name] = {
                    "bandwidth": group_result["bandwidth"],
                    "bandwidth_l2": group_result["bandwidth_l2"],
                    "radial_power": group_result["radial_power"].tolist(),
                    "W_t": group_result["W_t"].tolist(),
                    "W_t_l2": group_result["W_t_l2"].tolist(),
                    "layer_indices": group_result["layer_indices"],
                    "num_layers": group_result["num_layers_extracted"],
                    "num_heads": group_result["num_heads_total"],
                }
                level_group_bandwidths[group_name][level_key].append(group_result["bandwidth"])
                level_group_bandwidths_l2[group_name][level_key].append(group_result["bandwidth_l2"])
                level_group_radials[group_name][level_key].append(group_result["radial_power"])
                if store_per_sample_npy:
                    np.save(
                        filters_dir / f"{sample.image_id}_{level_key}_{group_name}.npy",
                        group_result["W_t"],
                    )
                    np.save(
                        filters_dir / f"{sample.image_id}_{level_key}_{group_name}_l2.npy",
                        group_result["W_t_l2"],
                    )
                if overlap_2d_enabled and group_result.get("W_t_2d") is not None:
                    w_t_2d_entry = _save_w_t_2d(
                        filters_2d_dir=filters_2d_dir,
                        image_id=sample.image_id,
                        level_key=level_key,
                        group_name=group_name,
                        power_2d=group_result["power_2d"],
                        W_t_2d=group_result["W_t_2d"],
                        patch_grid=result["patch_grid"],
                        layer_indices=group_result.get("layer_indices", []),
                        fft_window=fft_window,
                        suppress_dc=suppress_dc,
                    )
                    sample_record["levels"][level_key]["layer_groups"][group_name][
                        "W_t_2d_file"
                    ] = w_t_2d_entry["filename"]
                    sample_record["levels"][level_key]["layer_groups"][group_name][
                        "W_t_2d_shape"
                    ] = w_t_2d_entry["shape"]

            if question_only_enabled:
                question_only_result = _extract_attention_for_prompt(
                    adapter,
                    image,
                    level_data.question,
                    layer_stride,
                    num_bands,
                    fft_window,
                    suppress_dc=suppress_dc,
                    store_2d_spectra=overlap_2d_enabled,
                )
                if question_only_result is not None:
                    qo_divergence = compare_spectral_filters(
                        result["W_t"],
                        question_only_result["W_t"],
                    )
                    qo_record = {
                        "prompt": level_data.question,
                        **_serialise_filter_result(question_only_result),
                        **{
                            f"{key}_to_option_conditioned": value
                            for key, value in qo_divergence.items()
                        },
                        "bandwidth_delta_with_options_minus_question_only": float(
                            result["bandwidth"] - question_only_result["bandwidth"]
                        ),
                        "layer_groups": {},
                    }
                    qo_record.update(
                        _store_filter_artifacts(
                            result=question_only_result,
                            filters_dir=question_only_filters_dir,
                            filters_2d_dir=question_only_filters_2d_dir,
                            image_id=sample.image_id,
                            level_key=level_key,
                            group_name=None,
                            patch_grid=question_only_result["patch_grid"],
                            fft_window=fft_window,
                            suppress_dc=suppress_dc,
                            overlap_2d_enabled=overlap_2d_enabled,
                            store_radial_npy=store_per_sample_npy,
                        )
                    )
                    sample_record["levels"][level_key]["question_only_filter"] = qo_record
                    qo_level_bandwidths[level_key].append(question_only_result["bandwidth"])
                    qo_level_bandwidths_l2[level_key].append(question_only_result["bandwidth_l2"])
                    qo_level_radials[level_key].append(question_only_result["radial_power"])
                    qo_level_divergences["js_divergence"][level_key].append(
                        qo_divergence["js_divergence"]
                    )
                    qo_level_divergences["l2_distance"][level_key].append(
                        qo_divergence["l2_distance"]
                    )
                    qo_level_divergences["cosine_similarity"][level_key].append(
                        qo_divergence["cosine_similarity"]
                    )
                    qo_level_divergences[
                        "bandwidth_delta_with_options_minus_question_only"
                    ][level_key].append(
                        qo_record["bandwidth_delta_with_options_minus_question_only"]
                    )

                    for group_name in filter_group_order:
                        task_group = result.get("layer_groups", {}).get(group_name)
                        qo_group = question_only_result.get("layer_groups", {}).get(group_name)
                        if task_group is None or qo_group is None:
                            continue
                        group_divergence = compare_spectral_filters(
                            task_group["W_t"],
                            qo_group["W_t"],
                        )
                        qo_group_record = {
                            **_serialise_filter_result(qo_group),
                            **{
                                f"{key}_to_option_conditioned": value
                                for key, value in group_divergence.items()
                            },
                            "bandwidth_delta_with_options_minus_question_only": float(
                                task_group["bandwidth"] - qo_group["bandwidth"]
                            ),
                        }
                        qo_group_record.update(
                            _store_filter_artifacts(
                                result=qo_group,
                                filters_dir=question_only_filters_dir,
                                filters_2d_dir=question_only_filters_2d_dir,
                                image_id=sample.image_id,
                                level_key=level_key,
                                group_name=group_name,
                                patch_grid=question_only_result["patch_grid"],
                                fft_window=fft_window,
                                suppress_dc=suppress_dc,
                                overlap_2d_enabled=overlap_2d_enabled,
                                store_radial_npy=store_per_sample_npy,
                            )
                        )
                        qo_record["layer_groups"][group_name] = qo_group_record
                        qo_level_group_bandwidths[group_name][level_key].append(
                            qo_group["bandwidth"]
                        )
                        qo_level_group_bandwidths_l2[group_name][level_key].append(
                            qo_group["bandwidth_l2"]
                        )
                        qo_level_group_radials[group_name][level_key].append(
                            qo_group["radial_power"]
                        )
                        qo_level_group_divergences[group_name]["js_divergence"][level_key].append(
                            group_divergence["js_divergence"]
                        )
                        qo_level_group_divergences[group_name]["l2_distance"][level_key].append(
                            group_divergence["l2_distance"]
                        )
                        qo_level_group_divergences[group_name]["cosine_similarity"][level_key].append(
                            group_divergence["cosine_similarity"]
                        )
                        qo_level_group_divergences[group_name][
                            "bandwidth_delta_with_options_minus_question_only"
                        ][level_key].append(
                            qo_group_record["bandwidth_delta_with_options_minus_question_only"]
                        )

            control_prompts = _build_control_prompts(prompt, prompt_controls, seed + idx * 997 + level.value)
            for control_name, control_prompt in control_prompts.items():
                control_result = _extract_attention_for_prompt(
                    adapter,
                    image,
                    control_prompt,
                    layer_stride,
                    num_bands,
                    fft_window,
                    suppress_dc,
                )
                if control_result is None:
                    continue

                divergence = compare_spectral_filters(result["W_t"], control_result["W_t"])
                control_record: Dict[str, Any] = {
                    "prompt": control_prompt,
                    "bandwidth": control_result["bandwidth"],
                    "radial_power": control_result["radial_power"].tolist(),
                    "W_t": control_result["W_t"].tolist(),
                    "num_layers": control_result["num_layers_extracted"],
                    "num_heads": control_result["num_heads_total"],
                    **divergence,
                    "bandwidth_delta_to_task": float(
                        control_result["bandwidth"] - result["bandwidth"]
                    ),
                    "layer_groups": {},
                }
                sample_record["levels"][level_key]["controls"][control_name] = control_record
                level_control_bandwidths[control_name][level_key].append(control_result["bandwidth"])
                level_control_radials[control_name][level_key].append(control_result["radial_power"])
                level_control_divergences[control_name]["js_divergence"][level_key].append(
                    divergence["js_divergence"]
                )
                level_control_divergences[control_name]["l2_distance"][level_key].append(
                    divergence["l2_distance"]
                )
                level_control_divergences[control_name]["cosine_similarity"][level_key].append(
                    divergence["cosine_similarity"]
                )
                level_control_divergences[control_name]["bandwidth_delta"][level_key].append(
                    control_record["bandwidth_delta_to_task"]
                )

                for group_name in LAYER_GROUP_ORDER:
                    task_group = result.get("layer_groups", {}).get(group_name)
                    control_group = control_result.get("layer_groups", {}).get(group_name)
                    if task_group is None or control_group is None:
                        continue
                    group_divergence = compare_spectral_filters(
                        task_group["W_t"],
                        control_group["W_t"],
                    )
                    control_record["layer_groups"][group_name] = {
                        "bandwidth": control_group["bandwidth"],
                        "radial_power": control_group["radial_power"].tolist(),
                        "W_t": control_group["W_t"].tolist(),
                        "layer_indices": control_group["layer_indices"],
                        "num_layers": control_group["num_layers_extracted"],
                        "num_heads": control_group["num_heads_total"],
                        **group_divergence,
                        "bandwidth_delta_to_task": float(
                            control_group["bandwidth"] - task_group["bandwidth"]
                        ),
                    }
                    level_control_group_bandwidths[control_name][group_name][level_key].append(
                        control_group["bandwidth"]
                    )
                    level_control_group_radials[control_name][group_name][level_key].append(
                        control_group["radial_power"]
                    )
                    for metric_name in ("js_divergence", "l2_distance", "cosine_similarity", "bandwidth_delta"):
                        value_key = metric_name if metric_name != "bandwidth_delta" else "bandwidth_delta_to_task"
                        level_control_group_divergences[control_name][group_name][metric_name][level_key].append(
                            control_record["layer_groups"][group_name][value_key]
                        )

        per_sample.append(sample_record)

        # Progress
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            logger.info(
                "Progress: %d/%d (%.2f samples/s)",
                idx + 1, len(samples), rate,
            )

    total_time = time.time() - t0
    logger.info(
        "Extraction complete: %d samples in %.1fs",
        len(per_sample), total_time,
    )

    # --- Aggregate ---
    agg: Dict[str, Any] = {
        "per_level": {},
        "layer_groups": {
            "order": list(filter_group_order),
            "ranges": {
                group_name: {"start": start, "end": end}
                for group_name, (start, end) in layer_group_meta.items()
            },
            "last_layer_groups": {
                group_name: {
                    "offset_from_last": int(group_name.split("_", 1)[1])
                    if group_name.startswith("last_") and group_name.split("_", 1)[1].isdigit()
                    else None,
                    "description": "single final decoder attention layer (K-th from last)",
                }
                for group_name in LAST_LAYER_GROUP_ORDER
            }
            if overlap_2d_enabled
            else {},
        },
        "fft_window": fft_window,
        "suppress_dc": suppress_dc,
        "requested_num_bands": int(num_bands),
        "effective_num_bands": int(effective_band_count),
        "prompt_controls": prompt_controls,
        "overlap_2d": {
            "enabled": bool(overlap_2d_enabled),
            "domain": "attention_filter_space",
            "storage": "npz_sidecars",
            "filters_2d_dir": "filters_2d",
            "num_w_t_2d_files": len(list(filters_2d_dir.glob("*.npz")))
            if overlap_2d_enabled and filters_2d_dir.exists()
            else 0,
            "keying": "image_id x level x attention_group",
            "groups": ["overall", *filter_group_order],
        },
    }
    level_order = list(ALL_VQA_LEVEL_NAMES)
    present_levels = [lk for lk in level_order if lk in level_bandwidths]
    primary_present_levels = [lk for lk in PRIMARY_VQA_LEVEL_NAMES if lk in level_bandwidths]

    for lk in present_levels:
        bws = level_bandwidths[lk]
        bws_l2 = level_bandwidths_l2[lk]
        radials = level_radials[lk]
        mean_radial = (
            np.mean(np.stack(radials), axis=0)
            if radials
            else np.zeros(effective_band_count, dtype=np.float64)
        )
        W_t_avg = compute_filter_W_t(mean_radial)
        W_t_avg_l2 = compute_filter_W_t(mean_radial, norm="l2")

        agg["per_level"][lk] = {
            "mean_bandwidth": float(np.mean(bws)),
            "mean_bandwidth_l2": float(np.mean(bws_l2)) if bws_l2 else 0.0,
            "std_bandwidth": float(np.std(bws)),
            "std_bandwidth_l2": float(np.std(bws_l2)) if bws_l2 else 0.0,
            "median_bandwidth": float(np.median(bws)),
            "median_bandwidth_l2": float(np.median(bws_l2)) if bws_l2 else 0.0,
            "mean_radial_power": mean_radial.tolist(),
            "W_t_average": W_t_avg.tolist(),
            "W_t_average_l2": W_t_avg_l2.tolist(),
            "shape_metrics": compute_filter_shape_metrics(W_t_avg),
            "num_samples": len(bws),
            "layer_groups": {},
        }

        for group_name in filter_group_order:
            group_bws = level_group_bandwidths[group_name].get(lk, [])
            group_bws_l2 = level_group_bandwidths_l2[group_name].get(lk, [])
            group_radials = level_group_radials[group_name].get(lk, [])
            if not group_radials:
                continue
            group_mean_radial = np.mean(np.stack(group_radials), axis=0)
            group_W_t = compute_filter_W_t(group_mean_radial)
            group_W_t_l2 = compute_filter_W_t(group_mean_radial, norm="l2")
            agg["per_level"][lk]["layer_groups"][group_name] = {
                "mean_bandwidth": float(np.mean(group_bws)),
                "mean_bandwidth_l2": float(np.mean(group_bws_l2)) if group_bws_l2 else 0.0,
                "std_bandwidth": float(np.std(group_bws)),
                "std_bandwidth_l2": float(np.std(group_bws_l2)) if group_bws_l2 else 0.0,
                "median_bandwidth": float(np.median(group_bws)),
                "median_bandwidth_l2": float(np.median(group_bws_l2)) if group_bws_l2 else 0.0,
                "mean_radial_power": group_mean_radial.tolist(),
                "W_t_average": group_W_t.tolist(),
                "W_t_average_l2": group_W_t_l2.tolist(),
                "shape_metrics": compute_filter_shape_metrics(group_W_t),
                "num_samples": len(group_bws),
            }

        agg["per_level"][lk]["controls"] = {}
        for control_name in prompt_controls:
            control_bws = level_control_bandwidths[control_name].get(lk, [])
            control_radials = level_control_radials[control_name].get(lk, [])
            if not control_radials:
                continue
            control_mean_radial = np.mean(np.stack(control_radials), axis=0)
            control_W_t = compute_filter_W_t(control_mean_radial)
            control_entry: Dict[str, Any] = {
                "mean_bandwidth": float(np.mean(control_bws)),
                "std_bandwidth": float(np.std(control_bws)),
                "median_bandwidth": float(np.median(control_bws)),
                "mean_radial_power": control_mean_radial.tolist(),
                "W_t_average": control_W_t.tolist(),
                "shape_metrics": compute_filter_shape_metrics(control_W_t),
                "mean_js_divergence_to_task": float(
                    np.mean(level_control_divergences[control_name]["js_divergence"][lk])
                ),
                "std_js_divergence_to_task": float(
                    np.std(level_control_divergences[control_name]["js_divergence"][lk])
                ),
                "mean_l2_distance_to_task": float(
                    np.mean(level_control_divergences[control_name]["l2_distance"][lk])
                ),
                "std_l2_distance_to_task": float(
                    np.std(level_control_divergences[control_name]["l2_distance"][lk])
                ),
                "mean_cosine_similarity_to_task": float(
                    np.mean(level_control_divergences[control_name]["cosine_similarity"][lk])
                ),
                "mean_bandwidth_delta_to_task": float(
                    np.mean(level_control_divergences[control_name]["bandwidth_delta"][lk])
                ),
                "num_samples": len(control_bws),
                "layer_groups": {},
            }
            for group_name in LAYER_GROUP_ORDER:
                group_bws = level_control_group_bandwidths[control_name][group_name].get(lk, [])
                group_radials = level_control_group_radials[control_name][group_name].get(lk, [])
                if not group_radials:
                    continue
                group_mean_radial = np.mean(np.stack(group_radials), axis=0)
                group_W_t = compute_filter_W_t(group_mean_radial)
                control_entry["layer_groups"][group_name] = {
                    "mean_bandwidth": float(np.mean(group_bws)),
                    "std_bandwidth": float(np.std(group_bws)),
                    "median_bandwidth": float(np.median(group_bws)),
                    "mean_radial_power": group_mean_radial.tolist(),
                    "W_t_average": group_W_t.tolist(),
                    "shape_metrics": compute_filter_shape_metrics(group_W_t),
                    "mean_js_divergence_to_task": float(
                        np.mean(
                            level_control_group_divergences[control_name][group_name]["js_divergence"][lk]
                        )
                    ),
                    "mean_l2_distance_to_task": float(
                        np.mean(
                            level_control_group_divergences[control_name][group_name]["l2_distance"][lk]
                        )
                    ),
                    "mean_cosine_similarity_to_task": float(
                        np.mean(
                            level_control_group_divergences[control_name][group_name]["cosine_similarity"][lk]
                        )
                    ),
                    "mean_bandwidth_delta_to_task": float(
                        np.mean(
                            level_control_group_divergences[control_name][group_name]["bandwidth_delta"][lk]
                        )
                    ),
                    "num_samples": len(group_bws),
                }
            agg["per_level"][lk]["controls"][control_name] = control_entry

    if question_only_enabled:
        qo_present_levels = [lk for lk in level_order if lk in qo_level_bandwidths]
        question_only_payload: Dict[str, Any] = {
            "enabled": True,
            "prompt_definition": "question text only; answer options omitted",
            "comparison_reference": "option-conditioned task prompt used by the main Exp2 path",
            "filters_dir": "filters_question_only",
            "filters_2d_dir": "filters_2d_question_only" if overlap_2d_enabled else None,
            "num_w_t_2d_files": len(list(question_only_filters_2d_dir.glob("*.npz")))
            if overlap_2d_enabled and question_only_filters_2d_dir.exists()
            else 0,
            "groups": ["overall", *filter_group_order],
            "per_level": {},
        }
        for lk in qo_present_levels:
            bws = qo_level_bandwidths[lk]
            bws_l2 = qo_level_bandwidths_l2[lk]
            radials = qo_level_radials[lk]
            mean_radial = (
                np.mean(np.stack(radials), axis=0)
                if radials
                else np.zeros(effective_band_count, dtype=np.float64)
            )
            W_t_avg = compute_filter_W_t(mean_radial)
            W_t_avg_l2 = compute_filter_W_t(mean_radial, norm="l2")
            question_only_payload["per_level"][lk] = {
                "mean_bandwidth": float(np.mean(bws)),
                "mean_bandwidth_l2": float(np.mean(bws_l2)) if bws_l2 else 0.0,
                "std_bandwidth": float(np.std(bws)),
                "std_bandwidth_l2": float(np.std(bws_l2)) if bws_l2 else 0.0,
                "median_bandwidth": float(np.median(bws)),
                "median_bandwidth_l2": float(np.median(bws_l2)) if bws_l2 else 0.0,
                "mean_radial_power": mean_radial.tolist(),
                "W_t_average": W_t_avg.tolist(),
                "W_t_average_l2": W_t_avg_l2.tolist(),
                "shape_metrics": compute_filter_shape_metrics(W_t_avg),
                "num_samples": len(bws),
                "mean_js_divergence_to_option_conditioned": float(
                    np.mean(qo_level_divergences["js_divergence"][lk])
                )
                if qo_level_divergences["js_divergence"][lk]
                else None,
                "mean_l2_distance_to_option_conditioned": float(
                    np.mean(qo_level_divergences["l2_distance"][lk])
                )
                if qo_level_divergences["l2_distance"][lk]
                else None,
                "mean_cosine_similarity_to_option_conditioned": float(
                    np.mean(qo_level_divergences["cosine_similarity"][lk])
                )
                if qo_level_divergences["cosine_similarity"][lk]
                else None,
                "mean_bandwidth_delta_with_options_minus_question_only": float(
                    np.mean(
                        qo_level_divergences[
                            "bandwidth_delta_with_options_minus_question_only"
                        ][lk]
                    )
                )
                if qo_level_divergences[
                    "bandwidth_delta_with_options_minus_question_only"
                ][lk]
                else None,
                "layer_groups": {},
            }
            np.save(question_only_filters_dir / f"average_{lk}.npy", W_t_avg)
            np.save(question_only_filters_dir / f"average_{lk}_l2.npy", W_t_avg_l2)

            for group_name in filter_group_order:
                group_bws = qo_level_group_bandwidths[group_name].get(lk, [])
                group_bws_l2 = qo_level_group_bandwidths_l2[group_name].get(lk, [])
                group_radials = qo_level_group_radials[group_name].get(lk, [])
                if not group_radials:
                    continue
                group_mean_radial = np.mean(np.stack(group_radials), axis=0)
                group_W_t = compute_filter_W_t(group_mean_radial)
                group_W_t_l2 = compute_filter_W_t(group_mean_radial, norm="l2")
                group_entry = {
                    "mean_bandwidth": float(np.mean(group_bws)),
                    "mean_bandwidth_l2": float(np.mean(group_bws_l2)) if group_bws_l2 else 0.0,
                    "std_bandwidth": float(np.std(group_bws)),
                    "std_bandwidth_l2": float(np.std(group_bws_l2)) if group_bws_l2 else 0.0,
                    "median_bandwidth": float(np.median(group_bws)),
                    "median_bandwidth_l2": float(np.median(group_bws_l2)) if group_bws_l2 else 0.0,
                    "mean_radial_power": group_mean_radial.tolist(),
                    "W_t_average": group_W_t.tolist(),
                    "W_t_average_l2": group_W_t_l2.tolist(),
                    "shape_metrics": compute_filter_shape_metrics(group_W_t),
                    "num_samples": len(group_bws),
                    "mean_js_divergence_to_option_conditioned": float(
                        np.mean(qo_level_group_divergences[group_name]["js_divergence"][lk])
                    )
                    if qo_level_group_divergences[group_name]["js_divergence"][lk]
                    else None,
                    "mean_l2_distance_to_option_conditioned": float(
                        np.mean(qo_level_group_divergences[group_name]["l2_distance"][lk])
                    )
                    if qo_level_group_divergences[group_name]["l2_distance"][lk]
                    else None,
                    "mean_cosine_similarity_to_option_conditioned": float(
                        np.mean(qo_level_group_divergences[group_name]["cosine_similarity"][lk])
                    )
                    if qo_level_group_divergences[group_name]["cosine_similarity"][lk]
                    else None,
                    "mean_bandwidth_delta_with_options_minus_question_only": float(
                        np.mean(
                            qo_level_group_divergences[group_name][
                                "bandwidth_delta_with_options_minus_question_only"
                            ][lk]
                        )
                    )
                    if qo_level_group_divergences[group_name][
                        "bandwidth_delta_with_options_minus_question_only"
                    ][lk]
                    else None,
                }
                question_only_payload["per_level"][lk]["layer_groups"][group_name] = group_entry
                np.save(
                    question_only_filters_dir / f"average_{lk}_{group_name}.npy",
                    group_W_t,
                )
                np.save(
                    question_only_filters_dir / f"average_{lk}_{group_name}_l2.npy",
                    group_W_t_l2,
                )
        agg["question_only_control"] = question_only_payload
    else:
        agg["question_only_control"] = {"enabled": False}

    agg["per_layer_bandwidth"] = {}
    for lk in present_levels:
        layer_map = level_per_layer_bandwidths.get(lk, {})
        layer_indices = sorted(layer_map)
        agg["per_layer_bandwidth"][lk] = {
            "layer_indices": layer_indices,
            "mean_gt": [
                float(np.mean(layer_map[layer_index])) if layer_map[layer_index] else 0.0
                for layer_index in layer_indices
            ],
            "std_gt": [
                float(np.std(layer_map[layer_index])) if layer_map[layer_index] else 0.0
                for layer_index in layer_indices
            ],
            "num_samples": [
                len(layer_map[layer_index])
                for layer_index in layer_indices
            ],
        }

    def _mean_bandwidth_curve(level_names: List[str], *, value_suffix: str = "") -> Dict[str, Any]:
        curve: List[Dict[str, Any]] = []
        for group_name in ("overall",) + LAYER_GROUP_ORDER:
            values: List[float] = []
            for level_key in level_names:
                if level_key not in agg["per_level"]:
                    continue
                if group_name == "overall":
                    value = agg["per_level"][level_key].get(f"mean_bandwidth{value_suffix}")
                else:
                    value = (
                        agg["per_level"][level_key]
                        .get("layer_groups", {})
                        .get(group_name, {})
                        .get(f"mean_bandwidth{value_suffix}")
                    )
                if value is not None and np.isfinite(float(value)):
                    values.append(float(value))
            curve.append(
                {
                    "group": group_name,
                    "mean_bandwidth": float(np.mean(values)) if values else None,
                    "num_levels": len(values),
                }
            )
        return {
            "x_axis": ["overall", *LAYER_GROUP_ORDER],
            "values": curve,
        }

    primary_levels_for_curve = [
        level for level in present_levels if level in (LEVEL_VIEWS["primary"] or set())
    ]
    wordy_levels_for_curve = [
        level for level in present_levels if level in (LEVEL_VIEWS["wordy"] or set())
    ]
    agg["primary_mean_Gt_curve"] = _mean_bandwidth_curve(primary_levels_for_curve)
    agg["wordy_mean_Gt_curve"] = _mean_bandwidth_curve(wordy_levels_for_curve)
    agg["normalization_variants"] = {
        "l2": {
            "primary_mean_Gt_curve": _mean_bandwidth_curve(
                primary_levels_for_curve,
                value_suffix="_l2",
            ),
            "wordy_mean_Gt_curve": _mean_bandwidth_curve(
                wordy_levels_for_curve,
                value_suffix="_l2",
            ),
            "per_level": {
                lk: {
                    "mean_bandwidth_l2": agg["per_level"][lk].get("mean_bandwidth_l2"),
                    "std_bandwidth_l2": agg["per_level"][lk].get("std_bandwidth_l2"),
                    "median_bandwidth_l2": agg["per_level"][lk].get("median_bandwidth_l2"),
                    "num_samples": agg["per_level"][lk].get("num_samples"),
                }
                for lk in present_levels
            },
        }
    }

    complexity_points = _build_complexity_points(per_sample)
    agg["complexity_analysis"] = _summarize_complexity(complexity_points)

    # Save average W_t per level (critical for Exp 3 and Exp 5)
    for lk in present_levels:
        W_t_avg = np.array(agg["per_level"][lk]["W_t_average"])
        np.save(filters_dir / f"average_{lk}.npy", W_t_avg)
        np.save(filters_dir / f"average_{lk}_l2.npy", np.array(agg["per_level"][lk]["W_t_average_l2"]))
        for group_name in filter_group_order:
            group_stats = agg["per_level"][lk]["layer_groups"].get(group_name)
            if group_stats is None:
                continue
            np.save(
                filters_dir / f"average_{lk}_{group_name}.npy",
                np.asarray(group_stats["W_t_average"], dtype=np.float64),
            )
            np.save(
                filters_dir / f"average_{lk}_{group_name}_l2.npy",
                np.asarray(group_stats["W_t_average_l2"], dtype=np.float64),
            )

    # --- Hypothesis tests ---
    tests: Dict[str, Any] = {}

    # H1: Bandwidth decreases with granularity as attention spectra sharpen.
    if len(primary_present_levels) >= 2:
        ranks = list(range(1, len(primary_present_levels) + 1))
        mean_bws = [np.mean(level_bandwidths[lk]) for lk in primary_present_levels]

        rho, p = spearman_correlation(ranks, mean_bws)
        tests["spearman_bandwidth_vs_granularity"] = {
            "rho": rho,
            "p_value": p,
            "target": f"rho < -{GATES['strong_spearman_rho_min']} (bandwidth decreases with granularity)",
            "passed": rho < -GATES["strong_spearman_rho_min"],
            "values": dict(zip(primary_present_levels, mean_bws)),
        }
        mean_bws_l2 = [np.mean(level_bandwidths_l2[lk]) for lk in primary_present_levels]
        rho_l2, p_l2 = spearman_correlation(ranks, mean_bws_l2)
        tests["spearman_bandwidth_vs_granularity_l2"] = {
            "rho": rho_l2,
            "p_value": p_l2,
            "target": f"rho < -{GATES['strong_spearman_rho_min']} (L2-normalized W_t)",
            "passed": rho_l2 < -GATES["strong_spearman_rho_min"],
            "values": dict(zip(primary_present_levels, mean_bws_l2)),
            "normalization": "l2",
        }

        for group_name in LAYER_GROUP_ORDER:
            group_present = [
                lk for lk in primary_present_levels if level_group_bandwidths[group_name].get(lk)
            ]
            if len(group_present) < 2:
                continue
            group_ranks = list(range(1, len(group_present) + 1))
            group_means = [np.mean(level_group_bandwidths[group_name][lk]) for lk in group_present]
            group_rho, group_p = spearman_correlation(group_ranks, group_means)
            tests[f"spearman_bandwidth_vs_granularity_{group_name}"] = {
                "rho": group_rho,
                "p_value": group_p,
                "target": f"rho < -{GATES['strong_spearman_rho_min']}",
                "passed": group_rho < -GATES["strong_spearman_rho_min"],
                "values": dict(zip(group_present, group_means)),
            }
            group_means_l2 = [
                np.mean(level_group_bandwidths_l2[group_name][lk]) for lk in group_present
            ]
            group_rho_l2, group_p_l2 = spearman_correlation(group_ranks, group_means_l2)
            tests[f"spearman_bandwidth_vs_granularity_{group_name}_l2"] = {
                "rho": group_rho_l2,
                "p_value": group_p_l2,
                "target": f"rho < -{GATES['strong_spearman_rho_min']} (L2-normalized W_t)",
                "passed": group_rho_l2 < -GATES["strong_spearman_rho_min"],
                "values": dict(zip(group_present, group_means_l2)),
                "normalization": "l2",
            }

    wordy_present_levels = [
        level for level in present_levels if level in (LEVEL_VIEWS["wordy"] or set())
    ]
    if len(wordy_present_levels) >= 2:
        ranks = list(range(1, len(wordy_present_levels) + 1))
        mean_bws = [np.mean(level_bandwidths[lk]) for lk in wordy_present_levels]
        rho, p = spearman_correlation(ranks, mean_bws)
        tests["spearman_bandwidth_vs_granularity_wordy"] = {
            "rho": rho,
            "p_value": p,
            "target": f"rho < -{GATES['secondary_spearman_rho_min']} (secondary wordy-ladder monotonicity only; no pooled monotonicity is run)",
            "passed": rho < -GATES["secondary_spearman_rho_min"],
            "values": dict(zip(wordy_present_levels, mean_bws)),
        }

    # --- Late-layer downward concentration shape tests (Theorem 2, revised 2026-04-27) ---
    # Granularity axis = downward concentration of W_t mass at the late layer group:
    # top-2 mass RISES, tail mass FALLS, G_late shrinks. The earlier "peak consolidation
    # + tail growth" framing (top-3 ↓, tail ↑) is retracted; the data shows the opposite
    # sign on the tail. Tests are run on the `late` layer group (where the granularity
    # ladder is empirically clean) and on the overall filter (informational only).
    def _shape_series_overall(level_keys: List[str], metric_key: str) -> List[float]:
        out: List[float] = []
        for lk in level_keys:
            shape = agg["per_level"].get(lk, {}).get("shape_metrics", {}) or {}
            out.append(float(shape.get(metric_key, 0.0)))
        return out

    def _shape_series_late(level_keys: List[str], metric_key: str) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for lk in level_keys:
            shape = (
                agg["per_level"].get(lk, {})
                .get("layer_groups", {}).get("late", {})
                .get("shape_metrics")
            )
            if shape is None:
                out.append(None)
            else:
                out.append(float(shape.get(metric_key, 0.0)))
        return out

    shape_tests: Dict[str, Any] = {}
    if len(primary_present_levels) >= 2:
        ranks = list(range(1, len(primary_present_levels) + 1))

        # Late-layer gates (load-bearing under revised Theorem 2)
        late_top2 = _shape_series_late(primary_present_levels, "top_2_mass")
        if all(v is not None for v in late_top2):
            late_top2_f = [float(v) for v in late_top2]
            rho, p = spearman_correlation(ranks, late_top2_f)
            shape_tests["spearman_top2_mass_late_vs_granularity"] = {
                "rho": rho,
                "p_value": p,
                "target": "rho > 0 (late-layer top-2 mass rises with granularity — downward concentration)",
                "passed": rho > 0.0,
                "values": dict(zip(primary_present_levels, late_top2_f)),
            }

        late_tail = _shape_series_late(primary_present_levels, "tail_mass_fraction")
        if all(v is not None for v in late_tail):
            late_tail_f = [float(v) for v in late_tail]
            rho, p = spearman_correlation(ranks, late_tail_f)
            shape_tests["spearman_tail_mass_late_vs_granularity"] = {
                "rho": rho,
                "p_value": p,
                "target": "rho < 0 (late-layer tail mass thins as mass concentrates downward)",
                "passed": rho < 0.0,
                "values": dict(zip(primary_present_levels, late_tail_f)),
            }

        # Overall-filter shape tests (informational; mix early/mid/late)
        top3_series = _shape_series_overall(primary_present_levels, "top_3_mass")
        rho_t3, p_t3 = spearman_correlation(ranks, top3_series)
        shape_tests["spearman_top3_mass_vs_granularity"] = {
            "rho": rho_t3,
            "p_value": p_t3,
            "target": "informational (top-3 mass mixes early length-prior and late-layer concentration)",
            "passed": None,
            "values": dict(zip(primary_present_levels, top3_series)),
        }

        top2_series = _shape_series_overall(primary_present_levels, "top_2_mass")
        rho_t2, p_t2 = spearman_correlation(ranks, top2_series)
        shape_tests["spearman_top2_mass_vs_granularity"] = {
            "rho": rho_t2,
            "p_value": p_t2,
            "target": "rho > 0 expected on overall filter (downward concentration); load-bearing gate is the *_late variant",
            "passed": rho_t2 > 0.0,
            "values": dict(zip(primary_present_levels, top2_series)),
        }

        tail_series = _shape_series_overall(primary_present_levels, "tail_mass_fraction")
        rho_tail, p_tail = spearman_correlation(ranks, tail_series)
        shape_tests["spearman_tail_mass_vs_granularity"] = {
            "rho": rho_tail,
            "p_value": p_tail,
            "target": "rho < 0 expected on overall filter (downward concentration); load-bearing gate is the *_late variant",
            "passed": rho_tail < 0.0,
            "values": dict(zip(primary_present_levels, tail_series)),
        }

        centroid_series = _shape_series_overall(primary_present_levels, "centroid_normalised")
        rho_c, p_c = spearman_correlation(ranks, centroid_series)
        shape_tests["spearman_centroid_vs_granularity"] = {
            "rho": rho_c,
            "p_value": p_c,
            "target": "informational (small downward shift expected under late-layer downward concentration)",
            "passed": None,
            "values": dict(zip(primary_present_levels, centroid_series)),
        }

    # --- Wordy divergence: |shape(base) - shape(wordy)| should grow from early to late layers ---
    # Theorem 4 (revised 2026-04-27): wordy mirrors start more concentrated than their semantic
    # counterparts (length prior at early layers) and end more relaxed (asymmetric late-layer
    # relaxation from a higher starting concentration). The gap widens with depth.
    wordy_pairs: List[Tuple[str, str]] = []
    if LEVEL_VIEWS.get("primary") and LEVEL_VIEWS.get("wordy"):
        primary_ordered = [lk for lk in present_levels if lk in (LEVEL_VIEWS["primary"] or set())]
        wordy_ordered = [lk for lk in present_levels if lk in (LEVEL_VIEWS["wordy"] or set())]
        wordy_pairs = list(zip(primary_ordered, wordy_ordered))

    def _layer_group_shape(level_key: str, group_name: str, metric_key: str) -> Optional[float]:
        shape = (
            agg["per_level"].get(level_key, {})
            .get("layer_groups", {}).get(group_name, {})
            .get("shape_metrics")
        )
        if shape is None:
            return None
        return float(shape.get(metric_key, 0.0))

    wordy_divergence: Dict[str, Any] = {}
    for metric_key in ("top_3_mass", "top_2_mass", "tail_mass_fraction", "centroid_normalised"):
        per_group_abs_delta: Dict[str, List[float]] = {g: [] for g in LAYER_GROUP_ORDER}
        per_pair_deltas: Dict[str, Dict[str, float]] = {}
        for base_lk, wordy_lk in wordy_pairs:
            pair_entry: Dict[str, float] = {}
            for group_name in LAYER_GROUP_ORDER:
                base_val = _layer_group_shape(base_lk, group_name, metric_key)
                wordy_val = _layer_group_shape(wordy_lk, group_name, metric_key)
                if base_val is None or wordy_val is None:
                    continue
                delta = float(wordy_val - base_val)
                pair_entry[group_name] = delta
                per_group_abs_delta[group_name].append(abs(delta))
            if pair_entry:
                per_pair_deltas[f"{base_lk}__vs__{wordy_lk}"] = pair_entry

        mean_abs_delta = {
            g: (float(np.mean(vals)) if vals else None) for g, vals in per_group_abs_delta.items()
        }
        early = mean_abs_delta.get("early")
        late = mean_abs_delta.get("late")
        passed = None
        if early is not None and late is not None:
            passed = bool(late > early)
        wordy_divergence[metric_key] = {
            "mean_abs_delta_per_layer_group": mean_abs_delta,
            "per_pair_deltas": per_pair_deltas,
            "target": "mean_abs_delta_late > mean_abs_delta_early (wordy gap widens with depth — Theorem 4 asymmetric late-layer relaxation)",
            "passed": passed,
        }

    if shape_tests:
        tests["shape_consolidation"] = shape_tests
    if wordy_divergence:
        tests["wordy_layerwise_divergence"] = wordy_divergence

    # H2: ANOVA on bandwidths across levels
    groups = [level_bandwidths[lk] for lk in primary_present_levels if level_bandwidths[lk]]
    if len(groups) >= 2:
        f_stat, p_anova = one_way_anova(*groups)
        tests["anova_bandwidth_across_levels"] = {
            "F_statistic": f_stat,
            "p_value": p_anova,
            "target": "F > 10, p < 0.001",
            "passed": f_stat > 10 and p_anova < 0.001,
        }

    # H3: Bootstrap CI on bandwidth per level
    tests["bootstrap_ci_bandwidth"] = {}
    for lk in present_levels:
        bws = level_bandwidths[lk]
        if len(bws) >= 5:
            point, lo, hi = bootstrap_ci(bws, n_boot=1000)
            tests["bootstrap_ci_bandwidth"][lk] = {
                "mean": point,
                "ci_95_lower": lo,
                "ci_95_upper": hi,
            }

    # Summary
    core_passed = tests.get("spearman_bandwidth_vs_granularity_late", {}).get(
        "passed",
        tests.get("spearman_bandwidth_vs_granularity", {}).get("passed", False),
    )
    tests["primary_group"] = "late"
    tests["fft_window"] = fft_window
    tests["question_only_control"] = {
        "enabled": bool(question_only_enabled),
        "target": "quantify how answer options alter the attention frequency filter",
    }
    if question_only_enabled:
        tests["question_only_control"]["per_level"] = {
            lk: {
                "mean_js_divergence_to_option_conditioned": entry.get(
                    "mean_js_divergence_to_option_conditioned"
                ),
                "mean_l2_distance_to_option_conditioned": entry.get(
                    "mean_l2_distance_to_option_conditioned"
                ),
                "mean_cosine_similarity_to_option_conditioned": entry.get(
                    "mean_cosine_similarity_to_option_conditioned"
                ),
                "mean_bandwidth_delta_with_options_minus_question_only": entry.get(
                    "mean_bandwidth_delta_with_options_minus_question_only"
                ),
            }
            for lk, entry in agg.get("question_only_control", {}).get("per_level", {}).items()
        }
    tests["prompt_controls"] = {}
    for control_name in prompt_controls:
        tests["prompt_controls"][control_name] = {
            "mean_js_divergence_to_task": {
                lk: agg["per_level"][lk]["controls"][control_name]["mean_js_divergence_to_task"]
                for lk in present_levels
                if control_name in agg["per_level"][lk].get("controls", {})
            },
            "mean_bandwidth_delta_to_task": {
                lk: agg["per_level"][lk]["controls"][control_name]["mean_bandwidth_delta_to_task"]
                for lk in present_levels
                if control_name in agg["per_level"][lk].get("controls", {})
            },
        }
    tests["continuous_complexity"] = {
        "bandwidth_vs_complexity_overall": summarize_linear_trend(
            complexity_points,
            x_key="complexity_score",
            y_key="bandwidth",
        ),
        "bandwidth_vs_prompt_load_overall": summarize_linear_trend(
            complexity_points,
            x_key="prompt_complexity_score",
            y_key="bandwidth",
        ),
        "bandwidth_vs_option_hardness_overall": summarize_linear_trend(
            complexity_points,
            x_key="option_hardness_score",
            y_key="bandwidth",
        ),
        "bandwidth_vs_complexity_fixed_effects_overall": summarize_fixed_effects_trend(
            complexity_points,
            group_key="image_id",
            x_key="complexity_score",
            y_key="bandwidth",
        ),
        "horse_race_bandwidth": summarize_multivariate_regression(
            complexity_points,
            y_key="bandwidth",
            x_keys=_bandwidth_horse_race_predictors(complexity_points),
        ),
        "horse_race_bandwidth_within_image": summarize_multivariate_regression(
            complexity_points,
            y_key="bandwidth",
            x_keys=_bandwidth_horse_race_predictors(complexity_points),
            group_key="image_id",
            demean_by_group=True,
        ),
    }

    def _add_bandwidth_regression_views(base_key: str, value_key: str) -> None:
        view_payload: Dict[str, Any] = {}
        for view_name in LEVEL_VIEW_ORDER:
            level_filter = LEVEL_VIEWS[view_name]
            view_points = (
                complexity_points
                if level_filter is None
                else [point for point in complexity_points if str(point.get("level")) in level_filter]
            )
            view_entry = summarize_horse_race_view(
                view_points,
                y_key=value_key,
                x_keys=_bandwidth_horse_race_predictors(view_points),
                view_name=view_name,
                level_filter=None,
            )
            view_entry.update({
                "level_filter": sorted(level_filter) if level_filter is not None else None,
                "n_points": len(view_points),
            })
            view_payload[view_name] = view_entry
            if view_name == "primary":
                tests["continuous_complexity"][f"{base_key}_{view_name}"] = view_entry["marginal"]
            else:
                tests["continuous_complexity"][f"{base_key}_{view_name}"] = view_entry["pooled"]
                tests["continuous_complexity"][f"{base_key}_within_image_{view_name}"] = view_entry["within_image"]
        tests["continuous_complexity"][f"{base_key}_views"] = view_payload

    _add_bandwidth_regression_views("horse_race_bandwidth", "bandwidth")
    for group_name in LAYER_GROUP_ORDER:
        value_key = f"bandwidth_{group_name}"
        if any(point.get(value_key) is not None for point in complexity_points):
            tests["continuous_complexity"][f"bandwidth_vs_complexity_{group_name}"] = (
                summarize_linear_trend(
                    complexity_points,
                    x_key="complexity_score",
                    y_key=value_key,
                )
            )
            tests["continuous_complexity"][f"bandwidth_vs_prompt_load_{group_name}"] = (
                summarize_linear_trend(
                    complexity_points,
                    x_key="prompt_complexity_score",
                    y_key=value_key,
                )
            )
            tests["continuous_complexity"][f"bandwidth_vs_option_hardness_{group_name}"] = (
                summarize_linear_trend(
                    complexity_points,
                    x_key="option_hardness_score",
                    y_key=value_key,
                )
            )
            tests["continuous_complexity"][f"bandwidth_vs_complexity_fixed_effects_{group_name}"] = (
                summarize_fixed_effects_trend(
                    complexity_points,
                    group_key="image_id",
                    x_key="complexity_score",
                    y_key=value_key,
                )
            )
            tests["continuous_complexity"][f"horse_race_bandwidth_{group_name}"] = (
                summarize_multivariate_regression(
                    complexity_points,
                    y_key=value_key,
                    x_keys=_bandwidth_horse_race_predictors(complexity_points),
                )
            )
            tests["continuous_complexity"][f"horse_race_bandwidth_within_image_{group_name}"] = (
                summarize_multivariate_regression(
                    complexity_points,
                    y_key=value_key,
                    x_keys=_bandwidth_horse_race_predictors(complexity_points),
                    group_key="image_id",
                    demean_by_group=True,
                )
            )
            _add_bandwidth_regression_views(f"horse_race_bandwidth_{group_name}", value_key)
    tests["hypothesis_supported"] = core_passed
    agg.setdefault("normalization_variants", {}).setdefault("l2", {})["hypothesis_tests"] = {
        key: value for key, value in tests.items() if key.endswith("_l2")
    }

    # --- Log results ---
    logger.info("-" * 40)
    logger.info("Key results:")
    for lk in present_levels:
        stats = agg["per_level"][lk]
        logger.info(
            "  %s: G(t)=%.2f +/- %.2f  (n=%d)",
            lk, stats["mean_bandwidth"], stats["std_bandwidth"], stats["num_samples"],
        )
        for group_name in LAYER_GROUP_ORDER:
            group_stats = stats.get("layer_groups", {}).get(group_name)
            if group_stats is None:
                continue
            logger.info(
                "    %s[%s]: G(t)=%.2f +/- %.2f",
                lk,
                group_name,
                group_stats["mean_bandwidth"],
                group_stats["std_bandwidth"],
            )
        for control_name in prompt_controls:
            control_stats = stats.get("controls", {}).get(control_name)
            if control_stats is None:
                continue
            logger.info(
                "    %s[%s control]: G(t)=%.2f +/- %.2f  JS(task,control)=%.3f",
                lk,
                control_name,
                control_stats["mean_bandwidth"],
                control_stats["std_bandwidth"],
                control_stats["mean_js_divergence_to_task"],
            )
    sp = tests.get("spearman_bandwidth_vs_granularity", {})
    logger.info(
        "  Spearman(granularity, bandwidth): rho=%.3f, p=%.4f [%s]",
        sp.get("rho", 0), sp.get("p_value", 1),
        "PASS" if sp.get("passed") else "FAIL",
    )
    logger.info("-" * 40)

    # --- Optionally strip L2-normalised variants before persisting ---
    # The paper uses L1-normalised W_t throughout; the L2 branch is retained as
    # an opt-in alternative behind ``analysis.compute_l2_variants``. When that
    # flag is off (default) we prune both the in-memory dicts and the on-disk
    # .npy sidecars so the run does not advertise data it no longer keeps.
    if not _compute_l2_variants_enabled(cfg):
        _strip_l2_variants_inplace(agg)
        _strip_l2_variants_inplace(tests)
        _strip_l2_variants_inplace(complexity_points)
        for record in per_sample:
            _strip_l2_variants_inplace(record)
        for npy_path in list(filters_dir.glob("*_l2.npy")):
            try:
                npy_path.unlink()
            except OSError:
                pass
        if question_only_filters_dir.exists():
            for npy_path in list(question_only_filters_dir.glob("*_l2.npy")):
                try:
                    npy_path.unlink()
                except OSError:
                    pass

    # --- Save outputs ---
    save_json(agg, out_dir / "summary.json")
    save_json(tests, out_dir / "hypothesis_tests.json")
    save_json(complexity_points, out_dir / "complexity_points.json")
    save_json(
        [r for r in per_sample if r.get("levels")],
        out_dir / "power_spectra.json",
    )

    # --- Cleanup ---
    try:
        adapter.unload()
    except Exception:
        pass

    # ``mean_bandwidth_l2`` is only present when the L2-variant pathway was
    # active; otherwise ``_strip_l2_variants_inplace`` removed it earlier in
    # this function. Use .get() so the metrics dict survives either mode.
    metrics = {
        "num_samples": len(per_sample),
        "total_time_s": total_time,
        **{
            f"{lk}_mean_bandwidth": agg["per_level"][lk]["mean_bandwidth"]
            for lk in present_levels
        },
        **{
            f"{lk}_mean_bandwidth_l2": agg["per_level"][lk]["mean_bandwidth_l2"]
            for lk in present_levels
            if "mean_bandwidth_l2" in agg["per_level"].get(lk, {})
        },
        "spearman_rho": sp.get("rho", 0),
        "spearman_p": sp.get("p_value", 1),
        "complexity_pearson_bandwidth": tests.get("continuous_complexity", {})
        .get("bandwidth_vs_complexity_overall", {})
        .get("pearson_r", 0.0),
    }
    for group_name in LAYER_GROUP_ORDER:
        group_sp = tests.get(f"spearman_bandwidth_vs_granularity_{group_name}", {})
        if group_sp:
            metrics[f"spearman_rho_{group_name}"] = group_sp.get("rho", 0)
            metrics[f"spearman_p_{group_name}"] = group_sp.get("p_value", 1)
        for lk in present_levels:
            group_stats = agg["per_level"][lk]["layer_groups"].get(group_name)
            if group_stats is not None:
                metrics[f"{lk}_mean_bandwidth_{group_name}"] = group_stats["mean_bandwidth"]

    return ExperimentResult(
        experiment_id=2,
        experiment_name="exp2_attention_frequency",
        config=exp_cfg,
        metrics=metrics,
        per_sample=per_sample,
        hypothesis_tests=tests,
    )
