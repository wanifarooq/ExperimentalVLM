"""Spectral analysis of cross-attention maps.

This module contains the core novel analysis: computing the frequency structure
of VLM cross-attention maps to derive the task-specific frequency filter W_t(omega).

Key functions:
    - ``attention_to_spatial_grid``: reshape flattened attention to 2D spatial map
    - ``compute_attention_power_spectrum``: 2D FFT + radial band energies
    - ``compute_effective_bandwidth``: inverse participation ratio G(t)
    - ``compute_filter_W_t``: averaged normalized attention spectrum per task
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
_EPS = 1e-8


def spectral_vector_length(
    num_bands: int,
    suppress_dc: bool = False,
) -> int:
    return max(0, int(num_bands))


def spectral_band_centers(
    num_bands: int,
    suppress_dc: bool = False,
) -> np.ndarray:
    total_bands = max(0, int(num_bands))
    if total_bands == 0:
        return np.zeros(0, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, total_bands + 1, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def _normalize_window_name(window: Optional[str]) -> str:
    if window is None:
        return "none"
    if isinstance(window, bool):
        return "hann" if window else "none"

    name = str(window).strip().lower()
    if name in ("", "none", "off", "false", "0", "rect", "rectangular"):
        return "none"
    if name in ("on", "true", "1"):
        return "hann"
    return name


def _window_1d(length: int, window: Optional[str]) -> np.ndarray:
    if length <= 1:
        return np.ones(max(1, length), dtype=np.float64)

    name = _normalize_window_name(window)
    if name == "none":
        return np.ones(length, dtype=np.float64)
    if name in ("hann", "hanning"):
        return np.hanning(length).astype(np.float64)
    if name == "hamming":
        return np.hamming(length).astype(np.float64)
    raise ValueError(f"Unsupported window: {window}")


def build_window_2d(
    shape: Tuple[int, int],
    window: Optional[str] = "hann",
) -> np.ndarray:
    h, w = shape
    wy = _window_1d(h, window)
    wx = _window_1d(w, window)
    return np.outer(wy, wx)


def apply_window_2d(
    array_2d: np.ndarray,
    window: Optional[str] = "hann",
) -> np.ndarray:
    arr = np.asarray(array_2d, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for windowing, got shape {arr.shape}")
    if _normalize_window_name(window) == "none":
        return arr
    return arr * build_window_2d((arr.shape[0], arr.shape[1]), window)


def _radially_bin_power(
    power_2d: np.ndarray,
    num_bands: int,
    suppress_dc: bool = False,
) -> np.ndarray:
    power = np.asarray(power_2d, dtype=np.float64)
    h, w = power_2d.shape
    cy, cx = h // 2, w // 2
    if suppress_dc and h > 0 and w > 0:
        power = power.copy()
        power[cy, cx] = 0.0
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)
    if max_dist <= _EPS:
        radial_power = np.zeros(num_bands, dtype=np.float64)
        if num_bands > 0:
            radial_power[0] = float(power.sum())
        return radial_power

    edges = np.linspace(0, max_dist, num_bands + 1)
    radial_power = np.zeros(num_bands, dtype=np.float64)
    for band_idx in range(num_bands):
        if band_idx == num_bands - 1:
            mask = (dist >= edges[band_idx]) & (dist <= edges[band_idx + 1])
        else:
            mask = (dist >= edges[band_idx]) & (dist < edges[band_idx + 1])
        count = mask.sum()
        if count > 0:
            radial_power[band_idx] = power[mask].sum()
    return radial_power


def radially_bin_power(
    power_2d: np.ndarray,
    num_bands: int,
    suppress_dc: bool = True,
) -> np.ndarray:
    return _radially_bin_power(power_2d, num_bands, suppress_dc=suppress_dc)


def _distribution_cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    usable = min(len(p), len(q))
    if usable == 0:
        return 0.0
    p = p[:usable]
    q = q[:usable]
    denom = np.linalg.norm(p) * np.linalg.norm(q)
    if denom <= _EPS:
        return 0.0
    return float(np.dot(p, q) / denom)


def compute_distribution_js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = _EPS,
) -> float:
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    usable = min(len(p_arr), len(q_arr))
    if usable == 0:
        return 0.0
    p_arr = p_arr[:usable]
    q_arr = q_arr[:usable]
    p_sum = p_arr.sum()
    q_sum = q_arr.sum()
    if p_sum <= eps or q_sum <= eps:
        return 0.0
    p_norm = np.clip(p_arr / p_sum, eps, None)
    q_norm = np.clip(q_arr / q_sum, eps, None)
    p_norm /= p_norm.sum()
    q_norm /= q_norm.sum()
    mean = 0.5 * (p_norm + q_norm)
    kl_pm = np.sum(p_norm * np.log(p_norm / mean))
    kl_qm = np.sum(q_norm * np.log(q_norm / mean))
    return float(0.5 * (kl_pm + kl_qm))


def compare_spectral_filters(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    usable = min(len(ref), len(cand))
    if usable == 0:
        return {"l2_distance": 0.0, "cosine_similarity": 0.0, "js_divergence": 0.0}
    ref = ref[:usable]
    cand = cand[:usable]
    return {
        "l2_distance": float(np.linalg.norm(ref - cand)),
        "cosine_similarity": _distribution_cosine_similarity(ref, cand),
        "js_divergence": compute_distribution_js_divergence(ref, cand),
    }


def _features_to_spatial_grid(
    features: np.ndarray,
    patch_grid: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and patch_grid is not None:
        h, w = patch_grid
        if arr.shape[:2] == (h, w):
            return arr
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 3 and patch_grid is None and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        n_tokens = arr.shape[0]
        if patch_grid is not None and patch_grid[0] * patch_grid[1] <= n_tokens:
            h, w = patch_grid
        else:
            h = max(1, int(np.sqrt(n_tokens)))
            w = max(1, n_tokens // h)
        usable = min(n_tokens, h * w)
        return arr[:usable].reshape(h, w, -1)
    return arr


def _resize_spatial_grid(
    features: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    if features.shape[:2] == (target_h, target_w):
        return features
    y_idx = np.linspace(0, features.shape[0] - 1, target_h).round().astype(int)
    x_idx = np.linspace(0, features.shape[1] - 1, target_w).round().astype(int)
    return features[y_idx][:, x_idx, :]


def _align_feature_grids(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
    clean_patch_grid: Optional[Tuple[int, int]] = None,
    perturbed_patch_grid: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    clean = _features_to_spatial_grid(clean_features, clean_patch_grid)
    perturbed = _features_to_spatial_grid(perturbed_features, perturbed_patch_grid)

    if clean.ndim == 3 and perturbed.ndim == 3:
        target_h = min(clean.shape[0], perturbed.shape[0])
        target_w = min(clean.shape[1], perturbed.shape[1])
        return (
            _resize_spatial_grid(clean, target_h, target_w),
            _resize_spatial_grid(perturbed, target_h, target_w),
        )

    clean_tokens = clean.reshape(-1, clean.shape[-1])
    perturbed_tokens = perturbed.reshape(-1, perturbed.shape[-1])
    usable = min(clean_tokens.shape[0], perturbed_tokens.shape[0])
    side = max(1, int(np.sqrt(usable)))
    width = max(1, usable // side)
    usable = min(usable, side * width)
    return (
        clean_tokens[:usable].reshape(side, width, -1),
        perturbed_tokens[:usable].reshape(side, width, -1),
    )


def attention_to_spatial_grid(
    attention: np.ndarray,
    patch_grid: Tuple[int, int],
) -> np.ndarray:
    """Reshape flattened attention weights to a 2D spatial map.

    Args:
        attention: Attention weights with shape ``(..., num_vis_tokens)``.
            The last dimension corresponds to flattened patch tokens.
        patch_grid: ``(H_patches, W_patches)`` grid shape.

    Returns:
        Reshaped array with last two dims ``(H_patches, W_patches)``.
        Leading dimensions are preserved.
    """
    h, w = patch_grid
    n_vis = h * w
    if attention.shape[-1] < n_vis:
        # Pad with zeros if there's a CLS token mismatch
        pad_width = [(0, 0)] * (attention.ndim - 1) + [(0, n_vis - attention.shape[-1])]
        attention = np.pad(attention, pad_width, mode="constant")
    elif attention.shape[-1] > n_vis:
        # Truncate (extra tokens like CLS)
        attention = attention[..., :n_vis]

    new_shape = attention.shape[:-1] + (h, w)
    return attention.reshape(new_shape)


def compute_attention_power_spectrum(
    attention_2d: np.ndarray,
    num_bands: int = 10,
    window: Optional[str] = "hann",
    suppress_dc: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D FFT power spectrum of a spatial attention map.

    Args:
        attention_2d: 2D array of shape ``(H, W)`` representing one
            attention map in spatial layout.
        num_bands: Number of radial frequency bands.

    Returns:
        ``(radial_power, full_power_2d)`` where ``radial_power`` has shape
        ``(num_bands,)`` (summed band energy) and ``full_power_2d``
        is the full 2D power spectrum ``(H, W)``.
    """
    h, w = attention_2d.shape
    windowed = apply_window_2d(attention_2d, window=window)
    fft = np.fft.fft2(windowed.astype(np.float64))
    fft_shifted = np.fft.fftshift(fft)
    power_2d = np.abs(fft_shifted) ** 2
    radial_power = _radially_bin_power(power_2d, num_bands, suppress_dc=suppress_dc)
    return radial_power, power_2d


def compute_attention_power_spectrum_multi(
    attention_maps: List[np.ndarray],
    patch_grid: Tuple[int, int],
    num_bands: int = 10,
    window: Optional[str] = "hann",
    suppress_dc: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute and average power spectra across multiple attention heads/layers.

    Args:
        attention_maps: List of attention weight arrays, each with shape
            ``(num_heads, num_lang_tokens, num_vis_tokens)`` or
            ``(num_vis_tokens,)``.
        patch_grid: ``(H_patches, W_patches)``.
        num_bands: Number of radial frequency bands.

    Returns:
        ``(mean_radial_power, per_head_radials)`` where ``mean_radial_power``
        is the average across all heads and layers.
    """
    all_radials: List[np.ndarray] = []

    for attn in attention_maps:
        if attn.ndim == 1:
            # Single vector -> reshape to spatial
            grid = attention_to_spatial_grid(attn, patch_grid)
            radial, _ = compute_attention_power_spectrum(
                grid,
                num_bands,
                window=window,
                suppress_dc=suppress_dc,
            )
            all_radials.append(radial)
        elif attn.ndim == 2:
            # (lang_tokens, vis_tokens) -> average over lang tokens first
            mean_attn = attn.mean(axis=0)
            grid = attention_to_spatial_grid(mean_attn, patch_grid)
            radial, _ = compute_attention_power_spectrum(
                grid,
                num_bands,
                window=window,
                suppress_dc=suppress_dc,
            )
            all_radials.append(radial)
        elif attn.ndim == 3:
            # (heads, lang_tokens, vis_tokens)
            for head_idx in range(attn.shape[0]):
                mean_attn = attn[head_idx].mean(axis=0)  # avg over lang tokens
                grid = attention_to_spatial_grid(mean_attn, patch_grid)
                radial, _ = compute_attention_power_spectrum(
                    grid,
                    num_bands,
                    window=window,
                    suppress_dc=suppress_dc,
                )
                all_radials.append(radial)

    if not all_radials:
        return np.zeros(spectral_vector_length(num_bands, suppress_dc=suppress_dc)), []

    stacked = np.stack(all_radials)
    return stacked.mean(axis=0), all_radials


def compute_attention_power_spectrum_by_layer(
    attention_maps: List[np.ndarray],
    patch_grid: Tuple[int, int],
    layer_indices: List[int],
    num_bands: int = 10,
    window: Optional[str] = "hann",
    suppress_dc: bool = True,
) -> List[Dict[str, Any]]:
    """Compute mean radial spectra and bandwidth for each extracted layer."""
    per_layer: List[Dict[str, Any]] = []
    for layer_index, attn in zip(layer_indices, attention_maps):
        mean_radial, _ = compute_attention_power_spectrum_multi(
            [attn],
            patch_grid,
            num_bands=num_bands,
            window=window,
            suppress_dc=suppress_dc,
        )
        per_layer.append(
            {
                "layer_index": int(layer_index),
                "radial_power": mean_radial,
                "bandwidth": compute_effective_bandwidth(mean_radial),
            }
        )
    return per_layer


def compute_feature_spectral_signature_stats(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
    num_bands: int = 10,
    clean_patch_grid: Optional[Tuple[int, int]] = None,
    perturbed_patch_grid: Optional[Tuple[int, int]] = None,
    suppress_dc: bool = True,
) -> Dict[str, Any]:
    """Compute raw and relative feature-space spectral signatures."""
    clean_grid, perturbed_grid = _align_feature_grids(
        clean_features,
        perturbed_features,
        clean_patch_grid,
        perturbed_patch_grid,
    )
    clean_fft = np.fft.fft2(clean_grid, axes=(0, 1))
    perturbed_fft = np.fft.fft2(perturbed_grid, axes=(0, 1))
    clean_total_energy = float(np.sum(np.abs(clean_fft) ** 2))
    delta_fft = perturbed_fft - clean_fft
    delta_power_2d = np.abs(delta_fft) ** 2
    if delta_power_2d.ndim == 3:
        delta_power_for_bins = delta_power_2d.mean(axis=-1)
    else:
        delta_power_for_bins = delta_power_2d

    delta_shifted = np.fft.fftshift(delta_power_for_bins)
    radial_bins = _radially_bin_power(delta_shifted, num_bands, suppress_dc=suppress_dc)
    radial_bins_relative = radial_bins / max(clean_total_energy, _EPS)
    return {
        "delta_f": radial_bins,
        "delta_f_relative": radial_bins_relative,
        "delta_power_2d": delta_power_for_bins,
        "clean_total_energy": clean_total_energy,
    }


def compute_feature_spectral_signature(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
    num_bands: int = 10,
    clean_patch_grid: Optional[Tuple[int, int]] = None,
    perturbed_patch_grid: Optional[Tuple[int, int]] = None,
    suppress_dc: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radially binned FFT-difference energy in feature space."""
    stats = compute_feature_spectral_signature_stats(
        clean_features,
        perturbed_features,
        num_bands=num_bands,
        clean_patch_grid=clean_patch_grid,
        perturbed_patch_grid=perturbed_patch_grid,
        suppress_dc=suppress_dc,
    )
    return stats["delta_f"], stats["delta_power_2d"]


def compute_image_spectral_signature_stats(
    clean: Image.Image,
    perturbed: Image.Image,
    num_bands: int = 10,
    suppress_dc: bool = True,
) -> Dict[str, Any]:
    c_arr = np.array(clean.convert("L"), dtype=np.float32) / 255.0
    p_arr = np.array(perturbed.convert("L"), dtype=np.float32) / 255.0

    if c_arr.shape != p_arr.shape:
        h = min(c_arr.shape[0], p_arr.shape[0])
        w = min(c_arr.shape[1], p_arr.shape[1])
        c_arr = c_arr[:h, :w]
        p_arr = p_arr[:h, :w]

    fft_c = np.fft.fft2(c_arr)
    fft_p = np.fft.fft2(p_arr)
    clean_spectral_energy = float(np.sum(np.abs(fft_c) ** 2))
    delta = fft_p - fft_c
    delta_power_2d = np.abs(delta) ** 2
    delta_shifted = np.fft.fftshift(delta_power_2d)
    radial_bins = _radially_bin_power(delta_shifted, num_bands, suppress_dc=suppress_dc)
    radial_bins_relative = radial_bins / max(clean_spectral_energy, _EPS)
    return {
        "delta_f": radial_bins,
        "delta_f_relative": radial_bins_relative,
        "delta_2d": delta_power_2d,
        "clean_spectral_energy": clean_spectral_energy,
    }


def compute_image_spectral_signature(
    clean: Image.Image,
    perturbed: Image.Image,
    num_bands: int = 10,
    suppress_dc: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    stats = compute_image_spectral_signature_stats(
        clean,
        perturbed,
        num_bands=num_bands,
        suppress_dc=suppress_dc,
    )
    return stats["delta_f"], stats["delta_2d"]


def compute_effective_bandwidth(power_spectrum: np.ndarray) -> float:
    """Compute effective bandwidth G(t) via inverse participation ratio.

    The effective bandwidth measures how many frequency bands carry
    significant energy.  A narrow filter (low-pass only) gives small G(t),
    while a broad filter gives G(t) close to the total number of bands.

    .. math::

        G(t) = \\frac{(\\sum_\\omega |W(\\omega)|^2)^2}{\\sum_\\omega |W(\\omega)|^4}

    Args:
        power_spectrum: 1D radial power spectrum, shape ``(num_bands,)``.

    Returns:
        Effective bandwidth (1.0 = single-band, num_bands = uniform).
    """
    ps = np.asarray(power_spectrum, dtype=np.float64)
    if ps.size == 0:
        return 0.0
    if ps.sum() == 0:
        return 1.0
    # Normalize
    ps_norm = ps / ps.sum()
    sum_sq = np.sum(ps_norm ** 2)
    if sum_sq == 0:
        return 1.0
    # Inverse participation ratio
    return float(1.0 / sum_sq)


def compute_filter_W_t(
    radial_power: np.ndarray,
    norm: str = "l1",
) -> np.ndarray:
    """Normalize radial power spectrum to get the task-specific filter W_t(omega).

    W_t(omega) represents the fraction of attention energy at each frequency
    band. Under the default L1 normalization it integrates to 1 and serves as
    the task's frequency sensitivity profile. The L2 variant is a robustness
    check and has unit Euclidean norm instead.

    Args:
        radial_power: Raw radial power spectrum, shape ``(num_bands,)``.
        norm: Normalization mode, ``"l1"`` or ``"l2"``.

    Returns:
        Normalized filter W_t(omega), shape ``(num_bands,)``.
    """
    rp = np.asarray(radial_power, dtype=np.float64)
    if rp.size == 0:
        return np.zeros(0, dtype=np.float64)
    if norm == "l1":
        total = rp.sum()
    elif norm == "l2":
        total = float(np.sqrt(np.sum(rp ** 2)))
    else:
        raise ValueError(f"Unknown norm {norm!r}")
    if total == 0:
        if norm == "l2":
            return np.ones_like(rp) / np.sqrt(len(rp))
        return np.ones_like(rp) / len(rp)
    return rp / total


def compute_spectral_overlap(
    W_t: np.ndarray,
    delta_f: np.ndarray,
) -> float:
    """Compute predicted sensitivity as matched-filter energy overlap.

    .. math::

        S_{\\text{pred}} = \\sum_\\omega |W_t(\\omega)|^2 \\cdot |\\Delta F(\\omega)|^2

    This is the quadratic matched-filter energy overlap retained as a legacy
    robustness variant. The primary Exp 5 overlap uses
    :func:`compute_overlap_integral`; a probability-weighted power variant is
    available as :func:`compute_spectral_overlap_linear`.

    Args:
        W_t: Task-specific frequency filter, shape ``(num_bands,)``.
        delta_f: Perturbation spectral signature, shape ``(num_bands,)``.

    Returns:
        Predicted sensitivity (scalar).
    """
    w = np.asarray(W_t, dtype=np.float64)
    d = np.asarray(delta_f, dtype=np.float64)
    # Ensure same length
    min_len = min(len(w), len(d))
    return float(np.sum(w[:min_len] ** 2 * d[:min_len] ** 2))


def compute_spectral_overlap_linear(
    W_t: np.ndarray,
    delta_f: np.ndarray,
) -> float:
    """Linear overlap: S_pred_lin = Σ W_t(ω) · |ΔF(ω)|²."""
    w = np.asarray(W_t, dtype=np.float64)
    d = np.asarray(delta_f, dtype=np.float64)
    min_len = min(len(w), len(d))
    return float(np.sum(w[:min_len] * d[:min_len] ** 2))


def compute_overlap_integral(
    W_t: np.ndarray,
    delta_f: np.ndarray,
) -> float:
    """First-order overlap integral matching the theoretical drift form.

    .. math::

        \\langle W_t, \\Delta F \\rangle = \\sum_\\omega W_t(\\omega) \\cdot \\Delta F(\\omega)

    Both factors appear linearly. ``delta_f`` is the radially binned
    perturbation signature as stored by the pipeline (already non-negative).
    This is the quantity predicted to drive drift under the
    peak-consolidation / relocation story.
    """
    w = np.asarray(W_t, dtype=np.float64)
    d = np.asarray(delta_f, dtype=np.float64)
    min_len = min(len(w), len(d))
    if min_len == 0:
        return 0.0
    return float(np.sum(w[:min_len] * d[:min_len]))


def top_k_mass_fraction(
    W_t: np.ndarray,
    k: int,
) -> float:
    """Return the sum of the ``k`` largest values in ``W_t``.

    ``W_t`` is expected to be an l1-normalised filter (summing to 1). The
    return value is the cumulative mass captured by the top ``k`` bins and
    is a direct measure of how peaked the filter is.
    """
    w = np.asarray(W_t, dtype=np.float64)
    if w.size == 0 or k <= 0:
        return 0.0
    k_eff = int(min(k, w.size))
    # np.partition puts the k_eff largest values in the last k_eff positions
    # without full sort.
    top = np.partition(w, w.size - k_eff)[-k_eff:]
    return float(np.sum(top))


def tail_mass_fraction(
    W_t: np.ndarray,
    split: float = 0.5,
) -> float:
    """Return the mass of ``W_t`` lying on bands with index ≥ split·(N-1).

    ``split`` is expressed as a fraction of the band index range; 0.5 means
    "the upper half of bands." The index convention matches
    ``spectral_band_centers`` — lower indices are lower frequencies.
    """
    w = np.asarray(W_t, dtype=np.float64)
    if w.size == 0:
        return 0.0
    split_frac = float(np.clip(split, 0.0, 1.0))
    split_idx = int(np.ceil(split_frac * (w.size - 1)))
    split_idx = max(0, min(split_idx, w.size))
    return float(np.sum(w[split_idx:]))


def spectral_centroid(
    W_t: np.ndarray,
) -> float:
    """First moment of the filter along the band-index axis.

    Returns the mass-weighted mean band index. If ``W_t`` sums to zero the
    midpoint of the axis is returned (i.e. no information).
    """
    w = np.asarray(W_t, dtype=np.float64)
    if w.size == 0:
        return 0.0
    total = float(np.sum(w))
    if total <= _EPS:
        return float(w.size - 1) / 2.0
    idx = np.arange(w.size, dtype=np.float64)
    return float(np.sum(idx * w) / total)


def compute_filter_shape_metrics(
    W_t: np.ndarray,
    *,
    top_ks: Tuple[int, ...] = (1, 2, 3, 5),
    tail_split: float = 0.5,
) -> Dict[str, float]:
    """Bundle of shape descriptors for a filter W_t.

    Returns peak concentration (``top_k_mass``), high-frequency tail mass
    (``tail_mass_fraction``), spectral centroid (``centroid_index``), and the
    normalised centroid (``centroid_normalised``, in [0, 1] by dividing by
    ``num_bands-1``). These three together let us discriminate pure
    narrowing, relocation, and peak-consolidation scenarios without having
    to inspect the full W_t vector.
    """
    w = np.asarray(W_t, dtype=np.float64)
    num_bands = int(w.size)
    denom_bands = max(1, num_bands - 1)
    metrics: Dict[str, float] = {
        "num_bands": num_bands,
        "tail_mass_fraction": tail_mass_fraction(w, split=tail_split),
        "tail_split_fraction": float(tail_split),
        "centroid_index": spectral_centroid(w),
    }
    metrics["centroid_normalised"] = metrics["centroid_index"] / float(denom_bands)
    for k in top_ks:
        metrics[f"top_{int(k)}_mass"] = top_k_mass_fraction(w, int(k))
    return metrics
