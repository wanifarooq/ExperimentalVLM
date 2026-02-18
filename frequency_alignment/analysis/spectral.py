"""Spectral analysis of cross-attention maps.

This module contains the core novel analysis: computing the frequency structure
of VLM cross-attention maps to derive the task-specific frequency filter W_t(omega).

Key functions:
    - ``attention_to_spatial_grid``: reshape flattened attention to 2D spatial map
    - ``compute_attention_power_spectrum``: 2D FFT + radial averaging
    - ``compute_effective_bandwidth``: inverse participation ratio G(t)
    - ``compute_filter_W_t``: averaged normalized attention spectrum per task
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D FFT power spectrum of a spatial attention map.

    Args:
        attention_2d: 2D array of shape ``(H, W)`` representing one
            attention map in spatial layout.
        num_bands: Number of radial frequency bands.

    Returns:
        ``(radial_power, full_power_2d)`` where ``radial_power`` has shape
        ``(num_bands,)`` (radially averaged power) and ``full_power_2d``
        is the full 2D power spectrum ``(H, W)``.
    """
    h, w = attention_2d.shape
    # 2D FFT
    fft = np.fft.fft2(attention_2d.astype(np.float64))
    fft_shifted = np.fft.fftshift(fft)
    power_2d = np.abs(fft_shifted) ** 2

    # Radial binning
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)

    edges = np.linspace(0, max_dist, num_bands + 1)
    radial_power = np.zeros(num_bands, dtype=np.float64)

    for b in range(num_bands):
        mask = (dist >= edges[b]) & (dist < edges[b + 1])
        count = mask.sum()
        if count > 0:
            radial_power[b] = power_2d[mask].mean()

    return radial_power, power_2d


def compute_attention_power_spectrum_multi(
    attention_maps: List[np.ndarray],
    patch_grid: Tuple[int, int],
    num_bands: int = 10,
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
            radial, _ = compute_attention_power_spectrum(grid, num_bands)
            all_radials.append(radial)
        elif attn.ndim == 2:
            # (lang_tokens, vis_tokens) -> average over lang tokens first
            mean_attn = attn.mean(axis=0)
            grid = attention_to_spatial_grid(mean_attn, patch_grid)
            radial, _ = compute_attention_power_spectrum(grid, num_bands)
            all_radials.append(radial)
        elif attn.ndim == 3:
            # (heads, lang_tokens, vis_tokens)
            for head_idx in range(attn.shape[0]):
                mean_attn = attn[head_idx].mean(axis=0)  # avg over lang tokens
                grid = attention_to_spatial_grid(mean_attn, patch_grid)
                radial, _ = compute_attention_power_spectrum(grid, num_bands)
                all_radials.append(radial)

    if not all_radials:
        return np.zeros(num_bands), []

    stacked = np.stack(all_radials)
    return stacked.mean(axis=0), all_radials


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
) -> np.ndarray:
    """Normalize radial power spectrum to get the task-specific filter W_t(omega).

    W_t(omega) represents the fraction of attention energy at each frequency
    band.  It integrates to 1 and serves as the task's frequency sensitivity
    profile.

    Args:
        radial_power: Raw radial power spectrum, shape ``(num_bands,)``.

    Returns:
        Normalized filter W_t(omega), shape ``(num_bands,)``, sums to 1.
    """
    rp = np.asarray(radial_power, dtype=np.float64)
    total = rp.sum()
    if total == 0:
        return np.ones_like(rp) / len(rp)
    return rp / total


def compute_spectral_overlap(
    W_t: np.ndarray,
    delta_f: np.ndarray,
) -> float:
    """Compute predicted sensitivity as spectral overlap integral.

    .. math::

        S_{\\text{pred}} = \\sum_\\omega |W_t(\\omega)|^2 \\cdot |\\Delta F(\\omega)|^2

    This is the core quantity from Theorem 1: perturbation sensitivity is
    determined by how much the perturbation's spectral signature overlaps
    with the task's attended frequency bands.

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
