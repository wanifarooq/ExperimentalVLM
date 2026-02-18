"""Pre/post fusion drift analysis.

Measures how perturbation-induced drift in vision features gets amplified
or attenuated as features pass through cross-modal fusion layers.

Key concept: the amplification ratio R(omega) = ||DeltaZ(omega)|| / ||DeltaV(omega)|| tells us
which frequency bands the language conditioning amplifies.  If R(omega) correlates
with W_t(omega), it confirms that language conditioning selectively amplifies
drift in its attended frequency bands.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_band_drift(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
    num_bands: int = 10,
) -> np.ndarray:
    """Compute frequency-band-decomposed drift between clean and perturbed features.

    Reshapes 1D token features to a 2D spatial grid (assuming square layout),
    computes the FFT of the difference, and bins by radial frequency.

    Args:
        clean_features: Shape ``(num_tokens, hidden_dim)`` or ``(H, W, D)``.
        perturbed_features: Same shape as ``clean_features``.
        num_bands: Number of radial frequency bands.

    Returns:
        Band-decomposed drift energy, shape ``(num_bands,)``.
    """
    diff = perturbed_features.astype(np.float64) - clean_features.astype(np.float64)

    if diff.ndim == 2:
        # (tokens, dim) -> try to reshape to square spatial grid
        n_tokens, dim = diff.shape
        side = int(np.sqrt(n_tokens))
        if side * side != n_tokens:
            # Not a perfect square; use closest
            side = int(np.round(np.sqrt(n_tokens)))
            diff = diff[:side * side]
        diff = diff.reshape(side, side, -1)

    # diff now has shape (H, W, D)
    h, w = diff.shape[0], diff.shape[1]

    # Compute norm across hidden dim, then FFT of the spatial drift map
    drift_magnitude = np.linalg.norm(diff, axis=-1)  # (H, W)

    # 2D FFT
    fft = np.fft.fft2(drift_magnitude)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    # Radial binning
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)

    edges = np.linspace(0, max_dist, num_bands + 1)
    radial = np.zeros(num_bands, dtype=np.float64)
    for b in range(num_bands):
        mask = (dist >= edges[b]) & (dist < edges[b + 1])
        count = mask.sum()
        if count > 0:
            radial[b] = power[mask].mean()

    return radial


def compute_amplification_ratio(
    pre_drift_bands: np.ndarray,
    post_drift_bands: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute per-band amplification ratio R(omega) = post_drift / pre_drift.

    Args:
        pre_drift_bands: Pre-fusion drift energy per band, shape ``(num_bands,)``.
        post_drift_bands: Post-fusion drift energy per band, shape ``(num_bands,)``.
        epsilon: Small constant to avoid division by zero.

    Returns:
        Amplification ratio per band, shape ``(num_bands,)``.
        Values > 1 mean the fusion layer amplified drift at that frequency.
    """
    pre = np.asarray(pre_drift_bands, dtype=np.float64)
    post = np.asarray(post_drift_bands, dtype=np.float64)
    return post / (pre + epsilon)


def compute_scalar_drift(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
) -> float:
    """Compute scalar drift as mean L2 distance between features.

    Args:
        clean_features: Shape ``(num_tokens, hidden_dim)``.
        perturbed_features: Same shape.

    Returns:
        Mean per-token L2 drift.
    """
    diff = perturbed_features.astype(np.float64) - clean_features.astype(np.float64)
    per_token_norm = np.linalg.norm(diff, axis=-1)
    return float(per_token_norm.mean())


def compute_cosine_drift_per_token(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
) -> np.ndarray:
    """Compute per-token cosine distance.

    Returns:
        Array of shape ``(num_tokens,)`` with cosine distances in [0, 2].
    """
    c = clean_features.astype(np.float64)
    p = perturbed_features.astype(np.float64)
    c_norm = np.linalg.norm(c, axis=-1, keepdims=True)
    p_norm = np.linalg.norm(p, axis=-1, keepdims=True)
    # Avoid division by zero
    c_norm = np.maximum(c_norm, 1e-10)
    p_norm = np.maximum(p_norm, 1e-10)
    cos_sim = np.sum((c / c_norm) * (p / p_norm), axis=-1)
    return 1.0 - cos_sim
