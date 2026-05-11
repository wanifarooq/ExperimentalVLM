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

from .spectral import radially_bin_power

logger = logging.getLogger(__name__)


def _to_spatial_grid(
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


def _align_features(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
    clean_patch_grid: Optional[Tuple[int, int]] = None,
    perturbed_patch_grid: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    clean = _to_spatial_grid(clean_features, clean_patch_grid)
    perturbed = _to_spatial_grid(perturbed_features, perturbed_patch_grid)

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
    return clean_tokens[:usable], perturbed_tokens[:usable]


def _flatten_token_features(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim > 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def compute_band_drift(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
    num_bands: int = 10,
    clean_patch_grid: Optional[Tuple[int, int]] = None,
    perturbed_patch_grid: Optional[Tuple[int, int]] = None,
    suppress_dc: bool = True,
) -> np.ndarray:
    """Compute frequency-band-decomposed drift between clean and perturbed features.

    Reshapes 1D token features to a 2D spatial grid (assuming square layout),
    computes the FFT of the difference, and bins by radial frequency.

    Args:
        clean_features: Shape ``(num_tokens, hidden_dim)`` or ``(H, W, D)``.
        perturbed_features: Same feature layout as ``clean_features``.
        num_bands: Number of radial frequency bands.
        clean_patch_grid: Optional spatial grid for clean features.
        perturbed_patch_grid: Optional spatial grid for perturbed features.

    Returns:
        Band-decomposed drift energy, shape ``(num_bands,)``.
    """
    clean, perturbed = _align_features(
        clean_features,
        perturbed_features,
        clean_patch_grid,
        perturbed_patch_grid,
    )
    diff = perturbed - clean
    if diff.ndim == 2:
        n_tokens = diff.shape[0]
        h = max(1, int(np.sqrt(n_tokens)))
        w = max(1, n_tokens // h)
        usable = min(n_tokens, h * w)
        diff = diff[:usable].reshape(h, w, -1)

    # diff now has shape (H, W, D)
    h, w = diff.shape[0], diff.shape[1]

    # Compute norm across hidden dim, then FFT of the spatial drift map
    drift_magnitude = np.linalg.norm(diff, axis=-1)  # (H, W)

    # 2D FFT
    fft = np.fft.fft2(drift_magnitude)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    return radially_bin_power(power, num_bands, suppress_dc=suppress_dc)


def compute_scalar_drift(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
    clean_patch_grid: Optional[Tuple[int, int]] = None,
    perturbed_patch_grid: Optional[Tuple[int, int]] = None,
) -> float:
    """Compute scalar drift as mean L2 distance between features.

    Args:
        clean_features: Shape ``(num_tokens, hidden_dim)``.
        perturbed_features: Same shape.

    Returns:
        Mean per-token L2 drift.
    """
    clean, perturbed = _align_features(
        clean_features,
        perturbed_features,
        clean_patch_grid,
        perturbed_patch_grid,
    )
    diff = perturbed - clean
    if diff.ndim > 2:
        diff = diff.reshape(-1, diff.shape[-1])
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
