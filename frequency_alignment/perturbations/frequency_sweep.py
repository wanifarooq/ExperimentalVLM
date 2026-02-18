"""Progressive frequency ablation for Experiment 4.

Sweeps a low-pass or high-pass cutoff from near-DC to Nyquist in
equal steps, producing a series of filtered images.  Used to find the
critical frequency ω_c* where task accuracy drops below a threshold.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def frequency_sweep(
    image: Image.Image,
    num_steps: int = 20,
    mode: str = "lowpass",
) -> List[Tuple[float, Image.Image]]:
    """Sweep frequency cutoff and produce filtered images.

    Args:
        image: Input PIL image (RGB).
        num_steps: Number of cutoff steps between ``min_cutoff`` and 0.5.
        mode: ``"lowpass"`` (keep below cutoff) or ``"highpass"`` (keep above).

    Returns:
        List of ``(cutoff_normalized, filtered_image)`` tuples.
        Cutoffs are in [0, 0.5] (normalized to Nyquist = 0.5).
    """
    arr = np.array(image, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    c = arr.shape[2] if arr.ndim == 3 else 1
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    # Radial distance grid (fftshifted coordinates)
    cy, cx = h // 2, w // 2
    y_grid = np.arange(h) - cy
    x_grid = np.arange(w) - cx
    yy, xx = np.meshgrid(y_grid, x_grid, indexing="ij")
    # Normalize to [0, 0.5]
    max_freq = np.sqrt(cy ** 2 + cx ** 2)
    dist = np.sqrt(yy ** 2 + xx ** 2) / max_freq * 0.5

    cutoffs = np.linspace(0.02, 0.5, num_steps)
    results: List[Tuple[float, Image.Image]] = []

    for cutoff in cutoffs:
        if mode == "lowpass":
            mask = (dist <= cutoff).astype(np.float32)
        elif mode == "highpass":
            mask = (dist >= cutoff).astype(np.float32)
        else:
            raise ValueError(f"mode must be 'lowpass' or 'highpass', got {mode!r}")

        filtered = np.zeros_like(arr)
        for ch in range(c):
            fft = np.fft.fft2(arr[:, :, ch])
            fft_shifted = np.fft.fftshift(fft)
            fft_filtered = fft_shifted * mask
            filtered[:, :, ch] = np.real(
                np.fft.ifft2(np.fft.ifftshift(fft_filtered))
            )

        filtered = np.clip(filtered, 0, 1)
        if c == 1:
            filtered = filtered[:, :, 0]
        img_out = Image.fromarray((filtered * 255).astype(np.uint8))
        results.append((float(cutoff), img_out))

    return results


def compute_critical_cutoff(
    accuracies: List[float],
    cutoffs: List[float],
    threshold: float = 0.5,
) -> float:
    """Find ω_c* where accuracy drops below *threshold*.

    Uses linear interpolation between measured points.

    Args:
        accuracies: Accuracy at each cutoff (same length as cutoffs).
        cutoffs: Normalized cutoff values.
        threshold: Accuracy threshold.

    Returns:
        Critical cutoff value, or ``cutoffs[-1]`` if accuracy never drops
        below threshold.
    """
    for i in range(len(accuracies) - 1):
        a0, a1 = accuracies[i], accuracies[i + 1]
        c0, c1 = cutoffs[i], cutoffs[i + 1]
        if a0 >= threshold > a1:
            # Linear interpolation
            frac = (threshold - a0) / (a1 - a0) if a1 != a0 else 0.5
            return c0 + frac * (c1 - c0)
    # Never dropped below threshold
    return cutoffs[-1]
