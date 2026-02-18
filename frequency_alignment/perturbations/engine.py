"""Unified perturbation builder with spectral signature computation.

Wraps existing perturbation functions from the parent VLM driver and adds
spectral signature ``ΔF(ω) = |FFT(perturbed) - FFT(clean)|²`` for each
perturbation.  These signatures are used by Experiment 5 (overlap prediction)
to validate Theorem 1.
"""

from __future__ import annotations

import logging
import math
import random as _random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_PARENT_DIR = str(Path(__file__).resolve().parent.parent.parent)


def _import_parent_perturbations():
    """Lazy import perturbation functions from the main VLM driver."""
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    from vlm_invariance_check import (
        cyclic_horizontal_shift,
        pad_or_crop,
        scale_image,
        scale_and_pad,
        rotate_image,
        text_overlay,
        box_overlay,
    )
    return {
        "cyclic_horizontal_shift": cyclic_horizontal_shift,
        "pad_or_crop": pad_or_crop,
        "scale_image": scale_image,
        "scale_and_pad": scale_and_pad,
        "rotate_image": rotate_image,
        "text_overlay": text_overlay,
        "box_overlay": box_overlay,
    }


def _import_frequency_utils():
    """Lazy import FFT utilities from the frequency perturbation module."""
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    from frquencypertubation import (
        make_frequency_mask,
        fft_filter_keep,
    )
    return make_frequency_mask, fft_filter_keep


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PerturbationResult:
    """A perturbation applied to an image, with its spectral signature."""

    name: str                          # e.g. "Translation(+4)"
    family: str                        # "natural", "lowfreq", "highfreq"
    severity: int                      # 1, 2, or 3
    perturbed_image: Image.Image
    delta_f: np.ndarray                # |ΔF(ω)|², radially binned, shape (num_bands,)
    delta_f_2d: Optional[np.ndarray] = None  # full 2D spectral diff (optional)


# ---------------------------------------------------------------------------
# Spectral signature computation
# ---------------------------------------------------------------------------


def compute_spectral_signature(
    clean: Image.Image,
    perturbed: Image.Image,
    num_bands: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radially binned spectral difference |ΔF(ω)|².

    Args:
        clean: Original PIL image.
        perturbed: Perturbed PIL image.
        num_bands: Number of radial frequency bands.

    Returns:
        ``(radial_bins, delta_2d)`` where ``radial_bins`` has shape
        ``(num_bands,)`` and ``delta_2d`` has shape ``(H, W)``.
    """
    # Convert to grayscale float arrays
    c_arr = np.array(clean.convert("L"), dtype=np.float32) / 255.0
    p_arr = np.array(perturbed.convert("L"), dtype=np.float32) / 255.0

    # Resize to match if needed
    if c_arr.shape != p_arr.shape:
        h = min(c_arr.shape[0], p_arr.shape[0])
        w = min(c_arr.shape[1], p_arr.shape[1])
        c_arr = c_arr[:h, :w]
        p_arr = p_arr[:h, :w]

    # 2D FFT
    fft_c = np.fft.fft2(c_arr)
    fft_p = np.fft.fft2(p_arr)
    delta = fft_p - fft_c

    # 2D power difference
    delta_power_2d = np.abs(delta) ** 2

    # Radial binning
    h, w = delta_power_2d.shape
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    # Distance from center (shift FFT center)
    dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)

    delta_shifted = np.fft.fftshift(delta_power_2d)

    edges = np.linspace(0, max_dist, num_bands + 1)
    radial_bins = np.zeros(num_bands, dtype=np.float64)
    for b in range(num_bands):
        mask = (dist >= edges[b]) & (dist < edges[b + 1])
        count = mask.sum()
        if count > 0:
            radial_bins[b] = delta_shifted[mask].sum() / count

    return radial_bins, delta_power_2d


# ---------------------------------------------------------------------------
# Individual perturbation applicators
# ---------------------------------------------------------------------------


def _apply_frequency_perturbation(
    image: Image.Image,
    mode: str,
    cutoff: float,
    epsilon: float,
    rng: _random.Random,
) -> Image.Image:
    """Apply a frequency-domain perturbation.

    Modes: lowpass_keep, highpass_keep, lowband_noise, highband_noise, allband_noise.
    """
    import torch
    arr = np.array(image, dtype=np.float32) / 255.0

    if mode in ("lowpass_keep", "highpass_keep"):
        # Frequency ablation: keep only low or high frequencies
        make_mask, fft_keep = _import_frequency_utils()
        # Convert to tensor for FFT operations
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        keep_low = mode == "lowpass_keep"
        mask = make_mask(t.shape[-2:], cutoff, keep_low=keep_low)
        # Apply mask in frequency domain per channel
        result = torch.zeros_like(t)
        for c in range(t.shape[1]):
            fft = torch.fft.fft2(t[0, c])
            fft_shifted = torch.fft.fftshift(fft)
            fft_filtered = fft_shifted * mask
            fft_unshifted = torch.fft.ifftshift(fft_filtered)
            result[0, c] = torch.fft.ifft2(fft_unshifted).real
        result = result.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
        return Image.fromarray((result * 255).astype(np.uint8))

    elif mode in ("lowband_noise", "highband_noise", "allband_noise"):
        # Add band-limited noise
        h, w, c = arr.shape
        noise = rng.gauss(0, 1) * np.ones_like(arr)
        noise = np.random.RandomState(rng.randint(0, 2**31)).randn(h, w, c).astype(np.float32)

        if mode != "allband_noise":
            # Band-limit the noise
            import torch
            make_mask, _ = _import_frequency_utils()
            keep_low = mode == "lowband_noise"
            mask_2d = make_mask((h, w), cutoff, keep_low=keep_low).numpy()
            for ch in range(c):
                fft_n = np.fft.fft2(noise[:, :, ch])
                fft_n_shifted = np.fft.fftshift(fft_n)
                fft_n_shifted *= mask_2d
                noise[:, :, ch] = np.real(np.fft.ifft2(np.fft.ifftshift(fft_n_shifted)))

        result = arr + epsilon * noise
        result = np.clip(result, 0, 1)
        return Image.fromarray((result * 255).astype(np.uint8))

    raise ValueError(f"Unknown frequency mode: {mode}")


# ---------------------------------------------------------------------------
# Build full perturbation suite
# ---------------------------------------------------------------------------

# Default severity parameters matching the segmentation harness
_DEFAULT_SEVERITY = {
    1: {"translate": 4, "padcrop": 4, "scale": 0.95, "rotation": 10,
        "freq_cutoff": 0.18, "freq_epsilon": 8 / 255, "text_scale": 0.3},
    2: {"translate": 8, "padcrop": 8, "scale": 0.90, "rotation": 20,
        "freq_cutoff": 0.28, "freq_epsilon": 8 / 255, "text_scale": 0.5},
    3: {"translate": 12, "padcrop": 12, "scale": 0.85, "rotation": 30,
        "freq_cutoff": 0.38, "freq_epsilon": 8 / 255, "text_scale": 0.7},
}


def build_perturbation_suite(
    image: Image.Image,
    severity_levels: List[int] | None = None,
    include_natural: bool = True,
    include_frequency: bool = True,
    num_bands: int = 10,
    severity_params: Dict[int, Dict[str, Any]] | None = None,
    seed: int = 42,
) -> List[PerturbationResult]:
    """Build all perturbations for one image with spectral signatures.

    Args:
        image: Clean input image.
        severity_levels: List of severity levels (default [1, 2, 3]).
        include_natural: Include natural perturbations.
        include_frequency: Include frequency perturbations.
        num_bands: Number of radial bands for spectral signature.
        severity_params: Override severity parameters.
        seed: Random seed.

    Returns:
        List of :class:`PerturbationResult`.
    """
    if severity_levels is None:
        severity_levels = [1, 2, 3]
    params = severity_params or _DEFAULT_SEVERITY
    rng = _random.Random(seed)
    fns = _import_parent_perturbations()

    results: List[PerturbationResult] = []

    for sev in severity_levels:
        sp = params.get(sev, params.get(1, {}))
        tx = sp.get("translate", 4)
        pc = sp.get("padcrop", 4)
        sc = sp.get("scale", 0.95)
        rot = sp.get("rotation", 10)
        fc = sp.get("freq_cutoff", 0.18)
        fe = sp.get("freq_epsilon", 8 / 255)

        # -- Natural perturbations --
        if include_natural:
            natural_specs = [
                (f"Translation(+{tx})", "natural",
                 lambda img, n=tx: fns["cyclic_horizontal_shift"](img, n)),
                (f"Translation(-{tx})", "natural",
                 lambda img, n=tx: fns["cyclic_horizontal_shift"](img, -n)),
                (f"PadCrop(+{pc})", "natural",
                 lambda img, n=pc: fns["pad_or_crop"](img, n)),
                (f"PadCrop(-{pc})", "natural",
                 lambda img, n=pc: fns["pad_or_crop"](img, -n)),
                (f"Scale({sc})", "natural",
                 lambda img, s=sc: fns["scale_image"](img, s)),
                (f"ScalePadBlack({sc})", "natural",
                 lambda img, s=sc: fns["scale_and_pad"](img, s, background="black")),
                (f"ScalePadWhite({sc})", "natural",
                 lambda img, s=sc: fns["scale_and_pad"](img, s, background="white")),
                (f"Rotation(+{rot})", "natural",
                 lambda img, a=rot: fns["rotate_image"](img, a)),
                (f"Rotation(-{rot})", "natural",
                 lambda img, a=rot: fns["rotate_image"](img, -a)),
                (f"TextOverlay", "natural",
                 lambda img: fns["text_overlay"](img, "SAMPLE TEXT")),
                (f"BoxOverlay", "natural",
                 lambda img: fns["box_overlay"](img, [(10, 10, img.width // 3, img.height // 3)])),
                (f"RandomText", "natural",
                 lambda img: fns["text_overlay"](img, str(rng.randint(0, 99999)))),
            ]

            for name, family, fn in natural_specs:
                try:
                    perturbed = fn(image)
                    delta_f, delta_2d = compute_spectral_signature(
                        image, perturbed, num_bands
                    )
                    results.append(PerturbationResult(
                        name=f"{name}|sev{sev}",
                        family=family,
                        severity=sev,
                        perturbed_image=perturbed,
                        delta_f=delta_f,
                        delta_f_2d=delta_2d,
                    ))
                except Exception as e:
                    logger.warning("Perturbation %s failed: %s", name, e)

        # -- Frequency perturbations --
        if include_frequency:
            freq_specs = [
                (f"LowPassKeep({fc:.2f})", "lowfreq", "lowpass_keep"),
                (f"HighPassKeep({fc:.2f})", "highfreq", "highpass_keep"),
                (f"LowBandNoise({fc:.2f})", "lowfreq", "lowband_noise"),
                (f"HighBandNoise({fc:.2f})", "highfreq", "highband_noise"),
                (f"AllBandNoise({fc:.2f})", "highfreq", "allband_noise"),
            ]

            for name, family, mode in freq_specs:
                try:
                    perturbed = _apply_frequency_perturbation(
                        image, mode, fc, fe, rng
                    )
                    delta_f, delta_2d = compute_spectral_signature(
                        image, perturbed, num_bands
                    )
                    results.append(PerturbationResult(
                        name=f"{name}|sev{sev}",
                        family=family,
                        severity=sev,
                        perturbed_image=perturbed,
                        delta_f=delta_f,
                        delta_f_2d=delta_2d,
                    ))
                except Exception as e:
                    logger.warning("Perturbation %s failed: %s", name, e)

    logger.debug(
        "Built %d perturbations across %d severity levels",
        len(results), len(severity_levels),
    )
    return results
