"""Unified perturbation builder with spectral signature computation.

Wraps existing perturbation functions from the parent VLM driver and adds
spectral signature ``ΔF(ω) = |FFT(perturbed) - FFT(clean)|²`` for each
perturbation.  These signatures are used by Experiment 5 (overlap prediction)
to validate Theorem 1.
"""

from __future__ import annotations

import json
import logging
import math
import random as _random
import re
import string
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from ..analysis.spectral import compute_image_spectral_signature_stats

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


def _import_parent_overlay_utils():
    """Lazy import overlay helpers from the main VLM driver."""
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    from vlm_invariance_check import (
        overlay_font_for_image,
        overlay_box_for_text,
        text_overlay_in_box,
        text_overlay_phrases,
        random_text_phrases,
        box_overlay,
    )
    return {
        "overlay_font_for_image": overlay_font_for_image,
        "overlay_box_for_text": overlay_box_for_text,
        "text_overlay_in_box": text_overlay_in_box,
        "text_overlay_phrases": text_overlay_phrases,
        "random_text_phrases": random_text_phrases,
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
    delta_f_relative: Optional[np.ndarray] = None
    delta_f_2d: Optional[np.ndarray] = None  # full 2D spectral diff (optional)
    clean_spectral_energy: Optional[float] = None
    mask_transform: Optional[Callable[[Image.Image], Image.Image]] = None


def _safe_filename(value: str, max_chars: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        cleaned = "item"
    return cleaned[:max_chars]


def export_perturbation_suite_images(
    clean_image: Image.Image,
    perturbations: List[PerturbationResult],
    out_dir: Path,
    image_id: str,
) -> Path:
    """Save clean + perturbed images for one sample to disk."""
    sample_dir = Path(out_dir) / _safe_filename(image_id)
    sample_dir.mkdir(parents=True, exist_ok=True)

    clean_filename = "clean.png"
    clean_image.save(sample_dir / clean_filename)

    manifest: Dict[str, Any] = {
        "image_id": str(image_id),
        "clean_image": clean_filename,
        "num_perturbations": len(perturbations),
        "perturbations": [],
    }

    for idx, perturbation in enumerate(perturbations):
        filename = f"{idx:03d}_{_safe_filename(perturbation.name)}.png"
        perturbation.perturbed_image.save(sample_dir / filename)
        manifest["perturbations"].append(
            {
                "filename": filename,
                "name": perturbation.name,
                "family": perturbation.family,
                "severity": perturbation.severity,
            }
        )

    with open(sample_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return sample_dir


def compute_spectral_signature_stats(
    clean: Image.Image,
    perturbed: Image.Image,
    num_bands: int = 10,
    suppress_dc: bool = True,
) -> Dict[str, Any]:
    """Compute raw and relative image-space spectral signatures."""
    return compute_image_spectral_signature_stats(
        clean,
        perturbed,
        num_bands=num_bands,
        suppress_dc=suppress_dc,
    )


def compute_spectral_signature(
    clean: Image.Image,
    perturbed: Image.Image,
    num_bands: int = 10,
    suppress_dc: bool = True,
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
    stats = compute_spectral_signature_stats(
        clean,
        perturbed,
        num_bands=num_bands,
        suppress_dc=suppress_dc,
    )
    return stats["delta_f"], stats["delta_2d"]


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
        keep_mode = "low" if mode == "lowpass_keep" else "high"
        result = fft_keep(t, keep_mode, cutoff)
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
            mask_2d = make_mask(
                h,
                w,
                "low" if keep_low else "high",
                cutoff,
                torch.device("cpu"),
                torch.float32,
            ).numpy()
            for ch in range(c):
                fft_n = np.fft.fft2(noise[:, :, ch])
                fft_n_shifted = np.fft.fftshift(fft_n)
                fft_n_shifted *= mask_2d
                noise[:, :, ch] = np.real(np.fft.ifft2(np.fft.ifftshift(fft_n_shifted)))

        result = arr + epsilon * noise
        result = np.clip(result, 0, 1)
        return Image.fromarray((result * 255).astype(np.uint8))

    raise ValueError(f"Unknown frequency mode: {mode}")


def _identity_image(image: Image.Image) -> Image.Image:
    return image.copy()


def _apply_motion_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    """Apply a horizontal motion-blur kernel of arbitrary odd size.

    PIL's ``ImageFilter.Kernel`` rejects kernels larger than 5x5, so at
    severity 3 (kernel=7) we implement the uniform horizontal blur directly
    via a numpy cumsum moving average.
    """
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1

    arr = np.asarray(image)
    squeezed = False
    if arr.ndim == 2:
        arr = arr[..., None]
        squeezed = True
    pad = k // 2
    padded = np.pad(arr.astype(np.float32), ((0, 0), (pad, pad), (0, 0)), mode="edge")
    cs = np.cumsum(padded, axis=1)
    cs = np.concatenate(
        [np.zeros((cs.shape[0], 1, cs.shape[2]), dtype=cs.dtype), cs], axis=1
    )
    blurred = (cs[:, k:, :] - cs[:, :-k, :]) / float(k)
    blurred = np.clip(blurred, 0.0, 255.0).astype(np.uint8)
    if squeezed:
        blurred = blurred[..., 0]
    return Image.fromarray(blurred, mode=image.mode)


def _apply_occlusion(
    image: Image.Image,
    fraction: float,
    *,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Apply a centered cutout occlusion with area controlled by ``fraction``."""
    frac = max(0.0, min(0.8, float(fraction)))
    out = image.copy()
    w, h = out.size
    side_scale = math.sqrt(frac)
    box_w = max(1, int(w * side_scale))
    box_h = max(1, int(h * side_scale))
    x0 = max(0, (w - box_w) // 2)
    y0 = max(0, (h - box_h) // 2)
    x1 = min(w, x0 + box_w)
    y1 = min(h, y0 + box_h)
    draw = ImageDraw.Draw(out)
    if out.mode == "RGBA":
        fill_value: Any = (*fill, 255)
    elif out.mode == "RGB":
        fill_value = fill
    else:
        fill_value = 0
    draw.rectangle((x0, y0, x1, y1), fill=fill_value)
    return out


def _apply_gaussian_noise(
    image: Image.Image,
    std: float,
    rng: _random.Random,
) -> Image.Image:
    """Apply additive Gaussian pixel noise with std measured on [0, 1]."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    noisy = arr + np_rng.normal(0.0, float(std), size=arr.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8))


def _apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip through JPEG to simulate compression artifacts."""
    q = max(1, min(95, int(quality)))
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=q, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _centered_box_from_size(
    image: Image.Image,
    width: int,
    height: int,
    pad: int = 4,
) -> Tuple[int, int, int, int]:
    w, h = image.size
    x = max(0, (w - width) // 2)
    y = max(0, (h - height) // 2)
    return (x - pad, y - pad, x + width + pad, y + height + pad)


def _label_free_overlay_phrases(count: int) -> List[str]:
    base = [
        "NOTICE",
        "GENERAL NOTE",
        "TEXT INSERT",
    ]
    if count <= len(base):
        return base[:count]
    phrases = list(base)
    for idx in range(len(base), count):
        phrases.append(f"GENERIC TEXT {idx + 1}")
    return phrases


def _random_text_phrases_like(
    seed: int,
    reference_phrases: List[str],
) -> List[str]:
    rng = _random.Random(seed)
    alphabet = string.ascii_uppercase + string.digits
    phrases: List[str] = []
    for ref in reference_phrases:
        target_len = max(6, len(ref.replace(" ", "")))
        phrases.append("".join(rng.choice(alphabet) for _ in range(target_len)))
    return phrases


def _paired_overlay_boxes(
    image: Image.Image,
    font,
    phrases: List[str],
    random_phrases: List[str],
) -> List[Tuple[int, int, int, int]]:
    draw = ImageDraw.Draw(image)
    boxes: List[Tuple[int, int, int, int]] = []
    for phrase, random_phrase in zip(phrases, random_phrases):
        text_bbox = draw.textbbox((0, 0), phrase, font=font)
        rand_bbox = draw.textbbox((0, 0), random_phrase, font=font)
        width = max(text_bbox[2] - text_bbox[0], rand_bbox[2] - rand_bbox[0])
        height = max(text_bbox[3] - text_bbox[1], rand_bbox[3] - rand_bbox[1])
        boxes.append(_centered_box_from_size(image, width, height))
    return boxes


def _build_overlay_specs(
    image: Image.Image,
    *,
    enabled_natural: set[str],
    overlay_mode: str,
    overlay_count: int,
    overlay_options: Optional[Dict[str, str]],
    overlay_base_label: Optional[str],
    overlay_seed: int,
    fallback_text: str = "SAMPLE TEXT",
) -> List[Tuple[str, str, str, Callable[[Image.Image], Image.Image], Callable[[Image.Image], Image.Image]]]:
    """Build task-agnostic or answer-conditioned overlay perturbations once per image.

    Overlay perturbations are created once per image and reused across levels, so
    the default mode is task-agnostic ``label_free``. The legacy
    ``answer_conditioned`` mode remains available for controlled ablations.
    """
    overlay_utils = _import_parent_overlay_utils()
    mode = (overlay_mode or "label_free").strip().lower()
    count = max(1, int(overlay_count))
    if mode == "label_free":
        phrases = _label_free_overlay_phrases(count)
        random_phrases = _random_text_phrases_like(overlay_seed, phrases)
    elif mode == "answer_conditioned":
        options = overlay_options or {}
        phrases = overlay_utils["text_overlay_phrases"](
            options,
            overlay_base_label,
            limit=count,
        )
        if not phrases:
            phrases = [fallback_text]
        random_phrases = _random_text_phrases_like(overlay_seed, phrases)
    else:
        raise ValueError(f"Unknown overlay_mode: {overlay_mode}")

    font = overlay_utils["overlay_font_for_image"](image)
    boxes = _paired_overlay_boxes(image, font, phrases, random_phrases)

    specs: List[
        Tuple[str, str, str, Callable[[Image.Image], Image.Image], Callable[[Image.Image], Image.Image]]
    ] = []

    if not enabled_natural or "text_overlay" in enabled_natural:
        for idx, phrase in enumerate(phrases):
            box = boxes[idx]
            specs.append(
                (
                    "text_overlay",
                    f"TextOverlay({idx + 1})",
                    "natural",
                    lambda img, phrase=phrase, box=box, font=font: overlay_utils["text_overlay_in_box"](
                        img, phrase, box, font
                    ),
                    _identity_image,
                )
            )

    if not enabled_natural or "box_overlay" in enabled_natural:
        for idx, box in enumerate(boxes):
            specs.append(
                (
                    "box_overlay",
                    f"BoxOverlay({idx + 1})",
                    "natural",
                    lambda img, box=box: overlay_utils["box_overlay"](img, [box]),
                    _identity_image,
                )
            )

    if not enabled_natural or "random_text" in enabled_natural:
        for idx, phrase in enumerate(random_phrases[: len(boxes)]):
            box = boxes[idx]
            specs.append(
                (
                    "random_text",
                    f"RandomText({idx + 1})",
                    "natural",
                    lambda img, phrase=phrase, box=box, font=font: overlay_utils["text_overlay_in_box"](
                        img, phrase, box, font
                    ),
                    _identity_image,
                )
            )

    return specs


# ---------------------------------------------------------------------------
# Build full perturbation suite
# ---------------------------------------------------------------------------

# Default severity parameters matching the segmentation harness
_DEFAULT_SEVERITY = {
    1: {"translate": 4, "padcrop": 4, "scale": 0.95, "rotation": 10,
        "freq_cutoff": 0.18, "freq_epsilon": 8 / 255, "text_scale": 0.3,
        "gaussian_blur_radius": 1.0, "motion_blur_kernel": 3,
        "occlusion_fraction": 0.08, "brightness_delta": 0.15,
        "contrast_delta": 0.15, "gaussian_noise_std": 0.02,
        "jpeg_quality": 70},
    2: {"translate": 8, "padcrop": 8, "scale": 0.90, "rotation": 20,
        "freq_cutoff": 0.28, "freq_epsilon": 8 / 255, "text_scale": 0.5,
        "gaussian_blur_radius": 2.0, "motion_blur_kernel": 5,
        "occlusion_fraction": 0.14, "brightness_delta": 0.30,
        "contrast_delta": 0.30, "gaussian_noise_std": 0.04,
        "jpeg_quality": 50},
    3: {"translate": 12, "padcrop": 12, "scale": 0.85, "rotation": 30,
        "freq_cutoff": 0.38, "freq_epsilon": 8 / 255, "text_scale": 0.7,
        "gaussian_blur_radius": 3.0, "motion_blur_kernel": 7,
        "occlusion_fraction": 0.20, "brightness_delta": 0.45,
        "contrast_delta": 0.45, "gaussian_noise_std": 0.06,
        "jpeg_quality": 30},
}


def build_perturbation_suite(
    image: Image.Image,
    severity_levels: List[int] | None = None,
    include_natural: bool = True,
    include_frequency: bool = True,
    num_bands: int = 10,
    suppress_dc: bool = True,
    natural_types: List[str] | None = None,
    frequency_types: List[str] | None = None,
    severity_params: Dict[int, Dict[str, Any]] | None = None,
    seed: int = 42,
    overlay_mode: str = "label_free",
    overlay_count: int = 3,
    overlay_options: Optional[Dict[str, str]] = None,
    overlay_base_label: Optional[str] = None,
    overlay_seed: Optional[int] = None,
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
        overlay_mode: ``label_free`` for task-agnostic overlays, or
            ``answer_conditioned`` for legacy MCQ-conditioned overlays.
        overlay_count: Number of static overlay variants to generate per family.
        overlay_options: Optional MCQ options used only in ``answer_conditioned`` mode.
        overlay_base_label: Reference answer label excluded only in
            ``answer_conditioned`` mode.
        overlay_seed: Seed for random-text overlays.

    Returns:
        List of :class:`PerturbationResult`.
    """
    if severity_levels is None:
        severity_levels = [1, 2, 3]
    params = severity_params or _DEFAULT_SEVERITY
    rng = _random.Random(seed)
    fns = _import_parent_perturbations()
    enabled_natural = {name.lower() for name in (natural_types or [])}
    enabled_frequency = {name.lower() for name in (frequency_types or [])}

    results: List[PerturbationResult] = []
    static_overlay_specs: List[
        Tuple[str, str, str, Callable[[Image.Image], Image.Image], Callable[[Image.Image], Image.Image]]
    ] = []
    if include_natural:
        static_overlay_specs = _build_overlay_specs(
            image,
            enabled_natural=enabled_natural,
            overlay_mode=overlay_mode,
            overlay_count=overlay_count,
            overlay_options=overlay_options,
            overlay_base_label=overlay_base_label,
            overlay_seed=overlay_seed if overlay_seed is not None else seed,
        )

    for sev in severity_levels:
        sp = params.get(sev, params.get(1, {}))
        tx = sp.get("translate", 4)
        pc = sp.get("padcrop", 4)
        sc = sp.get("scale", 0.95)
        rot = sp.get("rotation", 10)
        fc = sp.get("freq_cutoff", 0.18)
        fe = sp.get("freq_epsilon", 8 / 255)
        blur_radius = sp.get("gaussian_blur_radius", sp.get("blur_radius", 1.0))
        motion_kernel = sp.get("motion_blur_kernel", 3)
        occlusion_fraction = sp.get("occlusion_fraction", 0.08)
        brightness_delta = sp.get("brightness_delta", 0.15)
        contrast_delta = sp.get("contrast_delta", 0.15)
        noise_std = sp.get("gaussian_noise_std", 0.02)
        jpeg_quality = sp.get("jpeg_quality", 70)

        # -- Natural perturbations --
        if include_natural:
            natural_specs = [
                (
                    "translation",
                    f"Translation(+{tx})",
                    "natural",
                    lambda img, n=tx: fns["cyclic_horizontal_shift"](img, n),
                    lambda img, n=tx: fns["cyclic_horizontal_shift"](img, n),
                ),
                (
                    "translation",
                    f"Translation(-{tx})",
                    "natural",
                    lambda img, n=tx: fns["cyclic_horizontal_shift"](img, -n),
                    lambda img, n=tx: fns["cyclic_horizontal_shift"](img, -n),
                ),
                (
                    "padcrop",
                    f"PadCrop(+{pc})",
                    "natural",
                    lambda img, n=pc: fns["pad_or_crop"](img, n),
                    lambda img, n=pc: fns["pad_or_crop"](img, n),
                ),
                (
                    "padcrop",
                    f"PadCrop(-{pc})",
                    "natural",
                    lambda img, n=pc: fns["pad_or_crop"](img, -n),
                    lambda img, n=pc: fns["pad_or_crop"](img, -n),
                ),
                (
                    "scale",
                    f"Scale({sc})",
                    "natural",
                    lambda img, s=sc: fns["scale_image"](img, s),
                    lambda img, s=sc: fns["scale_image"](img, s),
                ),
                (
                    "scalepad_black",
                    f"ScalePadBlack({sc})",
                    "natural",
                    lambda img, s=sc: fns["scale_and_pad"](img, s, background="black"),
                    lambda img, s=sc: fns["scale_and_pad"](img, s, background="black"),
                ),
                (
                    "scalepad_white",
                    f"ScalePadWhite({sc})",
                    "natural",
                    lambda img, s=sc: fns["scale_and_pad"](img, s, background="white"),
                    lambda img, s=sc: fns["scale_and_pad"](img, s, background="white"),
                ),
                (
                    "rotation",
                    f"Rotation(+{rot})",
                    "natural",
                    lambda img, a=rot: fns["rotate_image"](img, a),
                    lambda img, a=rot: fns["rotate_image"](img, a),
                ),
                (
                    "rotation",
                    f"Rotation(-{rot})",
                    "natural",
                    lambda img, a=rot: fns["rotate_image"](img, -a),
                    lambda img, a=rot: fns["rotate_image"](img, -a),
                ),
                (
                    "gaussian_blur",
                    f"GaussianBlur({blur_radius})",
                    "natural",
                    lambda img, r=blur_radius: img.filter(ImageFilter.GaussianBlur(radius=float(r))),
                    _identity_image,
                ),
                (
                    "motion_blur",
                    f"MotionBlur({motion_kernel})",
                    "natural",
                    lambda img, k=motion_kernel: _apply_motion_blur(img, int(k)),
                    _identity_image,
                ),
                (
                    "occlusion",
                    f"Occlusion({occlusion_fraction:.2f})",
                    "natural",
                    lambda img, frac=occlusion_fraction: _apply_occlusion(img, float(frac)),
                    _identity_image,
                ),
                (
                    "brightness",
                    f"Brightness(+{brightness_delta:.2f})",
                    "natural",
                    lambda img, d=brightness_delta: ImageEnhance.Brightness(img).enhance(1.0 + float(d)),
                    _identity_image,
                ),
                (
                    "brightness",
                    f"Brightness(-{brightness_delta:.2f})",
                    "natural",
                    lambda img, d=brightness_delta: ImageEnhance.Brightness(img).enhance(max(0.05, 1.0 - float(d))),
                    _identity_image,
                ),
                (
                    "contrast",
                    f"Contrast(+{contrast_delta:.2f})",
                    "natural",
                    lambda img, d=contrast_delta: ImageEnhance.Contrast(img).enhance(1.0 + float(d)),
                    _identity_image,
                ),
                (
                    "contrast",
                    f"Contrast(-{contrast_delta:.2f})",
                    "natural",
                    lambda img, d=contrast_delta: ImageEnhance.Contrast(img).enhance(max(0.05, 1.0 - float(d))),
                    _identity_image,
                ),
                (
                    "gaussian_noise",
                    f"GaussianNoise({noise_std:.3f})",
                    "natural",
                    lambda img, s=noise_std, rng=rng: _apply_gaussian_noise(img, float(s), rng),
                    _identity_image,
                ),
                (
                    "jpeg_compression",
                    f"JPEG({jpeg_quality})",
                    "natural",
                    lambda img, q=jpeg_quality: _apply_jpeg_compression(img, int(q)),
                    _identity_image,
                ),
            ]

            for pert_type, name, family, fn, mask_fn in natural_specs:
                if enabled_natural and pert_type not in enabled_natural:
                    continue
                try:
                    perturbed = fn(image)
                    spectral_stats = compute_spectral_signature_stats(
                        image,
                        perturbed,
                        num_bands,
                        suppress_dc=suppress_dc,
                    )
                    results.append(PerturbationResult(
                        name=f"{name}|sev{sev}",
                        family=family,
                        severity=sev,
                        perturbed_image=perturbed,
                        delta_f=spectral_stats["delta_f"],
                        delta_f_relative=spectral_stats["delta_f_relative"],
                        delta_f_2d=spectral_stats["delta_2d"],
                        clean_spectral_energy=spectral_stats["clean_spectral_energy"],
                        mask_transform=mask_fn,
                    ))
                except Exception as e:
                    logger.warning("Perturbation %s failed: %s", name, e)

        # -- Frequency perturbations --
        if include_frequency:
            freq_specs = [
                ("lowpass_keep", f"LowPassKeep({fc:.2f})", "lowfreq", "lowpass_keep"),
                ("highpass_keep", f"HighPassKeep({fc:.2f})", "highfreq", "highpass_keep"),
                ("lowband_noise", f"LowBandNoise({fc:.2f})", "lowfreq", "lowband_noise"),
                ("highband_noise", f"HighBandNoise({fc:.2f})", "highfreq", "highband_noise"),
                ("allband_noise", f"AllBandNoise({fc:.2f})", "highfreq", "allband_noise"),
            ]

            for pert_type, name, family, mode in freq_specs:
                if enabled_frequency and pert_type not in enabled_frequency:
                    continue
                try:
                    perturbed = _apply_frequency_perturbation(
                        image, mode, fc, fe, rng
                    )
                    spectral_stats = compute_spectral_signature_stats(
                        image,
                        perturbed,
                        num_bands,
                        suppress_dc=suppress_dc,
                    )
                    results.append(PerturbationResult(
                        name=f"{name}|sev{sev}",
                        family=family,
                        severity=sev,
                        perturbed_image=perturbed,
                        delta_f=spectral_stats["delta_f"],
                        delta_f_relative=spectral_stats["delta_f_relative"],
                        delta_f_2d=spectral_stats["delta_2d"],
                        clean_spectral_energy=spectral_stats["clean_spectral_energy"],
                        mask_transform=_identity_image,
                    ))
                except Exception as e:
                    logger.warning("Perturbation %s failed: %s", name, e)

    if include_natural and static_overlay_specs:
        overlay_severity = severity_levels[0] if severity_levels else 1
        for pert_type, name, family, fn, mask_fn in static_overlay_specs:
            if enabled_natural and pert_type not in enabled_natural:
                continue
            try:
                perturbed = fn(image)
                spectral_stats = compute_spectral_signature_stats(
                    image,
                    perturbed,
                    num_bands,
                    suppress_dc=suppress_dc,
                )
                results.append(PerturbationResult(
                    name=name,
                    family=family,
                    severity=overlay_severity,
                    perturbed_image=perturbed,
                    delta_f=spectral_stats["delta_f"],
                    delta_f_relative=spectral_stats["delta_f_relative"],
                    delta_f_2d=spectral_stats["delta_2d"],
                    clean_spectral_energy=spectral_stats["clean_spectral_energy"],
                    mask_transform=mask_fn,
                ))
            except Exception as e:
                logger.warning("Perturbation %s failed: %s", name, e)

    logger.debug(
        "Built %d perturbations across %d severity levels",
        len(results), len(severity_levels),
    )
    return results
