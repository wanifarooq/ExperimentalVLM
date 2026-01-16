#!/usr/bin/env python3
"""Adversarial frequency-constrained perturbations for VLMs."""

from __future__ import annotations

import argparse
import json
import os
import random
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlm_invariance_check import (
    DEFAULT_SEEDBENCH_ARCHIVE,
    DEFAULT_SEEDBENCH_REPO,
    DEFAULT_SEEDBENCH_TSV,
    build_seedbench_prompt,
    canonical_device_id,
    choose_label_via_loglik,
    ensure_model_available,
    get_context_length_cached,
    list_available_devices,
    load_samples_for_dataset,
    prepare_model,
    score_options_loglik_batch,
)

os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


MODES = ("pixel", "high", "low")


@dataclass
class AttackResult:
    mode: str
    adv_image: torch.Tensor
    adv_label: Optional[str]
    adv_scores: Dict[str, float]
    success: bool
    flip: bool
    fidelity: Optional[float]
    base_score: float
    adv_base_score: float
    loglik_drop: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    try:
        out = float(val)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    img = image.detach().cpu().clamp(0, 1)
    if img.dim() == 4:
        img = img[0]
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def tokens_to_image_batch(
    pixel_values: torch.Tensor,
    image_grid: torch.Tensor,
    processor,
) -> torch.Tensor:
    grid_t, grid_h_full, grid_w_full = [int(x) for x in image_grid[0].tolist()]
    merge_size = int(getattr(processor.image_processor, "merge_size", 1))
    patch_size = int(getattr(processor.image_processor, "patch_size", 16))
    temporal_patch = int(getattr(processor.image_processor, "temporal_patch_size", 1))

    tokens_per = grid_t * grid_h_full * grid_w_full
    if tokens_per == 0 or pixel_values.shape[0] % tokens_per != 0:
        raise ValueError(f"Incompatible pixel_values shape {pixel_values.shape} for grid {image_grid}")
    batch = pixel_values.shape[0] // tokens_per
    dim = pixel_values.shape[1]
    expected_dim = temporal_patch * 3 * patch_size * patch_size
    if dim < expected_dim:
        raise ValueError(f"Unexpected token dim {dim}, expected >= {expected_dim}")

    gh = grid_h_full // merge_size
    gw = grid_w_full // merge_size
    flat = pixel_values.view(batch, grid_t, gh, gw, merge_size, merge_size, 3, temporal_patch, patch_size, patch_size)
    # (b, grid_t, temp, grid_h, merge_h, patch_h, grid_w, merge_w, patch_w, c)
    patches = flat.permute(0, 1, 7, 2, 4, 8, 3, 5, 9, 6)
    frames = grid_t * temporal_patch
    height = grid_h_full * patch_size
    width = grid_w_full * patch_size
    images = patches.reshape(batch, frames, height, width, 3).permute(0, 1, 4, 2, 3)
    return images


def image_batch_to_tokens(
    images: torch.Tensor,
    image_grid: torch.Tensor,
    processor,
) -> torch.Tensor:
    b, frames, c, H, W = images.shape
    grid_t, grid_h_full, grid_w_full = [int(x) for x in image_grid[0].tolist()]
    merge_size = int(getattr(processor.image_processor, "merge_size", 1))
    patch_size = int(getattr(processor.image_processor, "patch_size", 16))
    temporal_patch = int(getattr(processor.image_processor, "temporal_patch_size", 1))
    expected_frames = grid_t * temporal_patch

    if c != 3 or H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Incompatible image shape {images.shape} for grid {image_grid}")
    if frames != expected_frames:
        if frames == 1 and expected_frames > 1:
            images = images.repeat(1, expected_frames, 1, 1, 1)
            frames = expected_frames
        else:
            raise ValueError(f"Frame count {frames} does not match grid {expected_frames}")

    h_calc = H // patch_size
    w_calc = W // patch_size
    if h_calc != grid_h_full or w_calc != grid_w_full:
        raise ValueError(f"Spatial mismatch for grid {image_grid} and image shape {images.shape}")

    patches = images.permute(0, 1, 3, 4, 2)  # b, frames, H, W, 3
    patches = patches.view(
        b,
        grid_t,
        temporal_patch,
        grid_h_full // merge_size,
        merge_size,
        patch_size,
        grid_w_full // merge_size,
        merge_size,
        patch_size,
        3,
    )
    patches = patches.permute(0, 1, 3, 7, 4, 8, 9, 2, 5, 6)
    tokens = patches.reshape(
        b * grid_t * grid_h_full * grid_w_full,
        temporal_patch * 3 * patch_size * patch_size,
    )
    return tokens


def tokens_to_attack_channels(
    pixel_values: torch.Tensor,
    image_grid: torch.Tensor,
    processor,
) -> Tuple[torch.Tensor, int, int, int]:
    images = tokens_to_image_batch(pixel_values, image_grid, processor)
    b, frames, c, H, W = images.shape
    channels = images.reshape(b, frames * c, H, W)
    return channels, frames, H, W


def attack_channels_to_tokens(
    channels: torch.Tensor,
    frames: int,
    image_grid: torch.Tensor,
    processor,
) -> torch.Tensor:
    b, c_total, H, W = channels.shape
    if c_total % 3 != 0:
        raise ValueError(f"Channel count {c_total} not divisible by 3")
    expected_frames = frames if frames > 0 else c_total // 3
    images = channels.view(b, expected_frames, 3, H, W)
    return image_batch_to_tokens(images, image_grid, processor)


def attack_channels_to_pil(
    channels: torch.Tensor,
    frames: int,
) -> Image.Image:
    b, c_total, H, W = channels.shape
    if c_total % 3 != 0:
        raise ValueError("Invalid channel count for PIL conversion")
    num_frames = max(1, frames)
    images = channels.view(b, c_total // 3, 3, H, W)
    # Average across frames to remove striping artifacts from temporal patch splits.
    merged = images[:, :num_frames].mean(dim=1)
    vis = (merged * 0.5) + 0.5
    return tensor_to_pil(vis)


def make_frequency_mask(height: int, width: int, mode: str, cutoff: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype).view(height, 1)
    fx = torch.fft.fftfreq(width, device=device, dtype=dtype).view(1, width)
    radius = torch.sqrt(fx**2 + fy**2)
    radius = radius / radius.max().clamp(min=1e-6)
    if mode == "high":
        mask = (radius >= cutoff).float()
    elif mode == "low":
        mask = (radius <= cutoff).float()
    else:
        mask = torch.ones_like(radius)
    return mask.to(dtype=dtype)


def fft_project(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    delta_f = delta.float()
    mask_f = mask.float()
    while mask_f.dim() < delta_f.dim():
        mask_f = mask_f.unsqueeze(0)
    proj = torch.fft.ifft2(torch.fft.fft2(delta_f, dim=(-2, -1)) * mask_f, dim=(-2, -1)).real
    return proj.to(dtype=delta.dtype)


def build_option_messages(image: Image.Image, prompt: str, options: Dict[str, str]) -> Tuple[List[str], List[list]]:
    labels: List[str] = []
    messages: List[list] = []
    for lab, opt_text in options.items():
        labels.append(lab)
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": opt_text}]},
            ]
        )
    return labels, messages


def compute_loglik_scores(
    log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    context_len: int,
    labels: List[str],
    detach: bool = False,
) -> Dict[str, torch.Tensor | float]:
    scores: Dict[str, torch.Tensor | float] = {}
    batch = input_ids.shape[0]
    for i in range(batch):
        seq_len = int(attn_mask[i].sum().item()) if attn_mask is not None else int(input_ids.shape[1])
        if seq_len <= context_len:
            continue
        target_ids = input_ids[i, context_len:seq_len]
        pred_lp = log_probs[i, context_len - 1 : seq_len - 1, :]
        score = pred_lp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).sum()
        scores[labels[i]] = float(score.item()) if detach else score
    return scores


def frequency_mask_for_mode(mode: str, base_image: torch.Tensor, cutoff: float, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if mode not in {"high", "low"}:
        return None
    cutoff = float(max(0.0, min(1.0, cutoff)))
    _, _, h, w = base_image.shape
    return make_frequency_mask(h, w, mode, cutoff, device, torch.float32)


def attack_mode(
    mode: str,
    base_channels: torch.Tensor,
    token_inputs: Dict[str, torch.Tensor],
    image_grid: torch.Tensor,
    frames: int,
    labels: List[str],
    context_len: int,
    target_label: str,
    base_scores: Dict[str, float],
    model,
    processor,
    device: torch.device,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    cutoff: float,
    compute_fidelity: bool,
    random_start: bool,
) -> AttackResult:
    dtype = next(model.parameters()).dtype
    delta = torch.zeros_like(base_channels, device=device, dtype=dtype)
    mask = frequency_mask_for_mode(mode, base_channels, cutoff, device, dtype)
    if mask is not None:
        mask_nonzero = float((mask != 0).float().mean().item())
        print(f"[debug] mask mode={mode} nonzero_fraction={mask_nonzero:.4f}", flush=True)
    model.zero_grad(set_to_none=True)

    if random_start:
        delta = torch.empty_like(delta).uniform_(-epsilon, epsilon)
        if mask is not None:
            delta = fft_project(delta, mask)
        delta = (base_channels + delta).clamp(-1.0, 1.0) - base_channels

    for _ in range(steps):
        adv_channels = (base_channels + delta).clamp(-1.0, 1.0)
        adv_channels.requires_grad_(True)
        adv_tokens = attack_channels_to_tokens(adv_channels, frames, image_grid, processor)
        inputs = dict(token_inputs)
        inputs["pixel_values"] = adv_tokens

        outputs = model(**inputs)
        log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        scores = compute_loglik_scores(
            log_probs,
            inputs["input_ids"],
            inputs.get("attention_mask"),
            context_len,
            labels,
            detach=False,
        )
        target_score = scores.get(target_label)
        if target_score is None:
            break

        loss = target_score
        loss.backward()
        grad_mean = float(adv_channels.grad.abs().mean().item()) if adv_channels.grad is not None else 0.0
        print(f"[debug] mode={mode} loss={float(loss.item()):.4f} grad_mean={grad_mean:.6f}", flush=True)
        if adv_channels.grad is None:
            break
        step_dir = adv_channels.grad.sign()
        delta = delta - step_size * step_dir.detach()
        if mask is not None:
            delta = fft_project(delta, mask)
        delta = delta.clamp(-epsilon, epsilon)
        delta = (base_channels + delta).clamp(-1.0, 1.0) - base_channels
        model.zero_grad(set_to_none=True)

    with torch.no_grad():
        final_channels = (base_channels + delta).clamp(-1.0, 1.0)
        adv_tokens = attack_channels_to_tokens(final_channels, frames, image_grid, processor)
        eval_inputs = dict(token_inputs)
        eval_inputs["pixel_values"] = adv_tokens
        outputs = model(**eval_inputs)
        log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        adv_scores = compute_loglik_scores(
            log_probs,
            eval_inputs["input_ids"],
            eval_inputs.get("attention_mask"),
            context_len,
            labels,
            detach=True,
        )
        adv_label = max(adv_scores, key=adv_scores.get) if adv_scores else None
        base_score = base_scores.get(target_label, float("nan"))
        adv_base_score = adv_scores.get(target_label, float("nan"))
        loglik_drop = adv_base_score - base_score
        flip = adv_label is not None and adv_label != target_label
        success = flip and adv_base_score < base_score
        fidelity = None
        if compute_fidelity:
            fidelity = image_fidelity(final_channels, base_channels, image_grid, frames, processor, model, device)

    return AttackResult(
        mode=mode,
        adv_image=final_channels.detach(),
        adv_label=adv_label,
        adv_scores={k: float(v) for k, v in adv_scores.items()},
        success=bool(success),
        flip=bool(flip),
        fidelity=fidelity,
        base_score=float(base_score),
        adv_base_score=float(adv_base_score),
        loglik_drop=float(loglik_drop),
    )


@torch.no_grad()
def image_fidelity(
    adv_channels: torch.Tensor,
    base_channels: torch.Tensor,
    image_grid: torch.Tensor,
    frames: int,
    processor,
    model,
    device: torch.device,
) -> Optional[float]:
    adv_pil = attack_channels_to_pil(adv_channels, frames)
    base_pil = attack_channels_to_pil(base_channels, frames)
    try:
        base_inputs = processor.image_processor(images=[base_pil], return_tensors="pt")
        adv_inputs = processor.image_processor(images=[adv_pil], return_tensors="pt")
        dtype = next(model.parameters()).dtype
        base_pixels = base_inputs["pixel_values"].to(device=device, dtype=dtype)
        adv_pixels = adv_inputs["pixel_values"].to(device=device, dtype=dtype)
        grid = base_inputs.get("image_grid_thw")
        base_feat = model.get_image_features(pixel_values=base_pixels, image_grid_thw=grid)
        adv_feat = model.get_image_features(pixel_values=adv_pixels, image_grid_thw=grid)
        base_emb = base_feat[0] if isinstance(base_feat, (list, tuple)) else base_feat
        adv_emb = adv_feat[0] if isinstance(adv_feat, (list, tuple)) else adv_feat
        if isinstance(base_emb, dict) or isinstance(adv_emb, dict):
            return None
        while base_emb.dim() > 2:
            base_emb = base_emb.mean(dim=1)
        while adv_emb.dim() > 2:
            adv_emb = adv_emb.mean(dim=1)
        if base_emb.shape != adv_emb.shape:
            return None
        return float(F.cosine_similarity(base_emb, adv_emb, dim=-1).mean().item())
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    cache_default = Path(__file__).resolve().parent / ".hf_cache"
    p = argparse.ArgumentParser(description="Frequency-constrained adversarial attacks for VLMs.")
    p.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model id to use.")
    p.add_argument("--model-path", type=str, default=None, help="Local path to a model directory (overrides model-id).")
    p.add_argument("--device", default="auto", help="Device: auto | cpu | mps | cuda | cuda:<idx>.")
    p.add_argument("--dataset", type=str, default="seedbench", help="Dataset adapter to use (currently: seedbench).")
    p.add_argument("--seedbench-tsv", type=str, default=None, help="Path to SEEDBench_IMG.tsv. If omitted, defaults to cache dir and auto-download if enabled.")
    p.add_argument("--image-root", type=str, default=None, help="Root directory for SEEDBench images. If omitted, defaults to cache dir and auto-download if enabled.")
    p.add_argument("--seedbench-repo", type=str, default=DEFAULT_SEEDBENCH_REPO, help="Hugging Face dataset repo id for SEEDBench.")
    p.add_argument("--seedbench-tsv-filename", type=str, default=DEFAULT_SEEDBENCH_TSV, help="Filename of the SEEDBench TSV inside the repo or cache.")
    p.add_argument("--seedbench-image-archive", type=str, default=DEFAULT_SEEDBENCH_ARCHIVE, help="Archive filename for SEEDBench images inside the repo (zip/tar).")
    p.add_argument("--data-dir", type=str, default=None, help="Base directory to cache SEEDBench data (defaults to <cache_dir>/seedbench).")
    p.add_argument("--max-samples", type=int, default=1, help="Maximum number of samples to evaluate.")
    p.add_argument("--cache-dir", type=str, default=str(cache_default), help="HF cache directory.")
    p.add_argument("--offline", action="store_true", help="Use HF cache only (no internet).")
    p.add_argument("--output-dir", type=str, default="adversarial_runs", help="Directory to save outputs.")
    p.add_argument("--epsilon", type=float, default=8 / 255.0, help="L-infinity bound in pixel space.")
    p.add_argument("--step-size", type=float, default=2 / 255.0, help="Step size in pixel space.")
    p.add_argument("--steps", type=int, default=10, help="Number of PGD steps.")
    p.add_argument("--freq-cutoff", type=float, default=0.25, help="Radial cutoff (normalized 0-1) for high/low frequency masks.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--compute-fidelity", action="store_true", help="Compute cosine similarity between clean and adversarial embeddings.")
    p.add_argument("--random-start", action="store_true", help="Start PGD from a random point within the epsilon ball.")
    return p.parse_args()


def run_sample(
    sample,
    model,
    processor,
    device: torch.device,
    args: argparse.Namespace,
    context_cache: Dict[tuple, int],
    out_dir: Path,
) -> Optional[dict]:
    try:
        image = Image.open(sample.image_path).convert("RGB")
    except Exception as exc:
        print(f"[warn] Failed to load image {sample.image_path}: {exc}", flush=True)
        return None

    prompt = build_seedbench_prompt(sample)
    context_len = get_context_length_cached(processor, image, prompt, device=str(device), cache=context_cache)
    base_label = choose_label_via_loglik(
        model,
        processor,
        image,
        prompt,
        sample.options,
        device=str(device),
        context_len_cache=context_cache,
    )
    if not base_label:
        print(f"[warn] No base label for sample {sample.index}", flush=True)
        return None
    base_scores = score_options_loglik_batch(
        model,
        processor,
        image,
        prompt,
        sample.options,
        device=str(device),
        context_len=context_len,
    )

    labels, messages = build_option_messages(image, prompt, sample.options)
    template_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    )

    token_inputs = {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in template_inputs.items()
        if k != "pixel_values"
    }
    pixel_values = template_inputs["pixel_values"].to(device)
    image_grid = template_inputs["image_grid_thw"].to(device)
    base_image, frames, height, width = tokens_to_attack_channels(
        pixel_values.to(dtype=next(model.parameters()).dtype),
        image_grid,
        processor,
    )

    sample_dir = out_dir / f"sample_{sample.index}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    attack_channels_to_pil(base_image, frames).save(sample_dir / "clean.png")

    mode_results: Dict[str, AttackResult] = {}
    for mode in MODES:
        res = attack_mode(
            mode,
            base_image,
            token_inputs,
            image_grid,
            frames,
            labels,
            context_len,
            base_label,
            base_scores,
            model,
            processor,
            device,
            epsilon=args.epsilon,
            step_size=args.step_size,
            steps=args.steps,
            cutoff=args.freq_cutoff,
            compute_fidelity=args.compute_fidelity,
            random_start=args.random_start,
        )
        mode_results[mode] = res
        attack_channels_to_pil(res.adv_image, frames).save(sample_dir / f"{mode}.png")
        save_fft_magnitude(
            res.adv_image - base_image,
            sample_dir / f"{mode}_fft.png",
            epsilon=args.epsilon,
            mode=mode,
            freq_cutoff=args.freq_cutoff,
        )

    record = {
        "sample_index": sample.index,
        "base_label": base_label,
        "base_scores": {k: safe_float(v) for k, v in base_scores.items()},
        "modes": {
            m: {
                "adv_label": r.adv_label,
                "adv_scores": {k: safe_float(v) for k, v in r.adv_scores.items()},
                "success": r.success,
                "flip": r.flip,
                "base_score": safe_float(r.base_score),
                "adv_base_score": safe_float(r.adv_base_score),
                "loglik_drop": safe_float(r.loglik_drop),
                "fidelity": safe_float(r.fidelity),
            }
            for m, r in mode_results.items()
        },
    }
    with open(out_dir / "attack_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def save_fft_magnitude(
    delta: torch.Tensor,
    path: Path,
    epsilon: float | None = None,
    *,
    mode: str | None = None,
    freq_cutoff: float | None = None,
) -> None:
    diff = delta.detach().cpu().float()
    raw_diff_full = diff.clone()
    print(f"[debug] delta shape={tuple(diff.shape)} std={float(diff.std().item()):.6f}", flush=True)
    raw_min = float(diff.min().item())
    raw_max = float(diff.max().item())
    raw_mean = float(diff.abs().mean().item())
    clamped_min = raw_min
    clamped_max = raw_max
    clamped_mean = raw_mean
    sat_pos = sat_neg = None
    unclamped = None
    if epsilon is not None:
        sat_pos = int((diff == epsilon).sum().item())
        sat_neg = int((diff == -epsilon).sum().item())
        clamped = diff.clamp(-epsilon, epsilon)
        clamped_min = float(clamped.min().item())
        clamped_max = float(clamped.max().item())
        clamped_mean = float(clamped.abs().mean().item())
        mask_unclamped = (diff != epsilon) & (diff != -epsilon)
        if mask_unclamped.any():
            unclamped_slice = diff[mask_unclamped]
            unclamped = (float(unclamped_slice.min().item()), float(unclamped_slice.max().item()))
    msg = (
        f"[debug] save_fft delta stats raw_min={raw_min:.6f} raw_max={raw_max:.6f} raw_mean_abs={raw_mean:.6f} "
        f"clamped_min={clamped_min:.6f} clamped_max={clamped_max:.6f} clamped_mean_abs={clamped_mean:.6f}"
    )
    if epsilon is not None:
        msg += f" sat_pos={sat_pos} sat_neg={sat_neg}"
        if unclamped is not None:
            msg += f" unclamped_min={unclamped[0]:.6f} unclamped_max={unclamped[1]:.6f}"
    print(msg, flush=True)
    if diff.dim() == 4:
        diff = diff.mean(dim=0)  # average batch if present
    # reshape to [frames, 3, H, W] if possible; use mean absolute to avoid cancellation and view/reshape safely.
    if diff.dim() == 3 and diff.shape[0] % 3 == 0:
        frames = max(1, diff.shape[0] // 3)
        diff = diff.reshape(frames, 3, diff.shape[1], diff.shape[2]).abs().mean(dim=0)
    elif diff.dim() == 3:
        diff = diff.abs()
    else:
        diff = diff.squeeze().abs()

    if diff.dim() != 3 or diff.shape[0] != 3:
        diff = diff.expand(3, *diff.shape[-2:]) if diff.dim() == 2 else diff[:3]
    # Log per-channel range after frame aggregation to spot flat channels.
    ch_ranges = [(float(diff[i].min().item()), float(diff[i].max().item())) for i in range(3)]
    print(f"[debug] channel ranges after combine {ch_ranges}", flush=True)
    # Fallback: if combined channels are flat but raw delta had variance, try a simpler reshape using absolute values.
    if all(r[0] == r[1] for r in ch_ranges) and raw_max != raw_min:
        raw_arr = raw_diff_full.abs()
        if raw_arr.dim() == 4:
            raw_arr = raw_arr.mean(dim=0)
        if raw_arr.dim() == 3 and raw_arr.shape[0] >= 3:
            diff = raw_arr[:3]
        elif raw_arr.dim() == 3 and raw_arr.shape[-1] == 3:
            diff = raw_arr.permute(2, 0, 1)
        elif raw_arr.dim() == 2:
            diff = raw_arr.expand(3, *raw_arr.shape)
        else:
            diff = diff
        ch_ranges = [(float(diff[i].min().item()), float(diff[i].max().item())) for i in range(min(3, diff.shape[0]))]
        print(f"[debug] fallback channel ranges {ch_ranges}", flush=True)

    def radial_profile(mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = mag.shape
        y, x = np.indices((h, w))
        cy, cx = h // 2, w // 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_max = int(r.max())
        radial_mean = np.zeros(r_max + 1, dtype=np.float32)
        for ri in range(r_max + 1):
            mask = (r >= ri) & (r < ri + 1)
            if mask.any():
                radial_mean[ri] = mag[mask].mean()
        return np.arange(r_max + 1), radial_mean

    def radial_energy(mag2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = mag2.shape
        y, x = np.indices((h, w))
        cy, cx = h // 2, w // 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
        r_max = int(r.max())
        energy = np.bincount(r.ravel(), weights=mag2.ravel(), minlength=r_max + 1)
        count = np.bincount(r.ravel(), minlength=r_max + 1)
        return np.arange(r_max + 1), energy, count

    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(5, 3, height_ratios=[3, 2, 1.5, 1.5, 1])
    heat_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    radial_ax = fig.add_subplot(gs[2, :])
    energy_ax = fig.add_subplot(gs[3, :])
    hist_ax = fig.add_subplot(gs[4, :])

    channel_names = ["R", "G", "B"]
    colors = ["r", "g", "b"]
    cutoff_px = None
    if freq_cutoff is not None:
        cutoff_px = float(freq_cutoff) * (max(diff.shape[1], diff.shape[2]) / 2.0)

    for idx in range(3):
        ch = diff[idx].numpy().astype(np.float32)
        fft = np.fft.fftshift(np.fft.fft2(ch))
        mag = np.log1p(np.abs(fft))
        # Use percentile-based scaling to avoid flat visuals on tiny ranges.
        vmin_p, vmax_p = np.percentile(mag, [1, 99])
        if vmax_p <= vmin_p:
            vmin_p, vmax_p = float(mag.min()), float(mag.max())
        denom = (vmax_p - vmin_p) + 1e-8
        mag_norm = np.clip((mag - vmin_p) / denom, 0.0, 1.0)

        ax = fig.add_subplot(gs[0, idx])
        # Gamma correction for visibility of low-magnitude structure (purely for display).
        gamma = 0.5
        im = ax.imshow(np.power(mag_norm, gamma), cmap="magma")
        ax.set_title(f"{channel_names[idx]} FFT")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        r, prof = radial_profile(mag_norm)
        radial_ax.plot(r, prof, color=colors[idx], label=channel_names[idx])

        # Delta spatial heatmap (abs, per-channel, per-image normalization).
        delta_ch = ch
        delta_min = float(delta_ch.min())
        delta_max = float(delta_ch.max())
        delta_norm = (delta_ch - delta_min) / (delta_max - delta_min + 1e-8)
        him = heat_axes[idx].imshow(delta_norm, cmap="coolwarm", vmin=0.0, vmax=1.0)
        heat_axes[idx].set_title(f"{channel_names[idx]} |delta|")
        heat_axes[idx].axis("off")
        fig.colorbar(him, ax=heat_axes[idx], fraction=0.046, pad=0.04)

    radial_ax.set_title("Radial FFT magnitude (log-scaled)")
    radial_ax.set_xlabel("Radius (pixels)")
    radial_ax.set_ylabel("Mean magnitude")
    radial_ax.legend()
    if cutoff_px is not None:
        radial_ax.axvline(cutoff_px, color="black", linestyle="--", linewidth=1, label="cutoff")

    # Energy profile: linear |F| and cumulative energy
    ch0 = diff[0].numpy().astype(np.float32)  # reuse first channel for radial axis length
    fft_any = np.fft.fftshift(np.fft.fft2(ch0))
    _, energy, _ = radial_energy(np.abs(fft_any) ** 2)
    energy_norm = energy / (energy.max() + 1e-12)
    cumulative = energy.cumsum()
    total_energy = cumulative[-1] + 1e-12
    cumulative_frac = cumulative / total_energy
    energy_ax.plot(np.arange(len(energy_norm)), energy_norm, label="energy (norm)")
    energy_ax.plot(np.arange(len(cumulative_frac)), cumulative_frac, label="cumulative energy")
    if cutoff_px is not None:
        energy_ax.axvline(cutoff_px, color="black", linestyle="--", linewidth=1, label="cutoff")
    energy_ax.set_title("Radial energy")
    energy_ax.set_xlabel("Radius (pixels)")
    energy_ax.set_ylabel("Energy (norm) / Cumulative")
    energy_ax.legend()

    # Low-frequency leakage / DC report
    try:
        h, w = diff.shape[1], diff.shape[2]
        cy, cx = h // 2, w // 2
        dc_vals = []
        leakage = None
        if freq_cutoff is not None and cutoff_px is not None:
            cutoff_bin = int(round(cutoff_px))
        else:
            cutoff_bin = None
        for idx in range(3):
            ch_fft = np.fft.fftshift(np.fft.fft2(diff[idx].numpy().astype(np.float32)))
            mag2 = np.abs(ch_fft) ** 2
            dc_vals.append(float(mag2[cy, cx]))
            _, energy_r, _ = radial_energy(mag2)
            cum = energy_r.cumsum()
            tot = cum[-1] + 1e-12
            if cutoff_bin is not None and cutoff_bin < len(cum):
                if mode == "high":
                    leakage = float(cum[cutoff_bin] / tot)
                elif mode == "low":
                    leakage = float((tot - cum[cutoff_bin]) / tot)
        print(f"[debug] DC components {dc_vals} leakage={leakage}", flush=True)
    except Exception as exc:
        print(f"[debug] energy leakage calc failed: {exc}", flush=True)

    # Histogram of delta values (all channels flattened, signed) on raw delta (pre-averaging)
    flat_vals = raw_diff_full.flatten().numpy()
    unique_vals = np.unique(flat_vals)
    bins = min(200, max(50, int(flat_vals.size / 2000)))  # scale bins with data size for detail
    hist_ax.hist(flat_vals, bins=bins, color="gray", alpha=0.85, log=True)
    if epsilon is not None:
        hist_ax.axvline(epsilon, color="red", linestyle="--", linewidth=1)
        hist_ax.axvline(-epsilon, color="blue", linestyle="--", linewidth=1)
    hist_ax.set_title("Delta value histogram")
    hist_ax.set_xlabel("Delta value")
    hist_ax.set_ylabel("Count")
    hist_ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    print(f"[debug] histogram unique values count={unique_vals.size}", flush=True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Save delta as an image for quick inspection (per-frame averaged).
    delta_path = path.with_name(f"{path.stem}_delta.png")
    vis = diff.mean(dim=0, keepdim=False) if diff.dim() == 3 else diff
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    delta_img = (vis.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if delta_img.ndim == 2:
        delta_img = np.stack([delta_img] * 3, axis=-1)
    else:
        delta_img = delta_img.transpose(1, 2, 0)
    Image.fromarray(delta_img).save(delta_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device_list = list_available_devices(args.device, max_workers=1)
    device = torch.device(canonical_device_id(device_list[0]))

    samples, _, _, _ = load_samples_for_dataset(args, cache_dir)

    model_source = ensure_model_available(
        args.model_id,
        args.model_path,
        cache_dir=cache_dir,
        offline=args.offline,
    )
    model, processor = prepare_model(
        str(model_source),
        device=str(device),
        cache_dir=str(cache_dir),
        local_files_only=args.offline,
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    context_cache: Dict[tuple, int] = {}
    metrics = {mode: {"count": 0, "success": 0, "flip": 0, "fidelity": []} for mode in MODES}

    for sample in samples:
        rec = run_sample(sample, model, processor, device, args, context_cache, out_dir)
        if rec is None:
            continue
        for mode in MODES:
            res = rec["modes"][mode]
            metrics[mode]["count"] += 1
            metrics[mode]["flip"] += int(res["flip"])
            metrics[mode]["success"] += int(res["success"])
            fid = res.get("fidelity")
            if fid is not None:
                metrics[mode]["fidelity"].append(fid)

    summary = {}
    for mode, vals in metrics.items():
        count = vals["count"]
        if count == 0:
            summary[mode] = {"asr": 0.0, "pfr": 0.0, "fidelity_mean": None, "samples": 0}
            continue
        summary[mode] = {
            "asr": vals["success"] / count,
            "pfr": vals["flip"] / count,
            "fidelity_mean": float(np.mean(vals["fidelity"])) if vals["fidelity"] else None,
            "samples": count,
        }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for mode, stats in summary.items():
        print(f"[{mode}] ASR={stats['asr']:.3f} PFR={stats['pfr']:.3f} samples={stats['samples']} fidelity={stats['fidelity_mean']}", flush=True)


if __name__ == "__main__":
    main()
