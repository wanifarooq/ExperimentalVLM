#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn a small adversarial perturbation for Qwen3-VL that flips the label.")
    parser.add_argument("--image", type=Path, required=True, help="Path to an RGB image.")
    parser.add_argument("--prompt", type=str, default="Answer with a single-word label for the main object in the photo.", help="User instruction for the model.")
    parser.add_argument("--target-text", type=str, required=True, help="Text we want the model to emit after perturbation.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="HF model id.")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu (auto prefers cuda, then mps).")
    parser.add_argument("--steps", type=int, default=60, help="Gradient steps.")
    parser.add_argument("--lr", type=float, default=0.05, help="Optimizer learning rate.")
    parser.add_argument("--epsilon", type=float, default=4.0 / 255.0, help="Max L_inf perturbation in raw pixel space.")
    parser.add_argument("--log-every", type=int, default=5, help="Log frequency.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max tokens to sample when decoding answers.")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the perturbed image reconstruction.")
    parser.add_argument("--seed", type=int, default=0, help="Optional RNG seed.")
    return parser.parse_args()


def detect_device(name: str) -> Tuple[torch.device, torch.dtype]:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float16
        if torch.backends.mps.is_available():
            return torch.device("mps"), torch.float16
        return torch.device("cpu"), torch.float32
    if name.startswith("cuda"):
        return torch.device(name), torch.float16
    if name == "mps":
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def move_to_device(batch: Dict[str, Any], device: torch.device, float_dtype: torch.dtype) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            if torch.is_floating_point(val):
                out[key] = val.to(device=device, dtype=float_dtype)
            else:
                out[key] = val.to(device=device)
        else:
            out[key] = val
    return out


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def make_conversation_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
    target_text: str,
    device: torch.device,
    float_dtype: torch.dtype,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
    messages_full = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]
    full = processor.apply_chat_template(messages_full, tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True)
    full = move_to_device(full, device=device, float_dtype=float_dtype)

    messages_user = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    gen = processor.apply_chat_template(messages_user, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    gen = move_to_device(gen, device=device, float_dtype=float_dtype)
    context_len = int(gen["input_ids"].shape[1])
    return full, gen, context_len


def make_labels(input_ids: torch.Tensor, context_len: int) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, :context_len] = -100
    return labels


def sequence_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    context_len: int,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float()[:, :-1, :], dim=-1)
    if attention_mask is not None:
        seq_lens = attention_mask.sum(dim=1).to(dtype=torch.long)
    else:
        seq_lens = torch.full((input_ids.shape[0],), input_ids.shape[1], device=logits.device, dtype=torch.long)
    scores: list[torch.Tensor] = []
    for idx in range(input_ids.shape[0]):
        end = int(seq_lens[idx].item())
        if end <= context_len:
            scores.append(torch.zeros((), device=logits.device))
            continue
        tgt = input_ids[idx, context_len:end]
        lp = log_probs[idx, context_len - 1 : end - 1, :]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        scores.append(tok_lp.sum())
    return torch.stack(scores)


def build_adv_pixels(
    base_pixels: torch.Tensor,
    patch_size: int,
    epsilon: float,
    std: torch.Tensor,
    opt_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patches = base_pixels.to(dtype=opt_dtype).reshape(-1, 2, 3, patch_size, patch_size)
    bound = torch.tensor(epsilon, device=patches.device, dtype=opt_dtype).view(1, 1, 1, 1, 1)
    bound = bound / std.view(1, 1, 3, 1, 1)
    perturb = torch.zeros_like(patches, requires_grad=True)
    return patches, bound, perturb


def stitch_image(
    patches: torch.Tensor,
    grid: torch.Tensor,
    patch_size: int,
    std: torch.Tensor,
) -> Image.Image:
    t, h, w = [int(x) for x in grid[0].tolist()]
    merged = patches.view(t, h, w, 2, 3, patch_size, patch_size).mean(dim=3)
    merged = merged.permute(0, 3, 1, 4, 2, 5).reshape(t, 3, h * patch_size, w * patch_size)
    img = merged[0]
    img = (img * std.view(3, 1, 1) + 0.5).clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr)


def generate_answer(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    gen_inputs: Dict[str, torch.Tensor],
    pixel_values: torch.Tensor,
    max_new_tokens: int,
) -> str:
    inputs = {k: v for k, v in gen_inputs.items() if k != "pixel_values"}
    inputs["pixel_values"] = pixel_values
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(inputs["input_ids"][i]) :] for i, o in enumerate(outputs)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


def main() -> None:
    args = parse_args()
    if args.seed:
        torch.manual_seed(args.seed)

    device, float_dtype = detect_device(args.device)
    print(f"[info] Using device {device} with dtype {float_dtype}")

    image = load_image(args.image)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, dtype=float_dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    full_inputs, gen_inputs, context_len = make_conversation_inputs(processor, image=image, prompt=args.prompt, target_text=args.target_text, device=device, float_dtype=float_dtype)
    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs.get("attention_mask")
    grid = full_inputs["image_grid_thw"]
    base_pixels = full_inputs["pixel_values"]
    labels = make_labels(input_ids, context_len).to(device)

    patch_size = int(processor.image_processor.patch_size)
    opt_dtype = torch.float32 if float_dtype == torch.float16 else float_dtype
    std = torch.tensor(processor.image_processor.image_std, device=device, dtype=opt_dtype)
    patches, bound, perturb = build_adv_pixels(base_pixels, patch_size=patch_size, epsilon=args.epsilon, std=std, opt_dtype=opt_dtype)

    text_inputs = {k: v for k, v in full_inputs.items() if k != "pixel_values"}
    optimizer = torch.optim.Adam([perturb], lr=args.lr)

    with torch.no_grad():
        base_out = generate_answer(model=model, processor=processor, gen_inputs=gen_inputs, pixel_values=base_pixels, max_new_tokens=args.max_new_tokens)
    print(f"[info] Baseline answer: {base_out}")

    for step in range(args.steps):
        optimizer.zero_grad()
        adv_patches = torch.clamp(patches + perturb, min=patches - bound, max=patches + bound)
        adv_flat = adv_patches.reshape_as(base_pixels).to(dtype=float_dtype)
        forward_inputs = dict(text_inputs)
        forward_inputs["pixel_values"] = adv_flat
        forward_inputs["labels"] = labels
        outputs = model(**forward_inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            perturb.data.clamp_(min=-bound, max=bound)
        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                lp = sequence_logprob(logits=outputs.logits, input_ids=input_ids, attention_mask=attention_mask, context_len=context_len)
                print(f"[step {step:03d}] loss={loss.item():.4f} target_logprob={lp.item():.4f}")

    with torch.no_grad():
        adv_patches = torch.clamp(patches + perturb, min=patches - bound, max=patches + bound)
        adv_flat = adv_patches.reshape_as(base_pixels).to(dtype=float_dtype)
        adv_out = generate_answer(model=model, processor=processor, gen_inputs=gen_inputs, pixel_values=adv_flat, max_new_tokens=args.max_new_tokens)
        delta = (adv_patches - patches) * std.view(1, 1, 3, 1, 1)
        l_inf = float(delta.abs().max().item())
        print(f"[info] Final answer: {adv_out}")
        print(f"[info] Max |delta| in pixel space: {l_inf:.6f}")
        if args.save is not None:
            adv_img = stitch_image(adv_patches, grid=grid, patch_size=patch_size, std=std)
            adv_img.save(args.save)
            print(f"[info] Saved adversarial image to {args.save}")


if __name__ == "__main__":
    main()
