"""Qwen3-VL / Qwen2.5-VL adapter with hook-based cross-attention extraction.

Architecture note
-----------------
Qwen VL models do **not** have explicit cross-attention layers.  Vision
tokens are projected and concatenated into the text token sequence before
entering the Transformer decoder.  The "cross-attention" we extract is the
sub-matrix of each self-attention layer where *language* query tokens attend
to *vision* key/value tokens.

We identify vision token positions by the ``<|vision_start|>`` /
``<|vision_end|>`` special tokens in ``input_ids``.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from .base import HookOutputs, VLMAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers (so the module loads even without the parent on path)
# ---------------------------------------------------------------------------

_PARENT_DIR = str(Path(__file__).resolve().parent.parent.parent)


def _import_parent_utils():
    """Import reusable functions from the main VLM driver."""
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    from vlm_invariance_check import (
        prepare_model,
        get_vision_tokens as _parent_get_vision_tokens,
    )
    return prepare_model, _parent_get_vision_tokens


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class QwenVLAdapter(VLMAdapter):
    """Adapter for Qwen3-VL and Qwen2.5-VL model families."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device_obj: Optional[torch.device] = None
        self._model_id: str = ""
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._hook_store: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(
        self,
        model_id: str,
        device: str,
        cache_dir: Optional[str] = None,
        *,
        quantization: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        prepare_model, _ = _import_parent_utils()

        load_kwargs: Dict[str, Any] = {}
        if quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["device_map"] = "auto"
            except ImportError:
                logger.warning("bitsandbytes not available; loading without quantization")
        elif quantization == "8bit":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                load_kwargs["device_map"] = "auto"
            except ImportError:
                logger.warning("bitsandbytes not available; loading without quantization")

        # prepare_model expects device as a string, not torch.device
        device_str = str(device) if not isinstance(device, str) else device
        self._model, self._processor = prepare_model(
            model_id,
            device_str,
            cache_dir=cache_dir,
            **load_kwargs,
        )
        self._model_id = model_id
        self._device_obj = next(self._model.parameters()).device
        logger.info(
            "Loaded %s on %s (params device: %s)",
            model_id, device, self._device_obj,
        )

    def unload(self) -> None:
        self._remove_hooks()
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> torch.device:
        return self._device_obj or torch.device("cpu")

    @property
    def num_layers(self) -> int:
        layers = getattr(getattr(self._model, "model", None), "layers", None)
        return len(layers) if layers is not None else 0

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _build_inputs(
        self,
        image: Image.Image,
        text: str,
        *,
        output_attentions: bool = False,
    ) -> Dict[str, Any]:
        """Build model inputs from (image, text) using the chat template."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        # Use processor's chat template
        prompt_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        # Move to device
        inputs = {
            k: v.to(self._device_obj) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if output_attentions:
            inputs["output_attentions"] = True
            inputs["return_dict"] = True
        return inputs

    def _find_vision_token_range(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Find start/end indices of vision tokens in the input sequence.

        Qwen VL uses special tokens to bracket vision content.  We search
        for ``<|vision_start|>`` and ``<|vision_end|>`` token IDs.
        """
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        ids_flat = input_ids[0].tolist()

        # Try to find the special token IDs
        for start_name in ("vision_start_token_id", "<|vision_start|>"):
            start_id = None
            if hasattr(tokenizer, start_name):
                start_id = getattr(tokenizer, start_name)
            elif hasattr(tokenizer, "convert_tokens_to_ids"):
                try:
                    cand = tokenizer.convert_tokens_to_ids(start_name)
                    if cand != tokenizer.unk_token_id:
                        start_id = cand
                except Exception:
                    pass
            if start_id is not None:
                break

        for end_name in ("vision_end_token_id", "<|vision_end|>"):
            end_id = None
            if hasattr(tokenizer, end_name):
                end_id = getattr(tokenizer, end_name)
            elif hasattr(tokenizer, "convert_tokens_to_ids"):
                try:
                    cand = tokenizer.convert_tokens_to_ids(end_name)
                    if cand != tokenizer.unk_token_id:
                        end_id = cand
                except Exception:
                    pass
            if end_id is not None:
                break

        if start_id is not None and end_id is not None:
            try:
                s = ids_flat.index(start_id)
                e = ids_flat.index(end_id, s)
                return (s + 1, e)  # exclude the bracket tokens themselves
            except ValueError:
                pass

        # Fallback: look for image_token_id from the processor
        image_token_id = getattr(tokenizer, "image_token_id", None)
        if image_token_id is not None:
            positions = [i for i, t in enumerate(ids_flat) if t == image_token_id]
            if positions:
                return (positions[0], positions[-1] + 1)

        # Last resort: heuristic — assume vision tokens are the first N
        # tokens where N is inferred from the image grid
        logger.warning(
            "Could not locate vision token boundaries; "
            "using heuristic based on image grid."
        )
        return (0, 0)

    def _get_patch_grid_from_inputs(
        self, inputs: Dict[str, Any]
    ) -> Tuple[int, int]:
        """Infer (H_patches, W_patches) from processor outputs."""
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            visual_module = getattr(
                getattr(self._model, "model", None), "visual", None
            )
            merge_size = int(getattr(visual_module, "spatial_merge_size", 1))
            _, h, w = [int(x) for x in grid[0].tolist()]
            return (max(1, h // merge_size), max(1, w // merge_size))

        # Fallback: guess from number of vision tokens
        vrange = self._find_vision_token_range(inputs["input_ids"])
        n_vis = vrange[1] - vrange[0]
        if n_vis > 0:
            side = int(math.isqrt(n_vis))
            if side * side == n_vis:
                return (side, side)
            return (side, max(1, n_vis // side))
        return (0, 0)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_options(
        self,
        image: Image.Image,
        question: str,
        options: Dict[str, str],
    ) -> Dict[str, float]:
        """Score MCQ options via log-likelihood."""
        scores: Dict[str, float] = {}
        for label, option_text in options.items():
            full_text = f"{question}\n{option_text}"
            inputs = self._build_inputs(image, full_text)

            with torch.inference_mode():
                outputs = self._model(**inputs)

            logits = outputs.logits  # (1, seq_len, vocab)
            # Get log-prob of the option tokens
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            option_ids = tokenizer.encode(option_text, add_special_tokens=False)

            if not option_ids:
                scores[label] = float("-inf")
                continue

            # Sum log-probs over the last len(option_ids) positions
            log_probs = torch.log_softmax(logits[0], dim=-1)
            total = 0.0
            seq_len = logits.shape[1]
            for i, tid in enumerate(option_ids):
                pos = seq_len - len(option_ids) + i - 1
                if 0 <= pos < seq_len:
                    total += log_probs[pos, tid].item()
            scores[label] = total

        return scores

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        inputs = self._build_inputs(image, prompt)
        with torch.inference_mode():
            gen_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
        # Decode only the generated tokens
        input_len = inputs["input_ids"].shape[1]
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        return tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._hook_store.clear()

    def _install_attention_hooks(self, layer_stride: int = 1) -> None:
        """Install hooks on self-attention layers to capture attention weights."""
        layers = getattr(getattr(self._model, "model", None), "layers", [])
        for i, layer in enumerate(layers):
            if i % layer_stride != 0:
                continue
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue

            key = f"attn_layer_{i}"

            def make_hook(k):
                def hook_fn(module, args, output):
                    # output is typically (hidden_states, attention_weights, ...)
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            # attn_weights: (batch, heads, seq, seq)
                            self._hook_store[k] = attn_weights[0].detach().cpu()
                return hook_fn

            h = attn.register_forward_hook(make_hook(key))
            self._hooks.append(h)

    def _install_pre_fusion_hook(self) -> None:
        """Hook the vision encoder output (before it enters the LLM)."""
        # The vision encoder is at model.model.visual or model.visual
        visual = getattr(getattr(self._model, "model", None), "visual", None)
        if visual is None:
            visual = getattr(self._model, "visual", None)
        if visual is None:
            logger.warning("Could not find vision encoder module for pre-fusion hook")
            return

        def hook_fn(module, args, output):
            # output may be a tensor or tuple
            feat = output
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if isinstance(feat, torch.Tensor):
                self._hook_store["pre_fusion"] = feat.detach().cpu()

        h = visual.register_forward_hook(hook_fn)
        self._hooks.append(h)

    def _install_post_fusion_hook(self, target_layer: int = 2) -> None:
        """Hook early decoder layers to capture post-fusion hidden states."""
        layers = getattr(getattr(self._model, "model", None), "layers", [])
        if target_layer >= len(layers):
            target_layer = min(len(layers) - 1, 2)
        layer = layers[target_layer]

        def hook_fn(module, args, output):
            feat = output
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if isinstance(feat, torch.Tensor):
                self._hook_store["post_fusion"] = feat.detach().cpu()

        h = layer.register_forward_hook(hook_fn)
        self._hooks.append(h)

    # ------------------------------------------------------------------
    # Internal extraction
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def extract_internals(
        self,
        image: Image.Image,
        prompt: str,
        *,
        extract_attention: bool = True,
        extract_pre_fusion: bool = True,
        extract_post_fusion: bool = True,
        layer_stride: int = 1,
    ) -> HookOutputs:
        self._remove_hooks()

        if extract_attention:
            self._install_attention_hooks(layer_stride)
        if extract_pre_fusion:
            self._install_pre_fusion_hook()
        if extract_post_fusion:
            self._install_post_fusion_hook()

        inputs = self._build_inputs(
            image, prompt, output_attentions=extract_attention
        )
        vision_range = self._find_vision_token_range(inputs["input_ids"])
        patch_grid = self._get_patch_grid_from_inputs(inputs)

        # Forward pass
        _ = self._model(**inputs)

        # Collect results
        result = HookOutputs(
            patch_grid=patch_grid,
            vision_token_range=vision_range,
        )

        if extract_attention and vision_range[1] > vision_range[0]:
            vis_start, vis_end = vision_range
            attn_list = []
            for key in sorted(self._hook_store):
                if not key.startswith("attn_layer_"):
                    continue
                full_attn = self._hook_store[key]  # (heads, seq, seq)
                # Extract language→vision sub-matrix
                # Language positions = everything NOT vision
                seq_len = full_attn.shape[1]
                lang_mask = torch.ones(seq_len, dtype=torch.bool)
                lang_mask[vis_start:vis_end] = False
                # cross_attn: (heads, num_lang, num_vis)
                cross_attn = full_attn[:, lang_mask, vis_start:vis_end]
                attn_list.append(cross_attn)
            result.cross_attention_weights = attn_list if attn_list else None

        if extract_pre_fusion and "pre_fusion" in self._hook_store:
            result.pre_fusion_features = self._hook_store["pre_fusion"]

        if extract_post_fusion and "post_fusion" in self._hook_store:
            result.post_fusion_features = self._hook_store["post_fusion"]

        self._remove_hooks()
        return result

    # ------------------------------------------------------------------
    # Vision tokens
    # ------------------------------------------------------------------

    def get_vision_tokens(self, image: Image.Image) -> Optional[torch.Tensor]:
        _, parent_get_vt = _import_parent_utils()
        return parent_get_vt(
            self._model,
            self._processor,
            image,
            str(self._device_obj),
        )

    def get_patch_grid_shape(self, image: Image.Image) -> Tuple[int, int]:
        inputs = self._build_inputs(image, "dummy")
        return self._get_patch_grid_from_inputs(inputs)
