"""LLaVA Hugging Face adapter."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from .base import HookOutputs
from .qwen_adapter import HFVLMAdapter

logger = logging.getLogger(__name__)


class LLaVAAdapter(HFVLMAdapter):
    """Adapter for HF LLaVA models such as llava-v1.6-mistral-7b-hf."""

    def __init__(self) -> None:
        super().__init__()
        self._last_vision_token_range: Tuple[int, int] = (0, 0)
        self._last_patch_grid: Optional[Tuple[int, int]] = None

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """Compatibility alias for older adapter call sites."""
        self.load(*args, **kwargs)

    def unload_model(self) -> None:
        """Compatibility alias for older adapter call sites."""
        self.unload()

    def _llava_image_token_id(self) -> Optional[int]:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        config = getattr(self._model, "config", None)
        for obj in (tokenizer, self._processor, config):
            if obj is None:
                continue
            for attr in ("image_token_id", "image_token_index"):
                value = getattr(obj, attr, None)
                if value is not None:
                    return int(value)
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            for token in ("<image>", "<im_patch>"):
                value = tokenizer.convert_tokens_to_ids(token)
                if value is not None and value != getattr(tokenizer, "unk_token_id", None):
                    return int(value)
        return None

    def _image_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        image_token_id = self._llava_image_token_id()
        if image_token_id is None:
            return []
        ids_flat = input_ids[0].tolist()
        return [idx for idx, token_id in enumerate(ids_flat) if token_id == image_token_id]

    # Aspect-ratio band within which an exact factorisation is preferred over
    # padded approximation. 1:8 is generous enough to cover landscape/portrait
    # AnyRes tiles while excluding pathological 1xN or Nx1 factorisations.
    _MAX_ASPECT_LOG_RATIO: float = math.log(8.0)

    @classmethod
    def _exact_factor_grid(
        cls,
        n_tokens: int,
        preferred_ratio: float,
    ) -> Optional[Tuple[int, int]]:
        """Return an exact factor pair of ``n_tokens`` closest to ``preferred_ratio``.

        Only considers factorisations whose log aspect ratio is within
        ``_MAX_ASPECT_LOG_RATIO`` of square (i.e. 1:8 to 8:1). Returns ``None``
        if no factorisation is in band, so the caller can fall back to padded
        approximation.
        """
        if n_tokens <= 0:
            return None
        if n_tokens == 1:
            return (1, 1)
        best: Optional[Tuple[float, int, int]] = None
        limit = int(math.isqrt(n_tokens))
        for h in range(1, limit + 1):
            if n_tokens % h != 0:
                continue
            for cand_h, cand_w in ((h, n_tokens // h), (n_tokens // h, h)):
                squareness = abs(math.log(max(cand_h, 1) / max(cand_w, 1)))
                if squareness > cls._MAX_ASPECT_LOG_RATIO:
                    continue
                score = abs(math.log(max(cand_h / max(cand_w, 1), 1e-12) / max(preferred_ratio, 1e-12)))
                if best is None or score < best[0]:
                    best = (score, cand_h, cand_w)
        if best is None:
            return None
        return (best[1], best[2])

    @staticmethod
    def _padded_grid_for_tokens(
        n_tokens: int,
        preferred_ratio: float,
    ) -> Optional[Tuple[int, int]]:
        """Fallback when no in-band exact factorisation exists.

        Returns a grid with ``h*w >= n_tokens``. The spectral pipeline
        zero-pads the attention tensor to fit, which introduces a small FFT
        leakage bias --- acceptable only when an exact factor would be
        pathologically thin (e.g. 1739 = 1 * 1739).
        """
        if n_tokens <= 0:
            return None
        if n_tokens == 1:
            return (1, 1)
        ratio = preferred_ratio if preferred_ratio > 0 else 1.0
        target_h = max(1, int(round(math.sqrt(n_tokens * ratio))))
        search_radius = max(8, int(math.sqrt(n_tokens)) + 4)
        best: Optional[Tuple[float, int, int]] = None
        for h in range(max(1, target_h - search_radius), target_h + search_radius + 1):
            w = max(1, math.ceil(n_tokens / h))
            area = h * w
            if area < n_tokens:
                continue
            grid_ratio = h / max(w, 1)
            ratio_penalty = abs(math.log(max(grid_ratio, 1e-12) / ratio))
            padding_penalty = (area - n_tokens) / max(n_tokens, 1)
            score = ratio_penalty + padding_penalty * 4.0
            if best is None or score < best[0]:
                best = (score, h, w)
        if best is None:
            side = int(math.ceil(math.sqrt(n_tokens)))
            return (side, side)
        return (int(best[1]), int(best[2]))

    def _infer_num_image_tokens(
        self,
        inputs: Dict[str, Any],
        image: Image.Image,
        *,
        expanded_seq_len: Optional[int] = None,
    ) -> int:
        input_ids = inputs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            positions = self._image_token_positions(input_ids)
            if len(positions) > 1:
                return max(1, positions[-1] - positions[0] + 1)
            if len(positions) == 1 and expanded_seq_len is not None:
                inferred = int(expanded_seq_len) - int(input_ids.shape[1]) + 1
                if inferred > 0:
                    return inferred

        config = getattr(self._model, "config", None)
        image_seq_length = getattr(config, "image_seq_length", None)
        if image_seq_length is not None:
            return max(1, int(image_seq_length))

        pixel_values = inputs.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim >= 4:
            height = int(pixel_values.shape[-2])
            width = int(pixel_values.shape[-1])
        else:
            width, height = image.size

        vision_config = getattr(config, "vision_config", None)
        patch_size = int(getattr(vision_config, "patch_size", 14) or 14)
        return max(1, math.ceil(height / patch_size) * math.ceil(width / patch_size))

    def _patch_grid_from_num_tokens(
        self,
        n_tokens: int,
        image: Image.Image,
    ) -> Optional[Tuple[int, int]]:
        """Prefer exact factorisation; only pad (and warn) when forced.

        Padding ``n_tokens`` up to ``h*w`` causes the downstream spectral
        pipeline (``analysis/spectral.py:280-282``) to zero-pad the attention
        tensor before FFT, which spreads spectral energy and biases ``W_t``.
        We avoid that path whenever an in-band exact factor exists.
        """
        preferred_ratio = image.height / max(image.width, 1)
        exact = self._exact_factor_grid(n_tokens, preferred_ratio)
        if exact is not None:
            return exact
        padded = self._padded_grid_for_tokens(n_tokens, preferred_ratio)
        if padded is not None and padded[0] * padded[1] > n_tokens:
            logger.warning(
                "LLaVA patch grid for n_tokens=%d has no in-band exact factor; "
                "using padded grid %dx%d (pad=%d tokens). Spectral W_t will "
                "carry mild FFT zero-padding bias for this sample.",
                n_tokens,
                padded[0],
                padded[1],
                padded[0] * padded[1] - n_tokens,
            )
        return padded

    def _infer_patch_grid_via_processor(
        self,
        image: Image.Image,
    ) -> Optional[Tuple[int, int]]:
        """LLaVA-OneVision patch grid from ``image_sizes`` + SigLIP patch_size.

        Avoids the (more expensive and fragile) parent implementation that
        assumes Qwen-specific ``image_grid_thw``. AnyRes tiles the image into
        N sub-tiles of fixed processor resolution; the per-tile grid is
        ``ceil(tile_H / patch_size) x ceil(tile_W / patch_size)`` and the
        total token count is N * tile_grid_h * tile_grid_w. We return the
        per-tile grid (the natural unit for radial FFT analysis); the caller
        slices attention per-tile or stacks tiles row-major as appropriate.
        Returns ``None`` if any of the required fields are unavailable.
        """
        if self._processor is None:
            return None
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is None:
            return None
        try:
            vision_inputs = image_processor(
                images=[image],
                return_tensors="pt",
            )
        except Exception as exc:
            logger.debug(
                "LLaVA vision-only preprocessing failed for patch-grid inference: %s",
                exc,
            )
            return None
        pixel_values = vision_inputs.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim < 4:
            return None
        tile_h = int(pixel_values.shape[-2])
        tile_w = int(pixel_values.shape[-1])
        config = getattr(self._model, "config", None)
        vision_config = getattr(config, "vision_config", None) if config is not None else None
        patch_size = getattr(vision_config, "patch_size", None) if vision_config is not None else None
        if not patch_size:
            return None
        return (max(1, math.ceil(tile_h / int(patch_size))),
                max(1, math.ceil(tile_w / int(patch_size))))

    def _build_inputs(
        self,
        image: Image.Image,
        text: str,
        *,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        if not hasattr(self._processor, "apply_chat_template"):
            raise RuntimeError(
                "LLaVA processor lacks apply_chat_template; refusing to fall "
                "back to a hardcoded USER:/ASSISTANT: format that would "
                "silently produce a wrong-format prompt for chat-tuned LLaVA "
                "variants (e.g. LLaVA-OneVision uses the Qwen2 <|im_start|> "
                "template). Upgrade transformers or pin a processor that "
                "supports apply_chat_template."
            )
        prompt_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        image_token_id = self._llava_image_token_id()
        self._last_vision_token_range = (0, 0)
        self._last_patch_grid = None
        if image_token_id is not None and "input_ids" in inputs:
            positions = self._image_token_positions(inputs["input_ids"])
            if positions:
                if len(positions) > 1:
                    self._last_vision_token_range = (positions[0], positions[-1] + 1)
                else:
                    n_image_tokens = self._infer_num_image_tokens(inputs, image)
                    self._last_vision_token_range = (positions[0], positions[0] + n_image_tokens)
                n_tokens = self._last_vision_token_range[1] - self._last_vision_token_range[0]
                self._last_patch_grid = self._patch_grid_from_num_tokens(n_tokens, image)

        moved: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self._device_obj)
            else:
                moved[key] = value
        moved["return_dict"] = True
        if output_attentions:
            moved["output_attentions"] = True
        if output_hidden_states:
            moved["output_hidden_states"] = True
        return moved

    def get_vision_token_span(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Return the contiguous expanded LLaVA image-token span.

        Always recomputed from the given ``input_ids`` --- never returns the
        cached ``_last_vision_token_range``, which can be stale across
        successive ``_build_inputs`` calls on different images.
        """
        positions = self._image_token_positions(input_ids)
        if not positions:
            return (0, 0)
        if len(positions) > 1:
            return (positions[0], positions[-1] + 1)
        return self._find_vision_token_range(input_ids)

    def _resolve_vision_token_range(
        self,
        inputs: Dict[str, Any],
        image: Image.Image,
        *,
        expanded_seq_len: Optional[int] = None,
    ) -> Tuple[int, int]:
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return (0, 0)
        positions = self._image_token_positions(input_ids)
        if not positions:
            return (0, 0)
        start = positions[0]
        if len(positions) > 1:
            end = positions[-1] + 1
        else:
            end = start + self._infer_num_image_tokens(
                inputs,
                image,
                expanded_seq_len=expanded_seq_len,
            )
        if expanded_seq_len is not None:
            end = min(end, int(expanded_seq_len))
        if end <= start:
            return (0, 0)
        self._last_vision_token_range = (start, end)
        self._last_patch_grid = self._patch_grid_from_num_tokens(end - start, image)
        return self._last_vision_token_range

    def _find_vision_token_range(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Range from input_ids alone (does not expand a single placeholder)."""
        positions = self._image_token_positions(input_ids)
        if not positions:
            return (0, 0)
        return (positions[0], positions[-1] + 1)

    def _get_patch_grid_from_inputs(
        self,
        inputs: Dict[str, Any],
        image: Image.Image,
    ) -> Optional[Tuple[int, int]]:
        if self._last_patch_grid is not None:
            return self._last_patch_grid
        input_ids = inputs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            positions = self._image_token_positions(input_ids)
            if positions:
                n_tokens = positions[-1] - positions[0] + 1
                patch_grid = self._patch_grid_from_num_tokens(n_tokens, image)
                if patch_grid is not None:
                    self._last_patch_grid = patch_grid
                    return patch_grid
        return super()._get_patch_grid_from_inputs(inputs, image)

    def prepare_inputs(self, image: Image.Image, text: str) -> Dict[str, Any]:
        return self._build_inputs(image, text)

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
        attention_extra_last_layers: int = 0,
        post_fusion_layer_index: Optional[int] = None,
        post_fusion_layer_fraction: float = 0.8,
    ) -> HookOutputs:
        inputs = self._build_inputs(
            image,
            prompt,
            output_attentions=extract_attention,
            output_hidden_states=extract_post_fusion,
        )
        backend = self._attention_extract_implementation if extract_attention else None
        with self._temporary_attention_backend(backend):
            outputs = self._model(**inputs)

        expanded_seq_len = None
        attentions = list(getattr(outputs, "attentions", None) or [])
        for attn in attentions:
            if attn is not None:
                expanded_seq_len = int(attn.shape[-1])
                break
        if expanded_seq_len is None:
            hidden_states = getattr(outputs, "hidden_states", None) or ()
            if hidden_states:
                expanded_seq_len = int(hidden_states[-1].shape[1])

        vision_range = self._resolve_vision_token_range(
            inputs,
            image,
            expanded_seq_len=expanded_seq_len,
        )
        patch_grid = self._get_patch_grid_from_inputs(inputs, image)

        pre_fusion_features = None
        if extract_pre_fusion:
            pre_fusion_features, direct_grid = self._flatten_vision_tokens(image)
            if direct_grid is not None:
                patch_grid = direct_grid

        result = HookOutputs(
            patch_grid=patch_grid,
            vision_token_range=vision_range,
            pre_fusion_features=pre_fusion_features,
        )

        vis_start, vis_end = vision_range
        if extract_attention:
            attn_list: List[torch.Tensor] = []
            layer_indices: List[int] = []
            extra_last = max(0, int(attention_extra_last_layers or 0))
            extra_last_start = max(0, len(attentions) - extra_last) if extra_last else len(attentions)
            for layer_index, attn in enumerate(attentions):
                if attn is None or vis_end <= vis_start:
                    continue
                is_stride_layer = layer_stride <= 1 or layer_index % layer_stride == 0
                is_extra_last_layer = extra_last > 0 and layer_index >= extra_last_start
                if not is_stride_layer and not is_extra_last_layer:
                    continue
                seq_len = int(attn.shape[-1])
                lang_mask = torch.ones(seq_len, dtype=torch.bool, device=attn.device)
                lang_mask[vis_start:vis_end] = False
                cross = attn[0, :, lang_mask, vis_start:vis_end].detach().cpu()
                attn_list.append(cross)
                layer_indices.append(layer_index)
            result.cross_attention_weights = attn_list or None
            result.cross_attention_layer_indices = layer_indices or None

        if extract_post_fusion:
            hidden_states = getattr(outputs, "hidden_states", None) or ()
            if hidden_states:
                from ..utils.layer_groups import resolve_post_fusion_hidden_state_index

                target_layer = resolve_post_fusion_hidden_state_index(
                    len(hidden_states),
                    explicit_index=post_fusion_layer_index,
                    fraction=post_fusion_layer_fraction,
                )
                post_all = hidden_states[target_layer][0]
                result.post_fusion_all_features = post_all.detach().cpu()
                post = post_all
                if vis_end > vis_start and post_all.shape[0] >= vis_end:
                    post = post_all[vis_start:vis_end]
                result.post_fusion_features = post.detach().cpu()
                result.post_fusion_layer_index = target_layer

        return result

    def get_patch_grid_shape(self, image: Image.Image) -> Tuple[int, int]:
        try:
            inputs = self._build_inputs(image, "")
            patch_grid = self._get_patch_grid_from_inputs(inputs, image)
            if patch_grid is not None and patch_grid[0] > 0 and patch_grid[1] > 0:
                return patch_grid
        except Exception:
            pass
        return super().get_patch_grid_shape(image)
