"""LLaVA Hugging Face adapter."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

from .base import HookOutputs
from .qwen_adapter import HFVLMAdapter


class LLaVAAdapter(HFVLMAdapter):
    """Adapter for HF LLaVA models such as llava-v1.6-mistral-7b-hf."""

    def __init__(self) -> None:
        super().__init__()
        self._last_vision_token_range: Tuple[int, int] = (0, 0)

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

    def _infer_num_image_tokens(self, inputs: Dict[str, Any], image: Image.Image) -> int:
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
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        if hasattr(self._processor, "apply_chat_template"):
            prompt_text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            eos = getattr(tokenizer, "eos_token", "") or ""
            prompt_text = f"USER: <image>\n{text}\nASSISTANT:{eos}"

        inputs = self._processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        image_token_id = self._llava_image_token_id()
        self._last_vision_token_range = (0, 0)
        if image_token_id is not None and "input_ids" in inputs:
            ids_flat = inputs["input_ids"][0].tolist()
            positions = [idx for idx, token_id in enumerate(ids_flat) if token_id == image_token_id]
            if positions:
                if len(positions) > 1:
                    self._last_vision_token_range = (positions[0], positions[-1] + 1)
                else:
                    n_image_tokens = self._infer_num_image_tokens(inputs, image)
                    self._last_vision_token_range = (positions[0], positions[0] + n_image_tokens)

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
        """Return the contiguous expanded LLaVA image-token span."""
        if self._last_vision_token_range != (0, 0):
            return self._last_vision_token_range
        return self._find_vision_token_range(input_ids)

    def _find_vision_token_range(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        if self._last_vision_token_range != (0, 0):
            return self._last_vision_token_range
        image_token_id = self._llava_image_token_id()
        if image_token_id is None:
            return (0, 0)
        ids_flat = input_ids[0].tolist()
        positions = [idx for idx, token_id in enumerate(ids_flat) if token_id == image_token_id]
        if not positions:
            return (0, 0)
        return (positions[0], positions[-1] + 1)

    def prepare_inputs(self, image: Image.Image, text: str) -> Dict[str, Any]:
        return self._build_inputs(image, text)

    def extract_attention(self, inputs: Dict[str, Any]) -> HookOutputs:
        with torch.inference_mode():
            outputs = self._model(**inputs, output_attentions=True, return_dict=True)
        vision_range = self.get_vision_token_span(inputs["input_ids"])
        vis_start, vis_end = vision_range
        attn_list = []
        layer_indices = []
        for layer_index, attn in enumerate(getattr(outputs, "attentions", None) or []):
            if attn is None or vis_end <= vis_start:
                continue
            seq_len = attn.shape[-1]
            lang_mask = torch.ones(seq_len, dtype=torch.bool, device=attn.device)
            lang_mask[vis_start:vis_end] = False
            attn_list.append(attn[0, :, lang_mask, vis_start:vis_end].detach().cpu())
            layer_indices.append(layer_index)
        return HookOutputs(
            cross_attention_weights=attn_list or None,
            cross_attention_layer_indices=layer_indices or None,
            vision_token_range=vision_range,
        )
