"""Shared Hugging Face VLM adapter for Qwen, LLaVA, and InternVL families."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

from ..utils.parent_bridge import (
    get_context_length_cached,
    get_vision_tokens as parent_get_vision_tokens,
    prepare_model,
    score_options_loglik_batch,
)
from .base import HookOutputs, VLMAdapter

logger = logging.getLogger(__name__)


class HFVLMAdapter(VLMAdapter):
    """Adapter for current Hugging Face image-text-to-text VLM families."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device_obj = torch.device("cpu")
        self._model_id = ""
        self._context_len_cache: Dict[tuple, int] = {}

    def load(
        self,
        model_id: str,
        device: str,
        cache_dir: Optional[str] = None,
        *,
        quantization: Optional[str] = None,
        trust_remote_code: bool = True,
        device_map: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> None:
        load_kwargs: Dict[str, Any] = {}
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = device_map or "auto"
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = device_map or "auto"
        elif device_map:
            load_kwargs["device_map"] = device_map

        self._model, self._processor = prepare_model(
            model_id,
            str(device),
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **load_kwargs,
        )
        self._model_id = model_id
        self._device_obj = next(self._model.parameters()).device
        self._context_len_cache.clear()
        logger.info("Loaded %s on %s", model_id, self._device_obj)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._context_len_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> torch.device:
        return self._device_obj

    @property
    def num_layers(self) -> int:
        return len(self._get_decoder_layers())

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
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
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

    def _iter_objects(self) -> Iterable[Any]:
        model = self._model
        if model is None:
            return []
        candidates = [model]
        for attr in ("model", "language_model"):
            child = getattr(model, attr, None)
            if child is not None:
                candidates.append(child)
        model_root = getattr(model, "model", None)
        if model_root is not None:
            for attr in ("language_model", "vision_tower", "visual"):
                child = getattr(model_root, attr, None)
                if child is not None:
                    candidates.append(child)
        return candidates

    def _get_decoder_layers(self) -> List[Any]:
        for obj in self._iter_objects():
            layers = getattr(obj, "layers", None)
            if layers is not None:
                return list(layers)
            model = getattr(obj, "model", None)
            layers = getattr(model, "layers", None)
            if layers is not None:
                return list(layers)
        return []

    def _image_token_id(self) -> Optional[int]:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        for obj in (tokenizer, self._processor, getattr(self._model, "config", None)):
            if obj is None:
                continue
            for attr in ("image_token_id", "image_token_index", "img_context_token_id"):
                value = getattr(obj, attr, None)
                if value is not None:
                    return int(value)
        return None

    def _find_special_token_id(self, *names: str) -> Optional[int]:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        for obj in (tokenizer, self._processor, getattr(self._model, "config", None)):
            if obj is None:
                continue
            for name in names:
                value = getattr(obj, name, None)
                if value is not None:
                    return int(value)
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            for name in names:
                try:
                    value = tokenizer.convert_tokens_to_ids(name)
                except Exception:
                    continue
                if value is not None and value != getattr(tokenizer, "unk_token_id", None):
                    return int(value)
        return None

    def _find_vision_token_range(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        ids_flat = input_ids[0].tolist()
        start_id = self._find_special_token_id("vision_start_token_id", "<|vision_start|>")
        end_id = self._find_special_token_id("vision_end_token_id", "<|vision_end|>")
        if start_id is not None and end_id is not None:
            try:
                start = ids_flat.index(start_id)
                end = ids_flat.index(end_id, start)
                return (start + 1, end)
            except ValueError:
                pass

        image_token_id = self._image_token_id()
        if image_token_id is not None:
            positions = [idx for idx, token_id in enumerate(ids_flat) if token_id == image_token_id]
            if positions:
                return (positions[0], positions[-1] + 1)

        return (0, 0)

    def _flatten_vision_tokens(
        self,
        image: Image.Image,
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]]]:
        if self._model is None or self._processor is None:
            return None, None
        tokens = parent_get_vision_tokens(
            self._model,
            self._processor,
            image,
            str(self._device_obj),
        )
        if tokens is None:
            return None, None
        if tokens.dim() == 4:
            _, h, w, dim = tokens.shape
            return tokens.reshape(-1, dim).detach().cpu(), (int(h), int(w))
        if tokens.dim() == 3:
            h, w, dim = tokens.shape
            return tokens.reshape(-1, dim).detach().cpu(), (int(h), int(w))
        if tokens.dim() == 2:
            n_tokens = int(tokens.shape[0])
            side = max(1, int(math.isqrt(n_tokens)))
            return tokens.detach().cpu(), (side, max(1, n_tokens // side))
        return None, None

    def _get_patch_grid_from_inputs(
        self,
        inputs: Dict[str, Any],
        image: Image.Image,
    ) -> Optional[Tuple[int, int]]:
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            visual_module = getattr(getattr(self._model, "model", None), "visual", None)
            merge_size = int(getattr(visual_module, "spatial_merge_size", 1))
            _, h, w = [int(x) for x in grid[0].tolist()]
            return (max(1, h // merge_size), max(1, w // merge_size))

        _, patch_grid = self._flatten_vision_tokens(image)
        return patch_grid

    def score_options(
        self,
        image: Image.Image,
        question: str,
        options: Dict[str, str],
    ) -> Dict[str, float]:
        if not options:
            return {}
        device_str = str(self._device_obj)
        context_len = get_context_length_cached(
            self._processor,
            image,
            question,
            device=device_str,
            cache=self._context_len_cache,
        )
        return score_options_loglik_batch(
            self._model,
            self._processor,
            image,
            question,
            options,
            device=device_str,
            context_len=context_len,
        )

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        inputs = self._build_inputs(image, prompt)
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        input_len = inputs["input_ids"].shape[1]
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        return tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)

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
        inputs = self._build_inputs(
            image,
            prompt,
            output_attentions=extract_attention,
            output_hidden_states=extract_post_fusion,
        )
        outputs = self._model(**inputs)

        vision_range = self._find_vision_token_range(inputs["input_ids"])
        pre_fusion_features = None
        patch_grid = None
        if extract_pre_fusion:
            pre_fusion_features, patch_grid = self._flatten_vision_tokens(image)
        else:
            patch_grid = self._get_patch_grid_from_inputs(inputs, image)

        result = HookOutputs(
            patch_grid=patch_grid,
            vision_token_range=vision_range,
            pre_fusion_features=pre_fusion_features,
        )

        vis_start, vis_end = vision_range
        if extract_attention:
            attn_list: List[torch.Tensor] = []
            for layer_index, attn in enumerate(getattr(outputs, "attentions", None) or []):
                if attn is None or vis_end <= vis_start:
                    continue
                if layer_stride > 1 and layer_index % layer_stride != 0:
                    continue
                seq_len = attn.shape[-1]
                lang_mask = torch.ones(seq_len, dtype=torch.bool, device=attn.device)
                lang_mask[vis_start:vis_end] = False
                cross = attn[0, :, lang_mask, vis_start:vis_end].detach().cpu()
                attn_list.append(cross)
            result.cross_attention_weights = attn_list or None

        if extract_post_fusion:
            hidden_states = getattr(outputs, "hidden_states", None) or ()
            if hidden_states:
                target_layer = min(3, len(hidden_states) - 1)
                post = hidden_states[target_layer][0]
                if vis_end > vis_start and post.shape[0] >= vis_end:
                    post = post[vis_start:vis_end]
                result.post_fusion_features = post.detach().cpu()

        return result

    def get_vision_tokens(self, image: Image.Image) -> Optional[torch.Tensor]:
        flat_tokens, patch_grid = self._flatten_vision_tokens(image)
        if flat_tokens is None or patch_grid is None:
            return None
        h, w = patch_grid
        if h * w != flat_tokens.shape[0]:
            return flat_tokens
        return flat_tokens.reshape(1, h, w, -1)

    def get_patch_grid_shape(self, image: Image.Image) -> Tuple[int, int]:
        _, patch_grid = self._flatten_vision_tokens(image)
        return patch_grid or (0, 0)


QwenVLAdapter = HFVLMAdapter
