"""Shared Hugging Face VLM adapter for Qwen, LLaVA, and InternVL families."""

from __future__ import annotations

from contextlib import contextmanager
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
from ..utils.layer_groups import resolve_post_fusion_hidden_state_index
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
        self._default_attn_implementation: Optional[str] = None
        self._attention_extract_implementation: Optional[str] = "eager"

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
        attn_implementation: Optional[str] = None,
        attention_extract_implementation: Optional[str] = "eager",
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
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

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
        self._attention_extract_implementation = attention_extract_implementation
        self._default_attn_implementation = self._current_attn_implementation()
        logger.info("Loaded %s on %s", model_id, self._device_obj)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._context_len_cache.clear()
        self._default_attn_implementation = None
        self._attention_extract_implementation = "eager"
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

    def _current_attn_implementation(self) -> Optional[str]:
        config = getattr(self._model, "config", None)
        if config is None:
            return None
        for attr in ("_attn_implementation", "_attn_implementation_internal"):
            value = getattr(config, attr, None)
            if value:
                return str(value)
        return None

    @contextmanager
    def _temporary_attention_backend(self, implementation: Optional[str]):
        previous = self._current_attn_implementation() or self._default_attn_implementation
        switched = False
        if implementation and implementation != previous:
            setter = getattr(self._model, "set_attn_implementation", None)
            if callable(setter):
                try:
                    setter(implementation)
                except Exception as exc:
                    logger.warning(
                        "Could not enable %s attention for %s: %s",
                        implementation,
                        self._model_id,
                        exc,
                    )
                else:
                    switched = self._current_attn_implementation() == implementation
                    if not switched:
                        logger.warning(
                            "Requested %s attention for %s but active backend is %s",
                            implementation,
                            self._model_id,
                            self._current_attn_implementation(),
                        )
            else:
                logger.warning(
                    "%s does not expose set_attn_implementation(); attention extraction may be unavailable",
                    self._model.__class__.__name__,
                )
        try:
            yield
        finally:
            if switched and previous and previous != implementation:
                try:
                    self._model.set_attn_implementation(previous)
                except Exception as exc:
                    logger.warning(
                        "Could not restore %s attention for %s: %s",
                        previous,
                        self._model_id,
                        exc,
                    )

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

    def _visual_module(self) -> Optional[Any]:
        candidates: List[Any] = []
        if self._model is not None:
            candidates.append(self._model)
            model_root = getattr(self._model, "model", None)
            if model_root is not None:
                candidates.append(model_root)
            language_model = getattr(self._model, "language_model", None)
            if language_model is not None:
                candidates.append(language_model)
        for obj in candidates:
            for attr in ("visual", "vision_tower", "vision_model"):
                module = getattr(obj, attr, None)
                if module is not None:
                    return module
        return None

    def _spatial_merge_size(self) -> int:
        visual_module = self._visual_module()
        value = getattr(visual_module, "spatial_merge_size", None)
        if value is None:
            value = getattr(getattr(self._model, "config", None), "spatial_merge_size", None)
        try:
            return max(1, int(value or 1))
        except (TypeError, ValueError):
            return 1

    def _vision_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        visual_module = self._visual_module()
        if visual_module is not None:
            try:
                param = next(visual_module.parameters())
                return param.device, param.dtype
            except StopIteration:
                pass
            except Exception:
                pass
            try:
                buffer = next(visual_module.buffers())
                return buffer.device, buffer.dtype if torch.is_floating_point(buffer) else torch.float16
            except StopIteration:
                pass
            except Exception:
                pass
        first_param = next(self._model.parameters(), None) if self._model is not None else None
        if first_param is not None:
            return first_param.device, first_param.dtype
        return self._device_obj, torch.float16

    @staticmethod
    def _first_tensor(value: Any) -> Optional[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            for item in value.values():
                tensor = HFVLMAdapter._first_tensor(item)
                if tensor is not None:
                    return tensor
            return None
        if hasattr(value, "to_tuple"):
            try:
                return HFVLMAdapter._first_tensor(value.to_tuple())
            except Exception:
                return None
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = HFVLMAdapter._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @staticmethod
    def _factor_grid(
        n_tokens: int,
        *,
        preferred_ratio: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        if n_tokens <= 0:
            return None
        if n_tokens == 1:
            return (1, 1)
        if preferred_ratio is None or preferred_ratio <= 0:
            side = int(math.isqrt(n_tokens))
            for h in range(side, 0, -1):
                if n_tokens % h == 0:
                    return (h, n_tokens // h)
            return (1, n_tokens)
        best: Optional[Tuple[float, int, int]] = None
        limit = int(math.isqrt(n_tokens))
        for h in range(1, limit + 1):
            if n_tokens % h != 0:
                continue
            for cand_h, cand_w in ((h, n_tokens // h), (n_tokens // h, h)):
                ratio = cand_h / max(cand_w, 1)
                score = abs(math.log(max(ratio, 1e-12) / preferred_ratio))
                if best is None or score < best[0]:
                    best = (score, cand_h, cand_w)
        if best is None:
            return None
        return (best[1], best[2])

    def _grid_from_image_grid(
        self,
        grid: Any,
        n_tokens: int,
    ) -> Optional[Tuple[int, int, int]]:
        if grid is None or n_tokens <= 0:
            return None
        try:
            if isinstance(grid, torch.Tensor):
                grid_values = grid[0].detach().cpu().tolist()
            else:
                grid_values = grid[0].tolist() if hasattr(grid[0], "tolist") else grid[0]
            t, h, w = [max(1, int(x)) for x in grid_values[:3]]
        except Exception:
            return None

        merge_candidates = [self._spatial_merge_size(), 1, 2, 4]
        seen = set()
        for merge_size in merge_candidates:
            if merge_size in seen or merge_size <= 0:
                continue
            seen.add(merge_size)
            h_tokens = max(1, h // merge_size)
            w_tokens = max(1, w // merge_size)
            if t * h_tokens * w_tokens == n_tokens:
                return (t, h_tokens, w_tokens)
            if h_tokens * w_tokens == n_tokens:
                return (1, h_tokens, w_tokens)

        # Dynamic-resolution Qwen outputs are often rectangular. If the model's
        # reported merge size does not match the returned token count, preserve
        # the processor's aspect ratio and factor the actual token count.
        preferred_ratio = h / max(w, 1)
        if n_tokens % t == 0:
            per_frame = n_tokens // t
            factored = self._factor_grid(per_frame, preferred_ratio=preferred_ratio)
            if factored is not None:
                return (t, factored[0], factored[1])
        factored = self._factor_grid(n_tokens, preferred_ratio=preferred_ratio)
        if factored is not None:
            return (1, factored[0], factored[1])
        return None

    def _normalise_vision_tokens(
        self,
        tokens: torch.Tensor,
        *,
        grid: Any = None,
        image: Optional[Image.Image] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]]]:
        if not isinstance(tokens, torch.Tensor):
            return None, None
        if tokens.dim() == 4:
            t, h, w, dim = tokens.shape
            return tokens.reshape(-1, dim).detach().cpu(), (int(h), int(w))
        if tokens.dim() == 3 and tokens.shape[0] == 1:
            tokens = tokens[0]
        elif tokens.dim() == 3 and grid is not None:
            tokens = tokens.reshape(-1, tokens.shape[-1])
        elif tokens.dim() == 3:
            h, w, dim = tokens.shape
            return tokens.reshape(-1, dim).detach().cpu(), (int(h), int(w))
        elif tokens.dim() > 3:
            tokens = tokens.reshape(-1, tokens.shape[-1])
        if tokens.dim() != 2:
            return None, None

        n_tokens = int(tokens.shape[0])
        grid_shape = self._grid_from_image_grid(grid, n_tokens)
        if grid_shape is not None:
            t, h, w = grid_shape
            if t * h * w == n_tokens:
                return tokens.reshape(-1, tokens.shape[-1]).detach().cpu(), (int(h), int(w))

        preferred_ratio = None
        if image is not None and image.width > 0:
            preferred_ratio = image.height / image.width
        factored = self._factor_grid(n_tokens, preferred_ratio=preferred_ratio)
        if factored is None:
            logger.debug(
                "Could not infer rectangular vision-token grid for %s tokens on %s",
                n_tokens,
                self._model_id,
            )
            return tokens.detach().cpu(), None
        return tokens.detach().cpu(), (int(factored[0]), int(factored[1]))

    @torch.inference_mode()
    def _extract_vision_tokens_direct(
        self,
        image: Image.Image,
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]]]:
        image_processor = getattr(self._processor, "image_processor", None)
        if self._model is None or image_processor is None:
            return None, None
        try:
            vision_inputs = image_processor(images=[image], return_tensors="pt")
        except Exception as exc:
            logger.warning(
                "Vision preprocessing failed for %s image size %s: %s",
                self._model_id,
                getattr(image, "size", None),
                exc,
            )
            return None, None

        pixel_values = vision_inputs.get("pixel_values")
        if pixel_values is None:
            logger.debug("Vision preprocessing for %s returned no pixel_values", self._model_id)
            return None, None

        vision_device, vision_dtype = self._vision_device_dtype()
        moved: Dict[str, Any] = {}
        for key, value in vision_inputs.items():
            if isinstance(value, torch.Tensor):
                if key == "pixel_values" and torch.is_floating_point(value):
                    moved[key] = value.to(device=vision_device, dtype=vision_dtype)
                else:
                    moved[key] = value.to(device=vision_device)
            else:
                moved[key] = value

        get_image_features = getattr(self._model, "get_image_features", None)
        if not callable(get_image_features):
            return None, None
        kwargs: Dict[str, Any] = {"pixel_values": moved["pixel_values"]}
        if moved.get("image_grid_thw") is not None:
            kwargs["image_grid_thw"] = moved["image_grid_thw"]
        elif moved.get("image_sizes") is not None:
            kwargs["image_sizes"] = moved["image_sizes"]

        try:
            outputs = get_image_features(**kwargs)
        except Exception as exc:
            logger.warning(
                "Vision encoder failed for %s image size %s: %s",
                self._model_id,
                getattr(image, "size", None),
                exc,
            )
            return None, None

        tokens = self._first_tensor(outputs)
        if tokens is None:
            logger.debug("Vision encoder for %s returned no tensor output", self._model_id)
            return None, None
        return self._normalise_vision_tokens(
            tokens,
            grid=moved.get("image_grid_thw"),
            image=image,
        )

    def _flatten_vision_tokens(
        self,
        image: Image.Image,
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]]]:
        if self._model is None or self._processor is None:
            return None, None

        flat_tokens, patch_grid = self._extract_vision_tokens_direct(image)
        if flat_tokens is not None and patch_grid is not None:
            return flat_tokens, patch_grid

        tokens = parent_get_vision_tokens(
            self._model,
            self._processor,
            image,
            str(self._device_obj),
        )
        if tokens is None:
            return None, None
        return self._normalise_vision_tokens(tokens, image=image)

    def _get_patch_grid_from_inputs(
        self,
        inputs: Dict[str, Any],
        image: Image.Image,
    ) -> Optional[Tuple[int, int]]:
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            merge_size = self._spatial_merge_size()
            _, h, w = [int(x) for x in grid[0].tolist()]
            return (max(1, h // merge_size), max(1, w // merge_size))

        _, patch_grid = self._flatten_vision_tokens(image)
        return patch_grid

    def _infer_patch_grid_via_processor(
        self,
        image: Image.Image,
    ) -> Optional[Tuple[int, int]]:
        """Infer patch grid without a vision forward pass when possible.

        For Qwen3-VL models, the processor emits ``image_grid_thw`` directly.
        That is cheaper and more robust than probing through
        ``get_vision_tokens`` and avoids silent failures in the auto-K probe.
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
                "Vision-only preprocessing failed for patch-grid inference on %s: %s",
                self._model_id,
                exc,
            )
            return None

        grid = vision_inputs.get("image_grid_thw")
        if grid is not None:
            try:
                merge_size = self._spatial_merge_size()
                _, h, w = [int(x) for x in grid[0].tolist()]
                return (max(1, h // merge_size), max(1, w // merge_size))
            except Exception as exc:
                logger.debug(
                    "image_grid_thw patch-grid inference failed for %s: %s",
                    self._model_id,
                    exc,
                )

        pixel_values = vision_inputs.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim >= 4:
            height = int(pixel_values.shape[-2])
            width = int(pixel_values.shape[-1])
            visual_module = self._visual_module()
            vision_config = getattr(getattr(self._model, "config", None), "vision_config", None)
            patch_size = (
                getattr(visual_module, "patch_size", None)
                or getattr(vision_config, "patch_size", None)
            )
            merge_size = self._spatial_merge_size()
            if patch_size:
                return (
                    max(1, math.ceil(height / int(patch_size)) // merge_size),
                    max(1, math.ceil(width / int(patch_size)) // merge_size),
                )

        return None

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
            layer_indices: List[int] = []
            attentions = list(getattr(outputs, "attentions", None) or [])
            extra_last = max(0, int(attention_extra_last_layers or 0))
            extra_last_start = max(0, len(attentions) - extra_last) if extra_last else len(attentions)
            for layer_index, attn in enumerate(attentions):
                if attn is None or vis_end <= vis_start:
                    continue
                is_stride_layer = layer_stride <= 1 or layer_index % layer_stride == 0
                is_extra_last_layer = extra_last > 0 and layer_index >= extra_last_start
                if not is_stride_layer and not is_extra_last_layer:
                    continue
                seq_len = attn.shape[-1]
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

    def get_vision_tokens(self, image: Image.Image) -> Optional[torch.Tensor]:
        flat_tokens, patch_grid = self._flatten_vision_tokens(image)
        if flat_tokens is None or patch_grid is None:
            return None
        h, w = patch_grid
        if h * w != flat_tokens.shape[0]:
            return flat_tokens
        return flat_tokens.reshape(1, h, w, -1)

    def get_patch_grid_shape(self, image: Image.Image) -> Tuple[int, int]:
        patch_grid = self._infer_patch_grid_via_processor(image)
        if patch_grid is not None and patch_grid[0] > 0 and patch_grid[1] > 0:
            return patch_grid

        try:
            inputs = self._build_inputs(image, "")
            patch_grid = self._get_patch_grid_from_inputs(inputs, image)
            if patch_grid is not None and patch_grid[0] > 0 and patch_grid[1] > 0:
                return patch_grid
        except Exception as exc:
            logger.debug(
                "Fallback patch-grid inference via full inputs failed for %s: %s",
                self._model_id,
                exc,
            )

        _, patch_grid = self._flatten_vision_tokens(image)
        return patch_grid or (0, 0)


QwenVLAdapter = HFVLMAdapter
