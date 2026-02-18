"""Abstract model adapter with hook-based internal extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image


@dataclass
class HookOutputs:
    """Container for extracted model internals.

    Attributes:
        cross_attention_weights: Per-layer attention from language tokens to
            vision tokens.  Each tensor has shape
            ``(num_heads, num_lang_tokens, num_vis_tokens)``.
        pre_fusion_features: Vision encoder output *before* it enters the
            language model (i.e. before cross-modal attention).
            Shape ``(num_vis_tokens, hidden_dim)``.
        post_fusion_features: Hidden states *after* the first decoder layers
            where vision and text interact.
            Shape ``(seq_len, hidden_dim)``.
        vision_tokens: Raw vision encoder output (same as pre_fusion but
            possibly un-projected).
        patch_grid: ``(H_patches, W_patches)`` so attention can be reshaped
            to a spatial map.
        vision_token_range: ``(start, end)`` indices of vision tokens in the
            full input sequence.
    """

    cross_attention_weights: Optional[List[torch.Tensor]] = None
    pre_fusion_features: Optional[torch.Tensor] = None
    post_fusion_features: Optional[torch.Tensor] = None
    vision_tokens: Optional[torch.Tensor] = None
    patch_grid: Optional[Tuple[int, int]] = None
    vision_token_range: Optional[Tuple[int, int]] = None


class VLMAdapter(ABC):
    """Abstract VLM adapter with hook-based internal extraction.

    Every concrete adapter must implement model loading, MCQ scoring,
    and internal extraction.  The adapters are designed to share a common
    interface so experiments can switch models via config.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
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
        """Load model and processor onto *device*."""

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @abstractmethod
    def score_options(
        self,
        image: Image.Image,
        question: str,
        options: Dict[str, str],
    ) -> Dict[str, float]:
        """Score MCQ options via log-likelihood.

        Returns:
            ``{label: log_p}`` for each option.
        """

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        """Open-ended generation."""

    # ------------------------------------------------------------------
    # Internal extraction (core novelty)
    # ------------------------------------------------------------------

    @abstractmethod
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
        """Run forward pass with hooks to extract internal representations.

        This is the key new capability for experiments 2, 3, and 5.

        Args:
            image: Input PIL image.
            prompt: Text prompt.
            extract_attention: Capture cross-attention weights.
            extract_pre_fusion: Capture vision features before fusion.
            extract_post_fusion: Capture features after fusion.
            layer_stride: Extract attention from every Nth layer (saves memory).

        Returns:
            :class:`HookOutputs` with the requested tensors.
        """

    @abstractmethod
    def get_vision_tokens(
        self, image: Image.Image
    ) -> Optional[torch.Tensor]:
        """Extract vision encoder output tokens.

        Returns:
            Tensor of shape ``(T, H_patches, W_patches, D)`` or ``None``.
        """

    @abstractmethod
    def get_patch_grid_shape(self, image: Image.Image) -> Tuple[int, int]:
        """Return ``(H_patches, W_patches)`` for the given image."""

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Currently loaded model identifier."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device the model lives on."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of decoder layers."""
