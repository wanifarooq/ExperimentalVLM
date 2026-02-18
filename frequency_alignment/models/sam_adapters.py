"""Re-export SAM adapters from the segmentation harness.

Provides thin wrappers around Sam2Adapter and Sam3Adapter from
``segmentation_robustness.segmentation_invariance_check`` so that
Experiment 6 can use them without importing the full segmentation module
directly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Path to segmentation code
_SEG_DIR = str(Path(__file__).resolve().parent.parent.parent / "segmentation_robustness")


def _import_segmentation_module():
    """Lazy import the segmentation invariance check module."""
    if _SEG_DIR not in sys.path:
        sys.path.insert(0, _SEG_DIR)
    import segmentation_invariance_check as sic
    return sic


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute mask IoU between predicted and ground truth boolean masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return inter / union if union > 0 else 0.0


class SegmentationAdapter:
    """Unified interface for SAM2 and SAM3 segmentation models."""

    def __init__(self, model_type: str = "sam3"):
        self.model_type = model_type
        self._adapter = None
        self._sic = None

    def load(
        self,
        model_type: str = "sam3",
        device: str = "cuda:0",
        **kwargs,
    ) -> None:
        """Load SAM2 or SAM3 adapter from segmentation harness."""
        self._sic = _import_segmentation_module()
        self.model_type = model_type

        if model_type == "sam2":
            config_name = kwargs.get("config_name", "sam2_hiera_large.yaml")
            checkpoint = kwargs.get("checkpoint", "")
            self._adapter = self._sic.Sam2Adapter(config_name, checkpoint, device)
        elif model_type == "sam3":
            module_path = kwargs.get("module_path", "sam3.model")
            class_name = kwargs.get("class_name", "Sam3Model")
            checkpoint = kwargs.get("checkpoint", "")
            self._adapter = self._sic.Sam3Adapter(module_path, class_name, checkpoint, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info("Loaded %s adapter", model_type)

    def predict_with_text(
        self,
        image: Image.Image,
        text_prompt: str,
    ) -> List[Tuple[np.ndarray, float]]:
        """Predict masks using text prompt (SAM3 only).

        Returns list of (mask, score) tuples.
        """
        if self.model_type != "sam3" or self._adapter is None:
            raise RuntimeError("Text prompting requires SAM3 adapter")

        predictions = self._adapter.predict(image, text_prompt)
        return [(p.mask, p.score) for p in predictions]

    def predict_with_box(
        self,
        image: Image.Image,
        boxes: List[Tuple[float, float, float, float]],
    ) -> List[Tuple[np.ndarray, float]]:
        """Predict masks using box prompts (SAM2).

        Returns list of (mask, score) tuples.
        """
        if self._adapter is None:
            raise RuntimeError("Adapter not loaded")

        if self.model_type == "sam2":
            predictions = self._adapter.predict(image, boxes=boxes, points=[])
        elif self.model_type == "sam3" and self._adapter.supports_visual_prompting():
            predictions = self._adapter.predict_visual(image, boxes=boxes, points=[])
        else:
            raise RuntimeError(f"{self.model_type} does not support box prompting")

        return [(p.mask, p.score) for p in predictions]

    def unload(self) -> None:
        """Release model."""
        self._adapter = None
