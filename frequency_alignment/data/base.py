"""Core data structures shared across all experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional


class GranularityLevel(IntEnum):
    """Task granularity levels, from coarse to fine."""

    L1_COARSE = 1       # Object presence / whole-object segmentation
    L2_MEDIUM = 2       # Attribute recognition / part-level segmentation
    L3_FINE = 3          # Spatial relationship / subpart segmentation
    L4_VERY_FINE = 4     # Compositional MCQ reasoning


@dataclass
class LevelData:
    """Task data for one granularity level on one image."""

    level: GranularityLevel
    question: str
    options: Dict[str, str]            # MCQ options, e.g. {"A": "yes", "B": "no"}
    answer_label: Optional[str] = None  # ground truth label key
    # For segmentation experiments:
    mask_path: Optional[Path] = None
    text_prompt: Optional[str] = None
    # Metadata:
    question_type: Optional[str] = None  # e.g. "object_presence", "attribute", etc.


@dataclass
class GranularitySample:
    """One image with tasks at multiple granularity levels.

    This is the primary sample type for the frequency alignment experiments.
    Each sample contains the same image but with questions/prompts ranging
    from coarse (L1) to fine (L4) granularity.
    """

    image_id: str
    image_path: Path
    levels: Dict[GranularityLevel, LevelData] = field(default_factory=dict)
    # Optional metadata from source dataset:
    dataset: str = ""
    split: str = "val"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def has_all_levels(self, required: List[GranularityLevel] | None = None) -> bool:
        """Check if sample has data for all required granularity levels."""
        if required is None:
            required = list(GranularityLevel)
        return all(lvl in self.levels for lvl in required)


@dataclass
class ExperimentResult:
    """Container for one experiment's outputs."""

    experiment_id: int
    experiment_name: str
    config: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    per_sample: List[Dict[str, Any]] = field(default_factory=list)
    hypothesis_tests: Dict[str, Any] = field(default_factory=dict)
