"""Core data structures shared across all experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GranularityLevel(IntEnum):
    """Task granularity levels, from coarse to fine."""

    L1_COARSE = 1       # Object presence / whole-object segmentation
    L2_MEDIUM = 2       # Attribute recognition / part-level segmentation
    L3_FINE = 3          # Spatial relationship / subpart segmentation
    L4_VERY_FINE = 4     # Compositional MCQ reasoning
    L5_WORDY_SIMPLETON = 5  # L1 semantics with inflated prompt load control
    L6_WORDY_MEDIUM = 6  # L2 semantics with inflated prompt load control
    L7_WORDY_FINE = 7  # L3 semantics with inflated prompt load control
    L8_WORDY_VERY_FINE = 8  # L4 semantics with inflated prompt load control


PRIMARY_VQA_LEVELS: List[GranularityLevel] = [
    GranularityLevel.L1_COARSE,
    GranularityLevel.L2_MEDIUM,
    GranularityLevel.L3_FINE,
    GranularityLevel.L4_VERY_FINE,
]
WORDY_CONTROL_VQA_LEVELS: List[GranularityLevel] = [
    GranularityLevel.L5_WORDY_SIMPLETON,
    GranularityLevel.L6_WORDY_MEDIUM,
    GranularityLevel.L7_WORDY_FINE,
    GranularityLevel.L8_WORDY_VERY_FINE,
]
ALL_VQA_LEVELS: List[GranularityLevel] = PRIMARY_VQA_LEVELS + WORDY_CONTROL_VQA_LEVELS
PRIMARY_VQA_LEVEL_NAMES: List[str] = [level.name for level in PRIMARY_VQA_LEVELS]
ALL_VQA_LEVEL_NAMES: List[str] = [level.name for level in ALL_VQA_LEVELS]
WORDY_CONTROL_LEVEL_NAME_PAIRS: List[Tuple[str, str]] = [
    (GranularityLevel.L1_COARSE.name, GranularityLevel.L5_WORDY_SIMPLETON.name),
    (GranularityLevel.L2_MEDIUM.name, GranularityLevel.L6_WORDY_MEDIUM.name),
    (GranularityLevel.L3_FINE.name, GranularityLevel.L7_WORDY_FINE.name),
    (GranularityLevel.L4_VERY_FINE.name, GranularityLevel.L8_WORDY_VERY_FINE.name),
]
VERIFICATION_VQA_LEVELS: List[GranularityLevel] = [
    GranularityLevel.L1_COARSE,
    GranularityLevel.L3_FINE,
    GranularityLevel.L5_WORDY_SIMPLETON,
    GranularityLevel.L7_WORDY_FINE,
]


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
    bbox: Optional[List[float]] = None
    segmentation: Any = None
    # Metadata:
    question_type: Optional[str] = None  # e.g. "object_presence", "attribute", etc.
    semantic_atoms: List[str] = field(default_factory=list)
    prompt_semantic_atoms: List[str] = field(default_factory=list)
    semantic_atom_counts: Dict[str, Any] = field(default_factory=dict)
    question_complexity_score: float = 0.0
    prompt_complexity_score: float = 0.0
    complexity_score: float = 0.0
    option_hardness_score: float = 0.0
    option_hardness_components: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GranularitySample:
    """One image with tasks at multiple granularity levels.

    This is the primary sample type for the frequency alignment experiments.
    Each sample contains the same image but with questions/prompts ranging
    from coarse (L1) to fine (L4) granularity, plus optional control levels.
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

    def reference_overlay_context(self) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        """Return the finest available MCQ context for legacy answer-conditioned overlays.

        The perturbation suite is built once per image and reused across levels,
        so the legacy answer-conditioned overlay mode still needs one reference
        MCQ context. The default overlay mode is label-free, so this fallback
        is only used when that legacy mode is explicitly enabled.
        """
        preferred_levels = list(reversed(PRIMARY_VQA_LEVELS))
        fallback_levels = [
            level for level in sorted(self.levels.keys(), key=int, reverse=True)
            if level not in preferred_levels
        ]
        for level in preferred_levels + fallback_levels:
            level_data = self.levels.get(level)
            if level_data and level_data.options:
                return level_data.options, level_data.answer_label
        return None, None


@dataclass
class ExperimentResult:
    """Container for one experiment's outputs."""

    experiment_id: int
    experiment_name: str
    config: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    per_sample: List[Dict[str, Any]] = field(default_factory=list)
    hypothesis_tests: Dict[str, Any] = field(default_factory=dict)
