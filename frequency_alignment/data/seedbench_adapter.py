"""Thin adapter to reuse existing SEEDBench loader from the parent module.

SEEDBench MCQ questions are treated as L4 (Very Fine) since they require
compositional visual reasoning.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

from .base import GranularityLevel, GranularitySample, LevelData

logger = logging.getLogger(__name__)

_PARENT_DIR = str(Path(__file__).resolve().parent.parent.parent)


def _import_seedbench_loader():
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    from vlm_invariance_check import load_seedbench_samples, Sample
    return load_seedbench_samples, Sample


def load_seedbench_as_granularity(
    tsv_path: Optional[Path] = None,
    image_root: Optional[Path] = None,
    max_samples: int = 1000,
    cache_dir: Optional[Path] = None,
) -> List[GranularitySample]:
    """Load SEEDBench samples as L4 GranularitySamples.

    Args:
        tsv_path: Path to SEEDBench_IMG.tsv.
        image_root: Path to image directory.
        max_samples: Max samples to load.
        cache_dir: HF cache directory for auto-download.

    Returns:
        List of :class:`GranularitySample` (L4 level only).
    """
    load_fn, _ = _import_seedbench_loader()

    # Try standard locations
    if tsv_path is None:
        candidates = []
        if cache_dir is not None:
            candidates.append(cache_dir / "seedbench" / "SEEDBench_IMG.tsv")
        candidates.append(Path(".hf_cache/seedbench/SEEDBench_IMG.tsv"))
        for c in candidates:
            if c.exists():
                tsv_path = c
                break
    if image_root is None and tsv_path is not None:
        image_root = tsv_path.parent / "images"

    if tsv_path is None or not tsv_path.exists():
        logger.warning("SEEDBench TSV not found; returning empty dataset.")
        return []

    raw_samples = load_fn(str(tsv_path), str(image_root), max_rows=max_samples)

    results: List[GranularitySample] = []
    for s in raw_samples:
        if not s.image_path or not Path(s.image_path).exists():
            continue
        level_data = LevelData(
            level=GranularityLevel.L4_VERY_FINE,
            question=s.question,
            options=s.options,
            answer_label=s.answer_label,
            question_type="seedbench_mcq",
        )
        gs = GranularitySample(
            image_id=str(s.index),
            image_path=Path(s.image_path),
            levels={GranularityLevel.L4_VERY_FINE: level_data},
            dataset="seedbench",
            split=s.split or "val",
        )
        results.append(gs)

    logger.info("Loaded %d SEEDBench samples as L4 granularity", len(results))
    return results
