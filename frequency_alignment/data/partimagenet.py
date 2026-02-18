"""PartImageNet loader with 3-level segmentation manifests.

Builds segmentation samples at three granularity levels:
- L1 (Coarse): Whole object mask + "segment the {object}"
- L2 (Medium): Part-level mask + "segment the {part} of the {object}"
- L3 (Fine): Subpart-level mask + "segment the {subpart} of the {part} of the {object}"

PartImageNet provides hierarchical part annotations for ~24K images
across 158 object categories from ImageNet.

Download: https://github.com/TACJu/PartImageNet
The dataset requires manual download and acceptance of ImageNet terms.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

from .base import GranularityLevel, GranularitySample, LevelData

logger = logging.getLogger(__name__)


def _find_partimagenet_root(cache_dir: Path) -> Optional[Path]:
    """Search for PartImageNet data in common locations."""
    candidates = [
        cache_dir / "partimagenet",
        cache_dir / "PartImageNet",
        Path.home() / "datasets" / "PartImageNet",
        Path("/data") / "PartImageNet",
    ]
    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c
    return None


def _parse_partimagenet_annotations(
    anno_path: Path,
) -> Dict[str, dict]:
    """Parse PartImageNet JSON annotations.

    Expected format (COCO-style):
    {
        "images": [{"id": int, "file_name": str, ...}],
        "annotations": [{"id": int, "image_id": int, "category_id": int,
                         "segmentation": [...], "area": float, ...}],
        "categories": [{"id": int, "name": str, "supercategory": str}]
    }

    Returns:
        {image_id: {"file_name": str, "annotations": [...]}}
    """
    with open(anno_path) as f:
        data = json.load(f)

    # Build category lookup
    cat_map = {}
    for cat in data.get("categories", []):
        cat_map[cat["id"]] = {
            "name": cat["name"],
            "supercategory": cat.get("supercategory", ""),
        }

    # Group annotations by image
    images_by_id = {}
    for img in data.get("images", []):
        images_by_id[img["id"]] = {
            "file_name": img["file_name"],
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "annotations": [],
        }

    for anno in data.get("annotations", []):
        img_id = anno["image_id"]
        if img_id in images_by_id:
            cat = cat_map.get(anno["category_id"], {})
            images_by_id[img_id]["annotations"].append({
                "category": cat.get("name", "unknown"),
                "supercategory": cat.get("supercategory", ""),
                "area": anno.get("area", 0),
                "bbox": anno.get("bbox", []),
                "segmentation": anno.get("segmentation", []),
            })

    return images_by_id


def _classify_part_level(
    category_name: str,
    supercategory: str,
) -> Optional[GranularityLevel]:
    """Classify an annotation into a granularity level based on naming.

    PartImageNet uses hierarchical names like:
    - "dog" (whole object) -> L1
    - "dog_head" (part) -> L2
    - "dog_head_ear" (subpart) -> L3
    """
    parts = category_name.split("_")
    if len(parts) == 1:
        return GranularityLevel.L1_COARSE
    elif len(parts) == 2:
        return GranularityLevel.L2_MEDIUM
    elif len(parts) >= 3:
        return GranularityLevel.L3_FINE
    return None


def build_partimagenet_dataset(
    cache_dir: Path,
    max_samples: int = 300,
    seed: int = 42,
    split: str = "val",
    allow_download: bool = True,
) -> List[GranularitySample]:
    """Build 3-level segmentation dataset from PartImageNet.

    Args:
        cache_dir: Root cache directory.
        max_samples: Maximum samples to return.
        seed: Random seed.
        split: Dataset split ("train" or "val").
        allow_download: Whether to download if not found.

    Returns:
        List of GranularitySample with L1-L3 segmentation prompts.
    """
    rng = random.Random(seed)

    root = _find_partimagenet_root(cache_dir)
    if root is None:
        logger.warning(
            "PartImageNet not found in %s. "
            "Download from https://github.com/TACJu/PartImageNet "
            "and place in %s/partimagenet/",
            cache_dir, cache_dir,
        )
        return []

    # Find annotation file
    anno_candidates = [
        root / f"{split}.json",
        root / "annotations" / f"{split}.json",
        root / f"partimagenet_{split}.json",
    ]
    anno_path = None
    for c in anno_candidates:
        if c.exists():
            anno_path = c
            break

    if anno_path is None:
        logger.warning("No PartImageNet annotations found at %s", root)
        return []

    logger.info("Loading PartImageNet annotations from %s", anno_path)
    images_data = _parse_partimagenet_annotations(anno_path)

    # Find image directory
    img_dir_candidates = [
        root / split,
        root / "images" / split,
        root / f"{split}_images",
    ]
    img_dir = None
    for c in img_dir_candidates:
        if c.exists():
            img_dir = c
            break

    if img_dir is None:
        logger.warning("Image directory not found under %s", root)
        return []

    # Build samples
    samples: List[GranularitySample] = []
    image_ids = list(images_data.keys())
    rng.shuffle(image_ids)

    for img_id in image_ids:
        if len(samples) >= max_samples:
            break

        img_data = images_data[img_id]
        file_name = img_data["file_name"]
        img_path = img_dir / file_name
        if not img_path.exists():
            continue

        # Classify annotations by level
        levels: Dict[GranularityLevel, LevelData] = {}
        for anno in img_data["annotations"]:
            level = _classify_part_level(anno["category"], anno["supercategory"])
            if level is None or level in levels:
                continue

            cat_name = anno["category"].replace("_", " ")
            if level == GranularityLevel.L1_COARSE:
                prompt = f"segment the {cat_name}"
            elif level == GranularityLevel.L2_MEDIUM:
                prompt = f"segment the {cat_name}"
            else:
                prompt = f"segment the {cat_name}"

            levels[level] = LevelData(
                level=level,
                question=prompt,
                options={},
                text_prompt=prompt,
                question_type="segmentation",
            )

        # Need at least 2 levels for meaningful comparison
        if len(levels) >= 2:
            samples.append(GranularitySample(
                image_id=str(img_id),
                image_path=img_path,
                levels=levels,
                dataset="partimagenet",
                split=split,
                metadata={
                    "num_annotations": len(img_data["annotations"]),
                    "num_levels": len(levels),
                },
            ))

    logger.info(
        "Built %d PartImageNet samples from %d images",
        len(samples), len(images_data),
    )
    return samples
