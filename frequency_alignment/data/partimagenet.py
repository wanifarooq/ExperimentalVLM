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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .base import GranularityLevel, GranularitySample, LevelData

logger = logging.getLogger(__name__)


def _find_partimagenet_root(
    cache_dir: Path,
    root_override: Optional[Path] = None,
) -> Optional[Path]:
    """Search for PartImageNet data in common locations."""
    candidates = []
    if root_override is not None:
        candidates.append(root_override)
    candidates.extend([
        cache_dir / "partimagenet",
        cache_dir / "PartImageNet",
        Path.home() / "datasets" / "PartImageNet",
        Path("/data") / "PartImageNet",
    ])
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


def _build_text_prompt(category_name: str) -> str:
    parts = category_name.replace("_", " ").split()
    if not parts:
        return "segment the object"
    if len(parts) == 1:
        return f"segment the {parts[0]}"
    if len(parts) == 2:
        return f"segment the {parts[1]} of the {parts[0]}"
    subject = parts[0]
    rel_chain = list(reversed(parts[1:]))
    phrase = f" of the ".join(rel_chain)
    return f"segment the {phrase} of the {subject}"


def bbox_xywh_to_xyxy(bbox: List[float]) -> Optional[List[float]]:
    if len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def segmentation_to_mask(
    segmentation,
    image_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    width, height = image_size
    if not segmentation:
        return None

    if isinstance(segmentation, list):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(points, outline=1, fill=1)
        return np.array(mask, dtype=bool)

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            return None

        rle = segmentation
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=-1)
        return decoded.astype(bool)

    return None


def mask_to_box(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def build_partimagenet_dataset(
    cache_dir: Path,
    max_samples: int = 300,
    seed: int = 42,
    split: str = "val",
    allow_download: bool = True,
    root_dir: Optional[Path] = None,
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

    root = _find_partimagenet_root(cache_dir, root_override=root_dir)
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

            prompt = _build_text_prompt(anno["category"])

            levels[level] = LevelData(
                level=level,
                question=prompt,
                options={},
                text_prompt=prompt,
                bbox=bbox_xywh_to_xyxy(anno["bbox"]),
                segmentation=anno["segmentation"],
                question_type="segmentation",
            )

        # Need a complete hierarchy for meaningful comparison
        required_levels = {
            GranularityLevel.L1_COARSE,
            GranularityLevel.L2_MEDIUM,
            GranularityLevel.L3_FINE,
        }
        if required_levels.issubset(levels):
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
