"""GQA dataset loader with 5-level granularity question generation.

Downloads GQA scene graphs and images, then generates questions at four
granularity levels from the same image:

- **L1 (Coarse)**: Object presence -- "Is there a {object} in this image?"
- **L2 (Medium)**: Attribute recognition -- "What {attr_type} is the {object}?"
- **L3 (Fine)**: Spatial relationship -- "Is the {obj1} to the {relation} of {obj2}?"
- **L4 (Very Fine)**: Compositional MCQ -- combines attributes + spatial reasoning
- **L5 (Wordy-Simpleton)**: L1 semantics with redundant filler text

GQA scene graph structure (per image)::

    {
        "imageId": {
            "width": int,
            "height": int,
            "objects": {
                "objId": {
                    "name": str,
                    "x": int, "y": int, "w": int, "h": int,
                    "attributes": [str, ...],
                    "relations": [
                        {"name": str, "object": str (target objId)},
                        ...
                    ]
                },
                ...
            }
        }
    }

Download URLs:
- Scene graphs: https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip (42.7 MB)
- Images: https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip (20.3 GB for images only)
"""

from __future__ import annotations

import json
import logging
import random
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from .base import (
    ALL_VQA_LEVELS,
    GranularityLevel,
    GranularitySample,
    LevelData,
    VERIFICATION_VQA_LEVELS,
)
from .complexity import build_semantic_complexity

logger = logging.getLogger(__name__)

_WORDY_FILLER_PREFIX = (
    "Please consider the visual scene carefully, without overthinking any hidden trick, "
    "and answer the following in a straightforward way after taking a moment to reflect on "
    "what is plainly visible. "
)
_WORDY_FILLER_SUFFIX = (
    "In other words, even though this may sound longer than necessary, the question is still "
    "asking only about the obvious visible presence of the object, so respond accordingly."
)

# ---------------------------------------------------------------------------
# Download URLs
# ---------------------------------------------------------------------------
_SCENE_GRAPHS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip"
_IMAGES_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"

# Common attribute categories for L2 questions
_ATTRIBUTE_CATEGORIES = {
    "color": {
        "white", "black", "red", "blue", "green", "yellow", "brown",
        "gray", "grey", "orange", "pink", "purple", "beige", "tan",
        "silver", "gold",
    },
    "material": {
        "wooden", "metal", "plastic", "glass", "leather", "fabric",
        "concrete", "brick", "stone", "ceramic", "rubber",
    },
    "shape": {
        "round", "square", "rectangular", "circular", "oval",
        "triangular", "flat", "curved",
    },
    "size": {"large", "small", "big", "little", "tall", "short", "long", "thin"},
    "texture": {
        "striped", "spotted", "plaid", "checkered", "smooth",
        "rough", "shiny", "matte",
    },
}

# Flatten for quick lookup: attribute_word -> category
_ATTR_TO_CATEGORY: Dict[str, str] = {}
for cat, words in _ATTRIBUTE_CATEGORIES.items():
    for w in words:
        _ATTR_TO_CATEGORY[w.lower()] = cat


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress logging."""
    if dest.exists():
        logger.info("Already downloaded: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", desc or url, dest)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (10 * 1024 * 1024) < 8192:
                logger.info("  %.1f / %.1f MB", downloaded / 1e6, total / 1e6)
    logger.info("Download complete: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def download_scene_graphs(cache_dir: Path) -> Path:
    """Download and extract GQA scene graphs.

    Returns:
        Directory containing ``val_sceneGraphs.json``.
    """
    sg_dir = cache_dir / "gqa" / "sceneGraphs"
    val_json = sg_dir / "val_sceneGraphs.json"
    if val_json.exists():
        return sg_dir

    zip_path = cache_dir / "gqa" / "sceneGraphs.zip"
    _download_file(_SCENE_GRAPHS_URL, zip_path, "GQA scene graphs")

    logger.info("Extracting scene graphs...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(sg_dir)
    # The zip may extract into a subdirectory; find the JSON
    for candidate in [sg_dir / "val_sceneGraphs.json",
                      sg_dir / "sceneGraphs" / "val_sceneGraphs.json"]:
        if candidate.exists():
            if candidate.parent != sg_dir:
                # Move files up one level
                for f in candidate.parent.iterdir():
                    f.rename(sg_dir / f.name)
            break
    assert val_json.exists() or (sg_dir / "val_sceneGraphs.json").exists(), \
        f"val_sceneGraphs.json not found in {sg_dir}"
    return sg_dir


def download_gqa_images(cache_dir: Path, allow_download: bool = True) -> Path:
    """Get GQA images directory.

    GQA uses Visual Genome images.  The full download is ~20 GB.
    If images are not available, returns the expected path (individual
    images will be downloaded on demand via :func:`download_single_image`).
    """
    img_dir = cache_dir / "gqa" / "images"
    if img_dir.exists() and any(img_dir.iterdir()):
        return img_dir

    img_dir.mkdir(parents=True, exist_ok=True)

    if allow_download:
        zip_path = cache_dir / "gqa" / "images.zip"
        if not zip_path.exists():
            logger.info(
                "GQA images not found locally. Individual images will be "
                "downloaded on demand from Visual Genome servers."
            )
            return img_dir
        logger.info("Extracting GQA images...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(img_dir.parent)
    return img_dir


# Visual Genome hosts GQA images in two directories
_VG_IMAGE_URLS = [
    "https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{image_id}.jpg",
]


def download_single_image(image_id: str, img_dir: Path) -> bool:
    """Download a single GQA image on demand from Visual Genome.

    Returns True if the image was successfully downloaded or already exists.
    """
    img_path = img_dir / f"{image_id}.jpg"
    if img_path.exists():
        return True

    for url_template in _VG_IMAGE_URLS:
        url = url_template.format(image_id=image_id)
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 1000:
                img_path.parent.mkdir(parents=True, exist_ok=True)
                with open(img_path, "wb") as f:
                    f.write(resp.content)
                return True
        except (requests.RequestException, OSError):
            continue
    return False


# ---------------------------------------------------------------------------
# Scene graph loading
# ---------------------------------------------------------------------------


def load_scene_graphs(
    sg_dir: Path, split: str = "val"
) -> Dict[str, dict]:
    """Load scene graphs JSON for the given split.

    Returns:
        ``{image_id: scene_graph_dict}``.
    """
    path = sg_dir / f"{split}_sceneGraphs.json"
    logger.info("Loading scene graphs from %s ...", path)
    with open(path, "r") as f:
        data = json.load(f)
    logger.info("Loaded %d scene graphs", len(data))
    return data


# ---------------------------------------------------------------------------
# Question generation per granularity level
# ---------------------------------------------------------------------------


def _classify_attributes(attributes: List[str]) -> Dict[str, str]:
    """Classify a list of free-form attribute strings into categories.

    Returns:
        ``{category: attribute_value}`` for recognized attributes.
    """
    result: Dict[str, str] = {}
    for attr in attributes:
        attr_lower = attr.lower().strip()
        cat = _ATTR_TO_CATEGORY.get(attr_lower)
        if cat and cat not in result:
            result[cat] = attr_lower
    return result


def _make_distractors(
    correct: str,
    category: str,
    n: int = 3,
    rng: random.Random | None = None,
) -> List[str]:
    """Generate distractor options for MCQ from the same attribute category."""
    rng = rng or random.Random()
    pool = list(_ATTRIBUTE_CATEGORIES.get(category, set()))
    pool = [p for p in pool if p.lower() != correct.lower()]
    rng.shuffle(pool)
    return pool[:n]


def _count_named_objects(objects: Dict[str, dict], object_name: str) -> int:
    object_name = str(object_name or "").strip().lower()
    if not object_name:
        return 0
    return sum(
        1
        for obj in objects.values()
        if str(obj.get("name", "")).strip().lower() == object_name
    )


def _count_named_pairs(
    objects: Dict[str, dict],
    subject_name: str,
    target_name: str,
) -> int:
    subject_name = str(subject_name or "").strip().lower()
    target_name = str(target_name or "").strip().lower()
    if not subject_name or not target_name:
        return 0
    subject_ids = [
        object_id
        for object_id, obj in objects.items()
        if str(obj.get("name", "")).strip().lower() == subject_name
    ]
    target_ids = [
        object_id
        for object_id, obj in objects.items()
        if str(obj.get("name", "")).strip().lower() == target_name
    ]
    return sum(1 for sid in subject_ids for tid in target_ids if sid != tid)


def _scene_attribute_inventory(objects: Dict[str, dict]) -> Set[str]:
    inventory: Set[str] = set()
    for obj in objects.values():
        for attr in obj.get("attributes", []) or []:
            attr_name = str(attr or "").strip().lower()
            if attr_name:
                inventory.add(attr_name)
    return inventory


def _compute_option_hardness(
    *,
    options: Dict[str, str],
    answer_label: Optional[str],
    attribute_category: Optional[str] = None,
    scene_attributes: Optional[Set[str]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Approximate MCQ discrimination difficulty.

    The primary term is the number of distractors drawn from the same semantic
    category as the correct answer. A secondary scene-plausibility term counts
    distractors that also occur as attributes somewhere in the scene.
    """

    if not options or answer_label not in options:
        return 0.0, {
            "num_distractors": 0,
            "same_category_distractors": 0,
            "scene_present_distractors": 0,
        }

    distractors = [
        str(value or "").strip().lower()
        for label, value in options.items()
        if label != answer_label
    ]
    same_category_distractors = 0
    if attribute_category:
        same_category_distractors = sum(
            1 for distractor in distractors
            if _ATTR_TO_CATEGORY.get(distractor) == attribute_category
        )
    scene_attributes = scene_attributes or set()
    scene_present_distractors = sum(
        1 for distractor in distractors if distractor in scene_attributes
    )
    hardness = float(same_category_distractors + 0.5 * scene_present_distractors)
    return hardness, {
        "num_distractors": len(distractors),
        "same_category_distractors": int(same_category_distractors),
        "scene_present_distractors": int(scene_present_distractors),
    }


def _build_object_presence_question(
    sg: dict,
    obj_id: str,
    obj: dict,
    rng: random.Random,
    all_object_names: Set[str],
    *,
    level: GranularityLevel,
    question_type: str,
    desired_answer_label: Optional[str] = None,
    wordy: bool = False,
) -> Optional[LevelData]:
    """Object-presence verification with optional prompt-load inflation."""
    name = obj.get("name", "").strip()
    if not name:
        return None

    if desired_answer_label not in {None, "A", "B"}:
        return None
    positive = desired_answer_label == "A" if desired_answer_label is not None else (rng.random() < 0.5)

    if positive:
        base_question = f"Is there a {name} in this image?"
        answer = "A"
        options = {"A": "yes", "B": "no"}
        grounding_candidates = _count_named_objects(sg.get("objects", {}), name)
    else:
        absent = list(all_object_names - {o.get("name", "") for o in sg.get("objects", {}).values()})
        if not absent:
            return None
        absent_name = rng.choice(absent)
        base_question = f"Is there a {absent_name} in this image?"
        answer = "B"
        options = {"A": "yes", "B": "no"}
        name = absent_name
        grounding_candidates = 0
    if wordy:
        question = f"{_WORDY_FILLER_PREFIX}{base_question} {_WORDY_FILLER_SUFFIX}"
    else:
        question = base_question
    complexity = build_semantic_complexity(
        entity_names=[name],
        reasoning_ops=["exist"],
        program_depth=1,
        grounding_candidate_count=grounding_candidates,
        question_text=question,
        options=options,
    )
    option_hardness_score, option_hardness_components = _compute_option_hardness(
        options=options,
        answer_label=answer,
    )

    return LevelData(
        level=level,
        question=question,
        options=options,
        answer_label=answer,
        question_type=question_type,
        option_hardness_score=option_hardness_score,
        option_hardness_components=option_hardness_components,
        **complexity,
    )


def build_l1_question(
    sg: dict,
    obj_id: str,
    obj: dict,
    rng: random.Random,
    all_object_names: Set[str],
    desired_answer_label: Optional[str] = None,
) -> Optional[LevelData]:
    """L1 (Coarse): Object presence Y/N."""

    return _build_object_presence_question(
        sg,
        obj_id,
        obj,
        rng,
        all_object_names,
        level=GranularityLevel.L1_COARSE,
        question_type="object_presence",
        desired_answer_label=desired_answer_label,
        wordy=False,
    )


def build_l5_question(
    sg: dict,
    obj_id: str,
    obj: dict,
    rng: random.Random,
    all_object_names: Set[str],
    desired_answer_label: Optional[str] = None,
) -> Optional[LevelData]:
    """L5 (Wordy-Simpleton): L1 semantics with redundant filler text."""

    return _build_object_presence_question(
        sg,
        obj_id,
        obj,
        rng,
        all_object_names,
        level=GranularityLevel.L5_WORDY_SIMPLETON,
        question_type="object_presence_wordy_control",
        desired_answer_label=desired_answer_label,
        wordy=True,
    )


def build_l2_question(
    sg: dict,
    obj_id: str,
    obj: dict,
    rng: random.Random,
) -> Optional[LevelData]:
    """L2 (Medium): Attribute recognition MCQ."""
    name = obj.get("name", "").strip()
    attributes = obj.get("attributes", [])
    if not name or not attributes:
        return None

    classified = _classify_attributes(attributes)
    if not classified:
        return None

    # Pick a random attribute category
    cat = rng.choice(list(classified.keys()))
    correct_val = classified[cat]
    distractors = _make_distractors(correct_val, cat, n=3, rng=rng)
    if len(distractors) < 2:
        return None

    all_opts = [correct_val] + distractors[:3]
    rng.shuffle(all_opts)
    labels = ["A", "B", "C", "D"]
    options = {labels[i]: opt for i, opt in enumerate(all_opts)}
    answer = next(l for l, v in options.items() if v == correct_val)

    question = f"What {cat} is the {name}?"
    complexity = build_semantic_complexity(
        entity_names=[name],
        attribute_queries=[cat],
        reasoning_ops=["query_attribute"],
        program_depth=2,
        grounding_candidate_count=_count_named_objects(sg.get("objects", {}), name),
        question_text=question,
        options=options,
    )
    option_hardness_score, option_hardness_components = _compute_option_hardness(
        options=options,
        answer_label=answer,
        attribute_category=cat,
        scene_attributes=_scene_attribute_inventory(sg.get("objects", {})),
    )
    return LevelData(
        level=GranularityLevel.L2_MEDIUM,
        question=question,
        options=options,
        answer_label=answer,
        question_type=f"attribute_{cat}",
        option_hardness_score=option_hardness_score,
        option_hardness_components=option_hardness_components,
        **complexity,
    )


def build_l3_question(
    sg: dict,
    obj_id: str,
    obj: dict,
    objects: dict,
    rng: random.Random,
    desired_answer_label: Optional[str] = None,
) -> Optional[LevelData]:
    """L3 (Fine): Spatial relationship Y/N."""
    name = obj.get("name", "").strip()
    relations = obj.get("relations", [])
    if not name or not relations:
        return None

    # Pick a random relation
    rel = rng.choice(relations)
    rel_name = rel.get("name", "").strip()
    target_id = rel.get("object", "")
    target_obj = objects.get(target_id)
    if not rel_name or target_obj is None:
        return None
    target_name = target_obj.get("name", "").strip()
    if not target_name:
        return None

    if desired_answer_label not in {None, "A", "B"}:
        return None
    positive = desired_answer_label == "A" if desired_answer_label is not None else (rng.random() < 0.5)

    if positive:
        question = f"Is the {name} {rel_name} the {target_name}?"
        answer = "A"
    else:
        other_rels = ["to the left of", "to the right of", "above", "below",
                      "behind", "in front of", "on top of", "next to"]
        other_rels = [r for r in other_rels if r != rel_name]
        if not other_rels:
            return None
        wrong_rel = rng.choice(other_rels)
        question = f"Is the {name} {wrong_rel} the {target_name}?"
        answer = "B"
        rel_name = wrong_rel

    options = {"A": "yes", "B": "no"}
    complexity = build_semantic_complexity(
        entity_names=[name, target_name],
        relation_labels=[rel_name],
        reasoning_ops=["verify_relation"],
        program_depth=2,
        grounding_candidate_count=_count_named_pairs(objects, name, target_name),
        question_text=question,
        options=options,
    )
    option_hardness_score, option_hardness_components = _compute_option_hardness(
        options=options,
        answer_label=answer,
    )
    return LevelData(
        level=GranularityLevel.L3_FINE,
        question=question,
        options=options,
        answer_label=answer,
        question_type="spatial_relationship",
        option_hardness_score=option_hardness_score,
        option_hardness_components=option_hardness_components,
        **complexity,
    )


def build_l4_question(
    sg: dict,
    objects: dict,
    rng: random.Random,
) -> Optional[LevelData]:
    """L4 (Very Fine): Compositional MCQ combining attributes + relations.

    Template: "What is the {attribute_category} of the {object} that is
    {relation} the {other_object}?"
    """
    # Find an object with both attributes AND outgoing relations
    candidates = []
    for oid, obj in objects.items():
        attrs = obj.get("attributes", [])
        rels = obj.get("relations", [])
        classified = _classify_attributes(attrs)
        if classified and rels:
            candidates.append((oid, obj, classified, rels))

    if not candidates:
        return None

    oid, obj, classified, rels = rng.choice(candidates)
    name = obj.get("name", "").strip()

    # Pick a relation for the compositional reference
    rel = rng.choice(rels)
    rel_name = rel.get("name", "").strip()
    target_id = rel.get("object", "")
    target_obj = objects.get(target_id)
    if not rel_name or target_obj is None:
        return None
    target_name = target_obj.get("name", "").strip()
    if not target_name:
        return None

    # Pick an attribute to ask about
    cat = rng.choice(list(classified.keys()))
    correct_val = classified[cat]
    distractors = _make_distractors(correct_val, cat, n=3, rng=rng)
    if len(distractors) < 2:
        return None

    all_opts = [correct_val] + distractors[:3]
    rng.shuffle(all_opts)
    labels = ["A", "B", "C", "D"]
    options = {labels[i]: opt for i, opt in enumerate(all_opts)}
    answer = next(l for l, v in options.items() if v == correct_val)

    question = (
        f"What {cat} is the {name} that is {rel_name} the {target_name}?"
    )
    complexity = build_semantic_complexity(
        entity_names=[name, target_name],
        attribute_queries=[cat],
        relation_labels=[rel_name],
        reasoning_ops=["restrict_by_relation", "query_attribute"],
        program_depth=3,
        grounding_candidate_count=_count_named_pairs(objects, name, target_name),
        question_text=question,
        options=options,
    )
    option_hardness_score, option_hardness_components = _compute_option_hardness(
        options=options,
        answer_label=answer,
        attribute_category=cat,
        scene_attributes=_scene_attribute_inventory(objects),
    )
    return LevelData(
        level=GranularityLevel.L4_VERY_FINE,
        question=question,
        options=options,
        answer_label=answer,
        question_type=f"compositional_{cat}",
        option_hardness_score=option_hardness_score,
        option_hardness_components=option_hardness_components,
        **complexity,
    )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _collect_all_object_names(scene_graphs: Dict[str, dict]) -> Set[str]:
    """Collect all unique object names across the dataset for negative sampling."""
    names: Set[str] = set()
    for sg in scene_graphs.values():
        for obj in sg.get("objects", {}).values():
            n = obj.get("name", "").strip()
            if n:
                names.add(n)
    return names


def _is_yes_label(level_data: LevelData) -> bool:
    return str(level_data.answer_label or "").strip().upper() == "A"


def _balance_verification_answers(
    candidates: List[GranularitySample],
    *,
    max_samples: int,
    rng: random.Random,
) -> List[GranularitySample]:
    """Select a subset with exact 50/50 yes/no balance on verification levels.

    Balance is enforced jointly across the verification levels by pairing each
    binary answer pattern with its complement.
    """

    if max_samples <= 0 or not candidates:
        return []

    verification_levels = [level for level in VERIFICATION_VQA_LEVELS if level in ALL_VQA_LEVELS]
    buckets: Dict[Tuple[int, ...], List[GranularitySample]] = {}
    for sample in candidates:
        if not sample.has_all_levels(list(ALL_VQA_LEVELS)):
            continue
        pattern = tuple(
            1 if _is_yes_label(sample.levels[level]) else 0
            for level in verification_levels
        )
        buckets.setdefault(pattern, []).append(sample)

    for samples in buckets.values():
        rng.shuffle(samples)

    unique_patterns = sorted(buckets)
    seen = set()
    pair_keys: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for pattern in unique_patterns:
        if pattern in seen:
            continue
        complement = tuple(1 - value for value in pattern)
        seen.add(pattern)
        seen.add(complement)
        if complement not in buckets:
            continue
        pair_keys.append((pattern, complement))

    pair_capacities = {
        (pattern, complement): min(len(buckets.get(pattern, [])), len(buckets.get(complement, [])))
        for pattern, complement in pair_keys
    }
    total_capacity = 2 * sum(pair_capacities.values())
    if total_capacity <= 0:
        return []

    target = min(max_samples, total_capacity)
    if target % 2 == 1:
        target -= 1
    if target <= 0:
        return []

    rng.shuffle(pair_keys)
    selected: List[GranularitySample] = []
    remaining_pairs = target // 2
    for pattern, complement in pair_keys:
        if remaining_pairs <= 0:
            break
        take = min(pair_capacities[(pattern, complement)], remaining_pairs)
        if take <= 0:
            continue
        selected.extend(buckets[pattern][:take])
        selected.extend(buckets[complement][:take])
        remaining_pairs -= take

    if len(selected) < target:
        logger.warning(
            "Balanced verification sampling could only provide %d samples (requested %d).",
            len(selected),
            target,
        )

    rng.shuffle(selected)
    return selected[:target]


def build_granularity_dataset(
    cache_dir: Path,
    max_samples: int = 1000,
    seed: int = 42,
    split: str = "val",
    allow_download: bool = True,
) -> List[GranularitySample]:
    """Build the 5-level granularity dataset from GQA.

    Downloads scene graphs and images if needed, generates questions at
    all primary levels plus the L5 control, and returns only images where all levels were
    successfully constructed.

    Args:
        cache_dir: Root cache directory.
        max_samples: Maximum number of complete samples to return.
        seed: Random seed for reproducibility.
        split: GQA split (``val`` or ``train``).
        allow_download: Whether to download missing data.

    Returns:
        List of :class:`GranularitySample`, each with L1-L5 questions.
    """
    rng = random.Random(seed)

    # Download data
    sg_dir = download_scene_graphs(cache_dir)
    img_dir = download_gqa_images(cache_dir, allow_download=allow_download)

    # Load scene graphs
    scene_graphs = load_scene_graphs(sg_dir, split)
    all_names = _collect_all_object_names(scene_graphs)

    # Generate questions per image
    candidate_samples: List[GranularitySample] = []
    image_ids = list(scene_graphs.keys())
    rng.shuffle(image_ids)

    for image_id in image_ids:
        sg = scene_graphs[image_id]
        objects = sg.get("objects", {})
        if len(objects) < 2:
            continue

        # Check image exists; download on demand if needed
        img_path = img_dir / f"{image_id}.jpg"
        if not img_path.exists():
            if not download_single_image(image_id, img_dir):
                continue

        # Pick a random object with enough annotations
        obj_ids = list(objects.keys())
        rng.shuffle(obj_ids)

        levels: Dict[GranularityLevel, LevelData] = {}

        for obj_id in obj_ids:
            obj = objects[obj_id]

            if GranularityLevel.L1_COARSE not in levels:
                q = build_l1_question(sg, obj_id, obj, rng, all_names)
                if q is not None:
                    levels[GranularityLevel.L1_COARSE] = q

            if GranularityLevel.L5_WORDY_SIMPLETON not in levels:
                desired_answer = None
                if GranularityLevel.L1_COARSE in levels:
                    desired_answer = levels[GranularityLevel.L1_COARSE].answer_label
                q = build_l5_question(sg, obj_id, obj, rng, all_names, desired_answer_label=desired_answer)
                if q is not None:
                    levels[GranularityLevel.L5_WORDY_SIMPLETON] = q

            if GranularityLevel.L2_MEDIUM not in levels:
                q = build_l2_question(sg, obj_id, obj, rng)
                if q is not None:
                    levels[GranularityLevel.L2_MEDIUM] = q

            if GranularityLevel.L3_FINE not in levels:
                q = build_l3_question(sg, obj_id, obj, objects, rng)
                if q is not None:
                    levels[GranularityLevel.L3_FINE] = q

            if len(levels) >= 4:
                break

        # L4 uses the full scene graph
        if GranularityLevel.L4_VERY_FINE not in levels:
            q = build_l4_question(sg, objects, rng)
            if q is not None:
                levels[GranularityLevel.L4_VERY_FINE] = q

        # Only keep images where all levels were generated
        sample = GranularitySample(
                image_id=image_id,
                image_path=img_path,
                levels=levels,
                dataset="gqa",
                split=split,
                metadata={"num_objects": len(objects)},
            )
        if sample.has_all_levels(list(ALL_VQA_LEVELS)):
            candidate_samples.append(sample)

    samples = _balance_verification_answers(
        candidate_samples,
        max_samples=max_samples,
        rng=rng,
    )
    logger.info(
        "Built %d complete balanced samples (all 5 levels) from %d scene graphs",
        len(samples), len(scene_graphs)
    )
    return samples
