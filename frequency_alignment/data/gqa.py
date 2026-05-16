"""GQA dataset loader with primary and wordy-control granularity questions.

Downloads GQA scene graphs and images, then generates questions at four
granularity levels from the same image:

- **L1 (Coarse)**: Object presence -- "Is there a {object} in this image?"
- **L2 (Medium)**: Attribute recognition -- "What {attr_type} is the {object}?"
- **L3 (Fine)**: Spatial relationship -- "Is the {obj1} to the {relation} of {obj2}?"
- **L4 (Very Fine)**: Compositional MCQ -- combines attributes + spatial reasoning
- **L5 (Wordy-Simpleton)**: L1 semantics with extra neutral filler
- **L6 (Wordy-Medium)**: L2 semantics with extra neutral filler
- **L7 (Wordy-Fine)**: L3 semantics with extra neutral filler
- **L8 (Wordy-Very-Fine)**: L4 semantics with extra neutral filler

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
from .complexity import (
    build_semantic_complexity,
    prompt_content_word_count,
    question_content_words,
    refresh_prompt_load,
)

logger = logging.getLogger(__name__)

_DATASET_CACHE_VERSION = "gqa_multilevel_v7"
_DATASET_MEMO: Dict[str, List[GranularitySample]] = {}
_WORDY_MIN_EXTRA_PROMPT_CONTENT_WORDS = 12

# v7: L3 negative-case restricted to relations with a known cardinal opposite,
# eliminating the "joint-truth" label-noise where the randomly-picked wrong
# relation might also be true in the scene (e.g. "to the left of" and "behind"
# can both hold). Cardinal opposites are mutually exclusive by physical
# geometry, so a flip is provably false.
_GQA_OPPOSITE_RELATION = {
    "to the left of": "to the right of",
    "to the right of": "to the left of",
    "above": "below",
    "below": "above",
    "behind": "in front of",
    "in front of": "behind",
    "on top of": "under",
    "under": "on top of",
}

_WORDY_FILLER_PREFIX = (
    "Please read the following question carefully and answer the same question directly. "
)
_WORDY_FILLER_SUFFIX = (
    "The extra wording is only polite framing and should not change what you are being asked to answer."
)
_WORDY_NEUTRAL_FILLER_WORDS = (
    "context",
    "review",
    "observation",
    "perspective",
    "analysis",
    "instance",
    "scene",
    "visual",
    "careful",
    "straightforward",
    "response",
    "neutral",
    "polite",
    "steady",
    "focused",
    "general",
    "supportive",
    "ordinary",
    "plain",
    "background",
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


def _neutral_filler_words(num_words: int) -> str:
    if num_words <= 0:
        return ""
    words = [
        _WORDY_NEUTRAL_FILLER_WORDS[idx % len(_WORDY_NEUTRAL_FILLER_WORDS)]
        for idx in range(num_words)
    ]
    return " ".join(words)


def _ensure_wordier_prompt(
    question: str,
    *,
    options: Optional[Dict[str, str]],
    base_prompt_content_words: int,
) -> str:
    """Pad only as needed so the wordy prompt is longer than its base prompt."""

    normalized = " ".join(str(question or "").strip().split())
    if not normalized:
        return normalized
    current = prompt_content_word_count(normalized, options)
    target_content_words = max(
        int(base_prompt_content_words) + 1,
        int(base_prompt_content_words) + _WORDY_MIN_EXTRA_PROMPT_CONTENT_WORDS,
    )
    if current >= target_content_words:
        return normalized
    needed = target_content_words - current
    return f"{normalized} {_neutral_filler_words(needed)}.".strip()


def _wordify_question(
    question: str,
    options: Optional[Dict[str, str]] = None,
    *,
    base_prompt_content_words: Optional[int] = None,
) -> str:
    base = str(question or "").strip()
    if not base:
        return base
    base_prompt_content_words = (
        int(base_prompt_content_words)
        if base_prompt_content_words is not None
        else prompt_content_word_count(base, options)
    )
    candidate = f"{_WORDY_FILLER_PREFIX}{base} {_WORDY_FILLER_SUFFIX}"
    return _ensure_wordier_prompt(
        candidate,
        options=options,
        base_prompt_content_words=base_prompt_content_words,
    )


def _build_wordy_variant(
    base_level: LevelData,
    *,
    level: GranularityLevel,
    question_type: str,
) -> LevelData:
    base_prompt_content_words = int(
        base_level.prompt_complexity_score
        if base_level.prompt_complexity_score is not None
        else prompt_content_word_count(base_level.question, base_level.options)
    )
    question = _wordify_question(
        base_level.question,
        options=base_level.options,
        base_prompt_content_words=base_prompt_content_words,
    )
    complexity = refresh_prompt_load(
        {
            "semantic_atoms": list(base_level.semantic_atoms),
            "prompt_semantic_atoms": list(base_level.prompt_semantic_atoms),
            "semantic_atom_counts": dict(base_level.semantic_atom_counts),
            "question_complexity_score": float(base_level.question_complexity_score),
            "prompt_complexity_score": float(base_level.prompt_complexity_score),
            "complexity_score": float(base_level.complexity_score),
        },
        question_text=question,
        options=base_level.options,
    )
    counts = dict(complexity.get("semantic_atom_counts", {}) or {})
    counts["wordy_base_prompt_content_words"] = base_prompt_content_words
    counts["wordy_min_extra_prompt_content_words"] = _WORDY_MIN_EXTRA_PROMPT_CONTENT_WORDS
    counts["wordy_prompt_content_words"] = prompt_content_word_count(question, base_level.options)
    counts["wordy_question_content_words"] = len(question_content_words(question))
    complexity["semantic_atom_counts"] = counts
    return LevelData(
        level=level,
        question=question,
        options=dict(base_level.options),
        answer_label=base_level.answer_label,
        question_type=question_type,
        option_hardness_score=float(base_level.option_hardness_score),
        option_hardness_components=dict(base_level.option_hardness_components),
        **complexity,
    )


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
    base_prompt_content_words = prompt_content_word_count(base_question, options)
    question = (
        _wordify_question(
            base_question,
            options=options,
            base_prompt_content_words=base_prompt_content_words,
        )
        if wordy
        else base_question
    )
    complexity = build_semantic_complexity(
        entity_names=[name],
        reasoning_ops=["exist"],
        program_depth=1,
        grounding_candidate_count=grounding_candidates,
        question_text=question,
        options=options,
    )
    if wordy:
        counts = dict(complexity.get("semantic_atom_counts", {}) or {})
        counts["wordy_base_prompt_content_words"] = base_prompt_content_words
        counts["wordy_min_extra_prompt_content_words"] = _WORDY_MIN_EXTRA_PROMPT_CONTENT_WORDS
        counts["wordy_prompt_content_words"] = prompt_content_word_count(question, options)
        counts["wordy_question_content_words"] = len(question_content_words(question))
        complexity["semantic_atom_counts"] = counts
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

    base_level = _build_object_presence_question(
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
    return build_l5_from_base(base_level)


def build_l5_from_base(base_level: Optional[LevelData]) -> Optional[LevelData]:
    """L5: exact wordy mirror of an already-built L1 question."""

    if base_level is None:
        return None
    return _build_wordy_variant(
        base_level,
        level=GranularityLevel.L5_WORDY_SIMPLETON,
        question_type=f"{base_level.question_type}_wordy_control",
    )


def build_l6_question(base_level: LevelData) -> Optional[LevelData]:
    """L6: L2 semantics with redundant filler text."""

    if base_level is None:
        return None
    return _build_wordy_variant(
        base_level,
        level=GranularityLevel.L6_WORDY_MEDIUM,
        question_type=f"{base_level.question_type}_wordy_control",
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
        # Restrict the wrong relation to a known cardinal opposite so it is
        # provably false. The previous random-from-fixed-list approach could
        # pick a relation that was also jointly true (e.g. "above" and "behind"
        # can both hold), producing label noise. Relations without a defined
        # opposite (e.g. "next to", "near") get filtered — caller retries.
        wrong_rel = _GQA_OPPOSITE_RELATION.get(rel_name)
        if not wrong_rel:
            return None
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


def build_l7_question(base_level: LevelData) -> Optional[LevelData]:
    """L7: L3 semantics with redundant filler text."""

    if base_level is None:
        return None
    return _build_wordy_variant(
        base_level,
        level=GranularityLevel.L7_WORDY_FINE,
        question_type=f"{base_level.question_type}_wordy_control",
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


def build_l8_question(base_level: LevelData) -> Optional[LevelData]:
    """L8: L4 semantics with redundant filler text."""

    if base_level is None:
        return None
    return _build_wordy_variant(
        base_level,
        level=GranularityLevel.L8_WORDY_VERY_FINE,
        question_type=f"{base_level.question_type}_wordy_control",
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


def _verification_pattern(sample: GranularitySample) -> Tuple[int, ...]:
    verification_levels = [level for level in VERIFICATION_VQA_LEVELS if level in ALL_VQA_LEVELS]
    return tuple(
        1 if _is_yes_label(sample.levels[level]) else 0
        for level in verification_levels
    )


def _balanced_target(max_samples: int) -> int:
    if max_samples <= 0:
        return 0
    return int(max_samples) if int(max_samples) % 2 == 0 else int(max_samples) - 1


def _balanced_capacity_from_counts(pattern_counts: Dict[Tuple[int, ...], int]) -> int:
    seen: Set[Tuple[int, ...]] = set()
    capacity = 0
    for pattern in sorted(pattern_counts):
        if pattern in seen:
            continue
        complement = tuple(1 - value for value in pattern)
        seen.add(pattern)
        seen.add(complement)
        if complement not in pattern_counts:
            continue
        capacity += 2 * min(pattern_counts.get(pattern, 0), pattern_counts.get(complement, 0))
    return int(capacity)


def _serialize_level_data(level_data: LevelData) -> Dict[str, Any]:
    return {
        "level": level_data.level.name,
        "question": level_data.question,
        "options": dict(level_data.options),
        "answer_label": level_data.answer_label,
        "question_type": level_data.question_type,
        "semantic_atoms": list(level_data.semantic_atoms),
        "prompt_semantic_atoms": list(level_data.prompt_semantic_atoms),
        "semantic_atom_counts": dict(level_data.semantic_atom_counts),
        "question_complexity_score": float(level_data.question_complexity_score),
        "prompt_complexity_score": float(level_data.prompt_complexity_score),
        "complexity_score": float(level_data.complexity_score),
        "option_hardness_score": float(level_data.option_hardness_score),
        "option_hardness_components": dict(level_data.option_hardness_components),
    }


def _deserialize_level_data(payload: Dict[str, Any]) -> LevelData:
    return LevelData(
        level=GranularityLevel[str(payload["level"])],
        question=str(payload.get("question", "")),
        options={str(k): str(v) for k, v in dict(payload.get("options", {})).items()},
        answer_label=payload.get("answer_label"),
        question_type=payload.get("question_type"),
        semantic_atoms=[str(value) for value in payload.get("semantic_atoms", [])],
        prompt_semantic_atoms=[str(value) for value in payload.get("prompt_semantic_atoms", [])],
        semantic_atom_counts=dict(payload.get("semantic_atom_counts", {})),
        question_complexity_score=float(payload.get("question_complexity_score", 0.0) or 0.0),
        prompt_complexity_score=float(payload.get("prompt_complexity_score", 0.0) or 0.0),
        complexity_score=float(payload.get("complexity_score", 0.0) or 0.0),
        option_hardness_score=float(payload.get("option_hardness_score", 0.0) or 0.0),
        option_hardness_components=dict(payload.get("option_hardness_components", {})),
    )


def _serialize_sample(sample: GranularitySample) -> Dict[str, Any]:
    return {
        "image_id": str(sample.image_id),
        "image_path": str(sample.image_path),
        "levels": {
            level.name: _serialize_level_data(level_data)
            for level, level_data in sample.levels.items()
        },
        "dataset": sample.dataset,
        "split": sample.split,
        "metadata": dict(sample.metadata),
    }


def _deserialize_sample(payload: Dict[str, Any]) -> GranularitySample:
    levels = {
        GranularityLevel[level_name]: _deserialize_level_data(level_payload)
        for level_name, level_payload in dict(payload.get("levels", {})).items()
    }
    return GranularitySample(
        image_id=str(payload.get("image_id", "")),
        image_path=Path(payload.get("image_path", "")),
        levels=levels,
        dataset=str(payload.get("dataset", "")),
        split=str(payload.get("split", "val")),
        metadata=dict(payload.get("metadata", {})),
    )


def _dataset_cache_key(*, split: str, seed: int, max_samples: int, allow_download: bool) -> str:
    return (
        f"{_DATASET_CACHE_VERSION}|split={split}|seed={int(seed)}|"
        f"max={int(max_samples)}|allow_download={int(bool(allow_download))}"
    )


def _dataset_cache_path(
    cache_dir: Path,
    *,
    split: str,
    seed: int,
    max_samples: int,
    allow_download: bool,
) -> Path:
    cache_root = cache_dir / "gqa" / "granularity_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / (
        f"{_DATASET_CACHE_VERSION}_{split}_seed{int(seed)}_max{int(max_samples)}_"
        f"allowdl{int(bool(allow_download))}.json"
    )


def _load_cached_dataset(path: Path, expected_key: str) -> Optional[List[GranularitySample]]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
        if payload.get("cache_key") != expected_key:
            return None
        samples = [_deserialize_sample(item) for item in payload.get("samples", [])]
        if not all(sample.image_path.exists() for sample in samples):
            logger.info("Ignoring stale dataset cache with missing images: %s", path)
            return None
        logger.info("Loaded cached GQA multilevel dataset: %s (%d samples)", path, len(samples))
        return samples
    except Exception as exc:
        logger.warning("Failed to load cached GQA multilevel dataset %s: %s", path, exc)
        return None


def _save_cached_dataset(path: Path, cache_key: str, samples: List[GranularitySample]) -> None:
    payload = {
        "cache_key": cache_key,
        "version": _DATASET_CACHE_VERSION,
        "num_samples": len(samples),
        "samples": [_serialize_sample(sample) for sample in samples],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def build_granularity_dataset(
    cache_dir: Path,
    max_samples: int = 1000,
    seed: int = 42,
    split: str = "val",
    allow_download: bool = True,
) -> List[GranularitySample]:
    """Build the multilevel granularity dataset from GQA.

    Downloads scene graphs and images if needed, generates questions at
    all primary levels plus the wordy controls, and returns only images where
    all levels were successfully constructed.

    Args:
        cache_dir: Root cache directory.
        max_samples: Maximum number of complete samples to return.
        seed: Random seed for reproducibility.
        split: GQA split (``val`` or ``train``).
        allow_download: Whether to download missing data.

    Returns:
        List of :class:`GranularitySample`, each with L1-L8 questions.
    """
    cache_key = _dataset_cache_key(
        split=split,
        seed=seed,
        max_samples=max_samples,
        allow_download=allow_download,
    )
    memoized = _DATASET_MEMO.get(cache_key)
    if memoized is not None:
        logger.info("Using in-memory cached GQA multilevel dataset (%d samples)", len(memoized))
        return list(memoized)

    cache_path = _dataset_cache_path(
        cache_dir,
        split=split,
        seed=seed,
        max_samples=max_samples,
        allow_download=allow_download,
    )
    cached_samples = _load_cached_dataset(cache_path, cache_key)
    if cached_samples is not None:
        _DATASET_MEMO[cache_key] = list(cached_samples)
        return list(cached_samples)

    rng = random.Random(seed)
    target_samples = _balanced_target(max_samples)
    if target_samples <= 0:
        logger.warning(
            "Requested max_samples=%d cannot satisfy exact 50/50 verification balancing; returning no samples.",
            max_samples,
        )
        _DATASET_MEMO[cache_key] = []
        _save_cached_dataset(cache_path, cache_key, [])
        return []

    # Download data
    sg_dir = download_scene_graphs(cache_dir)
    img_dir = download_gqa_images(cache_dir, allow_download=allow_download)

    # Load scene graphs
    scene_graphs = load_scene_graphs(sg_dir, split)
    all_names = _collect_all_object_names(scene_graphs)

    # Generate questions per image
    candidate_samples: List[GranularitySample] = []
    pattern_counts: Dict[Tuple[int, ...], int] = {}
    image_ids = list(scene_graphs.keys())
    rng.shuffle(image_ids)
    progress_interval = 250

    scanned_count = 0
    for index, image_id in enumerate(image_ids, start=1):
        scanned_count = index
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
                    q_wordy = build_l5_from_base(q)
                    if q_wordy is not None:
                        levels[GranularityLevel.L5_WORDY_SIMPLETON] = q_wordy

            if GranularityLevel.L2_MEDIUM not in levels:
                q = build_l2_question(sg, obj_id, obj, rng)
                if q is not None:
                    levels[GranularityLevel.L2_MEDIUM] = q
                    q_wordy = build_l6_question(q)
                    if q_wordy is not None:
                        levels[GranularityLevel.L6_WORDY_MEDIUM] = q_wordy

            if GranularityLevel.L3_FINE not in levels:
                q = build_l3_question(sg, obj_id, obj, objects, rng)
                if q is not None:
                    levels[GranularityLevel.L3_FINE] = q
                    q_wordy = build_l7_question(q)
                    if q_wordy is not None:
                        levels[GranularityLevel.L7_WORDY_FINE] = q_wordy

            if len(levels) >= 6:
                break

        # L4 uses the full scene graph
        if GranularityLevel.L4_VERY_FINE not in levels:
            q = build_l4_question(sg, objects, rng)
            if q is not None:
                levels[GranularityLevel.L4_VERY_FINE] = q
                q_wordy = build_l8_question(q)
                if q_wordy is not None:
                    levels[GranularityLevel.L8_WORDY_VERY_FINE] = q_wordy

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
            pattern = _verification_pattern(sample)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        if index % progress_interval == 0:
            balanced_capacity = _balanced_capacity_from_counts(pattern_counts)
            logger.info(
                "GQA dataset build progress: checked %d/%d scene graphs, complete=%d, balanced_capacity=%d/%d",
                index,
                len(image_ids),
                len(candidate_samples),
                balanced_capacity,
                target_samples,
            )

        if _balanced_capacity_from_counts(pattern_counts) >= target_samples:
            logger.info(
                "Reached balanced target after %d/%d scene graphs; stopping dataset construction early.",
                index,
                len(image_ids),
            )
            break

    samples = _balance_verification_answers(
        candidate_samples,
        max_samples=max_samples,
        rng=rng,
    )
    logger.info(
        "Built %d complete balanced samples (all %d levels) after scanning %d/%d scene graphs",
        len(samples), len(ALL_VQA_LEVELS), scanned_count, len(scene_graphs)
    )
    _DATASET_MEMO[cache_key] = list(samples)
    _save_cached_dataset(cache_path, cache_key, samples)
    return samples
