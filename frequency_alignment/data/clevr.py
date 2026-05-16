"""CLEVR loader with generated L1-L8 granularity questions.

The loader reads the official CLEVR scene files and creates one matched
multi-level VQA ladder per image. This keeps the causal setup identical to GQA
while using CLEVR's controlled visual grammar.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import GranularityLevel, GranularitySample, LevelData
from .complexity import (
    build_semantic_complexity,
    prompt_content_word_count,
    question_content_words,
    refresh_prompt_load,
)

logger = logging.getLogger(__name__)


_CLEVR_CACHE_VERSION = "clevr_multilevel_v1"
_DATASET_MEMO: Dict[str, List[GranularitySample]] = {}

_COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
_MATERIALS = ["rubber", "metal"]
_SHAPES = ["cube", "sphere", "cylinder"]
_SIZES = ["small", "large"]
_ATTR_VOCAB = {
    "color": _COLORS,
    "material": _MATERIALS,
    "shape": _SHAPES,
    "size": _SIZES,
}
_RELATION_TEXT = {
    "left": "to the left of",
    "right": "to the right of",
    "front": "in front of",
    "behind": "behind",
}
_OPPOSITE_RELATION = {
    "left": "right",
    "right": "left",
    "front": "behind",
    "behind": "front",
}

_WORDY_MIN_EXTRA_PROMPT_CONTENT_WORDS = 12
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
)


def _find_clevr_root(cache_dir: Path, root_override: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if root_override is not None:
        candidates.append(root_override)
    candidates.extend(
        [
            cache_dir / "clevr" / "CLEVR_v1.0",
            cache_dir / "CLEVR_v1.0",
            Path(".hf_cache") / "clevr" / "CLEVR_v1.0",
            Path("data") / "CLEVR_v1.0",
            Path.home() / "datasets" / "CLEVR_v1.0",
        ]
    )
    for candidate in candidates:
        root = candidate.expanduser()
        if (root / "scenes").exists() and (root / "images").exists():
            return root
    return None


def _split_name(split: str) -> str:
    normalized = str(split or "val").lower()
    if normalized in {"validation", "valid"}:
        return "val"
    if normalized in {"train", "test", "val"}:
        return normalized
    raise ValueError(f"Unsupported CLEVR split {split!r}; use train, val, or test")


def _scene_file(root: Path, split: str) -> Path:
    return root / "scenes" / f"CLEVR_{split}_scenes.json"


def _image_dir(root: Path, split: str) -> Path:
    return root / "images" / split


def _load_scenes(root: Path, split: str) -> List[Dict[str, Any]]:
    path = _scene_file(root, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing CLEVR scene file: {path}")
    with path.open("r") as handle:
        payload = json.load(handle)
    scenes = list(payload.get("scenes", []))
    logger.info("Loaded %d CLEVR %s scenes from %s", len(scenes), split, path)
    return scenes


def _objects(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(obj) for obj in scene.get("objects", []) or []]


def _attr(obj: Dict[str, Any], name: str) -> str:
    return str(obj.get(name, "")).strip().lower()


def _full_phrase(obj: Dict[str, Any]) -> str:
    return " ".join(
        part
        for part in [_attr(obj, "size"), _attr(obj, "color"), _attr(obj, "material"), _attr(obj, "shape")]
        if part
    )


def _phrase_excluding(obj: Dict[str, Any], exclude_attr: str) -> str:
    if exclude_attr == "color":
        parts = [_attr(obj, "size"), _attr(obj, "material"), _attr(obj, "shape")]
    elif exclude_attr == "size":
        parts = [_attr(obj, "color"), _attr(obj, "material"), _attr(obj, "shape")]
    elif exclude_attr == "material":
        parts = [_attr(obj, "size"), _attr(obj, "color"), _attr(obj, "shape")]
    elif exclude_attr == "shape":
        parts = [_attr(obj, "size"), _attr(obj, "color"), _attr(obj, "material"), "object"]
    else:
        parts = [_full_phrase(obj)]
    return " ".join(part for part in parts if part)


def _descriptor_count(objects: Sequence[Dict[str, Any]], target: Dict[str, Any], exclude_attr: str) -> int:
    phrase = _phrase_excluding(target, exclude_attr)
    return sum(1 for obj in objects if _phrase_excluding(obj, exclude_attr) == phrase)


def _shape_count(objects: Sequence[Dict[str, Any]], shape: str) -> int:
    return sum(1 for obj in objects if _attr(obj, "shape") == shape)


def _scene_attribute_inventory(objects: Sequence[Dict[str, Any]]) -> set[str]:
    inventory: set[str] = set()
    for obj in objects:
        for attr_name in ["color", "material", "shape", "size"]:
            value = _attr(obj, attr_name)
            if value:
                inventory.add(value)
    return inventory


def _make_options(correct: str, category: str, rng: random.Random, max_options: int = 4) -> Tuple[Dict[str, str], str]:
    vocab = [item for item in _ATTR_VOCAB[category] if item != correct]
    rng.shuffle(vocab)
    values = [correct] + vocab[: max(1, max_options - 1)]
    rng.shuffle(values)
    labels = ["A", "B", "C", "D"][: len(values)]
    options = {label: value for label, value in zip(labels, values)}
    answer = next(label for label, value in options.items() if value == correct)
    return options, answer


def _binary_options(answer_yes: bool) -> Tuple[Dict[str, str], str]:
    return {"A": "yes", "B": "no"}, "A" if answer_yes else "B"


def _option_hardness(
    *,
    options: Dict[str, str],
    answer_label: Optional[str],
    attribute_category: Optional[str],
    scene_attributes: set[str],
) -> Tuple[float, Dict[str, Any]]:
    if not options or answer_label not in options:
        return 0.0, {"num_distractors": 0, "same_category_distractors": 0, "scene_present_distractors": 0}
    distractors = [
        str(value or "").strip().lower()
        for label, value in options.items()
        if label != answer_label
    ]
    vocab = set(_ATTR_VOCAB.get(attribute_category or "", []))
    same_category = sum(1 for value in distractors if value in vocab)
    scene_present = sum(1 for value in distractors if value in scene_attributes)
    return float(same_category + 0.5 * scene_present), {
        "num_distractors": len(distractors),
        "same_category_distractors": int(same_category),
        "scene_present_distractors": int(scene_present),
    }


def _neutral_filler_words(num_words: int) -> str:
    return " ".join(
        _WORDY_NEUTRAL_FILLER_WORDS[idx % len(_WORDY_NEUTRAL_FILLER_WORDS)]
        for idx in range(max(0, num_words))
    )


def _wordify_question(question: str, options: Dict[str, str], base_prompt_content_words: int) -> str:
    base = str(question or "").strip()
    candidate = f"{_WORDY_FILLER_PREFIX}{base} {_WORDY_FILLER_SUFFIX}"
    current = prompt_content_word_count(candidate, options)
    target = int(base_prompt_content_words) + _WORDY_MIN_EXTRA_PROMPT_CONTENT_WORDS
    if current >= target:
        return candidate
    return f"{candidate} {_neutral_filler_words(target - current)}.".strip()


def _wordy_variant(base_level: LevelData, *, level: GranularityLevel, question_type: str) -> LevelData:
    base_prompt_content_words = int(base_level.prompt_complexity_score or prompt_content_word_count(base_level.question, base_level.options))
    question = _wordify_question(base_level.question, base_level.options, base_prompt_content_words)
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


def _relation_indices(scene: Dict[str, Any], relation: str, target_idx: int) -> List[int]:
    rel = scene.get("relationships", {}).get(relation)
    if not isinstance(rel, list) or target_idx >= len(rel):
        return []
    return [int(idx) for idx in rel[target_idx]]


def _has_relation(scene: Dict[str, Any], relation: str, target_idx: int, ref_idx: int) -> bool:
    return ref_idx in _relation_indices(scene, relation, target_idx)


def _candidate_attribute(
    objects: Sequence[Dict[str, Any]],
    target: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Pick the first uniquely-identifying attribute for *target*.

    Without ``rng`` this iterates ``[color, shape, material, size]`` in fixed
    order (legacy behaviour, preserved for callers that don't supply an rng).
    With ``rng`` the iteration order is shuffled per call so the chosen
    attribute category is balanced across runs instead of being color-heavy.
    Returns the first category in iteration order whose
    ``_descriptor_count == 1`` (uniquely identifies *target* in the scene);
    returns ``None`` if no attribute uniquely identifies the target.
    """
    categories = ["color", "shape", "material", "size"]
    if rng is not None:
        categories = list(categories)
        rng.shuffle(categories)
    for category in categories:
        if category not in _ATTR_VOCAB or not _attr(target, category):
            continue
        if _descriptor_count(objects, target, category) == 1:
            return category
    return None


def _l1_presence_query(
    target: Dict[str, Any],
    objects: Sequence[Dict[str, Any]],
    make_false: bool,
) -> Optional[Tuple[str, List[str], int, bool]]:
    if make_false:
        present_shapes = {_attr(obj, "shape") for obj in objects}
        absent_shapes = [shape for shape in _SHAPES if shape not in present_shapes]
        if absent_shapes:
            shape = absent_shapes[0]
            return f"Is there a {shape} in this image?", [shape], 0, False

        present_colors = {_attr(obj, "color") for obj in objects}
        absent_colors = [color for color in _COLORS if color not in present_colors]
        if absent_colors:
            color = absent_colors[0]
            return f"Is there a {color} object in this image?", [color, "object"], 0, False

        # No verifiably absent shape or color in this scene; refuse to silently
        # return a positive when a negative was requested. The caller will move
        # on to another target_idx (or scene) instead of injecting a label-
        # imbalance bias.
        return None

    shape = _attr(target, "shape")
    return f"Is there a {shape} in this image?", [shape], _shape_count(objects, shape), True


def _build_l1(
    scene: Dict[str, Any],
    target: Dict[str, Any],
    objects: Sequence[Dict[str, Any]],
    *,
    make_false: bool,
) -> Optional[LevelData]:
    result = _l1_presence_query(target, objects, make_false)
    if result is None:
        return None
    question, entity_names, candidate_count, answer_yes = result
    options, answer = _binary_options(answer_yes)
    complexity = build_semantic_complexity(
        entity_names=entity_names,
        reasoning_ops=["exist"],
        program_depth=1,
        grounding_candidate_count=candidate_count,
        question_text=question,
        options=options,
    )
    hardness, components = _option_hardness(
        options=options,
        answer_label=answer,
        attribute_category=None,
        scene_attributes=_scene_attribute_inventory(objects),
    )
    return LevelData(
        level=GranularityLevel.L1_COARSE,
        question=question,
        options=options,
        answer_label=answer,
        question_type="clevr_object_presence",
        option_hardness_score=hardness,
        option_hardness_components=components,
        **complexity,
    )


def _build_l2(
    scene: Dict[str, Any],
    target: Dict[str, Any],
    objects: Sequence[Dict[str, Any]],
    rng: random.Random,
) -> Optional[LevelData]:
    category = _candidate_attribute(objects, target, rng=rng)
    if category is None:
        return None
    correct = _attr(target, category)
    phrase = _phrase_excluding(target, category)
    question = f"What {category} is the {phrase}?"
    options, answer = _make_options(correct, category, rng)
    complexity = build_semantic_complexity(
        entity_names=[phrase],
        attribute_queries=[category],
        reasoning_ops=["query_attribute"],
        program_depth=2,
        grounding_candidate_count=_descriptor_count(objects, target, category),
        question_text=question,
        options=options,
    )
    hardness, components = _option_hardness(
        options=options,
        answer_label=answer,
        attribute_category=category,
        scene_attributes=_scene_attribute_inventory(objects),
    )
    return LevelData(
        level=GranularityLevel.L2_MEDIUM,
        question=question,
        options=options,
        answer_label=answer,
        question_type=f"clevr_attribute_{category}",
        option_hardness_score=hardness,
        option_hardness_components=components,
        **complexity,
    )


def _relation_pair(scene: Dict[str, Any], objects: Sequence[Dict[str, Any]], target_idx: int) -> Optional[Tuple[str, int]]:
    for relation in ["left", "right", "front", "behind"]:
        for ref_idx in _relation_indices(scene, relation, target_idx):
            if 0 <= ref_idx < len(objects) and ref_idx != target_idx:
                return relation, ref_idx
    return None


def _build_l3(
    scene: Dict[str, Any],
    target_idx: int,
    objects: Sequence[Dict[str, Any]],
    *,
    make_false: bool,
) -> Optional[LevelData]:
    pair = _relation_pair(scene, objects, target_idx)
    if pair is None:
        return None
    # _relation_pair returns (stored_rel, ref_idx) such that
    #   ref ∈ scene.relationships[stored_rel][target_idx]
    # i.e. *ref is stored_rel of target* in CLEVR's scene-graph convention.
    # Equivalently, *target is OPPOSITE_RELATION[stored_rel] of ref*.
    # The question phrases target as the subject ("Is target REL of ref?"), so
    # the literal-true relation between target→ref is the cardinal opposite of
    # what the scene graph stored. For the negative case we use stored_rel,
    # which is the cardinal opposite of the true direction and therefore
    # provably false (left/right and front/behind are mutually exclusive).
    stored_rel, ref_idx = pair
    correct_dir = _OPPOSITE_RELATION[stored_rel]
    relation = stored_rel if make_false else correct_dir
    answer_yes = not make_false
    target = objects[target_idx]
    ref = objects[ref_idx]
    target_phrase = _full_phrase(target)
    ref_phrase = _full_phrase(ref)
    relation_text = _RELATION_TEXT[relation]
    question = f"Is the {target_phrase} {relation_text} the {ref_phrase}?"
    options, answer = _binary_options(answer_yes)
    complexity = build_semantic_complexity(
        entity_names=[target_phrase, ref_phrase],
        relation_labels=[relation_text],
        reasoning_ops=["verify_relation"],
        program_depth=3,
        grounding_candidate_count=1,
        question_text=question,
        options=options,
    )
    hardness, components = _option_hardness(
        options=options,
        answer_label=answer,
        attribute_category=None,
        scene_attributes=_scene_attribute_inventory(objects),
    )
    return LevelData(
        level=GranularityLevel.L3_FINE,
        question=question,
        options=options,
        answer_label=answer,
        question_type="clevr_spatial_relationship",
        option_hardness_score=hardness,
        option_hardness_components=components,
        **complexity,
    )


def _build_l4(
    scene: Dict[str, Any],
    target_idx: int,
    objects: Sequence[Dict[str, Any]],
    rng: random.Random,
) -> Optional[LevelData]:
    target = objects[target_idx]
    category = _candidate_attribute(objects, target, rng=rng)
    pair = _relation_pair(scene, objects, target_idx)
    if category is None or pair is None:
        return None
    # See _build_l3: _relation_pair returns (stored_rel, ref) meaning ref is
    # stored_rel of target. The L4 descriptor phrases target as the subject
    # ("target that is REL of ref"), so we must use the cardinal opposite for
    # the relation_text to point to target's true position relative to ref.
    stored_rel, ref_idx = pair
    relation = _OPPOSITE_RELATION[stored_rel]
    ref = objects[ref_idx]
    correct = _attr(target, category)
    target_phrase = _phrase_excluding(target, category)
    ref_phrase = _full_phrase(ref)
    relation_text = _RELATION_TEXT[relation]
    question = f"What {category} is the {target_phrase} that is {relation_text} the {ref_phrase}?"
    options, answer = _make_options(correct, category, rng)
    complexity = build_semantic_complexity(
        entity_names=[target_phrase, ref_phrase],
        attribute_queries=[category],
        relation_labels=[relation_text],
        reasoning_ops=["restrict_by_relation", "query_attribute"],
        program_depth=4,
        grounding_candidate_count=1,
        question_text=question,
        options=options,
    )
    hardness, components = _option_hardness(
        options=options,
        answer_label=answer,
        attribute_category=category,
        scene_attributes=_scene_attribute_inventory(objects),
    )
    return LevelData(
        level=GranularityLevel.L4_VERY_FINE,
        question=question,
        options=options,
        answer_label=answer,
        question_type=f"clevr_compositional_{category}",
        option_hardness_score=hardness,
        option_hardness_components=components,
        **complexity,
    )


def _build_levels_for_scene(scene: Dict[str, Any], rng: random.Random, image_index: int) -> Optional[Dict[GranularityLevel, LevelData]]:
    objects = _objects(scene)
    if len(objects) < 2:
        return None
    indices = list(range(len(objects)))
    rng.shuffle(indices)
    for target_idx in indices:
        target = objects[target_idx]
        if not all(_attr(target, attr_name) for attr_name in ["color", "material", "shape", "size"]):
            continue
        l2 = _build_l2(scene, target, objects, rng)
        l3 = _build_l3(scene, target_idx, objects, make_false=bool(image_index % 2))
        l4 = _build_l4(scene, target_idx, objects, rng)
        if l2 is None or l3 is None or l4 is None:
            continue
        l1 = _build_l1(scene, target, objects, make_false=bool((image_index + 1) % 2))
        if l1 is None:
            continue
        levels: Dict[GranularityLevel, LevelData] = {
            GranularityLevel.L1_COARSE: l1,
            GranularityLevel.L2_MEDIUM: l2,
            GranularityLevel.L3_FINE: l3,
            GranularityLevel.L4_VERY_FINE: l4,
        }
        levels[GranularityLevel.L5_WORDY_SIMPLETON] = _wordy_variant(
            l1,
            level=GranularityLevel.L5_WORDY_SIMPLETON,
            question_type="clevr_object_presence_wordy_control",
        )
        levels[GranularityLevel.L6_WORDY_MEDIUM] = _wordy_variant(
            l2,
            level=GranularityLevel.L6_WORDY_MEDIUM,
            question_type=f"{l2.question_type}_wordy_control",
        )
        levels[GranularityLevel.L7_WORDY_FINE] = _wordy_variant(
            l3,
            level=GranularityLevel.L7_WORDY_FINE,
            question_type="clevr_spatial_relationship_wordy_control",
        )
        levels[GranularityLevel.L8_WORDY_VERY_FINE] = _wordy_variant(
            l4,
            level=GranularityLevel.L8_WORDY_VERY_FINE,
            question_type=f"{l4.question_type}_wordy_control",
        )
        return levels
    return None


def _cache_key(root: Path, split: str, seed: int, max_samples: int) -> str:
    return f"{_CLEVR_CACHE_VERSION}|root={root.resolve()}|split={split}|seed={seed}|max={max_samples}"


def build_clevr_granularity_dataset(
    *,
    cache_dir: Path,
    max_samples: int = 1000,
    seed: int = 42,
    split: str = "val",
    root_dir: Optional[Path] = None,
    allow_download: bool = False,
) -> List[GranularitySample]:
    """Build a CLEVR same-image L1-L8 granularity dataset.

    The official CLEVR v1.0 archive is large (~18 GB), so this loader expects a
    local extracted root unless users explicitly manage the download outside the
    experiment runner. Expected layout:

    ``CLEVR_v1.0/scenes/CLEVR_val_scenes.json`` and
    ``CLEVR_v1.0/images/val/*.png``.
    """

    split = _split_name(split)
    root = _find_clevr_root(cache_dir, root_override=root_dir)
    if root is None:
        message = (
            "CLEVR_v1.0 not found. Set data.clevr_dir to an extracted CLEVR root "
            "with scenes/ and images/ subdirectories."
        )
        if allow_download:
            logger.warning("%s Automatic CLEVR download is intentionally not triggered because the archive is ~18 GB.", message)
        else:
            logger.warning(message)
        return []
    key = _cache_key(root, split, int(seed), int(max_samples))
    if key in _DATASET_MEMO:
        return list(_DATASET_MEMO[key])

    scenes = _load_scenes(root, split)
    rng = random.Random(seed)
    rng.shuffle(scenes)
    images = _image_dir(root, split)
    samples: List[GranularitySample] = []
    for scene in scenes:
        if len(samples) >= int(max_samples):
            break
        filename = str(scene.get("image_filename") or "")
        if not filename:
            continue
        image_path = images / filename
        if not image_path.exists():
            continue
        image_index = int(scene.get("image_index", len(samples)) or 0)
        levels = _build_levels_for_scene(scene, rng, image_index)
        if levels is None:
            continue
        sample = GranularitySample(
            image_id=f"clevr_{split}_{image_index:06d}",
            image_path=image_path,
            levels=levels,
            dataset="clevr",
            split=split,
            metadata={
                "image_filename": filename,
                "image_index": image_index,
                "num_objects": len(scene.get("objects", []) or []),
                "source": "generated_from_clevr_scene",
            },
        )
        samples.append(sample)
    logger.info("Built %d CLEVR multilevel samples from %s", len(samples), root)
    _DATASET_MEMO[key] = list(samples)
    return samples
