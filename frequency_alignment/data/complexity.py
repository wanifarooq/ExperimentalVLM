"""Semantic complexity helpers for question-conditioned analyses.

The project keeps the original discrete L1-L4 labels (plus optional controls such as L5),
but also assigns each
task a continuous language-complexity score based on the semantic atoms that
the model sees in the prompt. This lets downstream experiments test smooth
relationships such as complexity vs bandwidth or complexity vs robustness.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence


SEMANTIC_COMPLEXITY_SCORE_NAME = "complexity_score"
SEMANTIC_COMPLEXITY_SCORE_LABEL = "Semantic Program Complexity"
SEMANTIC_COMPLEXITY_SCORE_DEFINITION = (
    "structured semantic-program complexity with grounding ambiguity"
)
SEMANTIC_COMPLEXITY_SCORE_FORMULA = (
    "entity refs + attribute refs + relation refs + reasoning ops + "
    "program depth + log1p(grounding candidates)"
)
PROMPT_COMPLEXITY_SCORE_NAME = "prompt_complexity_score"
PROMPT_COMPLEXITY_SCORE_LABEL = "Prompt Load Control"
PROMPT_COMPLEXITY_SCORE_DEFINITION = (
    "prompt load proxy from question and option content tokens"
)
OPTION_HARDNESS_SCORE_NAME = "option_hardness_score"
OPTION_HARDNESS_SCORE_LABEL = "Option Hardness Control"
OPTION_HARDNESS_SCORE_DEFINITION = (
    "MCQ distractor hardness: same-category distractors plus a scene-plausibility tie-break"
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_TEXT_STOPWORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "there",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "image",
    "picture",
    "photo",
    "of",
}
_BOOLEAN_OPTIONS = {"yes", "no", "true", "false"}
_RELATION_FILLERS = {"a", "an", "the", "to", "of", "is", "are"}


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in _WORD_RE.findall(text or "")]


def _entity_atoms(entity_names: Sequence[str]) -> List[str]:
    atoms: List[str] = []
    for name in entity_names:
        atoms.extend(token for token in _tokens(name) if token not in _TEXT_STOPWORDS)
    return atoms


def _attribute_atoms(attribute_queries: Sequence[str]) -> List[str]:
    atoms: List[str] = []
    for query in attribute_queries:
        atoms.extend(token for token in _tokens(query) if token not in _TEXT_STOPWORDS)
    return atoms


def _relation_atoms(relation_labels: Sequence[str]) -> List[str]:
    atoms: List[str] = []
    for relation in relation_labels:
        relation_tokens = [token for token in _tokens(relation) if token not in _RELATION_FILLERS]
        if relation_tokens:
            atoms.append("_".join(relation_tokens))
    return atoms


def _option_atoms(options: Optional[Mapping[str, str]]) -> List[str]:
    atoms: List[str] = []
    if not options:
        return atoms
    for option_text in options.values():
        option_tokens = [
            token for token in _tokens(option_text)
            if token not in _TEXT_STOPWORDS and token not in _BOOLEAN_OPTIONS
        ]
        atoms.extend(option_tokens)
    return atoms


def _question_load_atoms(question_text: str) -> List[str]:
    return [
        token
        for token in _tokens(question_text)
        if token not in _TEXT_STOPWORDS
    ]


def build_semantic_complexity(
    *,
    entity_names: Sequence[str] = (),
    attribute_queries: Sequence[str] = (),
    relation_labels: Sequence[str] = (),
    reasoning_ops: Sequence[str] = (),
    program_depth: float = 0.0,
    grounding_candidate_count: float = 0.0,
    question_text: str = "",
    options: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build structured semantic-complexity metadata.

    The primary ``complexity_score`` is a structured semantic-program score.
    Prompt/load complexity is kept separately in ``prompt_complexity_score``.
    """

    noun_atoms = list(entity_names)
    attribute_atoms = list(attribute_queries)
    relation_atoms = list(relation_labels)
    operator_atoms = [str(op).strip().lower() for op in reasoning_ops if str(op).strip()]
    question_atoms = (
        [f"entity:{name}" for name in _entity_atoms(entity_names)]
        + [f"attribute:{attr}" for attr in _attribute_atoms(attribute_queries)]
        + [f"relation:{rel}" for rel in _relation_atoms(relation_labels)]
        + [f"op:{op}" for op in operator_atoms]
    )
    entity_ref_count = len(noun_atoms)
    attribute_ref_count = len(attribute_atoms)
    relation_ref_count = len(relation_atoms)
    operator_count = len(operator_atoms)
    program_depth = max(0.0, float(program_depth or 0.0))
    grounding_candidate_count = max(0.0, float(grounding_candidate_count or 0.0))
    grounding_ambiguity = float(math.log1p(grounding_candidate_count))
    semantic_score = float(
        entity_ref_count
        + attribute_ref_count
        + relation_ref_count
        + operator_count
        + program_depth
        + grounding_ambiguity
    )

    question_load_atoms = _question_load_atoms(question_text)
    option_atoms = _option_atoms(options)
    prompt_atoms = question_load_atoms + list(option_atoms)
    prompt_score = float(len(prompt_atoms)) if prompt_atoms else float(len(question_load_atoms))

    return {
        "semantic_atoms": question_atoms,
        "prompt_semantic_atoms": prompt_atoms,
        "semantic_atom_counts": {
            "entity_refs": entity_ref_count,
            "attribute_refs": attribute_ref_count,
            "relation_refs": relation_ref_count,
            "reasoning_ops": operator_count,
            "program_depth": int(program_depth),
            "grounding_candidate_count": grounding_candidate_count,
            "grounding_ambiguity": grounding_ambiguity,
            "option_atoms": len(option_atoms),
            "question_atoms_total": len(question_atoms),
            "prompt_atoms_total": len(prompt_atoms),
        },
        "question_complexity_score": semantic_score,
        "prompt_complexity_score": prompt_score,
        "complexity_score": semantic_score,
    }


def refresh_prompt_load(
    existing_complexity: Mapping[str, Any],
    *,
    question_text: str,
    options: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Keep semantic-program complexity fixed while recomputing prompt load."""

    payload = dict(existing_complexity)
    question_load_atoms = _question_load_atoms(question_text)
    option_atoms = _option_atoms(options)
    prompt_atoms = question_load_atoms + list(option_atoms)
    counts = dict(payload.get("semantic_atom_counts", {}) or {})
    counts["option_atoms"] = len(option_atoms)
    counts["prompt_atoms_total"] = len(prompt_atoms)
    payload["prompt_semantic_atoms"] = prompt_atoms
    payload["semantic_atom_counts"] = counts
    payload["prompt_complexity_score"] = (
        float(len(prompt_atoms)) if prompt_atoms else float(len(question_load_atoms))
    )
    return payload


def fallback_text_complexity(
    question: str,
    options: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Fallback complexity for datasets without structured semantic metadata."""

    question_atoms = [token for token in _tokens(question) if token not in _TEXT_STOPWORDS]
    option_atoms = _option_atoms(options)
    prompt_atoms = list(question_atoms) + list(option_atoms)
    question_score = float(len(question_atoms))
    prompt_score = float(len(prompt_atoms)) if prompt_atoms else question_score
    return {
        "semantic_atoms": question_atoms,
        "prompt_semantic_atoms": prompt_atoms,
        "semantic_atom_counts": {
            "entity_refs": 0,
            "attribute_refs": 0,
            "relation_refs": 0,
            "reasoning_ops": 0,
            "program_depth": 0,
            "grounding_candidate_count": 0.0,
            "grounding_ambiguity": 0.0,
            "option_atoms": len(option_atoms),
            "question_atoms_total": len(question_atoms),
            "prompt_atoms_total": len(prompt_atoms),
        },
        "question_complexity_score": question_score,
        "prompt_complexity_score": prompt_score,
        "complexity_score": question_score,
    }


def ensure_level_complexity(level_data: Any) -> Dict[str, Any]:
    """Return a complete complexity payload for a ``LevelData``-like object."""

    if (
        getattr(level_data, "complexity_score", 0.0) > 0
        or getattr(level_data, "question_complexity_score", 0.0) > 0
        or getattr(level_data, "prompt_complexity_score", 0.0) > 0
        or getattr(level_data, "semantic_atoms", None)
    ):
        return {
            "semantic_atoms": list(getattr(level_data, "semantic_atoms", []) or []),
            "prompt_semantic_atoms": list(
                getattr(level_data, "prompt_semantic_atoms", []) or []
            ),
            "semantic_atom_counts": dict(
                getattr(level_data, "semantic_atom_counts", {}) or {}
            ),
            "question_complexity_score": float(
                getattr(level_data, "question_complexity_score", 0.0) or 0.0
            ),
            "prompt_complexity_score": float(
                getattr(level_data, "prompt_complexity_score", 0.0) or 0.0
            ),
            "complexity_score": float(getattr(level_data, "complexity_score", 0.0) or 0.0),
        }
    return fallback_text_complexity(
        getattr(level_data, "question", ""),
        getattr(level_data, "options", None),
    )
