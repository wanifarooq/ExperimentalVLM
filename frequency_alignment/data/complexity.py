"""Semantic complexity helpers for question-conditioned analyses.

The project keeps the original discrete L1-L4 labels, but also assigns each
task a continuous language-complexity score based on the semantic atoms that
the model sees in the prompt. This lets downstream experiments test smooth
relationships such as complexity vs bandwidth or complexity vs robustness.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


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


def build_semantic_complexity(
    *,
    entity_names: Sequence[str] = (),
    attribute_queries: Sequence[str] = (),
    relation_labels: Sequence[str] = (),
    options: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build structured semantic-complexity metadata.

    The primary ``complexity_score`` is based on the full language shown to the
    model during scoring, i.e. question semantics plus option semantics.
    """

    noun_atoms = _entity_atoms(entity_names)
    attribute_atoms = _attribute_atoms(attribute_queries)
    relation_atoms = _relation_atoms(relation_labels)
    option_atoms = _option_atoms(options)

    question_atoms = list(noun_atoms) + list(attribute_atoms) + list(relation_atoms)
    prompt_atoms = question_atoms + list(option_atoms)

    question_score = float(len(question_atoms))
    prompt_score = float(len(prompt_atoms)) if prompt_atoms else question_score

    return {
        "semantic_atoms": question_atoms,
        "prompt_semantic_atoms": prompt_atoms,
        "semantic_atom_counts": {
            "noun_atoms": len(noun_atoms),
            "attribute_atoms": len(attribute_atoms),
            "relation_atoms": len(relation_atoms),
            "option_atoms": len(option_atoms),
            "question_atoms_total": len(question_atoms),
            "prompt_atoms_total": len(prompt_atoms),
        },
        "question_complexity_score": question_score,
        "prompt_complexity_score": prompt_score,
        "complexity_score": prompt_score,
    }


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
            "noun_atoms": 0,
            "attribute_atoms": 0,
            "relation_atoms": 0,
            "option_atoms": len(option_atoms),
            "question_atoms_total": len(question_atoms),
            "prompt_atoms_total": len(prompt_atoms),
        },
        "question_complexity_score": question_score,
        "prompt_complexity_score": prompt_score,
        "complexity_score": prompt_score,
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
