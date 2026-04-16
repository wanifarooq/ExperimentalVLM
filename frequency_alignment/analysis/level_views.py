"""Shared level-view definitions for primary, wordy, and pooled analyses."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set

from ..data.base import PRIMARY_VQA_LEVEL_NAMES, WORDY_CONTROL_VQA_LEVELS

WORDY_VQA_LEVEL_NAMES = [level.name for level in WORDY_CONTROL_VQA_LEVELS]

LEVEL_VIEWS: Dict[str, Optional[Set[str]]] = {
    "primary": set(PRIMARY_VQA_LEVEL_NAMES),
    "wordy": set(WORDY_VQA_LEVEL_NAMES),
    "pooled": None,
}

LEVEL_VIEW_ORDER = ("primary", "wordy", "pooled")


def level_in_filter(level: Any, level_filter: Optional[Set[str]]) -> bool:
    """Return whether a row level belongs to the requested analysis view."""

    if level_filter is None:
        return True
    return str(level) in level_filter


def filter_rows_by_level(
    rows: Iterable[Dict[str, Any]],
    level_filter: Optional[Set[str]],
    *,
    level_key: str = "level",
) -> list[Dict[str, Any]]:
    """Filter dictionaries by level while preserving pooled compatibility."""

    if level_filter is None:
        return list(rows)
    return [row for row in rows if level_in_filter(row.get(level_key), level_filter)]
