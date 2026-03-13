"""Lazy access to reusable helpers from the parent repository."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_PARENT_DIR = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _parent_module():
    if str(_PARENT_DIR) not in sys.path:
        sys.path.insert(0, str(_PARENT_DIR))
    import vlm_invariance_check as parent

    return parent


def prepare_model(*args, **kwargs):
    return _parent_module().prepare_model(*args, **kwargs)


def get_vision_tokens(*args, **kwargs):
    return _parent_module().get_vision_tokens(*args, **kwargs)


def get_context_length_cached(*args, **kwargs):
    return _parent_module().get_context_length_cached(*args, **kwargs)


def score_options_loglik_batch(*args, **kwargs):
    return _parent_module().score_options_loglik_batch(*args, **kwargs)


def dirichlet_energy_from_tokens(*args, **kwargs):
    return _parent_module().dirichlet_energy_from_tokens(*args, **kwargs)
