from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..data.base import ALL_VQA_LEVEL_NAMES
from .io import load_json
from .layer_groups import LAYER_GROUP_ORDER

# Groups whose per-sample filters live under ``record.levels.<level>.layer_groups``
# in ``power_spectra.json``. ``last_2`` (the 2nd-from-last attention layer) is
# stored by Exp 2 alongside the early/mid/late tertile and needs to be loaded
# here for Exp 5's overlap law to evaluate it.
_LOAD_GROUPS = LAYER_GROUP_ORDER + ("last_2",)


def load_exp2_filter_bank(
    exp2_out_dir: Path,
    *,
    norm: str = "l1",
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[Tuple[str, str], np.ndarray]],
]:
    average_filters: Dict[str, Dict[str, np.ndarray]] = {"overall": {}}
    sample_filters: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {"overall": {}}
    for group_name in _LOAD_GROUPS:
        average_filters[group_name] = {}
        sample_filters[group_name] = {}

    filters_dir = exp2_out_dir / "filters"
    level_keys = list(ALL_VQA_LEVEL_NAMES)
    suffix = "_l2" if norm == "l2" else ""
    for level_key in level_keys:
        overall_path = filters_dir / f"average_{level_key}{suffix}.npy"
        if overall_path.exists():
            average_filters["overall"][level_key] = np.load(overall_path)
        for group_name in _LOAD_GROUPS:
            group_path = filters_dir / f"average_{level_key}_{group_name}{suffix}.npy"
            if group_path.exists():
                average_filters[group_name][level_key] = np.load(group_path)

    power_path = exp2_out_dir / "power_spectra.json"
    if not power_path.exists():
        return average_filters, sample_filters

    for record in load_json(power_path):
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            w_t = level_data.get("W_t_l2" if norm == "l2" else "W_t")
            if w_t:
                sample_filters["overall"][(image_id, level_key)] = np.asarray(w_t, dtype=np.float64)
            for group_name in _LOAD_GROUPS:
                group_w_t = level_data.get("layer_groups", {}).get(group_name, {}).get(
                    "W_t_l2" if norm == "l2" else "W_t"
                )
                if group_w_t:
                    sample_filters[group_name][(image_id, level_key)] = np.asarray(
                        group_w_t,
                        dtype=np.float64,
                    )

    return average_filters, sample_filters
