from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

LAYER_GROUP_ORDER = ("early", "mid", "late")

T = TypeVar("T")


def layer_group_ranges(total_layers: int) -> Dict[str, Tuple[int, int]]:
    if total_layers <= 0:
        return {name: (0, 0) for name in LAYER_GROUP_ORDER}

    first_split = max(1, total_layers // 3)
    second_split = max(first_split + 1, (2 * total_layers) // 3)
    second_split = min(second_split, total_layers)

    return {
        "early": (0, first_split),
        "mid": (first_split, second_split),
        "late": (second_split, total_layers),
    }


def layer_group_for_index(layer_index: int, total_layers: int) -> str:
    ranges = layer_group_ranges(total_layers)
    for group_name in LAYER_GROUP_ORDER:
        start, end = ranges[group_name]
        if start <= layer_index < end:
            return group_name
    return "late"


def split_by_layer_group(
    values: Sequence[T],
    layer_indices: Sequence[int],
    total_layers: int,
) -> Dict[str, Dict[str, List]]:
    grouped: Dict[str, Dict[str, List[T] | List[int]]] = {
        group_name: {"values": [], "layer_indices": []}
        for group_name in LAYER_GROUP_ORDER
    }

    for value, layer_index in zip(values, layer_indices):
        group_name = layer_group_for_index(int(layer_index), total_layers)
        grouped[group_name]["values"].append(value)
        grouped[group_name]["layer_indices"].append(int(layer_index))

    return grouped


def resolve_post_fusion_hidden_state_index(
    num_hidden_states: int,
    explicit_index: Optional[int] = None,
    fraction: float = 0.8,
) -> int:
    if num_hidden_states <= 0:
        return 0

    last_index = num_hidden_states - 1
    if explicit_index is not None:
        if explicit_index < 0:
            return max(0, num_hidden_states + explicit_index)
        return min(last_index, max(0, explicit_index))

    if num_hidden_states == 1:
        return 0

    fraction = min(max(fraction, 0.0), 1.0)
    decoder_layers = max(1, num_hidden_states - 1)
    return min(last_index, max(1, round(decoder_layers * fraction)))
