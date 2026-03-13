"""Dataset selection helpers for the experiment runners."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .base import GranularitySample
from .gqa import build_granularity_dataset
from .partimagenet import build_partimagenet_dataset
from .seedbench_adapter import load_seedbench_as_granularity


def load_multilevel_vqa_dataset(
    cfg: dict,
    *,
    max_samples: int,
    require_all_levels: bool = True,
) -> List[GranularitySample]:
    """Load the configured VQA dataset for experiments 1-5."""
    data_cfg = cfg.get("data", {})
    dataset = str(data_cfg.get("dataset", "gqa")).lower()
    cache_dir = Path(cfg.get("cache_dir", ".hf_cache"))
    seed = int(cfg.get("seed", 42))
    allow_download = bool(data_cfg.get("allow_download", True))

    if dataset == "gqa":
        return build_granularity_dataset(
            cache_dir=cache_dir,
            max_samples=max_samples,
            seed=seed,
            allow_download=allow_download,
        )

    if dataset == "seedbench":
        samples = load_seedbench_as_granularity(
            tsv_path=Path(data_cfg["seedbench_tsv"]) if data_cfg.get("seedbench_tsv") else None,
            image_root=Path(data_cfg["seedbench_image_root"]) if data_cfg.get("seedbench_image_root") else None,
            max_samples=max_samples,
            cache_dir=cache_dir,
        )
        if require_all_levels:
            raise ValueError(
                "SEEDBench currently provides L4-only samples. "
                "Experiments 1-5 require a same-image multi-granularity dataset such as GQA."
            )
        return samples

    raise ValueError(
        f"Unsupported dataset for VQA experiments: {dataset!r}. "
        "Use 'gqa' for experiments 1-5."
    )


def load_segmentation_dataset(
    cfg: dict,
    *,
    max_samples: int,
    dataset_override: str | None = None,
) -> List[GranularitySample]:
    """Load the configured segmentation dataset for experiment 6."""
    data_cfg = cfg.get("data", {})
    dataset = str(dataset_override or data_cfg.get("dataset", "partimagenet")).lower()
    cache_dir = Path(cfg.get("cache_dir", ".hf_cache"))
    seed = int(cfg.get("seed", 42))

    if dataset != "partimagenet":
        raise ValueError(
            f"Unsupported dataset for segmentation experiments: {dataset!r}. "
            "Use 'partimagenet' for experiment 6."
        )

    return build_partimagenet_dataset(
        cache_dir=cache_dir,
        max_samples=max_samples,
        seed=seed,
        root_dir=Path(data_cfg["partimagenet_dir"]) if data_cfg.get("partimagenet_dir") else None,
    )
