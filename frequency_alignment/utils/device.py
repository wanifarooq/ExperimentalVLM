"""Device selection and dtype utilities."""

from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Select compute device.

    Args:
        requested: "auto", "cpu", "cuda", "cuda:N", or "mps".
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def get_dtype(device: torch.device) -> torch.dtype:
    """Return appropriate dtype for the device."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def gpu_memory_gb(device: torch.device | None = None) -> float:
    """Return total GPU memory in GB, or 0 if not CUDA."""
    if device is None:
        device = select_device()
    if device.type != "cuda":
        return 0.0
    idx = device.index if device.index is not None else 0
    return torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
