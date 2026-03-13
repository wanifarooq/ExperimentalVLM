from .base import HookOutputs, VLMAdapter
from .qwen_adapter import HFVLMAdapter


def get_adapter(model_id: str) -> VLMAdapter:
    """Return the appropriate adapter for the given model ID."""
    model_lower = model_id.lower()
    if any(name in model_lower for name in ("qwen", "llava", "internvl")):
        return HFVLMAdapter()
    return HFVLMAdapter()
