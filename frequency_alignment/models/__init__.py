from .base import HookOutputs, VLMAdapter
from .llava_adapter import LLaVAAdapter
from .qwen_adapter import HFVLMAdapter


def get_adapter(model_id: str) -> VLMAdapter:
    """Return the appropriate adapter for the given model ID."""
    model_lower = model_id.lower()
    if "llava" in model_lower:
        return LLaVAAdapter()
    if any(name in model_lower for name in ("qwen", "internvl")):
        return HFVLMAdapter()
    return HFVLMAdapter()
