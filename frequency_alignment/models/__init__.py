from .base import VLMAdapter, HookOutputs
from .qwen_adapter import QwenVLAdapter


def get_adapter(model_id: str) -> VLMAdapter:
    """Return the appropriate adapter for the given model ID."""
    model_lower = model_id.lower()
    if "qwen" in model_lower:
        return QwenVLAdapter()
    if "llava" in model_lower:
        # Future: return LlavaAdapter()
        raise NotImplementedError(f"LLaVA adapter not yet implemented for {model_id}")
    if "internvl" in model_lower:
        # Future: return InternVLAdapter()
        raise NotImplementedError(f"InternVL adapter not yet implemented for {model_id}")
    # Default to Qwen adapter (most general with AutoModel)
    return QwenVLAdapter()
