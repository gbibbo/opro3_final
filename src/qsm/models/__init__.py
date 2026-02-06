"""Model wrappers for speech detection."""

from .qwen_audio import Qwen2AudioClassifier

# Qwen3-Omni requires transformers from GitHub - import conditionally
try:
    from .qwen3_omni import Qwen3OmniClassifier
    __all__ = ["Qwen2AudioClassifier", "Qwen3OmniClassifier"]
except ImportError:
    __all__ = ["Qwen2AudioClassifier"]
