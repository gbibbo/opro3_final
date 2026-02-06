"""
Qwen Speech Minimum (QSM)
Temporal threshold measurement and optimization for speech detection in Qwen models.
"""

__version__ = "0.1.0"

from pathlib import Path

import yaml

# Load global configuration
_config_path = Path(__file__).parent.parent.parent / "config.yaml"
with open(_config_path) as f:
    CONFIG = yaml.safe_load(f)

# Quick access to prototype mode setting
PROTOTYPE_MODE = CONFIG.get("PROTOTYPE_MODE", True)
PROTOTYPE_SAMPLES = CONFIG.get("PROTOTYPE_SAMPLES", 5)

__all__ = ["CONFIG", "PROTOTYPE_MODE", "PROTOTYPE_SAMPLES"]
