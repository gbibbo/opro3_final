"""
Voice Activity Detection (VAD) baseline.

Implements Silero-VAD neural network model for binary speech/nonspeech classification.
Optimized for segments ≥100ms with 95-100% accuracy on clean audio.
"""

from .base import VADModel, VADPrediction
from .silero import SileroVAD

__all__ = [
    "VADModel",
    "VADPrediction",
    "SileroVAD",
]
