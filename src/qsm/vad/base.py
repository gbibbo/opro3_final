"""
Base interface for VAD models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VADPrediction:
    """VAD prediction result."""

    label: str  # "SPEECH" or "NONSPEECH"
    confidence: float  # 0.0-1.0
    latency_ms: float  # Processing time in milliseconds
    frame_decisions: list[bool] | None = None  # Per-frame decisions (optional)


class VADModel(ABC):
    """Abstract base class for VAD models."""

    @abstractmethod
    def predict(self, audio_path: Path) -> VADPrediction:
        """
        Predict speech presence in audio file.

        Args:
            audio_path: Path to audio file (WAV, mono, 16kHz expected)

        Returns:
            VADPrediction with label, confidence, and latency
        """
        pass

    @abstractmethod
    def predict_frames(self, audio: np.ndarray, sample_rate: int) -> list[bool]:
        """
        Predict speech presence frame-by-frame.

        Args:
            audio: Audio samples (1D numpy array)
            sample_rate: Sample rate in Hz

        Returns:
            List of boolean decisions (True = speech, False = nonspeech) per frame
        """
        pass

    @property
    @abstractmethod
    def frame_duration_ms(self) -> int:
        """Frame duration in milliseconds."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging/results."""
        pass
