"""
Silero-VAD implementation.

Silero Voice Activity Detector uses neural network for speech detection.
Supports flexible frame sizes (typically 32-96ms) and confidence thresholds.

Reference: https://github.com/snakers4/silero-vad
"""

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .base import VADModel, VADPrediction


class SileroVAD(VADModel):
    """Silero Voice Activity Detector."""

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        window_size_samples: int = 512,  # 32ms at 16kHz
        device: str = "cpu",
    ):
        """
        Initialize Silero-VAD.

        Args:
            threshold: Confidence threshold for speech detection (0.0-1.0)
            sample_rate: Sample rate in Hz (8000 or 16000)
            window_size_samples: Window size in samples (512=32ms, 1536=96ms at 16kHz)
            device: Device to run model on ("cpu" or "cuda")

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

        if sample_rate not in [8000, 16000]:
            raise ValueError(f"sample_rate must be 8000 or 16000, got {sample_rate}")

        self._threshold = threshold
        self._sample_rate = sample_rate
        self._window_size_samples = window_size_samples
        self._device = device

        # Load Silero VAD model
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,  # Use PyTorch model
        )

        self.model.to(device)
        self.model.eval()

        # Extract utility functions
        (self.get_speech_timestamps, _, self.read_audio, *_) = utils

    @property
    def frame_duration_ms(self) -> int:
        """Frame duration in milliseconds."""
        return int(self._window_size_samples / self._sample_rate * 1000)

    @property
    def name(self) -> str:
        """Model name for logging/results."""
        return f"silero_vad_{self.frame_duration_ms}ms_thr{self._threshold:.2f}"

    def predict_frames(self, audio: np.ndarray, sample_rate: int) -> tuple[list[bool], list[float]]:
        """
        Predict speech presence frame-by-frame.

        Args:
            audio: Audio samples (1D numpy array, float32)
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (frame_decisions, frame_probabilities)
            - frame_decisions: List of boolean decisions (True = speech, False = nonspeech)
            - frame_probabilities: List of speech probabilities [0, 1]
        """
        if sample_rate != self._sample_rate:
            raise ValueError(
                f"Sample rate mismatch: expected {self._sample_rate}, got {sample_rate}"
            )

        # Convert to torch tensor
        if not isinstance(audio, torch.Tensor):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        audio_tensor = audio_tensor.to(self._device)

        # Ensure audio is float32 in [-1, 1]
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / 32768.0  # Assume int16 range

        # Pad audio to multiple of window size
        remainder = len(audio_tensor) % self._window_size_samples
        if remainder != 0:
            padding = self._window_size_samples - remainder
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

        # Process frame by frame
        num_frames = len(audio_tensor) // self._window_size_samples
        frame_decisions = []
        frame_probabilities = []

        with torch.no_grad():
            for i in range(num_frames):
                start = i * self._window_size_samples
                end = start + self._window_size_samples
                frame = audio_tensor[start:end]

                # Get speech probability
                speech_prob = self.model(frame, self._sample_rate).item()
                frame_probabilities.append(speech_prob)

                # Apply threshold
                is_speech = speech_prob >= self._threshold
                frame_decisions.append(is_speech)

        return frame_decisions, frame_probabilities

    def predict(self, audio_path: Path) -> VADPrediction:
        """
        Predict speech presence in audio file.

        Args:
            audio_path: Path to audio file (WAV, mono, 16kHz expected)

        Returns:
            VADPrediction with label, confidence, and latency
        """
        start_time = time.time()

        # Load audio
        audio, sr = sf.read(audio_path, dtype="float32")

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self._sample_rate:
            # Simple nearest-neighbor resampling
            ratio = self._sample_rate / sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
            audio = audio[indices]
            sr = self._sample_rate

        # Get frame decisions and probabilities
        frame_decisions, frame_probabilities = self.predict_frames(audio, sr)

        # NEW LOGIC: If ANY frame has speech probability > threshold, classify as SPEECH
        # This is more appropriate for short segments where even a brief speech detection matters
        max_speech_prob = max(frame_probabilities) if frame_probabilities else 0.0
        has_speech_frame = max_speech_prob >= self._threshold

        # For confidence, use the maximum probability found
        label = "SPEECH" if has_speech_frame else "NONSPEECH"
        confidence = max_speech_prob if label == "SPEECH" else (1.0 - max_speech_prob)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        return VADPrediction(
            label=label,
            confidence=confidence,
            latency_ms=latency_ms,
            frame_decisions=frame_decisions,
        )

    def get_speech_segments(
        self,
        audio_path: Path,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> list[tuple[float, float]]:
        """
        Get speech segments using Silero's built-in timestamp detection.

        Args:
            audio_path: Path to audio file
            min_speech_duration_ms: Minimum speech duration in ms
            min_silence_duration_ms: Minimum silence duration in ms

        Returns:
            List of (start_s, end_s) tuples for speech segments
        """
        # Load audio using Silero's utility
        audio = self.read_audio(str(audio_path), sampling_rate=self._sample_rate)

        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=self._sample_rate,
            threshold=self._threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        # Convert to seconds
        segments = [
            (ts["start"] / self._sample_rate, ts["end"] / self._sample_rate)
            for ts in speech_timestamps
        ]

        return segments
