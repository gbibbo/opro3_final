"""Qwen3-Omni wrapper for binary SPEECH/NONSPEECH classification.

This module provides the same interface as qwen_audio.py but for Qwen3-Omni model.
Requires transformers from GitHub (Qwen3-Omni support not yet in PyPI release).
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
import torch

# Import shared PredictionResult from qwen_audio
from .qwen_audio import PredictionResult

# Import normalization utilities
from ..utils.normalize import llm_fallback_interpret, normalize_to_binary

# Try to import Qwen3-Omni specific classes (from GitHub transformers)
try:
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )
    QWEN3_OMNI_AVAILABLE = True
except ImportError:
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
    QWEN3_OMNI_AVAILABLE = False

# Default model
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_CACHE_DIR = Path("/mnt/fast/nobackup/users/gb0048/.cache/huggingface")


class Qwen3OmniClassifier:
    """
    Qwen3-Omni classifier for binary speech detection.

    Provides the SAME interface as Qwen2AudioClassifier for compatibility
    with the existing OPRO pipeline and evaluation scripts.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        models_cache_dir: Path = DEFAULT_CACHE_DIR,
        **kwargs,  # Accept but ignore extra kwargs for compatibility
    ):
        """
        Initialize Qwen3-Omni classifier.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ("cuda" or "cpu")
            torch_dtype: Model precision ("auto", "bfloat16", "float16", "float32")
            load_in_4bit: Use 4-bit quantization (not recommended for Qwen3-Omni)
            load_in_8bit: Use 8-bit quantization (not recommended for Qwen3-Omni)
            models_cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.device = device
        self.models_cache_dir = Path(models_cache_dir)
        self.name = f"qwen3_omni_{model_name.split('/')[-1].lower()}"

        # Map dtype string to torch dtype
        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, "auto")

        print(f"Loading {model_name}...")
        print(f"  Device: {device}")
        print(f"  Dtype: {torch_dtype}")
        print(f"  Cache dir: {models_cache_dir}")

        # Create cache directory if needed
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)

        # Set HF cache directory
        os.environ["HF_HOME"] = str(self.models_cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.models_cache_dir)

        # Check and install dependencies if needed
        print("Checking dependencies...")
        try:
            import torchvision
            print("  torchvision: OK")
        except ImportError:
            print("  torchvision: MISSING - installing...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", "-q", "torchvision"],
                check=True
            )
            print("  torchvision: INSTALLED")

        # Load processor and model
        try:
            print("Loading processor...")
            if QWEN3_OMNI_AVAILABLE:
                print("  Using Qwen3OmniMoeProcessor (from GitHub transformers)")
                self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                    model_name,
                    cache_dir=self.models_cache_dir
                )
            else:
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=self.models_cache_dir,
                    trust_remote_code=True
                )

            print("Loading model...")
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": "auto",
                "cache_dir": self.models_cache_dir,
            }

            # Handle quantization
            if load_in_4bit or load_in_8bit:
                print("  Warning: Quantization not recommended for Qwen3-Omni")
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                )
                model_kwargs["quantization_config"] = quantization_config

            if QWEN3_OMNI_AVAILABLE:
                print("  Using Qwen3OmniMoeForConditionalGeneration (from GitHub transformers)")
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_name, **model_kwargs
                )
            else:
                model_kwargs["trust_remote_code"] = True
                print("  Attempting with AutoModel + trust_remote_code...")
                try:
                    self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
                except Exception:
                    print("  Attempting with AutoModelForCausalLM...")
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            # Disable audio generation (we only want text output)
            if hasattr(self.model, "disable_talker"):
                print("Disabling audio generation (text-only mode)...")
                self.model.disable_talker()
            else:
                print("Warning: Model does not have disable_talker() method")

            self.model.eval()

        except Exception as e:
            print(f"ERROR: Failed to load {model_name}")
            print(f"  Error: {e}")
            print(f"\nAborting: No fallback allowed. Please ensure:")
            print(f"  1. Sufficient VRAM (needs ~60GB for bf16, ~30GB for int8)")
            print(f"  2. HuggingFace token configured (if model requires auth)")
            print(f"  3. Correct transformers version installed (from GitHub)")
            raise RuntimeError(f"Failed to load {model_name}") from e

        # Default prompts (same as Qwen2)
        self.system_prompt = "You classify audio content."
        self.user_prompt = "Does this audio contain human speech? Answer SPEECH or NONSPEECH."

        print("Model loaded successfully!")

    def predict(
        self,
        audio_path: Path | str,
        decoding_mode: str = "auto",
        return_scores: bool = False,
    ) -> PredictionResult:
        """
        Predict SPEECH or NONSPEECH for an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            decoding_mode: Decoding mode (unused, for Qwen2 compatibility)
            return_scores: Whether to return token probabilities (unused, for compatibility)

        Returns:
            PredictionResult with label, confidence, raw output, and latency
        """
        audio_path = Path(audio_path)

        # Load audio (target 16kHz for Qwen3-Omni)
        audio, sr = sf.read(audio_path)

        # Get target sample rate from processor
        target_sr = getattr(
            self.processor.feature_extractor, "sampling_rate", 16000
        )

        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Start timing
        start_time = time.time()

        # Prepare conversation format (multimodal: audio + text)
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": self.user_prompt},
                ],
            },
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Process inputs
        inputs = self.processor(
            text=text,
            audio=[audio],
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )

        # Move to device and convert to model's dtype
        inputs = inputs.to(self.device)

        # Convert inputs to match model dtype (fix float/bfloat16 mismatch)
        try:
            model_dtype = next(self.model.parameters()).dtype
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].dtype in [
                    torch.float32, torch.float16, torch.float64
                ]:
                    inputs[key] = inputs[key].to(dtype=model_dtype)
        except StopIteration:
            pass

        # Generate response (deterministic for reproducibility)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # Greedy decoding
                temperature=0.0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                return_audio=False,  # CRITICAL: text-only output
            )

        # Decode ONLY the generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        output_text = self.processor.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Parse response (uses same normalization as Qwen2)
        label, confidence = self._parse_response(output_text)

        return PredictionResult(
            label=label,
            confidence=confidence,
            raw_output=output_text,
            latency_ms=latency_ms,
        )

    def _parse_response(self, text: str) -> tuple[str, float]:
        """
        Parse model response to extract label and confidence.

        Uses SAME parsing logic as Qwen2 pipeline (normalize_to_binary).
        """
        # Use existing normalization function
        label, confidence = normalize_to_binary(text)

        # If normalization fails, try LLM fallback
        if label is None:
            label, confidence = llm_fallback_interpret(text)
            if label is None:
                label = "UNKNOWN"
                confidence = 0.0

        return label, confidence

    def set_prompt(self, system_prompt: str | None = None, user_prompt: str | None = None):
        """
        Update the system and/or user prompts.

        Args:
            system_prompt: New system prompt (optional)
            user_prompt: New user prompt (optional)
        """
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if user_prompt is not None:
            self.user_prompt = user_prompt