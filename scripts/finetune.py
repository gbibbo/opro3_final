"""Fine-tune Qwen2-Audio for speech detection using LoRA/QLoRA.

This script fine-tunes the model on clean audio clips to improve
speech detection accuracy, especially in noisy conditions.

Includes Silero VAD filtering to remove low-speech-ratio clips from training.
"""

# Disable hf_transfer BEFORE any imports - must be set before transformers import
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa
from tqdm import tqdm
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Silero VAD Filtering (uses src/qsm/vad/silero.py module)
# ============================================================================

# Import the original SileroVAD implementation from opro2
try:
    from src.qsm.vad.silero import SileroVAD
    SILERO_VAD_AVAILABLE = True
except ImportError as e:
    print(f"  WARNING: Could not import SileroVAD module: {e}")
    SILERO_VAD_AVAILABLE = False


def load_silero_vad(threshold: float = 0.5, device: str = "cpu") -> Optional["SileroVAD"]:
    """
    Load Silero VAD model using the original implementation.

    Args:
        threshold: Confidence threshold for speech detection
        device: Device to run model on ("cpu" or "cuda")

    Returns:
        SileroVAD instance or None if unavailable
    """
    if not SILERO_VAD_AVAILABLE:
        print("  WARNING: SileroVAD module not available.")
        return None

    try:
        vad = SileroVAD(threshold=threshold, device=device)
        print(f"  Silero VAD loaded: {vad.name}")
        return vad
    except Exception as e:
        print(f"  WARNING: Could not load Silero VAD: {e}")
        return None


def compute_speech_ratio(
    audio_path: Path,
    vad: "SileroVAD",
    min_speech_duration_ms: int = 50,
    min_silence_duration_ms: int = 30,
) -> float:
    """
    Compute speech ratio for an audio file using Silero VAD.

    Uses the original SileroVAD implementation which loads audio with soundfile.

    Args:
        audio_path: Path to audio file
        vad: SileroVAD instance
        min_speech_duration_ms: Minimum speech segment duration
        min_silence_duration_ms: Minimum silence duration

    Returns:
        Speech ratio (0.0 to 1.0), or -1.0 on error
    """
    try:
        # Verify file exists
        if not Path(audio_path).exists():
            if not hasattr(compute_speech_ratio, '_missing_printed'):
                print(f"    [VAD ERROR] File not found: {audio_path}")
                compute_speech_ratio._missing_printed = True
            return -1.0

        # Get speech segments using the original implementation
        segments = vad.get_speech_segments(
            audio_path,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        # Load audio to get total duration (using soundfile like the original)
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        total_duration = len(audio) / sr

        if total_duration == 0:
            return 0.0

        # Calculate speech duration from segments
        speech_duration = sum(end - start for start, end in segments)
        speech_ratio = speech_duration / total_duration

        return float(speech_ratio)

    except Exception as e:
        # Return -1.0 to indicate error (print first error for debugging)
        if not hasattr(compute_speech_ratio, '_error_printed'):
            print(f"    [VAD ERROR] First error: {e}")
            print(f"    [VAD ERROR] Audio path: {audio_path}")
            compute_speech_ratio._error_printed = True
        return -1.0


def filter_dataframe_by_vad(
    df: pd.DataFrame,
    audio_column: str = "audio_path",
    min_speech_ratio: float = 0.8,
    data_root: Optional[Path] = None,
    label_column: str = "ground_truth",
) -> pd.DataFrame:
    """
    Filter DataFrame by Silero VAD speech ratio.

    Only applies filtering to SPEECH samples (NONSPEECH samples are kept as-is).

    Args:
        df: DataFrame with audio paths
        audio_column: Column name containing audio paths
        min_speech_ratio: Minimum speech ratio to keep (default: 0.8 per methodology)
        data_root: Root directory for relative audio paths
        label_column: Column containing ground truth labels

    Returns:
        Filtered DataFrame
    """
    print(f"\n{'='*60}")
    print("SILERO VAD FILTERING")
    print(f"{'='*60}")
    print(f"  Min speech ratio: {min_speech_ratio}")
    print(f"  Total samples before filtering: {len(df)}")

    # Load VAD model (uses original SileroVAD class from src/qsm/vad/silero.py)
    vad = load_silero_vad(threshold=0.5, device="cpu")

    if vad is None:
        print("  VAD model not available, skipping filtering.")
        return df

    # Separate SPEECH and NONSPEECH samples
    speech_mask = df[label_column] == "SPEECH"
    speech_df = df[speech_mask].copy()
    nonspeech_df = df[~speech_mask].copy()

    print(f"  SPEECH samples to filter: {len(speech_df)}")
    print(f"  NONSPEECH samples (kept as-is): {len(nonspeech_df)}")

    # Calculate speech ratios for SPEECH samples only
    speech_ratios = []
    print("  Computing speech ratios for SPEECH samples...")

    for _, row in tqdm(speech_df.iterrows(), total=len(speech_df), desc="  VAD"):
        path_str = str(row[audio_column])

        # Resolve path
        if path_str.startswith('/'):
            audio_path = Path(path_str)
        elif data_root:
            audio_path = data_root / path_str
        else:
            audio_path = project_root / path_str

        ratio = compute_speech_ratio(audio_path, vad)
        speech_ratios.append(ratio)

    speech_df["_vad_speech_ratio"] = speech_ratios

    # Filter by speech ratio
    valid_speech_mask = speech_df["_vad_speech_ratio"] >= min_speech_ratio
    error_mask = speech_df["_vad_speech_ratio"] < 0

    n_valid = valid_speech_mask.sum()
    n_low_ratio = (~valid_speech_mask & ~error_mask).sum()
    n_errors = error_mask.sum()

    print(f"\n  SPEECH samples passing VAD filter: {n_valid}")
    print(f"  SPEECH samples discarded (low ratio): {n_low_ratio}")
    print(f"  SPEECH samples with errors: {n_errors}")

    # Keep valid SPEECH samples and all NONSPEECH samples
    filtered_speech_df = speech_df[valid_speech_mask].drop(columns=["_vad_speech_ratio"])
    result_df = pd.concat([filtered_speech_df, nonspeech_df], ignore_index=True)

    # Shuffle to mix SPEECH and NONSPEECH
    result_df = result_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"\n  Final dataset size: {len(result_df)}")
    print(f"    SPEECH:    {(result_df[label_column] == 'SPEECH').sum()}")
    print(f"    NONSPEECH: {(result_df[label_column] == 'NONSPEECH').sum()}")
    print(f"{'='*60}\n")

    return result_df


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""

    # Model
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    use_4bit: bool = True

    # LoRA (paper configuration: r=64, alpha=16)
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Will default to attention + MLP

    # Training
    num_epochs: int = 3
    batch_size: int = 2  # Reduced from 4 to avoid OOM
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size of 16
    learning_rate: float = 5e-5  # Paper configuration
    warmup_steps: int = 100
    seed: int = 42  # Random seed for reproducibility

    # Paths
    train_csv: Path = project_root / "data" / "processed" / "normalized_clips" / "train_metadata.csv"
    test_csv: Path = project_root / "data" / "processed" / "normalized_clips" / "test_metadata.csv"
    output_dir: Path = project_root / "checkpoints" / "qwen2_audio_speech_detection_normalized"

    # Prompt
    system_prompt: str = "You classify audio content."
    user_prompt: str = (
        "Choose one:\n"
        "A) SPEECH (human voice)\n"
        "B) NONSPEECH (music/noise/silence/animals)\n\n"
        "Answer with A or B ONLY."
    )


class SpeechDetectionDataset(Dataset):
    """Dataset for speech detection fine-tuning."""

    def __init__(
        self,
        metadata_csv: Optional[Path],
        processor,
        system_prompt: str,
        user_prompt: str,
        max_audio_length: int = 30,  # seconds
        filter_duration: Optional[int] = None,  # ms
        filter_snr: Optional[float] = None,  # dB
        data_root: Optional[Path] = None,  # Root for relative audio paths
        dataframe: Optional[pd.DataFrame] = None,  # Pre-filtered DataFrame
    ):
        # Load from CSV or use provided DataFrame
        if dataframe is not None:
            self.df = dataframe.copy()
            source_name = "pre-filtered DataFrame"
        else:
            self.df = pd.read_csv(metadata_csv)
            source_name = metadata_csv.name

        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_audio_length = max_audio_length
        self.data_root = data_root if data_root else project_root

        print(f"  Loaded {len(self.df)} samples from {source_name}")

        # Apply filters if specified
        if filter_duration is not None:
            if 'duration_ms' in self.df.columns:
                orig_len = len(self.df)
                self.df = self.df[self.df['duration_ms'] == filter_duration]
                print(f"    Filtered by duration={filter_duration}ms: {orig_len} -> {len(self.df)}")
            else:
                print(f"    WARNING: filter_duration specified but 'duration_ms' column not found")

        if filter_snr is not None:
            if 'snr_db' in self.df.columns:
                orig_len = len(self.df)
                self.df = self.df[self.df['snr_db'] == filter_snr]
                print(f"    Filtered by SNR={filter_snr}dB: {orig_len} -> {len(self.df)}")
            else:
                print(f"    WARNING: filter_snr specified but 'snr_db' column not found")

        print(f"  Final dataset size: {len(self.df)}")
        print(f"    SPEECH:    {(self.df['ground_truth'] == 'SPEECH').sum()}")
        print(f"    NONSPEECH: {(self.df['ground_truth'] == 'NONSPEECH').sum()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load audio - handle both absolute and relative paths
        path_str = str(row['audio_path'])

        if path_str.startswith('/'):
            # Absolute path - use as-is
            audio_path = Path(path_str)
        else:
            # Relative path - use data_root
            audio_path = self.data_root / path_str
        audio, sr = sf.read(audio_path)

        # Resample if needed
        target_sr = self.processor.feature_extractor.sampling_rate
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Truncate if too long
        max_samples = int(self.max_audio_length * target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Ensure audio is float32 and 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono
        audio = audio.astype('float32')

        # Prepare conversation
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": self.user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": "A" if row['ground_truth'] == 'SPEECH' else "B",
            },
        ]

        # Get answer token for finding the offset
        answer_text = "A" if row['ground_truth'] == 'SPEECH' else "B"

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )

        # Process inputs
        # IMPORTANT: Pass sampling_rate explicitly to avoid silent errors
        inputs = self.processor(
            text=text,
            audio=[audio],
            sampling_rate=target_sr,  # Must pass explicitly for WhisperFeatureExtractor
            return_tensors="pt",
            padding=True,
        )

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Create labels with masking: only compute loss on the assistant's response token
        # This is much more efficient than computing loss on the entire prompt
        labels = inputs["input_ids"].clone()

        # Find the position of the assistant's response
        # Strategy: Find where the answer token appears near the end
        answer_token_ids = self.processor.tokenizer.encode(answer_text, add_special_tokens=False)

        # Search from the end backwards for the answer token
        input_ids_list = labels.tolist()
        assistant_response_start = -1

        # Look for the answer token in the last 20 tokens (should be at the very end)
        for i in range(len(input_ids_list) - 1, max(len(input_ids_list) - 20, -1), -1):
            if len(answer_token_ids) == 1 and input_ids_list[i] == answer_token_ids[0]:
                assistant_response_start = i
                break
            elif len(answer_token_ids) > 1 and i + len(answer_token_ids) <= len(input_ids_list):
                if input_ids_list[i:i+len(answer_token_ids)] == answer_token_ids:
                    assistant_response_start = i
                    break

        # Mask everything before the assistant's response with -100
        if assistant_response_start > 0:
            labels[:assistant_response_start] = -100

        inputs["labels"] = labels

        return inputs


def collate_fn(batch):
    """Collate function for DataLoader."""
    # Find max lengths
    max_input_len = max(item["input_ids"].shape[0] for item in batch)

    # For audio features, they should already be padded to 3000 by the processor
    # We just need to ensure they're all the same shape
    max_audio_len = 3000  # Fixed size for Qwen2-Audio

    # Check if all items have input_features
    has_audio = all("input_features" in item for item in batch)

    # Pad each item
    padded_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    if has_audio:
        padded_batch["input_features"] = []
        padded_batch["feature_attention_mask"] = []

    for item in batch:
        # Pad text tokens
        input_len = item["input_ids"].shape[0]
        pad_len = max_input_len - input_len

        padded_batch["input_ids"].append(
            torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=0)
        )
        padded_batch["attention_mask"].append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        padded_batch["labels"].append(
            torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100)
        )

        # Handle audio features if present
        if has_audio:
            # Audio features should be [128, 3000]
            audio_features = item["input_features"]
            feature_mask = item["feature_attention_mask"]

            # Pad if needed (should already be 3000 but just in case)
            if audio_features.shape[-1] < max_audio_len:
                pad_len = max_audio_len - audio_features.shape[-1]
                audio_features = torch.nn.functional.pad(audio_features, (0, pad_len), value=0.0)
                feature_mask = torch.nn.functional.pad(feature_mask, (0, pad_len), value=0)

            padded_batch["input_features"].append(audio_features)
            padded_batch["feature_attention_mask"].append(feature_mask)

    # Stack
    return {k: torch.stack(v) for k, v in padded_batch.items()}


def main():
    import argparse
    import random
    import numpy as np

    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-Audio")

    # Existing arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank (paper: 64)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (paper: 16)")
    parser.add_argument("--add_mlp_targets", action="store_true", help="Add MLP layers to LoRA targets")

    # NEW arguments that the pipeline is passing
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training metadata CSV")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation metadata CSV")
    parser.add_argument("--filter_duration", type=int, default=None, help="Filter by duration in ms (e.g., 1000). If None, no filtering.")
    parser.add_argument("--filter_snr", type=float, default=None, help="Filter by SNR in dB (e.g., 20). If None, no filtering.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (paper: 5e-5)")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for audio files (if paths in CSV are relative)")

    # VAD filtering (Silero)
    parser.add_argument(
        "--min_speech_ratio",
        type=float,
        default=0.8,
        help="Minimum speech ratio (Silero VAD) to keep SPEECH samples. Set to 0 to disable. (paper: 0.8)"
    )
    parser.add_argument(
        "--skip_vad_filter",
        action="store_true",
        help="Skip Silero VAD filtering entirely (faster, but may include low-quality samples)"
    )

    args = parser.parse_args()

    config = TrainingConfig()

    # Apply arguments to config
    config.seed = args.seed
    config.output_dir = Path(args.output_dir)
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.train_csv = Path(args.train_csv)
    config.test_csv = Path(args.val_csv)  # Note: using val_csv for test_csv (evaluation set)
    config.num_epochs = args.num_epochs
    config.batch_size = args.per_device_train_batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.learning_rate = args.learning_rate

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    print("=" * 80)
    print("QWEN2-AUDIO FINE-TUNING FOR SPEECH DETECTION")
    print("=" * 80)

    # Check if clean dataset exists
    if not config.train_csv.exists():
        print(f"\n❌ Training data not found: {config.train_csv}")
        print("   Run 'python scripts/create_clean_dataset.py' first!")
        return

    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  4-bit quantization: {config.use_4bit}")
    print(f"  LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Random seed: {config.seed}")

    # Load processor
    print(f"\nLoading processor...")
    processor = AutoProcessor.from_pretrained(config.model_name)

    # Resolve data_root
    data_root = Path(args.data_root) if args.data_root else None
    if data_root:
        print(f"  Data root: {data_root}")

    # Apply Silero VAD filtering to training data
    train_df = pd.read_csv(config.train_csv)
    if not args.skip_vad_filter and args.min_speech_ratio > 0:
        train_df = filter_dataframe_by_vad(
            train_df,
            audio_column="audio_path",
            min_speech_ratio=args.min_speech_ratio,
            data_root=data_root,
            label_column="ground_truth",
        )
        # Save filtered CSV for reference
        filtered_csv_path = config.output_dir / "train_filtered_vad.csv"
        config.output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(filtered_csv_path, index=False)
        print(f"  Saved filtered training data to: {filtered_csv_path}")
    else:
        print(f"\n  Skipping VAD filtering (skip_vad_filter={args.skip_vad_filter}, min_speech_ratio={args.min_speech_ratio})")

    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = SpeechDetectionDataset(
        metadata_csv=None,  # Not used when dataframe is provided
        processor=processor,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        filter_duration=args.filter_duration,
        filter_snr=args.filter_snr,
        data_root=data_root,
        dataframe=train_df,  # Use pre-filtered DataFrame
    )

    eval_dataset = SpeechDetectionDataset(
        config.test_csv,
        processor,
        config.system_prompt,
        config.user_prompt,
        filter_duration=args.filter_duration,
        filter_snr=args.filter_snr,
        data_root=data_root,
    )

    # Load model with quantization
    print(f"\nLoading model...")
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_config = None

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if not config.use_4bit else None,
    )

    # Prepare model for training
    print(f"\nPreparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing with use_reentrant=False (recommended by PyTorch 2.4+)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Apply LoRA
    print(f"\nApplying LoRA...")

    # Configure target modules
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    if args.add_mlp_targets:
        # Add MLP layers for increased capacity
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        print(f"  Adding MLP targets: gate_proj, up_proj, down_proj")

    print(f"  Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    config.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Trainer
    print(f"\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    # Train
    print(f"\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    trainer.train()

    # Save final model
    print(f"\nSaving final model...")
    final_dir = config.output_dir / "final"
    trainer.save_model(str(final_dir))

    print(f"\n[OK] Fine-tuning complete!")
    print(f"  Model saved to: {final_dir}")
    print("\nNext steps:")
    print(f"  1. Test the model: python scripts/test_finetuned_model.py")
    print(f"  2. Evaluate on full test set")
    print("=" * 80)


if __name__ == "__main__":
    main()
