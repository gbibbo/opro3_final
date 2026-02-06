#!/usr/bin/env python3
"""
OPRO Classic: Prompt Optimization with Local LLM

PURPOSE:
    Optimize prompts for Qwen2-Audio speech detection using OPRO algorithm.
    Part of pipeline block E (OPRO on base model) and G (OPRO on fine-tuned model).

INPUTS:
    - Manifest parquet with audio paths and labels (conditions_manifest_split.parquet)
    - Split name (dev for optimization, test for final eval)
    - Optional: LoRA checkpoint path for fine-tuned model

OUTPUTS:
    - Best prompt JSON (best_prompt.json)
    - Optimization history (opro_history.json)
    - Per-iteration metrics

CLUSTER vs LOCAL:
    - Requires GPU (A100 recommended, 8GB+ VRAM for 4-bit quantization)
    - Run via slurm/opro_classic_base.job or slurm/opro_classic_lora.job

Key features:
- Uses LOCAL LLM (Qwen2.5 / Qwen2-Audio / Llama) to generate prompts
- Optimizes reward: R = w_clip * BA_clip + w_cond * BA_conditions (NO length penalty)
- Evaluates prompts with Qwen2-Audio-7B-Instruct (base or with LoRA)
- Includes robust prompt sanitization and circuit breaker for error handling

Usage:
    # Base model (no fine-tuning)
    python scripts/opro_classic_optimize.py \\
        --manifest data/processed/conditions_final/conditions_manifest_split.parquet \\
        --split dev \\
        --output_dir results/opro_classic \\
        --no_lora

    # With LoRA checkpoint
    python scripts/opro_classic_optimize.py \\
        --manifest data/processed/conditions_final/conditions_manifest_split.parquet \\
        --split dev \\
        --output_dir results/opro_classic_lora \\
        --checkpoint checkpoints/qwen_lora_seed42/final
"""

import argparse
import gc
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.qsm.models.qwen_audio import Qwen2AudioClassifier
from src.qsm.utils.normalize import normalize_to_binary, llm_fallback_interpret

# Qwen3-Omni (requires transformers from GitHub)
try:
    from src.qsm.models.qwen3_omni import Qwen3OmniClassifier
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PromptCandidate:
    """A candidate prompt with its evaluation results."""

    prompt: str
    reward: float
    ba_clip: float
    ba_conditions: float
    prompt_length: int
    iteration: int
    timestamp: float
    metrics: dict


# ============================================================================
# Local LLM for Prompt Generation
# ============================================================================


class LocalLLMGenerator:
    """Local LLM for generating prompt candidates."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 2000,
        temperature: float = 0.7,
    ):
        """
        Initialize local LLM.

        Args:
            model_name: HuggingFace model name
            device: Device to use
            load_in_4bit: Use 4-bit quantization
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading local LLM: {model_name}...")
        print(f"  Device: {device}")
        print(f"  4-bit quantization: {load_in_4bit}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        model_kwargs = {"torch_dtype": torch.float16}

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = device if device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if device == "cpu" and not load_in_4bit:
            self.model = self.model.to(device)

        self.model.eval()

        print("Local LLM loaded successfully!")

    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # Format as chat if model supports it
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        generated_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return generated_text


# ============================================================================
# Prompt Sanitization and Validation
# ============================================================================


def sanitize_prompt(prompt: str) -> tuple[str, bool]:
    """
    Sanitize and validate prompt candidate.

    Returns:
        (cleaned_prompt, is_valid)
    """
    # Remove any audio special tokens
    forbidden_tokens = [
        "<|audio_bos|>",
        "<|AUDIO|>",
        "<|audio_eos|>",
        "<|im_start|>",
        "<|im_end|>",
        "<audio>",
        "</audio>",
    ]

    cleaned = prompt.strip()

    for token in forbidden_tokens:
        if token in cleaned:
            print(f"      ⚠️  Rejected: Contains forbidden token '{token}'")
            return cleaned, False

    # Check length
    if len(cleaned) < 10:
        print(f"      ⚠️  Rejected: Too short ({len(cleaned)} chars)")
        return cleaned, False

    if len(cleaned) > 300:
        print(f"      ⚠️  Rejected: Too long ({len(cleaned)} chars)")
        return cleaned, False

    # REMOVED: Keyword restriction to allow open-ended prompts
    # The normalize_to_binary() function handles various response formats including:
    # - Binary labels (SPEECH/NONSPEECH)
    # - Yes/No responses
    # - Synonyms (voice, talking, music, noise, etc.)
    # - Open descriptions

    # Remove multiple spaces and newlines
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()

    return cleaned, True


# ============================================================================
# OPRO Classic Optimizer
# ============================================================================


class OPROClassicOptimizer:
    """
    OPRO optimizer using LOCAL LLM.

    Uses transformers to run LLM locally on GPU.
    No API keys required.
    """

    def __init__(
        self,
        optimizer_llm: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        top_k: int = 10,
        candidates_per_iter: int = 3,
        reward_weights: dict = None,
        seed: int = 42,
        baseline_prompt: str = None,
        initial_prompts: list[str] = None,
        max_new_tokens: int = 2000,
        temperature: float = 0.7,
    ):
        """
        Initialize OPRO optimizer with local LLM.

        Args:
            optimizer_llm: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
            device: Device to use
            load_in_4bit: Use 4-bit quantization
            top_k: Number of best prompts to keep in memory
            candidates_per_iter: Number of candidates to generate per iteration
            reward_weights: Reward function weights
            seed: Random seed
            baseline_prompt: Initial baseline prompt
            initial_prompts: Optional list of initial prompts to seed memory
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        """
        self.optimizer_llm = optimizer_llm
        self.top_k = top_k
        self.candidates_per_iter = candidates_per_iter
        self.seed = seed

        # Default reward weights (NO length penalty - reward based purely on accuracy)
        if reward_weights is None:
            reward_weights = {
                "ba_clip": 1.0,
                "ba_cond": 0.25,
            }
        self.reward_weights = reward_weights

        # Initialize local LLM
        self.llm = LocalLLMGenerator(
            model_name=optimizer_llm,
            device=device,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Top-k memory
        self.memory: list[PromptCandidate] = []
        self.history: list[PromptCandidate] = []

        # Baseline prompt
        if baseline_prompt is None:
            baseline_prompt = (
                "Does this audio contain human speech?\n"
                "Reply with ONLY one word: SPEECH or NON-SPEECH."
            )
        self.baseline_prompt = baseline_prompt
        self.baseline_reward = None

        # Initial prompts (if provided)
        self.initial_prompts = initial_prompts or []

        print("\nOPRO Classic Optimizer initialized:")
        print(f"  LLM: {optimizer_llm}")
        print(f"  Device: {device}")
        print(f"  Top-k: {top_k}")
        print(f"  Candidates/iter: {candidates_per_iter}")
        print(f"  Reward weights: {reward_weights}")
        print(f"  Seed: {seed}")

    def compute_reward(self, ba_clip: float, ba_conditions: float, prompt_length: int) -> float:
        """Compute reward for a prompt.

        Reward is based purely on accuracy metrics (BA_clip and BA_conditions).
        No length penalty is applied - prompt length is constrained via sanitize_prompt()
        as a hard constraint (max 300 chars) rather than as part of the optimization objective.

        Args:
            ba_clip: Balanced accuracy at clip level (0-1)
            ba_conditions: Macro-average BA across psychoacoustic conditions (0-1)
            prompt_length: Character count (unused, kept for API compatibility)

        Returns:
            Reward score = w_clip * BA_clip + w_cond * BA_conditions
        """
        reward = (
            self.reward_weights["ba_clip"] * ba_clip
            + self.reward_weights["ba_cond"] * ba_conditions
        )
        return reward

    def build_meta_prompt(self, iteration: int) -> str:
        """Build meta-prompt for LLM."""
        sorted_memory = sorted(self.memory, key=lambda x: x.reward, reverse=True)

        history_str = ""
        for i, candidate in enumerate(sorted_memory[: self.top_k], 1):
            # Show only clean text (no special tokens)
            clean_prompt = candidate.prompt.replace(
                "<|audio_bos|><|AUDIO|><|audio_eos|>", ""
            ).strip()
            history_str += f"\n{i}. Reward={candidate.reward:.4f} | BA_clip={candidate.ba_clip:.3f} | BA_cond={candidate.ba_conditions:.3f}\n"
            history_str += f'   "{clean_prompt}"\n'

        meta_prompt = f"""TASK: Optimize prompts for audio speech detection (Qwen2-Audio-7B-Instruct).
The model receives audio and must determine if it contains human speech or not.

OBJECTIVE: Maximize performance on psychoacoustic degradations:
- Short durations (20-200ms clips)
- Low SNR (-10 to 0 dB, noise masked speech)
- Band-pass filtered audio (telephony, low-pass, high-pass)
- Reverberant audio (T60: 0-1.5s)

REWARD FUNCTION:
R = BA_clip + 0.25 × BA_conditions
(No length penalty - prompt length constrained to max 300 chars as hard limit)
where:
- BA_clip: Balanced accuracy at clip level (primary metric, range 0-1)
- BA_conditions: Macro-average BA across all psychoacoustic conditions
- len(prompt): Character count (penalty for verbosity)

BASELINE PROMPT:
Prompt: "{self.baseline_prompt}"
Reward: {f"{self.baseline_reward:.4f}" if self.baseline_reward is not None else "evaluating..."}

CURRENT ITERATION: {iteration}

TOP-{self.top_k} PROMPTS:{history_str}

INSTRUCTIONS:
Generate {self.candidates_per_iter} NEW prompt candidates that:
1. Are clear and concise (target <150 chars, absolute max 300 chars)
2. Encourage robust detection on SHORT and NOISY clips
3. Use simple, direct language (model is instruction-tuned)
4. Build on insights from top prompts above
5. Explore DIVERSE prompt formats:
   - Direct questions (e.g., "What do you hear?", "Describe the audio")
   - Binary choice prompts (e.g., "SPEECH or NONSPEECH?")
   - Command style (e.g., "Classify this audio")
   - Open-ended queries (e.g., "What type of sound is this?")
6. Consider emphasizing: brevity detection, noise robustness, voice/speech keywords

CONSTRAINTS:
- Each prompt must be COMPLETE and STANDALONE (no placeholders)
- Prompts can be ANY format: questions, commands, statements, binary choice, open-ended
- The model's response will be automatically parsed to determine speech detection
- Avoid overly complex or multi-step instructions
- Prompts must be plain text (NO special tokens, NO markup)

OUTPUT FORMAT (exactly {self.candidates_per_iter} prompts, one per line):
PROMPT_1: <your complete prompt here>
PROMPT_2: <your complete prompt here>
PROMPT_3: <your complete prompt here>
{f"PROMPT_{self.candidates_per_iter}: <your complete prompt here>" if self.candidates_per_iter > 3 else ""}

Generate the prompts now:"""

        return meta_prompt

    def generate_candidates(self, iteration: int) -> list[str]:
        """Use local LLM to generate new prompt candidates."""
        meta_prompt = self.build_meta_prompt(iteration)

        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Generating {self.candidates_per_iter} candidates...")
        print(f"{'='*60}")

        # Generate from local LLM
        llm_output = self.llm.generate(meta_prompt)

        # Parse and sanitize candidates
        candidates = self._parse_and_sanitize_candidates(llm_output)

        print(f"Generated {len(candidates)} valid candidates:")
        for i, prompt in enumerate(candidates, 1):
            print(f"  {i}. {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        return candidates

    def _parse_and_sanitize_candidates(self, llm_output: str) -> list[str]:
        """Parse candidates from LLM output and sanitize them."""
        candidates_raw = []
        lines = llm_output.strip().split("\n")

        # Try structured parsing first
        for line in lines:
            if "PROMPT_" in line and ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    prompt = parts[1].strip().strip('"').strip("'")
                    if prompt:
                        candidates_raw.append(prompt)

        # Fallback: split by double newline
        if len(candidates_raw) == 0:
            chunks = llm_output.split("\n\n")
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk and len(chunk) > 10 and "PROMPT" not in chunk:
                    candidates_raw.append(chunk)

        # Sanitize all candidates
        candidates_clean = []
        print(f"\n  Parsing {len(candidates_raw)} raw candidates...")

        for i, prompt_raw in enumerate(candidates_raw, 1):
            print(f"    Candidate {i}: {prompt_raw[:60]}{'...' if len(prompt_raw) > 60 else ''}")
            cleaned, is_valid = sanitize_prompt(prompt_raw)

            if is_valid:
                candidates_clean.append(cleaned)
                print("      ✓ Valid")

            if len(candidates_clean) >= self.candidates_per_iter:
                break

        # Fallback if no valid candidates generated
        if len(candidates_clean) == 0:
            print("\n  ⚠️  WARNING: No valid candidates generated!")
            print("  Falling back to baseline variations...")
            candidates_clean = [
                "Does this audio contain human speech? Answer: SPEECH or NON-SPEECH.",
                "Is there speech in this audio? Reply: SPEECH or NON-SPEECH.",
                "Audio classification: SPEECH or NON-SPEECH?",
            ][: self.candidates_per_iter]

        return candidates_clean[: self.candidates_per_iter]

    def update_memory(self, candidate: PromptCandidate):
        """Add candidate to memory and history."""
        self.history.append(candidate)
        self.memory.append(candidate)
        self.memory = sorted(self.memory, key=lambda x: x.reward, reverse=True)[: self.top_k]
        print(
            f"Memory updated: {len(self.memory)} prompts, best reward={self.memory[0].reward:.4f}"
        )

    def save_state(self, output_dir: Path):
        """Save optimizer state."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save history
        history_path = output_dir / "opro_prompts.jsonl"
        with open(history_path, "w") as f:
            for candidate in self.history:
                f.write(json.dumps(asdict(candidate)) + "\n")

        # Save memory
        memory_path = output_dir / "opro_memory.json"
        with open(memory_path, "w") as f:
            json.dump([asdict(c) for c in self.memory], f, indent=2)

        # Save best prompt
        if len(self.memory) > 0:
            best_prompt_path = output_dir / "best_prompt.txt"
            with open(best_prompt_path, "w") as f:
                f.write(self.memory[0].prompt)

            best_metrics_path = output_dir / "best_metrics.json"
            with open(best_metrics_path, "w") as f:
                json.dump(asdict(self.memory[0]), f, indent=2)

        # Save reward history
        rewards = [c.reward for c in self.history]
        iterations = [c.iteration for c in self.history]
        history_summary = {
            "iterations": iterations,
            "rewards": rewards,
            "best_reward_per_iteration": [],
        }

        best_so_far = float("-inf")
        for it in sorted(set(iterations)):
            iter_candidates = [c for c in self.history if c.iteration == it]
            max_reward = max([c.reward for c in iter_candidates])
            best_so_far = max(best_so_far, max_reward)
            history_summary["best_reward_per_iteration"].append(best_so_far)

        history_summary_path = output_dir / "opro_history.json"
        with open(history_summary_path, "w") as f:
            json.dump(history_summary, f, indent=2)

        print(f"\nSaved state to: {output_dir}")

    def run_optimization(
        self,
        evaluator_fn,
        n_iterations: int = 30,
        early_stopping_patience: int = 5,
        output_dir: Path = None,
    ) -> PromptCandidate:
        """Run OPRO optimization loop."""
        print(f"\n{'='*60}")
        print("STARTING OPRO CLASSIC OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Iterations: {n_iterations}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Output dir: {output_dir}")

        # Evaluate baseline
        if self.baseline_reward is None:
            print("\nEvaluating baseline prompt...")
            ba_clip, ba_cond, metrics = evaluator_fn(self.baseline_prompt)
            self.baseline_reward = self.compute_reward(ba_clip, ba_cond, len(self.baseline_prompt))

            baseline_candidate = PromptCandidate(
                prompt=self.baseline_prompt,
                reward=self.baseline_reward,
                ba_clip=ba_clip,
                ba_conditions=ba_cond,
                prompt_length=len(self.baseline_prompt),
                iteration=0,
                timestamp=time.time(),
                metrics=metrics,
            )
            self.update_memory(baseline_candidate)
            print(f"Baseline reward: {self.baseline_reward:.4f}")

        # Evaluate initial prompts if provided
        for i, prompt in enumerate(self.initial_prompts):
            print(f"\nEvaluating initial prompt {i+1}/{len(self.initial_prompts)}...")
            ba_clip, ba_cond, metrics = evaluator_fn(prompt)
            reward = self.compute_reward(ba_clip, ba_cond, len(prompt))

            candidate = PromptCandidate(
                prompt=prompt,
                reward=reward,
                ba_clip=ba_clip,
                ba_conditions=ba_cond,
                prompt_length=len(prompt),
                iteration=0,
                timestamp=time.time(),
                metrics=metrics,
            )
            self.update_memory(candidate)

        best_reward = self.memory[0].reward
        no_improvement_count = 0

        for iteration in range(1, n_iterations + 1):
            # Generate candidates
            candidates = self.generate_candidates(iteration)

            # Evaluate each candidate with circuit breaker
            for i, prompt in enumerate(candidates, 1):
                print(f"\nEvaluating candidate {i}/{len(candidates)}...")
                print(f"Prompt: {prompt[:150]}{'...' if len(prompt) > 150 else ''}")

                try:
                    ba_clip, ba_cond, metrics = evaluator_fn(prompt)
                    reward = self.compute_reward(ba_clip, ba_cond, len(prompt))

                    candidate = PromptCandidate(
                        prompt=prompt,
                        reward=reward,
                        ba_clip=ba_clip,
                        ba_conditions=ba_cond,
                        prompt_length=len(prompt),
                        iteration=iteration,
                        timestamp=time.time(),
                        metrics=metrics,
                    )

                    self.update_memory(candidate)
                    print(
                        f"Results: BA_clip={ba_clip:.3f}, BA_cond={ba_cond:.3f}, Reward={reward:.4f}"
                    )

                except Exception as e:
                    print(f"  ✗ ERROR evaluating candidate: {e}")
                    print("  Skipping this candidate...")
                    continue

            # Check for improvement
            current_best_reward = self.memory[0].reward
            if current_best_reward > best_reward:
                improvement = current_best_reward - best_reward
                print(f"\nNEW BEST REWARD: {current_best_reward:.4f} (+{improvement:.4f})")
                best_reward = current_best_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(
                    f"\nNo improvement (patience: {no_improvement_count}/{early_stopping_patience})"
                )

            # Save state
            if output_dir:
                self.save_state(output_dir)

            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                print(f"\nEarly stopping: No improvement for {early_stopping_patience} iterations")
                break

        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total iterations: {iteration}")
        print(f"Best reward: {self.memory[0].reward:.4f}")
        print(f"Best BA_clip: {self.memory[0].ba_clip:.3f}")
        print(f"Best prompt: {self.memory[0].prompt}")

        return self.memory[0]


# ============================================================================
# Evaluation Functions
# ============================================================================


def build_evaluator_from_args(args):
    """Build evaluator model from command-line arguments.

    Returns either Qwen2AudioClassifier or Qwen3OmniClassifier based on --model_type.
    """
    model_type = getattr(args, "model_type", "qwen2")

    if model_type == "qwen3_omni":
        # Qwen3-Omni model
        if not QWEN3_AVAILABLE:
            raise RuntimeError(
                "Qwen3-Omni not available. Install transformers from GitHub:\n"
                "pip install git+https://github.com/huggingface/transformers.git"
            )

        # Use default model name if not overridden
        model_name = args.evaluator_model_name
        if model_name == "Qwen/Qwen2-Audio-7B-Instruct":
            model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

        print(f"\nLoading Qwen3-Omni model: {model_name}...")
        print(f"  Device: {args.evaluator_device}")
        print("  NOTE: Qwen3-Omni does not support LoRA or 4-bit quantization")

        model = Qwen3OmniClassifier(
            model_name=model_name,
            device=args.evaluator_device,
        )

        # Warn if LoRA was requested
        if not args.no_lora and args.checkpoint is not None:
            print("  WARNING: LoRA not supported for Qwen3-Omni, ignoring checkpoint")

        return model

    else:
        # Qwen2-Audio model (default)
        print(f"\nLoading evaluator model: {args.evaluator_model_name}...")
        print(f"  Device: {args.evaluator_device}")
        print("  4-bit quantization: True")

        model = Qwen2AudioClassifier(
            model_name=args.evaluator_model_name,
            device=args.evaluator_device,
            load_in_4bit=True,
        )

        if not args.no_lora and args.checkpoint is not None:
            print(f"  Loading LoRA checkpoint: {args.checkpoint}")
            from peft import PeftModel

            model.model = PeftModel.from_pretrained(model.model, args.checkpoint)
            model.model.eval()
            print("  LoRA checkpoint loaded!")

        return model


def make_evaluator_fn(args, evaluator_model):
    """
    Create evaluator function that evaluates a prompt on the dev/test set.

    Args:
        args: Command-line arguments
        evaluator_model: Qwen2AudioClassifier instance

    Returns:
        Function that takes a prompt and returns (ba_clip, ba_cond, metrics)
    """
    # Load manifest (support both CSV and Parquet)
    manifest_path = Path(args.manifest)
    if manifest_path.suffix == ".csv":
        manifest_df = pd.read_csv(args.manifest)
        # Normalize column names if using experimental_variants CSV
        if "ground_truth" in manifest_df.columns and "label" not in manifest_df.columns:
            manifest_df["label"] = manifest_df["ground_truth"]
    else:
        manifest_df = pd.read_parquet(args.manifest)

    # Fix Windows-style paths (backslashes) to Unix-style
    if 'audio_path' in manifest_df.columns:
        manifest_df['audio_path'] = manifest_df['audio_path'].str.replace('\\', '/', regex=False)

    # Filter by split if split column exists
    if "split" in manifest_df.columns:
        split_df = manifest_df[manifest_df["split"] == args.split].copy()
    else:
        split_df = manifest_df.copy()

    print("\nDataset loaded:")
    print(f"  Manifest: {args.manifest}")
    print(f"  Split: {args.split}")
    print(f"  Samples: {len(split_df)}")

    # Resolve and validate audio paths
    def resolve_audio_path(p):
        """Resolve audio path, handling relative paths from repo root."""
        import os
        p_str = str(p).replace("\\", "/")

        # If absolute and exists, return it
        if os.path.isabs(p_str) and os.path.isfile(p_str):
            return p_str

        # Try as-is from current directory
        if os.path.isfile(p_str):
            return os.path.abspath(p_str)

        # If starts with "processed/", prepend "data/"
        if p_str.startswith("processed/"):
            candidate = os.path.join("data", p_str)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

        # Try from repo root
        candidate = os.path.join(os.getcwd(), p_str)
        if os.path.isfile(candidate):
            return candidate

        # Last resort: return original for error reporting
        return p_str

    split_df["audio_resolved"] = split_df["audio_path"].map(resolve_audio_path)

    # Validate that files exist
    existing_mask = split_df["audio_resolved"].map(lambda x: Path(x).is_file())
    exist_ratio = existing_mask.mean()

    print(f"  Files found: {exist_ratio:.1%} ({existing_mask.sum()}/{len(split_df)})")

    if exist_ratio < 0.95:
        print(f"\n  ⚠️  WARNING: Only {exist_ratio:.1%} of audio files found!")
        print("  Missing files (first 5):")
        for p in split_df[~existing_mask]["audio_path"].head(5):
            print(f"    - {p}")
        raise RuntimeError(
            f"Only {exist_ratio:.1%} of audio files exist. "
            "Check manifest paths and ensure data is accessible."
        )

    # Keep only existing files
    split_df = split_df[existing_mask].copy()
    print(f"  Proceeding with {len(split_df)} valid samples")

    # Apply sampling strategy if max_eval_samples is set
    def create_eval_subset(df, max_samples, strategy, per_condition_k, seed):
        """Create evaluation subset based on sampling strategy."""
        if max_samples <= 0 or max_samples >= len(df):
            return df

        n = min(max_samples, len(df))
        print(f"\n  Sampling {n} examples (strategy: {strategy})...")

        if strategy == "uniform":
            eval_df = df.sample(n=n, random_state=seed)

        elif strategy == "stratified":
            # Stratified sampling by ground_truth class
            if "label" not in df.columns:
                print("    WARNING: 'label' column not found, falling back to uniform")
                eval_df = df.sample(n=n, random_state=seed)
            else:
                cls_counts = df["label"].value_counts(normalize=True)
                parts = []
                for cls, frac in cls_counts.items():
                    k = max(1, int(round(n * frac)))
                    cls_df = df[df["label"] == cls]
                    sample_k = min(k, len(cls_df))
                    parts.append(cls_df.sample(n=sample_k, random_state=seed))
                eval_df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
                if len(eval_df) > n:
                    eval_df = eval_df.head(n)

        else:  # per_condition
            # Sample k examples per condition bucket
            bucket_cols = [c for c in ["duration_ms", "snr_db", "band_filter", "T60", "label"]
                          if c in df.columns]
            if not bucket_cols:
                bucket_cols = ["label"] if "label" in df.columns else []

            if not bucket_cols:
                print("    WARNING: No bucket columns found, falling back to uniform")
                eval_df = df.sample(n=n, random_state=seed)
            else:
                eval_df = (
                    df.groupby(bucket_cols, group_keys=False)
                    .apply(lambda g: g.sample(n=min(per_condition_k, len(g)), random_state=seed))
                )
                eval_df = eval_df.sample(frac=1.0, random_state=seed)
                if len(eval_df) > n:
                    eval_df = eval_df.head(n)

        print(f"    Sampled {len(eval_df)} examples")
        if "label" in eval_df.columns:
            class_dist = eval_df["label"].value_counts()
            print(f"    Class distribution: {class_dist.to_dict()}")

        return eval_df

    # Store full dataset and sampling params for evaluator closure
    full_split_df = split_df.copy()
    max_eval_samples = args.max_eval_samples
    sample_strategy = args.sample_strategy
    per_condition_k = args.per_condition_k
    eval_seed = args.seed

    def evaluator_fn(prompt: str) -> tuple[float, float, dict]:
        """
        Evaluate prompt on dataset.

        Returns:
            (ba_clip, ba_cond, metrics)
        """
        # Update model prompt
        evaluator_model.user_prompt = prompt

        # Create evaluation subset (may be full dataset if max_eval_samples=0)
        eval_df = create_eval_subset(
            full_split_df, max_eval_samples, sample_strategy, per_condition_k, eval_seed
        )

        # Evaluate on samples
        results = []
        for _, row in tqdm(
            eval_df.iterrows(), total=len(eval_df), desc="  Evaluating", leave=False
        ):
            audio_path = row["audio_resolved"]
            ground_truth = row["label"]

            try:
                result = evaluator_model.predict(audio_path, return_scores=True)

                # Normalize prediction
                normalized_label, confidence = normalize_to_binary(
                    result.raw_output,
                    probs=result.probs,
                    mode="auto",
                    verbalizers=["SPEECH", "NONSPEECH"],
                )

                # LLM fallback for ambiguous responses
                if normalized_label is None:
                    fallback_label, fallback_conf = llm_fallback_interpret(result.raw_output)
                    if fallback_label is not None:
                        normalized_label = fallback_label
                        confidence = fallback_conf
                    else:
                        # Last resort: use model's raw prediction
                        normalized_label = result.label

                is_correct = (normalized_label == ground_truth) if normalized_label else False

                # Create condition string from variant_type and corresponding value
                variant_type = row.get("variant_type", "unknown")

                # For standard single-dimension variants, use specific condition strings
                if variant_type == "duration":
                    condition = f"dur_{row.get('duration_ms', 'unk')}ms"
                elif variant_type == "snr":
                    condition = f"snr_{row.get('snr_db', 'unk')}dB"
                elif variant_type == "band":
                    condition = f"filter_{row.get('band_filter', 'unk')}"
                elif variant_type == "rir":
                    condition = f"reverb_T60_{row.get('T60', 'unk')}s"
                else:
                    # For "combo" or unknown variant_type, derive from duration_ms and snr_db
                    dm = row.get("duration_ms", None)
                    sn = row.get("snr_db", None)
                    if pd.notna(dm) and pd.notna(sn):
                        condition = f"dur_{int(dm)}_snr_{int(sn)}"
                    else:
                        condition = "unknown"

                results.append(
                    {
                        "audio_path": str(audio_path),
                        "ground_truth": ground_truth,
                        "prediction": normalized_label,
                        "correct": is_correct,
                        "condition": condition,
                        "variant_type": variant_type,
                        # NUEVO: dimensiones psicoacústicas explícitas
                        "duration_ms": row.get("duration_ms", None),
                        "snr_db": row.get("snr_db", None),
                        "band_filter": row.get("band_filter", None),
                        "T60": row.get("T60", None),
                    }
                )

            except Exception as e:
                print(f"  Error processing {audio_path}: {e}")
                continue

        # Calculate metrics
        results_df = pd.DataFrame(results)

        # Empty safeguard
        if len(results_df) == 0:
            return 0.0, 0.0, {}

        # ------------------------------------------------------------------
        # 1) Balanced accuracy at clip level
        # ------------------------------------------------------------------
        speech_samples = results_df[results_df["ground_truth"] == "SPEECH"]
        nonspeech_samples = results_df[results_df["ground_truth"] == "NONSPEECH"]

        speech_acc = speech_samples["correct"].mean() if len(speech_samples) > 0 else 0.0
        nonspeech_acc = nonspeech_samples["correct"].mean() if len(nonspeech_samples) > 0 else 0.0
        ba_clip = (speech_acc + nonspeech_acc) / 2.0

        # ------------------------------------------------------------------
        # 2) Balanced accuracy por condición compuesta (string 'condition')
        # ------------------------------------------------------------------
        condition_bas = []
        condition_metrics = {}

        for condition in results_df["condition"].unique():
            cond_df = results_df[results_df["condition"] == condition]
            cond_speech = cond_df[cond_df["ground_truth"] == "SPEECH"]
            cond_nonspeech = cond_df[cond_df["ground_truth"] == "NONSPEECH"]

            cond_speech_acc = cond_speech["correct"].mean() if len(cond_speech) > 0 else 0.0
            cond_nonspeech_acc = (
                cond_nonspeech["correct"].mean() if len(cond_nonspeech) > 0 else 0.0
            )
            cond_ba = (cond_speech_acc + cond_nonspeech_acc) / 2.0

            condition_bas.append(cond_ba)
            condition_metrics[str(condition)] = {
                "ba": float(cond_ba),
                "speech_acc": float(cond_speech_acc),
                "nonspeech_acc": float(cond_nonspeech_acc),
                "n_samples": int(len(cond_df)),
            }

        ba_conditions_old = float(np.mean(condition_bas)) if len(condition_bas) > 0 else 0.0

        # ------------------------------------------------------------------
        # 3) Helper para BA por columna (duración, SNR, filtro, reverb)
        # ------------------------------------------------------------------
        def compute_ba_by_column(df: pd.DataFrame, col: str):
            if col not in df.columns:
                return 0.0, {}

            values = [v for v in df[col].unique() if pd.notna(v)]
            if not values:
                return 0.0, {}

            bas = []
            metrics_by_value = {}

            for v in values:
                sub = df[df[col] == v]
                speech = sub[sub["ground_truth"] == "SPEECH"]
                nonspeech = sub[sub["ground_truth"] == "NONSPEECH"]

                speech_acc_v = speech["correct"].mean() if len(speech) > 0 else 0.0
                nonspeech_acc_v = nonspeech["correct"].mean() if len(nonspeech) > 0 else 0.0
                ba_v = (speech_acc_v + nonspeech_acc_v) / 2.0

                bas.append(ba_v)
                metrics_by_value[str(v)] = {
                    "ba": float(ba_v),
                    "speech_acc": float(speech_acc_v),
                    "nonspeech_acc": float(nonspeech_acc_v),
                    "n_samples": int(len(sub)),
                }

            return float(np.mean(bas)), metrics_by_value

        # ------------------------------------------------------------------
        # 4) BA por dimensión psicoacústica
        # ------------------------------------------------------------------
        ba_duration, duration_metrics = compute_ba_by_column(results_df, "duration_ms")
        ba_snr, snr_metrics = compute_ba_by_column(results_df, "snr_db")
        ba_filter, filter_metrics = compute_ba_by_column(results_df, "band_filter")
        ba_reverb, reverb_metrics = compute_ba_by_column(results_df, "T60")

        # ------------------------------------------------------------------
        # 5) BA_conditions = promedio de las 4 dimensiones independientes
        #    (NO combinaciones cruzadas)
        # ------------------------------------------------------------------
        dimension_bas = []
        if duration_metrics:
            dimension_bas.append(ba_duration)
        if snr_metrics:
            dimension_bas.append(ba_snr)
        if filter_metrics:
            dimension_bas.append(ba_filter)
        if reverb_metrics:
            dimension_bas.append(ba_reverb)

        # Si tenemos al menos una dimensión, usamos el promedio de dimensiones
        # Si no, fallback al cálculo antiguo por condiciones cruzadas
        if dimension_bas:
            ba_conditions = float(np.mean(dimension_bas))
        else:
            ba_conditions = ba_conditions_old

        # ------------------------------------------------------------------
        # 6) Empaquetar todas las métricas en un dict
        # ------------------------------------------------------------------
        metrics = {
            "ba_clip": float(ba_clip),
            "ba_conditions": float(ba_conditions),
            "ba_conditions_old": float(ba_conditions_old),  # promedio condiciones cruzadas (deprecated)
            "n_dimensions": len(dimension_bas),  # cuántas dimensiones se usaron
            "ba_duration": float(ba_duration),
            "ba_snr": float(ba_snr),
            "ba_filter": float(ba_filter),
            "ba_reverb": float(ba_reverb),
            "speech_acc": float(speech_acc),
            "nonspeech_acc": float(nonspeech_acc),
            "n_samples": int(len(results_df)),
            "condition_metrics": condition_metrics,
            "duration_metrics": duration_metrics,
            "snr_metrics": snr_metrics,
            "filter_metrics": filter_metrics,
            "reverb_metrics": reverb_metrics,
        }

        return ba_clip, ba_conditions, metrics

    return evaluator_fn


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(0.5)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser("OPRO Classic Prompt Optimization")

    # Data
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest Parquet with psychoacoustic conditions.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Split to use in manifest (e.g., dev, test).",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for best prompts, history, and logs.",
    )

    # Evaluator (Qwen2-Audio)
    parser.add_argument(
        "--evaluator_model_name",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
    )
    parser.add_argument(
        "--evaluator_device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="If passed, do not load LoRA (use base model).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional LoRA checkpoint (same as in evaluate_with_generation.py).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen2",
        choices=["qwen2", "qwen3_omni"],
        help="Model type: qwen2 (Qwen2-Audio-7B) or qwen3_omni (Qwen3-Omni-30B).",
    )

    # OPRO / LLM optimizer
    parser.add_argument(
        "--optimizer_llm",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model for generating prompts.",
    )
    parser.add_argument(
        "--optimizer_device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--optimizer_load_in_4bit",
        action="store_true",
        default=True,
        help="Load optimizer LLM in 4-bit.",
    )
    parser.add_argument(
        "--optimizer_max_new_tokens",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--optimizer_temperature",
        type=float,
        default=0.7,
    )

    # OPRO configuration
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--candidates_per_iter",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=5,
    )

    # Sampling configuration (for faster smoke tests)
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=0,
        help="Limit evaluation to N samples per iteration (0 = use all). "
        "Useful for smoke tests: try 200-600 for fast iteration.",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        choices=["uniform", "stratified", "per_condition"],
        default="stratified",
        help="How to sample subset: uniform (random), stratified (preserve class balance), "
        "per_condition (k samples per duration/SNR bucket).",
    )
    parser.add_argument(
        "--per_condition_k",
        type=int,
        default=5,
        help="Number of samples per condition bucket when sample_strategy=per_condition.",
    )

    # Reward weights
    parser.add_argument(
        "--reward_w_ba_clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--reward_w_ba_cond",
        type=float,
        default=0.25,
    )
    # NOTE: --reward_w_length_penalty removed. Prompt length is enforced as a hard
    # constraint in sanitize_prompt() (max 300 chars), NOT as part of the reward function.
    # This avoids biasing the optimizer toward artificially short prompts.

    # Baseline / initial prompts
    parser.add_argument(
        "--baseline_prompt",
        type=str,
        default="Does this audio contain human speech?\nReply with ONLY one word: SPEECH or NON-SPEECH.",
    )
    parser.add_argument(
        "--initial_prompts_json",
        type=str,
        default=None,
        help="Optional JSON file with initial prompts (list of strings).",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print("OPRO CLASSIC OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Manifest: {args.manifest}")
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir}")
    print(f"Evaluator: {args.evaluator_model_name}")
    print(f"Optimizer LLM: {args.optimizer_llm}")
    print(f"Seed: {args.seed}")

    # Load evaluator model
    evaluator_model = build_evaluator_from_args(args)

    # Prepare evaluator function
    evaluator_fn = make_evaluator_fn(args, evaluator_model)

    # Load initial prompts if provided
    initial_prompts = []
    if args.initial_prompts_json:
        with open(args.initial_prompts_json, encoding="utf-8") as f:
            initial_prompts = json.load(f)
        print(f"\nLoaded {len(initial_prompts)} initial prompts from {args.initial_prompts_json}")

    # Configure reward weights (no length penalty - accuracy only)
    reward_weights = {
        "ba_clip": args.reward_w_ba_clip,
        "ba_cond": args.reward_w_ba_cond,
    }

    # Instantiate OPRO optimizer
    optimizer = OPROClassicOptimizer(
        optimizer_llm=args.optimizer_llm,
        device=args.optimizer_device,
        load_in_4bit=args.optimizer_load_in_4bit,
        top_k=args.top_k,
        candidates_per_iter=args.candidates_per_iter,
        reward_weights=reward_weights,
        seed=args.seed,
        baseline_prompt=args.baseline_prompt,
        initial_prompts=initial_prompts,
        max_new_tokens=args.optimizer_max_new_tokens,
        temperature=args.optimizer_temperature,
    )

    # Run optimization
    best_candidate = optimizer.run_optimization(
        evaluator_fn=evaluator_fn,
        n_iterations=args.num_iterations,
        early_stopping_patience=args.early_stopping,
        output_dir=output_dir,
    )

    # Save final summary
    summary = {
        "best_prompt": best_candidate.prompt,
        "reward": best_candidate.reward,
        "ba_clip": best_candidate.ba_clip,
        "ba_conditions": best_candidate.ba_conditions,
        "prompt_length": best_candidate.prompt_length,
        "iteration": best_candidate.iteration,
    }
    with open(output_dir / "best_prompt_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BEST PROMPT ===")
    print(best_candidate.prompt)
    print(f"\nBA_clip: {best_candidate.ba_clip:.3f}")
    print(f"BA_conditions: {best_candidate.ba_conditions:.3f}")
    print(f"Reward: {best_candidate.reward:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
