#!/usr/bin/env python3
"""
OPRO Post Fine-Tuning: Prompt Optimization on Frozen Fine-Tuned Model (v2)

PURPOSE:
    Optimize prompts for fine-tuned Qwen2-Audio model (with LoRA adapters).
    Part of pipeline block G (OPRO on fine-tuned model).

INPUTS:
    - Train/dev CSV with audio paths and labels
    - LoRA checkpoint directory (or --no_lora for base model)

OUTPUTS:
    - Best prompt JSON (best_prompt.json)
    - Optimization history (opro_history.json)

CLUSTER vs LOCAL:
    - Requires GPU with 8GB+ VRAM
    - Run via slurm/opro_classic_lora.job

This version uses Qwen2AudioClassifier (generation-based) instead of logit extraction.
Based on evaluate_with_generation.py which is known to work correctly.
Reward is based on accuracy only (NO length penalty).

Usage:
    python scripts/opro_post_ft_v2.py \\
        --no_lora \\
        --train_csv data/processed/experimental_variants/dev_metadata.csv \\
        --output_dir results/opro_base \\
        --num_iterations 15 \\
        --samples_per_iter 20
"""

import argparse
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.qsm.utils.normalize import normalize_to_binary, detect_format

# Qwen3-Omni support (requires transformers from GitHub)
try:
    from src.qsm.models.qwen3_omni import Qwen3OmniClassifier
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False


def evaluate_sample_with_model(model, audio_path, ground_truth, decoding_mode="auto", mapping=None, verbalizers=None):
    """
    Evaluate a single sample using Qwen2AudioClassifier.

    Args:
        model: Qwen2AudioClassifier instance
        audio_path: Path to audio file
        ground_truth: Ground truth label (SPEECH/NONSPEECH)
        decoding_mode: Decoding mode (ab/mc/labels/open/auto)
        mapping: Optional letter-to-label mapping dict
        verbalizers: Optional list of valid label strings

    Returns:
        dict: {
            'correct': bool,
            'raw_text': str,
            'normalized_label': str,
            'p_first_token': float,
            'probs': dict
        }
    """
    from pathlib import Path

    # Hardening: resolve paths robustly
    audio_path = Path(audio_path)
    if not audio_path.exists():
        # Try with data/ prefix
        audio_path = Path("data") / audio_path

    if not audio_path.exists():
        print(f"  ERROR: File not found: {audio_path}")
        return {
            'correct': False,
            'raw_text': 'ERROR',
            'normalized_label': None,
            'p_first_token': 0.0,
            'probs': {}
        }

    try:
        # Get prediction with scores
        result = model.predict(str(audio_path.resolve()), decoding_mode=decoding_mode, return_scores=True)

        # Use robust normalization
        normalized_label, confidence = normalize_to_binary(
            result.raw_output,
            probs=result.probs,
            mode=decoding_mode,
            mapping=mapping,
            verbalizers=verbalizers
        )

        # Fallback to model's label if normalization failed
        if normalized_label is None:
            normalized_label = result.label

        is_correct = (normalized_label == ground_truth) if normalized_label else False

        return {
            'correct': is_correct,
            'raw_text': result.raw_output,
            'normalized_label': normalized_label,
            'p_first_token': result.probs.get('p_first_token', 0.0) if result.probs else 0.0,
            'probs': result.probs or {}
        }
    except Exception as e:
        print(f"  Error processing {audio_path}: {e}")
        return {
            'correct': False,
            'raw_text': f'ERROR: {e}',
            'normalized_label': None,
            'p_first_token': 0.0,
            'probs': {}
        }


def evaluate_prompt_on_samples(model, samples, prompt_data, decoding_mode="auto"):
    """
    Evaluate a prompt on a set of samples.

    Args:
        model: Qwen2AudioClassifier instance
        samples: List of {audio_path, ground_truth} dicts
        prompt_data: Either a string (prompt text) or dict with 'text', 'mapping', 'verbalizers'
        decoding_mode: Decoding mode (ab/mc/labels/open/auto)

    Returns:
        tuple: (accuracy: float, detailed_results: list)
    """
    # Handle both string prompts and template dicts
    if isinstance(prompt_data, str):
        prompt_text = prompt_data
        mapping = None
        verbalizers = ["SPEECH", "NONSPEECH"]
    else:
        prompt_text = prompt_data.get('text', prompt_data)
        mapping = prompt_data.get('mapping', None)
        verbalizers = prompt_data.get('verbalizers', ["SPEECH", "NONSPEECH"])

    # Update model's prompt
    model.user_prompt = prompt_text

    correct = 0
    total = len(samples)
    detailed_results = []

    for sample in tqdm(samples, desc="  Evaluating", leave=False):
        result = evaluate_sample_with_model(
            model,
            sample['audio_path'],
            sample['ground_truth'],
            decoding_mode=decoding_mode,
            mapping=mapping,
            verbalizers=verbalizers
        )

        if result['correct']:
            correct += 1

        # Collect detailed info
        detailed_results.append({
            'audio_path': sample['audio_path'],
            'ground_truth': sample['ground_truth'],
            'raw_text': result['raw_text'],
            'normalized_label': result['normalized_label'],
            'is_correct': result['correct'],
            'p_first_token': result['p_first_token'],
            **result['probs']  # Unpack A/B/C/D probabilities if available
        })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, detailed_results


def load_templates(templates_file):
    """
    Load prompt templates from JSON file.

    Args:
        templates_file: Path to JSON file with template data

    Returns:
        tuple: (templates_list, format_type, label_space)
    """
    with open(templates_file, 'r') as f:
        data = json.load(f)

    templates = data.get('templates', [])
    format_type = data.get('format', 'auto')
    label_space = data.get('label_space', ['SPEECH', 'NONSPEECH'])

    print(f"Loaded {len(templates)} templates from {templates_file}")
    print(f"  Format: {format_type}")
    print(f"  Label space: {label_space}")

    return templates, format_type, label_space


def generate_candidate_prompts(prompt_history, num_candidates=12):
    """
    Generate candidate prompts using best-practice templates.

    All templates request SPEECH or NONSPEECH (no A/B, no JSON) for deterministic parsing.

    Based on prompt engineering research:
    - Direct label output (SPEECH/NONSPEECH)
    - Label descriptions (verbalizers)
    - Constrained output space
    - Few-shot examples
    - Calibration-friendly formats
    """
    templates = [
        # 1) Minimal direct
        "Does this audio contain human speech? Answer exactly one token: SPEECH or NONSPEECH.",

        # 2) Binary decision
        "Binary decision. Output exactly one token: SPEECH or NONSPEECH.",

        # 3) Label descriptions (verbalizers)
        "Decide the dominant content.\nDefinitions:\n- SPEECH = human voice, spoken words, syllables, conversational cues.\n- NONSPEECH = music, tones/beeps, environmental noise, silence.\nOutput exactly: SPEECH or NONSPEECH.",

        # 4) Contrastive/counter-examples
        "Detect human speech. Treat the following as NONSPEECH: pure tones/beeps, clicks, clock ticks, music, environmental noise, silence.\nAnswer: SPEECH or NONSPEECH.",

        # 5) 1-shot consistency
        "Example:\nAudio→ crowd noise, music → Output: NONSPEECH\nNow classify the new audio. Output exactly ONE token: SPEECH or NONSPEECH.",

        # 6) Forced decision
        "Make a definite decision for the clip.\nOutput exactly one token: SPEECH or NONSPEECH.",

        # 7) Conservative (reduce false positives)
        "Label SPEECH only if human voice is clearly present; otherwise label NONSPEECH.\nAnswer: SPEECH or NONSPEECH.",

        # 8) Liberal (reduce false negatives)
        "If there is any hint of human voice (even faint/short), label SPEECH; otherwise NONSPEECH.\nAnswer: SPEECH or NONSPEECH.",

        # 9) Acoustic focus (vocal tract features)
        "Focus on cues of human vocal tract (formants, syllabic rhythm, consonant onsets).\nAnswer exactly: SPEECH or NONSPEECH.",

        # 10) Task-oriented
        "TASK: Speech detection. Is human voice/speech present in this audio?\nAnswer: SPEECH or NONSPEECH.",

        # 11) Confidence calibration
        "Binary classification task.\nQ: Does this contain human speech?\nIf confident YES → SPEECH\nIf confident NO → NONSPEECH\nAnswer:",

        # 12) Delimiters
        "You will answer with one token only.\n<question>Does this audio contain human speech?</question>\n<answer>SPEECH or NONSPEECH only</answer>",

        # 13) Explicit instruction
        "Classify this audio. Output only: SPEECH or NONSPEECH.",

        # 14) Focus instruction
        "Listen for human voice. If present: SPEECH. Otherwise: NONSPEECH.\nAnswer:",

        # 15) Simplified baseline
        "Human speech present? Answer: SPEECH or NONSPEECH.",
    ]

    # If we have history, include best performing prompt
    if len(prompt_history) > 0:
        best_prompt, best_acc = max(prompt_history, key=lambda x: x[1])
        candidates = [best_prompt]

        # Add templates
        random.shuffle(templates)
        candidates.extend(templates[:num_candidates-1])
    else:
        # First iteration: use templates
        candidates = templates[:num_candidates]

    return candidates


def stratified_sample_df(df, n, seed=42):
    """
    Stratified sampling: 50/50 SPEECH/NONSPEECH.
    """
    n_half = n // 2
    speech_df = df[df['ground_truth'] == 'SPEECH']
    nonspeech_df = df[df['ground_truth'] == 'NONSPEECH']

    # Sample with replacement if needed
    replace_speech = len(speech_df) < n_half
    replace_nonspeech = len(nonspeech_df) < (n - n_half)

    a = speech_df.sample(n=n_half, replace=replace_speech, random_state=seed)
    b = nonspeech_df.sample(n=n - n_half, replace=replace_nonspeech, random_state=seed)

    # Shuffle combined sample
    return pd.concat([a, b]).sample(frac=1, random_state=seed)


def opro_optimize(model, train_df, num_iterations=15, samples_per_iter=20, num_candidates=8, seed=42,
                  templates=None, decoding_mode="auto", output_dir=None, per_prompt_dump=False):
    """
    OPRO optimization loop using generation-based evaluation.

    Args:
        model: Qwen2AudioClassifier instance (frozen)
        train_df: DataFrame with training samples
        num_iterations: Number of optimization iterations
        samples_per_iter: Number of samples to evaluate per iteration
        num_candidates: Number of candidate prompts per iteration
        seed: Random seed for reproducibility
        templates: Optional list of template dicts from JSON file
        decoding_mode: Decoding mode (ab/mc/labels/open/auto)
        output_dir: Optional output directory for saving detailed CSVs
        per_prompt_dump: Whether to save detailed CSV for each prompt

    Returns:
        best_prompt, best_accuracy, history
    """
    # Configure model for deterministic generation
    import torch
    torch.manual_seed(seed)

    if hasattr(model, 'model') and hasattr(model.model, 'generation_config'):
        cfg = model.model.generation_config
        cfg.do_sample = False
        cfg.temperature = 0.0
        cfg.top_p = 1.0
        cfg.max_new_tokens = 3
        print("Model configured for deterministic generation (greedy, T=0)")

    prompt_history = []  # List of (prompt, accuracy) tuples

    # Initialize with baseline prompt
    best_prompt = "Does this audio contain human speech? Answer exactly one token: SPEECH or NONSPEECH."
    best_accuracy = 0.0

    print(f"\nStarting OPRO optimization:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Samples per iteration: {samples_per_iter} (stratified 50/50)")
    print(f"  Candidates per iteration: {num_candidates}")
    print(f"  Total samples available: {len(train_df)}")
    print(f"  Baseline prompt: {best_prompt[:60]}...")
    print()

    for iteration in range(num_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration+1}/{num_iterations}")
        print(f"{'='*80}")

        # Stratified sampling: 50/50 SPEECH/NONSPEECH
        iter_samples_df = stratified_sample_df(train_df, samples_per_iter, seed=seed+iteration)
        iter_samples = iter_samples_df[['audio_path', 'ground_truth']].to_dict('records')

        # Generate candidate prompts
        if templates:
            # Use templates from JSON file
            if iteration == 0:
                # First iteration: use all templates up to num_candidates
                candidates = templates[:num_candidates]
            else:
                # Subsequent iterations: best so far + random templates
                best_template, best_acc = max(prompt_history, key=lambda x: x[1])
                candidates = [best_template]
                remaining = [t for t in templates if t != best_template]
                random.shuffle(remaining)
                candidates.extend(remaining[:num_candidates-1])
        else:
            # Fallback to hardcoded templates
            candidates = generate_candidate_prompts(prompt_history, num_candidates)

        print(f"\nEvaluating {len(candidates)} candidate prompts...")

        # Evaluate each candidate
        candidate_results = []
        all_detailed_results = []  # Collect all detailed results for this iteration

        for i, prompt_data in enumerate(candidates):
            # Extract prompt text for display
            if isinstance(prompt_data, dict):
                prompt_text = prompt_data.get('text', str(prompt_data))
                prompt_id = prompt_data.get('id', f'prompt_{i:02d}')
            else:
                prompt_text = prompt_data
                prompt_id = f'prompt_{i:02d}'

            print(f"\n[{i+1}/{len(candidates)}] Testing prompt ({prompt_id}):")
            print(f"  {prompt_text[:80]}...")

            accuracy, detailed_results = evaluate_prompt_on_samples(
                model, iter_samples, prompt_data, decoding_mode=decoding_mode
            )

            # Add metadata to detailed results
            for result in detailed_results:
                result['prompt_id'] = prompt_id
                result['iteration'] = iteration + 1
                result['decoding_mode'] = decoding_mode
            all_detailed_results.extend(detailed_results)

            candidate_results.append((prompt_data, accuracy))
            print(f"  Accuracy: {accuracy:.1%}")

            # Save per-prompt CSV if requested
            if per_prompt_dump and output_dir:
                prompt_csv = output_dir / f"iter{iteration+1:02d}_{prompt_id}_predictions.csv"
                pd.DataFrame(detailed_results).to_csv(prompt_csv, index=False)

            # Update history
            prompt_history.append((prompt_data, accuracy))

            # Update best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_prompt = prompt_data
                print(f"  ✓ New best! {best_accuracy:.1%}")

        # Save aggregated CSV for this iteration
        if output_dir and all_detailed_results:
            iter_csv = output_dir / f"iter{iteration+1:02d}_all_predictions.csv"
            pd.DataFrame(all_detailed_results).to_csv(iter_csv, index=False)
            print(f"\nIteration results saved to: {iter_csv}")

        # Summary
        best_this_iter = max(candidate_results, key=lambda x: x[1])
        print(f"\nIteration {iteration+1} summary:")
        print(f"  Best this iteration: {best_this_iter[1]:.1%}")
        print(f"  Best overall:        {best_accuracy:.1%}")

    return best_prompt, best_accuracy, prompt_history


def main():
    parser = argparse.ArgumentParser(description="OPRO Post Fine-Tuning (Generation-based)")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--no_lora', action='store_true',
                        help='Use base model without LoRA')
    parser.add_argument('--train_csv', type=str, required=True,
                        help='CSV with training/dev samples for optimization')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--num_iterations', type=int, default=15,
                        help='Number of OPRO iterations')
    parser.add_argument('--samples_per_iter', type=int, default=20,
                        help='Number of samples to evaluate per iteration')
    parser.add_argument('--num_candidates', type=int, default=8,
                        help='Number of candidate prompts per iteration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--templates_file', type=str, default=None,
                        help='Path to JSON file with prompt templates (e.g., prompts/ab.json)')
    parser.add_argument('--decoding', type=str, default='auto',
                        choices=['auto', 'ab', 'mc', 'labels', 'open'],
                        help='Decoding mode: auto (detect from prompt), ab (A/B), mc (A/B/C/D), labels (SPEECH/NONSPEECH), open (free-form)')
    parser.add_argument('--per_prompt_dump', action='store_true',
                        help='Save detailed CSV for each prompt (for debugging)')
    parser.add_argument('--model_type', type=str, default='qwen2',
                        choices=['qwen2', 'qwen3_omni'],
                        help='Model type: qwen2 (Qwen2-Audio-7B) or qwen3_omni (Qwen3-Omni)')

    args = parser.parse_args()

    # Validate args - Qwen3 doesn't support LoRA
    if args.model_type == 'qwen3_omni':
        if args.checkpoint is not None:
            print("WARNING: LoRA not supported for Qwen3-Omni, ignoring --checkpoint")
        args.no_lora = True  # Force no_lora for Qwen3
    elif not args.no_lora and args.checkpoint is None:
        parser.error("--checkpoint is required unless --no_lora is specified (Qwen2)")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OPRO: Prompt Optimization (Generation-based)")
    print("=" * 80)
    print(f"\nModel type: {args.model_type}")
    print(f"Model: {'BASE (no LoRA)' if args.no_lora else args.checkpoint}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Output: {args.output_dir}")

    # Load model based on model_type
    print(f"\nLoading model...")

    if args.model_type == 'qwen3_omni':
        # Qwen3-Omni model
        if not QWEN3_AVAILABLE:
            raise RuntimeError(
                "Qwen3-Omni not available. Install transformers from GitHub:\n"
                "pip install git+https://github.com/huggingface/transformers.git"
            )
        print("  Loading Qwen3-Omni...")
        model = Qwen3OmniClassifier(
            model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            device="cuda",
            torch_dtype="auto",
        )
    else:
        # Qwen2-Audio model (default)
        from src.qsm.models.qwen_audio import Qwen2AudioClassifier
        print("  Loading Qwen2-Audio...")

        if args.no_lora:
            model = Qwen2AudioClassifier(load_in_4bit=True)
        else:
            # Load base model then apply LoRA
            model = Qwen2AudioClassifier(load_in_4bit=True)
            from peft import PeftModel
            model.model = PeftModel.from_pretrained(model.model, args.checkpoint)
            model.model.eval()
            print(f"  LoRA checkpoint loaded: {args.checkpoint}")

    print(f"Model loaded!")

    # Load data
    print(f"\nLoading training data...")
    train_df = pd.read_csv(args.train_csv)
    print(f"Loaded {len(train_df)} samples")
    print(f"  SPEECH:    {(train_df['ground_truth'] == 'SPEECH').sum()}")
    print(f"  NONSPEECH: {(train_df['ground_truth'] == 'NONSPEECH').sum()}")

    # Load templates if provided
    templates = None
    format_type = args.decoding
    if args.templates_file:
        templates, detected_format, label_space = load_templates(args.templates_file)
        if args.decoding == 'auto':
            format_type = detected_format
        print(f"\nUsing decoding mode: {format_type}")

    # Run OPRO
    best_prompt, best_accuracy, history = opro_optimize(
        model, train_df,
        num_iterations=args.num_iterations,
        samples_per_iter=args.samples_per_iter,
        num_candidates=args.num_candidates,
        seed=args.seed,
        templates=templates,
        decoding_mode=format_type,
        output_dir=output_dir,
        per_prompt_dump=args.per_prompt_dump
    )

    # Save results
    print(f"\n{'='*80}")
    print("OPRO OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest prompt (accuracy: {best_accuracy:.1%}):")
    print(f"{best_prompt}")

    # Save best prompt (handle both str and dict)
    best_prompt_file = output_dir / "best_prompt.txt"
    if isinstance(best_prompt, dict):
        best_prompt_file.write_text(best_prompt.get('text', ''))
        (output_dir / 'best_prompt.json').write_text(json.dumps(best_prompt, indent=2))
        print(f"\nBest prompt saved to: {best_prompt_file} and best_prompt.json")
    else:
        best_prompt_file.write_text(str(best_prompt))
        print(f"\nBest prompt saved to: {best_prompt_file}")

    # Save history
    history_file = output_dir / "optimization_history.json"
    with open(history_file, 'w') as f:
        json.dump({
            'best_accuracy': best_accuracy,
            'best_prompt': best_prompt,
            'history': [(p, float(a)) for p, a in history],
            'config': {
                'checkpoint': args.checkpoint,
                'num_iterations': args.num_iterations,
                'samples_per_iter': args.samples_per_iter,
                'num_candidates': args.num_candidates,
                'seed': args.seed,
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)
    print(f"History saved to: {history_file}")

    print(f"\n{'='*80}")
    print("Next steps:")
    print(f"  1. Evaluate best prompt on test set:")
    print(f"     sbatch eval_model.sh --no-lora")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
