#!/usr/bin/env python3
"""
Simple evaluation script for speech detection.
Evaluates a prompt on ALL samples, processing in batches to avoid CUDA OOM.
Reports per-condition metrics for the 22 independent conditions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from peft import PeftModel

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


def parse_args():
    parser = argparse.ArgumentParser(description="Simple evaluation script")
    parser.add_argument("--manifest", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to evaluate (alternative: use --prompt_file)")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to text file containing the prompt")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--model_type", type=str, default="qwen2", choices=["qwen2", "qwen3_omni"],
                        help="Model type: qwen2 (Qwen2-Audio-7B) or qwen3_omni (Qwen3-Omni-30B)")
    return parser.parse_args()


def get_condition_key(row):
    """Extract condition key from row (e.g., 'dur_20ms', 'snr_-10dB')."""
    variant_type = row.get("variant_type", "")

    if variant_type == "duration":
        dur = row.get("duration_ms", "")
        return f"dur_{int(dur)}ms"
    elif variant_type == "snr":
        snr = row.get("snr_db", "")
        return f"snr_{float(snr):.0f}dB"
    elif variant_type == "reverb":
        t60 = row.get("T60", "")
        if pd.isna(t60) or t60 == "" or t60 == "none":
            return "reverb_none"
        return f"reverb_{t60}s"
    elif variant_type == "filter":
        filt = row.get("band_filter", "none")
        return f"filter_{filt}"
    else:
        return "unknown"


def evaluate_samples(model, samples_df, prompt, batch_size=50):
    """
    Evaluate all samples in batches.
    Returns list of (prediction, ground_truth, condition_key) tuples.
    """
    results = []

    # Set the prompt once
    model.user_prompt = prompt

    # Process in batches
    total = len(samples_df)
    for start_idx in tqdm(range(0, total, batch_size), desc="Evaluating"):
        end_idx = min(start_idx + batch_size, total)
        batch_df = samples_df.iloc[start_idx:end_idx]

        for _, row in batch_df.iterrows():
            audio_path = row["audio_path"]
            ground_truth = row["ground_truth"]
            condition_key = get_condition_key(row)

            # Make absolute path if needed
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(os.getcwd(), audio_path)

            try:
                # Get model prediction
                result = model.predict(audio_path, return_scores=True)
                response = result.get("prediction", "") if isinstance(result, dict) else str(result)
                prediction, _ = normalize_to_binary(response)

                # LLM fallback for ambiguous responses
                if prediction is None:
                    fallback_label, _ = llm_fallback_interpret(response)
                    prediction = fallback_label if fallback_label is not None else "UNKNOWN"

            except Exception as e:
                print(f"  Error processing {audio_path}: {e}")
                prediction = "ERROR"

            results.append({
                "audio_path": audio_path,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "condition": condition_key,
                "variant_type": row.get("variant_type", "unknown")
            })

        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def compute_metrics(results):
    """Compute per-condition and aggregate metrics."""

    # Group by condition
    condition_results = defaultdict(list)
    for r in results:
        condition_results[r["condition"]].append(r)

    # Compute per-condition BA
    condition_metrics = {}
    for condition, cond_results in condition_results.items():
        speech_correct = sum(1 for r in cond_results
                           if r["ground_truth"] == "SPEECH" and r["prediction"] == "SPEECH")
        speech_total = sum(1 for r in cond_results if r["ground_truth"] == "SPEECH")

        nonspeech_correct = sum(1 for r in cond_results
                               if r["ground_truth"] == "NONSPEECH" and r["prediction"] == "NONSPEECH")
        nonspeech_total = sum(1 for r in cond_results if r["ground_truth"] == "NONSPEECH")

        speech_acc = speech_correct / speech_total if speech_total > 0 else 0
        nonspeech_acc = nonspeech_correct / nonspeech_total if nonspeech_total > 0 else 0
        ba = (speech_acc + nonspeech_acc) / 2

        condition_metrics[condition] = {
            "ba": ba,
            "speech_acc": speech_acc,
            "nonspeech_acc": nonspeech_acc,
            "n_samples": len(cond_results),
            "n_speech": speech_total,
            "n_nonspeech": nonspeech_total
        }

    # Group conditions by dimension
    dimension_conditions = {
        "duration": [k for k in condition_metrics if k.startswith("dur_")],
        "snr": [k for k in condition_metrics if k.startswith("snr_")],
        "reverb": [k for k in condition_metrics if k.startswith("reverb_")],
        "filter": [k for k in condition_metrics if k.startswith("filter_")]
    }

    # Compute per-dimension BA (mean of condition BAs)
    dimension_metrics = {}
    for dim, conditions in dimension_conditions.items():
        if conditions:
            bas = [condition_metrics[c]["ba"] for c in conditions]
            dimension_metrics[dim] = {
                "ba": sum(bas) / len(bas),
                "n_conditions": len(conditions),
                "conditions": conditions
            }

    # Compute overall metrics
    all_speech_correct = sum(1 for r in results
                            if r["ground_truth"] == "SPEECH" and r["prediction"] == "SPEECH")
    all_speech_total = sum(1 for r in results if r["ground_truth"] == "SPEECH")
    all_nonspeech_correct = sum(1 for r in results
                               if r["ground_truth"] == "NONSPEECH" and r["prediction"] == "NONSPEECH")
    all_nonspeech_total = sum(1 for r in results if r["ground_truth"] == "NONSPEECH")

    overall_speech_acc = all_speech_correct / all_speech_total if all_speech_total > 0 else 0
    overall_nonspeech_acc = all_nonspeech_correct / all_nonspeech_total if all_nonspeech_total > 0 else 0
    ba_clip = (overall_speech_acc + overall_nonspeech_acc) / 2

    # BA_conditions = mean of 4 dimension BAs
    if dimension_metrics:
        ba_conditions = sum(d["ba"] for d in dimension_metrics.values()) / len(dimension_metrics)
    else:
        ba_conditions = ba_clip

    return {
        "ba_clip": ba_clip,
        "ba_conditions": ba_conditions,
        "speech_acc": overall_speech_acc,
        "nonspeech_acc": overall_nonspeech_acc,
        "n_samples": len(results),
        "n_speech": all_speech_total,
        "n_nonspeech": all_nonspeech_total,
        "dimension_metrics": dimension_metrics,
        "condition_metrics": condition_metrics
    }


def main():
    args = parse_args()

    # Read prompt from file if --prompt_file is provided
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        print(f"Loaded prompt from file: {args.prompt_file}")
    elif args.prompt:
        prompt = args.prompt
    else:
        raise ValueError("Must provide either --prompt or --prompt_file")

    print("=" * 60)
    print("SIMPLE EVALUATION")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Prompt: {prompt}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA: {args.checkpoint or 'None (BASE model)'}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load manifest
    print("Loading manifest...")
    if args.manifest.endswith('.parquet'):
        df = pd.read_parquet(args.manifest)
    else:
        df = pd.read_csv(args.manifest)

    # Fix Windows-style paths (backslashes) to Unix-style
    if 'audio_path' in df.columns:
        df['audio_path'] = df['audio_path'].str.replace('\\', '/', regex=False)

    print(f"  Total samples: {len(df)}")
    print(f"  Variant types: {df['variant_type'].value_counts().to_dict()}")

    # Load model
    print("\nLoading model...")
    model_type = getattr(args, "model_type", "qwen2")

    if model_type == "qwen3_omni":
        if not QWEN3_AVAILABLE:
            raise RuntimeError(
                "Qwen3-Omni not available. Install transformers from GitHub:\n"
                "pip install git+https://github.com/huggingface/transformers.git"
            )
        print("  Model type: Qwen3-Omni")
        model = Qwen3OmniClassifier(
            model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            device=args.device,
            torch_dtype="auto",
        )
        # Qwen3-Omni doesn't support LoRA
        if args.checkpoint:
            print("  WARNING: LoRA not supported for Qwen3-Omni, ignoring checkpoint")
    else:
        print("  Model type: Qwen2-Audio")
        model = Qwen2AudioClassifier(
            model_name="Qwen/Qwen2-Audio-7B-Instruct",
            device=args.device,
            torch_dtype="auto",
            load_in_4bit=True
        )
        if args.checkpoint:
            print(f"  Loading LoRA checkpoint: {args.checkpoint}")
            model.model = PeftModel.from_pretrained(model.model, args.checkpoint)
            model.model.eval()
            print("  LoRA loaded!")

    # Evaluate
    print(f"\nEvaluating {len(df)} samples...")
    results = evaluate_samples(model, df, prompt, args.batch_size)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall:")
    print(f"  BA_clip: {metrics['ba_clip']:.4f} ({metrics['ba_clip']*100:.1f}%)")
    print(f"  BA_conditions: {metrics['ba_conditions']:.4f} ({metrics['ba_conditions']*100:.1f}%)")
    print(f"  Speech accuracy: {metrics['speech_acc']:.4f} ({metrics['speech_acc']*100:.1f}%)")
    print(f"  NonSpeech accuracy: {metrics['nonspeech_acc']:.4f} ({metrics['nonspeech_acc']*100:.1f}%)")
    print(f"  Samples: {metrics['n_samples']} (speech: {metrics['n_speech']}, nonspeech: {metrics['n_nonspeech']})")

    print("\nPer-dimension BA:")
    for dim, dm in sorted(metrics["dimension_metrics"].items()):
        print(f"  {dim}: {dm['ba']:.4f} ({dm['ba']*100:.1f}%) - {dm['n_conditions']} conditions")

    print("\nPer-condition BA:")
    for cond, cm in sorted(metrics["condition_metrics"].items()):
        print(f"  {cond}: {cm['ba']:.4f} ({cm['ba']*100:.1f}%) - n={cm['n_samples']}")

    # Save results
    metrics["prompt"] = args.prompt
    metrics["checkpoint"] = args.checkpoint
    metrics["manifest"] = args.manifest

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save predictions CSV
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    pd.DataFrame(results).to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
