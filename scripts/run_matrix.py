#!/usr/bin/env python3
"""
OPRO3 Matrix Orchestrator: Runs the 3x3 comparative experiment.

The 9-cell matrix:
| Model Config       | (A) Baseline    | (B) OPRO LLM    | (C) OPRO Template |
|--------------------|-----------------|-----------------|-------------------|
| 1. Qwen2-Base      | Direct Eval     | Opt -> Eval     | Opt -> Eval       |
| 2. Qwen2-LoRA      | Train -> Eval   | Opt -> Eval     | Opt -> Eval       |
| 3. Qwen3-Omni      | Direct Eval     | Opt -> Eval     | Opt -> Eval       |

Usage:
    python scripts/run_matrix.py --dry_run
    python scripts/run_matrix.py --cells all
    python scripts/run_matrix.py --cells 1A,2A,2B
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Project root
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

# Default baseline prompt
BASELINE_PROMPT = "Does this audio contain human speech? Answer SPEECH or NONSPEECH."

# Cell definitions: (cell_id, model_config, method, dependencies)
MATRIX_CELLS = {
    # Row 1: Qwen2-Audio Base
    "1A": {"name": "qwen2_base_baseline", "model": "qwen2", "method": "baseline", "lora": False, "deps": []},
    "1B": {"name": "qwen2_base_opro_llm", "model": "qwen2", "method": "opro_llm", "lora": False, "deps": []},
    "1C": {"name": "qwen2_base_opro_template", "model": "qwen2", "method": "opro_template", "lora": False, "deps": []},

    # Row 2: Qwen2-Audio LoRA
    "2A": {"name": "qwen2_lora_baseline", "model": "qwen2", "method": "baseline", "lora": True, "deps": ["2_train"]},
    "2B": {"name": "qwen2_lora_opro_llm", "model": "qwen2", "method": "opro_llm", "lora": True, "deps": ["2_train"]},
    "2C": {"name": "qwen2_lora_opro_template", "model": "qwen2", "method": "opro_template", "lora": True, "deps": ["2_train"]},

    # Row 3: Qwen3-Omni
    "3A": {"name": "qwen3_omni_baseline", "model": "qwen3_omni", "method": "baseline", "lora": False, "deps": []},
    "3B": {"name": "qwen3_omni_opro_llm", "model": "qwen3_omni", "method": "opro_llm", "lora": False, "deps": []},
    "3C": {"name": "qwen3_omni_opro_template", "model": "qwen3_omni", "method": "opro_template", "lora": False, "deps": []},
}


class MatrixOrchestrator:
    """Orchestrates the 3x3 experiment matrix."""

    def __init__(self, output_root: Path, seed: int = 42, dry_run: bool = False, resume_dir: Path = None):
        self.seed = seed
        self.dry_run = dry_run

        # Initialize tracking sets BEFORE using them
        self.completed = set()
        self.results = {}

        if resume_dir:
            # Resume into existing directory
            self.output_root = resume_dir
            print(f"[RESUME] Using existing output directory: {self.output_root}")
            # Check for existing LoRA checkpoint
            lora_checkpoint = self.output_root / "00_lora_training" / "checkpoints" / "final"
            if lora_checkpoint.exists():
                self.checkpoint_dir = lora_checkpoint
                self.completed.add("2_train")  # Mark training as done
                print(f"[RESUME] Found existing LoRA checkpoint: {self.checkpoint_dir}")
        else:
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_root = output_root / f"{timestamp}_COMPARATIVE_RUN"
            self.checkpoint_dir = None

    def setup(self):
        """Create output directory structure."""
        if not self.dry_run:
            self.output_root.mkdir(parents=True, exist_ok=True)
            (self.output_root / "logs").mkdir(exist_ok=True)

        print(f"Output root: {self.output_root}")

    def get_cell_output_dir(self, cell_id: str) -> Path:
        """Get output directory for a cell."""
        cell = MATRIX_CELLS[cell_id]
        idx = list(MATRIX_CELLS.keys()).index(cell_id) + 1
        folder_name = f"{idx:02d}_{cell['name']}"
        return self.output_root / folder_name

    def _find_train_csv(self) -> Path:
        """Find training CSV for finetuning.

        Uses experimental_variants (3,072 samples) to match opro2_clean methodology.
        This includes various SNR/duration conditions for robust training.
        """
        # FIX: Use experimental_variants (3,072 samples) instead of base_1000ms_large (200 samples)
        # This matches opro2_clean/scripts/run_complete_pipeline.py line 130
        candidates = [
            DATA_ROOT / "processed" / "experimental_variants" / "train_metadata.csv",  # 3,072 samples
            DATA_ROOT / "processed" / "experimental_variants_large" / "train_metadata.csv",  # 4,400 backup
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Training CSV not found. Tried: {candidates}")

    def _find_val_csv(self) -> Path:
        """Find validation CSV for finetuning.

        Uses experimental_variants dev split to match training data distribution.
        """
        # FIX: Use experimental_variants dev split to match training data
        candidates = [
            DATA_ROOT / "processed" / "experimental_variants" / "dev_metadata.csv",  # 3,456 samples
            DATA_ROOT / "processed" / "experimental_variants_large" / "dev_metadata.csv",  # backup
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Validation CSV not found. Tried: {candidates}")

    def _find_dev_manifest(self) -> Path:
        """Find dev manifest for OPRO optimization."""
        candidates = [
            DATA_ROOT / "processed" / "variants_validated_1000" / "dev_metadata.csv",
            DATA_ROOT / "processed" / "experimental_variants" / "dev_metadata.csv",
            DATA_ROOT / "processed" / "base_validated_1000" / "dev_metadata.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Dev manifest not found. Tried: {candidates}")

    def _find_test_manifest(self) -> Path:
        """Find test manifest for evaluation."""
        candidates = [
            DATA_ROOT / "processed" / "variants_validated_1000" / "test_metadata.csv",
            DATA_ROOT / "processed" / "experimental_variants" / "test_metadata.csv",
            DATA_ROOT / "processed" / "base_validated_1000" / "test_metadata.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Test manifest not found. Tried: {candidates}")

    def run_training(self) -> bool:
        """Run LoRA training for Qwen2-Audio."""
        print("\n" + "="*80)
        print("STEP: LoRA Training (Qwen2-Audio)")
        print("="*80)

        output_dir = self.output_root / "00_lora_training"

        if self.dry_run:
            print("[DRY RUN] Would run finetune.py")
            print(f"  Output: {output_dir}")
            self.checkpoint_dir = output_dir / "checkpoints" / "final"
            self.completed.add("2_train")
            return True

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            train_csv = self._find_train_csv()
            val_csv = self._find_val_csv()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

        cmd = [
            "python3", str(REPO_ROOT / "scripts" / "finetune.py"),
            "--train_csv", str(train_csv),
            "--val_csv", str(val_csv),
            "--output_dir", str(output_dir / "checkpoints"),
            "--seed", str(self.seed),
            "--num_epochs", "3",
            "--per_device_train_batch_size", "2",
            "--gradient_accumulation_steps", "8",
            "--skip_vad_filter",  # Match opro2_clean behavior (no VAD filtering)
        ]

        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            self.checkpoint_dir = output_dir / "checkpoints" / "final"
            self.completed.add("2_train")
            return True
        else:
            print(f"ERROR: Training failed with exit code {result.returncode}")
            return False

    def run_optimization(self, cell_id: str) -> bool:
        """Run OPRO optimization for a cell."""
        cell = MATRIX_CELLS[cell_id]
        method = cell["method"]

        if method == "baseline":
            # No optimization needed for baseline
            return True

        print(f"\n{'='*80}")
        print(f"STEP: Optimization ({cell_id}: {cell['name']})")
        print(f"{'='*80}")

        output_dir = self.get_cell_output_dir(cell_id)

        if self.dry_run:
            print(f"[DRY RUN] Would run {method}.py")
            print(f"  Model: {cell['model']}")
            print(f"  LoRA: {cell['lora']}")
            print(f"  Output: {output_dir}")
            return True

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            dev_manifest = self._find_dev_manifest()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

        # Build command based on method
        if method == "opro_llm":
            script = "opro_llm.py"
            cmd = [
                "python3", str(REPO_ROOT / "scripts" / script),
                "--manifest", str(dev_manifest),
                "--split", "dev",
                "--output_dir", str(output_dir / "optimization"),
                "--model_type", cell["model"],
                "--num_iterations", "30",
                "--candidates_per_iter", "3",
                "--early_stopping", "5",
                "--seed", str(self.seed),
            ]
        else:  # opro_template
            script = "opro_template.py"
            cmd = [
                "python3", str(REPO_ROOT / "scripts" / script),
                "--train_csv", str(dev_manifest),
                "--output_dir", str(output_dir / "optimization"),
                "--model_type", cell["model"],
                "--num_iterations", "15",
                "--samples_per_iter", "20",
                "--seed", str(self.seed),
            ]

        # Add LoRA checkpoint if applicable
        if cell["lora"] and self.checkpoint_dir:
            cmd.extend(["--checkpoint", str(self.checkpoint_dir)])
        else:
            cmd.append("--no_lora")

        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)

        return result.returncode == 0

    def run_evaluation(self, cell_id: str) -> bool:
        """Run evaluation for a cell."""
        cell = MATRIX_CELLS[cell_id]

        print(f"\n{'='*80}")
        print(f"STEP: Evaluation ({cell_id}: {cell['name']})")
        print(f"{'='*80}")

        output_dir = self.get_cell_output_dir(cell_id)

        if self.dry_run:
            print(f"[DRY RUN] Would run eval.py")
            print(f"  Model: {cell['model']}")
            print(f"  LoRA: {cell['lora']}")
            print(f"  Output: {output_dir}")
            return True

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            test_manifest = self._find_test_manifest()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

        # Determine prompt
        if cell["method"] == "baseline":
            prompt = BASELINE_PROMPT
        else:
            # Use optimized prompt from optimization step
            prompt_file = output_dir / "optimization" / "best_prompt.txt"
            if prompt_file.exists():
                prompt = prompt_file.read_text().strip()
            else:
                print(f"ERROR: Best prompt not found at {prompt_file}")
                return False

        # Build evaluation command
        cmd = [
            "python3", str(REPO_ROOT / "scripts" / "eval.py"),
            "--manifest", str(test_manifest),
            "--output_dir", str(output_dir / "evaluation"),
            "--model_type", cell["model"],
            "--prompt", prompt,
        ]

        # Add LoRA checkpoint if applicable
        if cell["lora"] and self.checkpoint_dir:
            cmd.extend(["--checkpoint", str(self.checkpoint_dir)])

        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            # Load and store metrics
            metrics_file = output_dir / "evaluation" / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    self.results[cell_id] = json.load(f)
            return True
        else:
            return False

    def run_cell(self, cell_id: str) -> bool:
        """Run a complete cell (optimization + evaluation)."""
        cell = MATRIX_CELLS[cell_id]

        # Check dependencies
        for dep in cell["deps"]:
            if dep not in self.completed:
                if dep == "2_train":
                    if not self.run_training():
                        return False
                else:
                    print(f"ERROR: Dependency {dep} not satisfied")
                    return False

        # Run optimization (if needed)
        if not self.run_optimization(cell_id):
            return False

        # Run evaluation
        if not self.run_evaluation(cell_id):
            return False

        self.completed.add(cell_id)
        return True

    def run_matrix(self, cells: list):
        """Run specified cells of the matrix."""
        self.setup()

        # Execution order: training first if needed, then cells in order
        needs_training = any(MATRIX_CELLS[c]["lora"] for c in cells)

        if needs_training and "2_train" not in self.completed:
            if not self.run_training():
                print("ERROR: Training failed, aborting")
                return False

        # Run each cell
        for cell_id in cells:
            print(f"\n{'#'*80}")
            print(f"# CELL {cell_id}: {MATRIX_CELLS[cell_id]['name']}")
            print(f"{'#'*80}")

            if not self.run_cell(cell_id):
                print(f"ERROR: Cell {cell_id} failed")
                # Continue with other cells

        # Save summary
        self.save_summary()
        return True

    def save_summary(self):
        """Save experiment summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "completed_cells": list(self.completed),
            "results": self.results,
        }

        if not self.dry_run:
            summary_path = self.output_root / "experiment_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {summary_path}")
        else:
            print(f"\n[DRY RUN] Would save summary to: {self.output_root / 'experiment_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="OPRO3 Matrix Orchestrator")
    parser.add_argument("--cells", type=str, default="all",
                        help="Cells to run: 'all' or comma-separated (e.g., '1A,1B,2A')")
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "results"),
                        help="Output directory root")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Resume into existing output directory (e.g., 'results/20260130_185046_COMPARATIVE_RUN')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    # Parse cell selection
    if args.cells.lower() == "all":
        cells = list(MATRIX_CELLS.keys())
    else:
        cells = [c.strip().upper() for c in args.cells.split(",")]
        # Validate
        for c in cells:
            if c not in MATRIX_CELLS:
                print(f"ERROR: Unknown cell '{c}'. Valid: {list(MATRIX_CELLS.keys())}")
                sys.exit(1)

    print("="*80)
    print("OPRO3 Matrix Orchestrator")
    print("="*80)
    print(f"  Cells to run: {cells}")
    print(f"  Seed: {args.seed}")
    print(f"  Dry run: {args.dry_run}")

    orchestrator = MatrixOrchestrator(
        output_root=Path(args.output_dir),
        seed=args.seed,
        dry_run=args.dry_run,
        resume_dir=Path(args.resume_dir) if args.resume_dir else None,
    )

    success = orchestrator.run_matrix(cells)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
