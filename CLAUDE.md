# OPRO3 FINAL — Comparative Experiment Matrix (Clean Slate)

## 1. Project Context & Goal
This is a **clean-slate** environment designed to execute a rigorous 3x3 comparative experiment. We compare two Prompt Optimization methods against a Baseline across three model configurations.
*   **Root Directory:** `/mnt/fast/nobackup/users/gb0048/opro3_final`
*   **Data Source:** Read-only Symlink at `data/` pointing to `../opro2_clean/data`. **NEVER** generate new audio files or metadata CSVs.

## 2. The Experimental Matrix (3x3)
All executions must align with one of these 9 cells. Do not run experiments outside this matrix.

| Model Config | (A) Baseline (Hand-crafted) | (B) OPRO LLM (Generative) | (C) OPRO Template (Deterministic) |
| :--- | :--- | :--- | :--- |
| **1. Qwen2-Audio (Base)** | Direct Eval | Opt (LLM) -> Eval | Opt (Templ) -> Eval |
| **2. Qwen2-Audio (LoRA)** | Train -> Eval | Opt (LLM) -> Eval | Opt (Templ) -> Eval |
| **3. Qwen3-Omni** | Direct Eval | Opt (LLM) -> Eval | Opt (Templ) -> Eval |

## 3. Environment & Slurm
*   **Host:** VS Code SSH on `datamove1` (Surrey HPC).
*   **Slurm Commands:** NOT in PATH. MUST use wrapper:
    `./slurm/tools/on_submit.sh <squeue|sbatch|scancel|sacct> <args...>`
*   **Job Templates:** New jobs must derive from `slurm/templates/matrix_job_template.job`.
*   **Time Limit:** Default `#SBATCH --time=48:00:00` for optimization/eval jobs.

## 4. Script Architecture (STRICT)
Do not create new logic scripts. Use only these consolidated, renamed scripts:

| New Name | Original Source | Purpose |
| :--- | :--- | :--- |
| `scripts/eval.py` | `evaluate_simple.py` | Universal evaluator (Base, LoRA, Qwen3). |
| `scripts/finetune.py` | `finetune_qwen_audio.py` | LoRA training (Qwen2 only). |
| `scripts/opro_llm.py` | `opro_classic_optimize.py` | Method B: Generative optimization via Meta-LLM. |
| `scripts/opro_template.py` | `opro_post_ft_v2.py` | Method C: Deterministic/Template optimization. |
| `scripts/stats.py` | `statistical_analysis.py` | Confidence intervals & McNemar tests. |
| `scripts/run_matrix.py` | *New* | **The Orchestrator.** Manages the sequence. |

## 5. Execution Rules (Containment Protocol)
1.  **Orchestrator First:** Prefer running the pipeline via `scripts/run_matrix.py` rather than loose Slurm jobs, to ensure folder structure consistency.
2.  **Single Output Root:** All outputs must flow into `results/<TIMESTAMP>_COMPARATIVE_RUN/`.
    *   Subfolders must follow: `01_qwen2_base_baseline`, `02_qwen2_base_opro_llm`, etc.
3.  **No Garbage:** Do not create intermediate files in the root. Use `/tmp` or the designated results folder.
4.  **Traceability:** Every result CSV must be traceable to the specific script arguments that generated it.

## 6. Technical Constraints & Lessons Learned
*   **Qwen3 vs Qwen2:** Qwen3-Omni requires `transformers` from GitHub (dev version). Qwen2 uses standard/bitsandbytes. Be careful with environment containers (`.sif`).
*   **Symlinks:** Ensure `opro3_final/data` is a valid symlink before running.
*   **OPRO Script Generalization:**
    *   `opro_llm.py` must accept `--checkpoint` (for LoRA) and `--model_type`.
    *   `opro_template.py` must NOT assume it only runs on LoRA; it must support Base models too.

## 7. Completion Protocol
1.  Verify the Symlink structure (`ls -l data/`).
2.  Ensure the Orchestrator (`run_matrix.py`) dry-run works.
3.  Submit the Master Job.
4.  Once finished, run `scripts/stats.py` on the consolidated results folder.