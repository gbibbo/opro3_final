# Audio-Language Models under Psychometric Degradations: Optimization for Voice Activity Detection

Robust Voice Activity Detection with audio-language models under short, noisy, reverberant, and filtered audio.

This repository contains an applied audio ML experiment testing whether a smaller adapted audio-language model can outperform a larger frozen model for degraded speech detection.

**Main result:** Qwen2-Audio-7B adapted with LoRA and OPRO-Template reached **93.3% balanced accuracy** on **21,340 degraded test clips**, outperforming a frozen Qwen3-Omni-30B baseline at **91.1% balanced accuracy**.

**Project case study:** https://gbibbo.github.io/work/audio-language-vad/

---

## Problem

Voice Activity Detection often works well on clean, well-segmented audio. It is harder when the model has to decide from short or degraded acoustic evidence.

This project evaluates VAD under four controlled degradation axes:

| Axis | Conditions |
|---|---|
| Segment duration | 20, 40, 60, 80, 100, 200, 500, 1000 ms |
| Additive noise | -10, -5, 0, +5, +10, +20 dB |
| Reverberation | RT60 = 0.0, 0.3, 1.0, 2.5 s |
| Spectral filtering | none, bandpass, lowpass, highpass |

The final test bank contains **21,340 clips** across speech and non-speech classes.

---

## What I built

The repository implements and documents a comparison matrix for robust binary VAD using audio-language models.

| Component | Description |
|---|---|
| Degradation benchmark | Controlled test bank across duration, SNR, reverberation, and filtering |
| Model comparison | Qwen2-Audio-7B base, Qwen2-Audio-7B + LoRA, Qwen3-Omni-30B frozen |
| Prompt comparison | Hand prompt, OPRO-LLM, OPRO-Template |
| Adaptation | LoRA fine-tuning for Qwen2-Audio-7B |
| Evaluation | Balanced accuracy, speech recall, non-speech recall, per-condition breakdowns |
| Reporting | JSON metrics, CSV predictions, LaTeX tables, audit reports, and figures |

### Experiment flow

```text
Speech and non-speech audio
        ↓
Controlled degradation bank
        ↓
Model configuration
  ├─ Qwen2-Audio-7B base
  ├─ Qwen2-Audio-7B + LoRA
  └─ Qwen3-Omni-30B frozen
        ↓
Prompt strategy
  ├─ Hand prompt
  ├─ OPRO-LLM
  └─ OPRO-Template
        ↓
Evaluation and audit artifacts
  ├─ metrics.json
  ├─ predictions.csv
  ├─ statistical tests
  ├─ LaTeX tables
  └─ figures
```

---

## Results

### Headline comparison

| System | Balanced accuracy | Speech recall | Non-speech recall |
|---|---:|---:|---:|
| Qwen2-Audio-7B + OPRO-LLM | 82.6% | 74.7% | 90.6% |
| **Qwen2-Audio-7B + LoRA + OPRO-Template** | **93.3%** | **92.8%** | **93.8%** |
| Qwen3-Omni-30B frozen + hand prompt | 91.1% | 87.4% | 94.7% |
| Silero VAD | 88.9% | 78.8% | 99.1% |

The best system is not the largest model. The adapted 7B model gives the strongest balanced result, while the specialist Silero baseline remains highly conservative: very strong non-speech recall, weaker speech recall.

Evidence files:

```text
audits/round2/b2_normalization/02_base_opro_llm/metrics.json
audits/round2/b2_normalization/06_lora_opro_template/metrics.json
audits/round2/b2_normalization/07_qwen3_baseline/metrics.json
audits/round2/B6_silero_results.md
results/CONSOLIDATED_MATRIX_RESULTS.md
```

---

## Robustness analysis

### Duration

<img src="figures/Fig_Duration.png" alt="Balanced accuracy across segment duration" width="720">

LoRA + OPRO-Template reaches **DT90 = 96 ms**, meaning it reaches 90% balanced accuracy with approximately 100 ms of audio. The unoptimized baseline does not reach 90% balanced accuracy at any tested duration.

### Noise

<img src="figures/Fig_SNR.png" alt="Balanced accuracy across SNR levels" width="720">

LoRA adaptation makes the model much more stable under additive noise. Prompt optimization improves the base model, but does not reproduce the same robustness profile.

### Sensitivity and specificity

<img src="figures/Fig_Tradeoff.png" alt="Speech recall versus non-speech recall trade-off" width="620">

The systems occupy different operating regimes:

| Regime | Behavior |
|---|---|
| Qwen2-Audio base | Conservative, biased toward non-speech under uncertainty |
| Base + OPRO | Recovers speech sensitivity, with more false alarms |
| LoRA + OPRO-Template | Best balance between speech and non-speech recall |
| Silero VAD | Very high non-speech recall, weaker speech recall |

### Reverberation

The original reverberation figure is included as a PDF in the repository:

```text
figures/Fig_Reverb.pdf
```

---

## Prompt optimization

The prompt optimization stage was evaluated as an experimental factor, not used as a cosmetic prompt rewrite.

| Search component | Count |
|---|---:|
| Total prompt evaluations | 435 |
| Unique prompts | 71 |
| OPRO-LLM evaluations | 75 |
| OPRO-Template evaluations | 360 |

Evidence file:

```text
audits/round1/B8_opro_prompt_analysis.md
```

Multi-seed OPRO-Template results:

| Model | Seeds | Mean BA | Std | Range |
|---|---:|---:|---:|---:|
| Base + OPRO-Template | 5 | 72.34% | 6.14 pp | 61.36 to 75.08% |
| **LoRA + OPRO-Template** | 5 | **91.80%** | **2.44 pp** | 87.66 to 93.29% |
| Qwen3 + OPRO-Template | 5 | 87.86% | 1.13 pp | 86.34 to 89.54% |

Evidence file:

```text
audits/round3/B1_multiseed_opro.md
```

---

## Failure analysis

The repository includes class-level analysis for ESC-50 non-speech categories. The hardest cases are mostly human or animal vocalizations, which are plausible VAD confounders.

| Category | Group | Mean accuracy across configs | LoRA + OPRO-Template accuracy |
|---|---|---:|---:|
| laughing | Human vocalizations | 43.9% | 31.8% |
| coughing | Human vocalizations | 56.4% | 56.6% |
| crying_baby | Human vocalizations | 60.4% | 77.3% |

Evidence file:

```text
audits/round1/B7_esc50_accuracy_report.md
```

This analysis is included because aggregate VAD accuracy is not enough for deployment. A useful system also needs to expose the sounds that produce false alarms or missed speech.

---

## Tech stack

| Area | Tools |
|---|---|
| Models | Qwen2-Audio-7B, Qwen3-Omni-30B, Silero VAD |
| Training and adaptation | PyTorch, LoRA, PEFT, 4-bit quantization |
| Model ecosystem | Hugging Face Transformers, bitsandbytes |
| Audio processing | 16 kHz mono audio, short-window evaluation, degradation banks |
| Evaluation | balanced accuracy, recall, bootstrap confidence intervals, McNemar tests |
| Experiment management | Python scripts, Slurm job support, JSON and CSV artifacts |
| Reporting | Matplotlib figures, LaTeX tables, markdown audit reports |

---

## Repository structure

```text
.
├── README.md
├── config.yaml
├── main.tex
├── figures/
│   ├── Fig_Duration.png
│   ├── Fig_SNR.png
│   ├── Fig_Tradeoff.png
│   ├── Fig_Reverb.pdf
│   └── esc50_heatmap.pdf
├── scripts/
│   ├── run_matrix.py
│   ├── finetune.py
│   ├── eval.py
│   ├── eval_silero.py
│   ├── opro_llm.py
│   ├── opro_template.py
│   ├── stats.py
│   ├── make_tables.py
│   └── plot_final_figures.py
├── results/
│   ├── CONSOLIDATED_MATRIX_RESULTS.json
│   └── CONSOLIDATED_MATRIX_RESULTS.md
├── tables/
│   ├── Tab_R02_OverallPerformance.tex
│   ├── Tab_R04_dimension_means.tex
│   ├── Tab_R05_ErrorCounts.tex
│   └── tab_primary_comparisons.tex
├── audits/
│   └── paper_audit_20260213.md
└── slurm/
    └── stats_rerun.job
```

---

## How to run / inspect

### 1. Clone

```bash
git clone <repo-url>
cd qwen-vad-lora
```

### 2. Inspect the consolidated results

```bash
cat results/CONSOLIDATED_MATRIX_RESULTS.md
```

### 3. Recompute the headline comparison from JSON metrics

```bash
python - <<'PY'
import json

systems = {
    "Base + OPRO-LLM": "audits/round2/b2_normalization/02_base_opro_llm/metrics.json",
    "LoRA + OPRO-Template": "audits/round2/b2_normalization/06_lora_opro_template/metrics.json",
    "Qwen3 + Hand": "audits/round2/b2_normalization/07_qwen3_baseline/metrics.json",
}

print(f"{'system':<24} {'BA':>8} {'speech':>8} {'nonspeech':>10} {'n':>8}")
for name, path in systems.items():
    with open(path) as f:
        m = json.load(f)
    print(
        f"{name:<24} "
        f"{100*m['ba_clip']:>7.1f}% "
        f"{100*m['speech_acc']:>7.1f}% "
        f"{100*m['nonspeech_acc']:>9.1f}% "
        f"{m['n_samples']:>8}"
    )
PY
```

Expected output:

```text
system                         BA   speech  nonspeech        n
Base + OPRO-LLM             82.6%    74.7%      90.6%    21340
LoRA + OPRO-Template        93.3%    92.8%      93.8%    21340
Qwen3 + Hand                91.1%    87.4%      94.7%    21340
```

### 4. Run analysis scripts

Examples:

```bash
python scripts/analyze_multiseed_opro.py
python scripts/analyze_normalization_levels.py
python scripts/analyze_silero.py
python scripts/plot_final_figures.py
python scripts/make_tables.py
```

### 5. Check the experiment orchestrator

```bash
python scripts/run_matrix.py --dry_run
```

Some scripts expect the original data and model-cache layout used during the experiment.

---

## Limitations

- Raw audio datasets are not included in this repository.
- Trained model checkpoints are not bundled.
- Some scripts depend on the original HPC data layout.
- The repository is strongest as an experiment, audit, and reporting package rather than a one-command training library.

---

## Author

Gabriel Bibbó  
Audio ML Research Engineer  
Sound event detection · Voice activity detection · Audio-language models

