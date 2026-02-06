# OPRO3 Comparative Matrix Results

**Generated:** 2026-02-05
**Test Samples:** 21,340
**Seed:** 42

---

## Summary Table

| Model | (A) Baseline | (B) OPRO LLM | (C) OPRO Template | Best | Gain |
|-------|-------------|--------------|-------------------|------|------|
| **Qwen2-Base** | 64.01% | **82.61%** | 75.08% | LLM | +18.60 pp |
| **Qwen2-LoRA** | 86.40% | 84.04% | **93.29%** | Template | +6.89 pp |
| **Qwen3-Omni** | 91.08% | **91.37%** | 87.80% | LLM | +0.29 pp |

---

## Detailed Results

### Row 1: Qwen2-Audio (Base)

| Cell | Method | BA_clip | Speech Acc | Nonspeech Acc | Prompt |
|------|--------|---------|------------|---------------|--------|
| 1A | Baseline | 64.01% | 32.08% | 95.93% | "Does this audio contain human speech? Answer SPEECH or NONSPEECH." |
| 1B | OPRO LLM | **82.61%** | 74.66% | 90.57% | "Is this audio human speech? Answer: SPEECH or NON-SPEECH." |
| 1C | OPRO Template | 75.08% | 71.74% | 78.43% | "Make a definite decision for the clip. Output exactly one token: SPEECH or NONSPEECH." |

**Winner:** OPRO LLM (+18.60 pp over baseline)

---

### Row 2: Qwen2-Audio (LoRA Fine-tuned)

| Cell | Method | BA_clip | Speech Acc | Nonspeech Acc | Prompt |
|------|--------|---------|------------|---------------|--------|
| 2A | Baseline | 86.40% | 82.45% | 90.35% | "Does this audio contain human speech? Answer SPEECH or NONSPEECH." |
| 2B | OPRO LLM | 84.04% | 80.00% | 88.08% | "Classify this audio as SPEECH or NON-SPEECH, focusing on short and noisy clips." |
| 2C | OPRO Template | **93.29%** | 92.95% | 93.63% | "Detect human speech. Treat the following as NONSPEECH: pure tones/beeps, clicks, clock ticks, music, environmental noise, silence. Answer: SPEECH or NONSPEECH." |

**Winner:** OPRO Template (+6.89 pp over baseline)
**Note:** OPRO LLM actually hurt performance (-2.36 pp)

---

### Row 3: Qwen3-Omni

| Cell | Method | BA_clip | Speech Acc | Nonspeech Acc | Prompt |
|------|--------|---------|------------|---------------|--------|
| 3A | Baseline | 91.08% | 87.42% | 94.74% | "Does this audio contain human speech? Answer SPEECH or NONSPEECH." |
| 3B | OPRO LLM | **91.37%** | 89.20% | 93.54% | "What type of sound is this? Respond: SPEECH or NON-SPEECH." |
| 3C | OPRO Template | 87.80% | 90.22% | 85.39% | "Decide the dominant content. Definitions: SPEECH = human voice... NONSPEECH = music, tones/beeps..." |

**Winner:** OPRO LLM (+0.29 pp over baseline)
**Note:** Minimal gain suggests Qwen3-Omni is already well-calibrated for this task

---

## Key Findings

1. **Base models prefer OPRO LLM:** Natural, concise prompts work better for non-fine-tuned models
2. **Fine-tuned models prefer OPRO Template:** Structured prompts with explicit definitions leverage learned representations
3. **OPRO LLM can hurt fine-tuned models:** -2.36 pp drop for LoRA suggests over-simplification
4. **Best absolute result:** Qwen2-LoRA + OPRO Template = **93.29%**
5. **Qwen3-Omni is already strong:** Achieves 91%+ with minimal optimization needed

---

## Methodology Comparison

| Aspect | OPRO LLM | OPRO Template |
|--------|----------|---------------|
| Samples/eval | 660 (full dev) | 20 (subset) |
| Iterations | Up to 30 (early stop) | 15 (fixed) |
| Candidates/iter | 3 | 8 |
| Generation | Meta-LLM adaptive | Fixed library + shuffle |
| Prompt style | Natural, concise | Structured, explicit |

---

## Source Runs

| Cells | Run Directory | Notes |
|-------|--------------|-------|
| 1A, 1B, 1C, 3A | `20260130_185046_COMPARATIVE_RUN` | Qwen2-Base + Qwen3 Baseline |
| 2A, 2B, 2C | `20260204_201138_COMPARATIVE_RUN` | LoRA with corrected data (3,072 samples) |
| 3B, 3C | `20260204_201131_COMPARATIVE_RUN` | Qwen3 OPRO optimization |
