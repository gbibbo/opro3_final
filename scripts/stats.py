#!/usr/bin/env python3
"""
Rigorous Statistical Analysis for Speech Detection Model Comparison.

Implements:
1. Wilson score intervals for per-class recalls
2. Cluster bootstrap (10,000 resamples) for BA and ΔBA confidence intervals
3. McNemar exact test for paired model comparisons
4. Holm-Bonferroni correction for multiple comparisons
5. Psychometric threshold estimation (DT50/75/90, SNR-75) with bootstrap CIs
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from tqdm import tqdm


# ============================================================================
# PART 1: Data Loading and Preparation
# ============================================================================

def canonicalize_label(label: str) -> str:
    """
    Canonicalize label to strict SPEECH, NONSPEECH, or UNKNOWN format.

    Handles variants like:
    - 'SPEECH', 'speech', 'Speech'
    - 'NONSPEECH', 'NON-SPEECH', 'NON_SPEECH', 'NON SPEECH', 'nonspeech'
    - 'UNKNOWN', 'unknown' (model abstained from prediction)
    - Trailing punctuation: 'NONSPEECH.', 'SPEECH!'

    Args:
        label: Raw label string

    Returns:
        Canonical label: 'SPEECH', 'NONSPEECH', or 'UNKNOWN'

    Raises:
        ValueError: If label cannot be mapped to known classes
    """
    if not isinstance(label, str):
        raise ValueError(f"Label must be string, got {type(label)}: {label}")

    # Uppercase, strip whitespace, remove trailing punctuation
    canonical = label.upper().strip().rstrip('.,!?;:')

    # Collapse separators (-, _, spaces) to nothing
    canonical = canonical.replace('-', '').replace('_', '').replace(' ', '')

    # Map to valid classes
    if canonical == 'SPEECH':
        return 'SPEECH'
    elif canonical == 'NONSPEECH':
        return 'NONSPEECH'
    elif canonical == 'UNKNOWN':
        return 'UNKNOWN'
    else:
        raise ValueError(
            f"Unknown label '{label}' (canonicalized to '{canonical}'). "
            f"Valid labels: SPEECH, NONSPEECH, UNKNOWN"
        )


def extract_clip_id(audio_path: str) -> str:
    """
    Extract base clip ID from audio path.

    Example:
        'esc50_1-45645-A-31_0152_1000ms_dur20ms.wav' -> 'esc50_1-45645-A-31_0152_1000ms'
    """
    filename = os.path.basename(audio_path)
    # Remove extension
    filename = filename.replace('.wav', '')

    # Split by underscore and remove the last part (variant)
    # Pattern: <base_clip_id>_<variant>
    parts = filename.split('_')

    # Find where the variant starts (dur/snr/reverb/filter)
    variant_indicators = ['dur', 'snr', 'reverb', 'filter']
    for i in range(len(parts) - 1, -1, -1):
        if any(parts[i].startswith(ind) for ind in variant_indicators):
            # Everything before this is the clip_id
            clip_id = '_'.join(parts[:i])
            return clip_id

    # Fallback: just remove last part
    return '_'.join(parts[:-1])


def load_predictions(csv_path: str) -> pd.DataFrame:
    """
    Load predictions CSV and add clip_id column.

    Returns DataFrame with columns:
        - clip_id: base clip identifier
        - condition: condition key (e.g., 'dur_20ms', 'snr_-10dB')
        - variant_type: type of degradation (duration, snr, reverb, filter)
        - ground_truth: SPEECH or NONSPEECH (canonicalized)
        - prediction: SPEECH or NONSPEECH (canonicalized)
        - correct: boolean, whether prediction matches ground truth

    Notes:
        - Rows where prediction == 'UNKNOWN' are filtered out with a warning
        - UNKNOWN indicates the model abstained from making a prediction

    Raises:
        ValueError: If any label cannot be canonicalized to SPEECH, NONSPEECH, or UNKNOWN
    """
    df = pd.read_csv(csv_path)
    n_total = len(df)

    # Extract clip_id
    df['clip_id'] = df['audio_path'].apply(extract_clip_id)

    # Canonicalize labels (strict validation - will raise if unknown label)
    df['ground_truth_canon'] = df['ground_truth'].apply(canonicalize_label)
    df['prediction_canon'] = df['prediction'].apply(canonicalize_label)

    # Replace original columns with canonical versions
    df['ground_truth'] = df['ground_truth_canon']
    df['prediction'] = df['prediction_canon']
    df.drop(columns=['ground_truth_canon', 'prediction_canon'], inplace=True)

    # Filter out UNKNOWN predictions (model abstained)
    unknown_mask = (df['prediction'] == 'UNKNOWN') | (df['ground_truth'] == 'UNKNOWN')
    n_unknown = unknown_mask.sum()

    if n_unknown > 0:
        print(f"  ⚠️  WARNING: Filtered {n_unknown}/{n_total} samples with UNKNOWN labels "
              f"({100*n_unknown/n_total:.2f}%)")
        df = df[~unknown_mask].copy()

    # Add correctness flag using canonical labels
    df['correct'] = (df['ground_truth'] == df['prediction']).astype(int)

    return df


def load_multiple_configs(config_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load predictions from multiple configurations.

    Args:
        config_paths: Dict mapping config_name -> predictions.csv path

    Returns:
        Dict mapping config_name -> DataFrame
    """
    configs = {}
    for name, path in config_paths.items():
        print(f"Loading {name} from {path}")
        configs[name] = load_predictions(path)
    return configs


# ============================================================================
# PART 2: Wilson Score Intervals for Proportions
# ============================================================================

def wilson_score_interval(n_success: int, n_total: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.

    More reliable than normal approximation (Wald) for proportions,
    especially for small samples or extreme proportions.

    Args:
        n_success: Number of successes
        n_total: Total number of trials
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    if n_total == 0:
        return 0.0, 0.0, 0.0

    p = n_success / n_total
    z = stats.norm.ppf(1 - alpha / 2)

    denominator = 1 + z**2 / n_total
    centre_adjusted = (p + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denominator

    lower = max(0.0, centre_adjusted - margin)
    upper = min(1.0, centre_adjusted + margin)

    return p, lower, upper


def compute_recalls_with_wilson(df: pd.DataFrame) -> Dict:
    """
    Compute per-class recalls with Wilson score 95% CIs.

    Returns dict with:
        - recall_speech: point estimate
        - recall_speech_ci: (lower, upper)
        - recall_nonspeech: point estimate
        - recall_nonspeech_ci: (lower, upper)
    """
    # Speech recall
    speech_mask = df['ground_truth'] == 'SPEECH'
    n_speech = speech_mask.sum()
    n_speech_correct = (speech_mask & (df['correct'] == 1)).sum()

    recall_speech, rs_lower, rs_upper = wilson_score_interval(n_speech_correct, n_speech)

    # NonSpeech recall
    nonspeech_mask = df['ground_truth'] == 'NONSPEECH'
    n_nonspeech = nonspeech_mask.sum()
    n_nonspeech_correct = (nonspeech_mask & (df['correct'] == 1)).sum()

    recall_nonspeech, rn_lower, rn_upper = wilson_score_interval(n_nonspeech_correct, n_nonspeech)

    return {
        'recall_speech': float(recall_speech),
        'recall_speech_ci': (float(rs_lower), float(rs_upper)),
        'n_speech': int(n_speech),
        'recall_nonspeech': float(recall_nonspeech),
        'recall_nonspeech_ci': (float(rn_lower), float(rn_upper)),
        'n_nonspeech': int(n_nonspeech)
    }


# ============================================================================
# PART 3: Cluster Bootstrap for BA and ΔBA
# ============================================================================

def compute_ba(df: pd.DataFrame) -> float:
    """Compute balanced accuracy from a DataFrame."""
    speech_mask = df['ground_truth'] == 'SPEECH'
    nonspeech_mask = df['ground_truth'] == 'NONSPEECH'

    recall_speech = (speech_mask & (df['correct'] == 1)).sum() / speech_mask.sum()
    recall_nonspeech = (nonspeech_mask & (df['correct'] == 1)).sum() / nonspeech_mask.sum()

    return (recall_speech + recall_nonspeech) / 2


def cluster_bootstrap_ba(
    df: pd.DataFrame,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Cluster bootstrap for BA confidence interval.

    Resamples base clips (with replacement), includes all samples from each clip.

    Args:
        df: DataFrame with clip_id, ground_truth, correct columns
        n_bootstrap: Number of bootstrap resamples
        random_state: Random seed

    Returns:
        (ba_point, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)

    # Precompute groups (much faster than repeated filtering)
    groups = [group for _, group in df.groupby('clip_id', sort=False)]
    n_clips = len(groups)

    ba_point = compute_ba(df)
    ba_samples = []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap BA", leave=False):
        # Resample clip indices with replacement
        sampled_idx = rng.choice(n_clips, size=n_clips, replace=True)

        # Build resampled dataset
        resampled_df = pd.concat([groups[i] for i in sampled_idx], ignore_index=True)

        # Compute BA on resampled data
        ba_boot = compute_ba(resampled_df)
        ba_samples.append(ba_boot)

    ba_samples = np.array(ba_samples)
    ci_lower = np.percentile(ba_samples, 2.5)
    ci_upper = np.percentile(ba_samples, 97.5)

    return float(ba_point), float(ci_lower), float(ci_upper)


def cluster_bootstrap_delta_ba(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Cluster bootstrap for paired ΔBA = BA(A) - BA(B).

    Uses the SAME resampled clips for both configurations to preserve pairing.

    Returns:
        (delta_ba_point, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)

    # Ensure both have same clips (should be same test set)
    clips_a = set(df_a['clip_id'].unique())
    clips_b = set(df_b['clip_id'].unique())
    unique_clips = sorted(clips_a & clips_b)

    # Precompute groups for both datasets
    groups_a = {clip_id: group for clip_id, group in df_a.groupby('clip_id', sort=False)}
    groups_b = {clip_id: group for clip_id, group in df_b.groupby('clip_id', sort=False)}

    # Keep only clips in both datasets
    groups_a = [groups_a[clip_id] for clip_id in unique_clips]
    groups_b = [groups_b[clip_id] for clip_id in unique_clips]
    n_clips = len(groups_a)

    # Point estimate
    ba_a = compute_ba(df_a)
    ba_b = compute_ba(df_b)
    delta_point = ba_a - ba_b

    delta_samples = []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap ΔBA", leave=False):
        # Resample clip indices (SAME for both)
        sampled_idx = rng.choice(n_clips, size=n_clips, replace=True)

        # Build resampled datasets
        resampled_a = pd.concat([groups_a[i] for i in sampled_idx], ignore_index=True)
        resampled_b = pd.concat([groups_b[i] for i in sampled_idx], ignore_index=True)

        # Compute ΔBA
        ba_boot_a = compute_ba(resampled_a)
        ba_boot_b = compute_ba(resampled_b)
        delta_samples.append(ba_boot_a - ba_boot_b)

    delta_samples = np.array(delta_samples)
    ci_lower = np.percentile(delta_samples, 2.5)
    ci_upper = np.percentile(delta_samples, 97.5)

    return float(delta_point), float(ci_lower), float(ci_upper)


# ============================================================================
# PART 4: McNemar Exact Test
# ============================================================================

def mcnemar_exact_test(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict:
    """
    McNemar exact test (binomial, two-tailed) for paired comparison.

    Constructs contingency table:
                B correct    B wrong
    A correct      n_00        n_01
    A wrong        n_10        n_11

    Test statistic uses only discordant pairs: n_01 and n_10.

    Returns:
        - n_01: A correct, B wrong
        - n_10: A wrong, B correct
        - n_total: total samples
        - discordant_rate: (n_01 + n_10) / n_total
        - p_value: two-tailed exact binomial p-value
    """
    # Merge on sample identifier (assume same order or use audio_path)
    merged = pd.merge(
        df_a[['audio_path', 'correct']].rename(columns={'correct': 'correct_a'}),
        df_b[['audio_path', 'correct']].rename(columns={'correct': 'correct_b'}),
        on='audio_path',
        how='inner'
    )

    n_total = len(merged)

    # Contingency table
    n_00 = ((merged['correct_a'] == 1) & (merged['correct_b'] == 1)).sum()
    n_01 = ((merged['correct_a'] == 1) & (merged['correct_b'] == 0)).sum()
    n_10 = ((merged['correct_a'] == 0) & (merged['correct_b'] == 1)).sum()
    n_11 = ((merged['correct_a'] == 0) & (merged['correct_b'] == 0)).sum()

    n_discordant = n_01 + n_10
    discordant_rate = n_discordant / n_total if n_total > 0 else 0.0

    # Exact binomial test (two-tailed)
    # Under null: n_01 ~ Binomial(n_01 + n_10, p=0.5)
    if n_discordant == 0:
        p_value = 1.0
    else:
        # Two-tailed: test if n_01 differs from n_10
        # Use binomtest (binom_test is deprecated in scipy >= 1.7)
        p_value = stats.binomtest(k=n_01, n=n_discordant, p=0.5, alternative='two-sided').pvalue

    return {
        'n_00': int(n_00),
        'n_01': int(n_01),
        'n_10': int(n_10),
        'n_11': int(n_11),
        'n_total': int(n_total),
        'n_discordant': int(n_discordant),
        'discordant_rate': float(discordant_rate),
        'p_value': float(p_value)
    }


# ============================================================================
# PART 5: Holm-Bonferroni Correction
# ============================================================================

def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate (default 0.05)

    Returns:
        List of adjusted p-values (same length as input)
    """
    n = len(p_values)

    # Create (index, p_value) pairs and sort by p_value
    indexed_p = list(enumerate(p_values))
    indexed_p.sort(key=lambda x: x[1])

    # Compute adjusted p-values
    adjusted = [None] * n
    for rank, (orig_idx, p) in enumerate(indexed_p, start=1):
        # Holm-Bonferroni: p_adj = min(1, p * (n - rank + 1))
        p_adj = min(1.0, p * (n - rank + 1))

        # Enforce monotonicity: p_adj[i] >= p_adj[i-1]
        if rank > 1:
            prev_p_adj = adjusted[indexed_p[rank - 2][0]]
            p_adj = max(p_adj, prev_p_adj)

        adjusted[orig_idx] = p_adj

    return adjusted


# ============================================================================
# PART 6: Primary Comparisons Report
# ============================================================================

def run_primary_comparisons(
    configs: Dict[str, pd.DataFrame],
    comparisons: List[Tuple[str, str, str]],
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> Dict:
    """
    Run the 4 primary comparisons with full statistical analysis.

    Args:
        configs: Dict mapping config_name -> DataFrame
        comparisons: List of (name_a, name_b, label) tuples
        n_bootstrap: Number of bootstrap resamples
        random_state: Random seed

    Returns:
        Dict with results for each comparison
    """
    results = {}
    p_values = []
    comparison_labels = []

    print("\n" + "=" * 80)
    print("PRIMARY COMPARISONS")
    print("=" * 80)

    for name_a, name_b, label in comparisons:
        print(f"\n{label}: {name_a} vs {name_b}")
        print("-" * 80)

        df_a = configs[name_a]
        df_b = configs[name_b]

        # 1. Compute ΔBA with bootstrap CI
        print("  Computing ΔBA with cluster bootstrap (10,000 resamples)...")
        delta_ba, delta_ci_lower, delta_ci_upper = cluster_bootstrap_delta_ba(
            df_a, df_b, n_bootstrap=n_bootstrap, random_state=random_state
        )

        # 2. McNemar exact test
        print("  Running McNemar exact test...")
        mcnemar_result = mcnemar_exact_test(df_a, df_b)

        results[label] = {
            'config_a': name_a,
            'config_b': name_b,
            'delta_ba': delta_ba,
            'delta_ba_ci': (delta_ci_lower, delta_ci_upper),
            'mcnemar': mcnemar_result,
            'p_value_raw': mcnemar_result['p_value']
        }

        p_values.append(mcnemar_result['p_value'])
        comparison_labels.append(label)

        print(f"  ΔBA = {delta_ba:.4f}, 95% CI: [{delta_ci_lower:.4f}, {delta_ci_upper:.4f}]")
        print(f"  McNemar p-value (raw): {mcnemar_result['p_value']:.6f}")
        print(f"  Discordant rate: {mcnemar_result['discordant_rate']:.4f}")

    # 3. Holm-Bonferroni correction
    print("\n" + "=" * 80)
    print("MULTIPLE COMPARISONS CORRECTION (Holm-Bonferroni)")
    print("=" * 80)

    p_adjusted = holm_bonferroni_correction(p_values, alpha=0.05)

    for label, p_raw, p_adj in zip(comparison_labels, p_values, p_adjusted):
        results[label]['p_value_adjusted'] = float(p_adj)
        results[label]['significant'] = p_adj < 0.05

        print(f"{label}:")
        print(f"  p-value (raw): {p_raw:.6f}")
        print(f"  p-value (adjusted): {p_adj:.6f}")
        print(f"  Significant at α=0.05: {p_adj < 0.05}")

    return results


# ============================================================================
# PART 7: Psychometric Thresholds
# ============================================================================

def parse_variant_value(variant_type: str, condition: str, audio_path: str) -> float:
    """
    Robustly parse variant severity value from condition or audio path.

    Handles multiple formats:
    - Duration: 'dur_20ms', 'dur20ms', 'dur_20.5ms'
    - SNR: 'snr_-10dB', 'snr-10dB', 'snr_+10dB', 'snr10db' (case-insensitive)

    Args:
        variant_type: 'duration' or 'snr'
        condition: Condition string (e.g., 'dur_20ms', 'snr_-10dB')
        audio_path: Full audio path (fallback if condition doesn't match)

    Returns:
        Numeric value (float), or np.nan if no match
    """
    import re

    if variant_type == 'duration':
        # Match: dur_20ms, dur20ms, dur_20.5ms (with/without underscore)
        pattern = r'dur_?(\d+(?:\.\d+)?)ms'
    elif variant_type == 'snr':
        # Match: snr_-10dB, snr-10dB, snr+10dB, snr10db (case-insensitive)
        pattern = r'snr_?([+-]?\d+(?:\.\d+)?)d[bB]'
    else:
        return np.nan

    # Try condition first
    match = re.search(pattern, condition, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Fallback: try audio filename
    filename = os.path.basename(audio_path)
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # No match
    return np.nan


def estimate_threshold_linear(
    values: np.ndarray,
    accuracies: np.ndarray,
    target_acc: float
) -> Tuple[Optional[float], str]:
    """
    Estimate threshold via linear interpolation with censoring support.

    In psychophysics, if the target accuracy is outside the tested range,
    we report a censored threshold at the boundary rather than "undefined".

    Args:
        values: Ordered condition values (e.g., durations in ms, SNRs in dB)
        accuracies: Corresponding accuracies
        target_acc: Target accuracy level (e.g., 0.50, 0.75, 0.90)

    Returns:
        (threshold_value, censoring_flag) where:
        - threshold_value: Interpolated or boundary value
        - censoring_flag: 'ok', 'below_range', 'above_range', or 'failed'
    """
    # Ensure sorted
    sorted_idx = np.argsort(values)
    x = values[sorted_idx]
    y = accuracies[sorted_idx]

    # Check if target is below range (e.g., model too robust)
    if target_acc < y.min():
        return float(x.min()), 'below_range'

    # Check if target is above range (e.g., model too weak)
    if target_acc > y.max():
        return float(x.max()), 'above_range'

    # Linear interpolation (target is within range)
    try:
        f = interp1d(y, x, kind='linear', bounds_error=False, fill_value='extrapolate')
        threshold = float(f(target_acc))
        return threshold, 'ok'
    except:
        return None, 'failed'


def compute_psychometric_thresholds(
    df: pd.DataFrame,
    variant_type: str,
    targets: List[float] = [0.50, 0.75, 0.90]
) -> Dict:
    """
    Compute psychometric thresholds (DT50/75/90 or SNR-75) for a variant type.

    Args:
        df: DataFrame with variant_type, condition, audio_path, correct columns
        variant_type: 'duration' or 'snr'
        targets: List of target accuracy levels

    Returns:
        Dict with thresholds for each target
    """
    # Filter to this variant type
    df_variant = df[df['variant_type'] == variant_type].copy()

    if len(df_variant) == 0:
        return {}

    # Extract numeric values using robust parser
    df_variant['value'] = df_variant.apply(
        lambda row: parse_variant_value(variant_type, row['condition'], row['audio_path']),
        axis=1
    )

    # Drop rows where value couldn't be parsed
    df_variant = df_variant.dropna(subset=['value'])

    if len(df_variant) == 0:
        print(f"  WARNING: No valid {variant_type} values found after parsing")
        return {}

    # Check if we have enough unique values for interpolation
    unique_values = df_variant['value'].nunique()
    if unique_values < 2:
        print(f"  WARNING: {variant_type} has only {unique_values} unique value(s), cannot compute thresholds")
        return {}

    # Compute accuracy per condition
    condition_acc = df_variant.groupby('value')['correct'].mean().reset_index()
    values = condition_acc['value'].values
    accuracies = condition_acc['correct'].values

    thresholds = {}
    for target in targets:
        thresh_value, censoring = estimate_threshold_linear(values, accuracies, target)
        if thresh_value is not None and censoring != 'failed':
            key = f"{'DT' if variant_type == 'duration' else 'SNR'}{int(target * 100)}"
            thresholds[key] = {
                'point': float(thresh_value),
                'censoring': censoring
            }

    return thresholds


def cluster_bootstrap_thresholds(
    df: pd.DataFrame,
    variant_type: str,
    targets: List[float] = [0.50, 0.75, 0.90],
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> Dict:
    """
    Compute psychometric thresholds with cluster bootstrap 95% CIs.

    Returns:
        Dict with {threshold_name: {'point': value, 'censoring': flag, 'ci': (lower, upper)}}
    """
    rng = np.random.RandomState(random_state)

    # Get unique clips
    unique_clips = df['clip_id'].unique()
    n_clips = len(unique_clips)

    # Point estimates (now returns dict with 'point' and 'censoring')
    point_thresholds = compute_psychometric_thresholds(df, variant_type, targets)

    if not point_thresholds:
        return {}

    # Precompute groups (much faster than repeated filtering)
    groups = [group for _, group in df.groupby('clip_id', sort=False)]

    # Bootstrap (collect only numeric values for CI computation)
    bootstrap_thresholds = {key: [] for key in point_thresholds.keys()}

    for _ in tqdm(range(n_bootstrap), desc=f"Bootstrap {variant_type} thresholds", leave=False):
        # Resample clip indices
        sampled_idx = rng.choice(n_clips, size=n_clips, replace=True)

        # Build resampled dataset
        resampled_df = pd.concat([groups[i] for i in sampled_idx], ignore_index=True)

        # Compute thresholds on resampled data
        boot_thresholds = compute_psychometric_thresholds(resampled_df, variant_type, targets)

        for key in point_thresholds.keys():
            if key in boot_thresholds:
                # Extract numeric point value
                bootstrap_thresholds[key].append(boot_thresholds[key]['point'])
            else:
                # Threshold not achievable in this resample
                bootstrap_thresholds[key].append(np.nan)

    # Compute CIs
    results = {}
    for key, point_info in point_thresholds.items():
        boot_samples = np.array(bootstrap_thresholds[key])
        boot_samples = boot_samples[~np.isnan(boot_samples)]

        if len(boot_samples) > 0:
            ci_lower = np.percentile(boot_samples, 2.5)
            ci_upper = np.percentile(boot_samples, 97.5)
        else:
            # No valid bootstrap samples - use point estimate
            ci_lower = ci_upper = point_info['point']

        results[key] = {
            'point': point_info['point'],
            'censoring': point_info['censoring'],
            'ci': (float(ci_lower), float(ci_upper))
        }

    return results


# ============================================================================
# PART 8: Main Analysis Function
# ============================================================================

def run_full_analysis(
    config_paths: Dict[str, str],
    output_dir: str,
    n_bootstrap: int = 10000,
    random_state: int = 42
):
    """
    Run complete statistical analysis.

    Args:
        config_paths: Dict mapping config_name -> predictions.csv path
        output_dir: Directory to save results
        n_bootstrap: Number of bootstrap resamples (default 10,000)
        random_state: Random seed
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all configurations
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    configs = load_multiple_configs(config_paths)

    # Create canonical aliases to avoid KeyError in comparisons
    # (CLI may pass 'lora_opro_classic' but comparisons expect 'lora_opro')
    if 'lora_opro_classic' in configs and 'lora_opro' not in configs:
        configs['lora_opro'] = configs['lora_opro_classic']
        print("  Created alias: lora_opro -> lora_opro_classic")

    if 'base_opro_classic' in configs and 'base_opro' not in configs:
        configs['base_opro'] = configs['base_opro_classic']
        print("  Created alias: base_opro -> base_opro_classic")

    # ========================================================================
    # 1. Per-Configuration Metrics
    # ========================================================================

    print("\n" + "=" * 80)
    print("PER-CONFIGURATION METRICS")
    print("=" * 80)

    config_metrics = {}

    for name, df in configs.items():
        print(f"\n{name}")
        print("-" * 80)

        # Wilson score intervals for recalls
        recalls = compute_recalls_with_wilson(df)

        # Bootstrap CI for BA
        ba, ba_lower, ba_upper = cluster_bootstrap_ba(
            df, n_bootstrap=n_bootstrap, random_state=random_state
        )

        config_metrics[name] = {
            'ba_clip': ba,
            'ba_clip_ci': (ba_lower, ba_upper),
            'recall_speech': recalls['recall_speech'],
            'recall_speech_ci': recalls['recall_speech_ci'],
            'n_speech': recalls['n_speech'],
            'recall_nonspeech': recalls['recall_nonspeech'],
            'recall_nonspeech_ci': recalls['recall_nonspeech_ci'],
            'n_nonspeech': recalls['n_nonspeech']
        }

        print(f"  BA_clip: {ba:.4f}, 95% CI: [{ba_lower:.4f}, {ba_upper:.4f}]")
        print(f"  Recall_Speech: {recalls['recall_speech']:.4f}, 95% CI: {recalls['recall_speech_ci']}")
        print(f"  Recall_NonSpeech: {recalls['recall_nonspeech']:.4f}, 95% CI: {recalls['recall_nonspeech_ci']}")

    # ========================================================================
    # 2. Primary Comparisons (dynamically constructed)
    # ========================================================================

    # Define Qwen2 primary comparisons
    primary_comparisons = [
        ('baseline', 'base_opro', 'Baseline vs Base+OPRO'),
        ('baseline', 'lora', 'Baseline vs LoRA+BasePrompt'),
        ('lora', 'lora_opro', 'LoRA+BasePrompt vs LoRA+OPRO'),
    ]

    # Add LoRA+OPRO_Classic vs LoRA+OPRO_Open if both exist
    if 'lora_opro_classic' in configs and 'lora_opro_open' in configs:
        primary_comparisons.append(
            ('lora_opro_classic', 'lora_opro_open', 'LoRA+OPRO_Classic vs LoRA+OPRO_Open')
        )

    # Add Qwen3 comparison if both exist
    if 'qwen3_baseline' in configs and 'qwen3_opro' in configs:
        primary_comparisons.append(
            ('qwen3_baseline', 'qwen3_opro', 'Qwen3 Baseline vs Qwen3+OPRO')
        )

    comparison_results = run_primary_comparisons(
        configs,
        primary_comparisons,
        n_bootstrap=n_bootstrap,
        random_state=random_state
    )

    # ========================================================================
    # 3. Psychometric Thresholds
    # ========================================================================

    print("\n" + "=" * 80)
    print("PSYCHOMETRIC THRESHOLDS")
    print("=" * 80)

    threshold_results = {}

    for name, df in configs.items():
        print(f"\n{name}")
        print("-" * 80)

        # Duration thresholds (DT50, DT75, DT90)
        print("  Computing duration thresholds...")
        dt_thresholds = cluster_bootstrap_thresholds(
            df, 'duration', targets=[0.50, 0.75, 0.90],
            n_bootstrap=n_bootstrap, random_state=random_state
        )

        # SNR threshold (SNR-75)
        print("  Computing SNR threshold...")
        snr_thresholds = cluster_bootstrap_thresholds(
            df, 'snr', targets=[0.75],
            n_bootstrap=n_bootstrap, random_state=random_state
        )

        threshold_results[name] = {
            'duration': dt_thresholds,
            'snr': snr_thresholds
        }

        # Print results
        for key, val in dt_thresholds.items():
            censoring_note = f" [{val['censoring']}]" if val['censoring'] != 'ok' else ''
            print(f"    {key}: {val['point']:.2f} ms, 95% CI: [{val['ci'][0]:.2f}, {val['ci'][1]:.2f}]{censoring_note}")

        for key, val in snr_thresholds.items():
            censoring_note = f" [{val['censoring']}]" if val['censoring'] != 'ok' else ''
            print(f"    {key}: {val['point']:.2f} dB, 95% CI: [{val['ci'][0]:.2f}, {val['ci'][1]:.2f}]{censoring_note}")

    # ========================================================================
    # 4. Save Results
    # ========================================================================

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results = {
        'config_metrics': config_metrics,
        'primary_comparisons': comparison_results,
        'psychometric_thresholds': threshold_results,
        'bootstrap_params': {
            'n_bootstrap': n_bootstrap,
            'random_state': random_state
        }
    }

    output_path = os.path.join(output_dir, 'statistical_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    # Generate summary report
    report_path = os.path.join(output_dir, 'statistical_report.txt')
    generate_text_report(results, report_path)
    print(f"Text report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def generate_text_report(results: Dict, output_path: str):
    """Generate a human-readable text report."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Configuration metrics
        f.write("1. PER-CONFIGURATION METRICS\n")
        f.write("-" * 80 + "\n\n")

        for name, metrics in results['config_metrics'].items():
            f.write(f"{name}:\n")
            f.write(f"  BA_clip: {metrics['ba_clip']:.4f} "
                   f"[{metrics['ba_clip_ci'][0]:.4f}, {metrics['ba_clip_ci'][1]:.4f}]\n")
            f.write(f"  Recall_Speech: {metrics['recall_speech']:.4f} "
                   f"[{metrics['recall_speech_ci'][0]:.4f}, {metrics['recall_speech_ci'][1]:.4f}] "
                   f"(n={metrics['n_speech']})\n")
            f.write(f"  Recall_NonSpeech: {metrics['recall_nonspeech']:.4f} "
                   f"[{metrics['recall_nonspeech_ci'][0]:.4f}, {metrics['recall_nonspeech_ci'][1]:.4f}] "
                   f"(n={metrics['n_nonspeech']})\n\n")

        # Primary comparisons
        f.write("\n2. PRIMARY COMPARISONS (Holm-Bonferroni corrected)\n")
        f.write("-" * 80 + "\n\n")

        for label, comp in results['primary_comparisons'].items():
            f.write(f"{label}:\n")
            f.write(f"  ΔBA: {comp['delta_ba']:.4f} "
                   f"[{comp['delta_ba_ci'][0]:.4f}, {comp['delta_ba_ci'][1]:.4f}]\n")
            f.write(f"  p-value (raw): {comp['p_value_raw']:.6f}\n")
            f.write(f"  p-value (adjusted): {comp['p_value_adjusted']:.6f}\n")
            f.write(f"  Significant (α=0.05): {comp['significant']}\n")
            f.write(f"  Discordant rate: {comp['mcnemar']['discordant_rate']:.4f}\n")
            f.write(f"  McNemar table: n_01={comp['mcnemar']['n_01']}, "
                   f"n_10={comp['mcnemar']['n_10']}\n\n")

        # Psychometric thresholds
        f.write("\n3. PSYCHOMETRIC THRESHOLDS\n")
        f.write("-" * 80 + "\n\n")

        for name, thresholds in results['psychometric_thresholds'].items():
            f.write(f"{name}:\n")

            if 'duration' in thresholds:
                f.write("  Duration thresholds:\n")
                for key, val in thresholds['duration'].items():
                    censoring_note = f" [{val['censoring']}]" if val['censoring'] != 'ok' else ''
                    f.write(f"    {key}: {val['point']:.2f} ms "
                           f"[{val['ci'][0]:.2f}, {val['ci'][1]:.2f}]{censoring_note}\n")

            if 'snr' in thresholds:
                f.write("  SNR thresholds:\n")
                for key, val in thresholds['snr'].items():
                    censoring_note = f" [{val['censoring']}]" if val['censoring'] != 'ok' else ''
                    f.write(f"    {key}: {val['point']:.2f} dB "
                           f"[{val['ci'][0]:.2f}, {val['ci'][1]:.2f}]{censoring_note}\n")

            f.write("\n")


# ============================================================================
# PART 9: CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rigorous statistical analysis for model comparison"
    )
    parser.add_argument(
        '--baseline', type=str, required=True,
        help='Path to baseline predictions.csv'
    )
    parser.add_argument(
        '--base_opro', type=str, required=True,
        help='Path to base+OPRO predictions.csv'
    )
    parser.add_argument(
        '--lora', type=str, required=True,
        help='Path to LoRA+BasePrompt predictions.csv'
    )
    parser.add_argument(
        '--lora_opro', type=str, required=True,
        help='Path to LoRA+OPRO (classic) predictions.csv'
    )
    parser.add_argument(
        '--lora_opro_open', type=str, default=None,
        help='Path to LoRA+OPRO (open) predictions.csv (optional)'
    )
    parser.add_argument(
        '--qwen3_baseline', type=str, default=None,
        help='Path to Qwen3 Baseline predictions.csv (optional)'
    )
    parser.add_argument(
        '--qwen3_opro', type=str, default=None,
        help='Path to Qwen3 OPRO predictions.csv (optional)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--n_bootstrap', type=int, default=10000,
        help='Number of bootstrap resamples (default: 10000)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Build config paths
    config_paths = {
        'baseline': args.baseline,
        'base_opro': args.base_opro,
        'lora': args.lora,
        'lora_opro_classic': args.lora_opro
    }

    if args.lora_opro_open:
        config_paths['lora_opro_open'] = args.lora_opro_open

    if args.qwen3_baseline:
        config_paths['qwen3_baseline'] = args.qwen3_baseline

    if args.qwen3_opro:
        config_paths['qwen3_opro'] = args.qwen3_opro

    # Run analysis
    run_full_analysis(
        config_paths=config_paths,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        random_state=args.seed
    )


if __name__ == '__main__':
    main()
