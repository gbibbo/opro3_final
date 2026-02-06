#!/usr/bin/env python3
"""
Output normalization for multi-format speech detection prompts.

Normalizes model outputs from different prompt formats (A/B, MC, labels, open)
to binary SPEECH/NONSPEECH labels with confidence scores.
"""

import re
from typing import Optional, Dict, List, Tuple


def normalize_to_binary(
    text: str,
    probs: Optional[Dict[str, float]] = None,
    mode: str = "auto",
    mapping: Optional[Dict[str, str]] = None,
    verbalizers: Optional[List[str]] = None,
) -> Tuple[Optional[str], float]:
    """
    Normalize model output to binary SPEECH/NONSPEECH label.

    Priority order (highest to lowest):
    1. NONSPEECH/NON-SPEECH (checked first to avoid substring match with SPEECH)
    2. SPEECH (only if NONSPEECH wasn't found)
    3. Letter mapping (A/B/C/D) via provided mapping dict
    4. Yes/No responses
    5. Synonyms (voice/talking vs music/noise/silence)
    6. Unknown (returns None)

    Semantic labels win over letters in ambiguous cases (e.g., "B) SPEECH" → SPEECH).

    Args:
        text: Raw model output text
        probs: Dict of token probabilities (optional, for confidence)
        mode: Format mode ("ab", "mc", "labels", "open", "auto")
        mapping: Dict mapping letters to labels (e.g., {"A": "SPEECH", "B": "NONSPEECH"})
        verbalizers: List of valid label strings (e.g., ["SPEECH", "NONSPEECH"])

    Returns:
        (label, confidence): Binary label (SPEECH/NONSPEECH/None) and confidence score

    Examples:
        >>> normalize_to_binary("A", mapping={"A": "SPEECH", "B": "NONSPEECH"})
        ('SPEECH', 1.0)

        >>> normalize_to_binary("NONSPEECH")
        ('NONSPEECH', 1.0)

        >>> normalize_to_binary("I hear music", mode="open")
        ('NONSPEECH', 0.8)
    """
    if not text:
        return None, 0.0

    # Normalize text
    text_clean = text.strip().upper()
    text_lower = text.strip().lower()

    # Default verbalizers
    if verbalizers is None:
        verbalizers = ["SPEECH", "NONSPEECH"]

    # Default confidence
    confidence = 1.0

    # Extract probability if available
    if probs:
        # Try to get confidence from first token probability
        if "p_first_token" in probs:
            confidence = probs["p_first_token"]

    # Priority 1: Check for NONSPEECH/NON-SPEECH FIRST (before SPEECH)
    # This avoids the substring bug where "NONSPEECH" contains "SPEECH"
    if (
        "NONSPEECH" in text_clean
        or "NON-SPEECH" in text_clean
        or "NON SPEECH" in text_clean
        or "NO SPEECH" in text_clean
    ):
        # Make sure it's not double-negated like "NOT NONSPEECH"
        if "NOT NONSPEECH" not in text_clean and "NOT NON-SPEECH" not in text_clean:
            return "NONSPEECH", confidence

    # Priority 2: Exact match with SPEECH (only if NONSPEECH wasn't found)
    if "SPEECH" in text_clean:
        # Check it's not negated
        if "NOT SPEECH" not in text_clean:
            return "SPEECH", confidence

    # Priority 3: Letter mapping (A/B/C/D)
    if mapping:
        # Extract first letter from response
        letter_match = re.match(r"^([A-D])", text_clean)
        if letter_match:
            letter = letter_match.group(1)
            if letter in mapping:
                label = mapping[letter]
                # Update confidence if we have letter probabilities
                if probs and letter in probs:
                    confidence = probs[letter]
                return label, confidence

    # Priority 4: Yes/No responses
    # Use word boundaries to avoid false positives (e.g., "SI" in "MUSIC", "NO" in "NOT")
    yes_patterns = ["YES", "SÍ", "SI", "AFFIRMATIVE", "TRUE", "CORRECT", "PRESENT"]
    no_patterns = ["NO", "NEGATIVE", "FALSE", "INCORRECT", "ABSENT", "NOT PRESENT"]

    for pattern in yes_patterns:
        # Use word boundaries to match whole words only
        if re.search(r'\b' + re.escape(pattern) + r'\b', text_clean):
            return "SPEECH", confidence * 0.95  # Slightly lower confidence for yes/no

    for pattern in no_patterns:
        # Use word boundaries to match whole words only
        if re.search(r'\b' + re.escape(pattern) + r'\b', text_clean):
            return "NONSPEECH", confidence * 0.95

    # Priority 5: Synonyms and semantic content
    speech_synonyms = [
        "voice",
        "voices",
        "talking",
        "spoken",
        "speaking",
        "speaker",
        "conversation",
        "conversational",
        "words",
        "utterance",
        "vocal",
        "human voice",
        "person talking",
        "dialogue",
        "speech",
        "syllables",
        "phonemes",
        "formants",
    ]

    nonspeech_synonyms = [
        "music",
        "musical",
        "song",
        "melody",
        "instrumental",
        "beep",
        "beeps",
        "tone",
        "tones",
        "pitch",
        "sine wave",
        "noise",
        "noisy",
        "static",
        "hiss",
        "white noise",
        "silence",
        "silent",
        "quiet",
        "nothing",
        "empty",
        "ambient",
        "environmental",
        "background",
        "click",
        "clicks",
        "clock",
        "tick",
        "ticking",
    ]

    # Count matches
    speech_score = sum(1 for syn in speech_synonyms if syn in text_lower)
    nonspeech_score = sum(1 for syn in nonspeech_synonyms if syn in text_lower)

    if speech_score > nonspeech_score:
        return "SPEECH", confidence * 0.8  # Lower confidence for synonym matching
    elif nonspeech_score > speech_score:
        return "NONSPEECH", confidence * 0.8

    # Priority 6: Unknown/unparseable
    return None, 0.0


def detect_format(text: str) -> str:
    """
    Auto-detect prompt format from text.

    Args:
        text: Prompt text

    Returns:
        Format string: "ab", "mc", "labels", or "open"
    """
    text_upper = text.upper()

    # Check for multiple choice with D option
    if "A)" in text_upper and "D)" in text_upper:
        return "mc"

    # Check for A/B binary
    if ("A)" in text_upper and "B)" in text_upper) or (
        "OPTION A" in text_upper and "OPTION B" in text_upper
    ):
        return "ab"

    # Check for explicit labels
    if "SPEECH" in text_upper and "NONSPEECH" in text_upper:
        return "labels"

    # Default to open
    return "open"


def validate_mapping(mapping: Optional[Dict[str, str]], label_space: List[str]) -> bool:
    """
    Validate that mapping dict maps to valid labels.

    Args:
        mapping: Letter to label mapping
        label_space: Valid label values

    Returns:
        True if valid, False otherwise
    """
    if not mapping:
        return True

    for letter, label in mapping.items():
        if label not in label_space:
            return False

    return True


##############################################################################
# LLM Fallback Confidence Levels (for paper reference: main.tex)
#
# These constants define the confidence scores returned by llm_fallback_interpret:
##############################################################################
LLM_FALLBACK_CONFIDENCE_CLEAR = 0.7      # Clear classification (one category dominant)
LLM_FALLBACK_CONFIDENCE_MIXED = 0.6      # Mixed content (both speech and nonspeech mentioned)
LLM_FALLBACK_CONFIDENCE_UNKNOWN = 0.0    # Unable to determine


def llm_fallback_interpret(
    response: str,
    model=None,
    prompt_template: Optional[str] = None,
) -> Tuple[Optional[str], float]:
    """
    LLM fallback for ambiguous responses that couldn't be parsed by normalize_to_binary.

    Uses heuristic-based semantic analysis to interpret the meaning of the response
    and determine if it indicates speech detection or not.

    Confidence Levels (for paper reference):
        - 0.7 (LLM_FALLBACK_CONFIDENCE_CLEAR): Clear classification with dominant indicators
        - 0.6 (LLM_FALLBACK_CONFIDENCE_MIXED): Mixed content with both speech and non-speech
        - 0.0 (LLM_FALLBACK_CONFIDENCE_UNKNOWN): Unable to determine, returns (None, 0.0)

    Args:
        response: The ambiguous model response text
        model: Optional classifier instance (reserved for future LLM-based interpretation)
        prompt_template: Optional custom prompt template (reserved for future use)

    Returns:
        (label, confidence): Binary label (SPEECH/NONSPEECH/None) and confidence score.

    Examples:
        >>> llm_fallback_interpret("The audio has music and maybe some talking")
        ('SPEECH', 0.6)  # Mixed content, speech present

        >>> llm_fallback_interpret("Just random noise")
        ('NONSPEECH', 0.7)  # Clear non-speech indicator

        >>> llm_fallback_interpret("I hear voices speaking")
        ('SPEECH', 0.7)  # Clear speech indicator
    """
    if not response or not response.strip():
        return None, 0.0

    # Default prompt template for interpretation
    if prompt_template is None:
        prompt_template = """The following text is a response from an audio classification model describing what it heard in an audio clip:

"{response}"

Based on this response, does it indicate that HUMAN SPEECH was detected in the audio?

Consider:
- If it mentions voices, talking, speaking, conversations, words, or speech → SPEECH
- If it mentions music, noise, silence, beeping, tones, or environmental sounds WITHOUT mentioning speech → NONSPEECH
- If it's ambiguous or mentions BOTH speech and non-speech, determine which is PRIMARY

Answer ONLY with one word: SPEECH or NONSPEECH"""

    # Format the prompt
    interpretation_prompt = prompt_template.format(response=response)

    # If model is provided, use it to interpret
    if model is not None:
        try:
            # Use the model to interpret (text-only, no audio)
            # This requires a text-only mode or we skip actual inference
            # For now, we'll implement a simpler heuristic-based approach
            pass
        except Exception:
            pass

    # Fallback: Use heuristic analysis with stronger semantic understanding
    # This is a more sophisticated version of the synonym matching
    response_lower = response.lower()

    # Strong speech indicators (person actively producing speech)
    strong_speech = [
        "talking", "speaking", "conversation", "voice", "voices",
        "said", "says", "spoken", "words", "dialogue", "narrator",
        "person speaking", "human voice", "someone talking"
    ]

    # Strong non-speech indicators (no human speech production)
    strong_nonspeech = [
        "music", "musical", "song", "melody", "instrumental",
        "beep", "tone", "noise", "static", "silence", "quiet",
        "environmental", "background", "nature sounds"
    ]

    # Negation patterns (flips the meaning)
    has_negation = any(neg in response_lower for neg in [
        "no voice", "no speech", "no talking", "not speaking",
        "without voice", "without speech", "no human"
    ])

    # Count strong indicators
    speech_count = sum(1 for indicator in strong_speech if indicator in response_lower)
    nonspeech_count = sum(1 for indicator in strong_nonspeech if indicator in response_lower)

    # Decision logic
    if has_negation:
        # Negation detected, likely NONSPEECH
        return "NONSPEECH", LLM_FALLBACK_CONFIDENCE_CLEAR

    if speech_count > nonspeech_count:
        # More speech indicators
        return "SPEECH", LLM_FALLBACK_CONFIDENCE_CLEAR
    elif nonspeech_count > speech_count:
        # More non-speech indicators
        return "NONSPEECH", LLM_FALLBACK_CONFIDENCE_CLEAR
    elif speech_count > 0 and nonspeech_count > 0:
        # Mixed content: if ANY speech mentioned, prioritize SPEECH
        # (for speech detection task, presence of speech is what matters)
        return "SPEECH", LLM_FALLBACK_CONFIDENCE_MIXED
    else:
        # No strong indicators found, truly ambiguous
        return None, LLM_FALLBACK_CONFIDENCE_UNKNOWN
