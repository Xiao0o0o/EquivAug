"""
GRPO reward for FOLIO logical reasoning.

This file is passed to verl via:
    custom_reward_function.path=reward_folio.py
    custom_reward_function.name=compute_score

The compute_score function receives the model's response and the
ground_truth label, extracts True/False/Uncertain from the response,
and returns a reward score.
"""

import re


def extract_folio_answer(response_text: str) -> str:
    """
    Extract the predicted label (true/false/uncertain) from model generation.

    Priority:
      1. Explicit "Final answer: <label>" patterns
      2. "The answer is: <label>" patterns
      3. Last occurrence of true/false/uncertain/unknown in text
    """
    # Remove <think>...</think> blocks (for reasoning models)
    cleaned = re.sub(
        r"^.*?</think>", "", response_text, flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # Priority patterns (most specific → least specific)
    # Target format: "Therefore, the answer is: True."
    priority_patterns = [
        r"\bthe\s+answer\s+is\b\s*[:\-]?\s*(true|false|uncertain|unknown)\b",
        r"\bthe\s+final\s+answer\s+is\b\s*[:\-]?\s*(true|false|uncertain|unknown)\b",
        r"\bfinal\s+answer\b\s*[:\-]?\s*(true|false|uncertain|unknown)\b",
        r"\banswer\b\s*[:\-]?\s*(true|false|uncertain|unknown)\b",
    ]

    for pat in priority_patterns:
        m = re.search(pat, cleaned, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).lower()
            return "uncertain" if ans == "unknown" else ans

    # Fallback: last occurrence of a label keyword
    matches = list(
        re.finditer(
            r"\b(true|false|uncertain|unknown)\b", cleaned, flags=re.IGNORECASE
        )
    )
    if matches:
        ans = matches[-1].group(1).lower()
        return "uncertain" if ans == "unknown" else ans

    return "unknown"  # no label found


def normalize_gt(label: str) -> str:
    """Normalize ground truth to lowercase: true / false / uncertain."""
    label = label.strip().lower()
    if "true" in label:
        return "true"
    if "false" in label:
        return "false"
    if "uncertain" in label or "unknown" in label:
        return "uncertain"
    return label


# --------------------------------------------------------------------------
# verl interface
# --------------------------------------------------------------------------

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Reward function called by verl's reward manager.

    Args:
        data_source (str): Dataset name, e.g. "folio".
        solution_str (str): The model's generated response text.
        ground_truth (str): The correct label ("True", "False", or "Uncertain").
        extra_info (dict, optional): Additional metadata.

    Returns:
        float: Reward score.
            1.0  — correct answer with proper "the answer is:" format
            0.5  — correct answer but without proper format
            0.1  — wrong answer but has proper "the answer is:" format
            0.0  — wrong answer, no proper format
    """
    pred = extract_folio_answer(solution_str)
    gt = normalize_gt(ground_truth)

    # Check if the response uses the expected format: "the answer is: <label>"
    has_format = bool(
        re.search(r"\bthe\s+answer\s+is\b", solution_str, re.IGNORECASE)
    )

    correct = pred == gt

    if correct and has_format:
        return 1.0
    elif correct and not has_format:
        return 0.5
    elif not correct and has_format:
        return 0.1
    else:
        return 0.0