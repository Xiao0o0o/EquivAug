"""
GRPO reward for GSM8K: 1.0 if the extracted final answer equals the gold answer, else 0.0.
The answer is read from "Final answer: <x>" (the SFT format), falling back to
"The answer is: <x>" and "#### <x>".

Usage in verl:
    custom_reward_function.path=reward_gsm8k.py
    custom_reward_function.name=compute_score
"""
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Optional


# =============================================
# Numeric comparison helpers
# =============================================

def _normalize_numeric_token(token: str) -> str:
    token = token.replace(",", "")
    token = token.strip()
    token = token.rstrip(' ."\'')
    if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
        if "." in token:
            token = token.rstrip("0").rstrip(".") or "0"
        return token
    return token


def _try_parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    text = text.rstrip(' ."\'')
    frac_match = re.fullmatch(r"-?\d+\s*/\s*-?\d+", text)
    if frac_match:
        try:
            return float(Fraction(text))
        except (ZeroDivisionError, ValueError):
            return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        try:
            return float(Decimal(text))
        except (InvalidOperation, ValueError):
            return None
    return None


def is_numerically_equal(ans1, ans2, tolerance: float = 1e-9) -> bool:
    num1 = _try_parse_number(ans1)
    num2 = _try_parse_number(ans2)
    if num1 is not None and num2 is not None:
        return abs(num1 - num2) < tolerance
    norm1 = _normalize_numeric_token(str(ans1))
    norm2 = _normalize_numeric_token(str(ans2))
    return norm1 == norm2


# =============================================
# Answer extraction
# =============================================

def extract_answer_from_model_output(ans_text: str) -> str:
    """
    Extract the answer from the model output.
    "Final answer:" is preferred; other formats are used as fallbacks.
    """
    # Remove <think> block
    cleaned_text = re.sub(
        r"^.*?</think>", "", ans_text, flags=re.DOTALL | re.IGNORECASE
    )

    # 1) Preferred: "Final answer:"
    matches = list(
        re.finditer(r"Final answer[^:]*:\s*([^\n]+)", cleaned_text, re.IGNORECASE)
    )

    # 2) Fallback: "The answer is:"
    if not matches:
        matches = list(
            re.finditer(r"The answer is[^:]*:\s*([^\n]+)", cleaned_text, re.IGNORECASE)
        )

    # 3) Fallback: "####" (original GSM8K format)
    if not matches:
        matches = list(re.finditer(r"####\s*(.+)", cleaned_text))

    if not matches:
        return ans_text.strip()

    ans = matches[-1].group(1).strip()

    # Strip trailing punctuation
    if re.match(r"^\d+\.$", ans):
        ans = ans[:-1]
    elif re.match(r"^\d+\.\d+\.$", ans):
        ans = ans[:-1]
    ans = ans.rstrip('."\'')
    ans = ans.strip()

    # Handle multiple decimal points
    first_dot = ans.find(".")
    if first_dot != -1:
        second_dot = ans.find(".", first_dot + 1)
        if second_dot != -1:
            ans = ans[:second_dot]

    ans = ans.strip()

    # Extract numeric candidates
    numeric_candidates = re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+\s*/\s*-?\d+",
        ans,
    )
    if numeric_candidates:
        candidate = numeric_candidates[-1]
        candidate = candidate.replace(" ", "")
        return candidate

    stripped = re.sub(r"[^0-9\-/.,]", "", ans).strip(" ,")
    return stripped if stripped else ans


def extract_answer_from_ground_truth(gt_text: str) -> str:
    """Extract the answer from the ground truth (GSM8K "####" format)."""
    match = re.search(r"####\s*(.+)", gt_text)
    if match:
        return match.group(1).strip()
    return gt_text.strip()


# =============================================
# verl reward function interface
# =============================================

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Custom reward function called by verl.

    Parameters (passed by verl):
        data_source: dataset name
        solution_str: generated response
        ground_truth: gold answer
        extra_info: optional extra information

    Returns:
        float: 1.0 = correct, 0.0 = wrong
    """
    model_answer = extract_answer_from_model_output(solution_str)
    gt_answer = extract_answer_from_ground_truth(str(ground_truth))

    if is_numerically_equal(model_answer, gt_answer):
        return 1.0
    else:
        return 0.0