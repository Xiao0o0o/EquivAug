"""
Evaluation (GSM8K): compute accuracy from the pickle written by generate_vllm.py.

Reports the accuracy of each sampled response, the mean accuracy over all
responses (ACC reported in the paper) and Pass@k.

Usage:
    python compute_accuracy.py --results-path model_responses_vllm/gsm8k_test_5responses_1024tokens.pkl
"""

import argparse
import os
import pickle
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Dict, Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accuracy from vLLM GSM8K response pickles."
    )
    parser.add_argument(
        "--results-path",
        required=True,
        help="Path to the pickle produced by generate_vllm.py.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Numeric tolerance when comparing answers.",
    )
    return parser.parse_args()


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


def extract_generate_answer(ans_text: str) -> str:
    cleaned_text = re.sub(r"^.*?</think>", "", ans_text, flags=re.DOTALL | re.IGNORECASE)
    matches = list(
        re.finditer(r"Final answer[^:]*:\s*([^\n]+)", cleaned_text, re.IGNORECASE)
    )
    if not matches:
        return ans_text.strip()

    ans = matches[-1].group(1).strip()
    if re.match(r"^\d+\.$", ans):
        ans = ans[:-1]
    elif re.match(r"^\d+\.\d+\.$", ans):
        ans = ans[:-1]
    ans = ans.rstrip('."\'' )
    ans = ans.strip()

    first_dot = ans.find(".")
    if first_dot != -1:
        second_dot = ans.find(".", first_dot + 1)
        if second_dot != -1:
            ans = ans[:second_dot]

    ans = ans.strip()

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


def evaluate(results: Dict[Any, Dict[str, Any]], tolerance: float) -> None:
    example = next(iter(results.values()))
    num_responses = len(example["generations"])

    # Accuracy of each response index
    correct_count = [0 for _ in range(num_responses)]

    # A question counts as solved if any of its responses is correct (Pass@k)
    any_correct_count = 0

    for item in results.values():
        gt_answer = item["answer"]

        per_item_correct_flags = []
        for i, gen in enumerate(item["generations"]):
            gen_answer = extract_generate_answer(gen)
            ok = is_numerically_equal(gen_answer, gt_answer, tolerance)
            per_item_correct_flags.append(ok)
            if ok:
                correct_count[i] += 1

        if any(per_item_correct_flags):
            any_correct_count += 1

    total_count = len(results)

    # Per-response accuracy
    for i, count in enumerate(correct_count):
        ratio = count / total_count if total_count else 0.0
        print(f"Response {i}: Correctness Ratio = {count}/{total_count} = {ratio:.4f}")

    # Pass@k (k = num_responses)
    any_ratio = any_correct_count / total_count if total_count else 0.0
    print(
        f"Any-of-{num_responses} (Pass@{num_responses}): Accuracy = "
        f"{any_correct_count}/{total_count} = {any_ratio:.4f}"
    )

    # Mean accuracy over all responses
    all_correct = sum(correct_count)
    all_ratio = all_correct / (total_count * num_responses) if total_count else 0.0
    print(
        f"All Responses (avg over {num_responses}): Correctness Ratio = "
        f"{all_correct}/{total_count * num_responses} = {all_ratio:.4f}"
    )


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.results_path):
        raise FileNotFoundError(f"Results file not found: {args.results_path}")

    with open(args.results_path, "rb") as f:
        results = pickle.load(f)

    if not results:
        print("No results found in the pickle. Nothing to evaluate.")
        return

    print(f"Loaded {len(results)} examples from {args.results_path}")
    evaluate(results, args.tolerance)


if __name__ == "__main__":
    main()
