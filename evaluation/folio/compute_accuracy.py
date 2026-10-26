"""
Evaluation (FOLIO): compute accuracy from the pickle written by generate_vllm.py.

Reports the mean accuracy over all sampled responses (ACC reported in the paper)
and Pass@k.

Usage:
    python compute_accuracy.py --results-path model_responses_vllm/folio_validation_5responses_1024tokens.pkl --k 1 5
"""

import argparse
import os
import pickle
import re
from math import comb
from typing import Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accuracy from vLLM FOLIO response pickles."
    )
    parser.add_argument(
        "--results-path",
        required=True,
        help="Path to the pickle produced by generate_vllm.py.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Values of k for Pass@k (e.g., --k 1 5 10 32 64)",
    )
    return parser.parse_args()


def extract_generate_answer(ans_text: str) -> str:
    """Extract label: true/false/uncertain from model generation."""
    # 1) remove <think>...</think> or any leading stuff ending with </think>
    cleaned = re.sub(r"^.*?</think>", "", ans_text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2) priority patterns: "the final answer is: True" etc. (case-insensitive)
    #    capture group only allows true/false/uncertain/unknown (word-boundary safe)
    priority_patterns = [
        r"\bthe\s+final\s+answer\s+is\b\s*[:\-]?\s*(true|false|uncertain|unknown)\b",
        r"\bfinal\s+answer\b\s*[:\-]?\s*(true|false|uncertain|unknown)\b",
        r"\bthe\s+answer\s+is\b\s*[:\-]?\s*(true|false|uncertain|unknown)\b",
        r"\banswer\b\s*[:\-]?\s*(true|false|uncertain|unknown)\b",
    ]

    for pat in priority_patterns:
        m = re.search(pat, cleaned, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).lower()
            return "uncertain" if ans == "unknown" else ans

    # 3) fallback: take the LAST occurrence in the entire text
    #    Use word boundaries to avoid matching inside other words.
    matches = list(re.finditer(r"\b(true|false|uncertain|unknown)\b", cleaned, flags=re.IGNORECASE))
    if matches:
        ans = matches[-1].group(1).lower()
        return "uncertain" if ans == "unknown" else ans

    return "unknown"


def normalize_label(label: str) -> str:
    """Normalize label to one of: true, false, uncertain."""
    label = label.lower().strip()
    if "true" in label:
        return "true"
    elif "false" in label:
        return "false"
    elif "uncertain" in label or "unknown" in label:
        return "uncertain"
    return label


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of Pass@k.

    n: total number of samples
    c: number of correct samples
    k: target k

    Pass@k = 1 - C(n-c, k) / C(n, k)

    If n-c < k (i.e., not enough wrong answers to fill k slots),
    then Pass@k = 1.0 (guaranteed at least one correct in k samples).
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def evaluate(results: Dict[Any, Dict[str, Any]], k_values: list) -> None:
    example = next(iter(results.values()))
    num_responses = len(example["generations"])

    # For each question, count how many of the n generations are correct
    per_question_stats = []
    for qid, item in results.items():
        gt_label = normalize_label(item["label"])
        n = len(item["generations"])
        c = 0
        for gen in item["generations"]:
            gen_answer = extract_generate_answer(gen)
            pred_label = normalize_label(gen_answer)
            if pred_label == gt_label:
                c += 1
        per_question_stats.append({"qid": qid, "n": n, "c": c})

    total_questions = len(per_question_stats)

    # Overall Pass@1 accuracy (= average correctness rate)
    total_correct = sum(s["c"] for s in per_question_stats)
    total_samples = sum(s["n"] for s in per_question_stats)
    avg_accuracy = total_correct / total_samples if total_samples else 0.0
    print(f"Average Accuracy (across all samples): {total_correct}/{total_samples} = {avg_accuracy:.4f}")
    print()

    # Compute Pass@k for each requested k
    print(f"{'Pass@k':<12} {'Value':<10} {'# Questions Passed'}")
    print("-" * 45)
    for k in sorted(k_values):
        if k > num_responses:
            print(f"Pass@{k:<6}  skipped (k={k} > n={num_responses})")
            continue

        # Pass@k for each question, then average
        question_pass_scores = []
        for s in per_question_stats:
            score = pass_at_k(s["n"], s["c"], k)
            question_pass_scores.append(score)

        avg_pass_k = sum(question_pass_scores) / total_questions
        num_passed = sum(1 for sc in question_pass_scores if sc >= 1.0)
        print(f"Pass@{k:<6}  {avg_pass_k:.4f}     {num_passed}/{total_questions}")

    # Print per-question details
    print()
    print("Per-question breakdown (sorted by correct count):")
    print(f"{'QID':<8} {'Correct':<10} {'Total':<8} {'Rate':<8}", end="")
    for k in sorted(k_values):
        if k <= num_responses:
            print(f" {'Pass@'+str(k):<10}", end="")
    print()
    print("-" * (36 + 10 * len([k for k in k_values if k <= num_responses])))

    for s in sorted(per_question_stats, key=lambda x: x["c"]):
        print(f"{s['qid']:<8} {s['c']:<10} {s['n']:<8} {s['c']/s['n']:<8.4f}", end="")
        for k in sorted(k_values):
            if k <= num_responses:
                score = pass_at_k(s["n"], s["c"], k)
                print(f" {score:<10.4f}", end="")
        print()


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
    print(f"Responses per question: {len(next(iter(results.values()))['generations'])}")
    print()
    evaluate(results, args.k)


if __name__ == "__main__":
    main()