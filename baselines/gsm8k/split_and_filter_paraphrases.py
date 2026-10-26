#!/usr/bin/env python3
"""
Baseline (GSM8K): post-process the output of paraphrasing.py into an SFT set.

1. Split the paraphrases that were returned together in one "answer" field
   (separated by '---' or by repeated '#### <answer>' lines) into one line each.
2. Keep only the questions that also appear in the EquivAug Step 3 output, so
   that all methods are trained on the same problems.
3. Append "Final answer: <answer>" (the format used by all SFT datasets) and shuffle.

Usage:
    python split_and_filter_paraphrases.py \
        --input gsm8k_paraphrased.jsonl \
        --filter_file train_expr_explained_all.jsonl \
        --output paraphrasing.jsonl
"""

import argparse
import json
import random
import re


def split_answer(answer_text: str, gt_number: str) -> list:
    """Split several paraphrases that were returned in one string."""

    # Strategy 1: split on '---' separators (also '----', '--- ' etc.)
    parts = re.split(r"\n\s*-{3,}\s*\n", answer_text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > 1:
        return parts

    # Strategy 2: split after each '#### <number>'
    gt_clean = gt_number.strip().replace(",", "")
    pattern = rf"(####\s*{re.escape(gt_clean)})\s*\n"
    segments = re.split(pattern, answer_text)

    # re.split keeps the captured separator, so glue each text back to its '#### <number>'
    result = []
    i = 0
    while i < len(segments):
        if i + 1 < len(segments) and re.match(rf"####\s*{re.escape(gt_clean)}$", segments[i + 1].strip()):
            combined = (segments[i] + segments[i + 1]).strip()
            if combined:
                result.append(combined)
            i += 2
        else:
            seg = segments[i].strip()
            if seg:
                result.append(seg)
            i += 1

    if len(result) > 1:
        return result

    # Fallback: keep the whole text as a single paraphrase
    return [answer_text.strip()]


def add_final_answer(expl: str) -> str:
    """Append 'Final answer: <x>' taken from the '#### <x>' line."""
    expl = expl.rstrip()
    match = re.search(r'####\s*(.+)', expl)
    if match:
        answer = match.group(1).strip()
        if not expl.endswith(f"Final answer: {answer}"):
            expl += f"\n Final answer: {answer}"
    return expl


def main():
    parser = argparse.ArgumentParser(description="Split, filter and format GSM8K paraphrases for SFT.")
    parser.add_argument("--input", required=True, help="Output of paraphrasing.py")
    parser.add_argument("--filter_file", required=True,
                        help="JSONL whose 'question' set defines the kept questions (Step 3 *_all.jsonl)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Step 1: split
    split_results = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for exp in split_answer(item["answer"], item["gt_number"]):
                split_results.append({
                    "question": item["question"],
                    "expr_explanations": [exp],
                    "original_answer": item["original_answer"],
                    "gt_number": item["gt_number"],
                })
    print(f"Split lines: {len(split_results)}")

    # Step 2: filter by question
    allowed_questions = set()
    with open(args.filter_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                allowed_questions.add(json.loads(line)["question"].strip())
    filtered = [item for item in split_results if item["question"].strip() in allowed_questions]
    print(f"After filtering: {len(filtered)} lines ({len(allowed_questions)} allowed questions)")

    # Step 3: format and shuffle
    for item in filtered:
        item["expr_explanations"] = [add_final_answer(e) for e in item["expr_explanations"]]
    random.seed(args.seed)
    random.shuffle(filtered)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(filtered)} examples to {args.output}")


if __name__ == "__main__":
    main()
