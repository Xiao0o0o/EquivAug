#!/usr/bin/env python3
"""
Baseline (GSM8K): Ground Truth.

Build the Ground-Truth SFT set from the annotated GSM8K solutions. For a fair
comparison we keep the same questions as EquivAug (records whose expression was
verified in Step 2) and repeat each solution --repeat times so that the number
of training steps is comparable to the augmented datasets.

The "#### <answer>" line is rewritten as "Final answer: <answer>", the format
used by all SFT datasets in this repo.

Usage:
    python ground_truth.py \
        --input train_expr_variants.jsonl \
        --output ground_truth.jsonl --repeat 5
"""

import argparse
import json
import random
import re


def main():
    parser = argparse.ArgumentParser(description="Build the GSM8K Ground-Truth SFT set.")
    parser.add_argument("--input", required=True,
                        help="JSONL with 'question', 'answer' and 'answer_match' (Step 2 output)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=5,
                        help="Number of copies of each ground-truth solution")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("answer_match", True):
                continue
            answer = re.sub(r"####\s*", "Final answer: ", item["answer"].strip())
            for _ in range(args.repeat):
                records.append({"question": item["question"], "answer": answer})

    random.seed(args.seed)
    random.shuffle(records)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} examples to {args.output}")


if __name__ == "__main__":
    main()
