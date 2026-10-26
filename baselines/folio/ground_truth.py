#!/usr/bin/env python3
"""
Baseline (FOLIO): Ground Truth.

FOLIO does not provide reasoning traces, so the Ground-Truth SFT target is the
gold label only: "Therefore, the answer is: <label>".

Usage:
    python ground_truth.py --output ground_truth.jsonl
"""

import argparse
import json

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Build the FOLIO Ground-Truth SFT set.")
    parser.add_argument("--dataset", default="yale-nlp/FOLIO")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in ds:
            record = {
                "premise-fol": ex["premises-FOL"],
                "conclusion-fol": ex["conclusion-FOL"],
                "premises_nl": ex["premises"],
                "conclusion_nl": ex["conclusion"],
                "explanation": f"Therefore, the answer is: {ex['label']}",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(ds)} examples to {args.output}")


if __name__ == "__main__":
    main()
