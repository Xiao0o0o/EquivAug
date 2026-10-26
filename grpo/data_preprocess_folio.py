"""
Convert FOLIO dataset to verl-compatible parquet format.

Usage:
    python data_preprocess_folio.py --local_dir ./data/folio

This script loads FOLIO from HuggingFace (or local jsonl files),
constructs the prompt in chat_template format, and saves as parquet
with the columns verl expects: data_source, prompt, ability,
reward_model (with ground_truth), extra_info.
"""

import argparse
import json
import os

import datasets
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./data/folio",
        help="Local directory to save parquet files.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="folio",
        help="Name tag used by reward function to dispatch.",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="yale-nlp/FOLIO",
        help="HuggingFace dataset name. Set to empty string to use local jsonl.",
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default="",
        help="Path to local folio-train.jsonl (only used if --hf_dataset is empty).",
    )
    parser.add_argument(
        "--val_jsonl",
        type=str,
        default="",
        help="Path to local folio-validation.jsonl (only used if --hf_dataset is empty).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

INSTRUCTION = (
    "Given the following premises, determine whether the conclusion is "
    "True, False, or Uncertain. Think step by step and output the final "
    'answer as "Therefore, the answer is: True.", '
    '"Therefore, the answer is: False.", or '
    '"Therefore, the answer is: Uncertain."'
)


def build_prompt(premises: str, conclusion: str) -> str:
    """Build the user-facing prompt string."""
    return (
        f"{INSTRUCTION}\n\n"
        f"Premises:\n{premises}\n\n"
        f"Conclusion:\n{conclusion}"
    )


def normalize_label(label: str) -> str:
    """Normalize FOLIO labels to: True / False / Uncertain."""
    label_lower = label.strip().lower()
    if "true" in label_lower:
        return "True"
    elif "false" in label_lower:
        return "False"
    elif "uncertain" in label_lower or "unknown" in label_lower:
        return "Uncertain"
    return label.strip()


# ---------------------------------------------------------------------------
# Map function
# ---------------------------------------------------------------------------

def make_map_fn(data_source: str, split: str):
    """Return a datasets.map-compatible function."""

    def process_fn(example, idx):
        # FOLIO fields: premises (str or list), conclusion (str), label (str)
        # Handle both list-of-strings and single-string premises
        premises = example.get("premises") or example.get("premises-NL", "")
        if isinstance(premises, list):
            premises = "\n".join(premises)

        conclusion = example.get("conclusion") or example.get("conclusion-NL", "")
        label = normalize_label(example.get("label", ""))

        question = build_prompt(premises, conclusion)

        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "logical_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": label,          # "True" / "False" / "Uncertain"
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "story_id": example.get("story-id", ""),
                "example_id": example.get("example-id", ""),
            },
        }
        return data

    return process_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.local_dir, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    if args.hf_dataset:
        print(f"Loading from HuggingFace: {args.hf_dataset}")
        ds = datasets.load_dataset(args.hf_dataset, trust_remote_code=True)
        train_dataset = ds.get("train") or ds.get("training")
        val_dataset = ds.get("validation") or ds.get("test")
    else:
        print("Loading from local jsonl files")
        train_dataset = datasets.Dataset.from_json(args.train_jsonl) if args.train_jsonl else None
        val_dataset = datasets.Dataset.from_json(args.val_jsonl) if args.val_jsonl else None

    # ---- Process ----------------------------------------------------------
    if train_dataset is not None:
        print(f"Train examples: {len(train_dataset)}")
        train_dataset = train_dataset.map(
            function=make_map_fn(args.data_source, "train"),
            with_indices=True,
        )
        out_path = os.path.join(args.local_dir, "train.parquet")
        train_dataset.to_parquet(out_path)
        print(f"Saved train → {out_path}")
    else:
        print("WARNING: No train split found.")

    if val_dataset is not None:
        print(f"Validation examples: {len(val_dataset)}")
        val_dataset = val_dataset.map(
            function=make_map_fn(args.data_source, "validation"),
            with_indices=True,
        )
        out_path = os.path.join(args.local_dir, "test.parquet")
        val_dataset.to_parquet(out_path)
        print(f"Saved validation → {out_path}")
    else:
        print("WARNING: No validation split found.")

    print("Done!")


if __name__ == "__main__":
    main()