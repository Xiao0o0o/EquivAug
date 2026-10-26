"""
Baseline (FOLIO): Rejection Sampling Fine-Tuning (RFT).

  Step 1: Generate multiple responses per train question (vLLM)
  Step 2: Filter correct ones, deduplicate
  Step 3: Filter by allowed questions, balanced sample
  Step 4: Add "Therefore, the answer is: <label>." format
  Step 5: Save as SFT-ready dataset matching folio_sft_nl_60.jsonl schema

Usage:
    python RFT_new.py \
        --model-path /path/to/sft_model_trained_on_ground_truth \
        --filter-file ../../data/folio/equivaug.jsonl \
        --num-responses 100 \
        --target-samples 7000
"""

import argparse
import json
import os
import pickle
import re
import random
from typing import List, Optional, Any
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams


# ============================================================
#  Defaults
# ============================================================

DEFAULT_DATASET_PATH = "yale-nlp/FOLIO"
DEFAULT_DATASET_NAME = "default"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_OUTPUT_DIR = "./RFT_results_folio"

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Determine whether the conclusion follows from the premises. "
    "Think step by step and explain your reasoning. "
    "End your response with \"The final answer is True/False/Uncertain.\"\n\n"
    "Premises: {premises}\n"
    "Conclusion: {conclusion}\n\n"
    "### Response:\n"
)

LABEL_MAP = {0: "True", 1: "False", 2: "Uncertain"}
VALID_LABELS = {"true", "false", "uncertain"}


# ============================================================
#  Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFT: collect + post-process on FOLIO train set.")

    # Model & generation args
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--dataset-cache", default="./dataset_cache")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--num-responses", type=int, default=100,
                        help="Number of candidate responses per question (k in RFT paper).")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-mem-util", type=float, default=0.95)
    parser.add_argument("--min-response-length", type=int, default=50,
                        help="Minimum character length for a valid CoT response.")

    # Post-processing args
    parser.add_argument("--filter-file", type=str, default=None,
                        help="JSONL file with allowed questions (premises_nl, conclusion_nl). "
                             "If not provided, all questions are kept.")
    parser.add_argument("--target-samples", type=int, default=7000,
                        help="Target number of balanced training samples for final output.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    return parser.parse_args()


# ============================================================
#  Answer extraction & comparison
# ============================================================

def normalize_label(label) -> str:
    if isinstance(label, int):
        return LABEL_MAP.get(label, str(label))
    s = str(label).strip().lower()
    if s in ("true", "yes", "correct", "valid"):
        return "True"
    elif s in ("false", "no", "incorrect", "invalid"):
        return "False"
    elif s in ("uncertain", "unknown", "undetermined"):
        return "Uncertain"
    return str(label).strip()


def extract_pred_label(text: str) -> str:
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    match = re.search(
        r"[Tt]he\s+final\s+answer\s+is\s+(True|False|Uncertain|Unknown)",
        text, re.IGNORECASE
    )
    if match:
        return normalize_label(match.group(1))

    match = re.search(
        r"[Tt]he\s+answer\s+is\s+(True|False|Uncertain|Unknown)",
        text, re.IGNORECASE
    )
    if match:
        return normalize_label(match.group(1))

    match = re.search(
        r"[Aa]nswer\s*:\s*(True|False|Uncertain|Unknown)",
        text, re.IGNORECASE
    )
    if match:
        return normalize_label(match.group(1))

    matches = re.findall(r"\b(True|False|Uncertain|Unknown)\b", text, re.IGNORECASE)
    if matches:
        return normalize_label(matches[-1])

    return ""


def is_valid_cot_response(text: str, min_length: int = 50) -> bool:
    if len(text.strip()) < min_length:
        return False
    if not re.search(
        r"[Tt]he\s+(final\s+)?answer\s+is\s+(True|False|Uncertain|Unknown)",
        text, re.IGNORECASE
    ):
        return False
    return True


def build_prompt(premises: str, conclusion: str) -> str:
    return PROMPT_TEMPLATE.format(premises=premises, conclusion=conclusion)


# ============================================================
#  Post-processing helpers
# ============================================================

def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())


def load_filter_file(filepath: str) -> dict:
    """Load allowed questions. Returns dict: (norm_premises, norm_conclusion) -> full record."""
    allowed = {}
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            premises = data.get('premises_nl') or data.get('premises', '')
            conclusion = data.get('conclusion_nl') or data.get('conclusion', '')
            key = (normalize_text(premises), normalize_text(conclusion))
            if key not in allowed:
                allowed[key] = data
    print(f"Loaded {len(allowed)} allowed question pairs from {filepath}")
    return allowed


def balanced_sample(question_groups: dict, target: int, rng: random.Random) -> list:
    """Sample `target` responses balanced across questions.
    Returns list of (key, rft_item) tuples."""
    num_questions = len(question_groups)
    if num_questions == 0:
        print("WARNING: No questions after filtering!")
        return []

    per_question = target // num_questions
    remainder = target - per_question * num_questions
    extra_keys = set(rng.sample(
        list(question_groups.keys()),
        min(remainder, num_questions)
    ))

    sampled = []
    for key, items in question_groups.items():
        n = per_question + (1 if key in extra_keys else 0)
        n = min(n, len(items))
        selected = rng.sample(items, n)
        for item in selected:
            sampled.append((key, item))

    rng.shuffle(sampled)
    return sampled


def add_final_answer_format(response: str, label: str) -> str:
    """Ensure response ends with 'Therefore, the answer is: <label>.'"""
    response = response.rstrip()
    expected_ending = f"Therefore, the answer is: {label}."

    if response.endswith(expected_ending):
        return response

    final_pattern = re.compile(
        r'[Tt]he\s+(final\s+)?answer\s+is\s*:?\s*(True|False|Uncertain)\.?',
        re.IGNORECASE
    )
    lines = response.split('\n')
    last_line = lines[-1].strip()
    if final_pattern.search(last_line):
        lines[-1] = expected_ending
        return '\n'.join(lines)
    else:
        return response + '\n' + expected_ending


def build_output_record(rft_item: dict, filter_record: dict) -> dict:
    """Build output record matching folio_sft_nl_60.jsonl schema."""
    label = rft_item['label']
    explanation = add_final_answer_format(rft_item['response'], label)

    out = {}
    for field in ['example_id', 'story_id', 'premises_fol', 'conclusion_fol',
                  'premises_p9', 'conclusion_p9', 'solution', 'solution_source',
                  'proof_index', 'total_proofs', 'strategy', 'transform',
                  'transform_desc']:
        if field in filter_record:
            out[field] = filter_record[field]

    out['premises_nl'] = rft_item['premises']
    out['conclusion_nl'] = rft_item['conclusion']
    out['label'] = label
    out['explanation'] = explanation

    return out


def build_output_record_no_filter(rft_item: dict) -> dict:
    """Build output record when no filter file is provided (no extra fields)."""
    label = rft_item['label']
    explanation = add_final_answer_format(rft_item['response'], label)

    return {
        'premises_nl': rft_item['premises'],
        'conclusion_nl': rft_item['conclusion'],
        'label': label,
        'explanation': explanation,
    }


# ============================================================
#  Main
# ============================================================

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)

    # ================================================================
    #  Part 1: Generate & filter (original rft_collect_folio.py)
    # ================================================================

    # Load train dataset
    dataset = load_dataset(
        args.dataset_path, args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache,
        download_mode="reuse_dataset_if_exists",
    )
    dataset = dataset.select(range(min(len(dataset), args.max_samples)))
    print(f"Loaded {len(dataset)} train samples.")
    print(f"Columns: {dataset.column_names}")
    print(f"First example: {dataset[0]}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Init vLLM
    tensor_parallel = max(torch.cuda.device_count(), 1)
    print(f"Initializing vLLM with tensor_parallel_size={tensor_parallel}")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16" if torch.cuda.is_available() else "auto",
        tensor_parallel_size=tensor_parallel,
        max_model_len=args.max_model_len or min(
            getattr(tokenizer, "model_max_length", 32768), 32768
        ),
        gpu_memory_utilization=args.gpu_mem_util,
        disable_log_stats=True,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_responses,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )

    # ---- Step 1: Generate all responses ----
    print(f"\nGenerating {args.num_responses} responses per question...")
    all_results = {}

    progress = tqdm(range(0, len(dataset), args.batch_size), desc="Generating", unit="batch")
    for start in progress:
        end = min(start + args.batch_size, len(dataset))
        batch_indices = list(range(start, end))
        batch_prompts = [
            build_prompt(dataset[i]["premises"], dataset[i]["conclusion"])
            for i in batch_indices
        ]

        outputs = llm.generate(batch_prompts, sampling_params)

        for ds_idx, output in zip(batch_indices, outputs):
            sample = dataset[ds_idx]
            generations = [o.text.strip() for o in output.outputs]
            gt_label = normalize_label(sample.get("label", ""))
            all_results[ds_idx] = {
                "id": ds_idx,
                "premises": sample["premises"],
                "premises_fol": sample["premises-FOL"],
                "conclusion": sample["conclusion"],
                "conclusion_fol": sample["conclusion-FOL"],
                "generations": generations,
                "label": gt_label,
            }

    model_short = os.path.basename(args.model_path.rstrip("/"))

    # Save raw results
    raw_file = os.path.join(
        args.output_dir,
        f"{model_short}_folio_train_raw_{args.num_responses}responses.pkl",
    )
    with open(raw_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Saved raw results to {raw_file}")

    # ---- Step 2: Filter correct CoT responses & deduplicate ----
    print("\nFiltering correct responses with valid CoT...")

    rft_data = []
    total_generated = 0
    total_correct = 0
    total_valid_cot = 0
    total_distinct = 0
    questions_with_correct = 0

    for item in all_results.values():
        gt_label = item["label"]
        correct_responses = []
        seen = set()

        for gen in item["generations"]:
            total_generated += 1
            pred_label = extract_pred_label(gen)

            if pred_label == gt_label:
                total_correct += 1

                if not is_valid_cot_response(gen, min_length=args.min_response_length):
                    continue
                total_valid_cot += 1

                if gen not in seen:
                    seen.add(gen)
                    correct_responses.append(gen)

        total_distinct += len(correct_responses)
        if correct_responses:
            questions_with_correct += 1

        for resp in correct_responses:
            rft_data.append({
                "premises": item["premises"],
                "conclusion": item["conclusion"],
                "response": resp,
                "label": gt_label,
            })

    # Save intermediate RFT data (unprocessed)
    jsonl_file = os.path.join(
        args.output_dir,
        f"{model_short}_folio_rft_{args.num_responses}responses.jsonl",
    )
    with open(jsonl_file, "w") as f:
        for sample in rft_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    pkl_file = os.path.join(
        args.output_dir,
        f"{model_short}_folio_rft_{args.num_responses}responses.pkl",
    )
    with open(pkl_file, "wb") as f:
        pickle.dump(rft_data, f)

    # ---- RFT collection stats ----
    print("\n" + "=" * 60)
    print("  RFT Collection Summary (FOLIO)")
    print("=" * 60)
    print(f"  Total questions:              {len(all_results)}")
    print(f"  Questions with >=1 correct:   {questions_with_correct} "
          f"({questions_with_correct/max(len(all_results),1)*100:.1f}%)")
    print(f"  Total generated:              {total_generated}")
    print(f"  Total correct label:          {total_correct} "
          f"({total_correct/max(total_generated,1)*100:.1f}%)")
    print(f"  Total valid CoT + correct:    {total_valid_cot} "
          f"({total_valid_cot/max(total_generated,1)*100:.1f}%)")
    print(f"  Total distinct correct:       {total_distinct}")
    print(f"  Avg distinct per question:    {total_distinct/max(len(all_results),1):.1f}")
    print(f"  Final RFT training samples:   {len(rft_data)}")
    print(f"\n  Saved JSONL: {jsonl_file}")
    print(f"  Saved PKL:   {pkl_file}")
    print("=" * 60)

    # ================================================================
    #  Part 2: Post-processing (filter, balanced sample, format)
    # ================================================================

    print("\n\n" + "=" * 60)
    print("  Starting Post-processing...")
    print("=" * 60)

    # Load filter file if provided
    allowed = None
    if args.filter_file:
        allowed = load_filter_file(args.filter_file)

    # Group by question, optionally filter
    question_groups = defaultdict(list)  # key -> list of rft_items
    skipped = 0
    for item in rft_data:
        key = (normalize_text(item['premises']), normalize_text(item['conclusion']))
        if allowed is not None and key not in allowed:
            skipped += 1
            continue
        question_groups[key].append(item)

    if allowed is not None:
        print(f"After question filtering: {sum(len(v) for v in question_groups.values())} responses "
              f"from {len(question_groups)} unique questions (skipped {skipped})")
    else:
        print(f"No filter file provided, keeping all {len(question_groups)} unique questions")

    # Balanced sampling
    print(f"\nBalanced sampling to {args.target_samples} samples...")
    sampled = balanced_sample(question_groups, args.target_samples, rng)
    print(f"  Sampled {len(sampled)} entries from {len(question_groups)} questions")

    # Build output records
    print("Building output records with 'Therefore, the answer is: <label>.' format...")
    output_data = []
    for key, rft_item in sampled:
        if allowed is not None and key in allowed:
            record = build_output_record(rft_item, allowed[key])
        else:
            record = build_output_record_no_filter(rft_item)
        output_data.append(record)

    # Save final dataset
    final_file = os.path.join(
        args.output_dir,
        f"{model_short}_folio_rft_{args.target_samples}_balanced_final.jsonl",
    )
    with open(final_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # ---- Post-processing stats ----
    print("\n" + "=" * 60)
    print("  Post-processing Summary")
    print("=" * 60)
    if allowed is not None:
        print(f"  Allowed questions:            {len(allowed)}")
    print(f"  Questions after filtering:    {len(question_groups)}")
    print(f"  Target samples:               {args.target_samples}")
    print(f"  Actual sampled:               {len(output_data)}")
    avg = len(output_data) / max(len(question_groups), 1)
    print(f"  Avg samples per question:     {avg:.1f}")

    label_counts = defaultdict(int)
    for item in output_data:
        label_counts[item['label']] += 1
    print(f"  Label distribution:           {dict(label_counts)}")
    print(f"\n  Intermediate JSONL: {jsonl_file}")
    print(f"  Intermediate PKL:   {pkl_file}")
    print(f"  Raw PKL:            {raw_file}")
    print(f"  Final JSONL:        {final_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()