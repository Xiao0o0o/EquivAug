"""
Baseline (GSM8K): Rejection Sampling Fine-Tuning (RFT).

Sample many responses per train question from a model that was fine-tuned on
the ground-truth data, keep the correct ones, deduplicate, sample a balanced
subset of --target-samples examples, and save it as an SFT-ready dataset.

Usage:
    python rft.py \
        --model-path /path/to/sft_model_trained_on_ground_truth \
        --num-responses 100 --temperature 0.7 --target-samples 30000 \
        --filter-questions-file ../../data/gsm8k/equivaug.jsonl
"""

import argparse
import json
import os
import pickle
import re
import random
from collections import defaultdict
from typing import List, Optional, Any
from decimal import Decimal, InvalidOperation
from fractions import Fraction

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams


# ============================================================
#  Defaults
# ============================================================

DEFAULT_DATASET_PATH = "openai/gsm8k"
DEFAULT_DATASET_NAME = "main"
DEFAULT_DATASET_SPLIT = "train"  # RFT collects on train set
DEFAULT_OUTPUT_DIR = "./RFT_results"

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{question}\n\n### Response:"
)

ANSWER_RE = re.compile(r"####\s*([^\n]+)")


# ============================================================
#  Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFT: collect correct responses on GSM8K train set.")
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
    parser.add_argument("--seed", type=int, default=42)

    # Post-processing args
    parser.add_argument("--target-samples", type=int, default=30000,
                        help="Target number of balanced training samples for final output.")
    parser.add_argument("--filter-questions-file", type=str, default=None,
                        help="Optional JSONL file with allowed questions (only keep questions present in this file). "
                             "If not provided, all questions are kept.")
    return parser.parse_args()


# ============================================================
#  Answer extraction & comparison
# ============================================================

def extract_gt_answer(answer_text: str) -> str:
    match = ANSWER_RE.search(answer_text or "")
    return match.group(1).strip() if match else (answer_text or "").strip()


def extract_pred_answer(text: str) -> str:
    """Extract predicted answer from model generation (looks for #### pattern)."""
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    matches = list(ANSWER_RE.finditer(text))
    if matches:
        ans = matches[-1].group(1).strip()
    else:
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
        return nums[-1].replace(",", "") if nums else ""
    ans = ans.rstrip('."\' ').replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", ans)
    return nums[-1] if nums else ans


def _try_parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").rstrip(' ."\'')
    if not text:
        return None
    if re.fullmatch(r"-?\d+\s*/\s*-?\d+", text):
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


def is_equal(ans1, ans2, tol=1e-9) -> bool:
    n1, n2 = _try_parse_number(ans1), _try_parse_number(ans2)
    if n1 is not None and n2 is not None:
        return abs(n1 - n2) < tol
    return str(ans1).strip().replace(",", "") == str(ans2).strip().replace(",", "")


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


# ============================================================
#  Post-processing: balanced sampling + Final answer formatting
# ============================================================

def load_filter_questions(filepath: str) -> set:
    """Load allowed questions from a JSONL file."""
    questions = set()
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            questions.add(data['question'])
    return questions


def balanced_sample(question_groups: dict, target: int, rng: random.Random) -> list:
    """
    Sample `target` responses balanced across questions.
    Each question gets roughly target // num_questions responses.
    Remainder is distributed randomly.
    """
    num_questions = len(question_groups)
    if num_questions == 0:
        return []

    per_question = target // num_questions
    remainder = target - per_question * num_questions
    extra_questions = set(rng.sample(
        list(question_groups.keys()),
        min(remainder, num_questions)
    ))

    sampled = []
    for q, responses in question_groups.items():
        n = per_question + (1 if q in extra_questions else 0)
        n = min(n, len(responses))
        selected = rng.sample(responses, n)
        for resp in selected:
            sampled.append({
                'question': q,
                'expr_explanations': [resp]
            })

    rng.shuffle(sampled)
    return sampled


def add_final_answer_format(data: list) -> list:
    """
    Ensure each explanation ends with 'Final answer: <number>'.
    Extracts the answer from '#### <number>' if present.
    """
    for item in data:
        new_expls = []
        for expl in item['expr_explanations']:
            expl = expl.rstrip()
            match = re.search(r'####\s*(.+)', expl)
            if match:
                answer = match.group(1).strip()
                if not expl.endswith(f"Final answer: {answer}"):
                    expl += f"\nFinal answer: {answer}"
            new_expls.append(expl)
        item['expr_explanations'] = new_expls
    return data


# ============================================================
#  Main
# ============================================================

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)

    # Load train dataset
    dataset = load_dataset(
        args.dataset_path, args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache,
        download_mode="reuse_dataset_if_exists",
    )
    dataset = dataset.select(range(min(len(dataset), args.max_samples)))
    print(f"Loaded {len(dataset)} train samples.")

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
        batch_prompts = [build_prompt(dataset[i]["question"]) for i in batch_indices]

        outputs = llm.generate(batch_prompts, sampling_params)

        for ds_idx, output in zip(batch_indices, outputs):
            sample = dataset[ds_idx]
            generations = [o.text.strip() for o in output.outputs]
            all_results[ds_idx] = {
                "id": ds_idx,
                "question": sample["question"],
                "generations": generations,
                "answer": extract_gt_answer(sample.get("answer", "")),
            }

    model_short = os.path.basename(args.model_path.rstrip("/"))

    # Save raw results
    raw_file = os.path.join(
        args.output_dir,
        f"{model_short}_gsm8k_train_raw_{args.num_responses}responses.pkl",
    )
    with open(raw_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Saved raw results to {raw_file}")

    # ---- Step 2: Filter correct responses & deduplicate ----
    print("\nFiltering correct responses...")

    # Optionally filter to a subset of questions
    filter_questions = None
    if args.filter_questions_file:
        filter_questions = load_filter_questions(args.filter_questions_file)
        print(f"Loaded {len(filter_questions)} allowed questions from {args.filter_questions_file}")

    question_groups = defaultdict(list)  # question -> list of correct responses
    total_generated = 0
    total_correct = 0
    total_distinct = 0
    questions_with_correct = 0

    for item in all_results.values():
        question = item["question"]

        # Skip if not in allowed set
        if filter_questions is not None and question not in filter_questions:
            continue

        gt = item["answer"]
        correct_responses = []
        seen = set()

        for gen in item["generations"]:
            total_generated += 1
            pred = extract_pred_answer(gen)
            if is_equal(pred, gt):
                total_correct += 1
                if gen not in seen:
                    seen.add(gen)
                    correct_responses.append(gen)

        total_distinct += len(correct_responses)
        if correct_responses:
            questions_with_correct += 1
            question_groups[question].extend(correct_responses)

    # Also save unprocessed RFT data (for compatibility)
    rft_data_raw = []
    for q, resps in question_groups.items():
        for resp in resps:
            rft_data_raw.append({"question": q, "response": resp})

    jsonl_raw_file = os.path.join(
        args.output_dir,
        f"{model_short}_gsm8k_rft_{args.num_responses}responses.jsonl",
    )
    with open(jsonl_raw_file, "w") as f:
        for sample in rft_data_raw:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # ---- Step 3: Balanced sampling ----
    print(f"\nBalanced sampling to {args.target_samples} samples...")
    sampled_data = balanced_sample(question_groups, args.target_samples, rng)
    print(f"  Sampled {len(sampled_data)} entries from {len(question_groups)} questions")

    # ---- Step 4: Add Final answer format ----
    print("Adding 'Final answer: <number>' format...")
    sampled_data = add_final_answer_format(sampled_data)

    # ---- Step 5: Save final dataset ----
    final_file = os.path.join(
        args.output_dir,
        f"{model_short}_gsm8k_rft_{args.target_samples}_balanced_final.jsonl",
    )
    with open(final_file, "w") as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- Stats ----
    print("\n" + "=" * 60)
    print("  RFT Collection Summary")
    print("=" * 60)
    print(f"  Total questions:              {len(all_results)}")
    if filter_questions is not None:
        print(f"  After question filtering:     {len(question_groups)}")
    print(f"  Questions with ≥1 correct:    {questions_with_correct} "
          f"({questions_with_correct/max(len(question_groups),1)*100:.1f}%)")
    print(f"  Total generated:              {total_generated}")
    print(f"  Total correct:                {total_correct} "
          f"({total_correct/max(total_generated,1)*100:.1f}%)")
    print(f"  Total distinct correct:       {total_distinct}")
    print(f"  Avg distinct per question:    {total_distinct/max(len(question_groups),1):.1f}")
    print(f"  Final balanced samples:       {len(sampled_data)}")
    print(f"\n  Raw JSONL:   {jsonl_raw_file}")
    print(f"  Raw PKL:     {raw_file}")
    print(f"  Final JSONL: {final_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()