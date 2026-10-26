"""
Evaluation (FOLIO): sample responses on the FOLIO validation set with vLLM.

The pickle written here is scored by compute_accuracy.py.

Usage:
    python generate_vllm.py --model-path /path/to/model \
        --num-responses 5 --temperature 0.6 --output-dir ./model_responses_vllm/my_model
"""

import argparse
import os
import pickle
import re
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams


DEFAULT_DATASET_PATH = "yale-nlp/FOLIO"
DEFAULT_DATASET_NAME = "default"
DEFAULT_DATASET_SPLIT = "validation"
DEFAULT_OUTPUT_DIR = "./model_responses_vllm"

# Same prompt template as SFT and GRPO
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Determine whether the conclusion follows from the premises. "
    "Think step by step. Answer True, False, or Uncertain.\n\n"
    "Premises: {premises}\n"
    "Conclusion: {conclusion}\n\n"
    "### Response:\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FOLIO responses with vLLM.")
    parser.add_argument("--model-path", required=True, help="Path or HF id of the fine-tuned model.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="HF dataset path (repo id).")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Subset name for the dataset.")
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT, help="Split to evaluate (e.g., validation/train).")
    parser.add_argument("--dataset-cache", default="./dataset_cache", help="Local cache dir for datasets.")
    parser.add_argument("--max-samples", type=int, default=10000, help="Upper bound on #examples to evaluate.")
    parser.add_argument("--num-responses", type=int, default=1, help="Number of samples per question.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling.")
    parser.add_argument("--batch-size", type=int, default=8, help="How many prompts to send to vLLM at once.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to store the pickle results.")
    parser.add_argument("--output-prefix", default="folio", help="Prefix for the pickle filename.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override vLLM max_model_len if set.")
    parser.add_argument("--gpu-mem-util", type=float, default=0.95, help="Fraction of GPU memory vLLM can use.")
    return parser.parse_args()


def build_prompt(premises: str, conclusion: str) -> str:
    return PROMPT_TEMPLATE.format(premises=premises, conclusion=conclusion)


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(
        args.dataset_path,
        args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache,
        download_mode="reuse_dataset_if_exists",
    )
    dataset = dataset.select(range(min(len(dataset), args.max_samples)))
    print(f"Loaded dataset '{args.dataset_path}' ({args.dataset_split}) with {len(dataset)} samples.")

    # Print first example to verify column names
    print(f"Example columns: {dataset.column_names}")
    print(f"First example: {dataset[0]}")

    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tensor_parallel = max(torch.cuda.device_count(), 1)
    print(f"Initializing vLLM with tensor_parallel_size={tensor_parallel}")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16" if torch.cuda.is_available() else "auto",
        tensor_parallel_size=tensor_parallel,
        max_model_len=args.max_model_len or min(getattr(tokenizer, "model_max_length", 32768), 32768),
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

    outfile = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_{args.dataset_split}_{args.num_responses}responses_{args.max_new_tokens}tokens.pkl",
    )
    results = {}

    progress = tqdm(range(0, len(dataset), args.batch_size), desc="Generating", unit="batch")
    for start in progress:
        end = min(start + args.batch_size, len(dataset))
        batch_indices = list(range(start, end))

        batch_prompts: List[str] = []
        for idx in batch_indices:
            sample = dataset[idx]
            # Use FOL columns: 'premises-FOL' and 'conclusion-FOL'
            #batch_prompts.append(build_prompt(sample["premises-FOL"], sample["conclusion-FOL"]))
            # Use NL columns: 'premises' and 'conclusion'
            batch_prompts.append(build_prompt(sample["premises"], sample["conclusion"]))

        outputs = llm.generate(batch_prompts, sampling_params)

        for ds_idx, output in zip(batch_indices, outputs):
            sample = dataset[ds_idx]
            generations = [o.text.strip() for o in output.outputs]
            results[ds_idx] = {
                "id": ds_idx,
                "premises": sample["premises"],
                "premises_fol": sample["premises-FOL"],
                "conclusion": sample["conclusion"],
                "conclusion_fol": sample["conclusion-FOL"],
                "generations": generations,
                "label": sample.get("label", ""),  # True / False / Uncertain
            }

    with open(outfile, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {len(results)} generations to {outfile}")


if __name__ == "__main__":
    main()