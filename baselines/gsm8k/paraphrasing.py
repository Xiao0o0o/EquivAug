"""
Paraphrase GSM8K solutions using GPT-4o-mini.
For each question+answer pair, generate 5 paraphrased solutions.
Save results to JSONL for SFT training.

Usage:
    export OPENAI_API_KEY="sk-..."
    python paraphrasing.py
    python paraphrasing.py --num-paraphrases 5 --output-file gsm8k_paraphrased.jsonl
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI


# ============================================================
#  Defaults
# ============================================================

DEFAULT_OUTPUT_FILE = "./gsm8k_paraphrased.jsonl"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_NUM_PARAPHRASES = 5
DEFAULT_DATASET_PATH = "openai/gsm8k"
DEFAULT_DATASET_NAME = "main"
DEFAULT_DATASET_SPLIT = "train"


# ============================================================
#  Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paraphrase GSM8K solutions with GPT-4o-mini.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-paraphrases", type=int, default=DEFAULT_NUM_PARAPHRASES)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--dataset-cache", default="./dataset_cache")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per API call on failure.")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save intermediate results every N questions.")
    parser.add_argument("--max-workers", type=int, default=100,
                        help="Max concurrent API threads.")
    parser.add_argument("--base_url", default=None,
                        help="Base URL of an OpenAI-compatible API (default: OpenAI).")
    return parser.parse_args()


# ============================================================
#  Prompt
# ============================================================

SYSTEM_PROMPT = (
    "You are a math tutor who rephrases solutions in different ways while keeping "
    "the math and logic exactly correct. You must preserve the final answer exactly."
)

def build_paraphrase_prompt(question: str, answer: str, num: int) -> str:
    return (
        f"Here is a math word problem and its solution:\n\n"
        f"**Question:**\n{question}\n\n"
        f"**Solution:**\n{answer}\n\n"
        f"Please paraphrase the above solution {num} different ways. "
        f"Each paraphrased solution must:\n"
        f"1. Use different wording, sentence structure, or explanation style\n"
        f"2. Keep all calculations and intermediate steps mathematically correct\n"
        f"3. End with the exact same final answer in the format: #### <number>\n\n"
        f"Return each paraphrased solution separated by a line containing only '---'.\n"
        f"Do NOT include the question, only the paraphrased solutions."
    )


# ============================================================
#  Parsing
# ============================================================

def extract_gt_answer(answer_text: str) -> str:
    m = re.search(r"####\s*([^\n]+)", answer_text)
    return m.group(1).strip() if m else answer_text.strip()


def parse_paraphrases(response_text: str, expected_answer: str, num: int) -> list:
    """Split response into individual paraphrases and validate each has correct ####."""
    # Split by --- separator
    parts = re.split(r"\n---\n|\n-{3,}\n", response_text)
    parts = [p.strip() for p in parts if p.strip()]

    valid = []
    for p in parts:
        # Check it contains the correct final answer
        m = re.search(r"####\s*([^\n]+)", p)
        if m:
            pred = m.group(1).strip().replace(",", "")
            gt = expected_answer.replace(",", "")
            if pred == gt:
                valid.append(p)

    return valid


# ============================================================
#  Main
# ============================================================

def main() -> None:
    args = parse_args()

    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    # Load dataset
    dataset = load_dataset(
        args.dataset_path, args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache,
        download_mode="reuse_dataset_if_exists",
    )
    dataset = dataset.select(range(min(len(dataset), args.max_samples)))
    print(f"Loaded {len(dataset)} samples from {args.dataset_path} ({args.dataset_split})")
    print(f"Model: {args.model}, Paraphrases per question: {args.num_paraphrases}")
    print(f"Output: {args.output_file}\n")

    results = []
    total_valid = 0
    total_failed = 0

    def process_one(idx, item):
        """Process a single question: call API and return parsed paraphrases."""
        question = item["question"]
        answer = item["answer"]
        gt_num = extract_gt_answer(answer)
        prompt = build_paraphrase_prompt(question, answer, args.num_paraphrases)

        response_text = None
        for attempt in range(args.max_retries):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=args.temperature,
                    max_tokens=4096,
                )
                response_text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                time.sleep(2 ** attempt)

        if response_text is None:
            return idx, None

        paraphrases = parse_paraphrases(response_text, gt_num, args.num_paraphrases)
        samples = []
        for p in paraphrases:
            samples.append({
                "question": question,
                "answer": p,
                "original_answer": answer,
                "gt_number": gt_num,
            })
        return idx, samples

    # Submit all tasks concurrently
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_one, idx, dataset[idx]): idx
            for idx in range(len(dataset))
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Paraphrasing"):
            idx, samples = future.result()
            if samples is None:
                total_failed += 1
                continue
            total_valid += len(samples)
            results.extend(samples)

            # Intermediate save
            if (len(results) // args.save_every) > ((len(results) - len(samples)) // args.save_every):
                with open(args.output_file, "w") as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Sort by original question order (optional, for reproducibility)
    # results are already collected, just save

    # Final save
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    total_questions = len(dataset)
    print("\n" + "=" * 60)
    print("  Paraphrasing Summary")
    print("=" * 60)
    print(f"  Total questions:          {total_questions}")
    print(f"  Failed API calls:         {total_failed}")
    print(f"  Total valid paraphrases:  {total_valid}")
    print(f"  Avg valid per question:   {total_valid / (total_questions - total_failed):.2f}")
    print(f"  Total training samples:   {len(results)}")
    print(f"  Saved to: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()