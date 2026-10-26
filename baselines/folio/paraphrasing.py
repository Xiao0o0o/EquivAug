"""
Paraphrase FOLIO solutions using GPT-4o-mini.
For each premises+conclusion pair, first generate a reasoning explanation,
validate correctness, then generate 9 paraphrased explanations.
Save results to JSONL for SFT training.

Usage:
    export OPENAI_API_KEY="sk-..."
    python paraphrasing.py
    python paraphrasing.py --num-paraphrases 9 --output-file folio_paraphrased.jsonl
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


# ============================================================
#  Defaults
# ============================================================

DEFAULT_INPUT_FILE = "folio_sft_nl.jsonl"  # output of Step 3 (translate mode)
DEFAULT_OUTPUT_FILE = "./folio_paraphrased.jsonl"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_NUM_PARAPHRASES = 9


# ============================================================
#  Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paraphrase FOLIO explanations with GPT-4o-mini.")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-paraphrases", type=int, default=DEFAULT_NUM_PARAPHRASES)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per API call on failure.")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save intermediate results every N items.")
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Max concurrent API threads.")
    parser.add_argument("--base_url", default=None,
                        help="Base URL of an OpenAI-compatible API (default: OpenAI).")
    return parser.parse_args()


# ============================================================
#  Prompts
# ============================================================

SYSTEM_PROMPT_GENERATE = (
    "You are a logic expert. Given a set of premises and a conclusion, determine whether "
    "the conclusion is True, False, or Uncertain based on the premises. Provide a clear, "
    "step-by-step natural language explanation of your reasoning, then end with your final "
    "answer on the last line in the exact format: Therefore, the answer is: <True/False/Uncertain>."
)

def build_generate_prompt(premises_nl: str, conclusion_nl: str) -> str:
    return (
        f"Given the following premises:\n{premises_nl}\n\n"
        f"Determine whether the following conclusion is True, False, or Uncertain:\n"
        f"{conclusion_nl}\n\n"
        f"Provide a step-by-step explanation, then end with:\n"
        f"Therefore, the answer is: <True/False/Uncertain>."
    )


SYSTEM_PROMPT_PARAPHRASE = (
    "You are a logic tutor who rephrases explanations in different ways while keeping "
    "the logic and reasoning exactly correct. You must preserve the final answer exactly."
)

def build_paraphrase_prompt(premises_nl: str, conclusion_nl: str,
                            explanation: str, num: int) -> str:
    return (
        f"Here is a logical reasoning problem and its explanation:\n\n"
        f"**Premises:**\n{premises_nl}\n\n"
        f"**Conclusion:**\n{conclusion_nl}\n\n"
        f"**Explanation:**\n{explanation}\n\n"
        f"Please paraphrase the above explanation {num} different ways. "
        f"Each paraphrased explanation must:\n"
        f"1. Use different wording, sentence structure, or explanation style\n"
        f"2. Keep all logical reasoning steps correct\n"
        f"3. End with the exact same final answer in the format: Therefore, the answer is: <True/False/Uncertain>.\n\n"
        f"Return each paraphrased explanation separated by a line containing only '---'.\n"
        f"Do NOT include the premises or conclusion, only the paraphrased explanations."
    )


# ============================================================
#  Parsing / Validation
# ============================================================

def extract_answer(text: str) -> str | None:
    """Extract the final answer (True/False/Uncertain) from explanation text."""
    # Try pattern: "Therefore, the answer is: True/False/Uncertain"
    m = re.search(r"[Tt]herefore,?\s*the\s+answer\s+is:?\s*(True|False|Uncertain)", text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip().capitalize()
        # Normalize
        if ans in ("True", "False", "Uncertain"):
            return ans
    return None


def parse_paraphrases(response_text: str, expected_label: str) -> list[str]:
    """Split response into individual paraphrases and validate each has correct answer."""
    parts = re.split(r"\n---\n|\n-{3,}\n", response_text)
    parts = [p.strip() for p in parts if p.strip()]

    valid = []
    for p in parts:
        ans = extract_answer(p)
        if ans and ans == expected_label:
            valid.append(p)
    return valid


# ============================================================
#  Load data
# ============================================================

def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ============================================================
#  Main
# ============================================================

def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    # Load input data and deduplicate by (premises_nl, conclusion_nl)
    raw_dataset = load_jsonl(args.input_file)
    print(f"Loaded {len(raw_dataset)} samples from {args.input_file}")

    seen = set()
    dataset = []
    for item in raw_dataset:
        key = (item["premises_nl"], item["conclusion_nl"])
        if key not in seen:
            seen.add(key)
            dataset.append(item)
    print(f"After deduplication: {len(dataset)} unique questions")
    print(f"Model: {args.model}, Paraphrases per question: {args.num_paraphrases}")
    print(f"Output: {args.output_file}\n")

    results = []
    total_gen_correct = 0
    total_gen_failed = 0
    total_valid_paraphrases = 0

    def call_api(messages: list[dict], temperature: float = 0.7) -> str | None:
        """Call OpenAI API with retries."""
        for attempt in range(args.max_retries):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=4096,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < args.max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def process_one(idx: int, item: dict):
        """
        Step 1: Generate an explanation from GPT-4o and check correctness.
        Step 2: If correct, paraphrase it N times.
        """
        premises_nl = item["premises_nl"]
        conclusion_nl = item["conclusion_nl"]
        label = item["label"]  # "True", "False", or "Uncertain"

        # --- Step 1: Generate explanation ---
        gen_prompt = build_generate_prompt(premises_nl, conclusion_nl)
        gen_response = call_api([
            {"role": "system", "content": SYSTEM_PROMPT_GENERATE},
            {"role": "user", "content": gen_prompt},
        ], temperature=args.temperature)

        if gen_response is None:
            return idx, None, "api_fail"

        predicted = extract_answer(gen_response)
        if predicted != label:
            return idx, None, "wrong_answer"

        # --- Step 2: Paraphrase the correct explanation ---
        para_prompt = build_paraphrase_prompt(
            premises_nl, conclusion_nl, gen_response, args.num_paraphrases
        )
        para_response = call_api([
            {"role": "system", "content": SYSTEM_PROMPT_PARAPHRASE},
            {"role": "user", "content": para_prompt},
        ], temperature=args.temperature)

        if para_response is None:
            # At least return the original explanation
            samples = [{
                "premises_nl": premises_nl,
                "conclusion_nl": conclusion_nl,
                "label": label,
                "explanation": gen_response,
            }]
            return idx, samples, "paraphrase_api_fail"

        paraphrases = parse_paraphrases(para_response, label)

        # Build output samples: original + valid paraphrases
        samples = []
        # Include the original generated explanation
        samples.append({
            "premises_nl": premises_nl,
            "conclusion_nl": conclusion_nl,
            "label": label,
            "explanation": gen_response,
        })
        for p in paraphrases:
            samples.append({
                "premises_nl": premises_nl,
                "conclusion_nl": conclusion_nl,
                "label": label,
                "explanation": p,
            })
        return idx, samples, "ok"

    # Submit all tasks concurrently
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_one, idx, dataset[idx]): idx
            for idx in range(len(dataset))
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx, samples, status = future.result()

            if status == "api_fail":
                total_gen_failed += 1
                continue
            elif status == "wrong_answer":
                total_gen_failed += 1
                continue

            if samples:
                total_gen_correct += 1
                total_valid_paraphrases += len(samples) - 1  # exclude original
                results.extend(samples)

            # Intermediate save
            if (len(results) // args.save_every) > ((len(results) - len(samples)) // args.save_every):
                with open(args.output_file, "w", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final save
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    total_questions = len(dataset)
    print("\n" + "=" * 60)
    print("  FOLIO Paraphrasing Summary")
    print("=" * 60)
    print(f"  Total questions:              {total_questions}")
    print(f"  Correct generations:          {total_gen_correct}")
    print(f"  Failed (API/wrong answer):    {total_gen_failed}")
    print(f"  Total valid paraphrases:      {total_valid_paraphrases}")
    if total_gen_correct > 0:
        print(f"  Avg paraphrases per correct:  {total_valid_paraphrases / total_gen_correct:.2f}")
    print(f"  Total training samples:       {len(results)}")
    print(f"  Saved to: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()