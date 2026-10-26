#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 (GSM8K): Natural Language Reconstruction.

For every transformed expression from Step 2, prompt an LLM to rewrite the
original solution so that its computation order and grouping follow the NEW
expression. Each generated solution is kept only if its "####" answer equals the
gold answer; a "Final answer: <x>" line is appended for SFT.

Each output line is one SFT example:
    {"question": ..., "expr": ..., "depth": d, "expr_explanations": [solution]}

Usage:
    export OPENAI_API_KEY="sk-..."
    python translate_back_gsm8k.py \
        --input train_expr_variants.jsonl \
        --output_dir ./output --per_question_max_expr 5
"""

import argparse
import os
import json
import re
import random
from typing import Any, Dict, List, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= Configuration =================

MAX_DEPTH = 5  # depth levels that may appear in the Step 2 output

def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate transformed GSM8K expressions back into natural-language solutions."
    )
    parser.add_argument("--input", required=True,
                        help="JSONL from Step 2 (ast_transformation.py)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for the output JSONL files")
    parser.add_argument("--output_prefix", default="train_expr_explained",
                        help="Outputs: {prefix}_depth_{d}.jsonl for each depth and {prefix}_all.jsonl")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base_url", default=None,
                        help="Base URL of an OpenAI-compatible API (default: OpenAI)")
    parser.add_argument("--per_question_max_expr", type=int, default=5,
                        help="Max expressions translated per depth per question")
    parser.add_argument("--max_workers", type=int, default=100)
    parser.add_argument("--test_limit", type=int, default=None,
                        help="Only process the first N valid questions (for quick tests)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# ================= Answer extraction =================

def extract_final_answer(text: str) -> Optional[float]:
    """Extract numerical answer after #### marker."""
    if '####' in text:
        answer_part = text.split('####')[-1].strip()
        answer_part = answer_part.replace(',', '')
        match = re.search(r'-?\d+\.?\d*', answer_part)
        if match:
            return float(match.group())
    return None

# ================= Prompt =================

def build_single_user_msg(
    question: str,
    expr_original: str,
    answer_text: str,
    expr_variant: str
) -> str:
    expr_variant_clean = expr_variant.strip()
    msg = (
        "Your task is to write a natural language solution for a math word problem based on a given algebraic expression.\n"
        "The solution should:\n"
        "1)Imitate the style, tone, and level of detail of the original natural language solution, but do not use any special formatting such as << >> or dollar signs.\n"
        "2) The final line starting with '####' and the numeric answer.\n"
        "3) Do NOT simply copy the original solution. Adjust the order and grouping of computations, so that they match the NEW expression.\n"
        "4) Only output the new natural language solution, with no extra comments or explanations.\n\n"
        "Here is the math problem, its ORIGINAL algebraic solution expression, "
        "and its original natural language solution:\n\n"
        f"[Problem]\n{question}\n\n"
        f"[Original Expression]\n{expr_original}\n\n"
        f"[Original Natural Language Solution]\n{answer_text}\n\n"
        "Here is the NEW algebraic solution expression that is mathematically equivalent and still solves the SAME word problem:\n\n"
        f"[New Expression]\n{expr_variant_clean}\n\n"
        "Write your new natural language solution below:"
    )
    return msg

# ================= OpenAI call =================

def explain_single_expression(
    question: str,
    expr_original: str,
    answer_text: str,
    expr_variant: str,
    client: OpenAI,
    model: str
) -> str:
    user_msg = build_single_user_msg(question, expr_original, answer_text, expr_variant)
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"

# ================= Group variants by depth =================

def group_variants_by_depth(obj: Dict) -> Dict[int, List[str]]:
    """
    Group expr variants by their depth level.
    Uses expr_variants_with_depth: [{"expr": "...", "depth": 1, "value": ...}, ...]
    Fallback: parse from expr_variants_final strings like "expr=value (depth=N)"
    """
    by_depth: Dict[int, List[str]] = {d: [] for d in range(1, MAX_DEPTH + 1)}

    variants_with_depth = obj.get("expr_variants_with_depth", [])

    if variants_with_depth:
        for vd in variants_with_depth:
            depth = vd.get("depth", 0)
            expr = vd.get("expr", "")
            if depth in by_depth and expr:
                by_depth[depth].append(expr)
    else:
        for entry in obj.get("expr_variants_final", []):
            m = re.search(r'\(depth=(\d+)\)', entry)
            if m:
                depth = int(m.group(1))
                expr = entry.split("=", 1)[0].strip()
                if depth in by_depth and expr:
                    by_depth[depth].append(expr)

    return by_depth

def dedup_and_limit(exprs: List[str], k: int) -> List[str]:
    seen = set()
    out = []
    for e in exprs:
        if e not in seen:
            seen.add(e)
            out.append(e)
            if len(out) >= k:
                break
    return out

# ================= Worker: one question x one expr =================

def process_one_expr(
    rec_idx: int,
    depth: int,
    question: str,
    expr_original: str,
    answer_text: str,
    expr_variant: str,
    correct_answer: float,
    client: OpenAI,
    model: str,
) -> Tuple[int, int, Optional[Dict]]:
    """
    Generate explanation for one (question, expr_variant) pair.
    Validates the explanation answer against ground truth.
    Returns (rec_idx, depth, output_dict or None).
    """
    explanation = explain_single_expression(
        question=question,
        expr_original=expr_original,
        answer_text=answer_text,
        expr_variant=expr_variant,
        client=client,
        model=model,
    )

    if explanation.startswith("[ERROR]"):
        return rec_idx, depth, None

    # Validate: check if explanation's #### answer matches ground truth
    expl_answer = extract_final_answer(explanation)
    if expl_answer is None:
        return rec_idx, depth, None

    if abs(expl_answer - correct_answer) > 1e-6:
        return rec_idx, depth, None

    # Format final answer string
    if expl_answer == int(expl_answer):
        final_answer_str = str(int(expl_answer))
    else:
        final_answer_str = str(expl_answer)

    modified_explanation = explanation.strip() + f"\n Final answer: {final_answer_str}"

    # Output format: one question + one explanation per line
    out = {
        "question": question,
        "expr": expr_variant,
        "depth": depth,
        "expr_explanations": [modified_explanation],
    }

    return rec_idx, depth, out

# ================= Main =================

def main():
    args = parse_args()
    if args.base_url is None and not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Please set OPENAI_API_KEY in your environment.")
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    INPUT_PATH = args.input
    OUTPUT_DIR = args.output_dir
    OUTPUT_PATHS = {
        d: os.path.join(OUTPUT_DIR, f"{args.output_prefix}_depth_{d}.jsonl")
        for d in range(1, MAX_DEPTH + 1)
    }
    OUTPUT_TOTAL_PATH = os.path.join(OUTPUT_DIR, f"{args.output_prefix}_all.jsonl")

    # ---- Step 1: Read and filter input ----
    print(f"Reading input: {INPUT_PATH}")
    all_tasks = []
    total_lines = 0
    skipped_no_match = 0
    skipped_no_answer = 0
    skipped_no_variants = 0
    valid_records = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Filter: only keep answer_match = True
            if not obj.get("answer_match", False):
                skipped_no_match += 1
                continue

            question = obj.get("question", "")
            expr_original = obj.get("expr_original", "")
            answer_text = obj.get("answer", "")

            # Extract ground truth answer
            correct_answer = extract_final_answer(answer_text)
            if correct_answer is None:
                skipped_no_answer += 1
                continue

            # Group variants by depth
            by_depth = group_variants_by_depth(obj)

            has_any = False
            for d in range(1, MAX_DEPTH + 1):
                variants = dedup_and_limit(by_depth[d], args.per_question_max_expr)
                for expr_variant in variants:
                    all_tasks.append((valid_records, d, question, expr_original,
                                     answer_text, expr_variant, correct_answer))
                    has_any = True

            if has_any:
                valid_records += 1
                if args.test_limit and valid_records >= args.test_limit:
                    break
            else:
                skipped_no_variants += 1

    print(f"Total lines read:           {total_lines}")
    print(f"Skipped (answer mismatch):  {skipped_no_match}")
    print(f"Skipped (no ground truth):  {skipped_no_answer}")
    print(f"Skipped (no variants):      {skipped_no_variants}")
    print(f"Valid records:              {valid_records}")
    print(f"Total API calls to make:    {len(all_tasks)}")

    # Count per depth
    tasks_per_depth = {d: 0 for d in range(1, MAX_DEPTH + 1)}
    for _, d, *_ in all_tasks:
        tasks_per_depth[d] += 1
    print("\nTasks per depth:")
    for d in range(1, MAX_DEPTH + 1):
        print(f"  Depth {d}: {tasks_per_depth[d]} expressions")
    print()

    # ---- Step 2: Submit all tasks ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_by_depth: Dict[int, List[Dict]] = {d: [] for d in range(1, MAX_DEPTH + 1)}

    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    futures = []

    for rec_idx, d, question, expr_original, answer_text, expr_variant, correct_answer in all_tasks:
        fut = executor.submit(
            process_one_expr,
            rec_idx, d, question, expr_original, answer_text,
            expr_variant, correct_answer, client, args.model
        )
        futures.append(fut)

    valid_per_depth = {d: 0 for d in range(1, MAX_DEPTH + 1)}
    invalid_count = 0

    for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        rec_idx, depth, out_dict = fut.result()
        if out_dict is not None:
            results_by_depth[depth].append(out_dict)
            valid_per_depth[depth] += 1
        else:
            invalid_count += 1

    # ---- Step 3: Shuffle and write per depth + total ----
    random.seed(args.seed)

    all_results = []
    for d in range(1, MAX_DEPTH + 1):
        data_list = results_by_depth[d]
        random.shuffle(data_list)

        with open(OUTPUT_PATHS[d], "w", encoding="utf-8") as fout:
            for item in data_list:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        all_results.extend(data_list)

    # Write total file (all depths combined, reshuffled)
    random.shuffle(all_results)
    with open(OUTPUT_TOTAL_PATH, "w", encoding="utf-8") as fout:
        for item in all_results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- Step 4: Final report ----
    print("\n" + "=" * 60)
    print("Final Report")
    print("=" * 60)
    print(f"Total API calls:            {len(all_tasks)}")
    print(f"Invalid/skipped responses:  {invalid_count}")
    print()
    for d in range(1, MAX_DEPTH + 1):
        print(f"  Depth {d}: {valid_per_depth[d]} valid lines -> {OUTPUT_PATHS[d]}")
    print(f"\nTotal valid lines written:  {sum(valid_per_depth.values())}")
    print("=" * 60)

if __name__ == "__main__":
    main()