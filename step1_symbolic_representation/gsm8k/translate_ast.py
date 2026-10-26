#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 (GSM8K): Symbolic Representation.

Convert each natural-language GSM8K solution into a single arithmetic
expression (the AST used by Step 2). Every expression is evaluated and compared
with the gold answer after "####"; only verified expressions are kept.

- Input:  GSM8K train set as JSONL (fields: "question", "answer").
- Output: Same records with an added "expr" field.

Usage:
    python translate_ast.py --input train.jsonl --output train_expr.jsonl
    python translate_ast.py --input train.jsonl --output train_expr.jsonl \
        --model gpt-4o-mini --base_url http://localhost:8000/v1
"""

import argparse
import json
import os
import sys
import time
import re
from typing import Any, Dict, Iterable, List, Tuple, Union
from tqdm import tqdm

from openai import OpenAI

# ---------- Validation and normalization ----------

ALLOWED_EXPR_RE = re.compile(r'^[0-9.\+\-\*\/\(\)\s]+$')

def normalize_to_ascii_ops(s: str) -> str:
    """Normalize unicode math symbols to ASCII, strip code fences, and remove spaces."""
    mapping = {
        '×': '*', '∙': '*', '·': '*', '✕': '*',
        '÷': '/', '⁄': '/', '／': '/',
        '−': '-', '–': '-', '—': '-',
        '（': '(', '）': ')',
    }
    s = s.strip()
    # Remove fenced code blocks if present
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            s = s.split("\n", 1)[1]
        s = s.strip()
    # Remove leading labels like "Expression:" if any
    if ":" in s and s.split(":", 1)[0].strip().lower() in {"expression", "expr"}:
        s = s.split(":", 1)[1].strip()
    for k, v in mapping.items():
        s = s.replace(k, v)
    # Remove spaces (parser can ignore spaces, but we keep it clean)
    s = re.sub(r'\s+', '', s)
    return s

def parentheses_balanced(expr: str) -> bool:
    """Check if parentheses are balanced."""
    depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0

def validate_allowed_tokens(expr: str) -> None:
    """Raise if the expression contains disallowed content."""
    if '=' in expr:
        raise ValueError("Equals sign '=' is not allowed.")
    if not ALLOWED_EXPR_RE.match(expr):
        raise ValueError("Illegal characters found.")
    if not parentheses_balanced(expr):
        raise ValueError("Unbalanced parentheses.")

# ---------- Expression evaluation ----------

def safe_eval_expr(expr: str) -> float:
    """
    Safely evaluate a pure arithmetic expression (digits, +-*/, parentheses only).
    Uses Python's eval with a restricted namespace.
    """
    # Double-check: only allowed characters
    if not re.match(r'^[0-9.\+\-\*\/\(\)\s]+$', expr):
        raise ValueError(f"Expression contains disallowed characters: {expr}")
    # Evaluate in empty namespace for safety
    return eval(expr, {"__builtins__": {}}, {})

def extract_answer_from_text(answer_text: str) -> float:
    """
    Extract the numerical answer from the answer text.
    The answer is after the #### marker.
    """
    if '####' in answer_text:
        answer_part = answer_text.split('####')[-1].strip()
        # Remove any commas and extract the number
        answer_part = answer_part.replace(',', '')
        match = re.search(r'-?\d+\.?\d*', answer_part)
        if match:
            return float(match.group())
    return None

# ---------- GPT prompt ----------

SYSTEM_PROMPT = (
    "You convert a natural-language math solution into exactly ONE arithmetic expression.\n"
    "Output rules:\n"
    "Output ONLY a single expression. Do not output any text, explanation, equals sign, numeric result, or markdown/code fences.\n"
    "Allowed characters: digits 0-9, decimal point '.', operators '+-*/', and parentheses '()' only. No variables, no units, no commas, no words.\n"
    "If the answer contains annotated intermediate steps of the form <<expr=result>>result (or similar like expr=<<expr=result>>result), then reconstruct the final expression by replacing numeric intermediate results with their original expressions recursively. Wrap each replacement in parentheses when substituting to preserve order.\n"
    "If the answer shows intermediate calculations without explicit <<...>> annotations (for example 3*2=6 then 6*2=12 then 12*52=624), infer the chain and expand likewise, replacing each derived number with its generating expression.\n"
    "Do not evaluate or simplify numeric subexpressions. Keep the original operations. For fractions prefer the original fraction notation rather than its decimal form.\n"
    "If an intermediate number has no known generating expression in the answer, leave that number as-is.\n"
    "Keep parentheses to preserve the intended order of operations. Add parentheses around substituted subexpressions to avoid ambiguity.\n"
    "If you cannot reconstruct an expanded expression, output a single expression built from the final arithmetic line as-is (still obey rules 1 and 2).\n"
    "Use deterministic settings (temperature=0) and return a single-line expression."
)

USER_TMPL = (
    "For example, given the natural-language answer:\n"
    "He writes each friend 3*2=<<3*2=6>>6 pages a week. So he writes 6*2=<<6*2=12>>12 pages every week. That means he writes 12*52=<<12*52=624>>624 pages a year."
    "\nThe correct final arithmetic expression is:\n"
    "((3*2)*2)*52\n\n"
    "Natural-language answer:\n"
    "{answer}\n\n"
    "Return only the single final arithmetic expression that expands intermediate results into their original subexpressions (see system rules)."
)

# ---------- GPT call with simple retry ----------

def call_gpt_expr_once(client: OpenAI, model: str, answer_text: str, retry_wait: float = 2.0) -> str:
    """Call GPT once and return the normalized, validated expression (may raise)."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TMPL.format(answer=answer_text)}
        ]
    )
    raw = resp.choices[0].message.content or ""
    expr = normalize_to_ascii_ops(raw)
    validate_allowed_tokens(expr)
    return expr

def call_gpt_expr_validated(client: OpenAI, model: str, answer_text: str,
                            expected_answer: float,
                            max_attempts: int = 5, retry_wait: float = 2.0) -> Tuple[str, bool]:
    """
    Call GPT up to max_attempts times. Each time:
      1. Get expression from GPT
      2. Evaluate it
      3. If it matches expected_answer, return immediately
    
    Returns:
        (expr, is_correct) - the best expression found and whether it matched
    """
    last_expr = None
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            expr = call_gpt_expr_once(client, model, answer_text)
            last_expr = expr

            # Evaluate and compare with ground truth
            try:
                computed = safe_eval_expr(expr)
                if expected_answer is not None and abs(computed - expected_answer) < 1e-4:
                    return expr, True  # correct!
            except Exception as eval_err:
                last_err = eval_err
                # Expression is syntactically valid but eval failed (e.g. division by zero)

            # If we get here, expression didn't match - retry with backoff
            time.sleep(retry_wait * attempt)

        except Exception as e:
            last_err = e
            time.sleep(retry_wait * attempt)

    # All attempts exhausted, return last expression as incorrect
    return last_expr, False

# ---------- I/O helpers ----------

def is_jsonl(path: str) -> bool:
    if path.lower().endswith(".jsonl"):
        return True
    return False

def read_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects.")
        return data

def write_json_array(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def stream_jsonl(path: str) -> Iterable[Tuple[Dict[str, Any], str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line), "\n"

def write_jsonl(path: str, objs: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------- Main pipeline ----------

def process_record(obj: Dict[str, Any], client: OpenAI, model: str,
                   max_attempts: int = 5) -> Tuple[Dict[str, Any], bool]:
    """
    Process a single record:
      1. Extract ground truth from #### in answer
      2. Call GPT up to max_attempts times until expression matches ground truth
      3. Return (record_with_expr, is_correct)
    
    If correct: record includes "expr" field
    If incorrect after all attempts: record includes "expr" (last attempt) and "expr_error"
    """
    if "answer" not in obj or not isinstance(obj["answer"], str):
        raise ValueError("Record missing 'answer' string.")

    answer_text = obj["answer"]
    expected_answer = extract_answer_from_text(answer_text)

    if expected_answer is None:
        # No ground truth available, just call once
        expr = call_gpt_expr_once(client, model, answer_text)
        out = dict(obj)
        out["expr"] = expr
        return out, True  # can't verify, assume ok

    expr, is_correct = call_gpt_expr_validated(
        client, model, answer_text, expected_answer, max_attempts=max_attempts
    )

    out = dict(obj)
    if is_correct:
        out["expr"] = expr
    else:
        out["expr"] = expr  # keep last attempt for debugging
        out["expr_error"] = f"Expression evaluates incorrectly after {max_attempts} attempts"
        if expr:
            try:
                computed = safe_eval_expr(expr)
                out["expr_computed"] = computed
            except:
                pass
        out["expected_answer"] = expected_answer

    return out, is_correct


def run(input_path: str, output_path: str, model: str, max_attempts: int = 5,
        base_url: str = None, limit: int = None) -> None:
    if base_url is None and not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Please set OPENAI_API_KEY in your environment.")

    # base_url allows any OpenAI-compatible server (e.g. a local vLLM server)
    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    total_count = 0
    correct_count = 0
    incorrect_count = 0
    error_count = 0

    if is_jsonl(input_path):
        out_objects: List[Dict[str, Any]] = []

        total_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
        total = min(total_lines, limit) if limit else total_lines

        with tqdm(total=total, desc="Processing JSONL", unit="item") as pbar:
            for i, (obj, _) in enumerate(stream_jsonl(input_path)):
                if limit and i >= limit:
                    break
                total_count += 1
                try:
                    result, is_correct = process_record(obj, client, model, max_attempts)
                    if is_correct:
                        correct_count += 1
                        out_objects.append(result)  # only save correct ones
                    else:
                        incorrect_count += 1
                        # Don't save incorrect ones (as requested)
                except Exception as e:
                    error_count += 1
                    # Don't save error ones either
                pbar.update(1)
                # Update progress bar description with accuracy
                if total_count > 0:
                    acc = correct_count / total_count * 100
                    pbar.set_postfix(correct=correct_count, total=total_count, acc=f"{acc:.1f}%")

        write_jsonl(output_path, out_objects)

    else:
        items = read_json_array(input_path)
        if limit:
            items = items[:limit]
        out_items: List[Dict[str, Any]] = []

        with tqdm(total=len(items), desc="Processing JSON", unit="item") as pbar:
            for obj in items:
                total_count += 1
                try:
                    result, is_correct = process_record(obj, client, model, max_attempts)
                    if is_correct:
                        correct_count += 1
                        out_items.append(result)
                    else:
                        incorrect_count += 1
                except Exception as e:
                    error_count += 1
                pbar.update(1)
                if total_count > 0:
                    acc = correct_count / total_count * 100
                    pbar.set_postfix(correct=correct_count, total=total_count, acc=f"{acc:.1f}%")

        write_json_array(output_path, out_items)

    # ---------- Final report ----------
    print("\n" + "=" * 60)
    print("Final Report")
    print("=" * 60)
    print(f"Total records processed:  {total_count}")
    print(f"Correct (saved):          {correct_count}")
    print(f"Incorrect (discarded):    {incorrect_count}")
    print(f"Errors (discarded):       {error_count}")
    if total_count > 0:
        print(f"Accuracy:                 {correct_count / total_count * 100:.2f}%")
    print("=" * 60)
    print(f"Output written to: {output_path}")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Translate natural-language answers to arithmetic expressions (with validation).")
    parser.add_argument("--input", required=True, help="Path to input .json or .jsonl file")
    parser.add_argument("--output", required=True, help="Path to output .json or .jsonl file")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--max-attempts", type=int, default=5, help="Max GPT attempts per record for validation (default: 5)")
    parser.add_argument("--base_url", default=None, help="Base URL of an OpenAI-compatible API (default: OpenAI)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records (for quick tests)")
    args = parser.parse_args()

    run(args.input, args.output, args.model, args.max_attempts, args.base_url, args.limit)

if __name__ == "__main__":
    main()