#!/usr/bin/env python3
"""
Step 3 (FOLIO): Natural Language Reconstruction
===============================================
Translate each Prover9 proof trace (True/False) or Mace4 solver output
(Uncertain) into a natural-language chain-of-thought with an LLM. Each
explanation is kept only if its final verdict matches the gold label.

Modes:
  translate  - translate every symbolic proof path into natural language.
  balance    - generate additional explanations for Uncertain examples.
               Uncertain examples have no proof trace (only Mace4 models), so
               Step 2 cannot produce multiple proof paths for them; this mode
               adds explanations from different angles until Uncertain reaches
               --target-ratio x max(#True, #False).

Usage:
    export OPENAI_API_KEY="sk-..."

    python translate_back_folio.py translate \
        --input  folio_sft_data.jsonl \
        --output folio_sft_nl.jsonl \
        --model gpt-4o-mini --max-workers 50

    python translate_back_folio.py balance \
        --input  folio_sft_nl.jsonl \
        --output folio_sft_nl_balanced.jsonl \
        --model gpt-4o-mini --max-workers 50 --target-ratio 0.7

    # Any OpenAI-compatible server can be used via --base_url
    python translate_back_folio.py translate --input ... --output ... \
        --model Qwen/Qwen2.5-72B-Instruct --base_url http://localhost:8000/v1
"""

import os
import json
import re
import argparse
import random
import math
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


# ============================================================================
# Prompt Templates — Original Translation (Mode 1)
# ============================================================================

SYSTEM_PROMPT = (
    "You are an expert at explaining logical reasoning in clear, natural English. "
    "You translate formal logical proofs into plain-language explanations that "
    "read like a thoughtful person working through the problem. "
    "Vary your style — sometimes start with the conclusion and work backward, "
    "sometimes build up from the premises, sometimes highlight the key insight first. "
    "Do NOT use rigid formatting like 'Step 1:', 'Step 2:'. "
    "Write flowing prose, as if explaining to a smart friend."
)


def build_prompt_true(record: dict) -> str:
    return (
        "Below is a logical reasoning problem. The premises logically entail the conclusion.\n"
        "I will show you the premises (in both natural language and formal logic), the conclusion, "
        "and a formal proof from an automated theorem prover.\n\n"
        "Your job: write a natural language explanation of WHY the conclusion follows from the premises. "
        "Trace the reasoning in your own words, referencing the key logical steps. "
        "End your explanation with a clear verdict line: 'Therefore, the answer is: True'\n\n"
        f"[Premises - Natural Language]\n{record.get('premises_nl', '(not available)')}\n\n"
        f"[Premises - Formal Logic]\n{record.get('premises_fol', '')}\n\n"
        f"[Conclusion]\n{record.get('conclusion_fol', '')}\n\n"
        f"[Formal Proof (Prover9)]\n{record.get('solution', '')}\n\n"
        "Write your natural language explanation below. Be natural and varied in style — "
        "do not mechanically list steps. End with 'The answer is: True'"
    )


def build_prompt_false(record: dict) -> str:
    return (
        "Below is a logical reasoning problem. The premises logically CONTRADICT the conclusion — "
        "the negation of the conclusion can be proved from the premises.\n"
        "I will show you the premises, the conclusion, "
        "and a formal proof that the negation holds.\n\n"
        "Your job: write a natural language explanation of WHY the conclusion is false given the premises. "
        "Show how the premises lead to the opposite of what the conclusion claims. "
        "End your explanation with: 'Therefore, the answer is: False'\n\n"
        f"[Premises - Natural Language]\n{record.get('premises_nl', '(not available)')}\n\n"
        f"[Premises - Formal Logic]\n{record.get('premises_fol', '')}\n\n"
        f"[Conclusion]\n{record.get('conclusion_fol', '')}\n\n"
        f"[Formal Proof (negation proved)]\n{record.get('solution', '')}\n\n"
        "Write your natural language explanation below. Be natural — "
        "do not mechanically list steps. End with 'The answer is: False'"
    )


def build_prompt_uncertain(record: dict) -> str:
    return (
        "Below is a logical reasoning problem. The conclusion is INDEPENDENT of the premises — "
        "it can be neither proved nor disproved from what is given. "
        "There exist scenarios where the premises are true and the conclusion holds, "
        "and other scenarios where the premises are true but the conclusion fails.\n\n"
        "Your job: write a natural language explanation of WHY we cannot determine the truth value. "
        "Point out what information is missing or why the premises don't constrain the conclusion. "
        "End your explanation with: 'Therefore, the answer is: Uncertain'\n\n"
        f"[Premises - Natural Language]\n{record.get('premises_nl', '(not available)')}\n\n"
        f"[Premises - Formal Logic]\n{record.get('premises_fol', '')}\n\n"
        f"[Conclusion]\n{record.get('conclusion_fol', '')}\n\n"
        f"[Solver Output]\n{record.get('solution', '')}\n\n"
        "Write your natural language explanation below. Be natural — "
        "explain what's missing or ambiguous. End with 'The answer is: Uncertain'"
    )


def build_prompt(record: dict) -> str:
    label = record.get('label', '')
    if label == 'True':
        return build_prompt_true(record)
    elif label == 'False':
        return build_prompt_false(record)
    else:
        return build_prompt_uncertain(record)


# ============================================================================
# Prompt Templates — Uncertain Augmentation (Mode 2)
# ============================================================================

# We use multiple strategies to generate diverse Uncertain explanations:
#   1. Fresh generation from different angles
#   2. Paraphrasing existing explanations

UNCERTAIN_FRESH_SYSTEM = (
    "You are an expert at explaining logical reasoning in clear, natural English. "
    "You write flowing prose explanations — no rigid 'Step 1, Step 2' formatting. "
    "Your explanations should read like a thoughtful person working through a logic problem."
)

# Multiple angle prompts for fresh generation — each encourages a different
# reasoning approach so the model sees diverse explanation styles during training.
UNCERTAIN_FRESH_ANGLES = [
    # Angle 0: Focus on missing information
    (
        "Below is a logical reasoning problem where the conclusion CANNOT be determined "
        "from the premises — it is neither provably true nor provably false.\n\n"
        "[Premises - Natural Language]\n{premises_nl}\n\n"
        "[Premises - Formal Logic]\n{premises_fol}\n\n"
        "[Conclusion - Natural Language]\n{conclusion_nl}\n\n"
        "[Conclusion - Formal Logic]\n{conclusion_fol}\n\n"
        "Write a natural language explanation focusing on WHAT INFORMATION IS MISSING "
        "from the premises that would be needed to determine the conclusion. "
        "Explain clearly why the premises leave a gap. "
        "but write your explanation in plain English. "
        "End with exactly: 'Therefore, the answer is: Uncertain'"
    ),
    # Angle 1: Construct two scenarios
    (
        "Below is a logical reasoning problem where the conclusion CANNOT be determined "
        "from the premises — it is neither provably true nor provably false.\n\n"
        "[Premises - Natural Language]\n{premises_nl}\n\n"
        "[Premises - Formal Logic]\n{premises_fol}\n\n"
        "[Conclusion - Natural Language]\n{conclusion_nl}\n\n"
        "[Conclusion - Formal Logic]\n{conclusion_fol}\n\n"
        "Write a natural language explanation by CONSTRUCTING TWO SCENARIOS: "
        "one where all premises are true AND the conclusion holds, "
        "and another where all premises are true BUT the conclusion fails. "
        "This demonstrates that the premises alone don't determine the conclusion. "
        "End with exactly: 'Therefore, the answer is: Uncertain'"
    ),
    # Angle 2: Analyze logical independence
    (
        "Below is a logical reasoning problem where the conclusion CANNOT be determined "
        "from the premises — it is neither provably true nor provably false.\n\n"
        "[Premises - Natural Language]\n{premises_nl}\n\n"
        "[Premises - Formal Logic]\n{premises_fol}\n\n"
        "[Conclusion - Natural Language]\n{conclusion_nl}\n\n"
        "[Conclusion - Formal Logic]\n{conclusion_fol}\n\n"
        "Write a natural language explanation analyzing the LOGICAL INDEPENDENCE "
        "between the premises and the conclusion. Identify which predicates or concepts "
        "in the conclusion are not constrained by any premise. "
        "Explain why no chain of reasoning can connect them. "
        "End with exactly: 'Therefore, the answer is: Uncertain'"
    ),
    # Angle 3: Start from conclusion, work backward
    (
        "Below is a logical reasoning problem where the conclusion CANNOT be determined "
        "from the premises — it is neither provably true nor provably false.\n\n"
        "[Premises - Natural Language]\n{premises_nl}\n\n"
        "[Premises - Formal Logic]\n{premises_fol}\n\n"
        "[Conclusion - Natural Language]\n{conclusion_nl}\n\n"
        "[Conclusion - Formal Logic]\n{conclusion_fol}\n\n"
        "Write a natural language explanation by STARTING FROM THE CONCLUSION and "
        "working backward: what would we need to prove it? What would we need to "
        "disprove it? Show that the premises provide neither. "
        "End with exactly: 'Therefore, the answer is: Uncertain'"
    ),
    # Angle 4: Highlight the key insight directly
    (
        "Below is a logical reasoning problem where the conclusion CANNOT be determined "
        "from the premises — it is neither provably true nor provably false.\n\n"
        "[Premises - Natural Language]\n{premises_nl}\n\n"
        "[Premises - Formal Logic]\n{premises_fol}\n\n"
        "[Conclusion - Natural Language]\n{conclusion_nl}\n\n"
        "[Conclusion - Formal Logic]\n{conclusion_fol}\n\n"
        "Write a natural language explanation that LEADS WITH THE KEY INSIGHT — "
        "immediately identify the core reason why the premises cannot determine "
        "the conclusion, then elaborate with supporting reasoning. "
        "End with exactly: 'Therefore, the answer is: Uncertain'"
    ),
]

UNCERTAIN_PARAPHRASE_SYSTEM = (
    "You are an expert at rewriting logical explanations. "
    "You preserve the core reasoning and logical accuracy while completely "
    "changing the wording, sentence structure, and flow. "
    "Your rewrite should feel like a DIFFERENT person explaining the same logic."
)

UNCERTAIN_PARAPHRASE_PROMPT = (
    "Below is a logical reasoning explanation for why a conclusion cannot be "
    "determined from the given premises (the answer is Uncertain).\n\n"
    "[Original Explanation]\n{explanation}\n\n"
    "[Premises - Natural Language]\n{premises_nl}\n\n"
    "[Premises - Formal Logic]\n{premises_fol}\n\n"
    "[Conclusion - Natural Language]\n{conclusion_nl}\n\n"
    "[Conclusion - Formal Logic]\n{conclusion_fol}\n\n"
    "REWRITE this explanation in a COMPLETELY DIFFERENT style. Requirements:\n"
    "- Preserve the logical reasoning and correctness\n"
    "- Use DIFFERENT vocabulary, sentence structure, and flow\n"
    "- Change the order of arguments if possible\n"
    "- Use the formal logic as reference to keep reasoning precise\n"
    "- {style_instruction}\n"
    "- End with exactly: 'Therefore, the answer is: Uncertain'\n\n"
    "Write your rewritten explanation below:"
)

PARAPHRASE_STYLES = [
    "Write more concisely — be direct and to the point",
    "Write more elaborately — add more detail and nuance to each point",
    "Start by analyzing the conclusion first, then examine what the premises offer",
    "Use a conversational tone, as if chatting with a colleague",
    "Start with the key insight immediately, then support it with details",
    "Use a methodical approach — examine each premise's relevance to the conclusion",
    "Write as if you're teaching a student who is learning logic for the first time",
    "Focus on constructing counterexamples to show independence",
    "Emphasize what we CAN determine vs what we CANNOT from the premises",
    "Write in a formal academic style with precise logical terminology",
]


# ============================================================================
# Answer Validation
# ============================================================================

def extract_predicted_label(explanation: str) -> Optional[str]:
    m = re.search(
        r'(?:the\s+answer\s+is|verdict|conclusion\s+is)\s*:\s*(True|False|Uncertain)',
        explanation,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).capitalize()
    tail = explanation[-100:].lower()
    if 'true' in tail and 'false' not in tail and 'uncertain' not in tail:
        return 'True'
    elif 'false' in tail and 'true' not in tail and 'uncertain' not in tail:
        return 'False'
    elif 'uncertain' in tail:
        return 'Uncertain'
    return None


def validate_explanation(explanation: str, expected_label: str) -> bool:
    predicted = extract_predicted_label(explanation)
    if predicted is None:
        return False
    return predicted.lower() == expected_label.lower()


# ============================================================================
# API Calls
# ============================================================================

def call_api_translate(
    record: dict,
    client: OpenAI,
    model: str,
    max_retries: int = 2,
) -> Tuple[int, Optional[str]]:
    """Original translation: formal proof -> NL explanation."""
    eid = record.get('example_id', -1)
    label = record.get('label', '')
    user_msg = build_prompt(record)

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.7 if attempt == 0 else 0.9,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            explanation = resp.choices[0].message.content.strip()
            if explanation.startswith("[ERROR]"):
                continue
            if validate_explanation(explanation, label):
                return eid, explanation
            if attempt == max_retries:
                explanation += f"\n\nTherefore, the answer is: {label}"
                return eid, explanation
        except Exception as e:
            if attempt == max_retries:
                return eid, None
    return eid, None


def call_api_uncertain_fresh(
    record: dict,
    angle_idx: int,
    client: OpenAI,
    model: str,
    max_retries: int = 2,
) -> Optional[str]:
    """Generate a fresh Uncertain explanation from a specific angle."""
    angle_template = UNCERTAIN_FRESH_ANGLES[angle_idx % len(UNCERTAIN_FRESH_ANGLES)]
    user_msg = angle_template.format(
        premises_nl=record.get('premises_nl', ''),
        premises_fol=record.get('premises_fol', ''),
        conclusion_nl=record.get('conclusion_nl', ''),
        conclusion_fol=record.get('conclusion_fol', ''),
    )

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.7 + 0.1 * attempt,
                messages=[
                    {"role": "system", "content": UNCERTAIN_FRESH_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            explanation = resp.choices[0].message.content.strip()
            if validate_explanation(explanation, 'Uncertain'):
                return explanation
            if attempt == max_retries:
                explanation += "\n\nTherefore, the answer is: Uncertain"
                return explanation
        except Exception:
            if attempt == max_retries:
                return None
    return None


def call_api_uncertain_paraphrase(
    record: dict,
    source_explanation: str,
    style_idx: int,
    client: OpenAI,
    model: str,
    max_retries: int = 2,
) -> Optional[str]:
    """Paraphrase an existing Uncertain explanation with a specific style."""
    style = PARAPHRASE_STYLES[style_idx % len(PARAPHRASE_STYLES)]
    user_msg = UNCERTAIN_PARAPHRASE_PROMPT.format(
        explanation=source_explanation,
        premises_nl=record.get('premises_nl', ''),
        premises_fol=record.get('premises_fol', ''),
        conclusion_nl=record.get('conclusion_nl', ''),
        conclusion_fol=record.get('conclusion_fol', ''),
        style_instruction=style,
    )

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.8 + 0.1 * attempt,
                messages=[
                    {"role": "system", "content": UNCERTAIN_PARAPHRASE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            explanation = resp.choices[0].message.content.strip()
            if validate_explanation(explanation, 'Uncertain'):
                return explanation
            if attempt == max_retries:
                explanation += "\n\nTherefore, the answer is: Uncertain"
                return explanation
        except Exception:
            if attempt == max_retries:
                return None
    return None


# ============================================================================
# Balance Task Generation
# ============================================================================

def plan_augmentation(
    uncertain_records: List[dict],
    target_count: int,
    num_fresh_angles: int = 5,
) -> List[dict]:
    """
    Plan augmentation tasks for Uncertain examples.
    
    Strategy:
    - First, generate fresh explanations from each angle (up to 5 per example)
    - Then, fill remaining with paraphrases of the original + fresh explanations
    
    Returns a list of task dicts with keys:
        'record', 'task_type' ('fresh'|'paraphrase'), 
        'angle_idx' or 'style_idx', 'source_explanation'
    """
    n_existing = len(uncertain_records)
    n_needed = target_count - n_existing
    
    if n_needed <= 0:
        print(f"  Already have {n_existing} >= target {target_count}, nothing to do.")
        return []
    
    # Calculate per-example budget
    per_example = math.ceil(n_needed / n_existing)
    
    # Cap the number of angles for fresh generation
    num_fresh = min(per_example, num_fresh_angles)
    num_paraphrase = per_example - num_fresh
    
    print(f"  Need {n_needed} more Uncertain samples")
    print(f"  {n_existing} base examples × ({num_fresh} fresh + {num_paraphrase} paraphrase) = {n_existing * per_example}")
    
    tasks = []
    for record in uncertain_records:
        # Fresh generation tasks (different angles)
        for i in range(num_fresh):
            tasks.append({
                'record': record,
                'task_type': 'fresh',
                'angle_idx': i,
            })
        
        # Paraphrase tasks (different styles)
        for i in range(num_paraphrase):
            tasks.append({
                'record': record,
                'task_type': 'paraphrase',
                'style_idx': i,
                'source_explanation': record.get('explanation', ''),
            })
    
    # Trim to exact target
    random.shuffle(tasks)
    tasks = tasks[:n_needed]
    
    task_types = Counter(t['task_type'] for t in tasks)
    print(f"  Final task plan: {dict(task_types)}")
    
    return tasks


def execute_task(
    task: dict,
    client: OpenAI,
    model: str,
    max_retries: int,
) -> Optional[dict]:
    """Execute a single augmentation task, return augmented record or None."""
    record = task['record']
    task_type = task['task_type']
    
    if task_type == 'fresh':
        explanation = call_api_uncertain_fresh(
            record, task['angle_idx'], client, model, max_retries
        )
        variant_info = f"fresh_angle_{task['angle_idx']}"
    else:
        explanation = call_api_uncertain_paraphrase(
            record, task['source_explanation'], task['style_idx'],
            client, model, max_retries,
        )
        variant_info = f"paraphrase_style_{task['style_idx']}"
    
    if explanation is None:
        return None
    
    # Build augmented record — copy all fields from original
    out = dict(record)
    out['explanation'] = explanation
    out['solution_source'] = 'uncertain_augmented'
    out['transform'] = variant_info
    out['transform_desc'] = f"Uncertain augmentation: {task_type}"
    
    return out


# ============================================================================
# Mode 1: Translate (original functionality)
# ============================================================================

def process_record_translate(
    idx: int,
    record: dict,
    client: OpenAI,
    model: str,
    max_retries: int,
) -> Tuple[int, int, Optional[dict]]:
    eid, explanation = call_api_translate(record, client, model, max_retries)
    if explanation is None:
        return idx, eid, None
    out = dict(record)
    out['explanation'] = explanation
    return idx, eid, out


def load_checkpoint(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                key = (
                    r.get('example_id', -1),
                    r.get('proof_index', 0),
                    r.get('solution_source', ''),
                )
                done.add(key)
            except json.JSONDecodeError:
                continue
    return done


def get_record_key(record: dict) -> tuple:
    return (
        record.get('example_id', -1),
        record.get('proof_index', 0),
        record.get('solution_source', ''),
    )


def run_translate(args, client):
    """Mode 1: Original translation of formal proofs to NL."""
    print(f"Loading input: {args.input}")
    records = []
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    print(f"  Total records: {len(records)}")

    done_keys = set()
    if args.resume:
        done_keys = load_checkpoint(args.output)
        print(f"  Resuming: {len(done_keys)} already processed")

    todo = [r for r in records if get_record_key(r) not in done_keys]
    if args.test_limit:
        todo = todo[:args.test_limit]
        print(f"  Test mode: processing {len(todo)} records")
    else:
        print(f"  To process: {len(todo)} records")

    if not todo:
        print("Nothing to do!")
        return

    label_dist = Counter(r['label'] for r in todo)
    print(f"\n  Label distribution: {dict(label_dist)}")

    valid_count = 0
    error_count = 0
    mode = 'a' if args.resume and os.path.exists(args.output) else 'w'
    fout = open(args.output, mode, buffering=1, encoding='utf-8')

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for idx, record in enumerate(todo):
                fut = executor.submit(
                    process_record_translate,
                    idx, record, client, args.model, args.max_retries,
                )
                futures[fut] = record

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
                idx, eid, out_dict = fut.result()
                if out_dict is not None:
                    fout.write(json.dumps(out_dict, ensure_ascii=False) + '\n')
                    valid_count += 1
                else:
                    error_count += 1
    finally:
        fout.close()

    print(f"\nDone: {valid_count} valid, {error_count} errors")
    print(f"Output: {args.output}")


# ============================================================================
# Mode 2: Balance Uncertain
# ============================================================================

def run_balance(args, client):
    """Mode 2: Augment Uncertain examples to balance label distribution."""
    print(f"Loading input: {args.input}")
    records = []
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    print(f"  Total records: {len(records)}")

    # Count labels
    label_counts = Counter(r['label'] for r in records)
    print(f"\n  Current label distribution:")
    for lbl in ['True', 'False', 'Uncertain']:
        print(f"    {lbl}: {label_counts.get(lbl, 0)}")

    # Determine target
    max_count = max(label_counts.get('True', 0), label_counts.get('False', 0))
    target_count = int(max_count * args.target_ratio)
    current_uncertain = label_counts.get('Uncertain', 0)

    print(f"\n  Target Uncertain count: {target_count} (ratio={args.target_ratio})")
    print(f"  Current Uncertain: {current_uncertain}")

    if current_uncertain >= target_count:
        print("  Already balanced! Copying input to output.")
        with open(args.output, 'w') as fout:
            for r in records:
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
        return

    # Get Uncertain records
    uncertain_records = [r for r in records if r['label'] == 'Uncertain']
    non_uncertain = [r for r in records if r['label'] != 'Uncertain']

    # Plan augmentation
    print(f"\n  Planning augmentation...")
    tasks = plan_augmentation(uncertain_records, target_count, num_fresh_angles=5)

    if not tasks:
        return

    # Check for resume: load already-generated augmentations
    existing_augmented = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    if r.get('solution_source') == 'uncertain_augmented':
                        existing_augmented.append(r)
                except json.JSONDecodeError:
                    continue
        print(f"  Resuming: {len(existing_augmented)} augmented already exist")
        # Reduce tasks needed
        n_still_needed = (target_count - current_uncertain) - len(existing_augmented)
        if n_still_needed <= 0:
            print("  All augmentation done from previous run!")
            tasks = []
        else:
            tasks = tasks[:n_still_needed]
            print(f"  Still need: {n_still_needed} augmentations")

    # Execute tasks
    augmented = list(existing_augmented)
    
    if tasks:
        error_count = 0
        print(f"\n  Generating {len(tasks)} augmented Uncertain explanations...")

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for i, task in enumerate(tasks):
                fut = executor.submit(execute_task, task, client, args.model, args.max_retries)
                futures[fut] = i

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Augmenting"):
                result = fut.result()
                if result is not None:
                    augmented.append(result)
                else:
                    error_count += 1

        print(f"\n  Generated: {len(augmented) - len(existing_augmented)} new")
        print(f"  Errors: {error_count}")

    # Write output: all original records + augmented uncertain, SHUFFLED
    all_records = records + augmented
    random.shuffle(all_records)
    print(f"\n  Writing output (shuffled to avoid label ordering bias)...")
    with open(args.output, 'w', encoding='utf-8') as fout:
        for r in all_records:
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Final stats
    total = len(records) + len(augmented)
    final_uncertain = current_uncertain + len(augmented)
    print(f"\n{'=' * 60}")
    print(f"BALANCE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Original records:    {len(records)}")
    print(f"Augmented Uncertain: {len(augmented)}")
    print(f"Total output:        {total}")
    print(f"\nFinal label distribution:")
    print(f"  True:      {label_counts.get('True', 0)}")
    print(f"  False:     {label_counts.get('False', 0)}")
    print(f"  Uncertain: {final_uncertain} (was {current_uncertain})")
    print(f"\nOutput: {args.output}")

    # Show samples
    if augmented:
        print(f"\n{'=' * 60}")
        print(f"SAMPLE AUGMENTED EXPLANATIONS")
        print(f"{'=' * 60}")
        samples = random.sample(augmented, min(3, len(augmented)))
        for s in samples:
            print(f"\n--- Example {s.get('example_id')} | {s.get('transform', '')} ---")
            expl = s.get('explanation', '')
            print(expl[:400])
            if len(expl) > 400:
                print("...")
            print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FOLIO proof translation & Uncertain balancing"
    )
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # ── Mode 1: translate ────────────────────────────────────────────────
    p_translate = subparsers.add_parser('translate', help='Translate formal proofs to NL')
    p_translate.add_argument("--input", type=str, required=True)
    p_translate.add_argument("--output", type=str, default="folio_sft_nl.jsonl")
    p_translate.add_argument("--model", type=str, default="gpt-4o-mini")
    p_translate.add_argument("--max-workers", type=int, default=50)
    p_translate.add_argument("--max-retries", type=int, default=2)
    p_translate.add_argument("--test-limit", type=int, default=None)
    p_translate.add_argument("--resume", action="store_true")
    p_translate.add_argument("--seed", type=int, default=42)
    p_translate.add_argument("--base_url", type=str, default=None,
                             help="Base URL of an OpenAI-compatible API (default: OpenAI)")

    # ── Mode 2: balance ──────────────────────────────────────────────────
    p_balance = subparsers.add_parser('balance', help='Balance Uncertain via augmentation')
    p_balance.add_argument("--input", type=str, required=True,
                           help="Already-translated JSONL (with explanations)")
    p_balance.add_argument("--output", type=str, default="folio_sft_nl_balanced.jsonl")
    p_balance.add_argument("--model", type=str, default="gpt-4o-mini")
    p_balance.add_argument("--max-workers", type=int, default=50)
    p_balance.add_argument("--max-retries", type=int, default=2)
    p_balance.add_argument("--target-ratio", type=float, default=1.0,
                           help="Target Uncertain count as ratio of max(True,False). "
                                "1.0 = match the largest class, 0.8 = 80%% of it")
    p_balance.add_argument("--resume", action="store_true")
    p_balance.add_argument("--seed", type=int, default=42)
    p_balance.add_argument("--base_url", type=str, default=None,
                           help="Base URL of an OpenAI-compatible API (default: OpenAI)")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    # Set seed
    random.seed(args.seed)

    # Check API key
    if args.base_url is None and not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "Please set OPENAI_API_KEY in your environment.\n"
            "  export OPENAI_API_KEY='sk-...'"
        )
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    if args.mode == 'translate':
        run_translate(args, client)
    elif args.mode == 'balance':
        run_balance(args, client)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
