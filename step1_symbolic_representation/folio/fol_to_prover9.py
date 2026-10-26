#!/usr/bin/env python3
"""
Step 1 (FOLIO): Symbolic Representation
=======================================
Reads the FOLIO train split from HuggingFace, converts FOL formulas to Prover9
syntax, runs Prover9/Mace4, and reports results with solution explanations.

Output: JSONL file where each line is one example with a `solution` field
explaining why the predicted label is True/False/Uncertain, including the
Prover9 proof or Mace4 countermodel when available.

Requirements:
    pip install datasets tqdm
    # Prover9/Mace4 must be installed and on PATH
    # Download LADR from: https://www.cs.unm.edu/~mccune/prover9/download/
    # Then: tar xzf LADR-2009-11A.tar.gz && cd LADR-2009-11A && make all
    # Add LADR-2009-11A/bin to your PATH

Usage:
    python fol_to_prover9.py [--timeout 60] [--output folio_prover9_results.jsonl]
"""

import re
import json
import subprocess
import tempfile
import os
import argparse
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional


# ============================================================================
# Part 1: FOL → Prover9 Conversion
# ============================================================================

class FOLConversionError(Exception):
    """Raised when FOL formula cannot be converted."""
    pass


def fol_to_prover9(fol: str) -> str:
    """
    Convert a single FOLIO-style FOL formula to Prover9 syntax.
    """
    s = fol.strip()
    if not s:
        raise FOLConversionError("Empty formula")

    # Step 0: Handle hyphens in identifier names FIRST
    s = re.sub(r'(\w)-(\w)', r'\1_\2', s)

    # Step 1: Handle XOR (⊕)
    s = _expand_xor(s)

    # Step 2: Quantifiers
    s = re.sub(r'∀\s*', 'all ', s)
    s = re.sub(r'∃\s*', 'exists ', s)
    s = re.sub(r'\\forall\s*', 'all ', s)
    s = re.sub(r'\\exists\s*', 'exists ', s)

    # Handle "all x,y" → "all x all y"
    def expand_multi_var_quant(match):
        quant = match.group(1)
        vars_str = match.group(2)
        variables = [v.strip() for v in vars_str.split(',')]
        return ' '.join(f'{quant} {v}' for v in variables)

    s = re.sub(r'(all|exists)\s+(\w+(?:\s*,\s*\w+)+)', expand_multi_var_quant, s)

    # Step 3: Connectives
    s = s.replace('∧', ' & ')
    s = s.replace('∨', ' | ')
    s = s.replace('↔', ' <-> ')
    s = s.replace('→', ' -> ')
    s = s.replace('¬', '-')
    s = s.replace('\\wedge', ' & ')
    s = s.replace('\\vee', ' | ')
    s = s.replace('\\neg', '-')
    s = s.replace('\\rightarrow', ' -> ')
    s = s.replace('\\leftrightarrow', ' <-> ')
    s = s.replace('\\land', ' & ')
    s = s.replace('\\lor', ' | ')
    s = s.replace('\\lnot', '-')

    # Step 4: Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # Step 5: Lowercase all non-variable identifiers for consistency
    # This fixes issues like Residentof vs ResidentOf being treated as different predicates
    s = _fix_casing(s)

    # Step 6: Fix unbalanced parentheses (common in FOLIO dataset)
    depth = 0
    for c in s:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
    if depth > 0:
        s += ')' * depth
    elif depth < 0:
        s = '(' * (-depth) + s

    return s


def _expand_xor(s: str) -> str:
    """Expand XOR operator ⊕: A ⊕ B → ((A) | (B)) & -((A) & (B))"""
    while '⊕' in s:
        idx = s.index('⊕')

        left_end = idx - 1
        while left_end >= 0 and s[left_end] == ' ':
            left_end -= 1
        if left_end < 0:
            raise FOLConversionError(f"No left operand for ⊕ in: {s}")
        left_start = _scan_operand_backward(s, left_end)
        left_operand = s[left_start:left_end + 1].strip()

        right_start = idx + len('⊕')
        while right_start < len(s) and s[right_start] == ' ':
            right_start += 1
        if right_start >= len(s):
            raise FOLConversionError(f"No right operand for ⊕ in: {s}")
        right_end = _scan_operand_forward(s, right_start)
        right_operand = s[right_start:right_end + 1].strip()

        expansion = f"(({left_operand}) | ({right_operand})) & -(({left_operand}) & ({right_operand}))"
        s = s[:left_start] + expansion + s[right_end + 1:]
    return s


def _scan_operand_backward(s: str, end: int) -> int:
    """Scan backward from position `end` to find where the operand starts."""
    pos = end
    if s[pos] == ')':
        depth = 1
        pos -= 1
        while pos >= 0 and depth > 0:
            if s[pos] == ')': depth += 1
            elif s[pos] == '(': depth -= 1
            pos -= 1
        pos += 1
        if pos > 0 and (s[pos - 1].isalnum() or s[pos - 1] == '_'):
            pos -= 1
            while pos > 0 and (s[pos - 1].isalnum() or s[pos - 1] == '_'):
                pos -= 1
    else:
        while pos > 0 and (s[pos - 1].isalnum() or s[pos - 1] == '_'):
            pos -= 1
    while pos > 0 and s[pos - 1] in ('¬', '-', '~'):
        pos -= 1
    return pos


def _scan_operand_forward(s: str, start: int) -> int:
    """Scan forward from position `start` to find where the operand ends."""
    pos = start
    length = len(s)
    while pos < length and s[pos] in ('¬', '-', '~'):
        pos += 1
    if pos >= length:
        return length - 1
    if s[pos] == '(':
        depth = 1
        pos += 1
        while pos < length and depth > 0:
            if s[pos] == '(': depth += 1
            elif s[pos] == ')': depth -= 1
            pos += 1
        return pos - 1
    else:
        while pos < length and (s[pos].isalnum() or s[pos] == '_'):
            pos += 1
        if pos < length and s[pos] == '(':
            depth = 1
            pos += 1
            while pos < length and depth > 0:
                if s[pos] == '(': depth += 1
                elif s[pos] == ')': depth -= 1
                pos += 1
            return pos - 1
        return pos - 1


def _fix_casing(s: str) -> str:
    """
    Fix casing for Prover9 compatibility.
    
    In Prover9, variables are u-z (single lowercase letters at the end of alphabet)
    or bound by quantifiers. Everything else should be consistently cased.
    We lowercase all identifiers that are NOT quantifier-bound variables.
    """
    # Find all variables bound by quantifiers
    bound_vars = set()
    for m in re.finditer(r'(?:all|exists)\s+(\w+)', s):
        bound_vars.add(m.group(1))

    result = []
    i = 0

    while i < len(s):
        if s[i].isalpha() or s[i] == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            token = s[i:j]

            # Keep keywords and bound variables as-is
            if token in ('all', 'exists'):
                result.append(token)
            elif token in bound_vars:
                result.append(token)
            elif token.lower() in {v.lower() for v in bound_vars} and len(token) == 1:
                # Single letter that matches a bound var (case insensitive)
                result.append(token)
            else:
                # Lowercase everything else for consistency
                result.append(token.lower())
            i = j
        else:
            result.append(s[i])
            i += 1

    return ''.join(result)


def split_premises_fol(premises_fol: str) -> list[str]:
    """Split a FOLIO premises-FOL string into individual formulas."""
    text = premises_fol.strip()
    if not text:
        return []

    lines = text.split('\n')

    # Try bracket-balanced merging
    formulas = []
    current = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        current += " " + line if current else line
        depth = sum(1 if c == '(' else -1 if c == ')' else 0 for c in current)
        if depth == 0:
            formulas.append(current.strip())
            current = ""

    if current.strip():
        # Unbalanced brackets — fallback to line-by-line
        formulas = [l.strip() for l in lines if l.strip()]

    return formulas


def build_prover9_input(
    premise_formulas: list[str],
    goal_formula: str,
    timeout: int = 60,
    negate_goal: bool = False,
) -> str:
    """Build a complete Prover9 input file."""
    lines = []
    lines.append(f"assign(max_seconds, {timeout}).")
    lines.append("")
    lines.append("formulas(assumptions).")
    for p in premise_formulas:
        clean = p.rstrip('.')
        lines.append(f"  {clean}.")
    lines.append("end_of_list.")
    lines.append("")

    goal_clean = goal_formula.rstrip('.')
    if negate_goal:
        goal_clean = f"-({goal_clean})"
    lines.append("formulas(goals).")
    lines.append(f"  {goal_clean}.")
    lines.append("end_of_list.")

    return "\n".join(lines)


def build_mace4_input(
    premise_formulas: list[str],
    goal_formula: str,
    timeout: int = 60,
    negate_goal: bool = False,
) -> str:
    """
    Build a Mace4 input file to find models.
    
    Mace4 searches for a model satisfying all assumptions.
    To check if "premises + conclusion" is satisfiable, put both in assumptions.
    To check if "premises + ¬conclusion" is satisfiable, put premises + negated goal in assumptions.
    """
    lines = []
    lines.append(f"assign(max_seconds, {timeout}).")
    lines.append("")
    lines.append("formulas(assumptions).")
    for p in premise_formulas:
        clean = p.rstrip('.')
        lines.append(f"  {clean}.")

    goal_clean = goal_formula.rstrip('.')
    if negate_goal:
        # We want model where premises hold AND goal is true
        lines.append(f"  {goal_clean}.")
    else:
        # We want model where premises hold AND goal is false (countermodel)
        lines.append(f"  -({goal_clean}).")
    lines.append("end_of_list.")

    return "\n".join(lines)


# ============================================================================
# Part 2: Running Prover9 and Mace4
# ============================================================================

@dataclass
class ProverResult:
    """Result from running Prover9/Mace4 on a FOLIO example."""
    example_id: int
    story_id: int
    label: str
    predicted_label: Optional[str] = None
    solution: Optional[str] = None
    proof_output: Optional[str] = None
    prover9_proved: Optional[bool] = None
    prover9_proved_neg: Optional[bool] = None
    mace4_countermodel: Optional[bool] = None
    mace4_model_neg: Optional[bool] = None
    conversion_error: Optional[str] = None
    prover9_error: Optional[str] = None
    premises_fol: str = ""
    conclusion_fol: str = ""
    premises_p9: list[str] = field(default_factory=list)
    conclusion_p9: str = ""


def run_prover9(input_text: str, prover9_path: str = "prover9", timeout: int = 60) -> tuple[bool, str]:
    """Run Prover9. Returns: (proved: bool, output: str)"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
        f.write(input_text)
        input_file = f.name

    try:
        result = subprocess.run(
            [prover9_path, '-f', input_file],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
        output = result.stdout + result.stderr
        proved = "THEOREM PROVED" in output or "Exiting with 1 proof" in output
        return proved, output

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError:
        return False, f"ERROR: {prover9_path} not found."
    finally:
        os.unlink(input_file)


def run_mace4(input_text: str, mace4_path: str = "mace4", timeout: int = 60) -> tuple[bool, str]:
    """Run Mace4. Returns: (found_model: bool, output: str)"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
        f.write(input_text)
        input_file = f.name

    try:
        result = subprocess.run(
            [mace4_path, '-f', input_file],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
        output = result.stdout + result.stderr
        found = "Exiting with 1 model" in output
        return found, output

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError:
        return False, f"ERROR: {mace4_path} not found."
    finally:
        os.unlink(input_file)


def determine_label(
    prover9_proved: bool,
    prover9_proved_neg: bool,
    mace4_countermodel: bool,
    mace4_model_neg: bool,
) -> tuple[str, str]:
    """Determine predicted label based on Prover9/Mace4 results."""
    if prover9_proved and not prover9_proved_neg:
        return "True", "Prover9 proved the conclusion from the premises."
    elif prover9_proved_neg and not prover9_proved:
        return "False", "Prover9 proved the negation of the conclusion from the premises."
    elif prover9_proved and prover9_proved_neg:
        return "True", "Prover9 proved both the conclusion and its negation (premises may be contradictory)."
    elif mace4_countermodel and mace4_model_neg:
        return "Uncertain", (
            "Mace4 found both a model where the conclusion holds and a model where it doesn't. "
            "The conclusion is independent of the premises."
        )
    elif mace4_countermodel and not prover9_proved:
        return "False", (
            "Mace4 found a countermodel (premises true, conclusion false). "
            "Prover9 could not prove the conclusion."
        )
    elif mace4_model_neg and not prover9_proved_neg:
        return "True", (
            "Mace4 found a model (premises true, conclusion true). "
            "Prover9 could not prove the negation."
        )
    else:
        return "Uncertain", (
            "Neither Prover9 nor Mace4 could determine the truth value within the time limit."
        )


# ============================================================================
# Part 3: Extraction helpers
# ============================================================================

def _extract_proof_section(prover9_output: str) -> Optional[str]:
    """Extract only the numbered proof steps from Prover9 output."""
    lines = prover9_output.split('\n')
    in_proof = False
    proof_lines = []
    for line in lines:
        if '============================== PROOF =====' in line:
            in_proof = True
            continue
        if in_proof and '============================== end of proof' in line:
            break
        if in_proof:
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                proof_lines.append(stripped)
    return '\n'.join(proof_lines) if proof_lines else None


def _extract_model_section(mace4_output: str) -> Optional[str]:
    """Extract the MODEL section from Mace4 output."""
    lines = mace4_output.split('\n')
    in_model = False
    model_lines = []
    for line in lines:
        if '============================== MODEL =====' in line or 'interpretation' in line.lower():
            in_model = True
        if in_model:
            model_lines.append(line)
        if in_model and '============================== end of model' in line:
            break
    return '\n'.join(model_lines) if model_lines else None


def _extract_model_readable(mace4_output: str) -> Optional[str]:
    """
    Extract a human-readable summary of the Mace4 model.
    Shows the domain, function assignments, and relation assignments.
    """
    model_section = _extract_model_section(mace4_output)
    if not model_section:
        return None
    
    lines = model_section.split('\n')
    readable_parts = []
    
    for line in lines:
        stripped = line.strip()
        # Domain size
        if stripped.startswith('interpretation'):
            m = re.search(r'\[\s*(\d+)\s*\]', stripped)
            if m:
                readable_parts.append(f"Domain size: {m.group(1)}")
        # Function/constant assignments like "function(name, [values])"
        elif stripped.startswith('function(') or stripped.startswith('relation('):
            readable_parts.append(stripped)
    
    if not readable_parts:
        # fallback: return the whole model section
        return model_section
    
    return '\n'.join(readable_parts)


# ============================================================================
# Part 4: Main Pipeline
# ============================================================================

def process_example(
    example: dict,
    prover9_path: str = "prover9",
    mace4_path: str = "mace4",
    timeout: int = 60,
) -> ProverResult:
    """Process a single FOLIO example through the full pipeline."""

    result = ProverResult(
        example_id=example.get('example_id', -1),
        story_id=example.get('story_id', -1),
        label=example.get('label', 'Unknown'),
        premises_fol=example.get('premises-FOL', ''),
        conclusion_fol=example.get('conclusion-FOL', ''),
    )

    # ---- Step 1: Convert FOL to Prover9 ----
    try:
        premise_fols = split_premises_fol(result.premises_fol)
        premise_p9 = [fol_to_prover9(p) for p in premise_fols]
        conclusion_p9 = fol_to_prover9(result.conclusion_fol)
        result.premises_p9 = premise_p9
        result.conclusion_p9 = conclusion_p9
    except Exception as e:
        result.conversion_error = str(e)
        result.solution = f"Conversion error: {e}"
        return result

    # ---- Step 2: Prover9 — prove the goal (True?) ----
    proof_output = None
    try:
        p9_input = build_prover9_input(premise_p9, conclusion_p9, timeout)
        result.prover9_proved, output = run_prover9(p9_input, prover9_path, timeout)
        if result.prover9_proved:
            proof_output = output
    except Exception as e:
        result.prover9_error = f"prover9 goal: {e}"

    # ---- Step 3: Prover9 — prove the negation (False?) ----
    proof_neg_output = None
    try:
        p9_input_neg = build_prover9_input(premise_p9, conclusion_p9, timeout, negate_goal=True)
        result.prover9_proved_neg, output_neg = run_prover9(p9_input_neg, prover9_path, timeout)
        if result.prover9_proved_neg:
            proof_neg_output = output_neg
    except Exception as e:
        result.prover9_error = f"prover9 neg: {e}"

    # ---- Step 4: Mace4 — find countermodel (premises true, goal false) ----
    mace4_counter_output = None
    try:
        m4_input = build_mace4_input(premise_p9, conclusion_p9, timeout, negate_goal=False)
        result.mace4_countermodel, mace4_counter_output = run_mace4(m4_input, mace4_path, timeout)
    except Exception as e:
        pass

    # ---- Step 5: Mace4 — find model (premises true, goal true) ----
    mace4_model_output = None
    try:
        m4_input_pos = build_mace4_input(premise_p9, conclusion_p9, timeout, negate_goal=True)
        result.mace4_model_neg, mace4_model_output = run_mace4(m4_input_pos, mace4_path, timeout)
    except Exception as e:
        pass

    # ---- Step 6: Determine predicted label and build solution ----
    predicted, explanation = determine_label(
        result.prover9_proved or False,
        result.prover9_proved_neg or False,
        result.mace4_countermodel or False,
        result.mace4_model_neg or False,
    )
    result.predicted_label = predicted

    # Build solution
    solution_parts = [explanation]

    if predicted == "True" and proof_output:
        proof_section = _extract_proof_section(proof_output)
        if proof_section:
            solution_parts.append(f"\n=== Prover9 Proof (conclusion follows from premises) ===\n{proof_section}")

    elif predicted == "False" and proof_neg_output:
        proof_section = _extract_proof_section(proof_neg_output)
        if proof_section:
            solution_parts.append(f"\n=== Prover9 Proof (negation of conclusion follows from premises) ===\n{proof_section}")

    elif predicted == "Uncertain":
        # Show both models to explain WHY it's uncertain
        if result.mace4_model_neg and mace4_model_output and mace4_model_output != "TIMEOUT":
            model_section = _extract_model_section(mace4_model_output)
            if model_section:
                solution_parts.append(
                    f"\n=== Mace4 Model 1: premises TRUE, conclusion TRUE ===\n"
                    f"(This model shows the conclusion CAN be true)\n{model_section}"
                )

        if result.mace4_countermodel and mace4_counter_output and mace4_counter_output != "TIMEOUT":
            model_section = _extract_model_section(mace4_counter_output)
            if model_section:
                solution_parts.append(
                    f"\n=== Mace4 Model 2: premises TRUE, conclusion FALSE ===\n"
                    f"(This model shows the conclusion CAN be false)\n{model_section}"
                )

        # If no models found, explain why
        if not result.mace4_countermodel and not result.mace4_model_neg:
            solution_parts.append(
                "\nNeither Prover9 could find a proof nor Mace4 could find models "
                "within the time limit."
            )

    # For False predicted by Mace4 (not Prover9), also show the countermodel
    elif predicted == "False" and not proof_neg_output:
        if result.mace4_countermodel and mace4_counter_output and mace4_counter_output != "TIMEOUT":
            model_section = _extract_model_section(mace4_counter_output)
            if model_section:
                solution_parts.append(
                    f"\n=== Mace4 Countermodel (premises TRUE, conclusion FALSE) ===\n{model_section}"
                )

    # For True predicted by Mace4 (not Prover9), also show the model
    elif predicted == "True" and not proof_output:
        if result.mace4_model_neg and mace4_model_output and mace4_model_output != "TIMEOUT":
            model_section = _extract_model_section(mace4_model_output)
            if model_section:
                solution_parts.append(
                    f"\n=== Mace4 Model (premises TRUE, conclusion TRUE) ===\n{model_section}"
                )

    result.solution = "\n".join(solution_parts)
    return result


def main():
    parser = argparse.ArgumentParser(description="FOLIO → Prover9 Pipeline")
    parser.add_argument("--timeout", type=int, default=60, help="Per-problem timeout in seconds")
    parser.add_argument("--output", type=str, default="folio_prover9_results.jsonl", help="Output JSONL file")
    parser.add_argument("--prover9", type=str, default="prover9", help="Path to prover9 binary")
    parser.add_argument("--mace4", type=str, default="mace4", help="Path to mace4 binary")
    parser.add_argument("--dataset", type=str, default="yale-nlp/FOLIO", help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to process")
    parser.add_argument("--conversion-only", action="store_true", help="Only test conversion, don't run provers")
    args = parser.parse_args()

    # ---- Load Dataset ----
    print(f"Loading dataset: {args.dataset} (split={args.split})")
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(ds)} examples")

    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))
        print(f"Processing first {len(ds)} examples")

    # ---- Check Prover9/Mace4 availability ----
    if not args.conversion_only:
        for tool, path in [("Prover9", args.prover9), ("Mace4", args.mace4)]:
            try:
                subprocess.run([path, '--help'], capture_output=True, timeout=5)
                print(f"✓ {tool} found at: {path}")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print(f"✗ WARNING: {tool} not found at '{path}'")
                if tool == "Prover9":
                    print("  Install LADR or use --conversion-only to test conversion only.")
                    print("  Download: https://www.cs.unm.edu/~mccune/prover9/download/")
                    args.conversion_only = True
                    break

    # ---- Process Examples ----
    from tqdm import tqdm

    conversion_success = 0
    conversion_fail = 0
    conversion_errors = []
    prover_stats = Counter()
    label_counts = Counter()

    with open(args.output, 'w', buffering=1) as fout:
        for example in tqdm(ds, desc="Processing", unit="ex"):
            example_id = example.get('example_id', -1)
            label = example.get('label', 'Unknown')
            label_counts[label] += 1

            if args.conversion_only:
                try:
                    premise_fols = split_premises_fol(example.get('premises-FOL', ''))
                    premise_p9 = [fol_to_prover9(p) for p in premise_fols]
                    conclusion_p9 = fol_to_prover9(example.get('conclusion-FOL', ''))
                    conversion_success += 1

                    record = {
                        'example_id': example_id,
                        'story_id': example.get('story_id', -1),
                        'label': label,
                        'premises_p9': premise_p9,
                        'conclusion_p9': conclusion_p9,
                        'conversion_ok': True,
                    }
                except Exception as e:
                    conversion_fail += 1
                    conversion_errors.append(f"Example {example_id}: {e}")
                    record = {
                        'example_id': example_id,
                        'story_id': example.get('story_id', -1),
                        'label': label,
                        'conversion_ok': False,
                        'error': str(e),
                    }

                fout.write(json.dumps(record, default=str) + '\n')

            else:
                result = process_example(example, args.prover9, args.mace4, args.timeout)

                if result.conversion_error:
                    conversion_fail += 1
                    conversion_errors.append(f"Example {example_id}: {result.conversion_error}")
                else:
                    conversion_success += 1
                    prover_stats[(label, result.predicted_label)] += 1

                record = {
                    'example_id': result.example_id,
                    'story_id': result.story_id,
                    'label': result.label,
                    'predicted_label': result.predicted_label,
                    'solution': result.solution,
                    'premises_fol': result.premises_fol,
                    'conclusion_fol': result.conclusion_fol,
                    'premises_p9': result.premises_p9,
                    'conclusion_p9': result.conclusion_p9,
                    'prover9_proved': result.prover9_proved,
                    'prover9_proved_neg': result.prover9_proved_neg,
                    'mace4_countermodel': result.mace4_countermodel,
                    'mace4_model_neg': result.mace4_model_neg,
                    'conversion_error': result.conversion_error,
                }
                fout.write(json.dumps(record, default=str) + '\n')

    # ---- Print Summary ----
    total = len(ds)
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples:       {total}")
    print(f"Conversion success:   {conversion_success} ({100*conversion_success/total:.1f}%)")
    print(f"Conversion failures:  {conversion_fail} ({100*conversion_fail/total:.1f}%)")
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    if conversion_errors:
        print(f"\nFirst 10 conversion errors:")
        for err in conversion_errors[:10]:
            print(f"  - {err}")

    if not args.conversion_only and prover_stats:
        print(f"\nProver9/Mace4 Results (actual → predicted):")
        correct = 0
        total_with_prediction = 0
        for (actual, predicted), count in sorted(prover_stats.items()):
            match = "✓" if actual == predicted else "✗"
            print(f"  {match} {actual:>10} → {predicted:<10}: {count}")
            total_with_prediction += count
            if actual == predicted:
                correct += count

        if total_with_prediction > 0:
            print(f"\nAccuracy: {correct}/{total_with_prediction} = {100*correct/total_with_prediction:.1f}%")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()