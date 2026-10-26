#!/usr/bin/env python3
"""
Step 2 (FOLIO): Equivalence Transformation with Prover9
=======================================================
Given the Prover9 results of Step 1 (fol_to_prover9.py), generate multiple
different proof paths for each successfully-proved example by combining:

1. Different Prover9 search strategies (hyper-resolution, UR-resolution,
   pick_given_ratio, ordering, ...).
2. Equivalence-preserving premise transformations (premise reordering,
   double negation, contraposition, De Morgan's laws, implication /
   biconditional expansion, distributivity, absorption, ...; see Table 2
   of the paper). Each transformed premise set is re-verified by Prover9.

Output: JSONL where each example has a list of proof paths ("proofs").

Usage:
    python folio_transform_prover9.py \
        --input folio_prover9_results.jsonl \
        --output folio_diverse_proofs_p9.jsonl \
        --prover9 prover9 --timeout 30 \
        --num-strategies 5 --num-transforms 12
"""



import re
import json
import subprocess
import tempfile
import os
import argparse
import random
from collections import Counter
from typing import Optional


# ============================================================================
# Part 1: Prover9 Strategy Configurations
# ============================================================================
# Prover9 strategy knobs used here:
#   - pick_given_ratio: balance between oldest-first vs lightest-first
#   - inference rules: hyper_resolution, ur_resolution, paramodulation
#   - demodulation, back_demod, lex_dep_demod
#   - ordering: lpo, rpo, kbo
#   - literal selection: max_weight, pick_given_ratio
#   - forward/back subsumption controls

PROVER9_STRATEGIES = [
    {
        "name": "default",
        "desc": "Prover9 default settings",
        "flags": [],
    },
    {
        "name": "hyper_res",
        "desc": "Hyperresolution enabled",
        "flags": [
            "set(hyper_resolution).",
        ],
    },
    {
        "name": "ur_res",
        "desc": "Unit-resulting resolution enabled",
        "flags": [
            "set(ur_resolution).",
        ],
    },
    {
        "name": "hyper_ur",
        "desc": "Both hyper and UR resolution",
        "flags": [
            "set(hyper_resolution).",
            "set(ur_resolution).",
        ],
    },
    {
        "name": "ratio_2",
        "desc": "pick_given_ratio = 2 (favor lighter clauses)",
        "flags": [
            "assign(pick_given_ratio, 2).",
        ],
    },
    {
        "name": "ratio_5",
        "desc": "pick_given_ratio = 5",
        "flags": [
            "assign(pick_given_ratio, 5).",
        ],
    },
    {
        "name": "ratio_10",
        "desc": "pick_given_ratio = 10 (strongly favor lighter)",
        "flags": [
            "assign(pick_given_ratio, 10).",
        ],
    },
    {
        "name": "no_back_demod",
        "desc": "Disable back demodulation",
        "flags": [
            "clear(back_demod).",
        ],
    },
    {
        "name": "hyper_ratio3",
        "desc": "Hyperresolution + pick_given_ratio 3",
        "flags": [
            "set(hyper_resolution).",
            "assign(pick_given_ratio, 3).",
        ],
    },
    {
        "name": "ur_ratio8",
        "desc": "UR resolution + pick_given_ratio 8",
        "flags": [
            "set(ur_resolution).",
            "assign(pick_given_ratio, 8).",
        ],
    },
    {
        "name": "age_weight_light",
        "desc": "Strongly favor lighter clauses with ratio 20",
        "flags": [
            "assign(pick_given_ratio, 20).",
        ],
    },
    {
        "name": "hyper_ur_ratio5",
        "desc": "Hyper + UR + ratio 5",
        "flags": [
            "set(hyper_resolution).",
            "set(ur_resolution).",
            "assign(pick_given_ratio, 5).",
        ],
    },
]

# Cache for strategies that Prover9 rejects (unknown flags etc.)
_bad_strategies: set[str] = set()


# ============================================================================
# Part 2: Prover9 Formula Parser Utilities
# ============================================================================
# Prover9 syntax:
#   Negation:      -
#   And:           &
#   Or:            |
#   Implication:   ->
#   Biconditional: <->
#   Universal:     all x (...)
#   Existential:   exists x (...)
#   Variables:     lowercase identifiers (x, y, z, ...)
#   Constants/Predicates: lowercase identifiers + parens

def _find_toplevel_connective(formula: str, connective: str) -> int:
    """
    Find the position of a top-level connective in formula (depth == 0).
    For '->', skips '<->'.
    Returns index or -1 if not found.
    """
    depth = 0
    clen = len(connective)
    i = 0
    while i < len(formula):
        ch = formula[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and formula[i:i + clen] == connective:
            # For '->', make sure it's not part of '<->'
            if connective == '->':
                if i > 0 and formula[i - 1] == '<':
                    i += clen
                    continue
            return i
        i += 1
    return -1


def _split_at_toplevel(formula: str, connective: str) -> Optional[tuple[str, str]]:
    """
    Split formula at the first top-level occurrence of connective.
    Returns (left, right) or None.
    """
    idx = _find_toplevel_connective(formula, connective)
    if idx == -1:
        return None
    left = formula[:idx].strip()
    right = formula[idx + len(connective):].strip()
    return left, right


def _strip_outer_parens(s: str) -> str:
    """Remove one layer of matching outer parentheses if they wrap the entire formula."""
    s = s.strip()
    if not s.startswith('(') or not s.endswith(')'):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth == 0 and i < len(s) - 1:
            return s  # opening paren closes before the end
    return s[1:-1].strip()


def _strip_quantifier_prefix(formula: str) -> tuple[str, str]:
    """
    Separate leading Prover9 quantifiers from the body.
    Returns (quantifier_prefix, body).
    E.g. "all x all y (body)" -> ("all x all y ", "(body)")
    """
    s = formula.strip()
    prefix = ""
    while True:
        m = re.match(r'^((?:all|exists)\s+\w+\s+)', s)
        if m:
            prefix += m.group(1)
            s = s[m.end():]
        else:
            break
    return prefix, s


def _negate(formula: str) -> str:
    """Negate a Prover9 formula, avoiding double negation where possible."""
    f = formula.strip()
    # If already negated: -(X) -> X
    if f.startswith('-(') and f.endswith(')'):
        inner = f[2:-1].strip()
        # verify balanced
        depth = 0
        balanced = True
        for ch in inner:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth < 0:
                balanced = False
                break
        if balanced and depth == 0:
            return inner
    # Simple negated atom: -p -> p
    if f.startswith('-') and not f.startswith('-('):
        rest = f[1:].strip()
        if '(' not in rest or rest.index('(') > 0:
            # It's a simple negated atom like -p or -pred(x)
            return rest
    return f"-({f})"


# ============================================================================
# Part 3: Equivalent Premise Transformations (Prover9 Syntax)
# ============================================================================

def apply_transform(
    premises_p9: list[str],
    transform_name: str,
    rng: random.Random,
) -> tuple[list[str], str]:
    """
    Apply a semantics-preserving transformation to the premises (Prover9 syntax).
    Returns: (transformed_premises, description)
    """
    if transform_name == "original":
        return premises_p9, "Original premises (no transformation)"

    elif transform_name == "shuffle":
        shuffled = premises_p9.copy()
        rng.shuffle(shuffled)
        return shuffled, "Shuffled premise order"

    elif transform_name == "reverse":
        return list(reversed(premises_p9)), "Reversed premise order"

    elif transform_name == "rotate":
        if len(premises_p9) <= 1:
            return premises_p9, "Rotated premises (no change, <=1 premise)"
        k = rng.randint(1, len(premises_p9) - 1)
        rotated = premises_p9[k:] + premises_p9[:k]
        return rotated, f"Rotated premises by {k} positions"

    elif transform_name == "pairwise_swap":
        result = premises_p9.copy()
        for i in range(0, len(result) - 1, 2):
            result[i], result[i + 1] = result[i + 1], result[i]
        return result, "Pairwise swapped adjacent premises"

    elif transform_name == "double_neg":
        result = []
        for p in premises_p9:
            if rng.random() < 0.5:
                result.append(f"-(-(({p})))")
            else:
                result.append(p)
        return result, "Double negation on some premises"

    elif transform_name == "contrapositive":
        result = [_apply_contrapositive(p) for p in premises_p9]
        return result, "Contrapositive of implications"

    elif transform_name == "demorgan":
        result = [_apply_demorgan(p) for p in premises_p9]
        return result, "De Morgan's law applied"

    elif transform_name == "implication_expand":
        result = [_expand_implication(p) for p in premises_p9]
        return result, "Implications expanded to disjunctions"

    elif transform_name == "biconditional_expand":
        result = [_expand_biconditional(p) for p in premises_p9]
        return result, "Biconditionals expanded to conjunctions of implications"

    elif transform_name == "biconditional_to_conj":
        result = [_expand_biconditional_conj(p) for p in premises_p9]
        return result, "Biconditionals expanded to (A&B)|(~A&~B)"

    elif transform_name == "commute_and_or":
        result = [_commute_connectives(p) for p in premises_p9]
        return result, "Commuted conjunctions and disjunctions"

    elif transform_name == "distributive_and_over_or":
        result = [_distribute_and_over_or(p) for p in premises_p9]
        return result, "Distributed & over |"

    elif transform_name == "distributive_or_over_and":
        result = [_distribute_or_over_and(p) for p in premises_p9]
        return result, "Distributed | over &"

    elif transform_name == "export_antecedent":
        result = [_export_antecedent(p) for p in premises_p9]
        return result, "Exported conjunction in antecedent: (A&B)->C to A->(B->C)"

    elif transform_name == "idempotent_cleanup":
        result = [_idempotent_cleanup(p) for p in premises_p9]
        return result, "Idempotent law applied (A&A->A, A|A->A)"

    elif transform_name == "absorption":
        result = [_absorption(p) for p in premises_p9]
        return result, "Absorption law applied"

    elif transform_name == "material_equivalence":
        result = [_material_equivalence(p) for p in premises_p9]
        return result, "Material equivalence: A->B to -(A) | B"

    elif transform_name == "shuffle_and_contrapositive":
        shuffled = premises_p9.copy()
        rng.shuffle(shuffled)
        result = [_apply_contrapositive(p) for p in shuffled]
        return result, "Shuffled + contrapositive"

    elif transform_name == "shuffle_and_expand":
        shuffled = premises_p9.copy()
        rng.shuffle(shuffled)
        result = [_expand_implication(p) for p in shuffled]
        return result, "Shuffled + implication expansion"

    elif transform_name == "shuffle_and_demorgan":
        shuffled = premises_p9.copy()
        rng.shuffle(shuffled)
        result = [_apply_demorgan(p) for p in shuffled]
        return result, "Shuffled + De Morgan's law"

    elif transform_name == "contrapositive_and_expand":
        result = [_expand_implication(_apply_contrapositive(p)) for p in premises_p9]
        return result, "Contrapositive then implication expansion"

    elif transform_name == "double_neg_and_demorgan":
        result = []
        for p in premises_p9:
            transformed = _apply_demorgan(p)
            if rng.random() < 0.4:
                transformed = f"-(-(({transformed})))"
            result.append(transformed)
        return result, "De Morgan + selective double negation"

    else:
        return premises_p9, f"Unknown transform: {transform_name}"


# ---------------------------------------------------------------------------
# Individual transformation implementations (Prover9 syntax)
# ---------------------------------------------------------------------------

def _apply_contrapositive(formula: str) -> str:
    """Convert top-level A -> B to -(B) -> -(A) in Prover9 syntax."""
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    split = _split_at_toplevel(inner, '->')
    if split is None:
        return formula

    antecedent, consequent = split
    neg_con = _negate(consequent)
    neg_ant = _negate(antecedent)
    return f"{prefix}({neg_con} -> {neg_ant})"


def _apply_demorgan(formula: str) -> str:
    """
    Apply De Morgan's laws:
      -(A & B) -> (-(A) | -(B))
      -(A | B) -> (-(A) & -(B))
    """
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    # Check if it's -(...)
    if not inner.startswith('-'):
        return formula

    rest = inner[1:].strip()
    if not rest.startswith('('):
        return formula

    negated_body = _strip_outer_parens(rest)

    # Try splitting on top-level &
    split_and = _split_at_toplevel(negated_body, '&')
    if split_and is not None:
        a, b = split_and
        return f"{prefix}({_negate(a)} | {_negate(b)})"

    # Try splitting on top-level |
    split_or = _split_at_toplevel(negated_body, '|')
    if split_or is not None:
        a, b = split_or
        return f"{prefix}({_negate(a)} & {_negate(b)})"

    return formula


def _expand_implication(formula: str) -> str:
    """Expand top-level A -> B to (-(A) | B)."""
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    split = _split_at_toplevel(inner, '->')
    if split is None:
        return formula

    antecedent, consequent = split
    return f"{prefix}({_negate(antecedent)} | ({consequent}))"


def _expand_biconditional(formula: str) -> str:
    """Expand top-level A <-> B to ((A -> B) & (B -> A))."""
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    split = _split_at_toplevel(inner, '<->')
    if split is None:
        return formula

    a, b = split
    return f"{prefix}((({a}) -> ({b})) & (({b}) -> ({a})))"


def _expand_biconditional_conj(formula: str) -> str:
    """
    Alternative biconditional expansion:
    A <-> B  ≡  (A & B) | (-(A) & -(B))
    """
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    split = _split_at_toplevel(inner, '<->')
    if split is None:
        return formula

    a, b = split
    neg_a = _negate(a)
    neg_b = _negate(b)
    return f"{prefix}((({a}) & ({b})) | (({neg_a}) & ({neg_b})))"


def _commute_connectives(formula: str) -> str:
    """Swap operands of top-level & or |."""
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    for conn in ('&', '|'):
        split = _split_at_toplevel(inner, conn)
        if split is not None:
            a, b = split
            return f"{prefix}({b} {conn} {a})"

    return formula


def _distribute_and_over_or(formula: str) -> str:
    """
    Distributive law: A & (B | C) -> (A & B) | (A & C)
    Only applies if top-level is & and one side is a top-level |.
    """
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    split_and = _split_at_toplevel(inner, '&')
    if split_and is None:
        return formula

    a, bc = split_and

    # Check if right side is a disjunction
    bc_inner = _strip_outer_parens(bc)
    split_or = _split_at_toplevel(bc_inner, '|')
    if split_or is not None:
        b, c = split_or
        return f"{prefix}((({a}) & ({b})) | (({a}) & ({c})))"

    # Try left side: (B | C) & A -> (B & A) | (C & A)
    a_inner = _strip_outer_parens(a)
    split_or_left = _split_at_toplevel(a_inner, '|')
    if split_or_left is not None:
        b, c = split_or_left
        return f"{prefix}((({b}) & ({bc})) | (({c}) & ({bc})))"

    return formula


def _distribute_or_over_and(formula: str) -> str:
    """
    Distributive law: A | (B & C) -> (A | B) & (A | C)
    Only applies if top-level is | and one side is a top-level &.
    """
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    split_or = _split_at_toplevel(inner, '|')
    if split_or is None:
        return formula

    a, bc = split_or

    bc_inner = _strip_outer_parens(bc)
    split_and = _split_at_toplevel(bc_inner, '&')
    if split_and is not None:
        b, c = split_and
        return f"{prefix}((({a}) | ({b})) & (({a}) | ({c})))"

    a_inner = _strip_outer_parens(a)
    split_and_left = _split_at_toplevel(a_inner, '&')
    if split_and_left is not None:
        b, c = split_and_left
        return f"{prefix}((({b}) | ({bc})) & (({c}) | ({bc})))"

    return formula


def _export_antecedent(formula: str) -> str:
    """
    Exportation: (A & B) -> C  ≡  A -> (B -> C)
    Only applies when top-level is -> and antecedent has top-level &.
    """
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    split_imp = _split_at_toplevel(inner, '->')
    if split_imp is None:
        return formula

    antecedent, consequent = split_imp
    ant_inner = _strip_outer_parens(antecedent)

    split_and = _split_at_toplevel(ant_inner, '&')
    if split_and is None:
        return formula

    a, b = split_and
    return f"{prefix}(({a}) -> (({b}) -> ({consequent})))"


def _idempotent_cleanup(formula: str) -> str:
    """
    Idempotent law: A & A -> A,  A | A -> A
    Checks if both sides of top-level & or | are identical.
    """
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    for conn in ('&', '|'):
        split = _split_at_toplevel(inner, conn)
        if split is not None:
            a, b = split
            if a.strip() == b.strip():
                return f"{prefix}{a.strip()}"

    return formula


def _absorption(formula: str) -> str:
    """
    Absorption: A & (A | B) -> A,  A | (A & B) -> A
    Checks if one side of & is also a disjunct/conjunct in the other side.
    """
    prefix, body = _strip_quantifier_prefix(formula)
    inner = _strip_outer_parens(body)

    # A & (A | B) -> A
    split_and = _split_at_toplevel(inner, '&')
    if split_and is not None:
        a, rhs = split_and
        rhs_inner = _strip_outer_parens(rhs)
        split_or = _split_at_toplevel(rhs_inner, '|')
        if split_or is not None:
            b, c = split_or
            if a.strip() == b.strip() or a.strip() == c.strip():
                return f"{prefix}{a.strip()}"

    # A | (A & B) -> A
    split_or = _split_at_toplevel(inner, '|')
    if split_or is not None:
        a, rhs = split_or
        rhs_inner = _strip_outer_parens(rhs)
        split_and2 = _split_at_toplevel(rhs_inner, '&')
        if split_and2 is not None:
            b, c = split_and2
            if a.strip() == b.strip() or a.strip() == c.strip():
                return f"{prefix}{a.strip()}"

    return formula


def _material_equivalence(formula: str) -> str:
    """
    Material equivalence / conditional: A -> B  ≡  -(A) | B
    Same as implication expansion but named differently as a standalone transform.
    This is a duplicate of _expand_implication but kept for explicit naming.
    """
    return _expand_implication(formula)


# ---------------------------------------------------------------------------
# Transform categories and selection
# ---------------------------------------------------------------------------

STRUCTURAL_TRANSFORMS = ["shuffle", "reverse", "rotate", "pairwise_swap"]
LOGICAL_TRANSFORMS = [
    "contrapositive",
    "demorgan",
    "implication_expand",
    "biconditional_expand",
    "biconditional_to_conj",
    "commute_and_or",
    "distributive_and_over_or",
    "distributive_or_over_and",
    "export_antecedent",
    "idempotent_cleanup",
    "absorption",
    "material_equivalence",
]
COMBINED_TRANSFORMS = [
    "shuffle_and_contrapositive",
    "shuffle_and_expand",
    "shuffle_and_demorgan",
    "contrapositive_and_expand",
    "double_neg_and_demorgan",
]
MISC_TRANSFORMS = ["double_neg"]

ALL_TRANSFORMS = (
    ["original"]
    + STRUCTURAL_TRANSFORMS
    + LOGICAL_TRANSFORMS
    + COMBINED_TRANSFORMS
    + MISC_TRANSFORMS
)


def select_diverse_transforms(
    num_transforms: int,
    rng: random.Random,
) -> list[str]:
    """
    Select transforms ensuring category diversity:
    - Always include "original"
    - Sample from each category
    - Fill remaining slots randomly
    """
    selected = ["original"]
    num_transforms = min(num_transforms, len(ALL_TRANSFORMS))

    if num_transforms <= 1:
        return selected

    remaining = num_transforms - 1

    categories = [
        STRUCTURAL_TRANSFORMS,
        LOGICAL_TRANSFORMS,
        COMBINED_TRANSFORMS,
        MISC_TRANSFORMS,
    ]
    for cat in categories:
        if remaining <= 0:
            break
        pick = rng.choice(cat)
        if pick not in selected:
            selected.append(pick)
            remaining -= 1

    pool = [t for t in ALL_TRANSFORMS if t not in selected]
    if remaining > 0 and pool:
        extras = rng.sample(pool, min(remaining, len(pool)))
        selected.extend(extras)

    return selected


# ============================================================================
# Part 4: Running Prover9
# ============================================================================

def build_prover9_input(
    premises: list[str],
    goal: str,
    timeout: int = 60,
    negate_goal: bool = False,
    strategy_flags: list[str] = None,
) -> str:
    """Build a complete Prover9 input file from premises and goal."""
    lines = []
    lines.append(f"assign(max_seconds, {timeout}).")
    lines.append("")

    # Strategy flags
    if strategy_flags:
        for flag in strategy_flags:
            lines.append(flag)
        lines.append("")

    lines.append("formulas(assumptions).")
    for p in premises:
        clean = p.strip().rstrip('.')
        lines.append(f"  {clean}.")
    lines.append("end_of_list.")
    lines.append("")

    goal_clean = goal.strip().rstrip('.')
    if negate_goal:
        goal_clean = f"-({goal_clean})"
    lines.append("formulas(goals).")
    lines.append(f"  {goal_clean}.")
    lines.append("end_of_list.")

    return "\n".join(lines)


def run_prover9_single(
    prover9_input: str,
    prover9_path: str,
    timeout: int,
) -> tuple[bool, Optional[str]]:
    """
    Run Prover9 once. Returns: (proved: bool, proof_steps: str or None)
    Detects unknown flags/errors to avoid wasting time on bad strategy configs.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
        f.write(prover9_input)
        input_file = f.name

    try:
        result = subprocess.run(
            [prover9_path, '-f', input_file],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
        output = result.stdout + result.stderr

        proved = ("THEOREM PROVED" in output) or ("Exiting with 1 proof" in output)
        if proved:
            proof = _extract_proof_section(output)
            return True, proof

        if re.search(r'(?i)\bunknown\b|\bunrecognized\b|\bfatal\b|\bsyntax error\b', output):
            return False, None
        # Detect unknown flags or fatal errors from Prover9
        # if re.search(
        #     r'(?i)\bunknown\b|\bunrecognized\b|\bfatal\b'
        #     r'|\berror\s*:|\bsyntax error\b',
        #     output
        # ):
        #     return False, None

        proved = "THEOREM PROVED" in output or "Exiting with 1 proof" in output

        if proved:
            proof = _extract_proof_section(output)
            return True, proof
        return False, None

    except subprocess.TimeoutExpired:
        return False, None
    except Exception:
        return False, None
    finally:
        os.unlink(input_file)


def _extract_proof_section(prover9_output: str) -> Optional[str]:
    """
    Extract numbered proof steps from Prover9 output.
    Robust to varying amounts of '=' in separator lines across Prover9 versions.
    """
    lines = prover9_output.splitlines()
    in_proof = False
    proof_lines = []
    for line in lines:
        stripped = line.strip()
        # Enter proof section: any separator line containing "PROOF"
        if not in_proof and 'PROOF' in stripped and '=' in stripped:
            in_proof = True
            continue
        # Exit proof section: any line containing "end of proof" (case-insensitive)
        if in_proof and 'end of proof' in stripped.lower():
            break
        if in_proof:
            # Proof steps start with a digit; skip blank lines and comments
            if stripped and re.match(r'^\d+\s', stripped):
                proof_lines.append(stripped)

    # Fallback: if no PROOF section found, try to grab numbered inference lines
    if not proof_lines:
        for line in lines:
            stripped = line.strip()
            if re.match(r'^\d+\s.*\[.*\]\.\s*$', stripped):
                proof_lines.append(stripped)

    return '\n'.join(proof_lines) if proof_lines else None


# ============================================================================
# Part 5: Main Pipeline
# ============================================================================

def validate_strategy(
    prover9_path: str,
    strategy: dict,
    timeout: int = 5,
) -> bool:
    """
    Test if a strategy's flags are accepted by the installed Prover9.
    Uses a trivial provable problem. Caches failures in _bad_strategies.
    """
    if strategy["name"] in _bad_strategies:
        return False
    if not strategy["flags"]:
        return True  # default strategy always works

    test_input = build_prover9_input(
        premises=["p"],
        goal="p",
        timeout=timeout,
        strategy_flags=strategy["flags"],
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
        f.write(test_input)
        input_file = f.name

    try:
        result = subprocess.run(
            [prover9_path, '-f', input_file],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        output = result.stdout + result.stderr
        if re.search(
            r'(?i)\bunknown\b|\bunrecognized\b|\bfatal\b'
            r'|\berror\s*:|\bsyntax error\b',
            output
        ):
            _bad_strategies.add(strategy["name"])
            return False
        return True
    except Exception:
        _bad_strategies.add(strategy["name"])
        return False
    finally:
        os.unlink(input_file)


def generate_diverse_proofs(
    example: dict,
    prover9_path: str,
    timeout: int,
    num_strategies: int = 5,
    num_transforms: int = 5,
    seed: int = 42,
    validated_strategies: list[dict] = None,
) -> dict:
    """
    Generate diverse proofs for a single example.

    1. Pick N different Prover9 strategies (pre-validated)
    2. Pick M different premise transformations
    3. Run all N×M combinations
    4. Collect unique proofs (deduplicated)
    """
    rng = random.Random(seed + example.get('example_id', 0))

    premises_p9 = example['premises_p9']
    conclusion_p9 = example['conclusion_p9']
    label = example['label']
    negate_goal = (label == 'False')

    # Use pre-validated strategies if provided, else sample and filter
    if validated_strategies is not None:
        pool = validated_strategies
    else:
        pool = [s for s in PROVER9_STRATEGIES if s["name"] not in _bad_strategies]

    n_strat = min(num_strategies, len(pool))
    strategies = rng.sample(pool, n_strat) if n_strat > 0 else []

    # Select transforms
    transforms = select_diverse_transforms(num_transforms, rng)

    # Run all combinations
    proofs = []
    seen_proofs = set()

    for strategy in strategies:
        for transform_name in transforms:
            transformed, desc = apply_transform(premises_p9, transform_name, rng)

            p9_input = build_prover9_input(
                transformed,
                conclusion_p9,
                timeout,
                negate_goal,
                strategy["flags"],
            )

            proved, proof_steps = run_prover9_single(
                p9_input,
                prover9_path,
                timeout,
            )

            if proved and proof_steps:
                normalized = re.sub(r'\s+', ' ', proof_steps.strip())
                if normalized not in seen_proofs:
                    seen_proofs.add(normalized)
                    proofs.append({
                        "strategy": strategy["name"],
                        "strategy_desc": strategy["desc"],
                        "transform": transform_name,
                        "transform_desc": desc,
                        "proof": proof_steps,
                    })

    return {
        'example_id': example.get('example_id', -1),
        'story_id': example.get('story_id', -1),
        'label': label,
        'predicted_label': example.get('predicted_label'),
        'premises_fol': example.get('premises_fol', ''),
        'conclusion_fol': example.get('conclusion_fol', ''),
        'premises_p9': premises_p9,
        'conclusion_p9': conclusion_p9,
        'num_proofs': len(proofs),
        'proofs': proofs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Diverse Proofs for FOLIO (Prover9 Edition)"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL from fol_to_prover9.py")
    parser.add_argument("--output", type=str, default="folio_diverse_proofs_p9.jsonl",
                        help="Output JSONL")
    parser.add_argument("--prover9", type=str, default="prover9",
                        help="Path to prover9 binary")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Per-attempt timeout (seconds)")
    parser.add_argument("--num-strategies", type=int, default=5,
                        help=f"Number of Prover9 strategies to try (max {len(PROVER9_STRATEGIES)})") #12
    parser.add_argument("--num-transforms", type=int, default=5,
                        help=f"Number of premise transforms to try (max {len(ALL_TRANSFORMS)})") #23
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Max examples to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Load input
    print(f"Loading results from: {args.input}")
    examples = []
    skipped_mismatch = 0
    skipped_other = 0
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Only use examples where label == predicted_label
            label = record.get('label')
            predicted = record.get('predicted_label')
            if label != predicted:
                skipped_mismatch += 1
                continue

            # Must be proved (True or False), with valid P9 formulas
            if (predicted not in ('True', 'False')
                    or record.get('conversion_error') is not None
                    or not record.get('premises_p9')
                    or not record.get('conclusion_p9')):
                skipped_other += 1
                continue

            examples.append(record)

    print(f"Found {len(examples)} correctly-proved examples (label == predicted)")
    print(f"Skipped {skipped_mismatch} label!=predicted, {skipped_other} other")

    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"Processing first {len(examples)} examples")

    n_trans = min(args.num_transforms, len(ALL_TRANSFORMS))
    print(f"\nAvailable transforms ({len(ALL_TRANSFORMS)}):")
    for t in ALL_TRANSFORMS:
        print(f"  - {t}")

    # Validate strategies against the installed Prover9
    print(f"\nValidating {len(PROVER9_STRATEGIES)} strategy configs against Prover9...")
    validated_strategies = []
    for strat in PROVER9_STRATEGIES:
        ok = validate_strategy(args.prover9, strat, timeout=3)
        status = "✓" if ok else "✗ SKIPPED"
        print(f"  {status} {strat['name']}: {strat['desc']}")
        if ok:
            validated_strategies.append(strat)

    if not validated_strategies:
        print("\nERROR: No valid strategies found. Check your Prover9 installation.")
        return

    n_strat = min(args.num_strategies, len(validated_strategies))
    combos = n_strat * n_trans
    print(f"\nEach example: up to {n_strat} strategies × {n_trans} transforms = {combos} attempts")
    print(f"Timeout per attempt: {args.timeout}s")
    print(f"Estimated max time: {len(examples) * combos * args.timeout / 3600:.1f} hours "
          f"(actual will be much less)")

    # Process
    from tqdm import tqdm

    total_proofs = 0
    examples_with_multiple = 0

    with open(args.output, 'w', buffering=1) as fout:
        for example in tqdm(examples, desc="Generating proofs", unit="ex"):
            result = generate_diverse_proofs(
                example,
                args.prover9,
                args.timeout,
                num_strategies=args.num_strategies,
                num_transforms=args.num_transforms,
                seed=args.seed,
                validated_strategies=validated_strategies,
            )

            total_proofs += result['num_proofs']
            if result['num_proofs'] > 1:
                examples_with_multiple += 1

            fout.write(json.dumps(result, default=str) + '\n')

    # Summary
    n = len(examples) if examples else 1
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Examples processed:        {len(examples)}")
    print(f"Total unique proofs:       {total_proofs}")
    print(f"Avg proofs per example:    {total_proofs / n:.1f}")
    print(f"Examples with 2+ proofs:   {examples_with_multiple} "
          f"({100 * examples_with_multiple / n:.1f}%)")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()