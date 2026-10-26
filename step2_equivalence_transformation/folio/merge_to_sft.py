#!/usr/bin/env python3
"""
Step 2 (FOLIO): Merge diverse proofs into one record per proof path
====================================================================
Combines three data sources into a single SFT training file:

1. folio_diverse_proofs_p9.jsonl  — multiple proof paths per example (True/False), from folio_transform_prover9.py
2. folio_prover9_results.jsonl     — original results including Uncertain examples, from fol_to_prover9.py
3. FOLIO HuggingFace dataset       — natural language premises and conclusions

Output format (one line per proof path / one line per Uncertain):
{
    "example_id": 21,
    "story_id": 7,
    "label": "True",

    # Natural language
    "premises_nl": "Miroslav is a Czech ...",
    "conclusion_nl": "A Czech published a book in 1Mo946.",

    # FOL (original FOLIO notation)
    "premises_fol": "∀x (Czech(x) → ...)",
    "conclusion_fol": "∃x ∃y (Czech(x) ∧ ...)",

    # Prover9 syntax
    "premises_p9": ["all x (czech(x) -> ...)", ...],
    "conclusion_p9": "exists x exists y (czech(x) & ...)",

    # Proof / solution
    "solution": "1 czech(miroslav). [assumption]. ...",
    "solution_source": "diverse_proof",       # or "original"
    "strategy": "hyper_res",                  # only for diverse proofs
    "transform": "contrapositive",            # only for diverse proofs
    "transform_desc": "Contrapositive of ...",
    "proof_index": 2,                         # which proof variant (0-based)
    "total_proofs": 5,                        # total variants for this example
}

Usage:
    python merge_to_sft.py \
        --diverse   /path/to/folio_diverse_proofs_p9.jsonl \
        --original  /path/to/folio_prover9_results.jsonl \
        --dataset   yale-nlp/FOLIO \
        --split     train \
        --output    folio_sft_data.jsonl

    # If HuggingFace is not accessible, provide a local FOLIO JSONL/CSV:
    python merge_to_sft.py \
        --diverse   /path/to/folio_diverse_proofs_p9.jsonl \
        --original  /path/to/folio_prover9_results.jsonl \
        --folio-local /path/to/folio_train.jsonl \
        --output    folio_sft_data.jsonl
"""

import json
import argparse
import os
from collections import defaultdict


def load_folio_nl(dataset_name: str, split: str, local_path: str = None) -> dict:
    """
    Load natural language premises/conclusions from FOLIO.
    Returns: {example_id: {"premises_nl": ..., "conclusion_nl": ..., "story_id": ...}}

    Tries three sources in order:
    1. Local JSONL file (--folio-local)
    2. HuggingFace datasets library
    3. Returns empty dict with warning
    """
    nl_data = {}

    # Source 1: Local file
    if local_path and os.path.exists(local_path):
        print(f"Loading NL from local file: {local_path}")
        with open(local_path) as f:
            # Support both JSONL and JSON array
            content = f.read().strip()
            if content.startswith('['):
                records = json.loads(content)
            else:
                records = [json.loads(line) for line in content.split('\n') if line.strip()]

        for r in records:
            eid = r.get('example_id')
            if eid is None:
                continue
            nl_data[eid] = {
                "premises_nl": r.get('premises', r.get('premises_nl', r.get('premises-NL', ''))),
                "conclusion_nl": r.get('conclusion', r.get('conclusion_nl', r.get('conclusion-NL', ''))),
                "story_id": r.get('story_id'),
            }
        print(f"  Loaded {len(nl_data)} examples from local file")
        return nl_data

    # Source 2: HuggingFace
    try:
        from datasets import load_dataset
        print(f"Loading NL from HuggingFace: {dataset_name} (split={split})")
        ds = load_dataset(dataset_name, split=split)
        for ex in ds:
            eid = ex.get('example_id')
            if eid is None:
                continue
            nl_data[eid] = {
                "premises_nl": ex.get('premises', ex.get('premises-NL', '')),
                "conclusion_nl": ex.get('conclusion', ex.get('conclusion-NL', '')),
                "story_id": ex.get('story_id'),
            }
        print(f"  Loaded {len(nl_data)} examples from HuggingFace")
        return nl_data
    except Exception as e:
        print(f"  WARNING: Could not load from HuggingFace: {e}")

    # Source 3: Empty
    print("  WARNING: No NL data source available. NL fields will be empty.")
    print("  Use --folio-local to provide a local FOLIO file with 'premises' and 'conclusion' fields.")
    return nl_data


def load_diverse_proofs(path: str) -> dict:
    """
    Load diverse proofs file.
    Returns: {example_id: record}
    """
    data = {}
    if not path or not os.path.exists(path):
        print(f"  Diverse proofs file not found: {path}")
        return data

    print(f"Loading diverse proofs: {path}")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            eid = r.get('example_id')
            data[eid] = r

    total_proofs = sum(r.get('num_proofs', 0) for r in data.values())
    print(f"  Loaded {len(data)} examples, {total_proofs} total proofs")
    return data


def load_original_results(path: str) -> dict:
    """
    Load original Prover9 results.
    Returns: {example_id: record}
    """
    data = {}
    print(f"Loading original results: {path}")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            eid = r.get('example_id')
            data[eid] = r

    print(f"  Loaded {len(data)} examples")
    return data


def build_sft_record(
    example_id: int,
    story_id: int,
    label: str,
    premises_nl: str,
    conclusion_nl: str,
    premises_fol: str,
    conclusion_fol: str,
    premises_p9: list,
    conclusion_p9: str,
    solution: str,
    solution_source: str,
    strategy: str = None,
    transform: str = None,
    transform_desc: str = None,
    proof_index: int = 0,
    total_proofs: int = 1,
) -> dict:
    """Build a single SFT training record."""
    record = {
        "example_id": example_id,
        "story_id": story_id,
        "label": label,
        # Natural language
        "premises_nl": premises_nl,
        "conclusion_nl": conclusion_nl,
        # FOL
        "premises_fol": premises_fol,
        "conclusion_fol": conclusion_fol,
        # Prover9
        "premises_p9": premises_p9,
        "conclusion_p9": conclusion_p9,
        # Solution
        "solution": solution,
        "solution_source": solution_source,
        "proof_index": proof_index,
        "total_proofs": total_proofs,
    }

    # Only add strategy/transform fields for diverse proofs
    if strategy is not None:
        record["strategy"] = strategy
    if transform is not None:
        record["transform"] = transform
    if transform_desc is not None:
        record["transform_desc"] = transform_desc

    return record


def main():
    parser = argparse.ArgumentParser(description="Merge FOLIO proofs into SFT-ready JSONL")
    parser.add_argument("--diverse", type=str, required=True,
                        help="Diverse proofs JSONL (from folio_transform_prover9.py)")
    parser.add_argument("--original", type=str, required=True,
                        help="Original Prover9 results JSONL (from fol_to_prover9.py)")
    parser.add_argument("--dataset", type=str, default="yale-nlp/FOLIO",
                        help="HuggingFace dataset name for NL premises/conclusions")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    parser.add_argument("--folio-local", type=str, default=None,
                        help="Local FOLIO file with NL fields (alternative to HuggingFace)")
    parser.add_argument("--output", type=str, default="folio_sft_data.jsonl",
                        help="Output SFT JSONL file")
    args = parser.parse_args()

    # ── Step 1: Load all data sources ────────────────────────────────────
    nl_data = load_folio_nl(args.dataset, args.split, args.folio_local)
    diverse_data = load_diverse_proofs(args.diverse)
    original_data = load_original_results(args.original)

    # ── Step 2: Merge ────────────────────────────────────────────────────
    print(f"\nMerging...")
    sft_records = []

    # Track stats
    from_diverse = 0
    from_original_true_false = 0
    from_uncertain = 0
    no_solution = 0

    # Collect all example_ids (union of both files)
    all_eids = set(original_data.keys())
    if diverse_data:
        all_eids |= set(diverse_data.keys())

    for eid in sorted(all_eids):
        orig = original_data.get(eid, {})
        diverse = diverse_data.get(eid)
        nl = nl_data.get(eid, {})

        # Common fields
        story_id = orig.get('story_id', diverse.get('story_id', -1) if diverse else -1)
        label = orig.get('label', diverse.get('label', '') if diverse else '')
        premises_fol = orig.get('premises_fol', diverse.get('premises_fol', '') if diverse else '')
        conclusion_fol = orig.get('conclusion_fol', diverse.get('conclusion_fol', '') if diverse else '')
        premises_p9 = orig.get('premises_p9', diverse.get('premises_p9', []) if diverse else [])
        conclusion_p9 = orig.get('conclusion_p9', diverse.get('conclusion_p9', '') if diverse else '')
        premises_nl = nl.get('premises_nl', '')
        conclusion_nl = nl.get('conclusion_nl', '')

        predicted = orig.get('predicted_label', '')

        # ── Case 1: Has diverse proofs → one row per proof ──────────
        if diverse and diverse.get('num_proofs', 0) > 0:
            proofs = diverse['proofs']
            total = len(proofs)
            for idx, proof in enumerate(proofs):
                rec = build_sft_record(
                    example_id=eid,
                    story_id=story_id,
                    label=label,
                    premises_nl=premises_nl,
                    conclusion_nl=conclusion_nl,
                    premises_fol=premises_fol,
                    conclusion_fol=conclusion_fol,
                    premises_p9=premises_p9,
                    conclusion_p9=conclusion_p9,
                    solution=proof['proof'],
                    solution_source="diverse_proof",
                    strategy=proof.get('strategy'),
                    transform=proof.get('transform'),
                    transform_desc=proof.get('transform_desc'),
                    proof_index=idx,
                    total_proofs=total,
                )
                sft_records.append(rec)
                from_diverse += 1
            continue

        # ── Case 2: label==predicted, True/False, but no diverse proofs
        #            (diverse file missing or 0 proofs) → use original solution
        if label == predicted and label in ('True', 'False'):
            solution = orig.get('solution', '')
            if solution:
                rec = build_sft_record(
                    example_id=eid,
                    story_id=story_id,
                    label=label,
                    premises_nl=premises_nl,
                    conclusion_nl=conclusion_nl,
                    premises_fol=premises_fol,
                    conclusion_fol=conclusion_fol,
                    premises_p9=premises_p9,
                    conclusion_p9=conclusion_p9,
                    solution=solution,
                    solution_source="original",
                )
                sft_records.append(rec)
                from_original_true_false += 1
            else:
                no_solution += 1
            continue

        # ── Case 3: Uncertain (label==predicted==Uncertain) → one row ──
        if label == 'Uncertain' and predicted == 'Uncertain':
            solution = orig.get('solution', '')
            if solution:
                rec = build_sft_record(
                    example_id=eid,
                    story_id=story_id,
                    label=label,
                    premises_nl=premises_nl,
                    conclusion_nl=conclusion_nl,
                    premises_fol=premises_fol,
                    conclusion_fol=conclusion_fol,
                    premises_p9=premises_p9,
                    conclusion_p9=conclusion_p9,
                    solution=solution,
                    solution_source="original_uncertain",
                )
                sft_records.append(rec)
                from_uncertain += 1
            else:
                no_solution += 1
            continue

        # ── Case 4: label != predicted → skip (wrong prediction) ────
        # These are excluded from SFT training

    # ── Step 3: Write output ─────────────────────────────────────────────
    print(f"\nWriting {len(sft_records)} records to: {args.output}")
    with open(args.output, 'w') as fout:
        for rec in sft_records:
            fout.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')

    # ── Step 4: Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SFT DATA SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total SFT records:         {len(sft_records)}")
    print(f"  From diverse proofs:     {from_diverse}")
    print(f"  From original (T/F):     {from_original_true_false}")
    print(f"  From Uncertain:          {from_uncertain}")
    print(f"  Skipped (no solution):   {no_solution}")
    print(f"  Skipped (label!=pred):   {len(all_eids) - from_diverse - from_original_true_false - from_uncertain - no_solution}")
    print()

    # Label distribution in output
    from collections import Counter
    label_dist = Counter(r['label'] for r in sft_records)
    source_dist = Counter(r['solution_source'] for r in sft_records)
    print(f"Label distribution:")
    for l, c in sorted(label_dist.items()):
        print(f"  {l:>12}: {c}")
    print(f"\nSource distribution:")
    for s, c in sorted(source_dist.items()):
        print(f"  {s:>25}: {c}")

    # Unique examples
    unique_eids = set(r['example_id'] for r in sft_records)
    print(f"\nUnique examples:           {len(unique_eids)}")
    print(f"Avg records per example:   {len(sft_records) / max(len(unique_eids), 1):.1f}")

    # NL coverage
    has_nl = sum(1 for r in sft_records if r['premises_nl'])
    print(f"Records with NL premises:  {has_nl} ({100 * has_nl / max(len(sft_records), 1):.1f}%)")

    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()