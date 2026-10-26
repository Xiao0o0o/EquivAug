#!/bin/bash
# EquivAug data generation for FOLIO (Steps 1-3).
#
# Requires prover9 and mace4 on PATH (see scripts/install_prover9.sh) and access to
# the yale-nlp/FOLIO dataset on HuggingFace (run `huggingface-cli login`).
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash scripts/build_folio.sh [output_dir]
#
# Optional environment variables:
#   MODEL=gpt-4o-mini   LLM used in Step 3
#   BASE_URL=...        OpenAI-compatible endpoint (e.g. a local vLLM server)
#   LIMIT=...           only process the first N problems (quick test)
set -ex

OUT=${1:-outputs/folio}
MODEL=${MODEL:-gpt-4o-mini}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p $OUT

LLM_ARGS="--model $MODEL"
[ -n "$BASE_URL" ] && LLM_ARGS="$LLM_ARGS --base_url $BASE_URL"

# 1) Symbolic Representation: FOL annotations -> Prover9 syntax; verify labels with Prover9/Mace4
python $ROOT/step1_symbolic_representation/folio/fol_to_prover9.py \
    --timeout 30 --output $OUT/folio_prover9_results.jsonl ${LIMIT:+--max-examples $LIMIT}

# 2) Equivalence Transformation: premise rewrites x Prover9 strategies -> diverse verified proofs
python $ROOT/step2_equivalence_transformation/folio/folio_transform_prover9.py \
    --input $OUT/folio_prover9_results.jsonl --output $OUT/folio_diverse_proofs_p9.jsonl \
    --timeout 30 --num-strategies 5 --num-transforms 12

python $ROOT/step2_equivalence_transformation/folio/merge_to_sft.py \
    --diverse $OUT/folio_diverse_proofs_p9.jsonl --original $OUT/folio_prover9_results.jsonl \
    --output $OUT/folio_sft_data.jsonl

# 3) Natural Language Reconstruction: proof trace -> natural-language explanation
python $ROOT/step3_nl_reconstruction/folio/translate_back_folio.py translate \
    --input $OUT/folio_sft_data.jsonl --output $OUT/folio_sft_nl.jsonl $LLM_ARGS

#    Uncertain examples have no proof trace; add explanations for them
python $ROOT/step3_nl_reconstruction/folio/translate_back_folio.py balance \
    --input $OUT/folio_sft_nl.jsonl --output $OUT/folio_sft_nl_balanced.jsonl \
    --target-ratio 0.7 $LLM_ARGS

echo "EquivAug SFT data: $OUT/folio_sft_nl_balanced.jsonl (same format as data/folio/equivaug.jsonl)"
