#!/bin/bash
# EquivAug data generation for GSM8K (Steps 1-3).
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash scripts/build_gsm8k.sh [output_dir]
#
# Optional environment variables:
#   MODEL=gpt-4o-mini   LLM used in Steps 1 and 3
#   BASE_URL=...        OpenAI-compatible endpoint (e.g. a local vLLM server)
#   LIMIT=...           only process the first N problems (quick test)
set -ex

OUT=${1:-outputs/gsm8k}
MODEL=${MODEL:-gpt-4o-mini}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p $OUT

LLM_ARGS="--model $MODEL"
[ -n "$BASE_URL" ] && LLM_ARGS="$LLM_ARGS --base_url $BASE_URL"

# 0) GSM8K train set
python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main', split='train').to_json('$OUT/train.jsonl', force_ascii=False)"

# 1) Symbolic Representation: natural-language solution -> arithmetic expression (AST)
python $ROOT/step1_symbolic_representation/gsm8k/translate_ast.py \
    --input $OUT/train.jsonl --output $OUT/train_expr.jsonl $LLM_ARGS ${LIMIT:+--limit $LIMIT}

# 2) Equivalence Transformation: rewrite the AST with equivalence rules, verify each variant
python $ROOT/step2_equivalence_transformation/gsm8k/ast_transformation.py \
    --input $OUT/train_expr.jsonl --output $OUT/train_expr_variants.jsonl \
    --max_transformations 1 --max_variants 5

# 3) Natural Language Reconstruction: expression variant -> natural-language solution
python $ROOT/step3_nl_reconstruction/gsm8k/translate_back_gsm8k.py \
    --input $OUT/train_expr_variants.jsonl --output_dir $OUT \
    --per_question_max_expr 5 $LLM_ARGS

# The SFT file is $OUT/train_expr_explained_depth_1.jsonl (same format as data/gsm8k/equivaug.jsonl)
echo "EquivAug SFT data: $OUT/train_expr_explained_depth_1.jsonl"
