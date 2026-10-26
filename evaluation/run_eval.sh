#!/bin/bash
# Sample 5 responses per test question with vLLM and report the mean accuracy.
#
# Usage:
#   bash evaluation/run_eval.sh <gsm8k|folio> /path/to/model <output_dir>
set -x

TASK=${1:?"usage: bash evaluation/run_eval.sh <gsm8k|folio> /path/to/model <output_dir>"}
MODEL_PATH=${2:?}
OUTPUT_DIR=${3:?}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_RESPONSES=${NUM_RESPONSES:-5}
TEMPERATURE=${TEMPERATURE:-0.6}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}

if [ "$TASK" == "gsm8k" ]; then
    SPLIT=test
else
    SPLIT=validation
fi

python $SCRIPT_DIR/$TASK/generate_vllm.py \
    --model-path $MODEL_PATH \
    --dataset-split $SPLIT \
    --num-responses $NUM_RESPONSES \
    --temperature $TEMPERATURE \
    --max-new-tokens $MAX_NEW_TOKENS \
    --batch-size 16 \
    --output-dir $OUTPUT_DIR

python $SCRIPT_DIR/$TASK/compute_accuracy.py \
    --results-path $OUTPUT_DIR/${TASK}_${SPLIT}_${NUM_RESPONSES}responses_${MAX_NEW_TOKENS}tokens.pkl
