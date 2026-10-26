#!/bin/bash
# Supervised fine-tuning.
#
# Usage:
#   bash sft/run_sft.sh <gsm8k|folio> <base_model> <data_path> <output_dir>
#
# Qwen2.5-1.5B / Qwen2.5-3B: full fine-tuning (trained_sft_<task>.py)
# Llama-3.1-8B:              LoRA (trained_sft_<task>_lora.py); the adapter is then merged
#                            into <output_dir>-merged, which is used for evaluation and GRPO.
set -ex

TASK=${1:?"usage: bash sft/run_sft.sh <gsm8k|folio> <base_model> <data_path> <output_dir>"}
BASE_MODEL=${2:?}
DATA_PATH=${3:?}
OUTPUT_DIR=${4:?}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "$BASE_MODEL" == *[Ll]lama* ]]; then
    SCRIPT=$SCRIPT_DIR/trained_sft_${TASK}_lora.py
    LR=${LR:-2e-4}
    PER_DEVICE_BS=${PER_DEVICE_BS:-4}
    GRAD_ACCUM=${GRAD_ACCUM:-4}
else
    SCRIPT=$SCRIPT_DIR/trained_sft_${TASK}.py
    LR=${LR:-2e-5}
    PER_DEVICE_BS=${PER_DEVICE_BS:-8}
    GRAD_ACCUM=${GRAD_ACCUM:-4}
fi

python $SCRIPT \
    --model_name_or_path $BASE_MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 20 \
    --eval_strategy steps \
    --eval_steps 50 \
    --validation_split 0.1 \
    --report_to tensorboard \
    --bf16 \
    --group_by_length \
    --gradient_checkpointing \
    "${@:5}"

if [[ "$BASE_MODEL" == *[Ll]lama* ]]; then
    python $SCRIPT_DIR/merge_lora.py --adapter_path $OUTPUT_DIR --output_path ${OUTPUT_DIR}-merged
fi
