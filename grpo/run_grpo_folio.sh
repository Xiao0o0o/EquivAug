#!/bin/bash
# GRPO on FOLIO with verl, starting from an SFT checkpoint.
#
# Usage:
#   bash grpo/run_grpo_folio.sh /path/to/sft_model [experiment_name] [extra verl options ...]
#
# The defaults are the settings used for the Qwen2.5 models. Settings used for Llama-3.1-8B
# (2 GPUs; start from the merged LoRA model, see sft/merge_lora.py):
#   N_GPUS=2 RAY_NUM_CPUS=16 KL_COEF=0 TOTAL_EPOCHS=12 MICRO_BS=4 OPTIMIZER_OFFLOAD=True \
#   bash grpo/run_grpo_folio.sh /path/to/llama_sft_model-merged llama3.1-8b \
#       data.train_batch_size=64 data.val_batch_size=32 data.max_prompt_length=1024 \
#       actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.rollout.max_model_len=2048 \
#       +actor_rollout_ref.model.override_config.max_position_embeddings=2048
set -x

MODEL_PATH=${1:?"usage: bash grpo/run_grpo_folio.sh /path/to/sft_model [experiment_name] [extra verl options ...]"}
EXP_NAME=${2:-folio_grpo}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR=${DATA_DIR:-$SCRIPT_DIR/data/folio}
REWARD_FILE=$SCRIPT_DIR/reward_folio.py

N_GPUS=${N_GPUS:-1}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
KL_COEF=${KL_COEF:-0.001}
MICRO_BS=${MICRO_BS:-8}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-False}
LOGGER=${LOGGER:-'["console","wandb"]'}

# 1) Parquet data (see data_preprocess_folio.py)
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    python $SCRIPT_DIR/data_preprocess_folio.py --local_dir $DATA_DIR
fi

export VLLM_USE_V1=1

# 2) GRPO
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS \
    data.dataloader_num_workers=4 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=204 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=compute_score \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BS \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BS \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BS \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.project_name=grpo_folio \
    trainer.experiment_name=$EXP_NAME \
    trainer.logger="$LOGGER" \
    "${@:3}"
