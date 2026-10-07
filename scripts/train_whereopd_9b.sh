#!/bin/bash
# WhereOPD training for Qwen3.5-9B: scripts/train_whereopd.sh plus the 9B memory
# settings. On 4x H100 the 9B student, its frozen 9B teacher and vLLM only fit with
# actor offload, a small vLLM share, shorter prompts and 2-way sequence parallelism;
# batch 48 (62 steps on 3k rows) is the 9B recipe.
#
#   DATA_DIR=.../whereopd_count_open_3k MODEL_PATH=Qwen/Qwen3.5-9B \
#   EXPERIMENT_NAME=WhereOPD-Qwen3.5-9B bash scripts/train_whereopd_9b.sh

set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.32}"
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-1}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"

exec bash "$(dirname "$0")/train_whereopd.sh" \
    data.train_batch_size=48 \
    actor_rollout_ref.actor.ppo_mini_batch_size=48 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    ++ray_kwargs.ray_init.object_store_memory=8589934592 \
    trainer.rollout_data_dir=null \
    trainer.save_freq=2 \
    trainer.max_actor_ckpt_to_keep=2 \
    "$@"
