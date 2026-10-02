#!/bin/bash
set -euo pipefail

# =============================================================================
# WhereOPD training: on-policy self-distillation with a frozen teacher that sees
# the student's prompt plus the scene's object-location hint (where_hint).
# The recipe (loss, teacher prompt mode, frozen teacher) lives in
# verl/trainer/config/whereopd.yaml; this script sets data, model and batch.
#
#   DATA_DIR=.../whereopd_count_mcq_3k MODEL_PATH=Qwen/Qwen3.5-4B \
#   EXPERIMENT_NAME=WhereOPD-Qwen3.5-4B bash scripts/train_whereopd.sh
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_NAME="whereopd"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.0}"   # 0.0 = frozen teacher (the WhereOPD recipe)

TRAIN_BATCH_SIZE=96
PPO_MIMI_BATCH_SIZE=96
ROLLOUT_N=8
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
LR=2e-6
DONT_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
ACTOR_USE_DYNAMIC_BSZ=True
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
REF_PARAM_OFFLOAD=False
TRAINER_N_GPUS_PER_NODE=4
TRAINER_NNODES="${SLURM_NNODES:-1}"
TRAINER_SAVE_FREQ=-1
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=null
TRAINER_LOGGER='["console","tensorboard"]'
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-8}"
CUSTOM_CHAT_TEMPLATE_FILE="${CUSTOM_CHAT_TEMPLATE_FILE-${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen35.jinja}"

# --- Data Paths ---
DATA_DIR="${DATA_DIR:?set DATA_DIR to a folder holding train.parquet (see whereopd_data/build_dataset.py)}"
TASK_TRAIN_FILE="${DATA_DIR}/train.parquet"

MODEL_NAME=$(basename "$MODEL_PATH")
EXPERIMENT_NAME="${EXPERIMENT_NAME:-WhereOPD-${MODEL_NAME}}"
PROJECT_NAME="WhereOPD"
# Checkpoints go to $CHECKPOINT_ROOT/checkpoints/<experiment> (default: the repo).
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT}"
TRAINER_DEFAULT_LOCAL_DIR="${CHECKPOINT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
TRAINER_ROLLOUT_DATA_DIR="${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME}"
mkdir -p "$TRAINER_ROLLOUT_DATA_DIR"

EXTRA_ARGS=("$@")

# =============================================================================
# ENVIRONMENT
# =============================================================================
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
ulimit -c 0

CHAT_TEMPLATE_ARGS=()
if [[ -n "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
    if [[ ! -f "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
        echo "Custom chat template file not found: ${CUSTOM_CHAT_TEMPLATE_FILE}" >&2
        exit 1
    fi
    CHAT_TEMPLATE_ARGS+=(actor_rollout_ref.model.custom_chat_template_file="$CUSTOM_CHAT_TEMPLATE_FILE")
fi

echo "Running: $EXPERIMENT_NAME"
echo "Teacher update rate: $TEACHER_UPDATE_RATE"

python3 -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    +ray_kwargs.ray_init.object_store_memory=42949672960 \
    data.train_files="[\"$TASK_TRAIN_FILE\"]" \
    data.val_files="[]" \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation=error \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key=images \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.dataloader_num_workers=$DATA_DATALOADER_NUM_WORKERS \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MIMI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_AGENT_NUM_WORKERS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    reward_model.enable=False \
    critic.model.path=$MODEL_PATH \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    trainer.project_name=$PROJECT_NAME \
    trainer.group_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS_PER_NODE \
    trainer.nnodes=$TRAINER_NNODES \
    trainer.save_freq=$TRAINER_SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=$TRAINER_MAX_ACTOR_CKPT_TO_KEEP \
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
