#!/usr/bin/env bash
# Evaluate one merged checkpoint, then print its table.
#
#   CHECKPOINTS_DIR=/path/to/checkpoints MODEL=WhereOPD-Qwen3.5-4B STEP=global_step_31 \
#   GPUS=0,1 bash scripts/launch_eval_suite.sh

# The model directory is $CHECKPOINTS_DIR/$MODEL/$STEP (a merged HF checkpoint, see
# scripts/merge_checkpoint.sh); the judge model is $JUDGE_MODEL_PATH (default
# $CHECKPOINTS_DIR/Qwen3.5-4B). A rerun resumes: finished answers and judgements are kept.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${MODEL:?set MODEL=<experiment name under CHECKPOINTS_DIR>}"
: "${CHECKPOINTS_DIR:?set CHECKPOINTS_DIR}"
export CHECKPOINTS_DIR STEP="${STEP:-global_step_31}" MODELS="${MODEL}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0}"
PORT0="${PORT0:-12300}"
LOGS="${ROOT}/eval/results/logs"; mkdir -p "${LOGS}"

MME=mme-realworld
REST=cvbench,blink,gqa,countqa,chartqa,docvqa,ocrbench,evochart,hallusionbench,amber,vstar,hrbench-4k,hrbench-8k,zoombench

if (( ${#GPU_LIST[@]} >= 2 )); then
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" VLLM_PORT=$PORT0 JUDGE_PORT=$((PORT0 + 1)) BENCHMARKS=$MME \
    bash "${ROOT}/scripts/eval_whereopd.sh" > "${LOGS}/${MODEL}_mme.log" 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" VLLM_PORT=$((PORT0 + 2)) JUDGE_PORT=$((PORT0 + 3)) BENCHMARKS=$REST \
    bash "${ROOT}/scripts/eval_whereopd.sh" > "${LOGS}/${MODEL}_rest.log" 2>&1 &
  p2=$!
  echo "running: MME on GPU ${GPU_LIST[0]} (log ${LOGS}/${MODEL}_mme.log), 14 others on GPU ${GPU_LIST[1]} (log ${LOGS}/${MODEL}_rest.log)"
  rc=0; wait $p1 || rc=$?; wait $p2 || rc=$?
else
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" VLLM_PORT=$PORT0 JUDGE_PORT=$((PORT0 + 1)) \
    bash "${ROOT}/scripts/eval_whereopd.sh" 2>&1 | tee "${LOGS}/${MODEL}_suite.log"
  rc=${PIPESTATUS[0]}
fi

echo
python3 "${ROOT}/eval/summarize.py" "${MODEL}"
exit $rc
