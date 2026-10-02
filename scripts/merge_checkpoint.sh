#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_BASE_DIR="${PROJECT_ROOT}/checkpoints/WhereOPD-Qwen3.5-4B/global_step_31/"
BASE_DIR="${BASE_DIR:-${1:-${DEFAULT_BASE_DIR}}}"
BASE_DIR="${BASE_DIR%/}"
ACTOR_DIR="${BASE_DIR}/actor"

if [ ! -d "${ACTOR_DIR}" ]; then
  echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
  exit 1
fi

echo "Merging ${ACTOR_DIR} -> ${BASE_DIR}"


find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type f -print -delete

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${BASE_DIR}"


TRAINED_TEMPLATE="${ACTOR_DIR}/huggingface/chat_template.jinja"
BASE_MODEL_TEMPLATE="${PROJECT_ROOT}/checkpoints/Qwen3.5-4B/chat_template.jinja"
if [ ! -f "${BASE_DIR}/chat_template.jinja" ]; then
  if [ -f "${TRAINED_TEMPLATE}" ]; then
    echo "chat_template.jinja missing in merge output; copying the checkpoint's own training-time template."
    cp "${TRAINED_TEMPLATE}" "${BASE_DIR}/"
  elif [ -f "${BASE_MODEL_TEMPLATE}" ]; then
    echo "WARNING: no training-time template found; copying STOCK base template (thinking ON by default!)."
    cp "${BASE_MODEL_TEMPLATE}" "${BASE_DIR}/"
  fi
fi

echo "Merge completed."
