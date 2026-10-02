#!/usr/bin/env bash
# Evaluate merged WhereOPD checkpoints).
#
#   PHASE 1  inference: serve the model with vLLM, answer every benchmark, stop the server
#   PHASE 2  judge:     serve the Qwen3.5-4B judge, grade every answer file, print accuracy
#
# Grading is: mathruler, then an MCQ letter match, then the
# Qwen3.5-4B LLM judge (eval/judge_qwenlm.py). Decoding is greedy,
# thinking off (the checkpoints carry the training chat template, which closes
# <think></think>), images at native resolution.
#
# Usage:
#   CHECKPOINTS_DIR=/path/to/checkpoints \
#   MODELS=WhereOPD-Qwen3.5-4B STEP=global_step_31 \
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_whereopd.sh
#
# BENCHMARKS defaults to the whole 15-benchmark suite (eval/benchmarks.py); pass a
# comma-separated subset to run part of it. 
#
# A model is read from $CHECKPOINTS_DIR/<name>/$STEP (merged HF dir). The judge
# model is $JUDGE_MODEL_PATH (default $CHECKPOINTS_DIR/Qwen3.5-4B). Reruns skip
# answer files that are complete and judge outputs that exist, so a killed job
# resumes where it stopped.
set -euo pipefail

VENV="${VENV:-./envs/where-opd}"
export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_DIR="${PROJECT_ROOT}/eval"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${PROJECT_ROOT}/checkpoints}"
STEP="${STEP:-global_step_31}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${CHECKPOINTS_DIR}/Qwen3.5-4B}"
LOG_DIR="${EVAL_DIR}/results/logs"

VLLM_PORT="${VLLM_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
TP="${TP:-1}"
GPU_MEM="${GPU_MEM:-0.85}"
JUDGE_GPU_MEM="${JUDGE_GPU_MEM:-0.4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-64}"
SERVER_WAIT_TRIES="${SERVER_WAIT_TRIES:-240}"   # x5 s = 20 min
PARALLEL_WORKERS="${PARALLEL_WORKERS:-256}"
SEED=42
PHASE="${PHASE:-all}"                            # all | infer | judge

MODEL_TAG="whereopd-checkpoint"
JUDGE_MODEL_TAG="qwen-judge"
API_BASE="http://localhost:${VLLM_PORT}/v1/"
JUDGE_API_BASE="http://localhost:${JUDGE_PORT}/v1/"

IFS=',' read -r -a MODEL_NAMES <<< "${MODELS:?set MODELS=name1,name2}"
SUITE_ALL=cvbench,mme-realworld,blink,gqa,countqa,chartqa,docvqa,ocrbench,evochart,hallusionbench,amber,vstar,hrbench-4k,hrbench-8k,zoombench
IFS=',' read -r -a BENCHMARKS <<< "${BENCHMARKS:-${SUITE_ALL}}"
declare -A BENCH_JSON=(
  [cvbench]="cvbench.json"
  [mme-realworld]="MME_RealWorld.json"
  [blink]="blink_full.json"
  [gqa]="gqa.json"
  [countqa]="countqa.json"
  [chartqa]="chartqa.json"
  [docvqa]="docvqa.json"
  [ocrbench]="ocrbench.json"
  [evochart]="evochart.json"
  [hallusionbench]="hallusionbench.json"
  [amber]="amber.json"
  [vstar]="vstar.json"
  [hrbench-4k]="hr_bench_4k.json"
  [hrbench-8k]="hr_bench_8k.json"
  [zoombench]="zoombench.json"
)
for b in "${BENCHMARKS[@]}"; do
  [[ -n "${BENCH_JSON[$b]+x}" ]] || { echo "unknown benchmark: $b (known: ${!BENCH_JSON[*]})"; exit 1; }
done
mkdir -p "${LOG_DIR}"

start_vllm() {   # model_path served_name port log gpu_mem -> prints pid
  # setsid: own process group, so the whole server tree can be killed without
  # touching another eval running on the same machine
  PYTHONUNBUFFERED=1 setsid python -m vllm.entrypoints.openai.api_server \
    --model "$1" --served-model-name "$2" --port "$3" \
    --tensor-parallel-size "${TP}" --gpu-memory-utilization "$5" \
    --max-model-len "${MAX_MODEL_LEN}" --trust-remote-code \
    > "$4" 2>&1 &
  echo $!
}

wait_for_server() {   # port label log
  local port="$1" label="$2" log_file="$3" tries=0
  local fatal='Engine core initialization failed|is less than desired GPU memory utilization|Error in memory profiling|CUDA out of memory|Address already in use|torch.OutOfMemoryError'
  echo -n "  [${label}] waiting"
  until python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${port}/health', timeout=3)" 2>/dev/null; do
    if [[ -s "${log_file}" ]] && grep -q -E "${fatal}" "${log_file}"; then
      echo; echo "  ERROR: ${label} server died during startup:"; grep -m1 -E "${fatal}" "${log_file}" | cut -c1-200
      return 1
    fi
    sleep 5; tries=$((tries + 1)); echo -n "."
    if (( tries > SERVER_WAIT_TRIES )); then echo; echo "  ERROR: ${label} not up after $((SERVER_WAIT_TRIES * 5 / 60)) min (see ${log_file})"; return 1; fi
  done
  echo " ready."
}

stop_server() {   # pid: kill its process group (API server + EngineCore)
  local pid="$1"
  [[ -z "${pid}" ]] && return 0
  kill -TERM -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  sleep 5
  kill -9 -- "-${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

# =========================================================================== inference
if [[ "${PHASE}" == "all" || "${PHASE}" == "infer" ]]; then
for model_name in "${MODEL_NAMES[@]}"; do
  step_dir="${CHECKPOINTS_DIR}/${model_name}/${STEP}"
  model_tag="${model_name}_seed${SEED}"
  [[ -f "${step_dir}/config.json" ]] || { echo "not a merged model: ${step_dir}"; exit 1; }
  echo; echo "--- ${model_name} (${step_dir}) ---"
  EVAL_LOG="${LOG_DIR}/${model_name}_vllm_p${VLLM_PORT}.log"
  EVAL_PID=$(start_vllm "${step_dir}" "${MODEL_TAG}" "${VLLM_PORT}" "${EVAL_LOG}" "${GPU_MEM}")
  trap "stop_server ${EVAL_PID}" EXIT INT TERM
  wait_for_server "${VLLM_PORT}" "eval" "${EVAL_LOG}"

  for bench in "${BENCHMARKS[@]}"; do
    bench_json="${EVAL_DIR}/${BENCH_JSON[$bench]}"
    answer_file="${EVAL_DIR}/model_answer/${bench}/${model_tag}_answer.jsonl"
    # skip only when every item has a real answer; a partial file, or one with
    # error rows ([API_ERROR]/[FUTURE_ERROR]), resumes and infer.py retries just those
    if [[ -f "${answer_file}" ]]; then
      n_have=$(grep -vc '"model_answer": "\[\(API\|FUTURE\)_ERROR\]' "${answer_file}" || true)
      n_need=$(python3 -c "import json;print(len(json.load(open('${bench_json}'))))")
      if (( n_have >= n_need )); then echo "  [skip] ${bench}: ${n_have}/${n_need} answered"; continue; fi
      echo "  [resume] ${bench}: ${n_have}/${n_need} answered, retrying the rest"
    fi
    echo "  [infer] ${bench} ..."
    set +e
    ( cd "${EVAL_DIR}"
      python3 prepare_data.py --benchmark "${bench}" --data_dir "${EVAL_DIR}"
      python3 infer.py --benchmark "${bench}" --benchmark_json "${bench_json}" \
        --out_dir model_answer --model_name "${model_tag}" --seed "${SEED}" \
        --api_base "${API_BASE}" --api_key EMPTY --model_id "${MODEL_TAG}" \
        --max_tokens "${MAX_TOKENS}" --max_retries 3 --parallel_workers "${PARALLEL_WORKERS}" \
        --temperature 0 )
    rc=$?
    set -e
    if (( rc != 0 )); then
      # infer.py exits non-zero when the server stops answering: stop this model
      # rather than write error stubs for every remaining benchmark
      if ! python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${VLLM_PORT}/health', timeout=5)" 2>/dev/null; then
        echo "  FATAL: eval server is down (infer rc=${rc}); resubmit to resume."; exit 3
      fi
      echo "  WARNING: infer.py rc=${rc} on ${bench}, server is up; continuing."
    fi
  done
  trap - EXIT INT TERM
  stop_server "${EVAL_PID}"
done
fi

# =========================================================================== judge
if [[ "${PHASE}" == "all" || "${PHASE}" == "judge" ]]; then
JUDGE_LOG="${LOG_DIR}/judge_vllm_p${JUDGE_PORT}.log"
JUDGE_PID=$(start_vllm "${JUDGE_MODEL_PATH}" "${JUDGE_MODEL_TAG}" "${JUDGE_PORT}" "${JUDGE_LOG}" "${JUDGE_GPU_MEM}")
trap "stop_server ${JUDGE_PID}" EXIT INT TERM
wait_for_server "${JUDGE_PORT}" "judge" "${JUDGE_LOG}"

for model_name in "${MODEL_NAMES[@]}"; do
  model_tag="${model_name}_seed${SEED}"
  echo; echo "--- ${model_name} ---"
  for bench in "${BENCHMARKS[@]}"; do
    answers="${EVAL_DIR}/model_answer/${bench}/${model_tag}_answer.jsonl"
    judged="judge/${bench}/${model_tag}_answer.jsonl"
    [[ -s "${answers}" ]] || { echo "  [skip] ${bench}: no answer file"; continue; }
    # (re)judge when there is no judge file, the answers changed since it was
    # written, or it holds failed items (api_error / judge_error)
    if [[ ! -s "${EVAL_DIR}/${judged}" || "${answers}" -nt "${EVAL_DIR}/${judged}" ]] \
        || grep -q '"judge_source": "\(api\|judge\)_error"' "${EVAL_DIR}/${judged}"; then
      echo "  [judge] ${bench} ..."
      ( cd "${EVAL_DIR}" && python3 judge_qwenlm.py --benchmark "${bench}" --model "${model_tag}" \
          --api_base "${JUDGE_API_BASE}" --judge_model "${JUDGE_MODEL_TAG}" \
          --judge_max_tokens "${JUDGE_MAX_TOKENS}" )
    fi
    if [[ -s "${EVAL_DIR}/${judged}" ]]; then
      ( cd "${EVAL_DIR}" && python3 cal_acc.py --benchmark "${bench}" --judge_json "${judged}" \
          --benchmark_json "${EVAL_DIR}/${BENCH_JSON[$bench]}" )
    else
      echo "  [acc] ${bench}: judge wrote no output"
    fi
  done
done
trap - EXIT INT TERM
stop_server "${JUDGE_PID}"
fi
