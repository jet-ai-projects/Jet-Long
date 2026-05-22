#!/bin/bash
# Launch HELMET evaluation for a jetlm model, sharding tasks across 8 GPUs.
#
# Usage:
#   bash scripts/eval/helmet/run_helmet.sh <MODEL_NAME> [CATEGORIES] [LENGTHS]
#
# Examples:
#   bash scripts/eval/helmet/run_helmet.sh Qwen3-1.7B-Base-jetlong
#   bash scripts/eval/helmet/run_helmet.sh Qwen3-1.7B-Base-jetlong "rag longqa" "8192 131072"
#
# Environment overrides:
#   HELMET_DIR=tmp/helmet/repo          where eval.py lives
#   MODEL_CACHE_ROOT=model_cache        where the jetlm variants live
#   GPUS=0,1,2,3,4,5,6,7                gpu list
#   MAX_TEST_SAMPLES=100                samples per (dataset,length)
#   TAG=v1                              output filename tag
#
# Skipped by default (datasets 4.x compat break): ICL, Summ (Multi-LexSum).
# See scripts/eval/helmet/README.md for details.
set -euo pipefail

MODEL_NAME="${1:?model name required, e.g. Qwen3-1.7B-Base-jetlong}"
# Categories to run; defaults to RAG only. Pass a space-separated list as the
# 2nd arg to add others (subject to the datasets>=4.0 compat skips above).
CATEGORIES="${2:-rag}"
LENGTHS="${3:-8192 16384 32768 65536 131072}"

HELMET_DIR="${HELMET_DIR:-tmp/helmet/repo}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-model_cache}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-100}"
TAG="${TAG:-v1}"

# Locate the project root (dir containing this script's grandparent).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="${MODEL_CACHE_ROOT}/${MODEL_NAME}"
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: model dir not found: ${MODEL_PATH}"; exit 1; }
[[ -f "${HELMET_DIR}/eval.py" ]] || { echo "ERROR: eval.py not under ${HELMET_DIR}"; exit 1; }

OUTPUT_DIR="results/helmet/${MODEL_NAME}"
LOG_DIR_ROOT="logs/helmet"
MODEL_LOG_DIR="${LOG_DIR_ROOT}/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}" "${MODEL_LOG_DIR}"

RUN_LOG="${MODEL_LOG_DIR}/run.log"
exec > >(tee "${RUN_LOG}") 2>&1

echo "[helmet] model         = ${MODEL_NAME}  (${MODEL_PATH})"
echo "[helmet] categories    = ${CATEGORIES}"
echo "[helmet] lengths       = ${LENGTHS}"
echo "[helmet] gpus          = ${GPUS}"
echo "[helmet] output_dir    = ${OUTPUT_DIR}"
echo "[helmet] log_dir       = ${MODEL_LOG_DIR}"
echo "[helmet] tag           = ${TAG}"
echo "[helmet] samples/task  = ${MAX_TEST_SAMPLES}"
echo "[helmet] python        = $(which python)"
echo "[helmet] torch/transformers: $(python -c 'import torch,transformers;print(torch.__version__,transformers.__version__)')"

START=$SECONDS

python jetlm/evaluation/helmet/helmet_shard.py \
    --helmet_dir "${HELMET_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    --configs ${CATEGORIES} \
    --lengths ${LENGTHS} \
    --gpus "${GPUS}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${MODEL_LOG_DIR}" \
    --tag "${TAG}" \
    --max_test_samples "${MAX_TEST_SAMPLES}"
shard_rc=$?

# Always aggregate whatever tasks completed, even if some failed. This writes
# result.json alongside the per-task JSON/score files.
echo "[helmet] aggregating results → ${OUTPUT_DIR}/result.json"
python jetlm/evaluation/helmet/aggregate_results.py \
    --output_dir "${OUTPUT_DIR}" \
    --tag "${TAG}" || echo "[helmet] aggregation failed (non-fatal)"

ELAPSED=$(( SECONDS - START ))
echo "[helmet] elapsed: $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s"
exit $shard_rc
