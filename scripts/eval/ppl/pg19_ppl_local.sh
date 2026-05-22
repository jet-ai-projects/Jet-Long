#!/bin/bash
# Local single-node launcher for the PG-19 PPL eval. Activate the python env
# first, then run.
#
# Usage:
#   MODEL_PATH=model_cache/Qwen3-1.7B-Base L_MAX=8192 NUM_BOOKS=4 \
#     bash scripts/eval/ppl/pg19_ppl_local.sh

set -e

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
# Disable the NCCL heartbeat watchdog — the eval loop is silent on collectives
# for hours, and the watchdog otherwise misreads that as a deadlock and
# SIGABRTs the run at ~98% complete.
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

MODEL_PATH=${MODEL_PATH:?must set MODEL_PATH (e.g. model_cache/Qwen3-1.7B-Base)}
VERSION_STR=${VERSION_STR:-smoke}
RUN_NAME="pg19_ppl-$(basename "$MODEL_PATH")"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

EXTRA_ARGS=""
if [[ -n "${L_MAX:-}"      ]]; then EXTRA_ARGS="$EXTRA_ARGS --l_max $L_MAX";          fi
if [[ -n "${STRIDE:-}"     ]]; then EXTRA_ARGS="$EXTRA_ARGS --stride $STRIDE";        fi
if [[ -n "${NUM_BOOKS:-}"  ]]; then EXTRA_ARGS="$EXTRA_ARGS --num_books $NUM_BOOKS";  fi
if [[ -n "${BATCH_SIZE:-}" ]]; then EXTRA_ARGS="$EXTRA_ARGS --batch_size $BATCH_SIZE"; fi

MASTER_PORT="${MASTER_PORT:-29500}"

echo "[pg19_ppl_local] MODEL_PATH=$MODEL_PATH gpu_count=$gpu_count VERSION_STR=$VERSION_STR MASTER_PORT=$MASTER_PORT"

torchrun \
  --nnodes 1 --nproc_per_node="$gpu_count" --master_port "$MASTER_PORT" \
  jetlm/evaluation/ppl/pg19_ppl.py \
    --model_name_or_path "$MODEL_PATH" \
    --eval_config        jetlm/evaluation/configs/pg19_ppl.yaml \
    --output_dir         "results/pg19_ppl" \
    --run_name           "$RUN_NAME" \
    --version_str        "$VERSION_STR" \
    $EXTRA_ARGS
