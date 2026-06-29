#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Single-node bash launcher for RULER with the fused CuTe kernel (jetlong_fused).
# Run from the FUSED conda env (flash_attn_4 / flash_attn.cute), e.g. `jtl-fused`:
#     conda activate jtl-fused
#     bash scripts/eval/ruler/qwen1d7B_base_jetlong_fused.sh
#
# The model is loaded with attn_implementation="sdpa" automatically (load_model.py
# detects the fused architecture). The fused kernel runs once the context passes the
# 32K window and rejects left-padded batches. Generation crosses that window during
# decode starting at length 32768, so eval_batch_size=8 makes every length >= 32768
# (32768, 65536, 131072) run at batch 1 (auto_batch_size = bs*4096//length); shorter
# lengths stay batched on the SDPA path. --prefill_chunk_size 16384 keeps the 131072
# prefill within a single H100's memory.

set -e

export OMP_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True  # torch>=2.9 renamed the var; the old one is ignored
export TOKENIZERS_PARALLELISM=false
export LM_HARNESS_CACHE_PATH=".cache/lm_harness_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1

export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="jet-long"
export WANDB_RESUME="allow"
export WANDB_RUNTYPE="eval_ruler"

MODEL_CACHE_ROOT="model_cache"
MODEL_NAME="Qwen3-1.7B-Base-jetlong_fused"
MODEL_PATH="${MODEL_CACHE_ROOT}/${MODEL_NAME}"
LOG_DIR_ROOT="logs/evaluation"
RUN_NAME="${MODEL_NAME}"

mkdir -p .cache/lm_harness_cache
mkdir -p "$LOG_DIR_ROOT"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
echo "Load model from ${MODEL_PATH}"
echo "Available GPU count: $gpu_count"

exec > >(tee "${LOG_DIR_ROOT}/${RUN_NAME}.log") 2>&1
START_TIME=$SECONDS
echo "Start at $(date '+%F %T')"

torchrun --nnodes 1 --nproc_per_node="$gpu_count" \
    --rdzv_id "$RANDOM" --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    jetlm/evaluation/meta_eval.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "results/${MODEL_NAME}/baseline" \
    --version_str "$RUN_NAME" \
    --run_name "$RUN_NAME" \
    --eval_config jetlm/evaluation/configs/ruler.yaml \
    --eval_batch_size 8 \
    --prefill_chunk_size 16384

ELAPSED=$((SECONDS - START_TIME))
echo "Ruler evaluation took: $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s."
