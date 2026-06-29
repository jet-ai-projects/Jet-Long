#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage: bash scripts/eval/ruler/vram_check.sh <MODEL_NAME> <EVAL_BS>
# Activate the python env first; this mimics a cluster eval (8 ranks).
set -e

MODEL_NAME="${1:?MODEL_NAME required}"
EVAL_BS="${2:?EVAL_BS required}"

export OMP_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

torchrun --nnodes 1 --nproc_per_node="${gpu_count}" \
    --rdzv_id "$RANDOM" --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    scripts/eval/ruler/vram_check.py \
    --model_name_or_path "model_cache/${MODEL_NAME}" \
    --eval_batch_size "${EVAL_BS}"
