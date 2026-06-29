#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup model variants by:
#   1. Downloading base models from HuggingFace
#   2. Materializing each variant under model_cache/<variant>/ as a directory of
#      symlinks pointing at:
#        - config.json from model_configs/<variant>/
#        - modeling_qwen3_<method>.py from jetlm/modeling/<method>/ (if any)
#        - shared weights/tokenizer from the base model
#
# Usage: bash scripts/model/setup_model_variants.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_CACHE="$REPO_ROOT/model_cache"
MODEL_CONFIGS="$REPO_ROOT/model_configs"
JETLM_MODELING="$REPO_ROOT/jetlm/modeling"

# --- Base models to download ---
declare -A BASE_MODELS=(
    ["Qwen3-1.7B-Base"]="Qwen/Qwen3-1.7B-Base"
    ["Qwen3-4B-Base"]="Qwen/Qwen3-4B-Base"
    ["Qwen3-8B-Base"]="Qwen/Qwen3-8B-Base"
)

# --- Variant -> Base mapping ---
# 1.7B ships every method; 4B and 8B only ship jetlong. The jetlong_fused variants
# wrap jetlong with the fused FA4/CuTe SM90 kernel (optional, separate env).
declare -A VARIANT_BASE=(
    # 1.7B variants
    ["Qwen3-1.7B-Base-yarn"]="Qwen3-1.7B-Base"
    ["Qwen3-1.7B-Base-dntk"]="Qwen3-1.7B-Base"
    ["Qwen3-1.7B-Base-selfext"]="Qwen3-1.7B-Base"
    ["Qwen3-1.7B-Base-dca"]="Qwen3-1.7B-Base"
    ["Qwen3-1.7B-Base-jetlong_freq"]="Qwen3-1.7B-Base"
    ["Qwen3-1.7B-Base-jetlong"]="Qwen3-1.7B-Base"
    ["Qwen3-1.7B-Base-jetlong_fused"]="Qwen3-1.7B-Base"
    # 4B and 8B variants (jetlong + jetlong_fused only)
    ["Qwen3-4B-Base-jetlong"]="Qwen3-4B-Base"
    ["Qwen3-4B-Base-jetlong_fused"]="Qwen3-4B-Base"
    ["Qwen3-8B-Base-jetlong"]="Qwen3-8B-Base"
    ["Qwen3-8B-Base-jetlong_fused"]="Qwen3-8B-Base"
)

# Step 1: Download base models
for model_name in "${!BASE_MODELS[@]}"; do
    repo_id="${BASE_MODELS[$model_name]}"
    if [ -d "$MODEL_CACHE/$model_name" ]; then
        echo "[skip] $model_name already exists"
    else
        echo "[download] $repo_id -> $MODEL_CACHE/$model_name"
        python "$REPO_ROOT/scripts/model/download_model_hf.py" "$repo_id" --no-load
    fi
done

# Step 2: Create variant directories with symlinks
for variant in "${!VARIANT_BASE[@]}"; do
    base="${VARIANT_BASE[$variant]}"
    base_dir="$MODEL_CACHE/$base"
    variant_dir="$MODEL_CACHE/$variant"
    config_dir="$MODEL_CONFIGS/$variant"

    if [ ! -d "$base_dir" ]; then
        echo "[error] Base model $base not found at $base_dir" >&2
        continue
    fi

    if [ ! -d "$config_dir" ]; then
        echo "[error] No configs for $variant in $config_dir" >&2
        continue
    fi

    mkdir -p "$variant_dir"

    # Symlink config files from model_configs/<variant>/ (config.json + any
    # variant-specific overrides). Edits land in the git-tracked source.
    echo "[setup] $variant: symlinking config files from model_configs/"
    for f in "$config_dir"/*; do
        fname="$(basename "$f")"
        target="$variant_dir/$fname"
        # Remove stale copy/link if it exists, config source takes priority
        [ -e "$target" ] || [ -L "$target" ] && rm -f "$target"
        ln -sr "$f" "$target"
        echo "  -> symlinked $fname (config)"
    done

    # Symlink the modeling file from jetlm/modeling/<method>/ if this method
    # ships custom modeling code. Method = variant name with the
    # "Qwen3-<size>-Base-" prefix stripped (e.g. jetlong, dca, selfext).
    # Config-only methods (yarn, dntk) have no matching modeling dir — skip.
    method="${variant#Qwen3-*-Base-}"
    modeling_dir="$JETLM_MODELING/$method"
    if [ -d "$modeling_dir" ]; then
        for f in "$modeling_dir"/modeling_qwen3_*.py; do
            [ -f "$f" ] || continue
            fname="$(basename "$f")"
            target="$variant_dir/$fname"
            [ -e "$target" ] || [ -L "$target" ] && rm -f "$target"
            ln -sr "$f" "$target"
            echo "  -> symlinked $fname (modeling, from jetlm/modeling/$method/)"
        done
    fi

    # Symlink shared files from base (weights, tokenizer, etc.)
    for f in "$base_dir"/*; do
        fname="$(basename "$f")"
        target="$variant_dir/$fname"
        # Skip if already exists (config/modeling overrides take priority)
        if [ -e "$target" ] || [ -L "$target" ]; then
            continue
        fi
        ln -sr "$f" "$target"
        echo "  -> symlinked $fname"
    done

    echo "[done] $variant"
done

echo ""
echo "All model variants set up in $MODEL_CACHE/"
