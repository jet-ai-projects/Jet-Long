#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pull every eval dataset into its canonical local path under processed_data/
# by calling scripts/data/download_data.py once per dataset.
#
# Usage:
#   bash scripts/data/download_data.sh                     # default: ruler + pg19sub
#   bash scripts/data/download_data.sh --token $HF_TOKEN   # pass token through
#   DATASETS="ruler" bash scripts/data/download_data.sh    # override which datasets
#
# Exits non-zero only if EVERY dataset failed.

set -uo pipefail

DATASETS=${DATASETS:-"ruler pg19sub"}
PY=scripts/data/download_data.py

if [[ ! -f "$PY" ]]; then
    echo "[download_data] cannot find $PY — run from repo root."
    exit 2
fi

declare -i ok=0 fail=0
declare -a failed=()

for DS in $DATASETS; do
    echo ""
    echo "============================================================"
    echo "  dataset: $DS"
    echo "============================================================"
    if python "$PY" --dataset "$DS" "$@"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        failed+=("$DS")
    fi
done

echo ""
echo "============================================================"
echo "[download_data] done: $ok ok, $fail failed"
if (( fail > 0 )); then
    echo "[download_data] failed datasets: ${failed[*]}"
    if (( ok == 0 )); then
        exit 1
    fi
fi
