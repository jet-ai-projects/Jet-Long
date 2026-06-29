# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None


def _source_paths() -> list[str]:
    root = Path(__file__).resolve().parent / "csrc"
    return [str(root / "merge_fused.cpp"), str(root / "merge_fused.cu")]


def _build_directory() -> str:
    explicit = os.environ.get("JETLM_TORCH_EXTENSIONS_DIR")
    if explicit:
        root = Path(explicit).expanduser()
    elif os.environ.get("TORCH_EXTENSIONS_DIR"):
        root = Path(os.environ["TORCH_EXTENSIONS_DIR"]).expanduser()
    else:
        root = Path(__file__).resolve().parents[2] / ".torch_extensions"
    build_dir = root / "jetlm_merge_fused"
    build_dir.mkdir(parents=True, exist_ok=True)
    return str(build_dir)


def load_extension():
    """Load the fused CUDA merge ops."""
    global _EXT
    if _EXT is None:
        _EXT = load(
            name="jetlm_merge_fused",
            sources=_source_paths(),
            build_directory=_build_directory(),
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            verbose=os.environ.get("JETLM_JETLONG_KERNEL_VERBOSE", "0") == "1",
        )
    return _EXT


def _ensure_bshd_contiguous(out: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if out.dim() != 4 or lse.dim() != 3:
        raise ValueError("expected out=(B,S,H,D) and lse=(B,H,S)")
    if not out.is_cuda or not lse.is_cuda:
        raise ValueError("fused merge requires CUDA tensors")
    return out.contiguous(), lse.float().contiguous()


def merge2_bshd(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge two attention branches in one CUDA kernel.

    Inputs use FlashAttention output layout:
    - out_*: (batch, seqlen, nheads, head_dim)
    - lse_*: (batch, nheads, seqlen)
    """
    out_a, lse_a = _ensure_bshd_contiguous(out_a, lse_a)
    out_b, lse_b = _ensure_bshd_contiguous(out_b, lse_b)
    if out is None:
        return load_extension().merge2_bshd(out_a, lse_a, out_b, lse_b)
    return load_extension().merge2_bshd_out(out_a, lse_a, out_b, lse_b, out)
