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

import math

import torch

from jetlm.kernels.cute_decode_sm90 import (
    _SUPPORTED_CUTE_DTYPES,
    _ensure_fa4_importable,
    _get_device_arch,
    _require_contiguous,
)
from jetlm.kernels.merge_fused import merge2_bshd


_BLOCK_SPARSE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = {}


def _make_mask_mod(base_seqlen_q: int, base_seqlen_k: int, seqstart_q: int, window_size_left: int):
    base_seqlen_q = int(base_seqlen_q)
    base_seqlen_k = int(base_seqlen_k)
    seqstart_q = int(seqstart_q)
    window_size_left = int(window_size_left)

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del batch_idx, head_idx, seqlen_info, aux_tensors
        q_abs_local = q_idx + seqstart_q
        q_abs_dist = q_idx + seqstart_q - base_seqlen_q
        local_keep = (
            (q_idx < base_seqlen_q)
            & (kv_idx < base_seqlen_k)
            & (kv_idx <= q_abs_local)
            & (kv_idx >= q_abs_local - window_size_left)
        )
        dist_keep = (
            (q_idx >= base_seqlen_q)
            & (kv_idx >= base_seqlen_k)
            & (kv_idx - base_seqlen_k <= q_abs_dist)
            & (kv_idx - base_seqlen_k < q_abs_dist - window_size_left)
        )
        return local_keep | dist_keep

    mask_mod.use_fast_sampling = True
    return mask_mod


def _get_block_sparse_tensors(
    batch_size: int,
    num_q_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    seqstart_q: int,
    window_size_left: int,
    tile_m: int,
    tile_n: int,
    device: torch.device,
):
    key = (
        int(batch_size),
        int(num_q_heads),
        int(seqlen_q),
        int(seqlen_k),
        int(seqstart_q),
        int(window_size_left),
        int(tile_m),
        int(tile_n),
        device.type,
        device.index,
    )
    cached = _BLOCK_SPARSE_CACHE.get(key)
    if cached is not None:
        return cached

    from flash_attn.cute.compute_block_sparsity import compute_block_sparsity

    mask_mod = _make_mask_mod(seqlen_q, seqlen_k, seqstart_q, window_size_left)
    _, sparse_tensors = compute_block_sparsity(
        tile_m=tile_m,
        tile_n=tile_n,
        batch_size=batch_size,
        num_heads=num_q_heads,
        seqlen_q=2 * seqlen_q,
        seqlen_k=2 * seqlen_k,
        mask_mod=mask_mod,
        aux_tensors=None,
        device=device,
        compute_full_blocks=True,
        use_fast_sampling=True,
    )
    cached = (
        sparse_tensors.full_block_cnt,
        sparse_tensors.full_block_idx,
        sparse_tensors.mask_block_cnt,
        sparse_tensors.mask_block_idx,
    )
    _BLOCK_SPARSE_CACHE[key] = cached
    return cached


def fa4_cute_prefill_sm90_jetlong_fused(
    q_base: torch.Tensor,
    q_group: torch.Tensor,
    k_base: torch.Tensor,
    k_group: torch.Tensor,
    value: torch.Tensor,
    *,
    seqstart_q: torch.Tensor,
    seqstart_q_scalar: int | None = None,
    window_size_left: int,
    softmax_scale: float | None = None,
    tile_m: int = 128,
    tile_n: int = 128,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    inv_freq: torch.Tensor | None = None,
    group_size: int | None = None,
    num_stages: int | None = None,
    deterministic: bool = False,
):
    op = "fa4_cute_prefill_sm90_jetlong_fused"
    unsupported = {
        "seqused_q": seqused_q,
        "seqused_k": seqused_k,
        "inv_freq": inv_freq,
        "group_size": group_size,
        "num_stages": num_stages,
    }
    used_unsupported = [name for name, value in unsupported.items() if value is not None]
    if used_unsupported:
        raise ValueError(
            f"{op}: unsupported metadata arguments for dense CuTe prefill: {', '.join(used_unsupported)}"
        )
    if not all(t.is_cuda for t in (q_base, q_group, k_base, k_group, value, seqstart_q)):
        raise ValueError(f"{op} requires CUDA tensors")
    if _get_device_arch() != 90:
        raise RuntimeError(f"{op} requires SM90/H100")
    if not all(t.dim() == 4 for t in (q_base, q_group, k_base, k_group, value)):
        raise ValueError(f"{op} expects rank-4 BSHD tensors")
    if q_base.dtype not in _SUPPORTED_CUTE_DTYPES:
        raise ValueError(f"{op}: q_base must have dtype torch.float16 or torch.bfloat16, got {q_base.dtype}")
    if not all(t.dtype == q_base.dtype for t in (q_group, k_base, k_group, value)):
        raise ValueError(f"{op}: Q/K/V tensors must share dtype")
    if not all(t.device == q_base.device for t in (q_group, k_base, k_group, value, seqstart_q)):
        raise ValueError(f"{op}: all tensors must be on the same CUDA device")
    for name, tensor in (
        ("q_base", q_base),
        ("q_group", q_group),
        ("k_base", k_base),
        ("k_group", k_group),
        ("value", value),
    ):
        _require_contiguous(op, name, tensor)
    if q_base.shape != q_group.shape:
        raise ValueError(f"{op}: q_base and q_group must share shape")
    if k_base.shape != k_group.shape or k_base.shape != value.shape:
        raise ValueError(f"{op}: k_base, k_group, and value must share shape")
    if q_base.shape[0] != k_base.shape[0] or q_base.shape[3] != k_base.shape[3]:
        raise ValueError(f"{op}: Q/K batch and head-dim must match")
    if q_base.shape[2] % k_base.shape[2] != 0:
        raise ValueError(f"{op}: q heads must be divisible by kv heads")
    if seqstart_q.dtype != torch.int32 or seqstart_q.dim() != 1 or seqstart_q.shape[0] != q_base.shape[0]:
        raise ValueError(f"{op}: seqstart_q must be int32 with shape (batch,)")
    if tile_m <= 0 or tile_n <= 0:
        raise ValueError(f"{op}: tile_m and tile_n must be positive")

    _ensure_fa4_importable(None)
    from flash_attn.cute.interface import flash_attn_func

    batch_size, seqlen_q, num_q_heads, _ = q_base.shape
    seqlen_k = k_base.shape[1]
    if seqstart_q_scalar is None:
        seqstart_vals = tuple(int(x) for x in seqstart_q.detach().cpu().tolist())
        if len(set(seqstart_vals)) != 1:
            raise ValueError("prefill CuTe fastpath requires uniform seqstart_q across batch")
        seqstart_scalar = seqstart_vals[0]
    else:
        seqstart_scalar = int(seqstart_q_scalar)

    q_packed = torch.cat((q_base, q_group), dim=1).contiguous()
    k_packed = torch.cat((k_base, k_group), dim=1).contiguous()
    v_packed = torch.cat((value, value), dim=1).contiguous()

    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx = _get_block_sparse_tensors(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        seqstart_q=seqstart_scalar,
        window_size_left=window_size_left,
        tile_m=tile_m,
        tile_n=tile_n,
        device=q_base.device,
    )
    out_packed, lse_packed = flash_attn_func(
        q_packed,
        k_packed,
        v_packed,
        softmax_scale=softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q_base.shape[-1]),
        causal=False,
        deterministic=deterministic,
        mask_mod=_make_mask_mod(seqlen_q, seqlen_k, seqstart_scalar, window_size_left),
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        block_size=(tile_m, tile_n),
        return_lse=True,
    )
    out = merge2_bshd(
        out_packed[:, :seqlen_q],
        lse_packed[:, :, :seqlen_q],
        out_packed[:, seqlen_q:],
        lse_packed[:, :, seqlen_q:],
    )
    lse = torch.logaddexp(lse_packed[:, :, :seqlen_q], lse_packed[:, :, seqlen_q:])
    return out, lse
