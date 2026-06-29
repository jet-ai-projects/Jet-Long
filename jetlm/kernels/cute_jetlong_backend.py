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
import os

import torch

# Decode bucket: the CuTe decode compiles per K/V sequence length, and the distant
# region grows by one token per decode step. Padding the distant length up to this
# bucket keeps the compile key stable across steps (one compile per bucket instead of
# one per token); seqused_distant carries the true length so the kernel masks the pad.
_DECODE_DISTANT_BUCKET = 8192


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _pad_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a (B, S, H, D) tensor along the sequence dim with zeros up to target_len."""
    cur = x.shape[1]
    if cur >= target_len:
        return x.contiguous()
    pad = x.new_zeros(x.shape[0], target_len - cur, x.shape[2], x.shape[3])
    return torch.cat([x, pad], dim=1).contiguous()


def fused_jetlong_prefill(
    query_base: torch.Tensor,
    query_group: torch.Tensor,
    key_base: torch.Tensor,
    key_group: torch.Tensor,
    value: torch.Tensor,
    *,
    seqstart_q: int,
    window_size_left: int,
    softmax_scale: float,
    has_any_pad: bool,
    training: bool,
) -> torch.Tensor:
    if training:
        raise RuntimeError("JetLong backend 'fused' is inference-only")
    if has_any_pad:
        raise RuntimeError("JetLong backend 'fused' does not support left-padded prefill batches")

    from jetlm.kernels.cute_prefill_sm90 import fa4_cute_prefill_sm90_jetlong_fused

    batch = query_base.shape[0]
    seqstart_q = int(seqstart_q)
    out, _ = fa4_cute_prefill_sm90_jetlong_fused(
        query_base.transpose(1, 2).contiguous(),
        query_group.transpose(1, 2).contiguous(),
        key_base.transpose(1, 2).contiguous(),
        key_group.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        seqstart_q=torch.full((batch,), seqstart_q, device=query_base.device, dtype=torch.int32),
        seqstart_q_scalar=seqstart_q,
        window_size_left=int(window_size_left),
        softmax_scale=softmax_scale,
    )
    return out


def _decode_region_attention(q, k, v, scale):
    """Single-region attention for one decode step (q_len == 1), GQA-aware, fp32 math.

    q: (B, 1, H, D)   k, v: (B, N, Hk, D)   ->   out: (B, 1, H, D) fp32, lse: (B, H, 1) fp32.
    """
    rep = q.shape[2] // k.shape[2]
    kf = k.repeat_interleave(rep, dim=2).float()
    vf = v.repeat_interleave(rep, dim=2).float()
    scores = torch.einsum("bqhd,bnhd->bhqn", q.float(), kf) * scale
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqn,bnhd->bqhd", probs, vf)
    return out, lse


def _reference_jetlong_decode(query_base, query_group, key_near, value_near, key_distant_group, value_distant):
    """Exact near + distant LSE-merge decode (q_len == 1). The distant K is already
    grouped by the caller, so this is the canonical JetLong decode: base RoPE over the
    nearby window, grouped RoPE over the distant region, merged in fp32."""
    scale = 1.0 / math.sqrt(query_base.shape[-1])
    out_near, lse_near = _decode_region_attention(query_base, key_near, value_near, scale)
    out_dist, lse_dist = _decode_region_attention(query_group, key_distant_group, value_distant, scale)
    max_lse = torch.maximum(lse_near, lse_dist)
    w_near = torch.exp(lse_near - max_lse)
    w_dist = torch.exp(lse_dist - max_lse)
    denom = (w_near + w_dist).clamp_min(1e-20)
    wn = (w_near / denom).transpose(1, 2).unsqueeze(-1)
    wd = (w_dist / denom).transpose(1, 2).unsqueeze(-1)
    return (out_near * wn + out_dist * wd).to(query_base.dtype)


def fused_jetlong_decode(
    query_base: torch.Tensor,
    query_group: torch.Tensor,
    key_near: torch.Tensor,
    value_near: torch.Tensor,
    key_distant_group: torch.Tensor,
    value_distant: torch.Tensor,
    *,
    inv_freq: torch.Tensor,
    group_size: int,
    has_any_pad: bool,
    training: bool,
) -> torch.Tensor:
    if training:
        raise RuntimeError("JetLong backend 'fused' is inference-only")
    if has_any_pad:
        raise RuntimeError("JetLong backend 'fused' does not support left-padded decode batches")
    if key_distant_group.shape[1] == 0:
        raise RuntimeError("JetLong backend 'fused' requires a non-empty distant decode region")

    # Decode through the fused CuTe kernel. The pure-PyTorch near+distant LSE-merge
    # reference (cheap since q_len == 1) remains available via
    # JETLM_JETLONG_FUSED_DECODE=ref as a correctness oracle / fallback.
    if os.environ.get("JETLM_JETLONG_FUSED_DECODE", "kernel") == "ref":
        return _reference_jetlong_decode(
            query_base, query_group, key_near, value_near, key_distant_group, value_distant
        )

    from jetlm.kernels.cute_decode_sm90 import fa4_cute_decode_sm90_jetlong_fused

    batch, _, num_heads, _ = query_base.shape
    tile_n = 128
    near_len = key_near.shape[1]
    distant_len = key_distant_group.shape[1]
    # Pad the distant K/V up to a bucket so the compile key is stable across decode steps;
    # the true length is passed via seqused_distant and the kernel masks the padding.
    distant_padded = _round_up(distant_len, _DECODE_DISTANT_BUCKET)
    key_distant_b = _pad_seq(key_distant_group, distant_padded)
    value_distant_b = _pad_seq(value_distant, distant_padded)
    num_splits = min(4, max(1, math.ceil((near_len + distant_len) / tile_n)))
    out, _ = fa4_cute_decode_sm90_jetlong_fused(
        query_base.contiguous(),
        query_group.contiguous(),
        key_near.contiguous(),
        value_near.contiguous(),
        key_distant_b,
        value_distant_b,
        seqused_near=torch.full((batch,), near_len, device=query_base.device, dtype=torch.int32),
        seqused_distant=torch.full((batch,), distant_len, device=query_base.device, dtype=torch.int32),
        inv_freq=inv_freq,
        group_size=int(group_size),
        lse=torch.empty(batch, num_heads, 1, device=query_base.device, dtype=torch.float32),
        k_mode="consumer_near_offset",
        tile_n=tile_n,
        num_splits=num_splits,
        distant_k_grouped=True,
    )
    return out
