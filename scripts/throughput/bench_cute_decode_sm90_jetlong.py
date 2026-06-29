#!/usr/bin/env python3
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

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# FA2 is only used for the optional baseline. In the fused (FA4-only) environment
# flash_attn.flash_attn_interface is absent — fall back to timing the fused kernel
# alone (the fused-vs-FA2 ratio requires a dual-stack env with both FA2 and FA4).
try:
    from flash_attn.flash_attn_interface import flash_attn_gpu, maybe_contiguous
    HAVE_FA2 = True
except ImportError:
    flash_attn_gpu = None
    HAVE_FA2 = False

    def maybe_contiguous(x):
        return x.contiguous() if x.stride(-1) != 1 else x

from jetlm.kernels.cute_decode_sm90 import fa4_cute_decode_sm90_jetlong_fused


_L2_FLUSH_BUFFER = None


def flush_l2() -> None:
    global _L2_FLUSH_BUFFER
    size = 256 * 1024 * 1024 // 4
    if _L2_FLUSH_BUFFER is None or _L2_FLUSH_BUFFER.numel() < size:
        _L2_FLUSH_BUFFER = torch.empty(size, device="cuda", dtype=torch.int32)
    _L2_FLUSH_BUFFER.zero_()
    torch.cuda.synchronize()


def bench_ms(fn, warmup: int, iters: int, cold_l2: bool) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if cold_l2:
            flush_l2()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def make_graph_callable(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph.replay


def fa2_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, out: torch.Tensor | None = None):
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, lse, _, _ = flash_attn_gpu.fwd(q, k, v, out, None, 0.0, scale, False, -1, -1, 0.0, False, None)
    return out, lse


def merge2_bshd_ref(
    out_near: torch.Tensor,
    lse_near: torch.Tensor,
    out_dist: torch.Tensor,
    lse_dist: torch.Tensor,
) -> torch.Tensor:
    lse_n = lse_near.float().squeeze(-1)
    lse_d = lse_dist.float().squeeze(-1)
    max_lse = torch.maximum(lse_n, lse_d)
    weight_near = torch.exp(lse_n - max_lse)
    weight_dist = torch.exp(lse_d - max_lse)
    denom = torch.clamp_min(weight_near + weight_dist, 1e-20)
    return (
        out_near * (weight_near / denom)[:, None, :, None].to(out_near.dtype)
        + out_dist * (weight_dist / denom)[:, None, :, None].to(out_dist.dtype)
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def correction_rotate_ref(x: torch.Tensor, delta_positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    angles = delta_positions.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
    emb = torch.cat((angles, angles), dim=-1)
    cos = emb.cos().to(x.dtype).unsqueeze(1)
    sin = emb.sin().to(x.dtype).unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


def make_relaxed_inputs(args, context: int):
    dtype = torch.bfloat16
    group_size = max(1, math.ceil(context / args.jetlong_w))
    boundary = max(context - 1 - args.jetlong_w0, 0)
    if boundary == 0:
        raise ValueError("relaxed JetLong decode benchmark requires context > jetlong_w0 + 1")

    q_base = torch.randn(args.batch, 1, args.nheads, args.head_dim, device="cuda", dtype=dtype)
    key_base = torch.randn(args.batch, context, args.nheads_kv, args.head_dim, device="cuda", dtype=dtype)
    value = torch.randn(args.batch, context, args.nheads_kv, args.head_dim, device="cuda", dtype=dtype)
    position_ids = torch.full((args.batch, 1), context - 1, device="cuda", dtype=torch.long)
    inv_freq = torch.rand(args.head_dim // 2, device="cuda", dtype=torch.float32)

    delta_q = torch.floor(position_ids.float() / group_size) - position_ids.float()
    q_group = correction_rotate_ref(q_base.transpose(1, 2), delta_q, inv_freq).transpose(1, 2).contiguous()

    key_distant = key_base[:, :boundary].contiguous()
    idx = torch.arange(boundary, device="cuda", dtype=torch.float32)
    delta_k = torch.floor(idx.unsqueeze(0) / group_size) - idx.unsqueeze(0)
    key_distant_grouped = correction_rotate_ref(key_distant.transpose(1, 2), delta_k, inv_freq).transpose(1, 2).contiguous()

    return {
        "group_size": group_size,
        "boundary": boundary,
        "q_base": q_base,
        "q_group": q_group,
        "key_near": key_base[:, boundary:].contiguous(),
        "value_near": value[:, boundary:].contiguous(),
        "key_distant_grouped": key_distant_grouped,
        "value_distant": value[:, :boundary].contiguous(),
        "seqused_near": torch.full((args.batch,), context - boundary, device="cuda", dtype=torch.int32),
        "seqused_distant": torch.full((args.batch,), boundary, device="cuda", dtype=torch.int32),
        "inv_freq": inv_freq,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark relaxed CuTe SM90 JetLong decode.")
    parser.add_argument("--contexts", type=int, nargs="+", default=[32769, 65537, 131073])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--nheads", type=int, default=16)
    parser.add_argument("--nheads-kv", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--jetlong-w", type=int, default=32768)
    parser.add_argument("--jetlong-w0", type=int, default=4096)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=96)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--num-splits", type=int, default=8)
    parser.add_argument("--pack-gqa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--cold-l2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-diff", type=float, default=5e-2)
    return parser.parse_args()


def main():
    args = parse_args()
    scale = args.head_dim ** -0.5
    if HAVE_FA2:
        print("context,fa2_relaxed_ref_ms,cute_relaxed_fused_ms,cute_vs_ref,max_abs_diff")
    else:
        print("# flash_attn (FA2) not installed — timing the fused CuTe decode only "
              "(no FA2 baseline / parity ratio; needs a dual-stack FA2+FA4 env for those)")
        print("context,cute_relaxed_fused_ms")

    for context in args.contexts:
        tensors = make_relaxed_inputs(args, context)
        q_base = tensors["q_base"]
        q_group = tensors["q_group"]
        key_near = tensors["key_near"]
        value_near = tensors["value_near"]
        key_distant_grouped = tensors["key_distant_grouped"]
        value_distant = tensors["value_distant"]

        ref_near_out = torch.empty_like(q_base)
        ref_dist_out = torch.empty_like(q_base)
        cute_out = torch.empty_like(q_base)
        cute_lse = torch.empty(args.batch, args.nheads, 1, device="cuda", dtype=torch.float32)

        def fa2_relaxed_ref_call():
            out_near, lse_near = fa2_decode(q_base, key_near, value_near, scale, ref_near_out)
            out_dist, lse_dist = fa2_decode(q_group, key_distant_grouped, value_distant, scale, ref_dist_out)
            return merge2_bshd_ref(out_near, lse_near, out_dist, lse_dist)

        def cute_relaxed_call():
            return fa4_cute_decode_sm90_jetlong_fused(
                q_base,
                q_group,
                key_near,
                value_near,
                key_distant_grouped,
                value_distant,
                seqused_near=tensors["seqused_near"],
                seqused_distant=tensors["seqused_distant"],
                inv_freq=tensors["inv_freq"],
                group_size=tensors["group_size"],
                out=cute_out,
                lse=cute_lse,
                tile_m=args.tile_m,
                tile_n=args.tile_n,
                num_stages=args.num_stages,
                pack_gqa=args.pack_gqa,
                k_mode="consumer_near_offset",
                num_splits=args.num_splits,
                distant_k_grouped=True,
            )

        cute_result = cute_relaxed_call()
        cute_tensor = cute_result[0] if isinstance(cute_result, tuple) else cute_result

        cute_call = cute_relaxed_call
        if args.cuda_graph:
            cute_call = make_graph_callable(cute_call)

        if HAVE_FA2:
            ref_out = fa2_relaxed_ref_call()
            diff = (ref_out - cute_tensor).abs().max().item()
            if diff > args.max_diff:
                raise RuntimeError(f"context={context} max_diff={diff} exceeds {args.max_diff}")
            ref_call = fa2_relaxed_ref_call
            if args.cuda_graph:
                ref_call = make_graph_callable(ref_call)
            ref_ms = bench_ms(ref_call, args.warmup, args.iters, args.cold_l2)
            cute_ms = bench_ms(cute_call, args.warmup, args.iters, args.cold_l2)
            print(f"{context},{ref_ms:.4f},{cute_ms:.4f},{cute_ms / ref_ms:.3f},{diff:.6f}")
        else:
            cute_ms = bench_ms(cute_call, args.warmup, args.iters, args.cold_l2)
            print(f"{context},{cute_ms:.4f}")


if __name__ == "__main__":
    main()
