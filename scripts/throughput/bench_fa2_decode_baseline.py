#!/usr/bin/env python3
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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

"""FA2 decode-latency baseline, for comparison against the fused CuTe decode.

Run in the FA2 (`jtl`) env. FA2 and FA4 cannot share one environment, so the FA2
reference and the fused decode are timed in separate processes; this script reuses
the fused bench's timing harness so the two sets of numbers line up.

Per context it reports:
  fa2_kvcache_ms         - one flash_attn_with_kvcache over the full context, i.e. a
                           plain FA2 per-step decode (the baseline the fused targets).
  fa2_unfused_jetlong_ms - near + distant FA2 decodes plus the LSE merge: the unfused
                           form of what the fused kernel does in a single launch.

Usage (repo root, FA2 env), alongside the fused env's decode bench:
  python scripts/throughput/bench_fa2_decode_baseline.py     --cuda-graph --nheads 32 --nheads-kv 8 --head-dim 128
  python scripts/throughput/bench_cute_decode_sm90_jetlong.py --cuda-graph --nheads 32 --nheads-kv 8 --head-dim 128
"""
import argparse
import sys
from pathlib import Path

import torch

# Reuse the fused decode bench's timing harness and input builder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cute_decode_sm90_jetlong as B  # noqa: E402

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache


def parse_args():
    p = argparse.ArgumentParser(description="FA2 decode-latency baseline for the fused-decode comparison.")
    p.add_argument("--contexts", type=int, nargs="+", default=[32769, 65537, 131073])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--nheads", type=int, default=32)
    p.add_argument("--nheads-kv", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--jetlong-w", type=int, default=32768)
    p.add_argument("--jetlong-w0", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main():
    assert B.HAVE_FA2, "run this in the FA2 env (flash_attn_interface is missing here)"
    args = parse_args()
    scale = args.head_dim ** -0.5
    dtype = torch.bfloat16
    print("context,fa2_kvcache_ms,fa2_unfused_jetlong_ms")
    for context in args.contexts:
        inp = B.make_relaxed_inputs(args, context)

        # FA2 optimal single decode (Flash-Decoding) over the full context.
        q = torch.randn(args.batch, 1, args.nheads, args.head_dim, device="cuda", dtype=dtype)
        k = torch.randn(args.batch, context, args.nheads_kv, args.head_dim, device="cuda", dtype=dtype)
        v = torch.randn(args.batch, context, args.nheads_kv, args.head_dim, device="cuda", dtype=dtype)

        def kvcache():
            return flash_attn_with_kvcache(q, k, v, softmax_scale=scale, causal=False)

        # Unfused JetLong reference: near fwd + distant fwd + merge.
        rn = torch.empty_like(inp["q_base"])
        rd = torch.empty_like(inp["q_base"])

        def unfused():
            on, ln = B.fa2_decode(inp["q_base"], inp["key_near"], inp["value_near"], scale, rn)
            od, ld = B.fa2_decode(inp["q_group"], inp["key_distant_grouped"], inp["value_distant"], scale, rd)
            return B.merge2_bshd_ref(on, ln, od, ld)

        kv_call = B.make_graph_callable(kvcache) if args.cuda_graph else kvcache
        unf_call = B.make_graph_callable(unfused) if args.cuda_graph else unfused
        kv_ms = B.bench_ms(kv_call, args.warmup, args.iters, True)
        u_ms = B.bench_ms(unf_call, args.warmup, args.iters, True)
        print(f"{context},{kv_ms:.4f},{u_ms:.4f}")


if __name__ == "__main__":
    main()
