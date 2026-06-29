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

"""Prefill-throughput benchmark (tok/s) on a single GPU.

Times one forward pass over an L-token prompt with CUDA events
(prefill_tok/s = B * L / median_forward_time). Generation throughput is timed
separately, at the decode-kernel level, by bench_cute_decode_sm90_jetlong.py.

Usage (from repo root, in the env matching the method):
  python scripts/throughput/bench_prefill_throughput.py --model_path model_cache/Qwen3-8B-Base               --attn flash_attention_2
  python scripts/throughput/bench_prefill_throughput.py --model_path model_cache/Qwen3-8B-Base-jetlong_fused --attn sdpa
"""

import argparse
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def _cuda_time_ms(fn, warmup: int, iters: int) -> float:
    """Median wall time (ms) of fn() over `iters` timed runs after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--attn", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768, 65536, 98304, 131072])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda")
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    vocab = cfg.vocab_size
    print(f"# model={args.model_path} attn={args.attn} batch={args.batch} "
          f"sm={torch.cuda.get_device_capability()}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        attn_implementation=args.attn, torch_dtype=torch.bfloat16,
    ).to(dev).eval()

    print("length,prefill_tok_s")
    for L in args.lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # avoid pad/eos so nothing early-stops; keep tokens in-vocab
        ids = torch.randint(5, vocab, (args.batch, L), device=dev, dtype=torch.long)

        def prefill():
            return model(input_ids=ids, use_cache=True)

        try:
            prefill_ms = _cuda_time_ms(prefill, args.warmup, args.iters)
        except torch.cuda.OutOfMemoryError:
            print(f"{L},OOM")
            continue
        prefill_tps = args.batch * L / (prefill_ms / 1e3)
        peak_gb = torch.cuda.max_memory_reserved() / 1e9
        print(f"{L},{prefill_tps:.0f}    # peak={peak_gb:.1f}GB prefill={prefill_ms:.1f}ms")


if __name__ == "__main__":
    main()
