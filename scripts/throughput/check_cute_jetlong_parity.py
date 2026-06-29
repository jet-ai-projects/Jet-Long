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
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Smoke-test FA4 CuTe JetLong kernels against reference attention using "
            "Q/K/V tensors produced by a real JetLong model layer."
        )
    )
    p.add_argument(
        "--model-path",
        default="",
        help="Optional local or Hugging Face JetLong checkpoint. If omitted, build a small random model from --config-path.",
    )
    p.add_argument("--config-path", default="model_cache/Qwen3-1.7B-Base-jetlong_fused")
    p.add_argument(
        "--num-hidden-layers",
        type=int,
        default=1,
        help="Number of layers to instantiate when --model-path is omitted.",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
        help="Vocab size for the random model built when --model-path is omitted.",
    )
    p.add_argument("--layer-idx", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=768)
    p.add_argument("--context-len", type=int, default=769)
    p.add_argument("--jetlong-w", type=int, default=512)
    p.add_argument("--w0", type=int, default=64)
    p.add_argument("--token-id", type=int, default=42)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--tile-m-prefill", type=int, default=128)
    p.add_argument("--tile-n-prefill", type=int, default=128)
    p.add_argument("--tile-m-decode", type=int, default=64)
    p.add_argument("--tile-n-decode", type=int, default=96)
    p.add_argument("--num-splits", type=int, default=4)
    p.add_argument("--max-prefill-diff", type=float, default=1e-1)
    p.add_argument("--max-decode-diff", type=float, default=7e-2)
    p.add_argument("--max-prefill-lse-diff", type=float, default=2e-2)
    p.add_argument("--max-decode-lse-diff", type=float, default=2e-2)
    p.add_argument("--json-output", default="")
    p.add_argument("--skip-prefill", action="store_true")
    p.add_argument(
        "--run-decode",
        action="store_true",
        help=(
            "Also run the model-layer decode diagnostic. Synthetic decode "
            "correctness is covered by tests/test_fa4_cute_kernels.py."
        ),
    )
    p.add_argument(
        "--skip-decode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return p.parse_args()


def dtype_from_arg(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def max_mean_diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max": float(diff.max().item()),
        "mean": float(diff.mean().item()),
    }


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, threshold: float) -> dict[str, float]:
    stats = max_mean_diff(actual, expected)
    if stats["max"] > threshold:
        raise RuntimeError(f"{name} max_diff={stats['max']:.6f} exceeds threshold={threshold}")
    return stats


def repeat_kv(x: torch.Tensor, nheads: int) -> torch.Tensor:
    return x if x.shape[2] == nheads else x.repeat_interleave(nheads // x.shape[2], dim=2)


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = repeat_kv(k, q.shape[2])
    v = repeat_kv(v, q.shape[2])
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * softmax_scale
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None], -float("inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(torch.isfinite(lse).unsqueeze(-1), probs, torch.zeros_like(probs))
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v.float()).to(q.dtype)
    return out, lse


def merge2_bshd(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_lse = torch.maximum(lse_a.float(), lse_b.float())
    weight_a = torch.exp(lse_a.float() - max_lse)
    weight_b = torch.exp(lse_b.float() - max_lse)
    denom = torch.clamp_min(weight_a + weight_b, 1e-20)
    out = (
        out_a * (weight_a / denom).transpose(1, 2).unsqueeze(-1).to(out_a.dtype)
        + out_b * (weight_b / denom).transpose(1, 2).unsqueeze(-1).to(out_b.dtype)
    )
    return out, torch.logaddexp(lse_a.float(), lse_b.float())


def override_jetlong_config(model, *, jetlong_w: int, w0: int) -> None:
    jetlong = getattr(model.config, "jetlong", None)
    if jetlong is None:
        raise ValueError("model config does not expose a JetLong config")
    jetlong["w"] = int(jetlong_w)
    jetlong["w_0"] = int(w0)
    rotary = model.model.rotary_emb
    if hasattr(rotary, "jetlong_w"):
        rotary.jetlong_w = int(jetlong_w)
    if hasattr(rotary, "jetlong_w_0"):
        rotary.jetlong_w_0 = int(w0)
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "jetlong_w"):
            attn.jetlong_w = int(jetlong_w)
        if hasattr(attn, "jetlong_w_0"):
            attn.jetlong_w_0 = int(w0)


def load_model(args, dtype: torch.dtype):
    from transformers import AutoConfig, AutoModelForCausalLM

    if args.model_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, dtype=dtype)
        return model.cuda().eval()

    config = AutoConfig.from_pretrained(args.config_path, trust_remote_code=True)
    config.num_hidden_layers = max(args.num_hidden_layers, args.layer_idx + 1)
    config.vocab_size = max(args.vocab_size, args.token_id + 1)
    if hasattr(config, "max_window_layers"):
        config.max_window_layers = min(config.max_window_layers, config.num_hidden_layers)
    if getattr(config, "torch_dtype", None) is None:
        config.torch_dtype = dtype
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.to(dtype=dtype)
    return model.cuda().eval()


@torch.no_grad()
def make_layer_states(model, *, seq_len: int, token_id: int, layer_idx: int):
    layer = model.model.layers[layer_idx].self_attn
    device = next(model.parameters()).device
    input_ids = torch.full((1, seq_len), token_id, device=device, dtype=torch.long)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    hidden = model.model.embed_tokens(input_ids)
    hidden = model.model.layers[layer_idx].input_layernorm(hidden)
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.head_dim)

    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    if len(position_embeddings) < 5:
        raise ValueError(
            f"JetLong path is inactive at seq_len={seq_len}; use --jetlong-w smaller than the smoke length"
        )
    base_cos, base_sin, group_size, inv_freq, position_ids = position_embeddings[:5]
    apply_rotary = layer.forward.__globals__["apply_rotary_pos_emb"]
    apply_group = layer.forward.__globals__["correction_rotate_per_sample"]

    query_states = layer.q_norm(layer.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
    key_states = layer.k_norm(layer.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
    value_states = layer.v_proj(hidden).view(hidden_shape).transpose(1, 2)

    query_base, key_base = apply_rotary(query_states, key_states, base_cos, base_sin)
    delta = torch.floor(position_ids.float() / group_size) - position_ids.float()
    query_group = apply_group(query_base, delta, inv_freq, hidden.dtype)
    key_group = apply_group(key_base, delta, inv_freq, hidden.dtype)

    return {
        "layer": layer,
        "input_shape": input_shape,
        "group_size": int(group_size),
        "inv_freq": inv_freq,
        "query_base": query_base.transpose(1, 2).contiguous(),
        "query_group": query_group.transpose(1, 2).contiguous(),
        "key_base": key_base.transpose(1, 2).contiguous(),
        "key_group": key_group.transpose(1, 2).contiguous(),
        "value": value_states.transpose(1, 2).contiguous(),
    }


@torch.no_grad()
def run_prefill_smoke(model, args) -> dict[str, object]:
    from jetlm.kernels.cute_prefill_sm90 import fa4_cute_prefill_sm90_jetlong_fused

    states = make_layer_states(
        model,
        seq_len=args.seq_len,
        token_id=args.token_id,
        layer_idx=args.layer_idx,
    )
    q_base = states["query_base"]
    q_group = states["query_group"]
    k_base = states["key_base"]
    k_group = states["key_group"]
    value = states["value"]
    softmax_scale = states["layer"].scaling
    seqlen = q_base.shape[1]

    q_idx = torch.arange(seqlen, device=q_base.device).view(1, seqlen, 1)
    k_idx = torch.arange(seqlen, device=q_base.device).view(1, 1, seqlen)
    local_mask = (k_idx <= q_idx) & (k_idx >= q_idx - args.w0)
    distant_mask = k_idx < q_idx - args.w0

    local_out, local_lse = attention_ref(q_base, k_base, value, mask=local_mask, softmax_scale=softmax_scale)
    distant_out, distant_lse = attention_ref(q_group, k_group, value, mask=distant_mask, softmax_scale=softmax_scale)
    expected_out, expected_lse = merge2_bshd(local_out, local_lse, distant_out, distant_lse)

    actual_out, actual_lse = fa4_cute_prefill_sm90_jetlong_fused(
        q_base,
        q_group,
        k_base,
        k_group,
        value,
        seqstart_q=torch.zeros(q_base.shape[0], device=q_base.device, dtype=torch.int32),
        seqstart_q_scalar=0,
        window_size_left=args.w0,
        softmax_scale=softmax_scale,
        tile_m=args.tile_m_prefill,
        tile_n=args.tile_n_prefill,
    )

    expected_proj = states["layer"].o_proj(expected_out.reshape(*states["input_shape"], -1).contiguous())
    actual_proj = states["layer"].o_proj(actual_out.reshape(*states["input_shape"], -1).contiguous())
    return {
        "seq_len": seqlen,
        "group_size": states["group_size"],
        "attn": assert_close("prefill.attn", actual_out, expected_out, args.max_prefill_diff),
        "lse": assert_close("prefill.lse", actual_lse, expected_lse, args.max_prefill_lse_diff),
        "o_proj": assert_close("prefill.o_proj", actual_proj, expected_proj, args.max_prefill_diff),
    }


@torch.no_grad()
def run_decode_smoke(model, args) -> dict[str, object]:
    from jetlm.kernels.cute_decode_sm90 import fa4_cute_decode_sm90_jetlong_fused

    states = make_layer_states(
        model,
        seq_len=args.context_len,
        token_id=args.token_id,
        layer_idx=args.layer_idx,
    )
    q_base = states["query_base"][:, -1:].contiguous()
    q_group = states["query_group"][:, -1:].contiguous()
    key_base = states["key_base"]
    key_group = states["key_group"]
    value = states["value"]
    softmax_scale = states["layer"].scaling
    boundary = max(args.context_len - 1 - args.w0, 0)
    if boundary == 0:
        raise ValueError("decode smoke requires context_len > w0 + 1")

    key_near = key_base[:, boundary:].contiguous()
    value_near = value[:, boundary:].contiguous()
    key_distant = key_group[:, :boundary].contiguous()
    value_distant = value[:, :boundary].contiguous()

    near_out, near_lse = attention_ref(q_base, key_near, value_near, softmax_scale=softmax_scale)
    distant_out, distant_lse = attention_ref(q_group, key_distant, value_distant, softmax_scale=softmax_scale)
    expected_out, expected_lse = merge2_bshd(near_out, near_lse, distant_out, distant_lse)

    actual_out, actual_lse = fa4_cute_decode_sm90_jetlong_fused(
        q_base,
        q_group,
        key_near,
        value_near,
        key_distant,
        value_distant,
        seqused_near=torch.full((q_base.shape[0],), key_near.shape[1], device=q_base.device, dtype=torch.int32),
        seqused_distant=torch.full((q_base.shape[0],), key_distant.shape[1], device=q_base.device, dtype=torch.int32),
        inv_freq=states["inv_freq"],
        group_size=states["group_size"],
        lse=torch.empty(q_base.shape[0], q_base.shape[2], 1, device=q_base.device, dtype=torch.float32),
        tile_m=args.tile_m_decode,
        tile_n=args.tile_n_decode,
        num_splits=args.num_splits,
        k_mode="consumer_near_offset",
        distant_k_grouped=True,
    )

    expected_proj = states["layer"].o_proj(expected_out.reshape(*states["input_shape"][:1], 1, -1).contiguous())
    actual_proj = states["layer"].o_proj(actual_out.reshape(*states["input_shape"][:1], 1, -1).contiguous())
    return {
        "context_len": args.context_len,
        "boundary": boundary,
        "group_size": states["group_size"],
        "attn": assert_close("decode.attn", actual_out, expected_out, args.max_decode_diff),
        "lse": assert_close("decode.lse", actual_lse, expected_lse, args.max_decode_lse_diff),
        "o_proj": assert_close("decode.o_proj", actual_proj, expected_proj, args.max_decode_diff),
    }


def main():
    args = parse_args()
    run_decode = args.run_decode and not args.skip_decode
    if args.skip_prefill and not run_decode:
        raise ValueError("at least one of prefill or decode smoke must run")
    if not torch.cuda.is_available():
        raise RuntimeError("CuTe JetLong parity smoke requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("CuTe JetLong parity smoke requires SM90/H100")

    model = load_model(args, dtype_from_arg(args.dtype))
    override_jetlong_config(model, jetlong_w=args.jetlong_w, w0=args.w0)

    results: dict[str, object] = {
        "model_path": args.model_path or None,
        "config_path": args.config_path if not args.model_path else None,
        "random_init": not bool(args.model_path),
        "layer_idx": args.layer_idx,
        "dtype": args.dtype,
        "jetlong_w": args.jetlong_w,
        "w0": args.w0,
    }
    if not args.skip_prefill:
        results["prefill"] = run_prefill_smoke(model, args)
    if run_decode:
        results["decode"] = run_decode_smoke(model, args)

    text = json.dumps(results, indent=2)
    print(text)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(text + "\n")


if __name__ == "__main__":
    main()
