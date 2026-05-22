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

import math
from importlib.util import find_spec

import pytest
import torch


def _require_fa4_sm90():
    if not torch.cuda.is_available():
        pytest.skip("FA4 CuTe kernel tests require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FA4 CuTe kernel tests require SM90/H100")
    for module_name in ("flash_attn.cute.interface", "cutlass.cute", "quack"):
        if find_spec(module_name) is None:
            pytest.skip(f"FA4 CuTe dependency is missing: {module_name}")


def _repeat_kv(x: torch.Tensor, nheads: int) -> torch.Tensor:
    return x if x.shape[2] == nheads else x.repeat_interleave(nheads // x.shape[2], dim=2)


def _attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    k = _repeat_kv(k, q.shape[2])
    v = _repeat_kv(v, q.shape[2])
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * (q.shape[-1] ** -0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None], -float("inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(torch.isfinite(lse).unsqueeze(-1), probs, torch.zeros_like(probs))
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v.float()).to(q.dtype)
    return out, lse


def _merge_ref(out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor) -> torch.Tensor:
    max_lse = torch.maximum(lse_a, lse_b)
    weight_a = torch.exp(lse_a - max_lse)
    weight_b = torch.exp(lse_b - max_lse)
    denom = torch.clamp_min(weight_a + weight_b, 1e-20)
    return (
        out_a * (weight_a / denom).transpose(1, 2).unsqueeze(-1).to(out_a.dtype)
        + out_b * (weight_b / denom).transpose(1, 2).unsqueeze(-1).to(out_b.dtype)
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _correction_rotate_ref(x: torch.Tensor, delta: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    angles = delta.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
    emb = torch.cat((angles, angles), dim=-1)
    cos = emb.cos().to(x.dtype).unsqueeze(1)
    sin = emb.sin().to(x.dtype).unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


def _assert_allclose(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float = 1e-2):
    diff = (actual.float() - expected.float()).abs().max().item()
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol), f"max_diff={diff}"


def _tiny_jetlong_model():
    # The jetlong_fused variant always runs the fused CuTe kernel (no backend knob).
    from transformers import AutoConfig, AutoModelForCausalLM

    # Use the materialized model_cache variant (config.json + the modeling file are
    # symlinked adjacent there, which trust_remote_code needs). Run
    # `bash scripts/model/setup_model_variants.sh` first.
    config = AutoConfig.from_pretrained("model_cache/Qwen3-1.7B-Base-jetlong_fused", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.vocab_size = 128
    if hasattr(config, "max_window_layers"):
        config.max_window_layers = min(config.max_window_layers, config.num_hidden_layers)
    config.torch_dtype = torch.bfloat16
    config.jetlong["w"] = 128
    config.jetlong["w_0"] = 32

    torch.manual_seed(123)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return model.cuda().to(dtype=torch.bfloat16).eval()


def _jetlong_layer_states(model, seq_len: int):
    layer = model.model.layers[0]
    attn = layer.self_attn
    input_ids = (torch.arange(seq_len, device="cuda", dtype=torch.long) % model.config.vocab_size).unsqueeze(0)
    hidden = model.model.embed_tokens(input_ids)
    hidden = layer.input_layernorm(hidden)
    position_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    cache_position = torch.arange(seq_len, device="cuda", dtype=torch.long)

    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    query_states = attn.q_norm(attn.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    base_cos, base_sin, group_size, inv_freq, position_ids = position_embeddings
    query_base, key_base = attn.forward.__globals__["apply_rotary_pos_emb"](
        query_states, key_states, base_cos, base_sin
    )
    delta = torch.floor(position_ids.float() / group_size) - position_ids.float()
    query_group = attn.forward.__globals__["correction_rotate_per_sample"](
        query_base, delta, inv_freq, hidden.dtype
    )
    key_group = attn.forward.__globals__["correction_rotate_per_sample"](
        key_base, delta, inv_freq, hidden.dtype
    )
    return {
        "layer": layer,
        "attn": attn,
        "hidden": hidden,
        "position_embeddings": position_embeddings,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "input_shape": input_shape,
        "query_base": query_base.transpose(1, 2).contiguous(),
        "query_group": query_group.transpose(1, 2).contiguous(),
        "key_base": key_base.transpose(1, 2).contiguous(),
        "key_group": key_group.transpose(1, 2).contiguous(),
        "value": value_states.transpose(1, 2).contiguous(),
    }


@torch.inference_mode()
def test_qwen3_jetlong_fused_backend_prefill_matches_torch_native():
    _require_fa4_sm90()
    model = _tiny_jetlong_model()
    states = _jetlong_layer_states(model, 256)
    attn = states["attn"]

    seq_len = states["query_base"].shape[1]
    q_idx = torch.arange(seq_len, device="cuda").view(1, seq_len, 1)
    k_idx = torch.arange(seq_len, device="cuda").view(1, 1, seq_len)
    local_mask = (k_idx <= q_idx) & (k_idx >= q_idx - model.config.jetlong["w_0"])
    distant_mask = k_idx < q_idx - model.config.jetlong["w_0"]
    local_out, local_lse = _attention_ref(states["query_base"], states["key_base"], states["value"], local_mask)
    distant_out, distant_lse = _attention_ref(
        states["query_group"], states["key_group"], states["value"], distant_mask
    )
    expected_attn = _merge_ref(local_out, local_lse, distant_out, distant_lse)
    expected = attn.o_proj(expected_attn.reshape(*states["input_shape"], -1).contiguous())

    actual, _ = attn(states["hidden"], states["position_embeddings"], None, cache_position=states["cache_position"])

    _assert_allclose(actual, expected, atol=1e-1)


@torch.inference_mode()
def test_qwen3_jetlong_fused_backend_decode_matches_torch_native():
    _require_fa4_sm90()
    from transformers.cache_utils import DynamicCache

    model = _tiny_jetlong_model()
    context_len = 256
    states = _jetlong_layer_states(model, context_len + 1)
    attn = states["attn"]

    boundary = context_len - model.config.jetlong["w_0"]
    near_out, near_lse = _attention_ref(
        states["query_base"][:, -1:].contiguous(),
        states["key_base"][:, boundary:].contiguous(),
        states["value"][:, boundary:].contiguous(),
    )
    distant_out, distant_lse = _attention_ref(
        states["query_group"][:, -1:].contiguous(),
        states["key_group"][:, :boundary].contiguous(),
        states["value"][:, :boundary].contiguous(),
    )
    expected_attn = _merge_ref(near_out, near_lse, distant_out, distant_lse)
    expected = attn.o_proj(expected_attn.reshape(1, 1, -1).contiguous())

    cache = DynamicCache(config=model.config)
    hidden_context = states["hidden"][:, :context_len]
    position_ids_context = states["position_ids"][:, :context_len]
    position_embeddings_context = model.model.rotary_emb(hidden_context, position_ids_context)
    attn(
        hidden_context,
        position_embeddings_context,
        None,
        past_key_values=cache,
        cache_position=torch.arange(context_len, device="cuda"),
    )

    hidden_decode = states["hidden"][:, context_len:]
    position_ids_decode = states["position_ids"][:, context_len:]
    position_embeddings_decode = model.model.rotary_emb(hidden_decode, position_ids_decode)
    actual, _ = attn(
        hidden_decode,
        position_embeddings_decode,
        None,
        past_key_values=cache,
        cache_position=torch.tensor([context_len], device="cuda"),
    )

    _assert_allclose(actual, expected, atol=1.2e-1)


@torch.inference_mode()
def test_fa4_cute_decode_matches_torch_native():
    _require_fa4_sm90()
    from jetlm.kernels.cute_decode_sm90 import fa4_cute_decode_sm90_jetlong_fused

    torch.manual_seed(2)
    batch, context, nheads, nheads_kv, head_dim = 1, 4609, 8, 4, 128
    jetlong_w, jetlong_w0 = 4096, 512
    group_size = math.ceil(context / jetlong_w)
    boundary = context - 1 - jetlong_w0
    q_base = torch.randn(batch, 1, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
    key_base = torch.randn(batch, context, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(batch, context, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    inv_freq = torch.rand(head_dim // 2, device="cuda", dtype=torch.float32)

    position_ids = torch.full((batch, 1), context - 1, device="cuda", dtype=torch.long)
    delta_q = torch.floor(position_ids.float() / group_size) - position_ids.float()
    q_group = _correction_rotate_ref(q_base.transpose(1, 2), delta_q, inv_freq).transpose(1, 2).contiguous()
    idx = torch.arange(boundary, device="cuda", dtype=torch.float32)
    delta_k = torch.floor(idx.unsqueeze(0) / group_size) - idx.unsqueeze(0)
    key_distant = key_base[:, :boundary].contiguous()
    key_distant_grouped = _correction_rotate_ref(key_distant.transpose(1, 2), delta_k, inv_freq).transpose(1, 2).contiguous()

    near_out, near_lse = _attention_ref(q_base, key_base[:, boundary:].contiguous(), value[:, boundary:].contiguous())
    dist_out, dist_lse = _attention_ref(q_group, key_distant_grouped, value[:, :boundary].contiguous())
    expected = _merge_ref(near_out, near_lse, dist_out, dist_lse)

    expected_lse = torch.logaddexp(near_lse, dist_lse)

    actual, actual_lse = fa4_cute_decode_sm90_jetlong_fused(
        q_base,
        q_group,
        key_base[:, boundary:].contiguous(),
        value[:, boundary:].contiguous(),
        key_distant_grouped,
        value[:, :boundary].contiguous(),
        seqused_near=torch.full((batch,), context - boundary, device="cuda", dtype=torch.int32),
        seqused_distant=torch.full((batch,), boundary, device="cuda", dtype=torch.int32),
        inv_freq=inv_freq,
        group_size=group_size,
        lse=torch.empty(batch, nheads, 1, device="cuda", dtype=torch.float32),
        tile_m=64,
        tile_n=96,
        pack_gqa=True,
        k_mode="consumer_near_offset",
        num_splits=4,
        distant_k_grouped=True,
    )
    _assert_allclose(actual, expected, atol=7e-2)
    _assert_allclose(actual_lse, expected_lse, atol=7e-2)


@torch.inference_mode()
def test_fa4_cute_decode_includes_near_when_distant_larger():
    """Regression: the fused decode must include the near region even when the distant
    region is larger than it. The other decode tests are distant-dominant (near weight
    negligible), so a kernel that dropped the near region still matched the reference.
    Here the near keys are aligned to the query (high near scores) while the distant
    region is bigger, so dropping near changes the result substantially. The original
    bug read the near tensor offset by seqlen_distant -> out of bounds whenever
    seqlen_distant >= seqlen_near -> near silently dropped."""
    _require_fa4_sm90()
    from jetlm.kernels.cute_decode_sm90 import fa4_cute_decode_sm90_jetlong_fused

    torch.manual_seed(7)
    batch, nheads, nheads_kv, head_dim = 1, 8, 4, 128
    near, distant = 512, 8192  # distant > near -> triggers the (former) out-of-bounds near read

    q_base = torch.randn(batch, 1, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
    q_group = torch.randn(batch, 1, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
    # near keys aligned to the query's kv-direction -> high near scores -> near dominates
    qkv = q_base.view(batch, 1, nheads_kv, nheads // nheads_kv, head_dim).mean(3)
    key_near = (
        qkv.expand(batch, near, nheads_kv, head_dim)
        + 0.1 * torch.randn(batch, near, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    ).contiguous()
    value_near = torch.randn(batch, near, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    key_distant = (0.3 * torch.randn(batch, distant, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)).contiguous()
    value_distant = torch.randn(batch, distant, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    inv_freq = torch.rand(head_dim // 2, device="cuda", dtype=torch.float32)

    near_out, near_lse = _attention_ref(q_base, key_near, value_near)
    dist_out, dist_lse = _attention_ref(q_group, key_distant, value_distant)
    expected = _merge_ref(near_out, near_lse, dist_out, dist_lse)
    # the test is only meaningful if near carries real weight (else dropping it is invisible)
    near_frac = (near_lse - torch.logaddexp(near_lse, dist_lse)).exp().mean().item()
    assert near_frac > 0.3, f"near weight {near_frac} too small to detect a near-drop regression"

    actual, _ = fa4_cute_decode_sm90_jetlong_fused(
        q_base,
        q_group,
        key_near,
        value_near,
        key_distant,
        value_distant,
        seqused_near=torch.full((batch,), near, device="cuda", dtype=torch.int32),
        seqused_distant=torch.full((batch,), distant, device="cuda", dtype=torch.int32),
        inv_freq=inv_freq,
        group_size=2,
        lse=torch.empty(batch, nheads, 1, device="cuda", dtype=torch.float32),
        tile_m=64,
        tile_n=96,
        pack_gqa=True,
        k_mode="consumer_near_offset",
        num_splits=4,
        distant_k_grouped=True,
    )
    _assert_allclose(actual, expected, atol=7e-2)


@torch.inference_mode()
def test_fa4_cute_decode_rejects_noncontiguous_output():
    _require_fa4_sm90()
    from jetlm.kernels.cute_decode_sm90 import fa4_cute_decode_sm90_jetlong_fused

    batch, context, nheads, nheads_kv, head_dim = 1, 16, 4, 2, 64
    q_base = torch.randn(batch, 1, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
    q_group = torch.randn_like(q_base)
    key_near = torch.randn(batch, 4, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    value_near = torch.randn_like(key_near)
    key_distant = torch.randn(batch, context - 4, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    value_distant = torch.randn_like(key_distant)
    out_storage = torch.empty(batch, 1, nheads, head_dim * 2, device="cuda", dtype=torch.bfloat16)
    out = out_storage[..., ::2]
    assert not out.is_contiguous()

    with pytest.raises(ValueError, match="out must be contiguous"):
        fa4_cute_decode_sm90_jetlong_fused(
            q_base,
            q_group,
            key_near,
            value_near,
            key_distant,
            value_distant,
            seqused_near=torch.full((batch,), key_near.shape[1], device="cuda", dtype=torch.int32),
            seqused_distant=torch.full((batch,), key_distant.shape[1], device="cuda", dtype=torch.int32),
            inv_freq=torch.rand(head_dim // 2, device="cuda", dtype=torch.float32),
            group_size=4,
            out=out,
        )


@torch.inference_mode()
def test_fa4_cute_decode_rejects_bad_metadata():
    _require_fa4_sm90()
    from jetlm.kernels.cute_decode_sm90 import fa4_cute_decode_sm90_jetlong_fused

    batch, context, nheads, nheads_kv, head_dim = 1, 16, 4, 2, 64
    q_base = torch.randn(batch, 1, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
    q_group = torch.randn_like(q_base)
    key_near = torch.randn(batch, 4, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    value_near = torch.randn_like(key_near)
    key_distant = torch.randn(batch, context - 4, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    value_distant = torch.randn_like(key_distant)

    with pytest.raises(ValueError, match="active_splits must have dtype torch.int32"):
        fa4_cute_decode_sm90_jetlong_fused(
            q_base,
            q_group,
            key_near,
            value_near,
            key_distant,
            value_distant,
            seqused_near=torch.full((batch,), key_near.shape[1], device="cuda", dtype=torch.int32),
            seqused_distant=torch.full((batch,), key_distant.shape[1], device="cuda", dtype=torch.int32),
            inv_freq=torch.rand(head_dim // 2, device="cuda", dtype=torch.float32),
            group_size=4,
            active_splits=torch.ones(batch, device="cuda", dtype=torch.int64),
        )


@torch.inference_mode()
def test_fa4_cute_prefill_matches_torch_native():
    _require_fa4_sm90()
    from jetlm.kernels.cute_prefill_sm90 import fa4_cute_prefill_sm90_jetlong_fused

    torch.manual_seed(3)
    batch, seqlen, nheads, nheads_kv, head_dim = 1, 256, 8, 4, 128
    w0, seqstart = 64, 0
    q_base = torch.randn(batch, seqlen, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
    q_group = torch.randn(batch, seqlen, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_base = torch.randn(batch, seqlen, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    k_group = torch.randn(batch, seqlen, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(batch, seqlen, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    q_idx = torch.arange(seqlen, device="cuda").view(1, seqlen, 1)
    k_idx = torch.arange(seqlen, device="cuda").view(1, 1, seqlen)
    local_mask = (k_idx <= q_idx + seqstart) & (k_idx >= q_idx + seqstart - w0)
    distant_mask = k_idx < q_idx + seqstart - w0

    local_out, local_lse = _attention_ref(q_base, k_base, value, local_mask.expand(batch, -1, -1))
    distant_out, distant_lse = _attention_ref(q_group, k_group, value, distant_mask.expand(batch, -1, -1))
    expected = _merge_ref(local_out, local_lse, distant_out, distant_lse)

    expected_lse = torch.logaddexp(local_lse, distant_lse)

    actual, actual_lse = fa4_cute_prefill_sm90_jetlong_fused(
        q_base,
        q_group,
        k_base,
        k_group,
        value,
        seqstart_q=torch.full((batch,), seqstart, device="cuda", dtype=torch.int32),
        seqstart_q_scalar=seqstart,
        window_size_left=w0,
        tile_m=128,
        tile_n=128,
    )
    _assert_allclose(actual, expected, atol=1e-1)
    assert actual_lse.shape == expected_lse.shape
    _assert_allclose(actual_lse, expected_lse, atol=1e-1)


@torch.inference_mode()
def test_fa4_cute_prefill_rejects_unsupported_metadata():
    _require_fa4_sm90()
    from jetlm.kernels.cute_prefill_sm90 import fa4_cute_prefill_sm90_jetlong_fused

    batch, seqlen, nheads, nheads_kv, head_dim = 1, 16, 4, 2, 64
    q_base = torch.randn(batch, seqlen, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
    q_group = torch.randn_like(q_base)
    k_base = torch.randn(batch, seqlen, nheads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    k_group = torch.randn_like(k_base)
    value = torch.randn_like(k_base)

    with pytest.raises(ValueError, match="unsupported metadata arguments.*seqused_q"):
        fa4_cute_prefill_sm90_jetlong_fused(
            q_base,
            q_group,
            k_base,
            k_group,
            value,
            seqstart_q=torch.zeros(batch, device="cuda", dtype=torch.int32),
            window_size_left=4,
            seqused_q=torch.full((batch,), seqlen, device="cuda", dtype=torch.int32),
        )
