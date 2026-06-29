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

# SelfExtend for Qwen3
#
# Training-free context window extension via bi-level attention.
# Reference: "LLM Maybe LongLM: SelfExtend LLM Context Window Without Tuning"
#            Jin et al., ICML 2024.  https://arxiv.org/abs/2401.01325
#
# Two attention mechanisms merged at inference:
#   1. Neighbor attention: standard RoPE for tokens within a local window
#   2. Grouped attention:  floor-divided RoPE for distant tokens
#
# Max extended context = (L - w_n) * G_s + w_n
# where L = pretrain length, G_s = group_size, w_n = window_size

import math
from collections.abc import Callable
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
)

try:
    from flash_attn import flash_attn_varlen_func as _fa_flash_attn_varlen_func
    _FA2_AVAILABLE = True
except ImportError:
    _FA2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_q_or_k(x, cos, sin, unsqueeze_dim=1):
    """Apply rotary embedding to a single tensor (Q or K only)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def _apply_rope_flat(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a packed [total, heads, dim] tensor with per-token cos/sin.

    cos, sin: [total, dim].
    """
    cos = cos.unsqueeze(1)  # [total, 1, dim]
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def gqa_scores(query: torch.Tensor, key: torch.Tensor, num_kv_heads: int, groups: int) -> torch.Tensor:
    """QK^T for GQA without materializing a full-head key tensor.

    query: [bsz, num_heads, q_len, head_dim]  (num_heads == num_kv_heads * groups)
    key:   [bsz, num_kv_heads, kv_len, head_dim]
    returns: [bsz, num_heads, q_len, kv_len]
    """
    bsz, num_heads, q_len, head_dim = query.shape
    _, _, kv_len, _ = key.shape
    q = query.view(bsz, num_kv_heads, groups, q_len, head_dim)
    scores = torch.einsum("bhgqd,bhkd->bhgqk", q, key)
    return scores.reshape(bsz, num_heads, q_len, kv_len)


def gqa_apply_value(attn: torch.Tensor, value: torch.Tensor, num_kv_heads: int, groups: int) -> torch.Tensor:
    """attn @ V for GQA without materializing a full-head value tensor.

    attn:  [bsz, num_heads, q_len, kv_len]
    value: [bsz, num_kv_heads, kv_len, head_dim]
    returns: [bsz, num_heads, q_len, head_dim]
    """
    bsz, num_heads, q_len, kv_len = attn.shape
    _, _, _, head_dim = value.shape
    a = attn.view(bsz, num_kv_heads, groups, q_len, kv_len)
    out = torch.einsum("bhgqk,bhkd->bhgqd", a, value)
    return out.reshape(bsz, num_heads, q_len, head_dim)


def _get_selfext_cfg(config):
    """Extract SelfExtend config from a Qwen3Config."""
    cfg = getattr(config, "selfext", None)
    if cfg is None and hasattr(config, "to_dict"):
        cfg = config.to_dict().get("selfext", None)
    return cfg


# ---------------------------------------------------------------------------
# SelfExtend Attention
# ---------------------------------------------------------------------------

class Qwen3SelfExtendAttention(nn.Module):
    """Multi-headed attention with SelfExtend bi-level position encoding.

    Two execution paths:
    * Sequence <= window_size: standard attention (FlashAttention compatible)
    * Sequence > window_size: SelfExtend with neighbor + grouped attention
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        _ltypes = getattr(config, "layer_types", None) or []
        self.layer_type = _ltypes[layer_idx] if layer_idx < len(_ltypes) else None
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.hidden_size = config.num_attention_heads * self.head_dim

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )

        # SelfExtend hyper-parameters
        se_cfg = _get_selfext_cfg(config)
        self.selfext_enabled = se_cfg is not None
        if self.selfext_enabled:
            self.group_size = se_cfg.get("group_size", 8)
            self.window_size = se_cfg.get("window_size", 1024)
            self.scale_base = se_cfg.get("scale_base", -1)
            self.use_flash_attn = se_cfg.get("use_flash_attn", True) and _FA2_AVAILABLE

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project and normalize Q, K
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        bsz, _, q_len, _ = query_states.shape
        # Only the standard (non-selfext) path consumes cos/sin; in the
        # selfext path each attention computes its own rotations.
        cos, sin = position_embeddings if position_embeddings is not None else (None, None)

        # Determine total KV length
        past_kv_len = 0
        if past_key_values is not None:
            past_kv_len = past_key_values.get_seq_length(self.layer_idx)
        kv_seq_len = past_kv_len + q_len

        if not self.selfext_enabled:
            return self._standard_forward(
                query_states, key_states, value_states,
                cos, sin, input_shape, attention_mask,
                past_key_values, cache_position, **kwargs,
            )

        # FA2 path: O(N*w) memory vs O(N^2) for eager — required for long contexts.
        # Falls back to eager if FA2 unavailable or q_len pattern is unsupported.
        if self.use_flash_attn and not self.training and (q_len == 1 or q_len == kv_seq_len):
            return self._fa2_varlen_forward(
                query_states, key_states, value_states,
                input_shape, attention_mask,
                past_key_values, cache_position,
                bsz, q_len, kv_seq_len,
                position_ids=position_ids, **kwargs,
            )

        return self._selfextend_forward(
            query_states, key_states, value_states,
            input_shape, attention_mask,
            past_key_values, cache_position,
            bsz, q_len, kv_seq_len,
            position_ids=position_ids, **kwargs,
        )

    def _standard_forward(
        self, query_states, key_states, value_states,
        cos, sin, input_shape, attention_mask,
        past_key_values, cache_position, **kwargs,
    ):
        """Standard attention path (FlashAttention compatible)."""
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _selfextend_forward(
        self, query_states, key_states, value_states,
        input_shape, attention_mask,
        past_key_values, cache_position,
        bsz, q_len, kv_seq_len, position_ids=None, **kwargs,
    ):
        """SelfExtend bi-level attention (eager, materializes attention matrices).

        Uses semantic position ids when provided (preferred under left-padding),
        falling back to cache_position for unpadded inputs. The neighbor/group
        selection is a vectorized distance mask — supports arbitrary q_len
        including chunked prefill and speculative decoding.
        """
        group_size = self.group_size
        window_size = self.window_size
        rotary_emb = self._rotary_emb

        # Store un-rotated keys in cache so we can apply different RoPE.
        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Resolve query / key semantic positions. Prefer position_ids when given;
        # otherwise derive from cache_position (unpadded fast path).
        device = query_states.device
        if position_ids is not None:
            query_position = position_ids.to(device=device, dtype=torch.long)
        elif cache_position is not None:
            query_position = cache_position.unsqueeze(0).long().expand(bsz, -1)
        else:
            query_position = torch.arange(
                kv_seq_len - q_len, kv_seq_len, device=device, dtype=torch.long,
            ).unsqueeze(0).expand(bsz, -1)

        if q_len == kv_seq_len:
            key_position = query_position
        else:
            # For unpadded cache: keys occupy semantic positions
            # [last_q - kv + 1, ..., last_q]. Under left-padding this is
            # approximate; FA2 varlen is the correct path there.
            last_q = query_position[:, -1:]
            key_position = last_q - kv_seq_len + 1 + torch.arange(
                kv_seq_len, device=device, dtype=torch.long,
            ).unsqueeze(0)
            key_position = torch.clamp(key_position, min=0)

        # Optional query log-scaling.
        if self.scale_base > 0:
            scaled_query = query_states * (
                (query_position + 1)[:, None, :, None].float().log()
                / np.log(self.scale_base)
            ).clip(1).to(query_states.dtype)
        else:
            scaled_query = query_states

        # Grouped positions (unconditional shift — no batch-dependent .max()).
        shift = window_size - (window_size // group_size)
        group_query_position = (query_position // group_size) + shift
        group_key_position = key_position // group_size

        dummy = value_states[:, :, :1, :]
        neighbor_q_cos, neighbor_q_sin = rotary_emb(dummy, query_position)
        neighbor_k_cos, neighbor_k_sin = rotary_emb(dummy, key_position)
        group_q_cos, group_q_sin = rotary_emb(dummy, group_query_position)
        group_k_cos, group_k_sin = rotary_emb(dummy, group_key_position)

        neighbor_query = apply_rotary_pos_emb_q_or_k(scaled_query, neighbor_q_cos, neighbor_q_sin)
        neighbor_key = apply_rotary_pos_emb_q_or_k(key_states, neighbor_k_cos, neighbor_k_sin)
        group_query = apply_rotary_pos_emb_q_or_k(scaled_query, group_q_cos, group_q_sin)
        group_key = apply_rotary_pos_emb_q_or_k(key_states, group_k_cos, group_k_sin)

        # GQA-native scoring (no repeat_kv blowup).
        neighbor_attn = gqa_scores(
            neighbor_query, neighbor_key, self.num_kv_heads, self.num_key_value_groups
        ) * self.scaling
        group_attn = gqa_scores(
            group_query, group_key, self.num_kv_heads, self.num_key_value_groups
        ) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :q_len, :kv_seq_len]
            neighbor_attn = neighbor_attn + causal_mask
            group_attn = group_attn + causal_mask

        # Vectorized distance-based selection — works for any (q_len, kv_seq_len).
        # Shape: [bsz, q_len, kv_seq_len] -> broadcast over heads as [bsz, 1, q, k].
        dist = query_position.unsqueeze(-1) - key_position.unsqueeze(-2)
        neighbor_mask = ((dist >= 0) & (dist < window_size)).unsqueeze(1)

        attn_weights = torch.where(neighbor_mask, neighbor_attn, group_attn)

        # Softmax in fp32 for stability
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = gqa_apply_value(
            attn_weights, value_states, self.num_kv_heads, self.num_key_value_groups
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _fa2_varlen_forward(
        self, query_states, key_states, value_states,
        input_shape, attention_mask,
        past_key_values, cache_position,
        bsz, q_len, kv_seq_len, **kwargs,
    ):
        """SelfExtend via Flash Attention v2 varlen — O(N*w) memory, handles padding.

        Packs all samples into one flat sequence using `flash_attn_varlen_func`
        with cu_seqlens. Supports arbitrary batch size and variable real lengths.
        Prefill: two varlen calls (neighbor sliding-window + grouped distant keys)
        merged via log-sum-exp. Decode: single varlen call with LongLM's
        negated-sin rotation trick on keys and an un-rotated query.
        """
        group_size = self.group_size
        window_size = self.window_size
        rotary_emb = self._rotary_emb
        device = query_states.device

        # Store un-rotated KV in the cache for next step.
        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        full_kv_len = key_states.shape[-2]

        # Padding-agnostic per-sample real lengths and physical offsets.
        real_lens, offsets = self._infer_real_lengths_and_offsets(
            attention_mask, bsz, full_kv_len, device
        )
        # One CPU sync per forward, then reuse lists in all loops.
        real_lens_cpu = real_lens.tolist()
        offsets_cpu = offsets.tolist()

        num_heads = query_states.shape[1]
        num_kv_heads = key_states.shape[1]
        head_dim = query_states.shape[-1]

        is_decode = (q_len == 1)

        # --- Build per-token semantic position ids (flat across the batch) ---
        k_positions_flat = torch.cat([
            torch.arange(rl, device=device, dtype=torch.long) for rl in real_lens_cpu
        ])  # [total_k]

        if is_decode:
            q_positions_flat = (real_lens - 1).to(torch.long)  # [bsz]
        else:
            q_positions_flat = k_positions_flat  # prefill: q_len == real_len per sample

        # --- Unpad Q, K, V into packed [total, heads, dim] ---
        k_packed = self._unpad_heads_seq(key_states, offsets_cpu, real_lens_cpu)
        v_packed = self._unpad_heads_seq(value_states, offsets_cpu, real_lens_cpu)
        if is_decode:
            # query_states: [bsz, num_heads, 1, dim] -> [bsz, num_heads, dim]
            q_packed = query_states[:, :, 0, :].contiguous()
        else:
            q_packed = self._unpad_heads_seq(query_states, offsets_cpu, real_lens_cpu)

        # cu_seqlens (int32 cumulative), built from CPU list.
        cu_k_cpu = [0]
        for rl in real_lens_cpu:
            cu_k_cpu.append(cu_k_cpu[-1] + rl)
        cu_k = torch.tensor(cu_k_cpu, dtype=torch.int32, device=device)
        max_k = max(real_lens_cpu) if bsz > 0 else 0

        if is_decode:
            cu_q = torch.arange(bsz + 1, dtype=torch.int32, device=device)
            max_q = 1
        else:
            cu_q = cu_k
            max_q = max_k

        # Optional query log-scaling.
        if self.scale_base > 0:
            scale = (
                (q_positions_flat + 1).float().log()
                / np.log(self.scale_base)
            ).clip(1).to(q_packed.dtype)
            q_packed = q_packed * scale.view(-1, 1, 1)

        dummy = key_states[:, :, :1, :]
        # Unconditional shift — no batch-dependent .max() coupling.
        shift = window_size - (window_size // group_size)

        if is_decode:
            # Decode path: single varlen call with LongLM's trick rotation.
            parts = []
            q_pos_cpu = q_positions_flat.tolist()
            for b in range(bsz):
                rl = real_lens_cpu[b]
                last_q = q_pos_cpu[b]
                k_pos = torch.arange(rl, device=device, dtype=torch.long)
                neighbor = last_q - k_pos
                grouped = (last_q // group_size) - (k_pos // group_size) + shift
                if rl > window_size:
                    parts.append(torch.cat([grouped[:-window_size], neighbor[-window_size:]]))
                else:
                    parts.append(neighbor)
            decode_k_pos_flat = torch.cat(parts)  # [total_k]

            cos, sin = rotary_emb(dummy, decode_k_pos_flat.unsqueeze(0))  # [1, total_k, head_dim]
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
            k_rot = _apply_rope_flat(k_packed, cos, -sin)

            attn_output = _fa_flash_attn_varlen_func(
                q_packed, k_rot, v_packed,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=max_q, max_seqlen_k=max_k,
                dropout_p=0.0, causal=True,
            )  # [total_q, num_heads, dim]
        else:
            # Prefill path. (Unconditional `shift` above; no `.max()` sync.)

            # Neighbor rotations (standard RoPE on absolute positions).
            cos_n, sin_n = rotary_emb(dummy, k_positions_flat.unsqueeze(0))
            cos_n = cos_n.squeeze(0); sin_n = sin_n.squeeze(0)
            n_q = _apply_rope_flat(q_packed, cos_n, sin_n)
            n_k = _apply_rope_flat(k_packed, cos_n, sin_n)

            # Check if any sample extends beyond the window — otherwise skip grouped.
            if max_k <= window_size:
                attn_output = _fa_flash_attn_varlen_func(
                    n_q, n_k, v_packed,
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=max_q, max_seqlen_k=max_k,
                    dropout_p=0.0, causal=True,
                )
            else:
                # Grouped rotations (strict integer division, unconditional shift).
                g_q_pos = (q_positions_flat // group_size) + shift
                g_k_pos = k_positions_flat // group_size
                cos_gq, sin_gq = rotary_emb(dummy, g_q_pos.unsqueeze(0))
                cos_gk, sin_gk = rotary_emb(dummy, g_k_pos.unsqueeze(0))
                cos_gq = cos_gq.squeeze(0); sin_gq = sin_gq.squeeze(0)
                cos_gk = cos_gk.squeeze(0); sin_gk = sin_gk.squeeze(0)
                g_q = _apply_rope_flat(q_packed, cos_gq, sin_gq)
                g_k = _apply_rope_flat(k_packed, cos_gk, sin_gk)

                # Neighbor varlen with sliding window = last `window_size` keys.
                ngb_out, ngb_lse, _ = _fa_flash_attn_varlen_func(
                    n_q, n_k, v_packed,
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=max_q, max_seqlen_k=max_k,
                    dropout_p=0.0, causal=True,
                    window_size=(window_size - 1, 0),
                    return_attn_probs=True,
                )
                # ngb_out: [total_q, num_heads, dim]
                # ngb_lse: [num_heads, total_q]  (FA2 2.8 varlen packed LSE layout)

                # Per-sample grouped lengths and cu_seqlens — all CPU arithmetic.
                group_lens_cpu = [max(0, rl - window_size) for rl in real_lens_cpu]
                cu_g_cpu = [0]
                for gl in group_lens_cpu:
                    cu_g_cpu.append(cu_g_cpu[-1] + gl)
                cu_g = torch.tensor(cu_g_cpu, dtype=torch.int32, device=device)
                max_g = max(group_lens_cpu) if group_lens_cpu else 0

                # Build index tensors into the packed neighbor/group tensors.
                g_q_idx_parts = []
                g_k_idx_parts = []
                for b in range(bsz):
                    rl = real_lens_cpu[b]
                    gl = group_lens_cpu[b]
                    base = cu_k_cpu[b]
                    if gl == 0:
                        continue
                    g_q_idx_parts.append(torch.arange(base + window_size, base + rl, device=device))
                    g_k_idx_parts.append(torch.arange(base, base + gl, device=device))
                g_q_idx = torch.cat(g_q_idx_parts) if g_q_idx_parts else torch.empty(0, dtype=torch.long, device=device)
                g_k_idx = torch.cat(g_k_idx_parts) if g_k_idx_parts else torch.empty(0, dtype=torch.long, device=device)

                grp_out_packed, grp_lse, _ = _fa_flash_attn_varlen_func(
                    g_q[g_q_idx], g_k[g_k_idx], v_packed[g_k_idx],
                    cu_seqlens_q=cu_g, cu_seqlens_k=cu_g,
                    max_seqlen_q=max_g, max_seqlen_k=max_g,
                    dropout_p=0.0, causal=True,
                    return_attn_probs=True,
                )
                # grp_out_packed: [sum(group_lens), num_heads, dim]
                # grp_lse: [num_heads, sum(group_lens)]

                # LSE merge over the tail queries (q at position >= window per sample).
                # ngb_out at rows g_q_idx corresponds 1:1 with grp_out_packed rows.
                lse_n_tail = ngb_lse[:, g_q_idx].transpose(0, 1).unsqueeze(-1)  # [g_total, heads, 1]
                lse_g_tail = grp_lse.transpose(0, 1).unsqueeze(-1)              # [g_total, heads, 1]
                diff = lse_g_tail - lse_n_tail
                w_n = torch.sigmoid(-diff).to(ngb_out.dtype)
                w_g = torch.sigmoid(diff).to(grp_out_packed.dtype)

                merged = ngb_out.clone()
                merged[g_q_idx] = (
                    w_n * ngb_out[g_q_idx] + w_g * grp_out_packed
                )
                attn_output = torch.nan_to_num(merged, nan=0.0)

        # Re-pad: scatter attn_output back into [bsz, q_len, num_heads*head_dim].
        hidden_size = num_heads * head_dim
        padded = torch.zeros(
            bsz, q_len, hidden_size, dtype=attn_output.dtype, device=device
        )
        attn_output_flat = attn_output.reshape(-1, hidden_size)  # [total_q, hidden]
        if is_decode:
            # total_q == bsz; one row per sample at position 0.
            padded[:, 0, :] = attn_output_flat
        else:
            # Scatter each sample's packed rows back into [offset, offset+real_len).
            start = 0
            for b in range(bsz):
                rl = real_lens_cpu[b]
                off = offsets_cpu[b]
                padded[b, off : off + rl, :] = attn_output_flat[start : start + rl]
                start += rl

        attn_output = self.o_proj(padded)
        return attn_output, None

    @staticmethod
    def _infer_real_lengths_and_offsets(attention_mask, bsz, kv_len, device):
        """Return (real_lens, offsets) per sample — agnostic to padding side.

        A key column is "real" iff *some* query is allowed to attend to it.
        Works for both left- and right-padded harnesses. `offsets` is the
        physical starting index of real tokens per sample.
        """
        if attention_mask is None:
            real_lens = torch.full((bsz,), kv_len, dtype=torch.long, device=device)
            offsets = torch.zeros((bsz,), dtype=torch.long, device=device)
            return real_lens, offsets
        # Two layouts are possible:
        #   - 2D [bsz, kv_len]: raw padding mask (1=real, 0=pad).
        #   - 4D [bsz, 1, q_len, kv_len]: additive causal+padding (0=allowed, -inf=blocked).
        if attention_mask.dim() == 2:
            valid = attention_mask.bool()
        elif attention_mask.dim() == 4:
            # A key column is real iff some query row allows it.
            valid = (attention_mask[:, 0, :, :] > -1e4).any(dim=-2)
        else:
            raise ValueError(
                f"Unexpected attention_mask shape {tuple(attention_mask.shape)}"
            )
        real_lens = valid.sum(dim=-1).to(torch.long)
        # argmax over a bool/uint8 gives the first True index (== offset).
        offsets = valid.to(torch.uint8).argmax(dim=-1).to(torch.long)
        return real_lens, offsets

    @staticmethod
    def _unpad_heads_seq(x, offsets_cpu, real_lens_cpu):
        """x: [bsz, num_heads, full_seq, dim] -> [total, num_heads, dim] packed.

        Uses pre-materialized CPU lists (single sync up-front) to avoid
        per-iteration `.item()` calls that serialize the layer stack.
        """
        bsz = x.shape[0]
        chunks = []
        for b in range(bsz):
            rl = real_lens_cpu[b]
            off = offsets_cpu[b]
            chunks.append(x[b, :, off:off + rl, :].transpose(0, 1).contiguous())
        return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Decoder Layer (swaps in SelfExtend attention)
# ---------------------------------------------------------------------------

class Qwen3SelfExtendDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3SelfExtendAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        _ltypes = getattr(config, "layer_types", None) or []
        self.attention_type = _ltypes[layer_idx] if layer_idx < len(_ltypes) else "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen3SelfExtendModel(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3SelfExtendDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        _ltypes = getattr(config, "layer_types", None) or []
        self.has_sliding_layers = "sliding_attention" in _ltypes

        # Store rotary_emb reference on each attention layer for SelfExtend
        for layer in self.layers:
            layer.self_attn._rotary_emb = self.rotary_emb

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Prepare causal masks
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        # SelfExtend attention computes its own RoPE per-rotation; skip the
        # model-level call when every layer is selfext-enabled.
        if all(l.self_attn.selfext_enabled for l in self.layers):
            position_embeddings = None
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping.get(decoder_layer.attention_type, causal_mask_mapping.get("full_attention")),
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------

class Qwen3SelfExtendForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3SelfExtendModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            if logits_to_keep > 0:
                hidden_states = hidden_states[:, -logits_to_keep:, :]
            elif labels is not None:
                hidden_states = hidden_states  # keep all for loss
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
