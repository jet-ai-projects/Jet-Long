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

# JetLong for Qwen3  —  Bifocal Dynamic Self-Extend
#
# Drops the YaRN squeeze/temperature machinery. The distant branch replaces
# NTK-by-parts frequency correction with Self-Extend-style integer-floor
# position grouping: p -> floor(p / G), G = ceil(L_curr / w). This caps every
# evaluated phase at <= w (the pretrained window) on every frequency channel,
# eliminating OOD phase drift on unscaled high-freq dims that YaRN leaves
# untouched.
#
# Attention structure is unchanged:
#   Call A — local base RoPE over last w_0 tokens (exact-match zone)
#   Call B — distant grouped-RoPE, full causal
#   Call C — distant grouped-RoPE, windowed (subtracted to avoid double-counting)
# All three calls use self.scaling (t=1), restoring single-temperature softmax
# semantics in the A+B-C merge. No YaRN ramp, no temperature, no calibration.

import math
from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

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
    eager_attention_forward,
)


# ---------------------------------------------------------------------------
# Rotary Embedding (JetLong — Dynamic Self-Extend)
# ---------------------------------------------------------------------------

class Qwen3JetLongRotaryEmbedding(nn.Module):
    """
    Dynamic-SE rotary: returns base (cos, sin) plus, when L_curr > w, the
    dynamic group size G and base inv_freq so attention can compute grouped
    queries/keys on-the-fly via correction_rotate.

    Returns:
        If jetlong disabled or L_curr <= w: (base_cos, base_sin) — 2-tuple
        If jetlong enabled and L_curr > w:   (base_cos, base_sin, G, inv_freq) — 4-tuple
    """
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        # --- Base inv_freq (identical to vanilla Qwen3RotaryEmbedding) ---
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        # --- JetLong configuration ---
        jetlong_cfg = getattr(config, "jetlong", None)
        if jetlong_cfg is None:
            jetlong_cfg = config.to_dict().get("jetlong", None) if hasattr(config, "to_dict") else None

        self.jetlong_enabled = jetlong_cfg is not None

        if self.jetlong_enabled:
            self._jetlong_cfg = jetlong_cfg
            self.jetlong_w = jetlong_cfg["w"]
            self.jetlong_w_0 = jetlong_cfg["w_0"]
            self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3Config | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        device = x.device
        dtype = x.dtype

        def _compute_base_cos_sin():
            inv_exp = self.inv_freq[None, :, None].float().expand(
                position_ids.shape[0], -1, 1
            ).to(device)
            pos_exp = position_ids[:, None, :].float().to(device)

            device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
            with maybe_autocast(device_type=device_type, enabled=False):
                freqs = (inv_exp.float() @ pos_exp.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
            return cos.to(dtype), sin.to(dtype)

        base_cos, base_sin = _compute_base_cos_sin()

        if not self.jetlong_enabled:
            return base_cos, base_sin

        max_pos = position_ids.max().item()
        L_curr = max_pos + 1

        if L_curr <= self.jetlong_w:
            # Within pretrained range: pure base, no grouping
            return base_cos, base_sin

        # --- Dynamic Self-Extend group size ---
        # G = ceil(L_curr / w) keeps every grouped position <= w-1 (in-distribution phase).
        G = max(1, math.ceil(L_curr / self.jetlong_w))

        # Pass position_ids through so attention can compute per-sample grouping
        # deltas — required for correctness with left-padded batches.
        return base_cos, base_sin, G, self.inv_freq.to(device), position_ids


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_q_only(q, cos, sin, unsqueeze_dim=1):
    """Apply rotary embedding to query states only."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


def correction_rotate(x, positions, delta_inv_freq, dtype, seq_dim=2):
    """Apply correction rotation with batch-shared positions (seq_len,).

    Args:
        x: tensor with seq_len at dimension seq_dim
        positions: (seq_len,) integer or float positions
        delta_inv_freq: (dim/2,) delta frequency vector
        dtype: output dtype
        seq_dim: which dimension of x holds the sequence length
    """
    angles = torch.outer(positions.float(), delta_inv_freq.float())  # (S, D/2)
    emb = torch.cat((angles, angles), dim=-1)  # (S, D)
    corr_cos = emb.cos().to(dtype)
    corr_sin = emb.sin().to(dtype)

    shape = [1] * x.dim()
    shape[seq_dim] = positions.shape[0]
    shape[-1] = x.shape[-1]

    return x * corr_cos.view(*shape) + rotate_half(x) * corr_sin.view(*shape)


def correction_rotate_per_sample(x, positions_2d, delta_inv_freq, dtype):
    """Apply correction rotation with per-sample positions (B, S).

    Needed when a batch is left-padded: HF adjusts base RoPE via per-sample
    position_ids, so the correction must also be per-sample to stay aligned.

    Args:
        x: (B, H, S, D) base-rotated tensor.
        positions_2d: (B, S) delta positions per sample.
        delta_inv_freq: (D/2,) delta frequency vector.
        dtype: output dtype.
    """
    # angles: (B, S, D/2) = positions_2d (B, S, 1) * delta_inv_freq (1, 1, D/2)
    angles = positions_2d.float().unsqueeze(-1) * delta_inv_freq.float().view(1, 1, -1)
    emb = torch.cat((angles, angles), dim=-1)  # (B, S, D)
    corr_cos = emb.cos().to(dtype).unsqueeze(1)  # (B, 1, S, D), broadcasts over heads
    corr_sin = emb.sin().to(dtype).unsqueeze(1)
    return x * corr_cos + rotate_half(x) * corr_sin


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads: (batch, kv_heads, seq, dim) -> (batch, heads, seq, dim)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Attention (JetLong v1)
# ---------------------------------------------------------------------------

class Qwen3JetLongAttention(nn.Module):
    """Multi-headed attention with JetLong v1 (Naive Hard Split)."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        # --- JetLong parameters ---
        jetlong_cfg = config.to_dict().get("jetlong", None) if hasattr(config, "to_dict") else None
        self.jetlong_enabled = jetlong_cfg is not None
        if self.jetlong_enabled:
            self.jetlong_w = jetlong_cfg["w"]
            self.jetlong_w_0 = jetlong_cfg["w_0"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # --- Project & QK-norm ---
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # --- Dispatch on position_embeddings tuple length ---
        is_extended = len(position_embeddings) >= 4

        if not is_extended:
            # ---- PATH 1: Pure base attention (L_curr <= w or jetlong disabled) ----
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        # ---- Extended paths (JetLong: Bifocal Dynamic Self-Extend) ----
        base_cos, base_sin, G, inv_freq, position_ids = position_embeddings
        w_0 = self.jetlong_w_0
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Apply base RoPE to Q and K
        query_base, key_base = apply_rotary_pos_emb(
            query_states, key_states, base_cos, base_sin
        )

        B = query_base.shape[0]

        # Cache base-rotated keys (invariant: cache always stores base-RoPE'd keys)
        if past_key_values is not None:
            cache_kwargs = {"sin": base_sin, "cos": base_cos, "cache_position": cache_position}
            key_base, value_states = past_key_values.update(
                key_base, value_states, self.layer_idx, cache_kwargs
            )

        max_q_pos = cache_position[-1].item() if cache_position is not None else 0
        seq_len = query_base.shape[2]
        total_kv_len = key_base.shape[2]

        # O(1) pad detection: position_ids[b, -1] is sample b's last real position.
        # For fresh prefill: pad_b = (total_kv_len - 1) - last_real_pos = # of left-pads.
        # For chunked prefill & decode: the same identity holds because each chunk's
        # last position_ids element is the most recent real position in cache.
        if position_ids is not None and B > 1:
            pad_counts_kv = (total_kv_len - 1 - position_ids[:, -1]).clamp(min=0).long()
            has_any_pad = bool((pad_counts_kv > 0).any().item())
        else:
            pad_counts_kv = torch.zeros(B, device=device, dtype=torch.long)
            has_any_pad = False

        from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_varlen_forward
        from flash_attn.bert_padding import unpad_input, pad_input

        def _fa_with_lse(q, k, v, scale, causal=False, window_size=(-1, -1)):
            """FlashAttention forward returning (output, lse) without materializing attn probs."""
            out, lse, _, _ = _flash_attn_forward(
                q, k, v,
                dropout_p=0.0, softmax_scale=scale, causal=causal,
                window_size_left=window_size[0], window_size_right=window_size[1],
                softcap=0.0, alibi_slopes=None, return_softmax=False,
            )
            return out, lse

        def _fa_varlen_with_lse(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal=False, window_size=(-1, -1)):
            """Varlen FA: q, k, v are packed (total, H, D); pads are excluded entirely."""
            out, lse, _, _ = _flash_attn_varlen_forward(
                q, k, v, cu_q, cu_k, max_q, max_k,
                dropout_p=0.0, softmax_scale=scale, causal=causal,
                window_size_left=window_size[0], window_size_right=window_size[1],
                softcap=0.0, alibi_slopes=None, return_softmax=False,
            )
            return out, lse

        if seq_len > 1:
            # ---- PATH 2: Prefill (3-call A+B-C merge, uniform scale) ----

            # Per-sample positions. In fresh prefill, K positions = Q positions.
            q_pos_2d = position_ids.float()  # (B, S)
            if total_kv_len == seq_len:
                k_pos_2d = q_pos_2d
            else:
                # Chunked prefill with past cache: reconstruct K positions from
                # per-sample left-padding offset.
                idx_range_kc = torch.arange(total_kv_len, device=device, dtype=torch.float32)
                k_pos_2d = (idx_range_kc.unsqueeze(0) - pad_counts_kv.float().unsqueeze(-1)).clamp(min=0)

            # Dynamic Self-Extend: target phase = floor(p/G) * theta
            delta_q_pos_2d = torch.floor(q_pos_2d / G) - q_pos_2d  # (B, S)
            delta_k_pos_2d = torch.floor(k_pos_2d / G) - k_pos_2d  # (B, total_kv_len)

            q_group = correction_rotate_per_sample(query_base, delta_q_pos_2d, inv_freq, dtype)
            k_group = correction_rotate_per_sample(key_base,   delta_k_pos_2d, inv_freq, dtype)

            scale = self.scaling

            if has_any_pad:
                # ---- Varlen path: batch has left-padding -> pack to skip pads ----
                # Build Q-side and K-side real-token masks from pad_counts_kv.
                # In fresh prefill Q and K share the same seq_len and pad pattern.
                seq_idx = torch.arange(seq_len, device=device).unsqueeze(0)
                attn_mask_q = (seq_idx >= pad_counts_kv.unsqueeze(-1))  # (B, S)

                q_base_bshd  = query_base.transpose(1, 2)
                q_group_bshd = q_group.transpose(1, 2)
                v_bshd       = value_states.transpose(1, 2)

                q_base_up, indices, cu_q, max_q, *_ = unpad_input(q_base_bshd, attn_mask_q)
                q_group_up, *_ = unpad_input(q_group_bshd, attn_mask_q)

                if total_kv_len == seq_len:
                    k_base_up,  *_ = unpad_input(key_base.transpose(1, 2),  attn_mask_q)
                    k_group_up, *_ = unpad_input(k_group.transpose(1, 2), attn_mask_q)
                    v_up,       *_ = unpad_input(v_bshd, attn_mask_q)
                    cu_seqlens, max_seqlen = cu_q, max_q
                    cu_k, max_k = cu_q, max_q
                else:
                    idx_k = torch.arange(total_kv_len, device=device).unsqueeze(0)
                    attn_mask_k = (idx_k >= pad_counts_kv.unsqueeze(-1))
                    k_base_up,  _, cu_k, max_k, *_ = unpad_input(key_base.transpose(1, 2), attn_mask_k)
                    k_group_up, *_ = unpad_input(k_group.transpose(1, 2), attn_mask_k)
                    v_up,       *_ = unpad_input(v_bshd, attn_mask_k)
                    cu_seqlens, max_seqlen = cu_q, max_q

                out_A, lse_A = _fa_varlen_with_lse(q_base_up,  k_base_up,  v_up, cu_q, cu_k,
                                                   max_q, max_k, scale, causal=True, window_size=(w_0, 0))
                out_B, lse_B = _fa_varlen_with_lse(q_group_up, k_group_up, v_up, cu_q, cu_k,
                                                   max_q, max_k, scale, causal=True)
                out_C, lse_C = _fa_varlen_with_lse(q_group_up, k_group_up, v_up, cu_q, cu_k,
                                                   max_q, max_k, scale, causal=True, window_size=(w_0, 0))

                # varlen out shape: (total, H, D); lse shape: (H, total)
                lse_A = lse_A.transpose(0, 1).float()  # (total, H)
                lse_B = lse_B.transpose(0, 1).float()
                lse_C = lse_C.transpose(0, 1).float()

                max_lse = torch.maximum(lse_A, torch.maximum(lse_B, lse_C))
                w_A = torch.exp(lse_A - max_lse).unsqueeze(-1)  # (total, H, 1)
                w_B = torch.exp(lse_B - max_lse).unsqueeze(-1)
                w_C = torch.exp(lse_C - max_lse).unsqueeze(-1)

                w_dist = (w_B - w_C).clamp(min=0.0)
                out_dist_unnorm = w_B * out_B.float() - w_C * out_C.float()
                out_dist_unnorm = torch.where(w_dist > 1e-6, out_dist_unnorm, torch.zeros_like(out_dist_unnorm))

                num = w_A * out_A.float() + out_dist_unnorm
                den = torch.clamp_min(w_A + w_dist, 1e-7)
                attn_output_up = (num / den).to(dtype)  # (total, H, D)

                # Re-pad to (B, S, H, D)
                attn_output = pad_input(attn_output_up, indices, B, seq_len)  # (B, S, H, D)
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, None

            # ---- Non-padded path (single sample or all samples same length) ----
            q_base_fa  = query_base.transpose(1, 2)
            q_group_fa = q_group.transpose(1, 2)
            k_base_fa  = key_base.transpose(1, 2)
            k_group_fa = k_group.transpose(1, 2)
            v_fa       = value_states.transpose(1, 2)

            out_A, lse_A = _fa_with_lse(q_base_fa,  k_base_fa,  v_fa, scale, causal=True, window_size=(w_0, 0))
            out_B, lse_B = _fa_with_lse(q_group_fa, k_group_fa, v_fa, scale, causal=True)
            out_C, lse_C = _fa_with_lse(q_group_fa, k_group_fa, v_fa, scale, causal=True, window_size=(w_0, 0))

            out_A = out_A.transpose(1, 2)
            out_B = out_B.transpose(1, 2)
            out_C = out_C.transpose(1, 2)

            lse_A = lse_A.float()
            lse_B = lse_B.float()
            lse_C = lse_C.float()

            max_lse = torch.maximum(lse_A, torch.maximum(lse_B, lse_C))
            w_A = torch.exp(lse_A - max_lse).unsqueeze(-1)
            w_B = torch.exp(lse_B - max_lse).unsqueeze(-1)
            w_C = torch.exp(lse_C - max_lse).unsqueeze(-1)

            w_dist = (w_B - w_C).clamp(min=0.0)
            out_dist_unnorm = w_B * out_B - w_C * out_C
            out_dist_unnorm = torch.where(w_dist > 1e-6, out_dist_unnorm, torch.zeros_like(out_dist_unnorm))

            num = w_A * out_A + out_dist_unnorm
            den = torch.clamp_min(w_A + w_dist, 1e-7)
            attn_output = (num / den).to(dtype)

            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        # ---- PATH 3: Decode (seq_len == 1) ----
        # Q is (B, 1, H, D) — all real. Only K (past cache) has potential pads
        # for samples that had left-padding during prefill. Use varlen FA on
        # the distant branch; nearby is past the pad region so regular FA is OK.

        max_q_pos_batch = int(position_ids.max().item())
        boundary = max(max_q_pos_batch - w_0, 0)

        # Grouped query per sample
        q_pos_2d = position_ids.float()  # (B, 1)
        delta_q_2d = torch.floor(q_pos_2d / G) - q_pos_2d
        q_group = correction_rotate_per_sample(query_base, delta_q_2d, inv_freq, dtype)

        # FA layout: (B, H, S, D) -> (B, S, H, D)
        q_base_fa  = query_base.transpose(1, 2)
        q_group_fa = q_group.transpose(1, 2)

        if boundary == 0:
            # No distant keys — pure base attention. If any sample has pads in
            # its K cache (edge case: real_len_b < w_0 AND L_curr_batch > w), use
            # varlen; otherwise the regular path is fine.
            k_fa = key_base.transpose(1, 2)
            v_fa = value_states.transpose(1, 2)
            if has_any_pad:
                idx_kv = torch.arange(total_kv_len, device=device)
                mask_kv = (idx_kv.unsqueeze(0) >= pad_counts_kv.unsqueeze(-1))
                k_up, _, cu_k, max_k, *_ = unpad_input(k_fa, mask_kv)
                v_up, *_ = unpad_input(v_fa, mask_kv)
                q_packed = q_base_fa.reshape(B, q_base_fa.shape[2], q_base_fa.shape[3])
                cu_q = torch.arange(B + 1, device=device, dtype=torch.int32)
                out_nearby, _ = _fa_varlen_with_lse(
                    q_packed, k_up, v_up, cu_q, cu_k, 1, max_k, self.scaling, causal=False,
                )
                attn_output = out_nearby.reshape(B, 1, q_base_fa.shape[2], q_base_fa.shape[3])
            else:
                out_nearby, _ = _fa_with_lse(q_base_fa, k_fa, v_fa, self.scaling, causal=False)
                attn_output = out_nearby
        else:
            k_nearby_base = key_base[:, :, boundary:, :]
            v_nearby_base = value_states[:, :, boundary:, :]
            k_distant_base = key_base[:, :, :boundary, :]
            v_distant_base = value_states[:, :, :boundary, :]

            # Per-sample K positions for correction (pads mapped to real pos 0 but skipped in FA)
            pad_counts_f = pad_counts_kv.float()
            idx_range_d = torch.arange(boundary, device=device, dtype=torch.float32)
            k_pos_distant_2d = (idx_range_d.unsqueeze(0) - pad_counts_f.unsqueeze(-1)).clamp(min=0)
            delta_k_distant_2d = torch.floor(k_pos_distant_2d / G) - k_pos_distant_2d
            k_distant_group = correction_rotate_per_sample(
                k_distant_base, delta_k_distant_2d, inv_freq, dtype
            )

            k_nearby           = k_nearby_base.transpose(1, 2)
            v_nearby           = v_nearby_base.transpose(1, 2)
            k_distant_group_fa = k_distant_group.transpose(1, 2)
            v_distant          = v_distant_base.transpose(1, 2)

            scale = self.scaling

            # Call 1: base near. Nearby covers batch_idx [boundary, total_kv_len).
            # Pads only intrude here if pad_b > boundary (i.e., real_len_b < w_0);
            # detect this and use varlen to skip pads. Otherwise regular FA.
            nearby_len = total_kv_len - boundary
            nearby_has_pad = has_any_pad and bool((pad_counts_kv > boundary).any().item())
            if nearby_has_pad:
                # Nearby-region pad count per sample = max(pad_b - boundary, 0)
                pad_in_near = (pad_counts_kv - boundary).clamp(min=0)
                idx_n = torch.arange(nearby_len, device=device)
                mask_n = (idx_n.unsqueeze(0) >= pad_in_near.unsqueeze(-1))
                k_n_up, _, cu_n, max_n, *_ = unpad_input(k_nearby, mask_n)
                v_n_up, *_ = unpad_input(v_nearby, mask_n)
                q_packed_base = q_base_fa.reshape(B, q_base_fa.shape[2], q_base_fa.shape[3])
                cu_q_near = torch.arange(B + 1, device=device, dtype=torch.int32)
                out_nearby, lse_nearby = _fa_varlen_with_lse(
                    q_packed_base, k_n_up, v_n_up, cu_q_near, cu_n, 1, max_n, scale, causal=False,
                )
                out_nearby = out_nearby.reshape(B, 1, q_base_fa.shape[2], q_base_fa.shape[3])
                # varlen lse shape (H, B) -> (B, H, 1) to match regular FA format downstream
                lse_nearby = lse_nearby.transpose(0, 1).unsqueeze(-1)
            else:
                out_nearby, lse_nearby = _fa_with_lse(
                    q_base_fa, k_nearby, v_nearby, scale, causal=False,
                )

            # Call 2: grouped distant. If any sample has pad_count > 0, use varlen
            # to strictly exclude pad keys from the softmax denominator.
            if has_any_pad:
                idx_kv = torch.arange(boundary, device=device)
                distant_mask_real = (idx_kv.unsqueeze(0) >= pad_counts_kv.unsqueeze(-1))  # (B, boundary)
                k_distant_up, _k_idx, cu_seqlens_kd, max_kd, *_ = unpad_input(k_distant_group_fa, distant_mask_real)
                v_distant_up, *_ = unpad_input(v_distant, distant_mask_real)
                q_group_packed = q_group_fa.reshape(B, q_group_fa.shape[2], q_group_fa.shape[3])
                cu_seqlens_q_ = torch.arange(B + 1, device=device, dtype=torch.int32)
                if max_kd > 0:
                    out_distant, lse_distant = _fa_varlen_with_lse(
                        q_group_packed, k_distant_up, v_distant_up,
                        cu_seqlens_q_, cu_seqlens_kd, 1, max_kd, scale, causal=False,
                    )
                    out_distant = out_distant.reshape(B, 1, q_group_fa.shape[2], q_group_fa.shape[3])
                    lse_distant = lse_distant.transpose(0, 1).unsqueeze(-1)  # (B, H, 1)
                else:
                    # All samples' distant region is entirely pads — no distant contribution.
                    out_distant = torch.zeros(B, 1, q_group_fa.shape[2], q_group_fa.shape[3],
                                              device=device, dtype=dtype)
                    lse_distant = torch.full((B, q_group_fa.shape[2], 1), float('-inf'),
                                             device=device, dtype=torch.float32)
            else:
                out_distant, lse_distant = _fa_with_lse(
                    q_group_fa, k_distant_group_fa, v_distant, scale, causal=False,
                )

            lse_n = lse_nearby.float().squeeze(-1)
            lse_d = lse_distant.float().squeeze(-1)
            max_lse = torch.maximum(lse_n, lse_d)
            exp_n = torch.exp(lse_n - max_lse)
            exp_d = torch.exp(lse_d - max_lse)
            denom = torch.clamp_min(exp_n + exp_d, 1e-7)
            w_n = (exp_n / denom)[:, None, :, None].to(out_nearby.dtype)
            w_d = (exp_d / denom)[:, None, :, None].to(out_distant.dtype)
            attn_output = w_n * out_nearby + w_d * out_distant

        # (B, 1, num_heads, D) -> (B, 1, hidden_size)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


# ---------------------------------------------------------------------------
# Decoder layer — swap in JetLong v1 attention
# ---------------------------------------------------------------------------

class Qwen3JetLongDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3JetLongAttention(config=config, layer_idx=layer_idx)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen3JetLongModel(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3JetLongDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3JetLongRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

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
        **kwargs: Unpack[TransformersKwargs],
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
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

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
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
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
# Causal LM head
# ---------------------------------------------------------------------------

class Qwen3JetLongForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3JetLongModel(config)
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

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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
        **kwargs: Unpack[TransformersKwargs],
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
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["Qwen3JetLongForCausalLM", "Qwen3JetLongModel", "Qwen3JetLongRotaryEmbedding"]
