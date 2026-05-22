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

# JetLongFreq for Qwen3
#
# Same as v3 (Dynamic Squeeze) but with w_0=4096. YaRN softmax temperature
# (t = 0.1 * ln(s_eff) + 1) is kept in attn_scale.

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
# NTK-by-parts static ramp
# ---------------------------------------------------------------------------

def compute_yarn_ramp(dim, w, rope_theta, beta_fast=32, beta_slow=1):
    """
    Compute the static NTK-by-parts interpolation ramp.

    Returns ramp tensor of shape [dim/2] where ramp=0 means extrapolation
    (keep original frequency) and ramp=1 means full interpolation.
    The ramp bounds are computed using the pretrained window w.

    Usage with dynamic s_eff:
        theta_yarn = theta_base * (1 - ramp * (1 - 1/s_eff))
        delta = -theta_base * ramp * (1 - 1/s_eff)
    """
    base_theta = float(rope_theta)

    def find_correction_dim(num_rotations):
        return (dim * math.log(w / (num_rotations * 2 * math.pi))) / (2 * math.log(base_theta))

    low = max(math.floor(find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(beta_slow)), dim // 2 - 1)

    linear = (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 0.001)
    ramp = torch.clamp(linear, 0, 1)
    return ramp


# ---------------------------------------------------------------------------
# Rotary Embedding (JetLongFreq)
# ---------------------------------------------------------------------------

class Qwen3JetLongFreqRotaryEmbedding(nn.Module):
    """
    JetLongFreq (Naive Hard Split) embedding.

    Returns:
        If jetlong_freq disabled or L_curr <= w: (base_cos, base_sin) — 2-tuple
        If jetlong_freq enabled and L_curr > w:
            (base_cos, base_sin, delta_inv_freq, attn_scale) — 4-tuple
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

        # --- JetLongFreq configuration ---
        jetlong_freq_cfg = getattr(config, "jetlong_freq", None)
        if jetlong_freq_cfg is None:
            jetlong_freq_cfg = config.to_dict().get("jetlong_freq", None) if hasattr(config, "to_dict") else None

        self.jetlong_freq_enabled = jetlong_freq_cfg is not None

        if self.jetlong_freq_enabled:
            self._jetlong_freq_cfg = jetlong_freq_cfg
            self.jetlong_freq_w = jetlong_freq_cfg["w"]
            self.jetlong_freq_w_0 = jetlong_freq_cfg["w_0"]
            self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

            # Precompute static NTK ramp (depends only on w, theta, beta — not on s_eff)
            ramp = compute_yarn_ramp(
                self.head_dim, self.jetlong_freq_w,
                float(config.rope_parameters["rope_theta"]),
                jetlong_freq_cfg.get("yarn_beta", 32), jetlong_freq_cfg.get("yarn_alpha", 1),
            )
            self.register_buffer("ntk_ramp", ramp, persistent=False)

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

        if not self.jetlong_freq_enabled:
            return base_cos, base_sin

        # After _init_weights re-initializes buffers from meta device,
        # recompute ntk_ramp. Use a flag to ensure single recomputation.
        if not getattr(self, '_ramp_ready', False):
            cfg = self._jetlong_freq_cfg
            self.ntk_ramp = compute_yarn_ramp(
                self.head_dim, self.jetlong_freq_w,
                float(self.config.rope_parameters["rope_theta"]),
                cfg.get("yarn_beta", 32), cfg.get("yarn_alpha", 1),
            ).to(device)
            self._ramp_ready = True

        # Dynamic s_eff from current context length
        max_pos = position_ids.max().item()
        L_curr = max_pos + 1

        if L_curr <= self.jetlong_freq_w:
            # Within pretrained range: pure base
            return base_cos, base_sin

        # --- v3: Dynamic Squeeze ---
        s_eff = (L_curr - self.jetlong_freq_w_0) / (self.jetlong_freq_w - self.jetlong_freq_w_0)
        s_eff = max(s_eff, 1.0)
        factor = 1.0 - 1.0 / s_eff

        ntk_ramp = self.ntk_ramp.to(device)
        delta_inv_freq = -self.inv_freq.to(device) * ntk_ramp * factor

        # v4: w_0=4096 with YaRN temperature t = 0.1 * ln(s_eff) + 1
        t = 0.1 * math.log(s_eff) + 1.0
        attn_scale = t / math.sqrt(self.head_dim)

        return base_cos, base_sin, delta_inv_freq, attn_scale


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
    """Apply correction rotation to convert base-RoPE'd tensors toward yarn.

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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads: (batch, kv_heads, seq, dim) -> (batch, heads, seq, dim)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Attention (JetLongFreq)
# ---------------------------------------------------------------------------

class Qwen3JetLongFreqAttention(nn.Module):
    """Multi-headed attention with JetLongFreq (Naive Hard Split)."""

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

        # --- JetLongFreq parameters ---
        jetlong_freq_cfg = config.to_dict().get("jetlong_freq", None) if hasattr(config, "to_dict") else None
        self.jetlong_freq_enabled = jetlong_freq_cfg is not None
        if self.jetlong_freq_enabled:
            self.jetlong_freq_w = jetlong_freq_cfg["w"]
            self.jetlong_freq_w_0 = jetlong_freq_cfg["w_0"]

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
        is_extended = len(position_embeddings) == 4

        if not is_extended:
            # ---- PATH 1: Pure base attention (L_curr <= w or jetlong_freq disabled) ----
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

        # ---- Extended paths (JetLongFreq, L_curr > w) ----
        base_cos, base_sin, delta_inv_freq, attn_scale = position_embeddings
        w_0 = self.jetlong_freq_w_0
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Apply base RoPE to Q and K
        query_base, key_base = apply_rotary_pos_emb(
            query_states, key_states, base_cos, base_sin
        )

        # Cache base-rotated keys (invariant: cache always stores base-RoPE'd keys)
        if past_key_values is not None:
            cache_kwargs = {"sin": base_sin, "cos": base_cos, "cache_position": cache_position}
            key_base, value_states = past_key_values.update(
                key_base, value_states, self.layer_idx, cache_kwargs
            )

        max_q_pos = cache_position[-1].item() if cache_position is not None else 0
        seq_len = query_base.shape[2]
        total_kv_len = key_base.shape[2]

        from flash_attn.flash_attn_interface import _flash_attn_forward

        def _fa_with_lse(q, k, v, scale, causal=False, window_size=(-1, -1)):
            """FlashAttention forward returning (output, lse) without materializing attn probs."""
            out, lse, _, _ = _flash_attn_forward(
                q, k, v,
                dropout_p=0.0, softmax_scale=scale, causal=causal,
                window_size_left=window_size[0], window_size_right=window_size[1],
                softcap=0.0, alibi_slopes=None, return_softmax=False,
            )
            return out, lse

        if seq_len > 1:
            # ---- PATH 2: Exact dual prefill (3-call FTVS-style merge) ----

            # --- v2/v3: Continuous Phase Stitching ---
            clamped_q_pos = (cache_position.float() - w_0).clamp(min=0)
            all_k_pos = torch.arange(total_kv_len, device=device)

            # Correction-rotate in (B, H, S, D) layout (seq_dim=2)
            q_dist = correction_rotate(query_base, clamped_q_pos, delta_inv_freq, dtype, seq_dim=2)
            k_yarn = correction_rotate(key_base, all_k_pos, delta_inv_freq, dtype, seq_dim=2)

            # Transpose to FA layout: (B, H, S, D) -> (B, S, H, D)
            q_base_fa = query_base.transpose(1, 2)
            q_dist_fa = q_dist.transpose(1, 2)
            k_base_fa = key_base.transpose(1, 2)
            k_yarn_fa = k_yarn.transpose(1, 2)
            v_fa = value_states.transpose(1, 2)

            # Call A: base near (windowed)
            out_A, lse_A = _fa_with_lse(q_base_fa, k_base_fa, v_fa, attn_scale, causal=True, window_size=(w_0, 0))
            # Call B: dist full (causal)
            out_B, lse_B = _fa_with_lse(q_dist_fa, k_yarn_fa, v_fa, attn_scale, causal=True)
            # Call C: dist window (subtract double-counted local region)
            out_C, lse_C = _fa_with_lse(q_dist_fa, k_yarn_fa, v_fa, attn_scale, causal=True, window_size=(w_0, 0))

            # LSE merge (Compute in float32 for stability)
            out_A = out_A.transpose(1, 2)
            out_B = out_B.transpose(1, 2)
            out_C = out_C.transpose(1, 2)

            lse_A = lse_A.float()
            lse_B = lse_B.float()
            lse_C = lse_C.float()

            max_lse = torch.maximum(lse_A, torch.maximum(lse_B, lse_C))

            w_A = torch.exp(lse_A - max_lse).unsqueeze(-1)  # (B, H, S, 1)
            w_B = torch.exp(lse_B - max_lse).unsqueeze(-1)
            w_C = torch.exp(lse_C - max_lse).unsqueeze(-1)

            # STABILITY FIX: Filter fp16 cancellation noise
            w_dist = (w_B - w_C).clamp(min=0.0)

            # Mask unnormalized distant output where weight is numerical dust
            out_dist_unnorm = w_B * out_B - w_C * out_C
            out_dist_unnorm = torch.where(w_dist > 1e-6, out_dist_unnorm, torch.zeros_like(out_dist_unnorm))

            num = w_A * out_A + out_dist_unnorm
            den = torch.clamp_min(w_A + w_dist, 1e-7)
            attn_output = (num / den).to(dtype)

            # (B, H, S, D) -> (B, S, H, D) -> reshape
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        # ---- PATH 3: Exact dual decode (seq_len == 1) ----

        # --- v2: Continuous Phase Stitching ---
        clamped_q = max(max_q_pos - w_0, 0)
        if clamped_q > 0:
            q_delta_angles = clamped_q * delta_inv_freq.float()  # (dim/2,)
            q_delta_emb = torch.cat((q_delta_angles, q_delta_angles), dim=-1)  # (dim,)
            q_corr_cos = q_delta_emb.cos().to(dtype)[None, None, None, :]  # (1, 1, 1, dim)
            q_corr_sin = q_delta_emb.sin().to(dtype)[None, None, None, :]
            q_dist = query_base * q_corr_cos + rotate_half(query_base) * q_corr_sin
        else:
            q_dist = query_base

        # Boundary: keys [0, boundary) are distant, [boundary, total_kv_len) are nearby
        boundary = max(max_q_pos - w_0, 0)

        # FA layout: (B, H, S, D) -> (B, S, H, D)
        q_base_fa = query_base.transpose(1, 2)   # (B, 1, num_heads, D)
        q_dist_fa = q_dist.transpose(1, 2)       # (B, 1, num_heads, D)

        if boundary == 0:
            # No distant keys — pure base attention
            k_fa = key_base.transpose(1, 2)
            v_fa = value_states.transpose(1, 2)
            out_nearby, _ = _fa_with_lse(q_base_fa, k_fa, v_fa, attn_scale, causal=False)
            attn_output = out_nearby
        else:
            # Split KV cache BEFORE transpose to keep (B, H, S, D) layout
            k_nearby_base = key_base[:, :, boundary:, :]
            v_nearby_base = value_states[:, :, boundary:, :]
            k_distant_base = key_base[:, :, :boundary, :]
            v_distant_base = value_states[:, :, :boundary, :]

            # Correction-rotate distant keys in (B, H, S, D) layout (seq_dim=2)
            distant_positions = torch.arange(boundary, device=device)
            k_distant_yarn = correction_rotate(
                k_distant_base, distant_positions, delta_inv_freq, dtype, seq_dim=2
            )

            # Transpose to FA layout
            k_nearby = k_nearby_base.transpose(1, 2)
            v_nearby = v_nearby_base.transpose(1, 2)
            k_distant_yarn_fa = k_distant_yarn.transpose(1, 2)
            v_distant = v_distant_base.transpose(1, 2)

            # FA call 1: base near
            out_nearby, lse_nearby = _fa_with_lse(
                q_base_fa, k_nearby, v_nearby, attn_scale, causal=False,
            )
            # FA call 2: dist distant
            out_distant, lse_distant = _fa_with_lse(
                q_dist_fa, k_distant_yarn_fa, v_distant, attn_scale, causal=False,
            )

            # Logsumexp merge (float32 for stability)
            lse_n = lse_nearby.float().squeeze(-1)    # (B, H)
            lse_d = lse_distant.float().squeeze(-1)    # (B, H)
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
# Decoder layer — swap in JetLongFreq attention
# ---------------------------------------------------------------------------

class Qwen3JetLongFreqDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3JetLongFreqAttention(config=config, layer_idx=layer_idx)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen3JetLongFreqModel(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3JetLongFreqDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3JetLongFreqRotaryEmbedding(config=config)
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

class Qwen3JetLongFreqForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3JetLongFreqModel(config)
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


__all__ = ["Qwen3JetLongFreqForCausalLM", "Qwen3JetLongFreqModel", "Qwen3JetLongFreqRotaryEmbedding"]
