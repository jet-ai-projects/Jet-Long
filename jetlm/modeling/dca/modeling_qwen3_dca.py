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

# Dual Chunk Attention (DCA) for Qwen3
#
# Training-free context window extension via three-region attention:
#   1. Intra-chunk:  queries attend to keys in the same chunk
#                    (positions cyclic within chunk size s)
#   2. Successive:   queries attend to keys in the immediately previous chunk
#                    (queries get position s..s+w-1 then clamp at s+w)
#   3. Inter-chunk:  queries attend to keys in chunks 2+ steps back
#                    (queries pinned at constant s+w)
#
# Reference: "Training-Free Long-Context Scaling of Large Language Models"
#            An et al., ICML 2024.  https://arxiv.org/abs/2402.17463
# Code reference: https://github.com/HKUNLP/ChunkLlama
#                 (chunkllama_attn_replace.py, chunkqwen_attn_replace.py)
#
# Notation: matches the paper.
#   s = chunk_size       (modulo divisor; paper's "chunk size")
#   w = local_window     (ramp width preserving locality across chunks)
#   s + w                (saturation value for clamped P_q_succ tail and constant P_q_inter)
#   c = pretraining context length (32768 for Qwen3-1.7B / 4B Base)
#
# Default values (s=20480, w=4096 for c=32768) come from the released ref
# code, NOT the paper text. The paper §4.1 claims s = 3c/4, but
# chunkllama_attn_replace.py sets `chunk_size = 3c/4` and
# `local_window = c/8`, then computes the actual modulo divisor as
# `chunk_len = chunk_size - local_window = 5c/8`. Paper's `s` is this
# `chunk_len`, i.e. 5c/8 in practice, not the claimed 3c/4. We follow the
# released code's 5c/8.
#
# Note: paper text uses c-1 as the saturation, but both released ref impls
# (Llama and Qwen) clamp at s+w. We follow the released code here too.

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
    from flash_attn import flash_attn_func as _fa_flash_attn_func
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
    """Apply rotary embedding to a single tensor (Q or K).

    x:   [B, H, T, D]
    cos: [B, T, D]   (or broadcastable)
    sin: [B, T, D]
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _get_dca_cfg(config):
    cfg = getattr(config, "dca", None)
    if cfg is None and hasattr(config, "to_dict"):
        cfg = config.to_dict().get("dca", None)
    return cfg


def _do_fa(q, k, v, causal):
    """Wrapper around flash_attn_func that takes [B, H, T, D] in/out and also
    returns the per-row LSE [B, H, T] needed for DCA's cross-region merge.
    """
    out, lse, _ = _fa_flash_attn_func(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        causal=causal,
        return_attn_probs=True,
    )
    return out.transpose(1, 2), lse


def _lse_merge(outs, lses):
    """Numerically-stable softmax-of-softmaxes across disjoint key regions.

    Equivalent to a single softmax over the union of region keys, given each
    region's already-softmaxed output and its log-sum-exp.

    outs: list of [B, H, T, D]
    lses: list of [B, H, T]
    returns:    [B, H, T, D]
    """
    stacked_outs = torch.stack(outs, dim=0)              # [N, B, H, T, D]
    stacked_lses = torch.stack(lses, dim=0).to(torch.float32)  # [N, B, H, T]
    max_lse = torch.max(stacked_lses, dim=0, keepdim=True).values
    weights = torch.exp(stacked_lses - max_lse)
    weights = weights / weights.sum(dim=0, keepdim=True)
    weights = weights.to(stacked_outs.dtype)
    return (stacked_outs * weights.unsqueeze(-1)).sum(dim=0)


def _infer_real_lengths_and_offsets(attention_mask, bsz, kv_len, device):
    """Per-sample (real_len, offset) regardless of padding side.

    A key column is "real" iff some query is allowed to attend to it, so this
    handles 2D padding masks (1=real, 0=pad) and 4D additive causal+padding
    masks (0=allowed, -inf=blocked) uniformly.
    """
    if attention_mask is None:
        real_lens = torch.full((bsz,), kv_len, dtype=torch.long, device=device)
        offsets = torch.zeros((bsz,), dtype=torch.long, device=device)
        return real_lens, offsets
    if attention_mask.dim() == 2:
        valid = attention_mask.bool()
    elif attention_mask.dim() == 4:
        valid = (attention_mask[:, 0, :, :] > -1e4).any(dim=-2)
    else:
        raise ValueError(f"Unexpected attention_mask shape {tuple(attention_mask.shape)}")
    real_lens = valid.sum(dim=-1).to(torch.long)
    offsets = valid.to(torch.uint8).argmax(dim=-1).to(torch.long)
    return real_lens, offsets


# ---------------------------------------------------------------------------
# DCA Attention
# ---------------------------------------------------------------------------

class Qwen3DcaAttention(nn.Module):
    """Multi-headed attention with Dual Chunk Attention.

    Three execution paths:
      * kv_seq_len <= s:                   _standard_forward (DCA inactive)
      * q_len > 1 (prefill, dca active):   _dca_prefill
      * q_len == 1 (decode, dca active):   _dca_decode
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

        dca_cfg = _get_dca_cfg(config)
        self.dca_enabled = dca_cfg is not None
        if self.dca_enabled:
            self.chunk_size = int(dca_cfg["chunk_size"])      # s
            self.local_window = int(dca_cfg["local_window"])  # w
            self.saturation = self.chunk_size + self.local_window  # s + w
            self.use_flash_attn = _FA2_AVAILABLE

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project + norm + view (Qwen3 ordering: norm BEFORE RoPE).
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        bsz, _, q_len, _ = query_states.shape
        cos, sin = position_embeddings if position_embeddings is not None else (None, None)

        past_kv_len = 0
        if past_key_values is not None:
            past_kv_len = past_key_values.get_seq_length(self.layer_idx)
        kv_seq_len = past_kv_len + q_len

        if not self.dca_enabled:
            if cos is None:
                dummy = value_states[:, :, :1, :]
                cos, sin = self._rotary_emb(dummy, position_ids)
            return self._standard_forward(
                query_states, key_states, value_states,
                cos, sin, input_shape, attention_mask,
                past_key_values, cache_position, **kwargs,
            )

        # DCA inactive when entire context fits in one chunk: standard path
        # is bit-equivalent because (pos mod s) == pos for pos < s.
        if kv_seq_len <= self.chunk_size:
            if cos is None:
                dummy = value_states[:, :, :1, :]
                cos, sin = self._rotary_emb(dummy, position_ids)
            return self._standard_forward(
                query_states, key_states, value_states,
                cos, sin, input_shape, attention_mask,
                past_key_values, cache_position, **kwargs,
            )

        # DCA active.
        if q_len == 1:
            return self._dca_decode(
                query_states, key_states, value_states,
                input_shape, attention_mask,
                past_key_values, cache_position,
                bsz, position_ids=position_ids,
            )
        return self._dca_prefill(
            query_states, key_states, value_states,
            input_shape, attention_mask,
            past_key_values, cache_position,
            bsz, q_len, position_ids=position_ids,
        )

    # -----------------------------------------------------------------------
    # Path 1: standard (DCA inactive)
    # -----------------------------------------------------------------------

    def _standard_forward(
        self, query_states, key_states, value_states,
        cos, sin, input_shape, attention_mask,
        past_key_values, cache_position, **kwargs,
    ):
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

    # -----------------------------------------------------------------------
    # Path 2: DCA prefill (q_len > 1, kv_seq_len > s)
    # -----------------------------------------------------------------------

    def _dca_prefill(
        self, query_states, key_states, value_states,
        input_shape, attention_mask,
        past_key_values, cache_position,
        bsz, q_len, position_ids=None,
    ):
        """Full prefill with DCA. Per-sample chunk loop with up to 3 FA2 calls
        per chunk, merged via LSE.

        Requires past_kv_len == 0 (full prefill) -- chunked prefill with prior
        cache content is not supported in this version.
        """
        s = self.chunk_size
        w = self.local_window
        sw = self.saturation
        device = query_states.device

        past_kv_len = 0
        if past_key_values is not None:
            past_kv_len = past_key_values.get_seq_length(self.layer_idx)
        if past_kv_len != 0:
            raise NotImplementedError(
                "DCA chunked prefill (past_kv_len > 0 with q_len > 1) is not "
                f"supported; got past_kv_len={past_kv_len}, q_len={q_len}."
            )

        full_kv_len = q_len  # full prefill
        real_lens, offsets = _infer_real_lengths_and_offsets(
            attention_mask, bsz, full_kv_len, device
        )
        real_lens_cpu = real_lens.tolist()
        offsets_cpu = offsets.tolist()

        # Real-position-mod-s for every cache slot (B, T).
        # Pad slots get clamped to 0; their K is unused (masked out by per-sample
        # slicing) but we still need a defined value to populate the cache.
        arange_t = torch.arange(q_len, device=device).unsqueeze(0)  # [1, T]
        offsets_t = offsets.unsqueeze(-1)                            # [B, 1]
        real_pos = arange_t - offsets_t                              # [B, T] (negative at pad)
        real_pos_clamped = real_pos.clamp(min=0)
        pos_mod_s = real_pos_clamped % s                             # [B, T]

        dummy = value_states[:, :, :1, :]

        # Rotate K with P_k = pos mod s.
        k_cos, k_sin = self._rotary_emb(dummy, pos_mod_s)
        key_states_rot = apply_rotary_pos_emb_q_or_k(key_states, k_cos, k_sin)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states_cached, value_states_cached = past_key_values.update(
                key_states_rot, value_states, self.layer_idx, cache_kwargs
            )
        else:
            key_states_cached = key_states_rot
            value_states_cached = value_states

        # Three Q rotations (computed for ALL positions, including pad slots
        # which won't be used). For each query at chunk-local position p:
        #   intra: rotate at p
        #   succ:  rotate at min(s + p, s + w)
        #   inter: rotate at constant s + w
        intra_pos = pos_mod_s
        succ_pos = torch.clamp(s + pos_mod_s, max=sw)
        inter_pos = torch.full_like(pos_mod_s, sw)

        q_intra_cos, q_intra_sin = self._rotary_emb(dummy, intra_pos)
        q_succ_cos, q_succ_sin = self._rotary_emb(dummy, succ_pos)
        q_inter_cos, q_inter_sin = self._rotary_emb(dummy, inter_pos)

        q_intra = apply_rotary_pos_emb_q_or_k(query_states, q_intra_cos, q_intra_sin)
        q_succ = apply_rotary_pos_emb_q_or_k(query_states, q_succ_cos, q_succ_sin)
        q_inter = apply_rotary_pos_emb_q_or_k(query_states, q_inter_cos, q_inter_sin)

        # Per-sample chunk loop. Each sample's real content lives in
        # cache slots [offset, offset + real_len); index into the full tensors
        # at those slots.
        outputs = []
        for b in range(bsz):
            rl_b = real_lens_cpu[b]
            off_b = offsets_cpu[b]

            # Slice this sample's real content (the rotated K/V we just computed
            # are equivalent at real positions to the cache contents -- use the
            # local tensors to avoid a re-fetch).
            q_b_intra = q_intra[b:b+1, :, off_b:off_b + rl_b]
            q_b_succ = q_succ[b:b+1, :, off_b:off_b + rl_b]
            q_b_inter = q_inter[b:b+1, :, off_b:off_b + rl_b]
            k_b = key_states_rot[b:b+1, :, off_b:off_b + rl_b]
            v_b = value_states[b:b+1, :, off_b:off_b + rl_b]

            if rl_b <= s:
                # Sample below DCA threshold: pure intra (== standard FA).
                out_b, _ = _do_fa(q_b_intra, k_b, v_b, causal=True)
            else:
                out_b = self._chunked_attention_single_sample(
                    q_b_intra, q_b_succ, q_b_inter, k_b, v_b, rl_b, s,
                )

            # Pad back to [1, Hq, q_len, D].
            out_padded = torch.zeros_like(query_states[b:b+1])
            out_padded[:, :, off_b:off_b + rl_b, :] = out_b
            outputs.append(out_padded)

        attn_output = torch.cat(outputs, dim=0)  # [B, Hq, q_len, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def _chunked_attention_single_sample(
        self, q_intra, q_succ, q_inter, k_rot, v, rl, s,
    ):
        """One sample's DCA prefill (chunk loop + LSE merge)."""
        # First chunk [0, s): intra only (causal).
        chunk_outputs = []
        out0, _ = _do_fa(q_intra[:, :, :s], k_rot[:, :, :s], v[:, :, :s], causal=True)
        chunk_outputs.append(out0)

        # Subsequent chunks [s, 2s), [2s, 3s), ...
        chunk_idx = 1
        while chunk_idx * s < rl:
            begin = chunk_idx * s
            end = min(begin + s, rl)
            prev_begin = begin - s

            # Intra: this chunk's keys, causal.
            out_intra, lse_intra = _do_fa(
                q_intra[:, :, begin:end],
                k_rot[:, :, begin:end],
                v[:, :, begin:end],
                causal=True,
            )
            # Succ: previous chunk's keys, full attention (no causal mask
            # needed -- all prev keys are strictly before all current queries).
            out_succ, lse_succ = _do_fa(
                q_succ[:, :, begin:end],
                k_rot[:, :, prev_begin:begin],
                v[:, :, prev_begin:begin],
                causal=False,
            )

            region_outs = [out_intra, out_succ]
            region_lses = [lse_intra, lse_succ]

            # Inter: all keys before previous chunk (chunks 2..n-2).
            if prev_begin > 0:
                out_inter, lse_inter = _do_fa(
                    q_inter[:, :, begin:end],
                    k_rot[:, :, :prev_begin],
                    v[:, :, :prev_begin],
                    causal=False,
                )
                region_outs.append(out_inter)
                region_lses.append(lse_inter)

            chunk_outputs.append(_lse_merge(region_outs, region_lses))
            chunk_idx += 1

        return torch.cat(chunk_outputs, dim=-2)  # concat along seq dim

    # -----------------------------------------------------------------------
    # Path 3: DCA decode (q_len == 1, kv_seq_len > s)
    # -----------------------------------------------------------------------

    def _dca_decode(
        self, query_states, key_states_new, value_states_new,
        input_shape, attention_mask,
        past_key_values, cache_position,
        bsz, position_ids=None,
    ):
        """One-token decode. Per-sample three-slice eager attention.

        Each sample may be at a different real position (left-padded batches),
        so chunk arithmetic is computed per-sample.
        """
        s = self.chunk_size
        w = self.local_window
        sw = self.saturation
        device = query_states.device
        dummy = value_states_new[:, :, :1, :]

        # Per-sample real position of the current query.
        # position_ids is HF's authoritative source: it accounts for left-padding
        # in prepare_inputs_for_generation.
        if position_ids is None:
            raise ValueError("DCA decode requires position_ids.")
        q_pos_per_sample = position_ids[:, 0].to(device=device, dtype=torch.long)  # [B]
        q_pos_mod_s = (q_pos_per_sample % s).unsqueeze(-1)  # [B, 1]

        # Rotate the new K at (pos mod s).
        k_cos_new, k_sin_new = self._rotary_emb(dummy, q_pos_mod_s)
        key_states_rotated = apply_rotary_pos_emb_q_or_k(
            key_states_new, k_cos_new, k_sin_new
        )

        # Update cache with rotated K.
        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states_cached, value_states_cached = past_key_values.update(
                key_states_rotated, value_states_new, self.layer_idx, cache_kwargs
            )
        else:
            key_states_cached = key_states_rotated
            value_states_cached = value_states_new

        full_kv_len = key_states_cached.shape[-2]
        real_lens, offsets = _infer_real_lengths_and_offsets(
            attention_mask, bsz, full_kv_len, device
        )
        real_lens_cpu = real_lens.tolist()
        offsets_cpu = offsets.tolist()

        # Three Q rotations (per-sample because q_pos_mod_s differs).
        intra_pos = q_pos_mod_s
        succ_pos = torch.clamp(s + intra_pos, max=sw)
        inter_pos = torch.full_like(intra_pos, sw)

        q_intra_cos, q_intra_sin = self._rotary_emb(dummy, intra_pos)
        q_succ_cos, q_succ_sin = self._rotary_emb(dummy, succ_pos)
        q_inter_cos, q_inter_sin = self._rotary_emb(dummy, inter_pos)

        q_intra = apply_rotary_pos_emb_q_or_k(query_states, q_intra_cos, q_intra_sin)
        q_succ = apply_rotary_pos_emb_q_or_k(query_states, q_succ_cos, q_succ_sin)
        q_inter = apply_rotary_pos_emb_q_or_k(query_states, q_inter_cos, q_inter_sin)

        # Expand K/V heads for GQA (cheap at q_len=1).
        k_expanded = repeat_kv(key_states_cached, self.num_key_value_groups)
        v_expanded = repeat_kv(value_states_cached, self.num_key_value_groups)

        outputs = []
        for b in range(bsz):
            rl_b = real_lens_cpu[b]
            off_b = offsets_cpu[b]
            t_b = rl_b - 1               # real position of the current query
            n_b = t_b // s               # number of complete prior chunks

            # Convert real-index slices to absolute cache-index slices.
            score_list = []
            v_list = []

            if n_b >= 2:
                inter_real_start = 0
                inter_real_end = s * (n_b - 1)
                k_inter = k_expanded[b:b+1, :, off_b + inter_real_start : off_b + inter_real_end, :]
                v_inter = v_expanded[b:b+1, :, off_b + inter_real_start : off_b + inter_real_end, :]
                score_inter = torch.matmul(q_inter[b:b+1], k_inter.transpose(-1, -2)) * self.scaling
                score_list.append(score_inter)
                v_list.append(v_inter)

            if n_b >= 1:
                succ_real_start = s * (n_b - 1)
                succ_real_end = s * n_b
                k_succ = k_expanded[b:b+1, :, off_b + succ_real_start : off_b + succ_real_end, :]
                v_succ = v_expanded[b:b+1, :, off_b + succ_real_start : off_b + succ_real_end, :]
                score_succ = torch.matmul(q_succ[b:b+1], k_succ.transpose(-1, -2)) * self.scaling
                score_list.append(score_succ)
                v_list.append(v_succ)

            # Intra: current chunk keys [s*n_b, t_b + 1).
            intra_real_start = s * n_b
            intra_real_end = t_b + 1
            k_intra = k_expanded[b:b+1, :, off_b + intra_real_start : off_b + intra_real_end, :]
            v_intra = v_expanded[b:b+1, :, off_b + intra_real_start : off_b + intra_real_end, :]
            score_intra = torch.matmul(q_intra[b:b+1], k_intra.transpose(-1, -2)) * self.scaling
            score_list.append(score_intra)
            v_list.append(v_intra)

            assert all(s_.shape[-1] > 0 for s_ in score_list), \
                f"Empty slice in DCA decode (sample {b}, t={t_b}, n={n_b})"

            scores = torch.cat(score_list, dim=-1)  # [1, Hq, 1, sum_k]
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            v_cat = torch.cat(v_list, dim=-2)
            out_b = torch.matmul(probs, v_cat)
            outputs.append(out_b)

        attn_output = torch.cat(outputs, dim=0)  # [B, Hq, 1, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class Qwen3DcaDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DcaAttention(config=config, layer_idx=layer_idx)
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

class Qwen3DcaModel(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DcaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        _ltypes = getattr(config, "layer_types", None) or []
        self.has_sliding_layers = "sliding_attention" in _ltypes

        # DCA attention computes its own RoPE rotations on demand; share the
        # rotary_emb instance so each layer can call it with arbitrary positions.
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
        # DCA layers compute their own RoPE; skip the model-level call when
        # all layers are DCA-enabled.
        if all(l.self_attn.dca_enabled for l in self.layers):
            position_embeddings = None
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping.get(
                    decoder_layer.attention_type,
                    causal_mask_mapping.get("full_attention"),
                ),
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

class Qwen3DcaForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3DcaModel(config)
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
