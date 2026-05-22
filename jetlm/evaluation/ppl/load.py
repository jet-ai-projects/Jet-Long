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

"""Model loader for PG19 PPL eval.

Mirrors `jetlm/evaluation/ruler/load_model.py` but adds a `max_pe_override` knob so the
bare baseline (`Qwen3-{1.7B,4B,8B}-Base`, native max_position_embeddings=32768) can be
forwarded with sequences up to 131072. Above its native max_pe the baseline produces
extrapolated RoPE — that is the OOD-collapse signal we want to measure.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model_for_ppl(model_path: str, max_pe_override: int | None = None):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if max_pe_override is not None and getattr(cfg, "max_position_embeddings", 0) < max_pe_override:
        cfg.max_position_embeddings = max_pe_override

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=cfg,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.cuda().eval()
    return model, tokenizer
