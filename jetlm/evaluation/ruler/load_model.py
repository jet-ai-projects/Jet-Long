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

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from jetlm.utils import master_print


def load_model_for_eval(model_name_or_path, attn_implementation=None):
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    if attn_implementation is None:
        # jetlong_fused runs attention through the fused CuTe kernel and is loaded with
        # SDPA for the within-window path; every other method uses FlashAttention-2.
        architectures = getattr(cfg, "architectures", None) or []
        attn_implementation = "sdpa" if any("JetLongFused" in a for a in architectures) else "flash_attention_2"

    master_print(f"[LOADING MODEL] {model_name_or_path} (attn_implementation={attn_implementation})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=cfg,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
    )

    model.cuda()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
