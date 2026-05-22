# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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

from typing import Any

import numpy as np
import torch, re, os, time, random
import torch.nn as nn

from jetlm.utils.dist import get_dist_rank, sync_tensor

def val2tuple(x: tuple | list | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    if isinstance(x, (list, tuple)):
        x = list(x)
    else:
        x = [x]

    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def chunk_list_by_size(x: list, chunk_size: int) -> list[list]:
    x_chunks = []
    for i in range(0, len(x), chunk_size):
        x_chunks.append(x[i : i + chunk_size])
    return x_chunks

def str_to_dir_format(path: str):
    if not path.endswith('/'):
        return path+'/'
    return path

def chunk_list_interleaved(x: list, chunks: int) -> list[list]:
    x_chunks = []
    for i in range(chunks):
        x_chunks.append(x[i::chunks])
    return x_chunks


def get_device(model: nn.Module) -> torch.device:
    return model.parameters().__next__().device


def find_full_attn_layer_idxs(model) -> list[int]:
    """
    Return sorted indices N for modules named like:
        model.layers.N.self_attn  (where module class is JetLMAttention)
    """
    idxs = set()
    for i, name in enumerate(model.config.layer_types):  # iterates dotted paths + modules
        if name in ['attn', 'swa']:
            idxs.add(i)
    return sorted(idxs)


def numerize(x: int, unit="auto") -> str:
    # unit = None, K, M, B, auto
    if unit == "auto":
        if x < 1024:
            unit = None
        elif x < 1024**2:
            unit = "K"
        elif x < 1024**3:
            unit = "M"
        elif x < 1024**4:
            unit = "B"
        else:
            unit = "T"

    if unit is None:
        return f"{x}"
    elif unit == "K":
        return f"{x / 1024:.2f}K"
    elif unit == "M":
        return f"{x / 1024**2:.2f}M"
    elif unit == "B":
        return f"{x / 1024**3:.2f}B"
    elif unit == "T":
        return f"{x / 1024**4:.2f}T"
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    
    
def extract_log_name(model_path):
    parent_dir_name = os.path.basename(os.path.dirname(model_path))
    if parent_dir_name == "jet-ai":
        return "pretrained_base"
    else:
        return parent_dir_name

def to_uppercase_only_letters(s: str) -> str:
    return ''.join(ch.upper() if ch.isalpha() else ch for ch in s)


def get_amp_dtype(name: str) -> torch.dtype:
    amp_dict = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    if name in amp_dict:
        return amp_dict[name]
    else:
        raise ValueError(f"Unsupported amp_dtype: {name}")

def get_amp_dtype_by_bool(bf16: bool, fp16:bool) -> torch.dtype:
    if bf16: return torch.bfloat16
    elif fp16: return torch.float16
    else: return torch.float32

def seed_all(seed: int, reset: bool = False) -> int:
    if reset:
        seed = int(sync_tensor(int(time.time()), reduce="root"))
    seed += get_dist_rank()

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def print_trainable_modules(model, main_LOG=print):
    # Print header
    main_LOG(f"\033[92m{'Name':60}\033[0m | "
            f"\033[96mShape\033[0m | "
            f"\033[91mCount\033[0m")
    main_LOG("-" * 80)
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape_str = str(list(param.shape))
            main_LOG(f"\033[92m{name:60}\033[0m | "   # green
                    f"\033[96m{shape_str}\033[0m | " # cyan
                    f"\033[91m{param.numel():,}\033[0m")  # red
    trainable, total = 0, 0
    for n, p in model.named_parameters():
        ct = p.numel()
        total += ct
        if p.requires_grad:
            trainable += ct
    main_LOG(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")
