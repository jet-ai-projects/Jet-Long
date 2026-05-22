#!/usr/bin/env python
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

import argparse
import os
import sys
import yaml
import torch

from jetlm.utils import build_config
from jetlm.evaluation.ruler import load_model_for_eval
from jetlm.evaluation.ruler.data_constants import TASKS as TASKS_DATA
from jetlm.evaluation.ruler.eval_constants import TASKS as TASKS_EVAL
from jetlm.modeling import HFModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--eval_batch_size", type=int, required=True)
    p.add_argument("--eval_config", default="jetlm/evaluation/configs/ruler.yaml")
    p.add_argument("--prefill_chunk_size", type=int, default=None)
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    total_gb = torch.cuda.get_device_properties(local_rank).total_memory / (1024 ** 3)

    eval_config = build_config(args.eval_config)
    lengths = eval_config["lengths"]
    base_len = lengths[0]

    model, tokenizer = load_model_for_eval(model_name_or_path=args.model_name_or_path)
    if eval_config.get("apply_chat_template", False) and tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "enable_thinking is not defined or enable_thinking is false",
        )
    model = model.eval().to(device)
    model = HFModel(model)

    ruler_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                             "jetlm", "evaluation", "ruler")
    with open(os.path.join(ruler_dir, "config_tasks.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)
    for _, c in tasks_customized.items():
        c.update(TASKS_DATA[c["task"]])
        c.update(TASKS_EVAL[c["task"]])

    generation_config = dict(eval_config["generation_config"])
    generation_config.pop("stop", None)
    apply_chat_template = eval_config.get("apply_chat_template", False)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stop_token_list = [
        [tokenizer.eos_token_id],
        tokenizer.encode("\n", add_special_tokens=False),
        tokenizer.encode(".\n", add_special_tokens=False),
        tokenizer.encode("\n\n", add_special_tokens=False),
        tokenizer.encode(".\n\n", add_special_tokens=False),
    ]
    max_new_tokens = max(
        tasks_customized[t["name"]]["tokens_to_generate"] for t in eval_config["tasks"]
    )

    failed = False
    for length in [lengths[0], lengths[-1]]:
        bs = max(1, args.eval_batch_size * base_len // length)
        dummy_text = tokenizer.decode([pad_id] * length, skip_special_tokens=False)
        prompts = [dummy_text for _ in range(bs)]
        if apply_chat_template:
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": pp}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ) for pp in prompts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           padding_side="left", add_special_tokens=False)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        if inputs["input_ids"].shape[1] > length:
            inputs["input_ids"] = inputs["input_ids"][:, -length:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -length:]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(local_rank)
        try:
            with torch.no_grad():
                model.generate(
                    **inputs,
                    **generation_config,
                    prefill_chunk_size=args.prefill_chunk_size,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                    stop_token_list=stop_token_list,
                )
            peak_gb = torch.cuda.max_memory_reserved(local_rank) / (1024 ** 3)
            print(f"[rank {local_rank}/{world_size}] length={length} bs={bs} "
                  f"peak_reserved={peak_gb:.2f} GiB / total={total_gb:.2f} GiB "
                  f"({100 * peak_gb / total_gb:.1f}%)", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            peak_gb = torch.cuda.max_memory_reserved(local_rank) / (1024 ** 3)
            print(f"[rank {local_rank}/{world_size}] OOM length={length} bs={bs} "
                  f"peak_reserved={peak_gb:.2f} GiB / total={total_gb:.2f} GiB: {e}",
                  file=sys.stderr, flush=True)
            failed = True
            break
        del inputs

    if failed:
        sys.exit(1)
    if local_rank == 0:
        print("SUCCESS", flush=True)


if __name__ == "__main__":
    main()
