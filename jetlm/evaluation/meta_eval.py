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

import os
import json
import argparse

from jetlm.utils import (
    is_master,
    dist_init,
    dist_print,
    dist_close,
    dist_barrier,
    build_config,
    get_device,
    master_print, 
    extract_log_name,
    get_filename_without_ext,
    check_complete_marker,
)
from jetlm.evaluation.lm_eval_harness import LMEvalWrapper, LMHarnessEvaluator
from jetlm.evaluation.ruler import RULEREvaluator, load_model_for_eval
from jetlm.modeling import HFModel
from jetlm.constant import DEFAULT_EVAL_VERSION_STR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="jetlm-ai/JetLM-2B", help="Path to the model configuration or checkpoint.")
    parser.add_argument("--run_name", type=str, default="", help="Run name for wandb")
    parser.add_argument("--eval_config", type=str, default="jetlm/evaluation/configs/ruler.yaml", help="Path to the evaluation configuration file.")
    parser.add_argument("--output_dir", type=str, default="results/eval/", help="Directory to save evaluation results.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length for evaluation.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Maximum number of new tokens to generate during evaluation.")
    parser.add_argument("--prefill_chunk_size", type=int, default=None, help="Chunk size for prefill during evaluation.")
    parser.add_argument("--generation_num_chunks", type=int, default=1, help="Number of chunks for generation during evaluation.")
    parser.add_argument("--no_runtime_cache", action="store_true", help="Whether to disable runtime cache during evaluation.")
    parser.add_argument("--no_cache_requests", action="store_true", help="Whether to cache requests during evaluation.")
    parser.add_argument("--max_test_num", type=int, default=None, help="Maximum number of tests to run during evaluation.")
    parser.add_argument("--ckpt_step", type=int, default=-1, help="Number of steps trained, -1 for last step")
    parser.add_argument("--version_str", type=str, default=DEFAULT_EVAL_VERSION_STR, help="version string for evluation")
    
    # enable code eval
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    
    # parse args
    args, opt = parser.parse_known_args()
    VERSION_STR = args.version_str

    # setup dist env
    dist_init(gpu=None, cudnn_benchmark=False)

    dist_print("Additional Arguments: ", json.dumps(opt, indent=2))

    eval_config = build_config(args.eval_config)

    model, tokenizer = load_model_for_eval(model_name_or_path=args.model_name_or_path)

    # For chat models with thinking mode (e.g. Qwen3), disable thinking by default
    # so that lm_eval gets direct answers instead of <think>...</think> blocks
    if eval_config.get("apply_chat_template", False) and tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "enable_thinking is not defined or enable_thinking is false",
        )

    model = model.eval().cuda()
    master_print(model.device)
    model = HFModel(model)
    
    wandb_log_name = extract_log_name(model_path=args.model_name_or_path) if args.run_name=='' else args.run_name
    task_abbrev = get_filename_without_ext(args.eval_config)
    output_dir = os.path.join(args.output_dir, eval_config["name"])
    # results/eval/fsdp_mha+emb+norms_v1f_c32768_b32_s42_lul65536/ruler
    # results/eval/fsdp_mha+emb+norms_v1f_c32768_b32_s42_lul65536/ruler
    if is_master():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    res_path = os.path.join(output_dir, f"{task_abbrev}_results.json")
    dist_barrier()
    
    if not check_complete_marker(res_path, version_str=VERSION_STR, clean_dir=False):
        if eval_config["type"] == "lm_eval_harness":
            max_seq_len = args.max_length if args.max_length is not None else eval_config.get("max_length", None)
            assert max_seq_len is not None, "max_length must be specified either in the config or as an argument."
            max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else eval_config.get("max_new_tokens", 256)

            eval_wrapper = LMEvalWrapper(model, tokenizer, max_seq_len=max_seq_len,
                                        max_new_tokens=max_new_tokens,
                                        device=get_device(model),
                                        prefill_chunk_size=args.prefill_chunk_size,
                                        generation_num_chunks=args.generation_num_chunks,
                                        batch_size=args.eval_batch_size, 
                                        use_runtime_cache=not args.no_runtime_cache)
        
            evaluator = LMHarnessEvaluator(eval_wrapper, 
                                        eval_config["tasks"],
                                        subtasks=eval_config.get("subtasks", None),
                                        cache_requests=not args.no_cache_requests,
                                        max_test_num=args.max_test_num,
                                        apply_chat_template=eval_config.get("apply_chat_template", False),
                                        fewshot_as_multiturn=eval_config.get("fewshot_as_multiturn", False),
                                        system_instruction=eval_config.get("system_instruction", None),
                                        keys_to_remove_in_save=eval_config.get("keys_to_remove_in_save", None),
                                        wandb_log_name=f"{wandb_log_name}_full",
                                        version_str=VERSION_STR)
            
            
        elif eval_config["type"] == "ruler":
            evaluator = RULEREvaluator(model, tokenizer, eval_config,
                                    batch_size=args.eval_batch_size,
                                    prefill_chunk_size=args.prefill_chunk_size,
                                    res_path=res_path,
                                    wandb_log_name=wandb_log_name,
                                    version_str=VERSION_STR)
        else:
            raise ValueError(f"Unsupported evaluation type: {eval_config['type']}. Supported types are 'lm_eval_harness' and 'ruler'.")
        

        results = evaluator.evaluate(full_save_dir=os.path.join(output_dir, "full_results"))
        if is_master():
            master_print(f"Evaluation results:{json.dumps(results, indent=4)}")
            with open(res_path, "w") as f:
                json.dump(results, f, indent=4)
            evaluator.finish_and_log(json_file_path=res_path, task_abbrev=task_abbrev, train_step=args.ckpt_step)
    else:
        master_print('-'*10+f"[{task_abbrev} already evaluated for {wandb_log_name}]"+'-'*10)
    dist_close()


if __name__ == "__main__":
    main()
