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

import os, datetime
import yaml, hashlib
import json, wandb
import math, time
import numpy as np
from tqdm import tqdm
from typing import Optional
from copy import deepcopy

import torch.nn as nn
import torch.distributed as dist

from .data_constants import TASKS as TASKS_DATA
from .eval_constants import TASKS as TASKS_EVAL
from .scorer import run_evaluation_per_task
from jetlm.utils import (
    is_master,
    get_dist_rank,
    get_dist_size,
    get_device,
    chunk_list_interleaved,
    locked_atomic_write_json, 
    locked_read_json,
    log_and_flush,
    master_print,
    clear_dir_of_res_path,
    to_uppercase_only_letters,
    create_complete_marker
)
from jetlm.constant import (
    DEFAULT_WANDB_RESULT_LOG_SUFFIX, 
    DEFAULT_VERSION_STR
)



import os

class RULEREvaluator:
    def __init__(self, 
                 model: nn.Module,
                 tokenizer,
                 eval_config: dict,
                 res_path: str = 'tmp_res.json',
                 wandb_log_name: str = 'ruler_benchmark_result',
                 batch_size: int = 1,
                 prefill_chunk_size: Optional[int] = None, 
                 version_str: Optional[int] = DEFAULT_VERSION_STR):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tasks = eval_config["tasks"]
        self.lengths = eval_config["lengths"]
        self.data_dir = eval_config["data_dir"]
        self.prefill_chunk_size = prefill_chunk_size
        self.verbose = eval_config.get("verbose", False)
        self.apply_chat_template = eval_config.get("apply_chat_template", False)

        curr_folder = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curr_folder, f"config_tasks.yaml"), "r") as f:
            self.tasks_customized: dict = yaml.safe_load(f)
        
        for _, config in self.tasks_customized.items():
            config.update(TASKS_DATA[config['task']])
            config.update(TASKS_EVAL[config['task']])

        self.generation_config = eval_config["generation_config"]
        self.stop = self.generation_config.pop('stop')
        run_name = f"TMP_{wandb_log_name}"
        self.wandb_report_name = f"EVAL_{wandb_log_name}"
        
        self.version_str = version_str
        run_id = hashlib.sha1((run_name+self.version_str).encode("utf-8")).hexdigest()
        if is_master():
            print(run_name, run_id)
            self.wandb_run =  wandb.init(name=run_name, id=run_id, resume="allow", job_type=os.environ["WANDB_RUNTYPE"]+"_tmp")
        
        self.res_path = res_path
        dist.barrier()
        if is_master():
            if os.path.isfile(self.res_path):
                master_print(f"[STATUS] Found previous run, loading...")
                self.full_results = locked_read_json(path=self.res_path)
                if 'run_id' not in self.full_results or self.full_results['run_id'] != run_id:
                    self.full_results = {
                        "avg/full": [],
                        **{f"avg/{l}": [] for l in self.lengths},
                        **{f"eval_time/{l}": [] for l in self.lengths},
                        "verbose": {},
                        'run_id': run_id,
                    }
                    master_print(f"[STATUS] Found previous run, but run_id mismatch, clear and rerunning...")
                    clear_dir_of_res_path(self.res_path)
                    locked_atomic_write_json(self.res_path, self.full_results)
            else:
                self.full_results = {
                    "avg/full": [],
                    **{f"avg/{l}": [] for l in self.lengths},
                    **{f"eval_time/{l}": [] for l in self.lengths},
                    "verbose": {},
                    'run_id': run_id,
                }
                master_print(f"[STATUS] No previous runs, running new...")
                clear_dir_of_res_path(self.res_path)
                locked_atomic_write_json(self.res_path, self.full_results)
        dist.barrier()
        self.full_results = locked_read_json(path=self.res_path)
            

    def pack_batch(self, data: list, batch_size: int):
        batched_data = []
        batch = []
        for data_point in data:
            if len(batch) >= batch_size:
                batched_data.append(batch)
                batch = []

            batch.append(data_point)

        if len(batch):
            batched_data.append(batch)
            
        return batched_data

    def load_data(self, task: dict, length: int):
        task_name, task_subset = task["name"], task["subset"]
        if task_name not in self.tasks_customized:
            raise ValueError(f'{task} is not found in config_tasks.yaml')
        
        task_file = os.path.join(self.data_dir, str(length), "data", task_name, f'{task_subset}.jsonl')
        
        with open(task_file, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]

        for i, d in enumerate(data):
            d["idx"] = i
            d["task_key"] = f"{length}/{task_name}"
            d["input"] = d["input"].strip()
            if task_name in ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
                             "niah_multiquery", "niah_multivalue", 
                             "niah_single_1", "niah_single_2", "niah_single_3"]:
                d["input"] = d["input"] + ":"
        return data

    def evaluate(
        self, full_save_dir: Optional[str] = None, **kwargs):
        if is_master():
            master_print(f"tasks {self.tasks}")
        
        self.model.eval()
        
        device = get_device(self.model)
        

        for length in self.lengths:
            if isinstance(self.full_results[f"avg/{length}"], list):
                auto_batch_size = max(1, self.batch_size * self.lengths[0] // length)
                self.full_results[f"avg/{length}"] = []
                self.full_results[f"eval_time/{length}"] = []
                if is_master():
                    os.makedirs(f"{full_save_dir}/{length}", exist_ok=True)
                    print("auto_batch_size", auto_batch_size)
                    
                
                for task in self.tasks:
                    task_name = task["name"]
                    task_key = f"{length}/{task_name}"
                    complete_marker_file = os.path.join(full_save_dir, f'{task_key}.complete')
                    pred_file = os.path.join(full_save_dir, f'{task_key}.jsonl')
                    if os.path.isfile(complete_marker_file):
                        master_print(f"<< {task_key} already processed, skipping {task_key} >>")
                    else:
                        # The fused (jetlong_fused) method's long-context prefill accumulates
                        # GPU memory across tasks; free the previous task's caches/reserved
                        # blocks for it only. Gated on the fused architecture so every other
                        # method's FA2 eval path is byte-identical (no extra gc / empty_cache).
                        if any("JetLongFused" in a for a in (getattr(self.model.config, "architectures", None) or [])):
                            import gc
                            import torch
                            gc.collect()
                            torch.cuda.empty_cache()
                        if is_master():
                            print(f"Evaluatining task: {task_name}, length: {length}")
                        
                        data = self.load_data(task, length)
                        task_outputs = [
                            {
                                "idx": d["idx"], 
                                "pred": None,
                                "task_key": task_key,
                                "input": d["input"], 
                                "outputs": d["outputs"],
                                "others": d.get('others', {}),
                                "truncation_list": d.get('truncation', -1),
                                "length_list": d.get('length', -1),
                            } for d in data
                        ]

                        pad_to_size = math.ceil(len(data) / get_dist_size()) * get_dist_size()
                        padding_data = [deepcopy(data[0]) for _ in range(pad_to_size - len(data))]
                        for d in padding_data:
                            d["idx"] = -1
                        padded_data = data + padding_data

                        batched_data = self.pack_batch(padded_data, auto_batch_size)
                    
                        batched_data_on_rank = chunk_list_interleaved(batched_data, get_dist_size())[get_dist_rank()]
                                    
                        all_texts = []
                        start_time = time.perf_counter()
                        for batch in tqdm(batched_data_on_rank, desc="Generating", disable=not is_master()):
                            prompts = [d["input"] for d in batch]
                            if self.apply_chat_template:
                                prompts = [self.tokenizer.apply_chat_template(
                                    [{"role": "user", "content": p}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False,
                                ) for p in prompts]
                            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
                            for k in inputs:
                                inputs[k] = inputs[k].to(device)
                            # Chat template adds extra tokens (role markers, thinking suppression), allow some overhead
                            length_tolerance = length * 1.01 if self.apply_chat_template else length
                            assert inputs["input_ids"].shape[1] <= length_tolerance, f"Input length {inputs['input_ids'].shape[1]} exceeds the set length {length} (tolerance: {length_tolerance})"
                            generated_ids = self.model.generate(
                                **inputs,
                                **self.generation_config,
                                prefill_chunk_size=self.prefill_chunk_size,
                                max_new_tokens=self.tasks_customized[task_name]["tokens_to_generate"],
                                pad_token_id=self.tokenizer.pad_token_id,
                                stop_token_list=[
                                    [self.tokenizer.eos_token_id], 
                                    self.tokenizer.encode("\n", add_special_tokens=False),
                                    self.tokenizer.encode(".\n", add_special_tokens=False),
                                    self.tokenizer.encode("\n\n", add_special_tokens=False),
                                    self.tokenizer.encode(".\n\n", add_special_tokens=False)],
                            )
                            generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        
                            for text, d in zip(generated_texts, batch):
                                if self.stop is not None:
                                    for s in self.stop:
                                        text = text.split(s)[0]

                                all_texts.append((text, d["idx"]))
                        end_time = time.perf_counter()
                        time_elapsed = end_time - start_time
                        all_texts_gather = [[] for _ in range(get_dist_size())]
                        dist.all_gather_object(all_texts_gather, all_texts)
                        for _texts in all_texts_gather:
                            for text, idx in _texts:
                                if idx >= 0:
                                    task_outputs[idx]["pred"] = text
                                    assert task_outputs[idx]["idx"] == idx
                        
                        if is_master() and full_save_dir:
                            os.makedirs(full_save_dir, exist_ok=True)
                            with open(pred_file, 'w') as f:
                                for out in task_outputs:
                                    f.write(json.dumps(out) + "\n")
                                f.write(json.dumps({'time_elapsed': time_elapsed}) + "\n")
                            fd = os.open(complete_marker_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                            os.close(fd)
                            os.sync()
                
                dist.barrier()
                outputs = {}
                for task in self.tasks:
                    task_name = task["name"]
                    task_key = f"{length}/{task_name}"
                    complete_marker_file = os.path.join(full_save_dir, f'{task_key}.complete')
                    pred_file = os.path.join(full_save_dir, f'{task_key}.jsonl')
                    assert os.path.isfile(complete_marker_file), f"Missing prediction for {task_key}"
                    with open(pred_file, "r") as f:
                        outputs[task_key] = [json.loads(line) for line in f]
                    self.full_results[f"eval_time/{length}"].append((outputs[task_key].pop())['time_elapsed'])
                    
                for task_key, output in outputs.items():
                    _, task_name = task_key.split("/")
                    for d in output:
                        assert d["pred"] is not None, f"Missing prediction for {d}"

                    eval_results, subm_results = {}, {}
                    task_score, task_nulls, predicts, indices = run_evaluation_per_task(
                        outputs=outputs[task_key],
                        task_config=self.tasks_customized[task_name],
                    )
                    eval_results[task_key] = {
                        'score': task_score,
                        'nulls': task_nulls,
                    }
                    subm_results[task_key] = {
                        'predicts': predicts,
                        'indices': indices,
                    }
                    
                    if self.verbose:
                        self.full_results.update(eval_results)
                    self.full_results["avg/full"].append(task_score)
                    self.full_results[f"avg/{length}"].append(task_score)
                
                dist.barrier()
                self.full_results[f"avg/{length}"] = np.mean(self.full_results[f"avg/{length}"])
                
                if is_master():
                    self.full_results[f"eval_time/{length}"] = sum(self.full_results[f"eval_time/{length}"])
                    log_and_flush(
                        wandb_run= self.wandb_run,
                        length=length,
                        log_dict={"RULER_by_length(%)": self.full_results[f"avg/{length}"], 
                                  "n(length)": length, 
                                  "eval_time(s)": self.full_results[f"eval_time/{length}"]},
                    )
                    locked_atomic_write_json(self.res_path, self.full_results)
            else:
                master_print(f"<< Skip Length={length} >>")
        self.full_results["avg/full"] = np.mean(self.full_results["avg/full"])
        if is_master():
            self.wandb_run.log({"RULER_avg": self.full_results["avg/full"]}, step=length+1)
            wandb.summary.update(self.full_results)
        return self.full_results
    
    def save_res_artifact(self, json_path):
        run_name = self.wandb_run.name
        art = wandb.Artifact(name=f"{run_name}_ruler-eval-json", type="evaluation", description="Evaluation results")
        art.add_file(json_path)
        wandb.log_artifact(art)

    def finish_and_log(self, json_file_path: str ='', 
                       wandb_hash_suffix: str=DEFAULT_WANDB_RESULT_LOG_SUFFIX,
                       task_abbrev: str ="RULER", 
                       train_step: int = -1):
        task_abbrev = to_uppercase_only_letters(task_abbrev)
        if is_master():
            self.wandb_run.finish()
            finish_log_id = hashlib.sha1((self.wandb_report_name+wandb_hash_suffix+str(self.version_str)).encode("utf-8")).hexdigest()
            wandb_log_run =  wandb.init(name=self.wandb_report_name, id=finish_log_id, 
                                        resume="allow", job_type=os.environ["WANDB_RUNTYPE"])   
            for length in self.lengths:
                wandb_log_run.log(
                    {f"{task_abbrev}_by_length(%)": self.full_results[f"avg/{length}"], 
                    "eval_time(s)": self.full_results[f"eval_time/{length}"]},
                    step=length 
                )
            if train_step > 0:
                wandb_log_run.log({f"{task_abbrev}_avg": self.full_results["avg/full"], "step": train_step})
            else:
                for train_step in [0, 1000, 10000, 64000, 256000]:
                    wandb_log_run.log({f"{task_abbrev}_avg": self.full_results["avg/full"], "step": train_step}, step=length)
            wandb.summary.update(self.full_results)
            if json_file_path != '':
                wandb_log_run.save(json_file_path)
            wandb_log_run.finish()
            create_complete_marker(json_file_path, version_str=self.version_str)