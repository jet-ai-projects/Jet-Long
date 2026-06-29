#!/usr/bin/env python
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

"""
Task-sharded HELMET launcher for multi-GPU single-node evaluation.

For each (config, dataset, input_max_length) triple, launch `eval.py` as a
subprocess pinned to one GPU via CUDA_VISIBLE_DEVICES. A worker pool of
len(gpus) slots consumes the task queue; the next task grabs a freed GPU.
Outputs go into the standard HELMET `output/<tag>/` layout — one JSON per
(dataset, length) pair, so concurrent writes never collide.

Example:
    python helmet_shard.py \\
        --helmet_dir tmp/helmet/repo \\
        --model_name_or_path model_cache/Qwen3-1.7B-Base-selfext \\
        --configs recall rag \\
        --lengths 8192 16384 32768 65536 131072 \\
        --gpus 0,1,2,3,4,5,6,7 \\
        --output_dir results/helmet/Qwen3-1.7B-Base-selfext \\
        --tag v1 \\
        --max_test_samples 100
"""
import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

# Our output is almost certainly going through tee/sbatch capture, so reconfigure
# stdout to be line-buffered — otherwise "[gpu0] START …" sits in a 4 KB block
# buffer for minutes and the run looks stuck.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass


# HELMET categories with their "full-length" and "short-length" config files.
# Full = 128K (single length per dataset), Short = 8K/16K/32K/64K fan-out.
CATEGORY_CONFIGS = {
    "recall":  ("recall.yaml",  "recall_short.yaml"),
    "rag":     ("rag.yaml",     "rag_short.yaml"),
    "cite":    ("cite.yaml",    "cite_short.yaml"),
    "rerank":  ("rerank.yaml",  "rerank_short.yaml"),
    "longqa":  ("longqa.yaml",  "longqa_short.yaml"),
    "summ":    ("summ.yaml",    "summ_short.yaml"),
    "icl":     ("icl.yaml",     "icl_short.yaml"),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helmet_dir", default="tmp/helmet/repo",
                    help="Path to the HELMET repo (contains eval.py + configs/).")
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--configs", nargs="+", default=list(CATEGORY_CONFIGS),
                    choices=list(CATEGORY_CONFIGS),
                    help="Which HELMET categories to evaluate.")
    ap.add_argument("--lengths", nargs="+", type=int,
                    default=[8192, 16384, 32768, 65536, 131072],
                    help="Input context lengths to evaluate at.")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                    help="Comma-separated GPU indices to use.")
    ap.add_argument("--output_dir", required=True,
                    help="Where HELMET writes result JSONs.")
    ap.add_argument("--log_dir", default=None,
                    help="Where to write per-task subprocess logs. "
                         "Defaults to <output_dir>/logs for backwards compat.")
    ap.add_argument("--tag", default="v1",
                    help="Run tag embedded in output filenames.")
    ap.add_argument("--max_test_samples", type=int, default=100)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_chat_template", choices=["True", "False"], default=None,
                    help="Override; default comes from each config's YAML.")
    ap.add_argument("--no_torch_compile", action="store_true", default=True,
                    help="Disable torch.compile — safer with custom trust_remote_code modeling files.")
    ap.add_argument("--extra_args", default="",
                    help="Extra raw CLI args forwarded to eval.py (e.g. '--rope_theta 1000000').")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the command list without launching.")
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip tasks whose output file already exists (rely on eval.py's own check too).")
    return ap.parse_args()


def expand_config(cfg_path: Path) -> list[dict]:
    """Turn a HELMET config YAML with comma-lists into a list of single-task dicts."""
    cfg = yaml.safe_load(cfg_path.read_text())
    datasets = str(cfg["datasets"]).split(",")
    test_files = str(cfg.get("test_files", ",".join([""] * len(datasets)))).split(",")
    demo_files = str(cfg.get("demo_files", ",".join([""] * len(datasets)))).split(",")
    in_len = str(cfg["input_max_length"]).split(",")
    gen_len = str(cfg["generation_max_length"]).split(",")

    n = len(datasets)
    assert len(test_files) == n and len(demo_files) == n, f"{cfg_path}: list length mismatch"
    # input/generation can be scalar (broadcast) or full-length lists
    if len(in_len) == 1:  in_len  = in_len  * n
    if len(gen_len) == 1: gen_len = gen_len * n

    tasks = []
    for i in range(n):
        tasks.append({
            "dataset":            datasets[i].strip(),
            "test_file":          test_files[i].strip(),
            "demo_file":          demo_files[i].strip(),
            "input_max_length":   int(in_len[i]),
            "generation_max_length": int(gen_len[i]),
            "use_chat_template":  cfg.get("use_chat_template", False),
            "stop_new_line":      cfg.get("stop_new_line", False),
            "config_path":        str(cfg_path),
        })
    return tasks


def build_task_list(helmet_dir: Path, categories: list[str],
                    lengths_wanted: list[int]) -> list[dict]:
    """Enumerate all (category, dataset, length) tasks across categories."""
    cfg_dir = helmet_dir / "configs"
    all_tasks = []
    wanted = set(lengths_wanted)
    for cat in categories:
        long_yaml, short_yaml = CATEGORY_CONFIGS[cat]
        for yaml_name in (long_yaml, short_yaml):
            p = cfg_dir / yaml_name
            if not p.exists():
                print(f"[WARN] missing config {p}", file=sys.stderr); continue
            for t in expand_config(p):
                if t["input_max_length"] in wanted:
                    t["category"] = cat
                    all_tasks.append(t)
    # order by descending length so heaviest tasks start first
    all_tasks.sort(key=lambda x: -x["input_max_length"])
    return all_tasks


def expected_output_file(args, task: dict) -> Path:
    """Reconstruct eval.py's output filename so we can skip already-done tasks.

    Mirrors the path string in eval.py:37.
    """
    test_name = Path(task["test_file"] or "").stem  # empty → ""
    do_sample = False
    temperature = 0.0
    top_p = 1.0
    chat = task["use_chat_template"] if args.use_chat_template is None \
        else (args.use_chat_template == "True")
    fn = (
        f'{task["dataset"]}_{args.tag}_{test_name}_in{task["input_max_length"]}'
        f'_size{args.max_test_samples}_shots{args.shots}_samp{do_sample}'
        f'max{task["generation_max_length"]}min0t{temperature}p{top_p}'
        f'_chat{chat}_{args.seed}.json'
    )
    return Path(args.output_dir) / fn


def build_cmd(args, task: dict) -> list[str]:
    # eval.py runs with cwd=helmet_dir, so relative paths from the caller must be
    # absolutized before forwarding. Test/demo files live inside helmet_dir (data/...)
    # and stay relative; the model and config paths need to be made absolute.
    model_path = args.model_name_or_path
    if not os.path.isabs(model_path) and os.path.exists(model_path):
        model_path = os.path.abspath(model_path)
    # patched_eval.py lives alongside this launcher and applies compat fixes
    # (e.g. multi_lexsum on datasets>=4) before invoking HELMET's eval.py.
    patched_eval = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "patched_eval.py")
    cmd = [
        sys.executable, patched_eval,
        "--config", task["config_path"],
        "--model_name_or_path", model_path,
        "--datasets", task["dataset"],
        "--test_files", task["test_file"],
        "--demo_files", task["demo_file"],
        "--input_max_length", str(task["input_max_length"]),
        "--generation_max_length", str(task["generation_max_length"]),
        "--max_test_samples", str(args.max_test_samples),
        "--shots", str(args.shots),
        "--seed", str(args.seed),
        "--output_dir", os.path.abspath(args.output_dir),
        "--tag", args.tag,
    ]
    if args.use_chat_template is not None:
        cmd += ["--use_chat_template", args.use_chat_template]
    if args.no_torch_compile:
        cmd.append("--no_torch_compile")
    if args.extra_args:
        cmd += shlex.split(args.extra_args)
    return cmd


def worker(gpu_id: int, task_q: "queue.Queue[dict]", args, helmet_dir: Path,
           log_dir: Path, results: list, lock: threading.Lock):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")  # flush per-task log promptly

    while True:
        try:
            task = task_q.get_nowait()
        except queue.Empty:
            return

        out_file = expected_output_file(args, task)
        if args.skip_existing and out_file.exists():
            with lock:
                print(f"[gpu{gpu_id}] SKIP exists: {out_file.name}")
            results.append({"gpu": gpu_id, "task": task, "status": "skipped"})
            task_q.task_done()
            continue

        cmd = build_cmd(args, task)
        label = f'{task["category"]}/{task["dataset"]}@{task["input_max_length"]}'
        log_path = log_dir / f'{task["category"]}_{task["dataset"]}_in{task["input_max_length"]}.log'

        with lock:
            print(f"[gpu{gpu_id}] START {label}  → {log_path.name}")

        t0 = time.time()
        with open(log_path, "w") as logf:
            logf.write(f"# CMD: {' '.join(shlex.quote(c) for c in cmd)}\n")
            logf.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n\n")
            logf.flush()
            rc = subprocess.call(cmd, cwd=str(helmet_dir), env=env,
                                 stdout=logf, stderr=subprocess.STDOUT)
        dt = time.time() - t0

        with lock:
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[gpu{gpu_id}] {status}  {label}  {dt:6.1f}s")
        results.append({"gpu": gpu_id, "task": task, "status": status, "seconds": dt})
        task_q.task_done()


def main():
    args = parse_args()
    helmet_dir = Path(args.helmet_dir).resolve()
    assert (helmet_dir / "eval.py").exists(), f"eval.py not under {helmet_dir}"

    tasks = build_task_list(helmet_dir, args.configs, args.lengths)
    if not tasks:
        print("No tasks to run.", file=sys.stderr); sys.exit(1)

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tasks:    {len(tasks)} across {len(args.configs)} categories × {len(args.lengths)} lengths")
    print(f"GPUs:     {gpus}")
    print(f"Model:    {args.model_name_or_path}")
    print(f"Output:   {args.output_dir}  (tag={args.tag}, samples={args.max_test_samples})")
    print(f"Logs:     {log_dir}")

    if args.dry_run:
        for t in tasks:
            print(f"  {t['category']:6s}  {t['dataset']:40s}  L={t['input_max_length']}")
        sys.exit(0)

    q: "queue.Queue[dict]" = queue.Queue()
    for t in tasks: q.put(t)

    results: list = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(g, q, args, helmet_dir, log_dir, results, lock),
                         daemon=False, name=f"gpu{g}")
        for g in gpus
    ]
    t0 = time.time()
    for th in threads: th.start()
    for th in threads: th.join()
    dt = time.time() - t0

    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"].startswith("FAIL"))
    skip = sum(1 for r in results if r["status"] == "skipped")
    print(f"\n===== DONE {len(results)}/{len(tasks)}   ok={ok} fail={fail} skip={skip}   {dt/60:.1f} min =====")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
