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

"""
Generate RULER test data at a given sequence length, matching the format used
by processed_data/RULER-data/<length>/data/<task>/validation.jsonl:

    {"index": <sample_id>, "input": <prompt + answer_prefix>, "outputs": [...], "length": <int>}

Wraps NVIDIA RULER's per-task scripts (tasks configured via synthetic.yaml in
the RULER repo). Default tokenizer is Qwen3.

Usage:
    python scripts/data/generate_ruler.py \
        --length 98304 \
        --output_dir processed_data/RULER-data \
        --ruler_repo /path/to/RULER \
        --tokenizer Qwen/Qwen3-1.7B-Base \
        --num_samples 500
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe",
    "qa_1", "qa_2",
]


def postprocess_jsonl(path: Path):
    """Rewrite a RULER-produced jsonl to the 4-field jetlm format.

    RULER writes extra fields (answer_prefix, length_w_model_temp, ...) and
    stores `input` without the answer prefix. jetlm expects `input` to already
    contain the answer prefix and only the 4 canonical fields.
    """
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    out_rows = []
    for i, r in enumerate(rows):
        ap = r.get("answer_prefix", "")
        out_rows.append({
            "index": i,
            "input": r["input"] + ap,
            "outputs": r["outputs"],
            "length": r["length"],
        })

    with open(path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    return len(out_rows)


def run_one_task(ruler_data_dir: Path, task: str, save_dir: Path,
                 tokenizer: str, tokenizer_type: str, length: int,
                 num_samples: int, random_seed: int):
    cmd = [
        sys.executable, "prepare.py",
        "--save_dir", str(save_dir),
        "--benchmark", "synthetic",
        "--task", task,
        "--tokenizer_path", tokenizer,
        "--tokenizer_type", tokenizer_type,
        "--max_seq_length", str(length),
        "--model_template_type", "base",
        "--num_samples", str(num_samples),
        "--random_seed", str(random_seed),
    ]
    print(f"\n>>> [{task}] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ruler_data_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"prepare.py failed for task {task} (exit {proc.returncode})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("processed_data/RULER-data"),
                   help="root dir; per-length subdir <output_dir>/<length>/data/ is created")
    p.add_argument("--ruler_repo", type=Path, required=True,
                   help="path to cloned NVIDIA RULER repo (e.g. /path/to/RULER)")
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--tokenizer_type", type=str, default="hf")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--tasks", nargs="+", default=None,
                   help="subset of tasks to run; defaults to all 13")
    p.add_argument("--force", action="store_true",
                   help="regenerate even if target jsonl already exists")
    args = p.parse_args()

    ruler_data_dir = args.ruler_repo / "scripts" / "data"
    if not (ruler_data_dir / "prepare.py").exists():
        raise FileNotFoundError(f"prepare.py not found under {ruler_data_dir}")

    # Verify source data files.
    json_dir = ruler_data_dir / "synthetic" / "json"
    for fname in ("PaulGrahamEssays.json", "squad.json", "hotpotqa.json"):
        if not (json_dir / fname).exists():
            raise FileNotFoundError(
                f"missing RULER source file: {json_dir/fname}. "
                f"Run download_qa_dataset.sh and download_paulgraham_essay.py in {json_dir}."
            )

    save_dir = (args.output_dir / str(args.length) / "data").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or TASKS
    for task in tasks:
        target = save_dir / task / "validation.jsonl"
        if target.exists() and not args.force:
            with open(target) as f:
                n = sum(1 for _ in f)
            if n == args.num_samples:
                print(f"[skip] {target} already has {n} lines")
                continue
            print(f"[regen] {target} has {n} lines (expected {args.num_samples})")

        if target.exists():
            target.unlink()

        run_one_task(
            ruler_data_dir=ruler_data_dir,
            task=task,
            save_dir=save_dir,
            tokenizer=args.tokenizer,
            tokenizer_type=args.tokenizer_type,
            length=args.length,
            num_samples=args.num_samples,
            random_seed=args.random_seed,
        )
        if not target.exists():
            raise RuntimeError(
                f"prepare.py exited 0 but {target} was not produced — "
                f"check subprocess stderr above (likely task-script crash swallowed by prepare.py)."
            )
        n = postprocess_jsonl(target)
        if n != args.num_samples:
            raise RuntimeError(f"{task}: expected {args.num_samples} samples, got {n}")
        print(f"[done] {task}: {n} samples written to {target}")

    print(f"\nAll tasks complete under {save_dir}")


if __name__ == "__main__":
    main()
