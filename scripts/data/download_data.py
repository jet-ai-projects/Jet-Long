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

"""Download an HF eval dataset into its canonical local path under processed_data/.

Usage:
    # download a single dataset (all of it)
    python scripts/data/download_data.py --dataset ruler
    python scripts/data/download_data.py --dataset pg19sub --token $HF_TOKEN

    # download a single RULER length only
    python scripts/data/download_data.py --dataset ruler --length 98304
"""

import argparse
import multiprocessing
import os
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import login, snapshot_download


REPO_TYPE = "dataset"

DATASETS = {
    "ruler": {
        "repo_id": "jet-ai/RULER-500",
        "local_root": "processed_data/RULER-data",
        "max_workers": 16,
        "supports_length": True,
    },
    "pg19sub": {
        "repo_id": "jet-ai/pg19-subsample",
        "local_root": "processed_data/PG-19-subsample",
        "max_workers": 8,
        "supports_length": False,
    },
}

MAX_RETRIES = 5
STALL_TIMEOUT = 180  # seconds without on-disk growth before we kill and retry


def _dir_size(*paths) -> int:
    total = 0
    for p in paths:
        if not os.path.isdir(p):
            continue
        for dirpath, _, files in os.walk(p):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
    return total


def _do_download(repo_id, local_dir, max_workers, allow_patterns, err_q):
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            local_dir=local_dir,
            max_workers=max_workers,
            allow_patterns=allow_patterns,
            ignore_patterns=[".DS_Store", "*.tmp", "*.lock"],
        )
    except Exception as e:
        err_q.put(str(e))
        raise


def download_with_retry(repo_id, local_dir, max_workers, allow_patterns, label):
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    repo_cache_dir = os.path.join(
        hf_home, "hub", f"{REPO_TYPE}s--{repo_id.replace('/', '--')}",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n[Attempt {attempt}/{MAX_RETRIES}] Downloading {label} "
              f"(stall timeout={STALL_TIMEOUT}s)...")

        err_q = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_do_download,
            args=(repo_id, local_dir, max_workers, allow_patterns, err_q),
        )
        proc.start()

        last_size = _dir_size(local_dir, repo_cache_dir)
        last_progress = time.time()

        while proc.is_alive():
            time.sleep(10)
            cur = _dir_size(local_dir, repo_cache_dir)
            if cur != last_size:
                last_size = cur
                last_progress = time.time()
            elif time.time() - last_progress > STALL_TIMEOUT:
                print(f"[Attempt {attempt}] No progress for {STALL_TIMEOUT}s "
                      f"(dir size stuck at {last_size / 1e6:.1f} MB), killing...")
                proc.terminate()
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                break

        proc.join()
        if proc.exitcode == 0:
            print("Download finished successfully.")
            return

        err = err_q.get_nowait() if not err_q.empty() else ""
        print(f"[Attempt {attempt}] Failed (exit {proc.exitcode}): {err}")
        time.sleep(5)

    raise RuntimeError(f"Download of {label} failed after {MAX_RETRIES} attempts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=sorted(DATASETS.keys()),
                        help="Which dataset to download.")
    parser.add_argument("--length", type=int, default=None,
                        help="(length-partitioned datasets only) fetch a single length "
                             "(e.g. 98304). Omit to fetch the full dataset.")
    parser.add_argument("--local_root", type=str, default=None,
                        help="Override the dataset's default local root.")
    parser.add_argument("--token", type=str, default=None,
                        help="HF token (optional if already logged in)")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    repo_id = spec["repo_id"]
    local_root = args.local_root or spec["local_root"]
    max_workers = spec["max_workers"]

    if args.length is not None and not spec["supports_length"]:
        parser.error(f"--length is not supported for dataset '{args.dataset}' "
                     f"(layout is not partitioned by length).")

    if args.token:
        login(token=args.token)

    os.makedirs(local_root, exist_ok=True)

    if args.length is None:
        allow_patterns = None
        label = f"{args.dataset} (all)"
        remote_url = f"https://huggingface.co/datasets/{repo_id}/tree/main/"
        final_dst = local_root
    else:
        allow_patterns = [f"{args.length}/**"]
        label = f"{args.dataset} length {args.length}"
        remote_url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{args.length}"
        final_dst = os.path.join(local_root, str(args.length))

    if os.path.isdir(final_dst) and os.listdir(final_dst):
        print(f"[warn] {final_dst} already exists and is non-empty — "
              f"will overwrite conflicts during download.")

    print(f"Starting download of {label}")
    print(f"   Remote: {remote_url}")
    print(f"   Local:  {final_dst}")
    print("-" * 60)

    download_with_retry(repo_id, local_root, max_workers, allow_patterns, label)

    print(f"\nDone. {label} available at {final_dst}")


if __name__ == "__main__":
    main()
