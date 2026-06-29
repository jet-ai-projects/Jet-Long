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

"""Upload processed_data/PG-19-subsample/ to the jet-ai/pg19-subsample HF dataset.

Layout mirrors the local tree so the downloader can drop files straight into
processed_data/PG-19-subsample/ without any path remapping:

    jet-ai/pg19-subsample/
        books.jsonl
        manifest.json

Usage:
    python scripts/data/upload_pg19sub_hf.py
    python scripts/data/upload_pg19sub_hf.py --token $HF_WRITE_TOKEN
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo


REPO_ID = "jet-ai/pg19-subsample"
REPO_TYPE = "dataset"
DEFAULT_LOCAL_ROOT = "processed_data/PG-19-subsample"


def _ensure_repo():
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        exist_ok=True,
        private=False,
    )


def upload_all(local_root: str):
    if not os.path.isdir(local_root):
        raise FileNotFoundError(f"Local directory not found: {local_root}")

    api = HfApi()
    _ensure_repo()

    print("Starting upload (full tree)...")
    print(f"   Local Source: {local_root}")
    print(f"   Remote Dest:  https://huggingface.co/datasets/{REPO_ID}/tree/main/")
    print("-" * 60)

    api.upload_folder(
        folder_path=local_root,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo="",
        commit_message="Upload PG-19-subsample",
        ignore_patterns=[".DS_Store", "*.tmp", "*.lock"],
    )
    print("\nUpload Complete!")
    print(f"Verify: https://huggingface.co/datasets/{REPO_ID}/tree/main/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_root", type=str, default=DEFAULT_LOCAL_ROOT,
                        help="Local PG-19-subsample root (default: processed_data/PG-19-subsample)")
    parser.add_argument("--token", type=str, required=False,
                        help="HF write token (optional if already logged in)")
    args = parser.parse_args()

    if args.token:
        from huggingface_hub import login
        login(token=args.token)

    upload_all(args.local_root)
