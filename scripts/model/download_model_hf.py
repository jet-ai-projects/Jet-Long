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

import os
import argparse
from huggingface_hub import login, snapshot_download


def download_model(repo_id, local_dir=None, token=None):
    if token:
        login(token=token)

    if local_dir is None:
        model_name = repo_id.split("/")[-1]
        local_dir = os.path.join("model_cache", model_name)

    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {repo_id} to {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {local_dir}")
    return local_dir


def try_load_model(local_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {local_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    print(f"Tokenizer loaded: {type(tokenizer).__name__}")

    print(f"Loading model from {local_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, trust_remote_code=True, device_map="auto"
    )
    print(f"Model loaded: {type(model).__name__}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and optionally load a HuggingFace model")
    parser.add_argument("repo_id", type=str, help="HuggingFace model repo id, e.g. meta-llama/Llama-2-7b-hf")
    parser.add_argument("--local-dir", type=str, default=None, help="Local directory (default: model_cache/<model_name>)")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--no-load", action="store_true", help="Skip loading the model after download")
    args = parser.parse_args()

    local_dir = download_model(args.repo_id, args.local_dir, args.token)

    if not args.no_load:
        try_load_model(local_dir)
