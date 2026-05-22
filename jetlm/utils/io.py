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
import yaml
import json
import shutil
from typing import Any

import logging
logger = logging.getLogger("train.jetlm_le")


def load(path: str) -> dict|str|list:
    if path.endswith('.json'):
        return load_json(path)
    elif path.endswith('.jsonl'):
        return load_jsonl(path)
    elif path.endswith('.txt'):
        return load_txt(path)
    elif path.endswith('.yaml') or path.endswith('.yml'):
        return load_yaml(path)
    else:
        return load_raw(path)

def save(data: dict|list|str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith('.json'):
        save_json(data, path)
    elif path.endswith('.jsonl'):
        save_jsonl(data, path)
    elif path.endswith('.txt'):
        save_txt(data, path)
    elif path.endswith('.yaml') or path.endswith('.yml'):
        save_yaml(data, path)
    else:
        save_raw(data, path)
        

def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_jsonl(path: str) -> list:
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: list, path: str) -> None:
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def load_raw(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

def save_raw(data: Any, path: str) -> None:
    with open(path, 'w') as f:
        f.write(str(data))


def load_txt(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

def save_txt(data: str, path: str) -> None:
    with open(path, 'w') as f:
        f.write(data)


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: dict, path: str) -> None:
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)


def file_size(path: str) -> int:
    """
    st_size: It represents the size of the file in bytes.

    return #bytes of the given file
    """
    return os.stat(path).st_size


def read_bytes(path: str, bytes_start: int, num_bytes: int) -> bytes:
    with open(path, "rb") as fin:
        fin.seek(bytes_start)
        return fin.read(num_bytes)


def copy_files(src_path: str, dst_path: str) -> None:
    if os.path.isfile(src_path):
        shutil.copy(src_path, dst_path)
    elif os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        
        
def remove_files(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def remove_path(path: str) -> None:
    logger.warning(f"Removing path {path}")
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        

def clear_dir_of_res_path(res_path: str):
    dir_path = os.path.dirname(res_path)
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
            
def get_filename_without_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]