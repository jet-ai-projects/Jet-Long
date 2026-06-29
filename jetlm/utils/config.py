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

import os
import yaml
from copy import deepcopy

from jetlm.utils.io import load, save
from jetlm.utils.dist import DistLogger

import logging
logger = logging.getLogger("train.jetlm_le")
dist_logger = DistLogger(logger)


__all__ = [
    "parse_unknown_args",
    "resolve_and_load_config",
    "partial_update_config",
    "build_config",
    "dump_config",
]


def _parse_with_yaml(config_str: str) -> str | dict:
    try:
        # add space manually for dict
        if "{" in config_str and "}" in config_str and ":" in config_str:
            out_str = config_str.replace(":", ": ")
        else:
            out_str = config_str
        return yaml.safe_load(out_str)
    except ValueError:
        # return raw string if parsing fails
        return config_str


def parse_unknown_args(unknown: list) -> dict:
    """Parse unknown args."""
    index = 0
    parsed_dict = {}
    while index < len(unknown):
        key, val = unknown[index], unknown[index + 1]
        index += 2
        if not key.startswith("--"):
            continue
        key = key[2:]

        # try parsing with either dot notation or full yaml notation
        # Note that the vanilla case "--key value" will be parsed the same
        if "." in key:
            # key == a.b.c, val == val --> parsed_dict[a][b][c] = val
            keys = key.split(".")
            dict_to_update = parsed_dict
            for key in keys[:-1]:
                if not (key in dict_to_update and isinstance(dict_to_update[key], dict)):
                    dict_to_update[key] = {}
                dict_to_update = dict_to_update[key]
            dict_to_update[keys[-1]] = _parse_with_yaml(val)  # so we can parse lists, bools, etc...
        else:
            parsed_dict[key] = _parse_with_yaml(val)
    return parsed_dict


def resolve_and_load_config(path: str, config_name: str = None) -> dict:
    path = os.path.realpath(os.path.expanduser(path))
    if config_name is not None:
        path = os.path.join(path, config_name)
    config = load(path)
    return config


def partial_update_config(config: dict, partial_config: dict) -> dict:
    for key in partial_config:
        if key in config and isinstance(partial_config[key], dict) and isinstance(config[key], dict):
            partial_update_config(config[key], partial_config[key])
        else:
            config[key] = partial_config[key]
    return config


def build_config(config: str | dict, recursive=True) -> dict:
    """Build a configuration dictionary from a YAML file."""
    if isinstance(config, str):
        if not os.path.isfile(config):
            dist_logger.warning(f"Config file {config} does not exist.")
            default_config = {"Error": f"Config file {config} does not exist."}
        else:
            default_config = resolve_and_load_config(config)
    else:
        if not isinstance(config, dict):
            raise ValueError("Config must be a string path or a dictionary")
        default_config = config

    config = deepcopy(default_config)
    if recursive:
        for k in default_config:
            if isinstance(default_config[k], dict):
                _config = build_config(default_config[k], recursive=True)
                config[k] = _config

        if "include" in config:
            include_path = config.pop("include")
            _config = build_config(include_path, recursive=True)
            config = partial_update_config(_config, config)
            # config["include_resolved"] = include_path

    return config


def dump_config(config: dict, path: str, config_name: str = None) -> None:
    if config_name is not None:
        path = os.path.join(path, config_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save(config, path)
