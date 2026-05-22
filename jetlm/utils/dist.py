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
from datetime import timedelta
from typing import Optional
import logging

import torch
import torch.distributed


def _get_dist_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def dist_close() -> None:
    dist_barrier()
    torch.distributed.destroy_process_group()


def dist_init(gpu: Optional[str] = None, cudnn_benchmark: bool = False, timeout: int = 3600) -> None:
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.enabled = cudnn_benchmark
    torch.backends.cudnn.deterministic = not cudnn_benchmark

    torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout))
    assert torch.distributed.is_initialized()

    torch.cuda.set_device(_get_dist_local_rank())


def get_dist_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_master() -> bool:
    return get_dist_rank() == 0


def dist_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def dist_print(x: str, rank=0) -> None:
    if get_dist_rank() == rank:
        print(x)


class DistLogger():
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info(self, *args, **kwargs) -> None:
        rank = kwargs.pop("rank", 0)
        if get_dist_rank() == rank:
            self.logger.info(*args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        rank = kwargs.pop("rank", 0)
        if get_dist_rank() == rank:
            self.logger.error(*args, **kwargs)

    def warning(self, *args, **kwargs) -> None:
        rank = kwargs.pop("rank", 0)
        if get_dist_rank() == rank:
            self.logger.warning(*args, **kwargs)
            
    def debug(self, *args, **kwargs) -> None:
        rank = kwargs.pop("rank", 0)
        if get_dist_rank() == rank:
            self.logger.debug(*args, **kwargs)


def sync_tensor(tensor: torch.Tensor | float, reduce="root", dim=0) -> torch.Tensor | list[torch.Tensor]:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return torch.mean(torch.stack(tensor_list, dim=dim), dim=dim)
    elif reduce == "sum":
        return torch.sum(torch.stack(tensor_list, dim=dim), dim=dim)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=dim)
    elif reduce == "stack":
        return torch.stack(tensor_list, dim=dim)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list

def get_world_size_safe(
    default: int = 1,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
    verbose: bool = False,
) -> int:
    if not torch.distributed.is_available():
        if verbose:
            print("[world_size] torch.distributed not available. Using default.")
        return default

    if not torch.distributed.is_initialized():
        if verbose:
            print("[world_size] torch.distributed not initialized. Using default.")
        return default

    try:
        ws = torch.distributed.get_world_size(group=process_group)
        if isinstance(ws, int) and ws > 0:
            return ws
        if verbose:
            print(f"[world_size] Invalid world size ({ws}). Using default.")
        return default
    except Exception as e:
        if verbose:
            print(f"[world_size] Exception while querying world size: {e}")
        return default