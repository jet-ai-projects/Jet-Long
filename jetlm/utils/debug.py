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

import os, torch, hashlib, json
import torch.distributed as dist
from .dist import is_master

def is_benchmark_mode():
    return (os.environ.get("JetLM_BENCHMARK_MODE", "0") in ["1", "true", "True"])


def set_benchmark_mode():
    os.environ["JetLM_BENCHMARK_MODE"] = "1"


def unset_benchmark_mode():
    os.environ["JetLM_BENCHMARK_MODE"] = "0"
    
def master_print(s):
    if is_master():
        print(s)
        
        



DTYPE_BYTES = {
    torch.float64: 8, torch.float32: 4, torch.bfloat16: 2, torch.float16: 2,
    torch.int64: 8, torch.int32: 4, torch.int16: 2, torch.int8: 1, torch.uint8: 1,
    torch.bool: 1,
}

def every_device_report(tag=""):
    print(f"\n[{tag}, also this does not work after dist initialized in separate process]")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i)/2**20
        reserv = torch.cuda.memory_reserved(i)/2**20
        maxalloc = torch.cuda.max_memory_allocated(i)/2**20
        print(f"cuda:{i}  allocated={alloc:8.1f} MB  reserved={reserv:8.1f} MB  max_alloc={maxalloc:8.1f} MB")

def per_device_report(tag=""):
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / 2**20
    reserv = torch.cuda.memory_reserved(dev) / 2**20
    maxalloc = torch.cuda.max_memory_allocated(dev) / 2**20
    if dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
        print(f"[{tag}] rank {rank}/{world-1} device cuda:{dev} → "
              f"allocated={alloc:8.1f} MB  reserved={reserv:8.1f} MB  max_alloc={maxalloc:8.1f} MB\n")
    else:
        print(f"[{tag}] device cuda:{dev} → allocated={alloc:8.1f} MB  reserved={reserv:8.1f} MB  max_alloc={maxalloc:8.1f} MB\n")

def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * DTYPE_BYTES.get(t.dtype, t.element_size())

def module_device_report(model, *, only_nonempty=True, recurse_children=False, rank_prefix=""):
    """
    Prints, per module, which device(s) its *direct* params/buffers are on,
    and the memory they occupy. Set recurse_children=True to include children’s
    params in each line (slower, more verbose).
    """
    rows = []
    for name, module in model.named_modules():
        params = list(module.parameters(recurse=recurse_children))
        bufs   = list(module.buffers(recurse=recurse_children))
        if only_nonempty and not params and not bufs:
            continue

        devs = set()
        n_params = sum(p.numel() for p in params)
        n_bufs   = sum(b.numel() for b in bufs)
        bytes_params = sum(_tensor_nbytes(p) for p in params)
        bytes_bufs   = sum(_tensor_nbytes(b) for b in bufs)

        for t in params + bufs:
            # For FSDP, some tensors can be sharded/flat; still has a .device
            devs.add(str(t.device))

        rows.append((name or "<root>", ",".join(sorted(devs)) or "n/a",
                     n_params, bytes_params, n_bufs, bytes_bufs))

    # Pretty print, sorted by memory
    rows.sort(key=lambda r: r[3], reverse=True)
    header = f"{rank_prefix}Module                                Devs     #Params     Param MB   #Bufs     Buf MB"
    print(header)
    print("-"*len(header))
    for name, devs, n_params, b_params, n_bufs, b_bufs in rows:
        print(f"{name[:36]:<36}  {devs:<8} {n_params:>10}  {b_params/2**20:>10.2f} {n_bufs:>8}  {b_bufs/2**20:>8.2f}")

def cuda_mem_report(tag=""):
    dev = torch.cuda.current_device()
    print(f"[{tag}] cuda:{dev} allocated={torch.cuda.memory_allocated(dev)/2**20:.1f}MB "
          f"reserved={torch.cuda.memory_reserved(dev)/2**20:.1f}MB "
          f"max_alloc={torch.cuda.max_memory_allocated(dev)/2**20:.1f}MB")

# Example usage (per rank if distributed)
# if torch.distributed.is_initialized():
#     rank = torch.distributed.get_rank()
#     torch.distributed.barrier()
#     print(f"\n=== RANK {rank} ===")
#     module_device_report(model, rank_prefix=f"[rank {rank}] ")
# else:
#     module_device_report(model)
# cuda_mem_report("before-train")


def _sha256_tensor(t: torch.Tensor) -> str:
    # exact bytes hash; ok because you only do it for 2 steps x 2 micros
    x = t.detach().to("cpu", non_blocking=False).contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()
import os, json, hashlib
import torch
import torch.distributed as dist

try:
    from torch.distributed._tensor import DTensor
except Exception:
    DTensor = None


def _as_local_tensor(x: torch.Tensor) -> torch.Tensor:
    # Handle DTensor if present
    if DTensor is not None and isinstance(x, DTensor):
        x = x.full_tensor()
    return x


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = _as_local_tensor(t)
    t = t.detach()
    # keep exact bits; for ints this is stable, for floats too (but you mostly have ints here)
    t_cpu = t.to("cpu", non_blocking=False).contiguous()
    return t_cpu.numpy().tobytes()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _head_tail_list_1d(t: torch.Tensor, head_n: int, tail_n: int) -> dict:
    """
    Returns small previews for 1D/2D tensors (we assume per-sample is 1D seq).
    """
    t = _as_local_tensor(t).detach()
    t_cpu = t.to("cpu", non_blocking=False)
    # flatten per-sample to 1D
    if t_cpu.ndim != 1:
        t_cpu = t_cpu.reshape(-1)
    out = {}
    if head_n > 0:
        out["head"] = t_cpu[:head_n].tolist()
    if tail_n > 0:
        out["tail"] = t_cpu[-tail_n:].tolist()
    return out


def log_batch_multiset_signature(
    batch: dict,
    *,
    out_path: str,
    global_step: int,
    micro_step: int,
    tag: str,
    head_n: int = 16,
    tail_n: int = 8,
    max_samples_total: int | None = None,
) -> None:
    """
    Logs a signature invariant to order (multiset) + detailed per-sample previews
    so you can detect flipped/reordered batches.

    Expected keys in batch:
      - input_ids: [bs, seqlen]
      - attention_mask: [bs, seqlen] or None
      - labels: [bs, seqlen]
    """
    input_ids = batch.get("input_ids", None)
    attention_mask = batch.get("attention_mask", None)
    labels = batch.get("labels", None)

    if input_ids is None or labels is None:
        raise ValueError("batch must contain at least input_ids and labels (attention_mask optional).")

    input_ids = _as_local_tensor(input_ids)
    labels = _as_local_tensor(labels)
    if attention_mask is not None:
        attention_mask = _as_local_tensor(attention_mask)

    # Local per-sample records
    local_bs = int(input_ids.shape[0])
    sample_dicts = []

    for i in range(local_bs):
        ids_i = input_ids[i]
        lbl_i = labels[i]
        msk_i = attention_mask[i] if attention_mask is not None else None

        h = hashlib.sha256()
        h.update(_tensor_bytes(ids_i))
        if msk_i is not None:
            h.update(_tensor_bytes(msk_i))
        else:
            h.update(b"<no_attention_mask>")
        h.update(_tensor_bytes(lbl_i))
        sample_hash = h.hexdigest()

        rec = {
            "rank": int(dist.get_rank()) if (dist.is_available() and dist.is_initialized()) else 0,
            "local_idx": int(i),
            "sample_hash": sample_hash,
            "input_ids": _head_tail_list_1d(ids_i, head_n=head_n, tail_n=tail_n),
            "labels": _head_tail_list_1d(lbl_i, head_n=head_n, tail_n=tail_n),
            "input_shape": list(ids_i.shape),
            "labels_shape": list(lbl_i.shape),
            "input_dtype": str(ids_i.dtype),
            "labels_dtype": str(lbl_i.dtype),
        }

        if msk_i is not None:
            # attention_mask sum is often informative (padding differences)
            msk_cpu = msk_i.detach().to("cpu", non_blocking=False)
            rec["attention_mask"] = _head_tail_list_1d(msk_i, head_n=head_n, tail_n=tail_n)
            rec["attention_mask_sum"] = int(msk_cpu.sum().item())
            rec["attention_mask_dtype"] = str(msk_i.dtype)
            rec["attention_mask_shape"] = list(msk_i.shape)
        else:
            rec["attention_mask"] = None
            rec["attention_mask_sum"] = None

        # label stats (optional but useful): count ignore_index=-100
        lbl_cpu = lbl_i.detach().to("cpu", non_blocking=False)
        rec["labels_ignore_count"] = int((lbl_cpu == -100).sum().item())

        sample_dicts.append(rec)

    # Cap total samples logged (optional)
    # Apply cap AFTER gathering so the cap is global & deterministic (rank0 decides).
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, sample_dicts)
        all_samples = [s for rank_samples in gathered for s in rank_samples]
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        all_samples = sample_dicts
        rank = 0
        world_size = 1

    # Build hashes
    # 1) order-invariant multiset hash (sort by sample_hash)
    sample_hashes_sorted = sorted([s["sample_hash"] for s in all_samples])
    micro_multiset_hash = _sha256_bytes("|".join(sample_hashes_sorted).encode("utf-8"))

    # 2) order-sensitive hash (rank-major, local_idx within rank)
    all_samples_ordered = sorted(all_samples, key=lambda s: (s["rank"], s["local_idx"]))
    ordered_hash_list = [s["sample_hash"] for s in all_samples_ordered]
    micro_ordered_hash = _sha256_bytes("|".join(ordered_hash_list).encode("utf-8"))

    # Optionally cap the stored sample list to keep json small
    if max_samples_total is not None and len(all_samples_ordered) > max_samples_total:
        # keep first K in *ordered* view and also first K in *sorted-by-hash* view
        # so you can still compare both perspectives
        ordered_preview = all_samples_ordered[:max_samples_total]
        sorted_preview = sorted(all_samples, key=lambda s: s["sample_hash"])[:max_samples_total]
        stored = {
            "ordered_preview": ordered_preview,
            "sorted_by_hash_preview": sorted_preview,
            "cap": int(max_samples_total),
        }
    else:
        stored = {
            "ordered": all_samples_ordered,
            "sorted_by_hash": sorted(all_samples, key=lambda s: s["sample_hash"]),
            "cap": None,
        }

    # right before writing `rec = dict(...)` in rank==0 block
    micro_input_shape = list(input_ids.shape)
    if rank == 0:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        rec = dict(
            tag=tag,
            global_step=int(global_step),
            micro_step=int(micro_step),
            world_size=int(world_size),
            world_count=int(len(all_samples)),
            micro_multiset_hash=micro_multiset_hash,   # order-invariant
            micro_ordered_hash=micro_ordered_hash,     # order-sensitive
            # quick heads for diffing without scrolling massive lists
            sample_hashes_head_sorted=sample_hashes_sorted[:16],
            sample_hashes_head_ordered=ordered_hash_list[:16],
            samples=stored,
            micro_input_shape=micro_input_shape, 
        )
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")