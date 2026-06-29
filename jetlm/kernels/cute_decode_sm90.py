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

from importlib.util import find_spec
import math
import operator
import os
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Callable, Optional

import torch


def _ensure_fa4_importable(fa4_root: str | None = None) -> None:
    if fa4_root is not None:
        raise ValueError(
            "fa4_root path injection is no longer supported. "
            "Install the FA4 CuTe Python package so flash_attn.cute is importable."
        )
    try:
        fa4_spec = find_spec("flash_attn.cute.interface")
    except ModuleNotFoundError:
        fa4_spec = None
    if fa4_spec is not None:
        return
    raise ImportError(
        "Unable to import flash_attn.cute.interface. "
        "Install a FlashAttention build that includes the FA4 CuTe Python modules."
    )


def _lazy_imports(fa4_root: str | None = None):
    _ensure_fa4_importable(fa4_root)
    import cuda.bindings.driver as cuda  # noqa: F401
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32
    from cutlass.cute.nvgpu import cpasync, warpgroup
    from cutlass import pipeline
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass.base_dsl.arch import Arch

    from quack import copy_utils
    from quack import layout_utils
    from quack import sm90_utils

    from flash_attn.cute.cute_dsl_utils import to_cute_tensor
    from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
    from flash_attn.cute.mask import AttentionMask
    from flash_attn.cute.mask import mask_r2p_lambda, r2p_bitmask_below, sm90_col_to_r2p_idx
    from flash_attn.cute.named_barrier import NamedBarrierFwd
    from flash_attn.cute.pack_gqa import make_packgqa_tiled_tma_atom, pack_gqa_layout
    from flash_attn.cute.seqlen_info import SeqlenInfoQK
    from flash_attn.cute.softmax import Softmax
    from flash_attn.cute import utils
    from flash_attn.cute import pipeline as pipeline_custom
    from flash_attn.cute.cache_utils import get_jit_cache

    return {
        "cutlass": cutlass,
        "cute": cute,
        "Float32": Float32,
        "Int32": Int32,
        "cpasync": cpasync,
        "warpgroup": warpgroup,
        "pipeline": pipeline,
        "pipeline_init_arrive": pipeline_init_arrive,
        "pipeline_init_wait": pipeline_init_wait,
        "Arch": Arch,
        "copy_utils": copy_utils,
        "layout_utils": layout_utils,
        "sm90_utils": sm90_utils,
        "to_cute_tensor": to_cute_tensor,
        "FlashAttentionForwardSm90": FlashAttentionForwardSm90,
        "AttentionMask": AttentionMask,
        "mask_r2p_lambda": mask_r2p_lambda,
        "r2p_bitmask_below": r2p_bitmask_below,
        "sm90_col_to_r2p_idx": sm90_col_to_r2p_idx,
        "NamedBarrierFwd": NamedBarrierFwd,
        "make_packgqa_tiled_tma_atom": make_packgqa_tiled_tma_atom,
        "pack_gqa_layout": pack_gqa_layout,
        "SeqlenInfoQK": SeqlenInfoQK,
        "Softmax": Softmax,
        "utils": utils,
        "pipeline_custom": pipeline_custom,
        "get_jit_cache": get_jit_cache,
    }


_COMPILE_CACHE = None
_SPLITK_WORKSPACE_CACHE = {}


def _get_cache(fa4_root: str | None = None):
    global _COMPILE_CACHE
    if _COMPILE_CACHE is None:
        _COMPILE_CACHE = _lazy_imports(fa4_root)["get_jit_cache"]("jetlm_cute_decode_sm90")
    return _COMPILE_CACHE


@lru_cache(maxsize=None)
def _get_device_arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


_SUPPORTED_CUTE_DTYPES = (torch.float16, torch.bfloat16)


def _require_contiguous(op: str, name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{op}: {name} must be contiguous; call {name}.contiguous() before invoking the CuTe kernel")


def _require_output_tensor(
    op: str,
    name: str,
    tensor: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor is None:
        return
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{op}: {name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{op}: {name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{op}: {name} must be on device {device}, got {tensor.device}")
    _require_contiguous(op, name, tensor)


def _require_decode_metadata(
    op: str,
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{op}: {name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{op}: {name} must have dtype torch.int32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{op}: {name} must be on device {device}, got {tensor.device}")


def _get_splitk_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    seqlen_q: int,
    nheads: int,
    head_dim_v: int,
    num_splits: int,
):
    key = (device.type, device.index, dtype, batch, seqlen_q, nheads, head_dim_v, num_splits)
    cached = _SPLITK_WORKSPACE_CACHE.get(key)
    if cached is None:
        cached = (
            torch.empty(
                batch * num_splits,
                seqlen_q,
                nheads,
                head_dim_v,
                device=device,
                dtype=dtype,
            ),
            torch.empty(
                batch * num_splits,
                nheads,
                seqlen_q,
                device=device,
                dtype=torch.float32,
            ),
        )
        _SPLITK_WORKSPACE_CACHE[key] = cached
    return cached


def _merge_splitk_partials_ref(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    *,
    batch: int,
    num_splits: int,
    return_lse: bool,
):
    out_partial = out_partial.view(num_splits, batch, 1, out_partial.shape[-2], out_partial.shape[-1])
    lse_partial = lse_partial.view(num_splits, batch, lse_partial.shape[-2], 1)
    lse_max = lse_partial.amax(dim=0, keepdim=True)
    lse_weights = torch.exp(lse_partial - lse_max)
    out = (out_partial * lse_weights.unsqueeze(2)).sum(dim=0) / lse_weights.sum(dim=0).unsqueeze(1)
    if not return_lse:
        return out
    lse = lse_max.squeeze(0) + torch.log(lse_weights.sum(dim=0))
    return out, lse


def _merge_splitk_partials(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    *,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    batch: int,
    num_splits: int,
):
    merged = _merge_splitk_partials_ref(
        out_partial,
        lse_partial,
        batch=batch,
        num_splits=num_splits,
        return_lse=lse is not None,
    )
    if lse is None:
        out.copy_(merged)
        return out
    out_merged, lse_merged = merged
    out.copy_(out_merged)
    lse.copy_(lse_merged)
    return out, lse


def _define_decode_base_class(fa4_root: str | None = None):
    # JetLong decode reuses the FA4 SM90 decode pipeline/epilogue structure.
    # Keep this as a private base class rather than exposing a second public
    # generic decode API from JetLM.
    ns = _lazy_imports(fa4_root)
    cutlass = ns["cutlass"]
    cute = ns["cute"]
    Float32 = ns["Float32"]
    Int32 = ns["Int32"]
    cpasync = ns["cpasync"]
    warpgroup = ns["warpgroup"]
    pipeline = ns["pipeline"]
    pipeline_init_arrive = ns["pipeline_init_arrive"]
    pipeline_init_wait = ns["pipeline_init_wait"]
    Arch = ns["Arch"]
    copy_utils = ns["copy_utils"]
    layout_utils = ns["layout_utils"]
    sm90_utils = ns["sm90_utils"]
    FlashAttentionForwardSm90 = ns["FlashAttentionForwardSm90"]
    AttentionMask = ns["AttentionMask"]
    NamedBarrierFwd = ns["NamedBarrierFwd"]
    make_packgqa_tiled_tma_atom = ns["make_packgqa_tiled_tma_atom"]
    pack_gqa_layout = ns["pack_gqa_layout"]
    SeqlenInfoQK = ns["SeqlenInfoQK"]
    Softmax = ns["Softmax"]
    utils = ns["utils"]
    pipeline_custom = ns["pipeline_custom"]

    class FlashAttentionDecodeSm90(FlashAttentionForwardSm90):
        def __init__(
            self,
            *args,
            tile_m: int = 64,
            tile_n: int = 128,
            num_stages: int = 3,
            num_threads: int = 160,
            **kwargs,
        ):
            super().__init__(
                *args,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=num_stages,
                num_threads=num_threads,
                intra_wg_overlap=True,
                mma_pv_is_rs=True,
                **kwargs,
            )

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mO: cute.Tensor,
            mLSE: Optional[cute.Tensor],
            softmax_scale: Float32,
            mSeqUsedK: Optional[cute.Tensor] = None,
            num_splits: Int32 = 1,
            stream=None,
        ):
            self._check_type(
                *(
                    t.element_type if t is not None else None
                    for t in (mQ, mK, mV, mO, mLSE, None, None, None, mSeqUsedK)
                )
            )
            assert self.arch >= Arch.sm_90 and self.arch <= Arch.sm_90a
            assert mSeqUsedK is not None, "decode kernel requires seqused_k"

            mQ, mK, mV, mO = [t for t in (mQ, mK, mV, mO)]
            mQ, mO = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mQ, mO)]
            mK, mV = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mK, mV)]
            mLSE = layout_utils.select(mLSE, [2, 1, 0]) if cutlass.const_expr(mLSE is not None) else None
            mQ_og, mO_og = mQ, mO

            tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
            self.num_mma_threads = tiled_mma_qk.size
            self.num_threads_per_warp_group = 128
            self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
            assert self.num_wg_mma == 1
            self.num_producer_threads = self.num_threads_per_warp_group
            self.num_Q_load_threads = self.num_threads_per_warp_group
            self.num_epilogue_threads = self.num_mma_threads
            self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
            self.num_mma_regs, self.num_producer_regs = 256, 56
            self.use_scheduler_barrier = False
            self.use_tma_Q = True
            self.use_tma_KV = True
            self.use_tma_O = True
            self.rescale_O_before_gemm = False
            self.varlen_q = False
            self.use_block_sparsity = False
            self._setup_attributes()

            if cutlass.const_expr(self.pack_gqa):
                nheads_kv = mK.shape[2]
                mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
                mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
                if cutlass.const_expr(mLSE is not None):
                    mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

            make_tiled_tma_atom_fn = (
                partial(
                    make_packgqa_tiled_tma_atom,
                    qhead_per_kvhead=self.qhead_per_kvhead,
                    head_idx=2,
                )
                if cutlass.const_expr(self.pack_gqa)
                else cpasync.make_tiled_tma_atom
            )

            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                cpasync.CopyBulkTensorTileG2SOp(),
                mQ_og if cutlass.const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),
            )
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,
            )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                cpasync.CopyBulkTensorTileS2GOp(),
                mO_og if cutlass.const_expr(self.pack_gqa) else mO,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),
            )
            self.tma_copy_bytes = {
                name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
                for name, mX, layout in [
                    ("Q", mQ, self.sQ_layout),
                    ("K", mK, self.sK_layout),
                    ("V", mV, self.sV_layout),
                ]
            }
            SharedStorage = self._get_shared_storage_cls()

            grid = [mQ.shape[2], mQ.shape[3], num_splits]
            self.kernel(
                tma_tensor_Q,
                tma_tensor_K,
                tma_tensor_V,
                tma_tensor_O,
                mLSE,
                *utils.compute_softmax_scale_log2(softmax_scale, None),
                mSeqUsedK,
                num_splits,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_O,
                self.sQ_layout,
                self.sK_layout,
                self.sV_layout,
                self.sO_layout,
                self.sP_layout,
                self.gmem_tiled_copy_O,
                tiled_mma_qk,
                tiled_mma_pv,
                SharedStorage,
            ).launch(
                grid=grid,
                block=[self.num_threads, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mO: cute.Tensor,
            mLSE: Optional[cute.Tensor],
            softmax_scale_log2: Float32,
            softmax_scale: Optional[Float32],
            mSeqUsedK: cute.Tensor,
            num_splits: Int32,
            tma_atom_Q: cute.CopyAtom,
            tma_atom_K: cute.CopyAtom,
            tma_atom_V: cute.CopyAtom,
            tma_atom_O: cute.CopyAtom,
            sQ_layout: cute.ComposedLayout,
            sK_layout: cute.ComposedLayout,
            sV_layout: cute.ComposedLayout,
            sO_layout: cute.ComposedLayout,
            sP_layout: cute.ComposedLayout | None,
            gmem_tiled_copy_O: cute.TiledCopy,
            tiled_mma_qk: cute.TiledMma,
            tiled_mma_pv: cute.TiledMma,
            SharedStorage: cutlass.Constexpr[Callable],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            head_idx, batch_idx, split_idx = cute.arch.block_idx()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 0:
                for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                    cpasync.prefetch_descriptor(tma_atom)

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            sVt = layout_utils.transpose_view(sV)
            sP = None
            if cutlass.const_expr(sP_layout is not None):
                sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
            sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_Q.data_ptr(),
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
                ),
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
                ),
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
                ),
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )

            pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            seqlen = SeqlenInfoQK.create(
                batch_idx=batch_idx,
                seqlen_q_static=mQ.shape[0] if cutlass.const_expr(not self.pack_gqa) else mQ.shape[0][1],
                seqlen_k_static=mK.shape[0],
                mSeqUsedK=mSeqUsedK,
            )
            n_block_total = cute.ceil_div(seqlen.seqlen_k, self.tile_n)
            num_blocks_per_split = (
                Int32(0)
                if n_block_total <= 0
                else (n_block_total + num_splits - 1) // num_splits
            )
            n_block_min = split_idx * num_blocks_per_split
            n_block_max = cutlass.min(n_block_min + num_blocks_per_split, n_block_total)
            has_work = n_block_max > n_block_min

            if has_work and warp_idx < 4:
                cute.arch.setmaxregister_decrease(self.num_producer_regs)
                self.decode_load(
                    mQ,
                    mK,
                    mV,
                    sQ,
                    sK,
                    sV,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    pipeline_q,
                    pipeline_k,
                    pipeline_v,
                    batch_idx,
                    head_idx,
                    seqlen,
                    n_block_min,
                    n_block_max,
                )
            elif has_work:
                cute.arch.setmaxregister_increase(self.num_mma_regs)
                self.decode_mma(
                    mO,
                    mLSE,
                    sQ,
                    sK,
                    sVt,
                    sP,
                    sO,
                    pipeline_q,
                    pipeline_k,
                    pipeline_v,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    tidx - self.num_threads_per_warp_group,
                    softmax_scale_log2,
                    softmax_scale,
                    batch_idx,
                    head_idx,
                    seqlen,
                    split_idx,
                    num_splits,
                    mSeqUsedK.shape[0],
                    n_block_min,
                    n_block_max,
                    n_block_total,
                )

        @cute.jit
        def decode_load(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            sQ: cute.Tensor,
            sK: cute.Tensor,
            sV: cute.Tensor,
            tma_atom_Q: cute.CopyAtom,
            tma_atom_K: cute.CopyAtom,
            tma_atom_V: cute.CopyAtom,
            pipeline_q: pipeline.PipelineAsync,
            pipeline_k: pipeline.PipelineAsync,
            pipeline_v: pipeline.PipelineAsync,
            batch_idx: Int32,
            head_idx: Int32,
            seqlen: SeqlenInfoQK,
            n_block_min: Int32,
            n_block_max: Int32,
        ):
            warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
            q_state = Int32(1)
            kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            head_idx_kv = head_idx if cutlass.const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
            gQ = cute.local_tile(mQ[None, None, head_idx, batch_idx], (self.tile_m, self.tile_hdim), (0, 0))
            gK = cute.local_tile(mK[None, None, head_idx_kv, batch_idx], (self.tile_n, self.tile_hdim), (None, 0))
            gV = cute.local_tile(mV[None, None, head_idx_kv, batch_idx], (self.tile_n, self.tile_hdimv), (None, 0))
            load_Q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True)
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), gK, sK
            )
            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V, 0, cute.make_layout(1), gV, sV
            )
            load_K = copy_utils.tma_producer_copy_fn(load_K, pipeline_k)
            load_V = copy_utils.tma_producer_copy_fn(load_V, pipeline_v)

            if warp_idx_in_wg == 0:
                pipeline_k.producer_acquire(kv_state)
                load_K(src_idx=n_block_max - 1, producer_state=kv_state)
                pipeline_q.producer_acquire_w_index_phase(0, q_state)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                q_state ^= 1
                pipeline_v.producer_acquire(kv_state)
                load_V(src_idx=n_block_max - 1, producer_state=kv_state)
                kv_state.advance()

                for i in cutlass.range(n_block_max - n_block_min - 1, unroll=1):
                    n_block = n_block_max - 2 - i
                    pipeline_k.producer_acquire(kv_state)
                    load_K(src_idx=n_block, producer_state=kv_state)
                    pipeline_v.producer_acquire(kv_state)
                    load_V(src_idx=n_block, producer_state=kv_state)
                    kv_state.advance()
                pipeline_v.producer_tail(kv_state)

        @cute.jit
        def decode_mma(
            self,
            mO: cute.Tensor,
            mLSE: Optional[cute.Tensor],
            sQ: cute.Tensor,
            sK: cute.Tensor,
            sVt: cute.Tensor,
            sP: Optional[cute.Tensor],
            sO: cute.Tensor,
            pipeline_q: pipeline.PipelineAsync,
            pipeline_k: pipeline.PipelineAsync,
            pipeline_v: pipeline.PipelineAsync,
            gmem_tiled_copy_O: cute.TiledCopy,
            tma_atom_O: cute.CopyAtom,
            tiled_mma_qk: cute.TiledMma,
            tiled_mma_pv: cute.TiledMma,
            tidx: Int32,
            softmax_scale_log2: Float32,
            softmax_scale: Optional[Float32],
            batch_idx: Int32,
            head_idx: Int32,
            seqlen: SeqlenInfoQK,
            split_idx: Int32,
            num_splits: Int32,
            num_batch: Int32,
            n_block_min: Int32,
            n_block_max: Int32,
            n_block_total: Int32,
        ):
            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            wg_mma_qk = tiled_mma_qk.get_slice(cute.make_layout(1)(0))
            wg_mma_pv = tiled_mma_pv.get_slice(cute.make_layout(1)(0))
            _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
                wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
            )
            mma_qk_fn = partial(
                sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
            )
            acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
                wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
            )
            mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

            smem_copy_atom_P = utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, self.dtype)
            smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
            tPsP = smem_thr_copy_P.partition_D(sP) if cutlass.const_expr(sP is not None) else None

            self.mma_init()
            q_state = Int32(0)
            kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            pipeline_q.consumer_wait_w_index_phase(0, q_state)

            softmax = Softmax.create(
                softmax_scale_log2,
                num_rows=acc_O.shape[0][0] * acc_O.shape[1],
                softmax_scale=softmax_scale,
            )

            mask = AttentionMask(
                self.tile_m,
                self.tile_n,
                seqlen,
                qhead_per_kvhead_packgqa=self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1,
            )
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=0,
                thr_mma=thr_mma_qk,
                mask_causal=False,
                mask_local=False,
            )

            smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)
            scores_scale = None
            mma_one_n_block_all = partial(
                self.mma_one_n_block_intrawg_overlap
                if cutlass.const_expr(self.intra_wg_overlap)
                else self.mma_one_n_block,
                mma_qk_fn=mma_qk_fn,
                pipeline_k=pipeline_k,
                pipeline_v=pipeline_v,
                acc_O=acc_O,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                check_inf=True,
                scores_scale=scores_scale,
            )
            process_first_half_block = partial(
                self.first_half_block_overlap,
                mma_qk_fn=mma_qk_fn,
                pipeline_k=pipeline_k,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                scores_scale=scores_scale,
                softmax=softmax,
                acc_O=acc_O,
            )
            process_last_half_block = partial(
                self.last_half_block_overlap,
                pipeline_v=pipeline_v,
                mma_pv_fn=mma_pv_fn,
                scores_scale=scores_scale,
                softmax=softmax,
                acc_O=acc_O,
            )
            mma_one_n_block = partial(mma_one_n_block_all, seqlen=seqlen, softmax=softmax, score_mod_fn=None)

            o_should_accumulate = False
            if cutlass.const_expr(self.intra_wg_overlap):
                kv_state = process_first_half_block(
                    n_block=n_block_max - 1,
                    seqlen=seqlen,
                    kv_consumer_state=kv_state,
                    mask_fn=partial(mask_fn, mask_seqlen=n_block_max == n_block_total),
                    score_mod_fn=None,
                    is_first_block=True,
                )
            else:
                self.warp_scheduler_barrier_sync()
                kv_state = mma_one_n_block(
                    kv_state,
                    n_block=n_block_max - 1,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                    is_first_n_block=True,
                    mask_fn=partial(mask_fn, mask_seqlen=n_block_max == n_block_total),
                )
                o_should_accumulate = True
            n_block_last = n_block_max - 1
            for n_tile in cutlass.range(n_block_last - n_block_min, unroll=1):
                kv_state = mma_one_n_block(
                    kv_state,
                    n_block=n_block_last - 1 - n_tile,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not o_should_accumulate),
                    mask_fn=None,
                )
                o_should_accumulate = True
            pipeline_q.consumer_release_w_index(0)
            if cutlass.const_expr(self.intra_wg_overlap):
                kv_state = process_last_half_block(
                    kv_consumer_state=kv_state,
                    zero_init=not o_should_accumulate,
                )
            else:
                self.warp_scheduler_barrier_arrive()
            q_state ^= 1

            row_scale = softmax.finalize()
            softmax.rescale_O(acc_O, row_scale)
            batch_out_idx = batch_idx + split_idx * num_batch
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                SeqlenInfoQK.create(
                    batch_idx=batch_out_idx,
                    seqlen_q_static=seqlen.seqlen_q,
                    seqlen_k_static=seqlen.seqlen_q,
                ),
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                0,
                head_idx,
                batch_out_idx,
            )

    return FlashAttentionDecodeSm90, ns


def _define_jetlong_kernel_class(
    fa4_root: str | None = None,
    *,
    k_mode: str = "consumer",
    intra_kernel_reduce: bool = False,
    cluster_splits: int = 1,
    apply_distant_k_correction: bool = True,
    use_dynamic_metadata: bool = False,
    jetlong_meta_has_active_splits: bool = False,
    use_dual_branch_accumulator: bool = True,
    mma_regs: int = 192,
):
    FlashAttentionDecodeSm90, ns = _define_decode_base_class(fa4_root)
    cutlass = ns["cutlass"]
    cute = ns["cute"]
    Float32 = ns["Float32"]
    Int32 = ns["Int32"]
    cpasync = ns["cpasync"]
    warpgroup = ns["warpgroup"]
    pipeline = ns["pipeline"]
    pipeline_init_arrive = ns["pipeline_init_arrive"]
    pipeline_init_wait = ns["pipeline_init_wait"]
    Arch = ns["Arch"]
    layout_utils = ns["layout_utils"]
    sm90_utils = ns["sm90_utils"]
    copy_utils = ns["copy_utils"]
    utils = ns["utils"]
    pipeline_custom = ns["pipeline_custom"]
    SeqlenInfoQK = ns["SeqlenInfoQK"]
    Softmax = ns["Softmax"]
    make_packgqa_tiled_tma_atom = ns["make_packgqa_tiled_tma_atom"]
    pack_gqa_layout = ns["pack_gqa_layout"]
    mask_r2p_lambda = ns["mask_r2p_lambda"]
    r2p_bitmask_below = ns["r2p_bitmask_below"]
    sm90_col_to_r2p_idx = ns["sm90_col_to_r2p_idx"]
    NamedBarrierFwd = ns["NamedBarrierFwd"]
    k_mode_map = {"consumer": 0, "producer": 1, "manual": 2, "consumer_near_offset": 3}
    if k_mode not in k_mode_map:
        raise ValueError(f"Unsupported JetLong decode k_mode: {k_mode}")
    k_mode_idx = k_mode_map[k_mode]

    class FlashAttentionDecodeJetLongSm90(FlashAttentionDecodeSm90):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.jetlong_k_mode = k_mode_idx
            self.intra_kernel_reduce = intra_kernel_reduce
            self.cluster_splits = cluster_splits
            self.apply_distant_k_correction = apply_distant_k_correction
            self.use_dynamic_metadata = use_dynamic_metadata
            self.jetlong_meta_has_active_splits = jetlong_meta_has_active_splits
            self.use_dual_branch_accumulator = use_dual_branch_accumulator
            self.requested_mma_regs = mma_regs

        def _get_shared_storage_cls(self):
            assert not self.Q_in_regs, "JetLong decode kernel assumes Q_in_regs=False"
            sQ_struct, sK_struct, sV_struct = [
                cute.struct.Align[
                    cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
                ]
                for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
            ]
            cosize_sP = cute.cosize(self.sP_layout) if cutlass.const_expr(self.sP_layout is not None) else 0
            sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
            mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mbar_ptr_K_ready_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            split_scale_count = self.cluster_splits * (self.qhead_per_kvhead if self.pack_gqa else 1)
            split_scale_struct = cute.struct.Align[cute.struct.MemRange[Float32, split_scale_count], 16]

            @cute.struct
            class SharedStorageQDualKV:
                mbar_ptr_Q_base: mbar_ptr_Q_struct
                mbar_ptr_Q_group: mbar_ptr_Q_struct
                mbar_ptr_K: mbar_ptr_K_struct
                mbar_ptr_K_ready: mbar_ptr_K_ready_struct
                mbar_ptr_V: mbar_ptr_V_struct
                sV: sV_struct
                sQ_base: sQ_struct
                sQ_group: sQ_struct
                sK: sK_struct
                sP: sP_struct
                sSplitScale: split_scale_struct

            return SharedStorageQDualKV

        @cute.jit
        def apply_mixed_tail_mask(
            self,
            acc_S: cute.Tensor,
            n_block: Int32,
            thr_mma,
            n_block_total_distant: Int32,
            seqlen_distant: Int32,
            seqlen_near: Int32,
            mask_seqlen: cutlass.Constexpr[bool] = True,
        ):
            if mask_seqlen:
                local_n_block = n_block
                valid_cols = seqlen_distant
                if n_block >= n_block_total_distant:
                    local_n_block = n_block - n_block_total_distant
                    valid_cols = seqlen_near
                valid_cols = valid_cols - local_n_block * self.tile_n
                if valid_cols < self.tile_n:
                    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
                    thr_col_offset = tScS_mn[0][1]
                    valid_cols_r2p = sm90_col_to_r2p_idx(valid_cols - thr_col_offset)
                    mask_r2p_lambda(acc_S_mn, lambda s: r2p_bitmask_below(valid_cols_r2p, s))

        @cute.jit
        def finalize_softmax_div(self, softmax: Softmax) -> cute.Tensor:
            row_sum = softmax.row_sum
            row_max = softmax.row_max
            scale_log2 = softmax.scale_log2
            row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
            row_scale = cute.make_fragment_like(row_max, Float32)
            LN2 = math.log(2.0)
            for r in cutlass.range(cute.size(row_sum), unroll_full=True):
                bad_sum = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
                row_scale[r] = Float32(0.0) if bad_sum else Float32(1.0) / row_sum[r]
                row_sum_cur = row_sum[r]
                row_sum[r] = (
                    (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=True)) * LN2
                    if not bad_sum
                    else -Float32.inf
                )
            return row_scale

        @cute.jit
        def apply_jetlong_k_transform_smem(
            self,
            sK: cute.Tensor,
            stage: Int32,
            n_block: Int32,
            seqlen_distant: Int32,
            tidx: Int32,
            mInvFreq: cute.Tensor,
            group_size: Int32,
        ):
            half = self.tile_hdim // 2
            total = self.tile_n * half
            loop_count = cute.ceil_div(total, self.num_mma_threads)

            for i in cutlass.range(loop_count, unroll=1):
                linear = tidx + i * self.num_mma_threads
                if linear < total:
                    n = linear // half
                    d = linear - n * half
                    kv_idx = n_block * self.tile_n + n
                    if kv_idx < seqlen_distant:
                        delta_k = kv_idx // group_size - kv_idx
                        angle = Float32(delta_k) * Float32(mInvFreq[d])
                        cosv = cute.cos(angle, fastmath=True)
                        sinv = cute.sin(angle, fastmath=True)
                        x0 = Float32(sK[n, d, stage])
                        x1 = Float32(sK[n, d + half, stage])
                        sK[n, d, stage] = (x0 * cosv - x1 * sinv).to(self.dtype)
                        sK[n, d + half, stage] = (x1 * cosv + x0 * sinv).to(self.dtype)

            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PFull),
                number_of_threads=self.num_mma_threads,
            )

        @cute.jit
        def load_jetlong_k_tile_from_gmem(
            self,
            gK: cute.Tensor,
            sK: cute.Tensor,
            stage: Int32,
            n_block: Int32,
            seqlen_k: Int32,
            tidx: Int32,
            mInvFreq: cute.Tensor,
            group_size: Int32,
            apply_grouped_rotate: cutlass.Constexpr[bool],
        ):
            half = self.tile_hdim // 2
            total = self.tile_n * half
            loop_count = cute.ceil_div(total, self.num_producer_threads)

            for i in cutlass.range(loop_count, unroll=1):
                linear = tidx + i * self.num_producer_threads
                if linear < total:
                    n = linear // half
                    d = linear - n * half
                    kv_idx = n_block * self.tile_n + n
                    x0 = Float32(0.0)
                    x1 = Float32(0.0)
                    if kv_idx < seqlen_k:
                        x0 = Float32(gK[n, d, n_block])
                        x1 = Float32(gK[n, d + half, n_block])
                        if cutlass.const_expr(apply_grouped_rotate):
                            delta_k = kv_idx // group_size - kv_idx
                            angle = Float32(delta_k) * Float32(mInvFreq[d])
                            cosv = cute.cos(angle, fastmath=True)
                            sinv = cute.sin(angle, fastmath=True)
                            y0 = x0 * cosv - x1 * sinv
                            y1 = x1 * cosv + x0 * sinv
                            x0, x1 = y0, y1
                    sK[n, d, stage] = x0.to(self.dtype)
                    sK[n, d + half, stage] = x1.to(self.dtype)

            cute.arch.fence_view_async_shared()

        @cute.jit
        def load_jetlong_k_tile_from_tensor(
            self,
            mK: cute.Tensor,
            sK: cute.Tensor,
            stage: Int32,
            n_block: Int32,
            kv_start: Int32,
            seqlen_k: Int32,
            tidx: Int32,
            mInvFreq: cute.Tensor,
            group_size: Int32,
            head_idx_kv: Int32,
            batch_idx: Int32,
            apply_grouped_rotate: cutlass.Constexpr[bool],
        ):
            half = self.tile_hdim // 2
            total = self.tile_n * half
            loop_count = cute.ceil_div(total, self.num_producer_threads)
            kv_tile_start = kv_start + n_block * self.tile_n
            tile_offset = (
                kv_tile_start * mK.stride[0]
                + head_idx_kv * mK.stride[2]
                + batch_idx * mK.stride[3]
            )
            gK_tile = cute.make_tensor(
                mK.iterator + tile_offset,
                cute.make_layout((self.tile_n, self.tile_hdim), stride=(mK.stride[0], mK.stride[1])),
            )

            for i in cutlass.range(loop_count, unroll=1):
                linear = tidx + i * self.num_producer_threads
                if linear < total:
                    n = linear // half
                    d = linear - n * half
                    kv_rel = n_block * self.tile_n + n
                    x0 = Float32(0.0)
                    x1 = Float32(0.0)
                    if kv_rel < seqlen_k:
                        x0 = Float32(gK_tile[n, d])
                        x1 = Float32(gK_tile[n, d + half])
                        if cutlass.const_expr(apply_grouped_rotate):
                            delta_k = kv_rel // group_size - kv_rel
                            angle = Float32(delta_k) * Float32(mInvFreq[d])
                            cosv = cute.cos(angle, fastmath=True)
                            sinv = cute.sin(angle, fastmath=True)
                            y0 = x0 * cosv - x1 * sinv
                            y1 = x1 * cosv + x0 * sinv
                            x0, x1 = y0, y1
                    sK[n, d, stage] = x0.to(self.dtype)
                    sK[n, d + half, stage] = x1.to(self.dtype)

            cute.arch.fence_view_async_shared()

        @cute.jit
        def load_jetlong_v_tile_from_tensor(
            self,
            mV: cute.Tensor,
            sV: cute.Tensor,
            stage: Int32,
            n_block: Int32,
            kv_start: Int32,
            seqlen_k: Int32,
            tidx: Int32,
            head_idx_kv: Int32,
            batch_idx: Int32,
        ):
            total = self.tile_n * self.tile_hdimv
            loop_count = cute.ceil_div(total, self.num_producer_threads)
            kv_tile_start = kv_start + n_block * self.tile_n
            tile_offset = (
                kv_tile_start * mV.stride[0]
                + head_idx_kv * mV.stride[2]
                + batch_idx * mV.stride[3]
            )
            gV_tile = cute.make_tensor(
                mV.iterator + tile_offset,
                cute.make_layout((self.tile_n, self.tile_hdimv), stride=(mV.stride[0], mV.stride[1])),
            )

            for i in cutlass.range(loop_count, unroll=1):
                linear = tidx + i * self.num_producer_threads
                if linear < total:
                    n = linear // self.tile_hdimv
                    d = linear - n * self.tile_hdimv
                    kv_rel = n_block * self.tile_n + n
                    value = Float32(0.0)
                    if kv_rel < seqlen_k:
                        value = Float32(gV_tile[n, d])
                    sV[n, d, stage] = value.to(self.dtype)

            cute.arch.fence_view_async_shared()

        @cute.jit
        def release_k_pipelines(
            self,
            pipeline_k: pipeline.PipelineAsync,
            smem_pipe_read,
            pipeline_k_raw=None,
        ):
            pipeline_k.consumer_release(smem_pipe_read)
            if cutlass.const_expr(self.jetlong_k_mode == 1):
                pipeline_k_raw.consumer_release(smem_pipe_read)

        @cute.jit
        def first_half_block_overlap(
            self,
            n_block: Int32,
            mma_qk_fn: Callable,
            kv_consumer_state,
            pipeline_k,
            pipeline_k_raw,
            tOrP: cute.Tensor,
            smem_copy_params: SimpleNamespace,
            softmax: Softmax,
            seqlen: SeqlenInfoQK,
            scores_scale: Optional[cute.Tensor] = None,
            acc_O: Optional[cute.Tensor] = None,
            mask_fn: Callable = None,
            score_mod_fn: Optional[Callable] = None,
            is_first_block: bool = False,
            sK: Optional[cute.Tensor] = None,
            tidx: Optional[Int32] = None,
            n_block_total_distant: Optional[Int32] = None,
            seqlen_distant: Optional[Int32] = None,
            mInvFreq: Optional[cute.Tensor] = None,
            group_size: Optional[Int32] = None,
        ):
            pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
            if (
                cutlass.const_expr(self.jetlong_k_mode == 0 and self.apply_distant_k_correction)
                and n_block < n_block_total_distant
            ):
                self.apply_jetlong_k_transform_smem(
                    sK, kv_consumer_state.index, n_block, seqlen_distant, tidx, mInvFreq, group_size
                )
            acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
            self.release_k_pipelines(pipeline_k, kv_consumer_state, pipeline_k_raw)

            if cutlass.const_expr(score_mod_fn is not None):
                score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
            mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

            row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)
            tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
            tOrP_cur = (
                tOrP
                if cutlass.const_expr(self.mma_pv_is_rs)
                else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
            )
            tOrP_cur.store(tOrP_acc.load().to(self.dtype))

            if cutlass.const_expr(not self.mma_pv_is_rs):
                tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
                cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()

            if cutlass.const_expr(self.rescale_O_before_gemm):
                acc_O.fill(0.0)
                scores_scale.store(row_scale.load())

            return kv_consumer_state

        @cute.jit
        def mma_one_n_block(
            self,
            smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
            n_block: Int32,
            mma_qk_fn: Callable,
            mma_pv_fn: Callable,
            pipeline_k: pipeline.PipelineAsync,
            pipeline_k_raw,
            pipeline_v: pipeline.PipelineAsync,
            acc_O: cute.Tensor,
            tOrP: cute.Tensor,
            smem_copy_params: SimpleNamespace,
            softmax: Softmax,
            seqlen: SeqlenInfoQK,
            scores_scale: Optional[cute.Tensor] = None,
            score_mod_fn: Optional[Callable] = None,
            mask_fn: Optional[Callable] = None,
            is_first_n_block: cutlass.Constexpr = False,
            check_inf: cutlass.Constexpr = True,
            sK: Optional[cute.Tensor] = None,
            tidx: Optional[Int32] = None,
            n_block_total_distant: Optional[Int32] = None,
            seqlen_distant: Optional[Int32] = None,
            mInvFreq: Optional[cute.Tensor] = None,
            group_size: Optional[Int32] = None,
        ):
            pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
            if (
                cutlass.const_expr(self.jetlong_k_mode == 0 and self.apply_distant_k_correction)
                and n_block < n_block_total_distant
            ):
                self.apply_jetlong_k_transform_smem(
                    sK, smem_pipe_read.index, n_block, seqlen_distant, tidx, mInvFreq, group_size
                )
            acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
            self.warp_scheduler_barrier_arrive()
            warpgroup.wait_group(0)
            self.release_k_pipelines(pipeline_k, smem_pipe_read, pipeline_k_raw)

            if cutlass.const_expr(score_mod_fn is not None):
                score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
            if cutlass.const_expr(mask_fn is not None):
                mask_fn(acc_S=acc_S, n_block=n_block)

            row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
            tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
            tOrP_cur = (
                tOrP
                if cutlass.const_expr(self.mma_pv_is_rs)
                else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
            )
            utils.cvt_f16(tOrP_acc, tOrP_cur)
            if cutlass.const_expr(not self.mma_pv_is_rs):
                tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
                cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            softmax.rescale_O(acc_O, row_scale)
            if cutlass.const_expr(not self.mma_pv_is_rs):
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
            pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
            self.warp_scheduler_barrier_sync()
            mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
            pipeline_v.consumer_release(smem_pipe_read)
            smem_pipe_read.advance()
            return smem_pipe_read

        @cute.jit
        def mma_one_n_block_intrawg_overlap(
            self,
            smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
            n_block: Int32,
            mma_qk_fn: Callable,
            mma_pv_fn: Callable,
            pipeline_k: pipeline.PipelineAsync,
            pipeline_k_raw,
            pipeline_v: pipeline.PipelineAsync,
            acc_O: cute.Tensor,
            tOrP: cute.Tensor,
            smem_copy_params: SimpleNamespace,
            softmax: Softmax,
            seqlen: SeqlenInfoQK,
            scores_scale: Optional[cute.Tensor] = None,
            score_mod_fn: Optional[Callable] = None,
            mask_fn: Optional[Callable] = None,
            check_inf: cutlass.Constexpr = True,
            sK: Optional[cute.Tensor] = None,
            tidx: Optional[Int32] = None,
            n_block_total_distant: Optional[Int32] = None,
            seqlen_distant: Optional[Int32] = None,
            mInvFreq: Optional[cute.Tensor] = None,
            group_size: Optional[Int32] = None,
        ):
            smem_pipe_read_v = smem_pipe_read.clone()
            smem_pipe_read.advance()
            pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
            if (
                cutlass.const_expr(self.jetlong_k_mode == 0 and self.apply_distant_k_correction)
                and n_block < n_block_total_distant
            ):
                self.apply_jetlong_k_transform_smem(
                    sK, smem_pipe_read.index, n_block, seqlen_distant, tidx, mInvFreq, group_size
                )
            self.warp_scheduler_barrier_sync()
            acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
            if cutlass.const_expr(self.rescale_O_before_gemm):
                softmax.rescale_O(acc_O, scores_scale)
            pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))
            mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
            self.warp_scheduler_barrier_arrive()
            warpgroup.wait_group(1)
            self.release_k_pipelines(pipeline_k, smem_pipe_read, pipeline_k_raw)

            if cutlass.const_expr(score_mod_fn is not None):
                score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
            if cutlass.const_expr(mask_fn is not None):
                mask_fn(acc_S=acc_S, n_block=n_block)

            row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
            warpgroup.wait_group(0)
            pipeline_v.consumer_release(smem_pipe_read_v)
            tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
            tOrP_cur = (
                tOrP
                if cutlass.const_expr(self.mma_pv_is_rs)
                else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
            )
            utils.cvt_f16(tOrP_acc, tOrP_cur)
            if cutlass.const_expr(not self.mma_pv_is_rs):
                tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
                cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            if cutlass.const_expr(not self.rescale_O_before_gemm):
                softmax.rescale_O(acc_O, row_scale)
            if cutlass.const_expr(self.rescale_O_before_gemm):
                scores_scale.store(row_scale.load())
            if cutlass.const_expr(not self.mma_pv_is_rs):
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
            return smem_pipe_read

        @cute.jit
        def __call__(
            self,
            mQBase: cute.Tensor,
            mQGroup: cute.Tensor,
            mKNear: cute.Tensor,
            mVNear: cute.Tensor,
            mKDistant: cute.Tensor,
            mVDistant: cute.Tensor,
            mO: cute.Tensor,
            mLSE: Optional[cute.Tensor],
            mOFinal: Optional[cute.Tensor],
            mLSEFinal: Optional[cute.Tensor],
            softmax_scale: Float32,
            mSeqUsedNear: cute.Tensor,
            mSeqUsedDistant: cute.Tensor,
            mInvFreq: cute.Tensor,
            mJetLongMeta: Optional[cute.Tensor],
            mActiveSplits: Optional[cute.Tensor],
            group_size: Int32,
            num_splits: Int32 = 1,
            stream=None,
        ):
            self._check_type(
                *(
                    t.element_type if t is not None else None
                    for t in (mQBase, mKNear, mVNear, mO, mLSE, None, None, None, mSeqUsedNear)
                )
            )
            assert self.arch >= Arch.sm_90 and self.arch <= Arch.sm_90a

            mQBase, mQGroup, mKNear, mVNear, mKDistant, mVDistant, mO = [
                t for t in (mQBase, mQGroup, mKNear, mVNear, mKDistant, mVDistant, mO)
            ]
            mQBase, mQGroup, mO = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mQBase, mQGroup, mO)]
            mKNear, mVNear, mKDistant, mVDistant = [
                layout_utils.select(t, [1, 3, 2, 0]) for t in (mKNear, mVNear, mKDistant, mVDistant)
            ]
            mLSE = layout_utils.select(mLSE, [2, 1, 0]) if cutlass.const_expr(mLSE is not None) else None
            mO_direct = mO
            mLSE_direct = mLSE
            mOFinal = layout_utils.select(mOFinal, [1, 3, 2, 0]) if cutlass.const_expr(mOFinal is not None) else None
            mLSEFinal = (
                layout_utils.select(mLSEFinal, [2, 1, 0]) if cutlass.const_expr(mLSEFinal is not None) else None
            )
            mQBase_og, mQGroup_og, mO_og = mQBase, mQGroup, mO

            tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
            self.num_mma_threads = tiled_mma_qk.size
            self.num_threads_per_warp_group = 128
            self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
            assert self.num_wg_mma == 1
            self.num_producer_threads = self.num_threads_per_warp_group
            self.num_Q_load_threads = self.num_threads_per_warp_group
            self.num_epilogue_threads = self.num_mma_threads
            self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
            self.num_mma_regs, self.num_producer_regs = self.requested_mma_regs, 56
            self.use_scheduler_barrier = False
            self.intra_wg_overlap = False
            self.use_tma_Q = True
            self.use_tma_KV = True
            self.use_tma_O = True
            self.rescale_O_before_gemm = False
            self.varlen_q = False
            self.use_block_sparsity = False
            self._setup_attributes()

            if cutlass.const_expr(self.pack_gqa):
                nheads_kv = mKNear.shape[2]
                mQBase = pack_gqa_layout(mQBase, self.qhead_per_kvhead, nheads_kv, head_idx=2)
                mQGroup = pack_gqa_layout(mQGroup, self.qhead_per_kvhead, nheads_kv, head_idx=2)
                mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
                if cutlass.const_expr(mLSE is not None):
                    mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

            make_tiled_tma_atom_fn = (
                partial(
                    make_packgqa_tiled_tma_atom,
                    qhead_per_kvhead=self.qhead_per_kvhead,
                    head_idx=2,
                )
                if cutlass.const_expr(self.pack_gqa)
                else cpasync.make_tiled_tma_atom
            )
            tma_atom_Q_base, tma_tensor_Q_base = make_tiled_tma_atom_fn(
                cpasync.CopyBulkTensorTileG2SOp(),
                mQBase_og if cutlass.const_expr(self.pack_gqa) else mQBase,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),
            )
            tma_atom_Q_group, tma_tensor_Q_group = make_tiled_tma_atom_fn(
                cpasync.CopyBulkTensorTileG2SOp(),
                mQGroup_og if cutlass.const_expr(self.pack_gqa) else mQGroup,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),
            )
            tma_atom_K_near, tma_tensor_K_near = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mKNear,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_V_near, tma_tensor_V_near = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mVNear,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,
            )
            tma_atom_K_distant, tma_tensor_K_distant = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mKDistant,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_V_distant, tma_tensor_V_distant = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mVDistant,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,
            )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                cpasync.CopyBulkTensorTileS2GOp(),
                mO_og if cutlass.const_expr(self.pack_gqa) else mO,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),
            )
            self.tma_copy_bytes = {
                "Q": cute.size_in_bytes(mQBase.element_type, cute.select(self.sQ_layout, mode=[0, 1])),
                "K": cute.size_in_bytes(mKNear.element_type, cute.select(self.sK_layout, mode=[0, 1])),
                "V": cute.size_in_bytes(mVNear.element_type, cute.select(self.sV_layout, mode=[0, 1])),
            }
            SharedStorage = self._get_shared_storage_cls()

            grid = [mQBase.shape[2], mQBase.shape[3], num_splits]
            self.kernel(
                tma_tensor_Q_base,
                tma_tensor_Q_group,
                tma_tensor_K_near,
                tma_tensor_V_near,
                tma_tensor_K_distant,
                tma_tensor_V_distant,
                mKNear,
                mVNear,
                mKDistant,
                mVDistant,
                tma_tensor_O,
                mLSE,
                mO_direct,
                mLSE_direct,
                mOFinal,
                mLSEFinal,
                *utils.compute_softmax_scale_log2(softmax_scale, None),
                mSeqUsedNear,
                mSeqUsedDistant,
                mInvFreq,
                mJetLongMeta,
                mActiveSplits,
                group_size,
                num_splits,
                tma_atom_Q_base,
                tma_atom_Q_group,
                tma_atom_K_near,
                tma_atom_V_near,
                tma_atom_K_distant,
                tma_atom_V_distant,
                tma_atom_O,
                self.sQ_layout,
                self.sK_layout,
                self.sV_layout,
                self.sO_layout,
                self.sP_layout,
                self.gmem_tiled_copy_O,
                tiled_mma_qk,
                tiled_mma_pv,
                SharedStorage,
            ).launch(
                grid=grid,
                block=[self.num_threads, 1, 1],
                cluster=(1, 1, self.cluster_splits)
                if cutlass.const_expr(self.intra_kernel_reduce and self.cluster_splits > 1)
                else None,
                stream=stream,
                min_blocks_per_mp=1,
            )

        @cute.kernel
        def kernel(
            self,
            mQBase: cute.Tensor,
            mQGroup: cute.Tensor,
            mKNear: cute.Tensor,
            mVNear: cute.Tensor,
            mKDistant: cute.Tensor,
            mVDistant: cute.Tensor,
            mKNearRaw: cute.Tensor,
            mVNearRaw: cute.Tensor,
            mKDistantRaw: cute.Tensor,
            mVDistantRaw: cute.Tensor,
            mO: cute.Tensor,
            mLSE: Optional[cute.Tensor],
            mODirect: cute.Tensor,
            mLSEDirect: Optional[cute.Tensor],
            mOFinal: Optional[cute.Tensor],
            mLSEFinal: Optional[cute.Tensor],
            softmax_scale_log2: Float32,
            softmax_scale: Optional[Float32],
            mSeqUsedNear: cute.Tensor,
            mSeqUsedDistant: cute.Tensor,
            mInvFreq: cute.Tensor,
            mJetLongMeta: Optional[cute.Tensor],
            mActiveSplits: Optional[cute.Tensor],
            group_size: Int32,
            num_splits: Int32,
            tma_atom_Q_base: cute.CopyAtom,
            tma_atom_Q_group: cute.CopyAtom,
            tma_atom_K_near: cute.CopyAtom,
            tma_atom_V_near: cute.CopyAtom,
            tma_atom_K_distant: cute.CopyAtom,
            tma_atom_V_distant: cute.CopyAtom,
            tma_atom_O: cute.CopyAtom,
            sQ_layout: cute.ComposedLayout,
            sK_layout: cute.ComposedLayout,
            sV_layout: cute.ComposedLayout,
            sO_layout: cute.ComposedLayout,
            sP_layout: cute.ComposedLayout | None,
            gmem_tiled_copy_O: cute.TiledCopy,
            tiled_mma_qk: cute.TiledMma,
            tiled_mma_pv: cute.TiledMma,
            SharedStorage: cutlass.Constexpr[Callable],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            head_idx, batch_idx, split_idx = cute.arch.block_idx()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 0:
                for tma_atom in (
                    tma_atom_Q_base,
                    tma_atom_Q_group,
                    tma_atom_K_near,
                    tma_atom_V_near,
                    tma_atom_K_distant,
                    tma_atom_V_distant,
                    tma_atom_O,
                ):
                    cpasync.prefetch_descriptor(tma_atom)

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQBase = storage.sQ_base.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
            sQGroup = storage.sQ_group.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            sVt = layout_utils.transpose_view(sV)
            sSplitScale = storage.sSplitScale.get_tensor(
                cute.make_layout((self.cluster_splits, self.qhead_per_kvhead if self.pack_gqa else 1))
            )
            sP = None
            if cutlass.const_expr(sP_layout is not None):
                sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
            sO = storage.sQ_base.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

            ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
            tma_warp = ThreadCooperativeGroup(1)
            load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
            mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)

            pipeline_q_base = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_Q_base.data_ptr(),
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
            pipeline_q_group = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_Q_group.data_ptr(),
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
            if cutlass.const_expr(self.jetlong_k_mode == 2):
                pipeline_k = pipeline_custom.PipelineAsync.create(
                    barrier_storage=storage.mbar_ptr_K.data_ptr(),
                    num_stages=self.num_stages,
                    producer_group=load_threads,
                    consumer_group=mma_warps,
                    defer_sync=True,
                    elect_one_release=True,
                    syncwarp_before_release=False,
                )
                pipeline_k_raw = pipeline_k
            elif cutlass.const_expr(self.jetlong_k_mode == 1):
                pipeline_k_raw = pipeline_custom.PipelineTmaAsync.create(
                    barrier_storage=storage.mbar_ptr_K.data_ptr(),
                    num_stages=self.num_stages,
                    producer_group=tma_warp,
                    consumer_group=mma_warps,
                    tx_count=self.tma_copy_bytes["K"],
                    defer_sync=True,
                )
                pipeline_k = pipeline_custom.PipelineAsync.create(
                    barrier_storage=storage.mbar_ptr_K_ready.data_ptr(),
                    num_stages=self.num_stages,
                    producer_group=load_threads,
                    consumer_group=mma_warps,
                    defer_sync=True,
                    elect_one_release=True,
                    syncwarp_before_release=False,
                )
            else:
                pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                    barrier_storage=storage.mbar_ptr_K.data_ptr(),
                    num_stages=self.num_stages,
                    producer_group=tma_warp,
                    consumer_group=mma_warps,
                    tx_count=self.tma_copy_bytes["K"],
                    defer_sync=True,
                )
                pipeline_k_raw = pipeline_k
            if cutlass.const_expr(self.use_dynamic_metadata and self.jetlong_k_mode == 2):
                pipeline_v = pipeline_custom.PipelineAsync.create(
                    barrier_storage=storage.mbar_ptr_V.data_ptr(),
                    num_stages=self.num_stages,
                    producer_group=load_threads,
                    consumer_group=mma_warps,
                    defer_sync=True,
                    elect_one_release=True,
                    syncwarp_before_release=False,
                )
            else:
                pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                    barrier_storage=storage.mbar_ptr_V.data_ptr(),
                    num_stages=self.num_stages,
                    producer_group=tma_warp,
                    consumer_group=mma_warps,
                    tx_count=self.tma_copy_bytes["V"],
                    defer_sync=True,
                )

            pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            if cutlass.const_expr(self.use_dynamic_metadata):
                total_kv_len = mJetLongMeta[batch_idx, 0]
                seqlen_distant = mJetLongMeta[batch_idx, 1]
                group_size_cur = mJetLongMeta[batch_idx, 2]
                seqlen_near = total_kv_len - seqlen_distant
            else:
                seqlen_near = mSeqUsedNear[batch_idx]
                seqlen_distant = mSeqUsedDistant[batch_idx]
                group_size_cur = group_size
            n_block_total_distant = cute.ceil_div(seqlen_distant, self.tile_n)
            n_block_total_near = cute.ceil_div(seqlen_near, self.tile_n)
            n_block_total = n_block_total_distant + n_block_total_near
            active_splits = num_splits
            if cutlass.const_expr(mActiveSplits is not None):
                active_splits = mActiveSplits[batch_idx]
            elif cutlass.const_expr(self.jetlong_meta_has_active_splits):
                active_splits = mJetLongMeta[batch_idx, 4]
            if cutlass.const_expr(mActiveSplits is not None or self.jetlong_meta_has_active_splits):
                active_splits = cutlass.max(Int32(1), cutlass.min(active_splits, num_splits))
            num_blocks_per_split = Int32(0) if n_block_total <= 0 else (n_block_total + active_splits - 1) // active_splits
            n_block_min = split_idx * num_blocks_per_split
            n_block_max = cutlass.min(n_block_min + num_blocks_per_split, n_block_total)
            if split_idx >= active_splits:
                n_block_min = Int32(0)
                n_block_max = Int32(0)
            has_work = n_block_max > n_block_min

            if has_work and warp_idx < 4:
                cute.arch.setmaxregister_decrease(self.num_producer_regs)
                self.decode_load(
                    mQBase,
                    mQGroup,
                    mKNear,
                    mVNear,
                    mKDistant,
                    mVDistant,
                    mKNearRaw,
                    mVNearRaw,
                    mKDistantRaw,
                    mVDistantRaw,
                    sQBase,
                    sQGroup,
                    sK,
                    sV,
                    tma_atom_Q_base,
                    tma_atom_Q_group,
                    tma_atom_K_near,
                    tma_atom_V_near,
                    tma_atom_K_distant,
                    tma_atom_V_distant,
                    pipeline_q_base,
                    pipeline_q_group,
                    pipeline_k,
                    pipeline_k_raw,
                    pipeline_v,
                    batch_idx,
                    head_idx,
                    seqlen_near,
                    seqlen_distant,
                    mInvFreq,
                    group_size_cur,
                    n_block_total_distant,
                    n_block_min,
                    n_block_max,
                )
            elif has_work:
                cute.arch.setmaxregister_increase(self.num_mma_regs)
                self.decode_mma(
                    mO,
                    mLSE,
                    sQBase,
                    sQGroup,
                    sK,
                    sVt,
                    sP,
                    sO,
                    pipeline_q_base,
                    pipeline_q_group,
                    pipeline_k,
                    pipeline_k_raw,
                    pipeline_v,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    tidx - self.num_threads_per_warp_group,
                    softmax_scale_log2,
                    softmax_scale,
                    batch_idx,
                    head_idx,
                    split_idx,
                    num_splits,
                    mSeqUsedNear.shape[0],
                    seqlen_near,
                    seqlen_distant,
                    n_block_total_distant,
                    n_block_min,
                    n_block_max,
                    mInvFreq,
                    group_size_cur,
                )
            else:
                self.store_empty_split(
                    mODirect,
                    mLSEDirect,
                    batch_idx + split_idx * mSeqUsedNear.shape[0],
                    head_idx,
                    tidx,
                )
            if cutlass.const_expr(self.intra_kernel_reduce):
                cute.arch.sync_threads()
                cute.arch.fence_acq_rel_gpu()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if split_idx == 0:
                    self.combine_splitk_partials_in_kernel(
                        mODirect,
                        mLSEDirect,
                        mOFinal,
                        mLSEFinal,
                        sSplitScale,
                        batch_idx,
                        head_idx,
                        tidx,
                        mSeqUsedNear.shape[0],
                    )

        @cute.jit
        def store_empty_split(
            self,
            mO: cute.Tensor,
            mLSE: Optional[cute.Tensor],
            batch_idx: Int32,
            head_idx: Int32,
            tidx: Int32,
        ):
            q_heads_per_tile = self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1
            loop_count = cute.ceil_div(self.tile_hdimv, self.num_threads)
            for q_local in cutlass.range_constexpr(q_heads_per_tile):
                q_head = head_idx * q_heads_per_tile + q_local if cutlass.const_expr(self.pack_gqa) else head_idx
                if cutlass.const_expr(mLSE is not None):
                    if tidx == q_local:
                        mLSE[0, q_head, batch_idx] = -Float32.inf
                for i in cutlass.range(loop_count, unroll=1):
                    d = tidx + i * self.num_threads
                    if d < mO.shape[1]:
                        mO[0, d, q_head, batch_idx] = Float32(0.0).to(self.dtype)

        @cute.jit
        def combine_splitk_partials_in_kernel(
            self,
            mOPartial: cute.Tensor,
            mLSEPartial: cute.Tensor,
            mOFinal: cute.Tensor,
            mLSEFinal: Optional[cute.Tensor],
            sSplitScale: cute.Tensor,
            batch_idx: Int32,
            head_idx: Int32,
            tidx: Int32,
            num_batch: Int32,
        ):
            q_heads_per_tile = self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1
            LOG2_E = math.log2(math.e)
            for q_local in cutlass.range_constexpr(q_heads_per_tile):
                q_head = head_idx * q_heads_per_tile + q_local if cutlass.const_expr(self.pack_gqa) else head_idx
                if tidx == q_local:
                    lse_max = -Float32.inf
                    for s in cutlass.range_constexpr(self.cluster_splits):
                        lse_cur = Float32(mLSEPartial[0, q_head, batch_idx + s * num_batch])
                        lse_max = cutlass.max(lse_max, lse_cur)
                    denom = Float32(0.0)
                    lse_max_safe = Float32(0.0) if lse_max == -Float32.inf else lse_max
                    for s in cutlass.range_constexpr(self.cluster_splits):
                        lse_cur = Float32(mLSEPartial[0, q_head, batch_idx + s * num_batch])
                        weight = cute.math.exp2(lse_cur * LOG2_E - lse_max_safe * LOG2_E, fastmath=True)
                        denom += weight
                        sSplitScale[s, q_local] = weight
                    inv_denom = Float32(0.0) if (denom == 0.0 or denom != denom) else Float32(1.0) / denom
                    for s in cutlass.range_constexpr(self.cluster_splits):
                        sSplitScale[s, q_local] = sSplitScale[s, q_local] * inv_denom
                    if cutlass.const_expr(mLSEFinal is not None):
                        mLSEFinal[0, q_head, batch_idx] = cute.math.log(denom, fastmath=True) + lse_max

            cute.arch.sync_threads()
            loop_count = cute.ceil_div(self.tile_hdimv, self.num_threads)
            for q_local in cutlass.range_constexpr(q_heads_per_tile):
                q_head = head_idx * q_heads_per_tile + q_local if cutlass.const_expr(self.pack_gqa) else head_idx
                for i in cutlass.range(loop_count, unroll=1):
                    d = tidx + i * self.num_threads
                    if d < mOFinal.shape[1]:
                        acc = Float32(0.0)
                        for s in cutlass.range_constexpr(self.cluster_splits):
                            acc += (
                                Float32(mOPartial[0, d, q_head, batch_idx + s * num_batch])
                                * Float32(sSplitScale[s, q_local])
                            )
                        mOFinal[0, d, q_head, batch_idx] = acc.to(self.dtype)

        @cute.jit
        def decode_load(
            self,
            mQBase: cute.Tensor,
            mQGroup: cute.Tensor,
            mKNear: cute.Tensor,
            mVNear: cute.Tensor,
            mKDistant: cute.Tensor,
            mVDistant: cute.Tensor,
            mKNearRaw: cute.Tensor,
            mVNearRaw: cute.Tensor,
            mKDistantRaw: cute.Tensor,
            mVDistantRaw: cute.Tensor,
            sQBase: cute.Tensor,
            sQGroup: cute.Tensor,
            sK: cute.Tensor,
            sV: cute.Tensor,
            tma_atom_Q_base: cute.CopyAtom,
            tma_atom_Q_group: cute.CopyAtom,
            tma_atom_K_near: cute.CopyAtom,
            tma_atom_V_near: cute.CopyAtom,
            tma_atom_K_distant: cute.CopyAtom,
            tma_atom_V_distant: cute.CopyAtom,
            pipeline_q_base: pipeline.PipelineAsync,
            pipeline_q_group: pipeline.PipelineAsync,
            pipeline_k: pipeline.PipelineAsync,
            pipeline_k_raw: pipeline.PipelineAsync,
            pipeline_v: pipeline.PipelineAsync,
            batch_idx: Int32,
            head_idx: Int32,
            seqlen_near: Int32,
            seqlen_distant: Int32,
            mInvFreq: cute.Tensor,
            group_size: Int32,
            n_block_total_distant: Int32,
            n_block_min: Int32,
            n_block_max: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
            q_state = Int32(1)
            kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            head_idx_kv = head_idx if cutlass.const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
            gQBase = cute.local_tile(mQBase[None, None, head_idx, batch_idx], (self.tile_m, self.tile_hdim), (0, 0))
            gQGroup = cute.local_tile(mQGroup[None, None, head_idx, batch_idx], (self.tile_m, self.tile_hdim), (0, 0))
            mKNearCur = mKNear[None, None, head_idx_kv, batch_idx]
            mVNearCur = mVNear[None, None, head_idx_kv, batch_idx]
            # The near K/V are a separate, 0-based tensor (key_near), and both the block
            # iteration (n_block_total_near = ceil(seqlen_near/tile_n)) and the seqused
            # masking index the near region locally. The near tensor must therefore be
            # read from index 0 — NOT offset by seqlen_distant. The old domain_offset
            # assumed a combined [distant; near] buffer and silently dropped the near
            # region (read out of bounds) whenever seqlen_distant >= seqlen_near.
            gKNear = cute.local_tile(mKNearCur, (self.tile_n, self.tile_hdim), (None, 0))
            gVNear = cute.local_tile(mVNearCur, (self.tile_n, self.tile_hdimv), (None, 0))
            gKDistant = cute.local_tile(mKDistant[None, None, head_idx_kv, batch_idx], (self.tile_n, self.tile_hdim), (None, 0))
            gVDistant = cute.local_tile(mVDistant[None, None, head_idx_kv, batch_idx], (self.tile_n, self.tile_hdimv), (None, 0))
            load_Q_base, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q_base, 0, cute.make_layout(1), gQBase, sQBase, single_stage=True
            )
            load_Q_group, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q_group, 0, cute.make_layout(1), gQGroup, sQGroup, single_stage=True
            )
            load_K_near, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K_near, 0, cute.make_layout(1), gKNear, sK)
            load_V_near, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V_near, 0, cute.make_layout(1), gVNear, sV)
            load_K_distant, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K_distant, 0, cute.make_layout(1), gKDistant, sK
            )
            load_V_distant, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V_distant, 0, cute.make_layout(1), gVDistant, sV
            )
            load_K_near = copy_utils.tma_producer_copy_fn(load_K_near, pipeline_k)
            load_V_near = copy_utils.tma_producer_copy_fn(load_V_near, pipeline_v)
            load_K_distant = copy_utils.tma_producer_copy_fn(
                load_K_distant, pipeline_k_raw if cutlass.const_expr(self.jetlong_k_mode == 1) else pipeline_k
            )
            load_V_distant = copy_utils.tma_producer_copy_fn(load_V_distant, pipeline_v)

            if warp_idx_in_wg == 0:
                pipeline_q_base.producer_acquire_w_index_phase(0, q_state)
                load_Q_base(tma_bar_ptr=pipeline_q_base.sync_object_full.get_barrier(0))
                pipeline_q_group.producer_acquire_w_index_phase(0, q_state)
                load_Q_group(tma_bar_ptr=pipeline_q_group.sync_object_full.get_barrier(0))
                q_state ^= 1

            for i in cutlass.range(n_block_max - n_block_min, unroll=1):
                global_block = n_block_max - 1 - i
                is_distant = global_block < n_block_total_distant
                if cutlass.const_expr(self.use_dynamic_metadata and self.jetlong_k_mode == 2):
                    pipeline_k.producer_acquire(kv_state)
                    if is_distant:
                        self.load_jetlong_k_tile_from_tensor(
                            mKDistantRaw,
                            sK,
                            kv_state.index,
                            global_block,
                            Int32(0),
                            seqlen_distant,
                            tidx,
                            mInvFreq,
                            group_size,
                            head_idx_kv,
                            batch_idx,
                            apply_grouped_rotate=self.apply_distant_k_correction,
                        )
                    else:
                        self.load_jetlong_k_tile_from_tensor(
                            mKNearRaw,
                            sK,
                            kv_state.index,
                            global_block - n_block_total_distant,
                            seqlen_distant,
                            seqlen_near,
                            tidx,
                            mInvFreq,
                            group_size,
                            head_idx_kv,
                            batch_idx,
                            apply_grouped_rotate=False,
                        )
                    pipeline_k.producer_commit(kv_state)
                    pipeline_v.producer_acquire(kv_state)
                    if is_distant:
                        self.load_jetlong_v_tile_from_tensor(
                            mVDistantRaw,
                            sV,
                            kv_state.index,
                            global_block,
                            Int32(0),
                            seqlen_distant,
                            tidx,
                            head_idx_kv,
                            batch_idx,
                        )
                    else:
                        self.load_jetlong_v_tile_from_tensor(
                            mVNearRaw,
                            sV,
                            kv_state.index,
                            global_block - n_block_total_distant,
                            seqlen_distant,
                            seqlen_near,
                            tidx,
                            head_idx_kv,
                            batch_idx,
                        )
                    pipeline_v.producer_commit(kv_state)
                elif cutlass.const_expr(self.jetlong_k_mode == 2):
                    pipeline_k.producer_acquire(kv_state)
                    if is_distant:
                        self.load_jetlong_k_tile_from_gmem(
                            gKDistant,
                            sK,
                            kv_state.index,
                            global_block,
                            seqlen_distant,
                            tidx,
                            mInvFreq,
                            group_size,
                            apply_grouped_rotate=self.apply_distant_k_correction,
                        )
                    else:
                        self.load_jetlong_k_tile_from_gmem(
                            gKNear,
                            sK,
                            kv_state.index,
                            global_block - n_block_total_distant,
                            seqlen_near,
                            tidx,
                            mInvFreq,
                            group_size,
                            apply_grouped_rotate=False,
                        )
                    pipeline_k.producer_commit(kv_state)
                elif cutlass.const_expr(self.jetlong_k_mode == 1):
                    if warp_idx_in_wg == 0:
                        pipeline_k_raw.producer_acquire(kv_state)
                        if is_distant:
                            load_K_distant(src_idx=global_block, producer_state=kv_state)
                        else:
                            load_K_near(src_idx=global_block - n_block_total_distant, producer_state=kv_state)
                    pipeline_k_raw.consumer_wait(kv_state, pipeline_k_raw.consumer_try_wait(kv_state))
                    if is_distant:
                        if cutlass.const_expr(self.apply_distant_k_correction):
                            self.apply_jetlong_k_transform_smem(
                                sK, kv_state.index, global_block, seqlen_distant, tidx, mInvFreq, group_size
                            )
                        else:
                            cute.arch.fence_view_async_shared()
                    else:
                        cute.arch.fence_view_async_shared()
                    pipeline_k.producer_acquire(kv_state)
                    pipeline_k.producer_commit(kv_state)
                else:
                    if warp_idx_in_wg == 0:
                        pipeline_k.producer_acquire(kv_state)
                        if is_distant:
                            load_K_distant(src_idx=global_block, producer_state=kv_state)
                        else:
                            load_K_near(src_idx=global_block - n_block_total_distant, producer_state=kv_state)

                if cutlass.const_expr(not (self.use_dynamic_metadata and self.jetlong_k_mode == 2)) and warp_idx_in_wg == 0:
                    pipeline_v.producer_acquire(kv_state)
                    if is_distant:
                        load_V_distant(src_idx=global_block, producer_state=kv_state)
                    else:
                        load_V_near(src_idx=global_block - n_block_total_distant, producer_state=kv_state)
                kv_state.advance()
            if warp_idx_in_wg == 0:
                pipeline_v.producer_tail(kv_state)

        @cute.jit
        def decode_mma(
            self,
            mO: cute.Tensor,
            mLSE: Optional[cute.Tensor],
            sQBase: cute.Tensor,
            sQGroup: cute.Tensor,
            sK: cute.Tensor,
            sVt: cute.Tensor,
            sP: Optional[cute.Tensor],
            sO: cute.Tensor,
            pipeline_q_base: pipeline.PipelineAsync,
            pipeline_q_group: pipeline.PipelineAsync,
            pipeline_k: pipeline.PipelineAsync,
            pipeline_k_raw: pipeline.PipelineAsync,
            pipeline_v: pipeline.PipelineAsync,
            gmem_tiled_copy_O: cute.TiledCopy,
            tma_atom_O: cute.CopyAtom,
            tiled_mma_qk: cute.TiledMma,
            tiled_mma_pv: cute.TiledMma,
            tidx: Int32,
            softmax_scale_log2: Float32,
            softmax_scale: Optional[Float32],
            batch_idx: Int32,
            head_idx: Int32,
            split_idx: Int32,
            num_splits: Int32,
            num_batch: Int32,
            seqlen_near: Int32,
            seqlen_distant: Int32,
            n_block_total_distant: Int32,
            n_block_min: Int32,
            n_block_max: Int32,
            mInvFreq: cute.Tensor,
            group_size: Int32,
        ):
            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            wg_mma_qk = tiled_mma_qk.get_slice(cute.make_layout(1)(0))
            wg_mma_pv = tiled_mma_pv.get_slice(cute.make_layout(1)(0))
            _, tSrQBase, tSrK = sm90_utils.partition_fragment_ABC(
                wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQBase, sK
            )
            _, tSrQGroup, _ = sm90_utils.partition_fragment_ABC(
                wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQGroup, sK
            )
            mma_qk_base_fn = partial(
                sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQBase, tSrK
            )
            mma_qk_group_fn = partial(
                sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQGroup, tSrK
            )
            acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
                wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
            )
            mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)
            if cutlass.const_expr(self.use_dual_branch_accumulator):
                acc_ODist = cute.make_fragment_like(acc_O, Float32)
                mma_pv_dist_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_ODist, tOrP, tOrVt)

            smem_copy_atom_P = utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, self.dtype)
            smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
            tPsP = smem_thr_copy_P.partition_D(sP) if cutlass.const_expr(sP is not None) else None

            self.mma_init()
            q_state = Int32(0)
            kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            pipeline_q_base.consumer_wait_w_index_phase(0, q_state)
            pipeline_q_group.consumer_wait_w_index_phase(0, q_state)

            softmax = Softmax.create(
                softmax_scale_log2,
                num_rows=acc_O.shape[0][0] * acc_O.shape[1],
                softmax_scale=softmax_scale,
            )
            if cutlass.const_expr(self.use_dual_branch_accumulator):
                softmax_dist = Softmax.create(
                    softmax_scale_log2,
                    num_rows=acc_O.shape[0][0] * acc_O.shape[1],
                    softmax_scale=softmax_scale,
                )
            else:
                saved_branch_O = cute.make_fragment_like(acc_O, self.dtype)
                saved_branch_lse = cute.make_fragment_like(softmax.row_sum, Float32)
            if cutlass.const_expr(not self.intra_wg_overlap):
                acc_O.fill(0.0)
                softmax.reset()
                if cutlass.const_expr(self.use_dual_branch_accumulator):
                    acc_ODist.fill(0.0)
                    softmax_dist.reset()

            mixed_mask_fn = partial(
                self.apply_mixed_tail_mask,
                thr_mma=thr_mma_qk,
                n_block_total_distant=n_block_total_distant,
                seqlen_distant=seqlen_distant,
                seqlen_near=seqlen_near,
            )

            smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)
            scores_scale = None
            mma_one_n_block_all = partial(
                self.mma_one_n_block_intrawg_overlap
                if cutlass.const_expr(self.intra_wg_overlap)
                else self.mma_one_n_block,
                pipeline_k=pipeline_k,
                pipeline_k_raw=pipeline_k_raw,
                pipeline_v=pipeline_v,
                acc_O=acc_O,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                check_inf=True,
                scores_scale=scores_scale,
                softmax=softmax,
                seqlen=SeqlenInfoQK.create(batch_idx=batch_idx, seqlen_q_static=1, seqlen_k_static=1),
                score_mod_fn=None,
                sK=sK,
                tidx=tidx,
                n_block_total_distant=n_block_total_distant,
                seqlen_distant=seqlen_distant,
                mInvFreq=mInvFreq,
                group_size=group_size,
            )
            if cutlass.const_expr(self.use_dual_branch_accumulator):
                mma_one_n_block_dist = partial(
                    self.mma_one_n_block,
                    pipeline_k=pipeline_k,
                    pipeline_k_raw=pipeline_k_raw,
                    pipeline_v=pipeline_v,
                    acc_O=acc_ODist,
                    tOrP=tOrP,
                    smem_copy_params=smem_copy_params,
                    check_inf=True,
                    scores_scale=scores_scale,
                    softmax=softmax_dist,
                    seqlen=SeqlenInfoQK.create(batch_idx=batch_idx, seqlen_q_static=1, seqlen_k_static=1),
                    score_mod_fn=None,
                    sK=sK,
                    tidx=tidx,
                    n_block_total_distant=n_block_total_distant,
                    seqlen_distant=seqlen_distant,
                    mInvFreq=mInvFreq,
                    group_size=group_size,
                )
            process_first_half_block = partial(
                self.first_half_block_overlap,
                pipeline_k=pipeline_k,
                pipeline_k_raw=pipeline_k_raw,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                scores_scale=scores_scale,
                softmax=softmax,
                acc_O=acc_O,
                seqlen=SeqlenInfoQK.create(batch_idx=batch_idx, seqlen_q_static=1, seqlen_k_static=1),
                score_mod_fn=None,
                sK=sK,
                tidx=tidx,
                n_block_total_distant=n_block_total_distant,
                seqlen_distant=seqlen_distant,
                mInvFreq=mInvFreq,
                group_size=group_size,
            )
            process_last_half_block = partial(
                self.last_half_block_overlap,
                pipeline_v=pipeline_v,
                mma_pv_fn=mma_pv_fn,
                scores_scale=scores_scale,
                softmax=softmax,
                acc_O=acc_O,
            )

            o_should_accumulate = False
            near_should_accumulate = False
            saved_near = False
            first_block = n_block_max - 1
            if cutlass.const_expr(self.intra_wg_overlap):
                if first_block < n_block_total_distant:
                    kv_state = process_first_half_block(
                        n_block=first_block,
                        mma_qk_fn=mma_qk_group_fn,
                        kv_consumer_state=kv_state,
                        mask_fn=mixed_mask_fn,
                        is_first_block=True,
                    )
                else:
                    kv_state = process_first_half_block(
                        n_block=first_block,
                        mma_qk_fn=mma_qk_base_fn,
                        kv_consumer_state=kv_state,
                        mask_fn=mixed_mask_fn,
                        is_first_block=True,
                    )
            else:
                self.warp_scheduler_barrier_sync()
                if cutlass.const_expr(self.use_dual_branch_accumulator):
                    for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                        global_block = n_block_max - 1 - n_tile
                        if global_block < n_block_total_distant:
                            kv_state = mma_one_n_block_dist(
                                kv_state,
                                n_block=global_block,
                                mma_qk_fn=mma_qk_group_fn,
                                mma_pv_fn=partial(mma_pv_dist_fn, zero_init=False),
                                is_first_n_block=False,
                                mask_fn=mixed_mask_fn,
                            )
                        else:
                            kv_state = mma_one_n_block_all(
                                kv_state,
                                n_block=global_block,
                                mma_qk_fn=mma_qk_base_fn,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                                is_first_n_block=False,
                                mask_fn=mixed_mask_fn,
                            )
                            near_should_accumulate = True
                    o_should_accumulate = near_should_accumulate
                else:
                    near_block_min = cutlass.max(n_block_min, n_block_total_distant)
                    near_block_max = n_block_max
                    dist_block_min = n_block_min
                    dist_block_max = cutlass.min(n_block_max, n_block_total_distant)
                    has_near_work = near_block_max > near_block_min
                    has_dist_work = dist_block_max > dist_block_min
                    saved_near = has_near_work and has_dist_work
                    if has_near_work:
                        branch_first = near_block_max - 1
                        kv_state = mma_one_n_block_all(
                            kv_state,
                            n_block=branch_first,
                            mma_qk_fn=mma_qk_base_fn,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                            is_first_n_block=True,
                            mask_fn=mixed_mask_fn,
                        )
                        o_should_accumulate = True
                        for n_tile in cutlass.range(branch_first - near_block_min, unroll=1):
                            global_block = branch_first - 1 - n_tile
                            kv_state = mma_one_n_block_all(
                                kv_state,
                                n_block=global_block,
                                mma_qk_fn=mma_qk_base_fn,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                                mask_fn=mixed_mask_fn,
                            )
                        if saved_near:
                            row_scale_near = self.finalize_softmax_div(softmax)
                            softmax.rescale_O(acc_O, row_scale_near)
                            saved_branch_O.store(acc_O.load().to(self.dtype))
                            saved_branch_lse.store(softmax.row_sum.load())
                            acc_O.fill(0.0)
                            softmax.reset()
                            o_should_accumulate = False
                    if has_dist_work:
                        branch_first = dist_block_max - 1
                        kv_state = mma_one_n_block_all(
                            kv_state,
                            n_block=branch_first,
                            mma_qk_fn=mma_qk_group_fn,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                            is_first_n_block=True,
                            mask_fn=mixed_mask_fn,
                        )
                        o_should_accumulate = True
                        for n_tile in cutlass.range(branch_first - dist_block_min, unroll=1):
                            global_block = branch_first - 1 - n_tile
                            kv_state = mma_one_n_block_all(
                                kv_state,
                                n_block=global_block,
                                mma_qk_fn=mma_qk_group_fn,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                                mask_fn=mixed_mask_fn,
                            )
            n_block_last = n_block_max - 1
            if cutlass.const_expr(self.intra_wg_overlap):
                for n_tile in cutlass.range(n_block_last - n_block_min, unroll=1):
                    global_block = n_block_last - 1 - n_tile
                    if global_block < n_block_total_distant:
                        kv_state = mma_one_n_block_all(
                            kv_state,
                            n_block=global_block,
                            mma_qk_fn=mma_qk_group_fn,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not o_should_accumulate),
                            mask_fn=mixed_mask_fn,
                        )
                    else:
                        kv_state = mma_one_n_block_all(
                            kv_state,
                            n_block=global_block,
                            mma_qk_fn=mma_qk_base_fn,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not o_should_accumulate),
                            mask_fn=mixed_mask_fn,
                        )
                    o_should_accumulate = True
            pipeline_q_base.consumer_release_w_index(0)
            pipeline_q_group.consumer_release_w_index(0)
            if cutlass.const_expr(self.intra_wg_overlap):
                kv_state = process_last_half_block(
                    kv_consumer_state=kv_state,
                    zero_init=not o_should_accumulate,
                )
            else:
                self.warp_scheduler_barrier_arrive()
            q_state ^= 1

            if cutlass.const_expr(self.use_dual_branch_accumulator):
                row_scale = self.finalize_softmax_div(softmax)
                softmax.rescale_O(acc_O, row_scale)
            else:
                row_scale = self.finalize_softmax_div(softmax)
                softmax.rescale_O(acc_O, row_scale)
                if cutlass.const_expr(not self.intra_wg_overlap):
                    if saved_near:
                        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
                        saved_branch_O_mn = layout_utils.reshape_acc_to_mn(saved_branch_O)
                        for r in cutlass.range(cute.size(softmax.row_sum), unroll_full=True):
                            lse_near = saved_branch_lse[r]
                            lse_dist = softmax.row_sum[r]
                            max_lse = cutlass.max(lse_near, lse_dist)
                            max_lse_safe = Float32(0.0) if max_lse == -Float32.inf else max_lse
                            w_near = cute.math.exp(lse_near - max_lse_safe, fastmath=True)
                            w_dist = cute.math.exp(lse_dist - max_lse_safe, fastmath=True)
                            denom = Float32(0.0)
                            denom += w_near
                            denom += w_dist
                            inv_denom = Float32(0.0) if (denom == 0.0 or denom != denom) else Float32(1.0) / denom
                            wn = w_near * inv_denom
                            wd = w_dist * inv_denom
                            near_o = saved_branch_O_mn[r, None].load().to(Float32)
                            dist_o = acc_O_mn[r, None].load().to(self.dtype).to(Float32)
                            acc_O_mn[r, None].store(near_o * wn + dist_o * wd)
                            softmax.row_sum[r] = (
                                max_lse_safe + cute.math.log(denom, fastmath=True)
                                if denom > 0.0
                                else -Float32.inf
                            )
            if cutlass.const_expr(not self.intra_wg_overlap and self.use_dual_branch_accumulator):
                row_scale_dist = self.finalize_softmax_div(softmax_dist)
                softmax_dist.rescale_O(acc_ODist, row_scale_dist)
                acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
                acc_ODist_mn = layout_utils.reshape_acc_to_mn(acc_ODist)
                for r in cutlass.range(cute.size(softmax.row_sum), unroll_full=True):
                    lse_near = softmax.row_sum[r]
                    lse_dist = softmax_dist.row_sum[r]
                    max_lse = cutlass.max(lse_near, lse_dist)
                    max_lse_safe = Float32(0.0) if max_lse == -Float32.inf else max_lse
                    w_near = cute.math.exp(lse_near - max_lse_safe, fastmath=True)
                    w_dist = cute.math.exp(lse_dist - max_lse_safe, fastmath=True)
                    denom = Float32(0.0)
                    denom += w_near
                    denom += w_dist
                    inv_denom = Float32(0.0) if (denom == 0.0 or denom != denom) else Float32(1.0) / denom
                    wn = w_near * inv_denom
                    wd = w_dist * inv_denom
                    near_o = acc_O_mn[r, None].load().to(self.dtype).to(Float32)
                    dist_o = acc_ODist_mn[r, None].load().to(self.dtype).to(Float32)
                    acc_O_mn[r, None].store(near_o * wn + dist_o * wd)
                    softmax.row_sum[r] = (
                        max_lse_safe + cute.math.log(denom, fastmath=True)
                        if denom > 0.0
                        else -Float32.inf
                    )
            batch_out_idx = batch_idx + split_idx * num_batch
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                SeqlenInfoQK.create(
                    batch_idx=batch_out_idx,
                    seqlen_q_static=1,
                    seqlen_k_static=1,
                ),
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                0,
                head_idx,
                batch_out_idx,
            )

    return FlashAttentionDecodeJetLongSm90, ns


def _launch_jetlong_mixed_kernel(
    q_base: torch.Tensor,
    q_group: torch.Tensor,
    k_near: torch.Tensor,
    v_near: torch.Tensor,
    k_distant: torch.Tensor,
    v_distant: torch.Tensor,
    *,
    seqused_near: torch.Tensor,
    seqused_distant: torch.Tensor,
    jetlong_meta: torch.Tensor | None = None,
    inv_freq: torch.Tensor,
    group_size: int,
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    num_splits: int,
    fa4_root: str | None,
    tile_m: int,
    tile_n: int,
    num_stages: int,
    pack_gqa: bool | None,
    k_mode: str,
    intra_kernel_reduce: bool = False,
    apply_distant_k_correction: bool = True,
    active_splits: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    op = "fa4_cute_decode_sm90_jetlong_fused"
    qkv_tensors = (
        ("query_base", q_base),
        ("query_group", q_group),
        ("key_near", k_near),
        ("value_near", v_near),
        ("key_distant", k_distant),
        ("value_distant", v_distant),
    )
    if not all(tensor.dim() == 4 for _, tensor in qkv_tensors):
        raise ValueError(f"{op} expects rank-4 BSHD tensors")
    if q_base.shape != q_group.shape:
        raise ValueError(f"{op}: query_base and query_group must share shape")
    if q_base.shape[1] != 1:
        raise ValueError(f"{op} requires seqlen_q == 1")
    if q_base.dtype not in _SUPPORTED_CUTE_DTYPES:
        raise ValueError(f"{op}: query_base must have dtype torch.float16 or torch.bfloat16, got {q_base.dtype}")
    if not all(tensor.is_cuda for _, tensor in qkv_tensors):
        raise ValueError(f"{op} requires CUDA Q/K/V tensors")
    if not all(tensor.dtype == q_base.dtype for _, tensor in qkv_tensors):
        raise ValueError(f"{op}: Q/K/V tensors must share dtype")
    if not all(tensor.device == q_base.device for _, tensor in qkv_tensors):
        raise ValueError(f"{op}: Q/K/V tensors must be on the same CUDA device")
    if k_near.shape[:3] != v_near.shape[:3] or k_distant.shape[:3] != v_distant.shape[:3]:
        raise ValueError(f"{op}: each K/V branch must share batch, sequence, and head dimensions")
    if k_near.shape[0] != q_base.shape[0] or k_distant.shape[0] != q_base.shape[0]:
        raise ValueError(f"{op}: Q/K/V batch sizes must match")
    if k_near.shape[2] != k_distant.shape[2]:
        raise ValueError(f"{op}: near and distant KV head counts must match")
    if v_near.shape[-1] != v_distant.shape[-1]:
        raise ValueError(f"{op}: near and distant value head_dim must match")
    if q_base.shape[-1] != k_near.shape[-1] or q_base.shape[-1] != k_distant.shape[-1]:
        raise ValueError(f"{op}: query and key head_dim must match")
    if q_base.shape[2] % k_near.shape[2] != 0:
        raise ValueError(f"{op}: query heads must be divisible by kv heads")
    if num_splits < 1:
        raise ValueError(f"{op}: num_splits must be >= 1")
    if inv_freq.device != q_base.device:
        raise ValueError(f"{op}: inv_freq must be on device {q_base.device}, got {inv_freq.device}")
    if inv_freq.dim() != 1 or inv_freq.numel() < q_base.shape[-1] // 2:
        raise ValueError(f"{op}: inv_freq must have shape ({q_base.shape[-1] // 2},) or larger")
    if not torch.is_floating_point(inv_freq):
        raise ValueError(f"{op}: inv_freq must be a floating-point tensor, got {inv_freq.dtype}")
    for name, tensor in (
        ("q_base", q_base),
        ("q_group", q_group),
        ("k_near", k_near),
        ("v_near", v_near),
        ("k_distant", k_distant),
        ("v_distant", v_distant),
    ):
        _require_contiguous(op, name, tensor)
    _require_decode_metadata(op, "seqused_near", seqused_near, shape=(q_base.shape[0],), device=q_base.device)
    _require_decode_metadata(op, "seqused_distant", seqused_distant, shape=(q_base.shape[0],), device=q_base.device)
    _require_output_tensor(
        op,
        "out",
        out,
        shape=(q_base.shape[0], q_base.shape[1], q_base.shape[2], v_near.shape[-1]),
        dtype=q_base.dtype,
        device=q_base.device,
    )
    _require_output_tensor(
        op,
        "lse",
        lse,
        shape=(q_base.shape[0], q_base.shape[2], q_base.shape[1]),
        dtype=torch.float32,
        device=q_base.device,
    )
    seqused_near = seqused_near.contiguous()
    seqused_distant = seqused_distant.contiguous()
    if jetlong_meta is not None:
        if k_mode not in ("manual", "consumer", "consumer_near_offset"):
            raise ValueError("JetLong dynamic metadata requires k_mode='manual', 'consumer', or 'consumer_near_offset'")
        if jetlong_meta.dtype != torch.int32 or jetlong_meta.dim() != 2 or jetlong_meta.shape[1] < 3:
            raise ValueError("jetlong_meta must be int32 with shape (B, >=3): [total_kv_len, boundary, group_size]")
        if jetlong_meta.shape[0] != q_base.shape[0]:
            raise ValueError(f"jetlong_meta batch dimension must be {q_base.shape[0]}, got {jetlong_meta.shape[0]}")
        if jetlong_meta.device != q_base.device:
            raise ValueError(f"jetlong_meta must be on device {q_base.device}, got {jetlong_meta.device}")
        jetlong_meta = jetlong_meta.contiguous()
    if active_splits is not None:
        _require_decode_metadata(
            op,
            "active_splits",
            active_splits,
            shape=(q_base.shape[0],),
            device=q_base.device,
        )
        active_splits = active_splits.contiguous()
    inv_freq = inv_freq.float().contiguous()
    group_size = int(group_size)
    batch = q_base.shape[0]
    return_lse = lse is not None
    if intra_kernel_reduce and num_splits > 1:
        n_block_total = math.ceil((k_near.shape[1] + k_distant.shape[1]) / tile_n)
        if num_splits > n_block_total:
            raise ValueError(f"num_splits={num_splits} exceeds JetLong KV tiles={n_block_total}")
        if num_splits > 8:
            raise ValueError("SM90 cluster intra-kernel split-K reduce supports num_splits <= 8")
    if num_splits > 1:
        out_partial, lse_partial = _get_splitk_workspace(
            device=q_base.device,
            dtype=q_base.dtype,
            batch=batch,
            seqlen_q=q_base.shape[1],
            nheads=q_base.shape[2],
            head_dim_v=v_near.shape[-1],
            num_splits=num_splits,
        )
        if not intra_kernel_reduce:
            out_partial.zero_()
            lse_partial.fill_(-float("inf"))
        out = (
            torch.empty(
                batch,
                q_base.shape[1],
                q_base.shape[2],
                v_near.shape[-1],
                device=q_base.device,
                dtype=q_base.dtype,
            )
            if out is None
            else out
        )
        lse = (
            torch.empty(batch, q_base.shape[2], q_base.shape[1], device=q_base.device, dtype=torch.float32)
            if return_lse
            else None
        ) if lse is None else lse
    else:
        if out is None:
            out = torch.empty(
                q_base.shape[0], q_base.shape[1], q_base.shape[2], v_near.shape[-1], device=q_base.device, dtype=q_base.dtype
            )
        out_partial = out
        lse_partial = lse

    out_final = out if intra_kernel_reduce and num_splits > 1 else None
    lse_final = lse if intra_kernel_reduce and num_splits > 1 else None
    jetlong_meta_has_active_splits = jetlong_meta is not None and jetlong_meta.shape[1] > 4
    use_dual_branch_accumulator = os.environ.get("JETLM_JETLONG_CUTE_DUAL_ACC", "0") == "1"
    mma_regs = int(os.environ.get("JETLM_JETLONG_CUTE_MMA_REGS", "192"))

    FlashAttentionDecodeJetLongSm90, ns = _define_jetlong_kernel_class(
        fa4_root,
        k_mode=k_mode,
        intra_kernel_reduce=intra_kernel_reduce and num_splits > 1,
        cluster_splits=num_splits,
        apply_distant_k_correction=apply_distant_k_correction,
        use_dynamic_metadata=jetlong_meta is not None,
        jetlong_meta_has_active_splits=jetlong_meta_has_active_splits,
        use_dual_branch_accumulator=use_dual_branch_accumulator,
        mma_regs=mma_regs,
    )
    cute = ns["cute"]
    cutlass = ns["cutlass"]
    to_cute_tensor = ns["to_cute_tensor"]
    cache = _get_cache(fa4_root)

    num_head = q_base.shape[-2]
    num_head_kv = k_near.shape[-2]
    head_dim = q_base.shape[-1]
    head_dim_v = v_near.shape[-1]
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    compile_key = (
        "jetlong_mixed",
        q_base.dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        q_base.shape[2],
        k_near.shape[1],
        k_distant.shape[1],
        tile_m,
        tile_n,
        num_stages,
        pack_gqa,
        k_mode,
        apply_distant_k_correction,
        jetlong_meta is not None,
        jetlong_meta_has_active_splits,
        use_dual_branch_accumulator,
        mma_regs,
        active_splits.shape if active_splits is not None else None,
        lse_partial is not None,
        intra_kernel_reduce and num_splits > 1,
        num_splits,
        _get_device_arch(),
    )
    if compile_key not in cache:
        current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        dtype_map = {
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
        }
        fa = FlashAttentionDecodeJetLongSm90(
            dtype_map[q_base.dtype],
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=True,
            is_local=False,
            pack_gqa=pack_gqa,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=num_stages,
        )
        cache[compile_key] = cute.compile(
            fa,
            to_cute_tensor(q_base),
            to_cute_tensor(q_group),
            to_cute_tensor(k_near),
            to_cute_tensor(v_near),
            to_cute_tensor(k_distant),
            to_cute_tensor(v_distant),
            to_cute_tensor(out_partial),
            to_cute_tensor(lse_partial) if lse_partial is not None else None,
            to_cute_tensor(out_final) if out_final is not None else None,
            to_cute_tensor(lse_final) if lse_final is not None else None,
            1.0 / math.sqrt(head_dim),
            to_cute_tensor(seqused_near, assumed_align=4, leading_dim=0),
            to_cute_tensor(seqused_distant, assumed_align=4, leading_dim=0),
            to_cute_tensor(inv_freq, assumed_align=4, leading_dim=0),
            to_cute_tensor(jetlong_meta, assumed_align=4) if jetlong_meta is not None else None,
            to_cute_tensor(active_splits, assumed_align=4) if active_splits is not None else None,
            group_size,
            num_splits,
            current_stream,
            options="--enable-tvm-ffi",
        )
    cache[compile_key](
        q_base.detach(),
        q_group.detach(),
        k_near.detach(),
        v_near.detach(),
        k_distant.detach(),
        v_distant.detach(),
        out_partial.detach(),
        lse_partial.detach() if lse_partial is not None else None,
        out_final.detach() if out_final is not None else None,
        lse_final.detach() if lse_final is not None else None,
        1.0 / math.sqrt(head_dim),
        seqused_near,
        seqused_distant,
        inv_freq,
        jetlong_meta,
        active_splits,
        group_size,
        num_splits,
    )
    if intra_kernel_reduce and num_splits > 1:
        return (out, lse) if lse is not None else out
    if num_splits > 1:
        return _merge_splitk_partials(
            out_partial,
            lse_partial,
            out=out,
            lse=lse,
            batch=batch,
            num_splits=num_splits,
        )
    return (out, lse) if lse is not None else out


def fa4_cute_decode_sm90_jetlong_fused(
    query_base: torch.Tensor,
    query_group: torch.Tensor,
    key_near: torch.Tensor,
    value_near: torch.Tensor,
    key_distant: torch.Tensor,
    value_distant: torch.Tensor,
    *,
    seqused_near: torch.Tensor,
    seqused_distant: torch.Tensor,
    jetlong_meta: torch.Tensor | None = None,
    inv_freq: torch.Tensor,
    group_size: int,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    fa4_root: str | None = None,
    tile_m: int = 64,
    tile_n: int = 128,
    num_stages: int = 3,
    pack_gqa: bool | None = None,
    k_mode: str = "consumer",
    num_splits: int = 1,
    active_splits: torch.Tensor | None = None,
    distant_k_grouped: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if _get_device_arch() != 90:
        raise RuntimeError("fa4_cute_decode_sm90_jetlong_fused requires SM90/H100")
    return _launch_jetlong_mixed_kernel(
        query_base,
        query_group,
        key_near,
        value_near,
        key_distant,
        value_distant,
        seqused_near=seqused_near,
        seqused_distant=seqused_distant,
        jetlong_meta=jetlong_meta,
        inv_freq=inv_freq,
        group_size=group_size,
        out=out,
        lse=lse,
        num_splits=num_splits,
        fa4_root=fa4_root,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=num_stages,
        pack_gqa=pack_gqa,
        k_mode=k_mode,
        intra_kernel_reduce=1 < num_splits <= 8,
        apply_distant_k_correction=not distant_k_grouped,
        active_splits=active_splits,
    )
