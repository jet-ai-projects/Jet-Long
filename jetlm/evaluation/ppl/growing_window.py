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

"""Anchored growing-window NLL helper.

For one tokenized book of length >= stride:
    for current_len in [S, 2S, 3S, ..., min(book_len, l_max)]:
        logits = forward(tokens[0:current_len], logits_to_keep=S+1)
        score NLL on the LAST S targets (or stride-1 at the very first step,
        since token 0 has no prior context to be predicted from).

Each step is anchored at index 0 — the model sees the full prefix from the start of
the book. At current_len = k*S the scored tokens are [k*S - S, k*S), with (k-1)*S
tokens of preceding true context.

Steps run in ASCENDING current_len so a late-stage OOM still leaves a complete
short-context curve.

`logits_to_keep=S+1` caps the logits transient at ~600 MB even at L=131072 (versus
~39 GB for the full [1, 131072, 151936] bf16 logits tensor). Both the stock HF
Qwen3ForCausalLM and the jetlong/selfext/dca custom variants in this repo accept this
argument and slice the lm_head accordingly.

The batched variant `evaluate_books_batched_growing_window` runs multiple books
through one forward pass at each `current_len`. With bsz=4 books per forward at
current_len=131072 on Qwen3-8B, peak memory is ~62 GB (weights 16 + 4 * KV 9.5 +
logits transient + cross_entropy float32 ≈ 62 GB), which fits comfortably on B200
(192 GB). Books that drop below `current_len` in length are excluded from
subsequent steps; books that OOM are retried at bsz=1 before being marked OOM.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StepRecord:
    book_idx: int
    current_len: int
    sum_nll: float
    n_scored: int


def _slice_for_loss(logits: torch.Tensor, ids: torch.Tensor, stride: int):
    """Return (shift_logits, shift_labels) with proper alignment.

    Args:
        logits: [B, kept, V] — the last `kept` positions' logits.
        ids:    [B, current_len] — the input tokens.
        stride: scoring window size.

    On the first step (current_len == stride), kept == stride and we score tokens
    1..stride-1 (token 0 has no prior context). On steady-state steps kept == stride+1
    and we score the trailing `stride` tokens.
    """
    kept = logits.size(1)
    current_len = ids.size(1)
    if kept == current_len:
        # First step: predict tokens 1..stride-1 from logits 0..stride-2.
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
    else:
        # Steady state: kept == stride+1; predict the last `stride` tokens.
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, -stride:]
    return shift_logits, shift_labels


def _score_batch(model, batch_ids: torch.Tensor, stride: int) -> torch.Tensor:
    """Run one forward pass on a batched [B, current_len] tensor and return per-row sum_nll.

    Returns: [B] float tensor of summed NLL per row.
    Raises torch.cuda.OutOfMemoryError on OOM.
    """
    with torch.no_grad():
        out = model(batch_ids, use_cache=False, logits_to_keep=stride + 1)
    logits = out.logits  # [B, kept, V]
    shift_logits, shift_labels = _slice_for_loss(logits, batch_ids, stride)
    B, T, V = shift_logits.shape
    nll_flat = F.cross_entropy(
        shift_logits.float().reshape(-1, V),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(B, T)
    return nll_flat.sum(dim=1)


def evaluate_books_batched_growing_window(
    model,
    rank_books: list[tuple[int, torch.Tensor]],  # [(book_idx, ids[L_i])]
    stride: int,
    l_max: int,
    batch_size: int = 1,
    show_progress: bool = False,
    progress_desc: str = "pg19_ppl",
) -> tuple[list[StepRecord], list[tuple[int, int]]]:
    """Anchored growing-window across multiple books with data-parallel batching.

    At each current_len = k*S, batches all books still long enough into chunks of
    size `batch_size` and runs one forward per chunk. Books shorter than current_len
    drop out for that and all later steps. Books that OOM at the batch are retried
    one-by-one at bsz=1; if even bsz=1 OOMs they're marked OOM at this current_len
    and skipped from subsequent steps.

    Returns: (records, oom_list)
        records: list of StepRecord for every (book, successful current_len)
        oom_list: list of (book_idx, current_len_that_oomed)
    """
    records: list[StepRecord] = []
    oom_list: list[tuple[int, int]] = []
    n_steps = l_max // stride

    # alive_books shrinks as books either run out of tokens or OOM.
    alive_books = [(idx, ids) for idx, ids in rank_books if ids.numel() >= stride]

    # Progress bar — only on the rank that asked for it (typically master).
    # Weight each step by k² so the ETA reflects the quadratic attention cost rather
    # than uniform-step-time (step 1 vs step 128 differ by ~16,000× in compute).
    pbar = None
    if show_progress:
        from tqdm import tqdm
        # Weight progress by k² (≈ attention compute) so the ETA reflects real wall-time.
        total_units = sum(k * k for k in range(1, n_steps + 1))
        pbar = tqdm(
            total=total_units,
            desc=progress_desc,
            mininterval=10.0,           # at most one redraw per 10 s — slurm-log friendly
            miniters=1,
            bar_format="{desc}: {percentage:3.0f}% [{elapsed}<{remaining}] {postfix}",
        )

    for k in range(1, n_steps + 1):
        current_len = k * stride
        # Drop books not long enough for this step.
        alive_books = [(idx, ids) for idx, ids in alive_books if ids.numel() >= current_len]
        if not alive_books:
            break

        oomed_this_step: set[int] = set()

        # Iterate in chunks of batch_size.
        for batch_start in range(0, len(alive_books), batch_size):
            batch = alive_books[batch_start:batch_start + batch_size]
            batch_idx = [idx for idx, _ in batch]
            batch_ids = torch.stack([ids[:current_len] for _, ids in batch])  # [B, current_len]

            try:
                nll_per_book = _score_batch(model, batch_ids, stride)
                n_scored = (
                    current_len - 1 if current_len == stride else stride
                )
                for j, b_idx in enumerate(batch_idx):
                    records.append(StepRecord(
                        book_idx=b_idx,
                        current_len=current_len,
                        sum_nll=nll_per_book[j].item(),
                        n_scored=n_scored,
                    ))
                del nll_per_book
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Fall back to bsz=1 for this batch — maybe the batched memory was tight.
                for j, b_idx in enumerate(batch_idx):
                    one_ids = batch_ids[j:j + 1]
                    try:
                        nll_one = _score_batch(model, one_ids, stride)
                        n_scored = current_len - 1 if current_len == stride else stride
                        records.append(StepRecord(
                            book_idx=b_idx,
                            current_len=current_len,
                            sum_nll=nll_one[0].item(),
                            n_scored=n_scored,
                        ))
                        del nll_one
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        oom_list.append((b_idx, current_len))
                        oomed_this_step.add(b_idx)
                    finally:
                        torch.cuda.empty_cache()

            del batch_ids
            torch.cuda.empty_cache()

        # Drop any books that OOM'd at this step from future steps.
        if oomed_this_step:
            alive_books = [(idx, ids) for idx, ids in alive_books if idx not in oomed_this_step]

        if pbar is not None:
            pbar.update(k * k)
            pbar.set_postfix(L=current_len, alive=len(alive_books), oom=len(oom_list))

    if pbar is not None:
        pbar.close()
    return records, oom_list


# Back-compat: single-book API used by tests / debug. Calls the batched path with bsz=1.
def evaluate_book_growing_window(
    model,
    input_ids_full: torch.Tensor,  # [L_book] int64 on CUDA
    book_idx: int,
    stride: int,
) -> tuple[list[StepRecord], int | None]:
    """Single-book wrapper (kept for back-compat); prefer the batched variant."""
    records, oom_list = evaluate_books_batched_growing_window(
        model,
        [(book_idx, input_ids_full)],
        stride=stride,
        l_max=input_ids_full.numel(),
        batch_size=1,
    )
    oom_at = oom_list[0][1] if oom_list else None
    return records, oom_at
