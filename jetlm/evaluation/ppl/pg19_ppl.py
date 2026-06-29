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

"""PG19 anchored growing-window PPL eval — distributed entry point.

One torchrun invocation = one variant. Produces a 64-point degradation curve
(2k, 4k, ..., 131072) across the first 100 PG19 test books, distributed across all
ranks via `chunk_list_interleaved` (each rank loads the full model — pure data
parallelism over books). Output: per-variant `curve.json` plus a per-step CSV row
appended to a global aggregated table.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict

import torch
import torch.distributed as dist
import yaml

from jetlm.utils import (
    chunk_list_interleaved,
    create_complete_marker,
    check_complete_marker,
    dist_barrier,
    dist_close,
    dist_init,
    get_dist_rank,
    get_dist_size,
    is_master,
    locked_atomic_write_json,
    master_print,
)
from jetlm.evaluation.ppl.growing_window import (
    StepRecord,
    evaluate_books_batched_growing_window,
)
from jetlm.evaluation.ppl.load import load_model_for_ppl
from jetlm.evaluation.ppl.pg19_dataset import load_pg19_books


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--eval_config", default="jetlm/evaluation/configs/pg19_ppl.yaml")
    p.add_argument("--output_dir", default="results/pg19_ppl")
    p.add_argument("--run_name", default="")
    p.add_argument("--version_str", default="v1")
    # Allow CLI overrides of the YAML (handy for smoke tests):
    p.add_argument("--l_max", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--num_books", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None,
                   help="books per forward pass on each rank (data-parallel within rank)")
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def variant_id(model_path: str) -> tuple[str, str]:
    """Parse model_cache/Qwen3-4B-Base-yarn -> (size='4B', method='yarn'). Baseline => method='baseline'."""
    base = os.path.basename(model_path.rstrip("/"))
    # Expected: Qwen3-<size>-Base[-<method>]
    if base.startswith("Qwen3-") and "-Base" in base:
        head = base[len("Qwen3-"):]
        size, _, tail = head.partition("-Base")
        method = tail.lstrip("-") if tail else "baseline"
        return size, method or "baseline"
    return base, "unknown"


def main():
    args = parse_args()
    cfg = load_yaml(args.eval_config)

    l_max = args.l_max if args.l_max is not None else int(cfg["l_max"])
    stride = args.stride if args.stride is not None else int(cfg["stride"])
    num_books = args.num_books if args.num_books is not None else int(cfg["num_books"])
    batch_size = (
        args.batch_size if args.batch_size is not None
        else int(cfg.get("batch_size", 1))
    )
    target_lengths_for_csv = cfg.get("target_lengths_for_csv", [65536, 131072])
    assert l_max % stride == 0, f"l_max ({l_max}) must be a multiple of stride ({stride})"

    dist_init(gpu=None, cudnn_benchmark=False)
    rank, world = get_dist_rank(), get_dist_size()

    size, method = variant_id(args.model_name_or_path)
    run_name = args.run_name or f"pg19_ppl-{size}-{method}"
    variant_dir = os.path.join(args.output_dir, "per_variant", f"{size}_{method}")
    res_path = os.path.join(variant_dir, "curve.json")

    if is_master():
        os.makedirs(variant_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(args.output_dir, "_")), exist_ok=True)
    dist_barrier()

    # Resume: skip if already complete with matching version_str.
    if check_complete_marker(res_path, version_str=args.version_str, clean_dir=False):
        master_print(f"[{run_name}] already complete at {res_path}, skipping.")
        dist_close()
        return

    # ------- Load model -------
    # Bare-baseline override: bump max_position_embeddings up to l_max so the model
    # accepts >32k sequences (it'll produce extrapolated RoPE — that's the OOD signal).
    max_pe_override = l_max if method == "baseline" else None
    master_print(
        f"[{run_name}] rank {rank}/{world} loading {args.model_name_or_path} "
        f"(max_pe_override={max_pe_override})"
    )
    t_load_start = time.perf_counter()
    model, tokenizer = load_model_for_ppl(args.model_name_or_path, max_pe_override=max_pe_override)
    master_print(f"[{run_name}] model loaded in {time.perf_counter() - t_load_start:.1f}s")

    # ------- Load + tokenize all 100 books on every rank -------
    # PG19 books are <50MB raw; tokenizing all of them is fast and avoids cross-rank
    # data-shape mismatch issues during the gather.
    t_tok_start = time.perf_counter()
    books = load_pg19_books(num_books=num_books, split=cfg.get("split", "test"))
    tokenized: list[tuple[int, torch.Tensor | None]] = []
    n_too_short = 0
    for book_idx, text in books:
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        if len(ids) < stride:
            tokenized.append((book_idx, None))
            n_too_short += 1
            continue
        # Truncate to l_max and round down to a multiple of stride.
        keep = (min(len(ids), l_max) // stride) * stride
        tokenized.append((book_idx, torch.tensor(ids[:keep], dtype=torch.long, device="cuda")))
    master_print(
        f"[{run_name}] tokenized {len(books)} books in {time.perf_counter() - t_tok_start:.1f}s "
        f"({n_too_short} too short to score)"
    )

    # ------- Per-rank book shard -------
    rank_shard = chunk_list_interleaved(tokenized, world)[rank]
    # Books that are too short to score (ids is None) are tracked separately.
    rank_skipped_short = [b_idx for b_idx, ids in rank_shard if ids is None]
    rank_books = [(b_idx, ids) for b_idx, ids in rank_shard if ids is not None]

    # ------- Evaluate (batched anchored-growing-window over this rank's books) -------
    t_eval_start = time.perf_counter()
    records, oom_list = evaluate_books_batched_growing_window(
        model,
        rank_books,
        stride=stride,
        l_max=l_max,
        batch_size=batch_size,
        show_progress=is_master(),
        progress_desc=f"{run_name}",
    )
    rank_records: list[dict] = [asdict(r) for r in records]
    rank_oom: list[tuple[int, int]] = list(oom_list)
    if oom_list:
        for b_idx, oom_at in oom_list:
            print(
                f"[{run_name}] rank {rank} book {b_idx} OOM at current_len={oom_at}",
                flush=True,
            )

    rank_wallclock = time.perf_counter() - t_eval_start
    master_print(f"[{run_name}] rank-local eval done in {rank_wallclock:.1f}s, "
                 f"{len(rank_records)} step records, {len(rank_oom)} OOM books, "
                 f"batch_size={batch_size}, books_on_rank={len(rank_books)}")

    # ------- Cross-rank gather -------
    payload = {
        "records": rank_records,
        "oom": rank_oom,
        "skipped_short": rank_skipped_short,
        "rank_wallclock": rank_wallclock,
    }
    gathered: list[dict] = [None] * world  # type: ignore
    dist.all_gather_object(gathered, payload)

    if not is_master():
        dist_close()
        return

    # ------- Aggregate on master -------
    all_records: list[dict] = []
    all_oom: list[tuple[int, int]] = []
    all_skipped: list[int] = []
    max_rank_wallclock = 0.0
    for chunk in gathered:
        all_records.extend(chunk["records"])
        all_oom.extend(chunk["oom"])
        all_skipped.extend(chunk["skipped_short"])
        max_rank_wallclock = max(max_rank_wallclock, chunk["rank_wallclock"])

    # group by current_len -> sum_nll, n_scored, n_books_scored
    by_len = defaultdict(lambda: {"sum_nll": 0.0, "n_scored": 0, "books": set()})
    for r in all_records:
        L = int(r["current_len"])
        by_len[L]["sum_nll"] += float(r["sum_nll"])
        by_len[L]["n_scored"] += int(r["n_scored"])
        by_len[L]["books"].add(int(r["book_idx"]))

    # books OOM at-or-before each current_len:
    oom_books_below = defaultdict(set)  # current_len -> set of book_idx that OOMed strictly above this length? we want at_or_before
    for book_idx, oom_at in all_oom:
        # any current_len >= oom_at is unobserved for this book.
        oom_books_below[oom_at].add(book_idx)

    curve = []
    for L in sorted(by_len.keys()):
        d = by_len[L]
        # n_books_oom_at_or_before: number of books that hit OOM at any current_len <= L
        n_oom_le = sum(1 for _, oom_at in all_oom if oom_at <= L)
        nll_sum = d["sum_nll"]
        n_scored = d["n_scored"]
        ppl = math.exp(nll_sum / n_scored) if n_scored > 0 else float("nan")
        curve.append({
            "current_len": L,
            "ppl": ppl,
            "sum_nll": nll_sum,
            "n_scored_tokens": n_scored,
            "n_books_scored": len(d["books"]),
            "n_books_skipped_short": len(set(all_skipped)),
            "n_books_oom_at_or_before": n_oom_le,
        })

    summary = {
        "schema_version": "v1",
        "run_name": run_name,
        "size": size,
        "method": method,
        "model_path": args.model_name_or_path,
        "config": {
            "stride": stride,
            "l_max": l_max,
            "num_books": num_books,
            "batch_size": batch_size,
            "mode": "anchored_growing",
            "target_lengths_for_csv": target_lengths_for_csv,
        },
        "curve": curve,
        "wallclock_sec_max_rank": max_rank_wallclock,
        "world_size": world,
        "n_books_oom_total": len({b for b, _ in all_oom}),
        "n_books_skipped_short_total": len(set(all_skipped)),
        "status": "ok" if not all_oom else "partial_oom",
    }

    locked_atomic_write_json(res_path, summary, indent=2)

    # ------- Persist per-(book, current_len) records as JSONL ----------------
    # The curve already encodes the by-length aggregate; the JSONL keeps the
    # finer-grained per-book breakdown so downstream paired statistics
    # (paired t, sign test, bootstrap over books) are possible without
    # re-running the eval.
    per_book_path = os.path.join(variant_dir, "per_book_records.jsonl")
    _dump_per_book_records(per_book_path, all_records, curve)

    create_complete_marker(res_path, version_str=args.version_str)

    # ------- Append to global aggregated table -------
    agg_json = os.path.join(args.output_dir, "pg19_ppl_results.json")
    agg_csv = os.path.join(args.output_dir, "pg19_ppl_results.csv")
    _append_aggregated(agg_json, agg_csv, summary)
    master_print(f"[{run_name}] wrote {res_path} and updated {agg_json}/.csv")

    # ------- W&B logging (master only, opt-out via WANDB_DISABLED=true) -------
    _log_to_wandb(
        run_name=run_name, summary=summary, version_str=args.version_str,
        checkpoint_lengths=cfg.get("wandb_checkpoint_lengths", [32768, 65536, 98304, 131072]),
    )

    dist_close()


def _log_to_wandb(
    run_name: str,
    summary: dict,
    version_str: str,
    checkpoint_lengths: list[int],
) -> None:
    """Plot the full PPL-vs-context_length curve in W&B.

    Each curve point is logged with `step = current_len`, so the default x-axis is
    sequence length and y is `ppl`. The 4 user-flagged checkpoints
    (32768/65536/98304/131072 by default) are also dropped into `wandb.summary`
    as `ppl_at_<L>` for cross-run comparison tables.
    """
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return
    try:
        import wandb
    except ImportError:
        master_print(f"[{run_name}] wandb not installed; skipping W&B logging")
        return

    try:
        run_id = hashlib.sha1((run_name + version_str).encode()).hexdigest()[:32]
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "jet-long"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=run_name,
            id=run_id,
            config={
                "size": summary["size"],
                "method": summary["method"],
                "model_path": summary["model_path"],
                "world_size": summary["world_size"],
                **summary["config"],
            },
            resume=os.environ.get("WANDB_RESUME", "allow"),
            job_type=os.environ.get("WANDB_RUNTYPE", "eval_ppl"),
        )

        for pt in summary["curve"]:
            wandb_run.log(
                {
                    "ppl": pt["ppl"],
                    "n_books_scored": pt["n_books_scored"],
                    "n_books_oom_at_or_before": pt["n_books_oom_at_or_before"],
                },
                step=int(pt["current_len"]),
            )

        # Named checkpoints — copy onto wandb.summary for easy cross-run comparison.
        curve_by_len = {pt["current_len"]: pt for pt in summary["curve"]}
        ckpt_summary: dict = {}
        for L in checkpoint_lengths:
            pt = curve_by_len.get(L)
            if pt is not None:
                ckpt_summary[f"ppl_at_{L}"] = pt["ppl"]
                ckpt_summary[f"n_books_at_{L}"] = pt["n_books_scored"]
        ckpt_summary["status"] = summary["status"]
        ckpt_summary["wallclock_sec_max_rank"] = summary["wallclock_sec_max_rank"]
        wandb_run.summary.update(ckpt_summary)
        wandb_run.finish()
        master_print(f"[{run_name}] logged {len(summary['curve'])} curve points + "
                     f"{len([k for k in ckpt_summary if k.startswith('ppl_at_')])} "
                     f"named checkpoints to W&B (run_id={run_id})")
    except Exception as e:
        master_print(f"[{run_name}] W&B logging failed: {type(e).__name__}: {e}")


def _locked_atomic_write_csv(
    path: str, fieldnames: list[str], rows: list[dict], *, timeout: float = 10.0
) -> None:
    """Concurrent-safe CSV writer.

    Mirrors the lock + tmp+rename pattern of `locked_atomic_write_json` so that
    multiple sbatch array jobs racing on the same cross-variant CSV (e.g. with
    `--array=1-18%4`) cannot interleave bytes or leave a truncated file.
    """
    import csv as _csv
    import portalocker
    import tempfile as _tempfile

    lock_path = path + ".lock"
    with portalocker.Lock(lock_path, mode="a+", flags=portalocker.LOCK_EX, timeout=timeout):
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = _tempfile.mkstemp(prefix=".tmp.", suffix=".csv", dir=d, text=True)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass


def _dump_per_book_records(path: str, all_records: list[dict], curve: list[dict]) -> None:
    """Persist the per-(book_idx, current_len) NLL records as JSONL.

    Sort order is `(book_idx asc, current_len asc)` for stable diffs across reruns.
    Verifies that the per-book sums aggregate exactly back to the corresponding
    `curve` entries; raises AssertionError on drift (this catches any future bug
    in the gather / aggregation path).

    Atomic write: write to `path + ".tmp"`, fsync, then `os.replace`.
    """
    sorted_records = sorted(
        all_records,
        key=lambda r: (int(r["book_idx"]), int(r["current_len"])),
    )

    # Aggregation cross-check: per-book sums must match the by-length curve.
    by_len_check: dict[int, dict[str, float]] = defaultdict(
        lambda: {"sum_nll": 0.0, "n_scored": 0}
    )
    for r in sorted_records:
        L = int(r["current_len"])
        by_len_check[L]["sum_nll"] += float(r["sum_nll"])
        by_len_check[L]["n_scored"] += int(r["n_scored"])
    for pt in curve:
        L = int(pt["current_len"])
        check = by_len_check.get(L)
        assert check is not None, f"per-book records missing current_len={L}"
        # Allow tiny float drift from differing summation order; n_scored is exact.
        assert abs(check["sum_nll"] - float(pt["sum_nll"])) < max(
            1e-3, 1e-6 * abs(float(pt["sum_nll"]))
        ), (
            f"per-book sum_nll at L={L} disagrees with curve: "
            f"{check['sum_nll']} vs {pt['sum_nll']}"
        )
        assert int(check["n_scored"]) == int(pt["n_scored_tokens"]), (
            f"per-book n_scored at L={L} disagrees with curve: "
            f"{check['n_scored']} vs {pt['n_scored_tokens']}"
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for r in sorted_records:
            f.write(json.dumps({
                "book_idx": int(r["book_idx"]),
                "current_len": int(r["current_len"]),
                "sum_nll": float(r["sum_nll"]),
                "n_scored": int(r["n_scored"]),
            }) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _append_aggregated(agg_json: str, agg_csv: str, summary: dict) -> None:
    """Atomically merge this variant's summary into the cross-variant aggregated outputs."""
    from jetlm.utils.ckpt import locked_read_json

    existing = locked_read_json(agg_json, default={"schema_version": "v1", "rows": []})
    rows = list(existing.get("rows", []))
    # Remove any prior row for the same (size, method).
    rows = [r for r in rows if not (r.get("size") == summary["size"] and r.get("method") == summary["method"])]

    curve_by_len = {pt["current_len"]: pt for pt in summary["curve"]}
    targets = summary["config"]["target_lengths_for_csv"]
    new_row = {
        "size": summary["size"],
        "method": summary["method"],
        "status": summary["status"],
        **{
            f"ppl_at_{L}": (curve_by_len[L]["ppl"] if L in curve_by_len else None)
            for L in targets
        },
        **{
            f"n_books_at_{L}": (curve_by_len[L]["n_books_scored"] if L in curve_by_len else 0)
            for L in targets
        },
    }
    rows.append(new_row)
    locked_atomic_write_json(agg_json, {"schema_version": "v1", "rows": rows}, indent=2)

    # Long-format CSV — rebuild from every per-variant curve.json so a re-run of one
    # variant always yields a self-consistent CSV that includes all completed variants.
    out_dir = os.path.dirname(agg_json)
    per_variant_root = os.path.join(out_dir, "per_variant")
    csv_records: list[dict] = []
    if os.path.isdir(per_variant_root):
        for entry in sorted(os.listdir(per_variant_root)):
            curve_path = os.path.join(per_variant_root, entry, "curve.json")
            if not os.path.isfile(curve_path):
                continue
            try:
                with open(curve_path) as f:
                    s = json.load(f)
            except json.JSONDecodeError:
                continue
            for pt in s.get("curve", []):
                csv_records.append({
                    "size": s.get("size", ""),
                    "method": s.get("method", ""),
                    "current_len": pt["current_len"],
                    "ppl": pt["ppl"],
                    "n_books_scored": pt["n_books_scored"],
                    "n_books_oom_at_or_before": pt["n_books_oom_at_or_before"],
                    "status": s.get("status", ""),
                })
    if csv_records:
        fieldnames = ["size", "method", "current_len", "ppl",
                      "n_books_scored", "n_books_oom_at_or_before", "status"]
        _locked_atomic_write_csv(agg_csv, fieldnames, csv_records)

    # Per-(book, current_len) cross-variant CSV — rebuild from every per-variant
    # `per_book_records.jsonl`. Best-effort: skip variants that don't have one
    # (e.g. completed before this artifact was added).
    pb_csv = os.path.join(out_dir, "pg19_ppl_per_book_records.csv")
    pb_records: list[dict] = []
    missing_pb: list[str] = []
    if os.path.isdir(per_variant_root):
        for entry in sorted(os.listdir(per_variant_root)):
            curve_path = os.path.join(per_variant_root, entry, "curve.json")
            pb_path = os.path.join(per_variant_root, entry, "per_book_records.jsonl")
            if not os.path.isfile(curve_path):
                continue
            if not os.path.isfile(pb_path):
                missing_pb.append(entry)
                continue
            try:
                with open(curve_path) as f:
                    s = json.load(f)
            except json.JSONDecodeError:
                continue
            size = s.get("size", "")
            method = s.get("method", "")
            with open(pb_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pb_records.append({
                        "size": size,
                        "method": method,
                        "book_idx": int(r["book_idx"]),
                        "current_len": int(r["current_len"]),
                        "sum_nll": float(r["sum_nll"]),
                        "n_scored": int(r["n_scored"]),
                    })
    if pb_records:
        pb_fieldnames = ["size", "method", "book_idx", "current_len", "sum_nll", "n_scored"]
        _locked_atomic_write_csv(pb_csv, pb_fieldnames, pb_records)
    if missing_pb:
        print(
            f"[per_book_records] WARN: {len(missing_pb)} variant(s) have curve.json but "
            f"no per_book_records.jsonl (likely from a pre-patch run): {missing_pb}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
