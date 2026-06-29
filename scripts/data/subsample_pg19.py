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

"""Build a 100-book subsample of PG19 where every book has >= MIN_TOKENS Qwen3 tokens.

Layout produced:
    processed_data/PG-19-subsample/
    ├── manifest.json   # metadata for every book (no text)
    └── books.jsonl     # one record per line: {idx, source_split, source_index, n_tokens, text}

Selection rule (deterministic for a given --seed):
    - All books from PG19 *test* split that have >= MIN_TOKENS tokens (typically 18 books).
    - Fill the remainder up to TARGET_TOTAL with random books from PG19 *train* whose
      tokenized length is >= MIN_TOKENS (yield is ~25%, we walk a shuffled index list).
    - Records are written in (test-first, then train) order; `idx` is the subsample index
      in [0, TARGET_TOTAL).

Usage:
    python scripts/data/subsample_pg19.py
    python scripts/data/subsample_pg19.py --target_total 100 --min_tokens 131072 --seed 0
"""

import argparse
import json
import os
import random
import sys
import time

from datasets import load_dataset
from transformers import AutoTokenizer


PG19_REPO = "emozilla/pg19"
DEFAULT_TOKENIZER = "model_cache/Qwen3-1.7B-Base"  # any Qwen3 tokenizer is fine — same vocab
DEFAULT_OUT_DIR = "processed_data/PG-19-subsample"
DEFAULT_MIN_TOKENS = 131072
DEFAULT_TARGET_TOTAL = 100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min_tokens", type=int, default=DEFAULT_MIN_TOKENS,
                   help="filter threshold (default 131072)")
    p.add_argument("--target_total", type=int, default=DEFAULT_TARGET_TOTAL,
                   help="size of the subsample (default 100)")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="path to Qwen3 tokenizer (any size; same vocab)")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_train_scan", type=int, default=2000,
                   help="cap on how many train books we'll tokenize while searching")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # ----- test split: take ALL books that qualify -----
    print(f"Loading PG19 test split...", flush=True)
    test_ds = load_dataset(PG19_REPO, split="test")
    print(f"  scanning {len(test_ds)} test books for >= {args.min_tokens} tokens...", flush=True)

    test_long: list[dict] = []
    t0 = time.perf_counter()
    for i in range(len(test_ds)):
        text = test_ds[i]["text"]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= args.min_tokens:
            test_long.append({
                "source_split": "test",
                "source_index": i,
                "n_tokens": len(ids),
                "text": text,
            })
    print(f"  -> {len(test_long)} test books qualify "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    # ----- train split: random walk until we have enough -----
    n_needed_from_train = max(0, args.target_total - len(test_long))
    train_long: list[dict] = []
    if n_needed_from_train > 0:
        print(f"Loading PG19 train split (need {n_needed_from_train} more long books)...",
              flush=True)
        train_ds = load_dataset(PG19_REPO, split="train")
        rng = random.Random(args.seed)
        train_indices = list(range(len(train_ds)))
        rng.shuffle(train_indices)

        t1 = time.perf_counter()
        scanned = 0
        for i in train_indices:
            scanned += 1
            if scanned > args.max_train_scan:
                print(f"  [warn] hit --max_train_scan={args.max_train_scan} with only "
                      f"{len(train_long)}/{n_needed_from_train} train books found.",
                      flush=True)
                break
            text = train_ds[i]["text"]
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= args.min_tokens:
                train_long.append({
                    "source_split": "train",
                    "source_index": i,
                    "n_tokens": len(ids),
                    "text": text,
                })
                if len(train_long) % 10 == 0 or len(train_long) >= n_needed_from_train:
                    print(f"  found {len(train_long)} long books "
                          f"after scanning {scanned} ({time.perf_counter() - t1:.1f}s)",
                          flush=True)
                if len(train_long) >= n_needed_from_train:
                    break

    # ----- combine + write -----
    books = test_long + train_long
    if len(books) < args.target_total:
        print(f"[warn] subsample has only {len(books)} books "
              f"(target {args.target_total}). Bump --max_train_scan or lower --min_tokens.",
              flush=True)
    else:
        books = books[:args.target_total]

    os.makedirs(args.out_dir, exist_ok=True)

    manifest_records = [
        {
            "idx": i,
            "source_split": b["source_split"],
            "source_index": b["source_index"],
            "n_tokens": b["n_tokens"],
        }
        for i, b in enumerate(books)
    ]
    manifest = {
        "tokenizer": args.tokenizer,
        "min_tokens": args.min_tokens,
        "target_total": args.target_total,
        "actual_total": len(books),
        "n_from_test": sum(1 for b in books if b["source_split"] == "test"),
        "n_from_train": sum(1 for b in books if b["source_split"] == "train"),
        "seed": args.seed,
        "books": manifest_records,
    }

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    books_path = os.path.join(args.out_dir, "books.jsonl")
    with open(books_path, "w") as f:
        for i, b in enumerate(books):
            f.write(json.dumps({"idx": i, **b}) + "\n")

    total_size = (
        os.path.getsize(manifest_path) + os.path.getsize(books_path)
    )
    print(f"\n=== PG-19 Subsample ===", flush=True)
    print(f"  books:        {len(books)} (test={manifest['n_from_test']}, "
          f"train={manifest['n_from_train']})", flush=True)
    print(f"  min_tokens:   {args.min_tokens}", flush=True)
    print(f"  total bytes:  {total_size / 1e6:.1f} MB", flush=True)
    print(f"  manifest:     {manifest_path}", flush=True)
    print(f"  books.jsonl:  {books_path}", flush=True)
    if len(books) < args.target_total:
        sys.exit(2)


if __name__ == "__main__":
    main()
