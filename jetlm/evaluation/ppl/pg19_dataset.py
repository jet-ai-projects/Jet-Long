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

"""PG19 book loader for the perplexity eval.

Two sources, in priority order:

1. **Local subsample set** (`processed_data/PG-19-subsample/books.jsonl`) — the 100-book
   subset built by `scripts/data/subsample_pg19.py`, all guaranteed >= 131k Qwen3 tokens.
   Use this when you want every curve point to have identical `n_books_scored`.
   Selected when `split == "subsample"` (or `split == "auto"` and the subsample dir exists).

2. **HuggingFace `emozilla/pg19`** parquet mirror — full splits (test/val/train).
   The original `deepmind/pg19` dataset script is no longer supported by `datasets >= 3.x`.
   Selected when `split` is one of {test, validation, train}.
"""

import json
import os
from typing import Iterator


PG19_REPO = "emozilla/pg19"
PG19_SUBSAMPLE_DIR = "processed_data/PG-19-subsample"
PG19_SUBSAMPLE_BOOKS = os.path.join(PG19_SUBSAMPLE_DIR, "books.jsonl")


def _subsample_available() -> bool:
    return os.path.isfile(PG19_SUBSAMPLE_BOOKS)


def _iter_subsample(num_books: int) -> Iterator[tuple[int, str]]:
    with open(PG19_SUBSAMPLE_BOOKS) as f:
        for line in f:
            entry = json.loads(line)
            idx = int(entry["idx"])
            if idx >= num_books:
                break
            yield idx, entry["text"]


def _iter_hf(num_books: int, split: str) -> Iterator[tuple[int, str]]:
    from datasets import load_dataset

    ds = load_dataset(PG19_REPO, split=split)
    n = min(num_books, len(ds))
    for i in range(n):
        yield i, ds[i]["text"]


def iter_pg19_books(num_books: int = 100, split: str = "subsample") -> Iterator[tuple[int, str]]:
    """Yield (book_idx, text) for the first `num_books` books.

    `split` values:
      - "subsample"        → use processed_data/PG-19-subsample/books.jsonl (raises if missing)
      - "auto"             → "subsample" if available, else "test"
      - "test" / "validation" / "train" → HF emozilla/pg19 split
    """
    if split == "auto":
        split = "subsample" if _subsample_available() else "test"
    if split == "subsample":
        if not _subsample_available():
            raise FileNotFoundError(
                f"PG19 subsample set not found at {PG19_SUBSAMPLE_BOOKS}. "
                "Build it with `python scripts/data/subsample_pg19.py` or download "
                "with `python scripts/data/download_data.py --dataset pg19sub`."
            )
        return _iter_subsample(num_books)
    return _iter_hf(num_books, split)


def load_pg19_books(num_books: int = 100, split: str = "subsample") -> list[tuple[int, str]]:
    return list(iter_pg19_books(num_books=num_books, split=split))
