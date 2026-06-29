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

"""
Bypass `allenai/multi_lexsum`'s loading-script requirement by fetching the raw
JSON releases directly from the dataset repo and reassembling the DatasetDict
in-memory — no `trust_remote_code` needed.

Raw file layout (hf_hub, dataset repo):
    releases/<release>/sources.json   – dict[source_id → {doc_text, url, …}]  (2.2 GB)
    releases/<release>/train.json     – JSONL, one case per line
    releases/<release>/dev.json       – JSONL
    releases/<release>/test.json      – JSONL

Each case has `case_documents: list[source_id]` which we expand via the sources
map to produce the `sources: list[str]` field that HELMET expects, matching
exactly the schema produced by the upstream loading script.

Invoke `apply()` once after HELMET's `data.py` is importable. Returns True
if it patched, False if the original `load_dataset` already works (so this
module is harmless on `datasets<4`).
"""
from __future__ import annotations

import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

_RELEASE_DEFAULT = "v20230518"
_REPO_ID = "allenai/multi_lexsum"


def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _fetch(release: str) -> dict[str, str]:
    from huggingface_hub import hf_hub_download
    paths = {}
    for name in ("sources.json", "train.json", "dev.json", "test.json"):
        paths[name] = hf_hub_download(
            _REPO_ID, f"releases/{release}/{name}", repo_type="dataset",
        )
    return paths


def _build_split(case_rows: list[dict], sources: dict[str, dict],
                 include_sources: bool = True) -> list[dict]:
    """Join case rows with sources map; preserve HELMET's expected field names.

    When `include_sources=False`, `sources` is set to an empty list — cheap enough
    to build the train split for HELMET's 2-shot demo sampling, which only reads
    `summary/short` anyway.
    """
    out = []
    for case in case_rows:
        if include_sources:
            case_sources = [
                sources[sid]["doc_text"]
                for sid in case.get("case_documents", [])
                if sid in sources and sources[sid].get("doc_text")
            ]
        else:
            case_sources = []
        out.append({
            "id": case.get("case_id"),
            "sources": case_sources,
            "summary/long":  case.get("summary/long"),
            "summary/short": case.get("summary/short"),
            "summary/tiny":  case.get("summary/tiny"),
        })
    return out


def _cache_dir(release: str) -> str:
    base = os.environ.get(
        "HELMET_PATCH_CACHE",
        os.path.expanduser("~/.cache/helmet_patches"),
    )
    return os.path.join(base, f"multi_lexsum_{release}")


def _build_datasetdict(release: str = _RELEASE_DEFAULT):
    """Build a DatasetDict compatible with HELMET's usage, with on-disk cache.

    First run: parse 2.2 GB `sources.json`, join with each split, materialize
    to Arrow, and `save_to_disk` under `~/.cache/helmet_patches/…`. ~3 min.
    Subsequent runs: `load_from_disk` — seconds.

    Validation carries full `sources` (what HELMET actually summarizes); train
    keeps only the summary text for 2-shot demos (the sources column is an
    empty list). Test split is skipped — HELMET never reads it.
    """
    from datasets import Dataset, DatasetDict, load_from_disk
    import pyarrow as pa

    cache = _cache_dir(release)
    if os.path.isdir(cache) and os.path.isfile(os.path.join(cache, "dataset_dict.json")):
        return load_from_disk(cache)

    paths = _fetch(release)
    sources = _load_json(paths["sources.json"])  # ~2.2 GB

    val_rows   = _build_split(_load_jsonl(paths["dev.json"]),   sources, include_sources=True)
    train_rows = _build_split(_load_jsonl(paths["train.json"]), sources, include_sources=False)
    del sources  # free ~2 GB before Arrow materialization

    def _to_ds(rows: list[dict]):
        return Dataset(pa.Table.from_pylist(rows))

    dsd = DatasetDict({"train": _to_ds(train_rows), "validation": _to_ds(val_rows)})
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    dsd.save_to_disk(cache)
    return dsd


def apply() -> bool:
    """Patch HELMET's `data.load_dataset` to short-circuit on multi_lexsum.

    Returns True if patched, False if `datasets` still supports the upstream
    loader (nothing to do).
    """
    try:
        import datasets as _ds  # noqa: F401
        # cheap probe: attempt to import the loader-script mechanism
        from datasets import load_dataset
    except ImportError as e:
        logger.warning("datasets import failed, skipping multi_lexsum patch: %s", e)
        return False

    # We only patch when we know upstream loading scripts are disabled.
    # datasets 4+ removed support; check via version string.
    try:
        major = int(_ds.__version__.split(".", 1)[0])
    except Exception:
        major = 0
    if major < 4:
        return False

    import data as helmet_data
    original_load_dataset = helmet_data.load_dataset

    # Lazily build so we only pay the cost if the test actually requests it.
    _cache: dict[str, object] = {}

    def _patched(path, *args, **kwargs):
        if path == _REPO_ID:
            kwargs.pop("trust_remote_code", None)
            name = kwargs.get("name") or (args[0] if args else _RELEASE_DEFAULT)
            if name not in _cache:
                print(f"[multi_lexsum_patch] building DatasetDict for {_REPO_ID}:{name} …",
                      file=sys.stderr, flush=True)
                _cache[name] = _build_datasetdict(name)
                print(f"[multi_lexsum_patch] built splits: "
                      f"{ {k: len(v) for k, v in _cache[name].items()} }",
                      file=sys.stderr, flush=True)
            return _cache[name]
        return original_load_dataset(path, *args, **kwargs)

    helmet_data.load_dataset = _patched
    return True
