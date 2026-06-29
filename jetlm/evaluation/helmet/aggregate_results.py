#!/usr/bin/env python
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
Walk a HELMET output dir, read every `*.json.score` file, bucket results by
(category, dataset, length), and write `result.json` at the top level.

Canonical HELMET metrics (from `tmp/helmet/repo/scripts/collect_results.py`):
  substring_exact_match   (SubEM) — RAG (kilt_*), recall (json_kv)
  ruler_recall                    — recall (ruler_niah_*)
  NDCG@10                         — re-ranking
  exact_match                     — ICL, infbench_choice
  rougeL_f1                       — infbench_qa, and GPT-4 judge fallback
  gpt-4-score / gpt-4-f1          — NarrativeQA, infbench_sum, multi_lexsum
                                    (only present after running the separate
                                    GPT-4 judge scripts; we fall back to
                                    rougeL_f1 + flag it)
  str_em / citation_rec / citation_prec / qampari_rec_top5 — ALCE cite

`result.json` layout:
  {
    "model":        <str>,
    "tag":          <str>,
    "num_tasks":    <int>,
    "per_task":     [ {category, dataset, length, metric, score, judge_pending, file}, ... ],
    "category_length": { "rag": {"8192": 27.9, ..., "avg": 31.2}, ... },
    "category_avg":    { "rag": 29.3, ... },
    "length_avg":      { "8192": 31.2, ... },
    "overall_avg":  <float>
  }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


# Dataset-prefix → (metric_key, category, gpt4_judge_required?)
# Order matters: longer/more-specific prefixes first so they win on `startswith`.
DATASET_METRIC = [
    # synthetic recall
    ("json_kv",             "substring_exact_match", "recall", False),
    ("ruler_niah_mk_1",     "ruler_recall",          "recall", False),
    ("ruler_niah_mk_2",     "ruler_recall",          "recall", False),
    ("ruler_niah_mk_3",     "ruler_recall",          "recall", False),
    ("ruler_niah_mq",       "ruler_recall",          "recall", False),
    ("ruler_niah_mv",       "ruler_recall",          "recall", False),
    ("ruler_niah_s_1",      "ruler_recall",          "recall", False),
    ("ruler_niah_s_2",      "ruler_recall",          "recall", False),
    ("ruler_niah_s_3",      "ruler_recall",          "recall", False),
    ("ruler_fwe",           "ruler_recall",          "recall", False),
    ("ruler_cwe",           "ruler_recall",          "recall", False),
    ("ruler_vt",            "ruler_recall",          "recall", False),
    ("ruler_qa_1",          "substring_exact_match", "recall", False),
    ("ruler_qa_2",          "substring_exact_match", "recall", False),

    # RAG
    ("kilt_nq",             "substring_exact_match", "rag", False),
    ("kilt_triviaqa",       "substring_exact_match", "rag", False),
    ("kilt_hotpotqa",       "substring_exact_match", "rag", False),
    ("kilt_popqa",          "substring_exact_match", "rag", False),

    # citations (ALCE). Main score = str_em / qampari_rec_top5; citation metrics also tracked.
    ("alce_asqa",           "str_em",                "cite", False),
    ("alce_qampari",        "qampari_rec_top5",      "cite", False),

    # re-ranking
    ("msmarco_rerank_psg",  "NDCG@10",               "rerank", False),

    # long-doc QA — true metric = gpt-4-score; we fall back to rougeL_f1
    ("narrativeqa",         "rougeL_f1",             "longqa", True),
    ("infbench_qa_eng",     "rougeL_f1",             "longqa", False),  # upstream uses rougeL_f1
    ("infbench_choice_eng", "exact_match",           "longqa", False),

    # summarization — true metric = gpt-4-f1; fallback to rougeL_f1
    ("infbench_sum_eng",    "rougeL_f1",             "summ", True),
    ("multi_lexsum",        "rougeL_f1",             "summ", True),

    # ICL
    ("icl_trec_coarse",     "exact_match",           "icl", False),
    ("icl_trec_fine",       "exact_match",           "icl", False),
    ("icl_banking77",       "exact_match",           "icl", False),
    ("icl_clinic150",       "exact_match",           "icl", False),
    ("icl_nlu",             "exact_match",           "icl", False),
]


# Filename format from eval.py:37:
#   {dataset}_{tag}_{test_name}_in{L}_size{N}_shots{S}_samp{D}max{G}min{m}t{T}p{P}_chat{C}_{seed}.json
# `dataset` itself may contain underscores (e.g. `kilt_nq`, `ruler_niah_mk_2`,
# `narrativeqa_130772`). The trailing `_in<L>_size<N>_…_<seed>.json` fragment
# is stable, so we anchor on that and enumerate known dataset prefixes.
_TAIL_RE = re.compile(
    r"^(?P<head>.+)_in(?P<length>\d+)_size(?P<size>\d+)_.*_(?P<seed>\d+)\.json(?:\.score)?$"
)


def _split_head(head: str) -> tuple[str, str, str] | None:
    """Split `head = {dataset}_{tag}_{test_name}` by matching the longest dataset
    prefix from DATASET_METRIC. Returns (dataset, tag, test_name) or None.

    Dataset names with numeric suffixes (e.g. `narrativeqa_130772`) are matched
    by the base key (`narrativeqa`); the suffix stays with the dataset token.
    """
    for key, _, _, _ in sorted(DATASET_METRIC, key=lambda x: -len(x[0])):
        if head.startswith(key + "_"):
            remainder = head[len(key) + 1:]
            # Dataset may be `key` or `key_<numsuffix>`. Determine which by peeking.
            toks = remainder.split("_", 2)
            if toks and toks[0].isdigit():
                dataset = f"{key}_{toks[0]}"
                rest = remainder[len(toks[0]) + 1:]
            else:
                dataset = key
                rest = remainder
            # rest = `{tag}_{test_name}`; tag has no underscore.
            tag, _, test_name = rest.partition("_")
            return dataset, tag, test_name
        if head.startswith(key) and len(head) == len(key):
            return key, "", ""
    return None


def _dataset_metric(dataset: str) -> tuple[str, str, str, bool] | None:
    """Return (canonical_dataset, metric, category, judge_pending)."""
    # Strip numeric length suffix like `_130772`.
    parts = dataset.rsplit("_", 1)
    base = parts[0] if len(parts) == 2 and parts[1].isdigit() else dataset
    for key, metric, category, judge in sorted(DATASET_METRIC, key=lambda x: -len(x[0])):
        if base == key or base.startswith(key + "_"):
            return (key, metric, category, judge)
    return None


def parse_score_file(path: Path) -> dict | None:
    """Return a per-task record or None if we can't classify."""
    m = _TAIL_RE.match(path.name)
    if not m:
        return None
    length = int(m.group("length"))

    split = _split_head(m.group("head"))
    if split is None:
        return None
    dataset, tag, _test_name = split

    info = _dataset_metric(dataset)
    if info is None:
        return None
    canonical_dataset, metric, category, judge_pending = info

    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    # Prefer the GPT-4 judge output if someone ran it alongside (`*-gpt4eval_o.json`)
    gpt4 = path.with_name(path.name.replace(".json.score", "-gpt4eval_o.json"))
    if gpt4.exists():
        try:
            gpt4_data = json.loads(gpt4.read_text()).get("averaged_metrics", {})
            if category == "longqa" and canonical_dataset == "narrativeqa" and "gpt-4-score" in gpt4_data:
                return _record(path, canonical_dataset, length, tag, category,
                               "gpt-4-score", gpt4_data["gpt-4-score"] * (100 / 3), False, data)
            if category == "summ" and "gpt-4-f1" in gpt4_data:
                return _record(path, canonical_dataset, length, tag, category,
                               "gpt-4-f1", gpt4_data["gpt-4-f1"] * 100, False, data)
        except Exception:
            pass  # fall through to surrogate

    if metric not in data:
        return None
    return _record(path, canonical_dataset, length, tag, category,
                   metric, float(data[metric]), judge_pending, data)


def _record(path: Path, dataset: str, length: int, tag: str, category: str,
            metric: str, score: float, judge_pending: bool, raw: dict) -> dict:
    rec = {
        "category":      category,
        "dataset":       dataset,
        "length":        length,
        "tag":           tag,
        "metric":        metric,
        "score":         round(score, 2),
        "judge_pending": judge_pending,
        "file":          path.name,
    }
    # handy extras, kept small
    for k in ("input_len", "output_len"):
        if k in raw:
            rec[k] = round(float(raw[k]), 1)
    # For ALCE, also record citation metrics for inspection
    if category == "cite":
        for k in ("citation_rec", "citation_prec"):
            if k in raw:
                rec[k] = round(float(raw[k]), 2)
    return rec


def aggregate(output_dir: Path, tag_filter: str | None = None) -> dict:
    per_task: list[dict] = []
    for p in sorted(output_dir.glob("*.json.score")):
        rec = parse_score_file(p)
        if rec is None:
            continue
        if tag_filter is not None and rec["tag"] != tag_filter:
            continue
        per_task.append(rec)

    # bucket by (category, length) → [scores]
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in per_task:
        buckets[(r["category"], r["length"])].append(r["score"])

    category_length: dict[str, dict[str, float]] = defaultdict(dict)
    for (cat, length), scores in buckets.items():
        category_length[cat][str(length)] = round(mean(scores), 2)

    # category avg = mean over its per-length means
    category_avg = {
        cat: round(mean(v for v in lengths.values()), 2)
        for cat, lengths in category_length.items()
    }
    for cat, lengths in category_length.items():
        lengths["avg"] = category_avg[cat]

    # length avg = mean over categories at that length
    all_lengths = sorted({r["length"] for r in per_task})
    length_avg = {}
    for length in all_lengths:
        vals = [category_length[cat][str(length)]
                for cat in category_length
                if str(length) in category_length[cat]]
        if vals:
            length_avg[str(length)] = round(mean(vals), 2)

    overall = round(mean(category_avg.values()), 2) if category_avg else None

    return {
        "model":       output_dir.name,
        "tag":         tag_filter,
        "num_tasks":   len(per_task),
        "category_length": dict(category_length),
        "category_avg":    category_avg,
        "length_avg":      length_avg,
        "overall_avg":     overall,
        "judge_pending_categories": sorted({
            r["category"] for r in per_task if r["judge_pending"]
        }),
        "per_task":    per_task,
    }


def print_summary(agg: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"HELMET aggregate  model={agg['model']}  tag={agg['tag']}  tasks={agg['num_tasks']}")
    print(f"{'=' * 78}")
    if not agg["per_task"]:
        print("  (no results found)"); return
    lengths = sorted({int(L) for cat in agg["category_length"].values() for L in cat if L != "avg"})
    header = "  category    " + "  ".join(f"{l:>8}" for l in lengths) + "       avg"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cat in sorted(agg["category_length"]):
        row = [f"  {cat:<11}"]
        for L in lengths:
            v = agg["category_length"][cat].get(str(L))
            row.append(f"{v:>8.2f}" if v is not None else f"{'—':>8}")
        row.append(f"    {agg['category_avg'][cat]:>6.2f}")
        print("  ".join(row))
    print("  " + "-" * (len(header) - 2))
    lrow = ["  length_avg "] + [f"{agg['length_avg'].get(str(l),'—'):>8}" if agg['length_avg'].get(str(l)) is None
                                else f"{agg['length_avg'][str(l)]:>8.2f}" for l in lengths]
    lrow.append(f"    {agg['overall_avg']:>6.2f}" if agg['overall_avg'] is not None else "     —")
    print("  ".join(lrow))
    if agg["judge_pending_categories"]:
        print(f"\n  note: {', '.join(agg['judge_pending_categories'])} use surrogate metrics "
              f"(rougeL_f1); for paper-faithful numbers run the GPT-4 judge:")
        print(f"        python tmp/helmet/repo/scripts/eval_gpt4_longqa.py --tag {agg['tag']} "
              f"--model_name_or_path <model>")
        print(f"        python tmp/helmet/repo/scripts/eval_gpt4_summ.py   --tag {agg['tag']} "
              f"--model_name_or_path <model>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                    help="The HELMET per-model output dir (e.g. results/helmet/Qwen3-1.7B-Base-jetlong)")
    ap.add_argument("--tag", default=None,
                    help="Only aggregate files matching this tag (defaults to all)")
    ap.add_argument("--out", default="result.json",
                    help="Where to write the summary (relative to output_dir)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"ERROR: {output_dir} is not a directory", file=sys.stderr); sys.exit(2)

    agg = aggregate(output_dir, tag_filter=args.tag)
    out_path = output_dir / args.out
    out_path.write_text(json.dumps(agg, indent=2))
    print_summary(agg)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
