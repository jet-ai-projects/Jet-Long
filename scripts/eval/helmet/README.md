# HELMET evaluation launcher

Thin driver that runs the upstream HELMET benchmark
([princeton-nlp/HELMET](https://github.com/princeton-nlp/HELMET)) against the
jetlm model variants under `model_cache/`. HELMET is kept as a sibling
checkout in `tmp/helmet/repo` so we do not modify its code and our scores stay
comparable to the public leaderboard.

## Layout

Shell launchers (this directory):
- [set_up_helmet.sh](set_up_helmet.sh) — one-shot HELMET setup: clones the
  upstream repo into `tmp/helmet/repo`, downloads + extracts the ~11 GB
  `data.tar.gz`, pre-fetches HF datasets for LongQA + Summ, builds the
  multi_lexsum compat cache. Idempotent.
- [run_helmet.sh](run_helmet.sh) — single-model HELMET eval. Sets up output
  dirs, calls `helmet_shard.py`, then runs `aggregate_results.py`.
- [run_gpt4o_judges.sh](run_gpt4o_judges.sh) — sweep the GPT-4o judges over
  every model under `results/helmet/` (or a positional subset) and refresh
  each `result.json` with the paper-faithful `gpt-4-score` / `gpt-4-f1`
  metrics. Auto-discovers, file-presence-checks, dry-run-able.

Python driver code (under `jetlm/evaluation/helmet/`):
- [`helmet_shard.py`](../../../jetlm/evaluation/helmet/helmet_shard.py) —
  task-sharding driver. Parses a set of HELMET config YAMLs, expands their
  comma-lists into `(category, dataset, length)` tuples, and runs them as
  independent `eval.py` subprocesses, one per GPU, via a thread-based worker
  pool. Output filenames include every hyperparameter so concurrent writes
  across GPUs never collide.
- [`patched_eval.py`](../../../jetlm/evaluation/helmet/patched_eval.py) — thin
  wrapper that applies our compat patches (see below) before invoking HELMET's
  `eval.py`. Auto-used by `helmet_shard.py`.
- [`aggregate_results.py`](../../../jetlm/evaluation/helmet/aggregate_results.py)
  — walks the per-model output dir, reads every `*.json.score`, groups scores
  by (category, length), and writes `result.json` at the top level with
  category/length/overall averages plus a `per_task` array. Also prints a
  pretty table. Safe to re-run at any time.
- [`patches/`](../../../jetlm/evaluation/helmet/patches/) — compat patches
  injected by `patched_eval.py`.

## Output layout

```
results/helmet/<model_name>/
├── result.json                                     ← aggregated summary (what you want)
├── <dataset>_<tag>_..._in<L>_size<N>_...json       ← full per-task trace (inputs + outputs)
├── <dataset>_<tag>_..._in<L>_size<N>_...json.score ← per-task averaged metrics
└── logs/
    └── <category>_<dataset>_in<L>.log              ← subprocess stderr
```

`result.json` structure:
```json
{
  "model": "Qwen3-1.7B-Base-jetlong",
  "tag": "v1",
  "num_tasks": 40,
  "overall_avg": 28.5,
  "category_avg":    {"rag": 31.0, "longqa": 24.1, "summ": 6.7},
  "length_avg":      {"8192": 31.2, "131072": 26.8, ...},
  "category_length": {"rag": {"8192": 32.5, "131072": 29.6, "avg": 31.0}, ...},
  "judge_pending_categories": ["longqa", "summ"],
  "per_task": [ {category, dataset, length, metric, score, file, ...}, ... ]
}
```
`judge_pending_categories` lists categories whose canonical metric is a
GPT-4o judge (NarrativeQA, infbench_sum, multi_lexsum). Those tasks currently
report `rougeL_f1` as a surrogate; run `tmp/helmet/repo/scripts/eval_gpt4_*.py`
(or our `run_gpt4o_judges.sh` wrapper) to produce paper-faithful numbers, then
re-run `aggregate_results.py` — it will automatically pick up
`*-gpt4eval_o.json` outputs.

## Usage

```bash
# default: 5 lengths × 5 categories × ~3 datasets = ~70 tasks across 8 GPUs
bash scripts/eval/helmet/run_helmet.sh Qwen3-1.7B-Base-selfext

# a subset
bash scripts/eval/helmet/run_helmet.sh Qwen3-1.7B-Base-selfext "recall rag" "8192 131072"

# overrides via env
GPUS=0,1,2,3 MAX_TEST_SAMPLES=50 TAG=v2 \
  bash scripts/eval/helmet/run_helmet.sh Qwen3-1.7B-Base-jetlong
```

Direct driver invocation for finer control:

```bash
python jetlm/evaluation/helmet/helmet_shard.py \
    --helmet_dir tmp/helmet/repo \
    --model_name_or_path model_cache/Qwen3-1.7B-Base-selfext \
    --configs recall rag cite rerank longqa \
    --lengths 8192 16384 32768 65536 131072 \
    --gpus 0,1,2,3,4,5,6,7 \
    --output_dir results/helmet/Qwen3-1.7B-Base-selfext \
    --tag v1 \
    --max_test_samples 100 \
    --dry_run   # preview without launching
```

## How it works (short)
- `trust_remote_code=True` is already in HELMET's `HFModel`
  ([model_utils.py:893–910](../../../tmp/helmet/repo/model_utils.py#L893-L910)),
  so jetlm variants with a custom `modeling_qwen3_*.py` and an `auto_map` in
  `config.json` load unchanged.
- `CUDA_VISIBLE_DEVICES=<gpu>` pins each subprocess to one physical GPU.
  HELMET's `device_map="auto"` then places the whole model on that GPU.
- `torch.compile` is disabled by default (`--no_torch_compile`) — safer with
  trust-remote-code modeling files whose graphs may not be compile-clean.
- Results land in `results/helmet/<model_name>/` as one `*.json` +
  `*.json.score` per `(dataset, length)`. Aggregate with
  [`scripts/collect_results.py`](../../../tmp/helmet/repo/scripts/collect_results.py)
  from the HELMET repo, or with our `aggregate_results.py`.

## Compat patches

HELMET's upstream `data.py` relies on `load_dataset(..., trust_remote_code=True)`
for a handful of datasets whose HF repos still ship loading scripts.
`datasets>=4` removed that path, so in the `jtl` env (datasets 4.6.1) these
would break.

[`patched_eval.py`](../../../jetlm/evaluation/helmet/patched_eval.py) is a thin
wrapper that imports
[`patches/`](../../../jetlm/evaluation/helmet/patches/) before invoking
`eval.py`. `helmet_shard.py` automatically uses it. Each compat patch
monkey-patches HELMET's `data.load_dataset` on a per-dataset basis — HELMET's
own code is never touched.

Currently patched:
- **`allenai/multi_lexsum`** ([patches/multi_lexsum_patch.py](../../../jetlm/evaluation/helmet/patches/multi_lexsum_patch.py))
  — fetches the raw `releases/v20230518/{sources,train,dev}.json` files from
  the HF hub directly, joins `case_documents` IDs against the sources map, and
  materializes a `DatasetDict` matching the upstream schema exactly. The
  assembled dataset is cached to disk under `~/.cache/helmet_patches/`
  (override via `HELMET_PATCH_CACHE`). First invocation: ~3.5 min + 2.2 GB
  download; subsequent: ~1s load + whatever HELMET's own
  `filter_length(num_proc=32)` startup costs.

Not patched (Summ works without these; ICL is disabled by default):
- `CogComp/trec`, `PolyAI/banking77`, `xingkunliuxtracta/nlu_evaluation_data`
  (ICL) — would need similar patches. Add them as peer modules in `patches/`.

If you want the ICL category, either downgrade `datasets` to `<4` in a
separate env, or write analogous loading-script-bypass patches (the pattern
in `multi_lexsum_patch.py` is a template).

## Dependencies

Beyond what jetlm already needs:
```
pip install sentencepiece pytrec_eval
```
`sentencepiece` is required by several HF tokenizers; `pytrec_eval` is used
for NDCG@10 on MS MARCO reranking.
