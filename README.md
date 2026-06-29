



# Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE
<div align="center">
  <a href="https://arxiv.org/abs/2604.13010"><img src="https://img.shields.io/static/v1?label=arXiv&message=Jet-Long&color=red&logo=arxiv"></a> &ensp;
</div>

## 1. 💡 Introduction

https://github.com/user-attachments/assets/52d59246-f648-4fda-91b5-aa007cc2c30a

### News

- \[2026.05.21\] We released Jet-Long.

### Contents

+ [2. Installation](#2-installation)
+ [3. Quick Evaluation (Qwen3-1.7B-Base)](#3-quick-evaluation-qwen3-17b-base)
+ [4. Shipped methods](#4-shipped-methods)
+ [5. How a method is composed on top of a Qwen3 Base](#5-how-a-method-is-composed-on-top-of-a-qwen3-base)
+ [6. Hardware Requirements](#6-hardware-requirements)
+ [7. Project Structure](#7-project-structure)
+ [8. Acknowledgements](#8-acknowledgements)
+ [9. Contact](#9-contact)
+ [10. Contributing](#10-contributing)
+ [11. License](#11-license)
+ [12. BibTeX](#12-bibtex)

## 2. Installation

```bash
git clone https://github.com/jet-ai-projects/jet-long.git
cd jet-long
```

There are two installation profiles, both on Python 3.12 + torch 2.9.1+cu130:

- **Default (FlashAttention-2)** — Steps 1–3 below. Supports every method
  (`base`, `yarn`, `dntk`, `selfext`, `dca`, `jetlong`, `jetlong_freq`).
- **Fused CuTe kernel (optional, SM90/H100 only)** — the *Fused CuTe kernel
  environment* section below. A separate conda env for the `jetlong_fused` method.

The only out-of-band dep is the FlashAttention build, which must be linked
against the same CUDA toolkit as torch.

### Step 1: Create the conda env

```bash
conda create -n jtl python=3.12 -y
conda activate jtl
```

### Step 2: Install jetlm + Python deps

`pip install -e .` reads [`pyproject.toml`](pyproject.toml) and pulls torch 2.9.1,
transformers 5.3.0, datasets 4.6.1, accelerate, wandb, nvidia-ml-py, lm_eval
(jetlm-ai fork pinned to the `jetai` tag), and the RULER/PG-19 metric deps
(jieba, rouge, etc.):

```bash
pip install -e .
```

If your system CUDA major version doesn't match the torch wheel (the default
pip wheel ships cu12, but H100 / Blackwell hosts often expose CUDA 13), force
a matching torch wheel before building flash-attn:

```bash
# Example: cu13 host
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
```

### Step 3: Install flash-attention 2.8.3

Install or build flash-attn `2.8.3` against your torch + CUDA stack — either a
matching pre-built wheel from the
[Dao-AILab releases page](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3),
or a from-source build of the `v2.8.3` tag. If you build from source, set
`FLASH_ATTENTION_FORCE_BUILD=TRUE` so `pip` actually compiles instead of
falling back to a stale pre-built wheel.

> **Note:** `flash-attn==2.8.3` is required. The bleeding-edge `2.8.4` HEAD
> build has a small-q NaN regression that silently corrupts decode. See
> [`requirements.txt`](requirements.txt) for the pin.

### Fused CuTe kernel environment (optional — SM90/H100 only)

The `jetlong_fused` method runs JetLong through a fused FlashAttention-4 / CuTe
SM90 kernel ([`jetlm/kernels/`](jetlm/kernels)) instead of the FA2 A+B−C merge.
It needs a **separate conda env** with the CuTe-DSL runtime plus `flash_attn_4`
(which provides `flash_attn.cute`), on the same torch 2.9.1+cu130:

```bash
conda create -n jtl-fused python=3.12 -y
conda activate jtl-fused
pip install -e ".[fused]"     # base deps + nvidia-cutlass-dsl + quack-kernels + cuda-python + ...
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
# flash_attn_4 ships flash_attn.cute (beta) — install 4.0.0b10 (the block-sparse
# prefill API the kernel relies on changed in b11):
pip install flash_attn_4==4.0.0b10
```

> **⚠️ This environment runs only `jetlong_fused`.** `flash_attn_4` provides
> `flash_attn.cute` but **not** the FA2 forward kernels, so every other method —
> including the non-fused `jetlong`, `yarn`, `dntk`, `selfext`, `dca`,
> `jetlong_freq` — will **not** run here; they require the default `flash-attn`
> 2.8.3 environment above. Hard requirements: NVIDIA **SM90/H100** + **CUDA 13**.
> `jetlong_fused` is **long-context-only** (decode context must exceed `w_0`);
> use plain `jetlong` for short contexts.

Validate the kernel against reference attention with `pytest
tests/test_fa4_cute_kernels.py` and `python
scripts/throughput/check_cute_jetlong_parity.py --config-path
model_cache/Qwen3-1.7B-Base-jetlong_fused` (run
`scripts/model/setup_model_variants.sh` first so the modeling file is symlinked
adjacent to the config). Load `jetlong_fused` models with
`attn_implementation="sdpa"` (the fused kernel is invoked explicitly on the
extended path; SDPA covers the within-32K-window path).

### Benchmarking fused throughput

Measure prefill throughput and CUDA-graphed decode-kernel latency. The
`jetlong_fused` commands run in the fused env; the FA2 baselines run in the default
env (the two environments are mutually exclusive), so each comparison is two
processes:

```bash
# Prefill throughput (full-forward tok/s) — fused vs FA2 baseline, across lengths:
python scripts/throughput/bench_prefill_throughput.py --model_path model_cache/Qwen3-8B-Base-jetlong_fused --attn sdpa            # fused env
python scripts/throughput/bench_prefill_throughput.py --model_path model_cache/Qwen3-8B-Base --attn flash_attention_2            # default env

# Generation throughput (decode-kernel latency, CUDA-graphed) — fused vs single-FA2-decode baseline:
python scripts/throughput/bench_cute_decode_sm90_jetlong.py --cuda-graph --nheads 32 --nheads-kv 8 --head-dim 128                # fused env
python scripts/throughput/bench_fa2_decode_baseline.py --cuda-graph --nheads 32 --nheads-kv 8 --head-dim 128                     # default env
```

Prefill should get ~1.24–1.31× FA2 speedup past 32K.

> **Generation throughput shows no loss vs FA2 at the attention-kernel level** — the
> fused decode measures at parity with or faster than a single FA2 decode (6–15%
> faster across 32K–131K on H100), and attention is a very small portion of the per-token cost, so end-to-end is ≈1.0× FA2 (note that measuring *full* end-to-end generation throughput directly requires manually capturing a full CUDA graph of the decode loop).

## 3. Quick Evaluation (Qwen3-1.7B-Base)

Below is the complete from-scratch evaluation of the Qwen3-1.7B + jetlong row on all three benchmarks. All commands are run from the repo root with the python env activated.

### Step 1: Materialize the model_cache

```bash
bash scripts/model/setup_model_variants.sh
```

### Step 2: Fetch eval data

RULER-500 and PG-19-subsample are hosted on the HF Hub and downloaded into
`processed_data/` via a single dispatcher:

| Dataset | HF repo | Local path |
|---|---|---|
| `ruler` | [`jet-ai/RULER-500`](https://huggingface.co/datasets/jet-ai/RULER-500) | `processed_data/RULER-data/` |
| `pg19sub` | [`jet-ai/pg19-subsample`](https://huggingface.co/datasets/jet-ai/pg19-subsample) | `processed_data/PG-19-subsample/` |

```bash
bash scripts/data/download_data.sh
```

### Step 3: Run RULER

```bash
# One launcher per (size, method) under scripts/eval/ruler/
bash scripts/eval/ruler/qwen1d7B_base_jetlong.sh
```

### Step 4: Run PG-19 perplexity (anchored growing window, 1k → 131k)

```bash
MODEL_PATH=model_cache/Qwen3-1.7B-Base-jetlong bash scripts/eval/ppl/pg19_ppl_local.sh
```

### Step 5: Run HELMET-RAG (8 GPUs, 4 RAG sub-tasks × 5 lengths)

First clone the upstream HELMET repo + data:

```bash
bash scripts/eval/helmet/set_up_helmet.sh
bash scripts/eval/helmet/run_helmet.sh Qwen3-1.7B-Base-jetlong
```

### Evaluating `jetlong_fused`

`jetlong_fused` reuses the `model_cache` and data from Steps 1–2, but runs RULER
from the fused conda env (`jtl-fused`, see [Section 2](#2-installation)) because
the kernel needs `flash_attn.cute`:

```bash
conda activate jtl-fused
bash scripts/eval/ruler/qwen1d7B_base_jetlong_fused.sh
```

This mirrors Step 3 with three fused-specific settings: it loads with
`attn_implementation="sdpa"` (auto-detected from the architecture; the CuTe kernel
takes over once the context passes the 32K window), runs at `--eval_batch_size 8`
so the lengths that exercise the kernel (≥32K, where it rejects left-padded
batches) fall to batch 1, and sets `--prefill_chunk_size 16384` to keep the
131072 prefill within one H100. RULER scores match the non-fused `jetlong`
method — identical at ≤32K (both reduce to plain causal attention there) and
lossless at the 64K/128K lengths the kernel handles.

## 4. Shipped methods

| Method | Mechanism | Config knobs |
|---|---|---|
| `yarn` | YaRN NTK-by-parts via stock HF `rope_scaling` | `factor=4`, `original_max_position_embeddings=32768` |
| `dntk` | Dynamic NTK via stock HF `rope_scaling` | `factor=4`, dynamic |
| `selfext` | SelfExtend bi-level attention (Jin et al., ICML 2024) | `window_size=1024`, `group_size=8` |
| `dca` | Dual Chunk Attention (An et al., ICML 2024) | `chunk_size=20480`, `local_window=4096` |
| `jetlong` | JetLong — Bifocal Dynamic Self-Extend (`G=⌈L/w⌉`, single-temperature 3-call A+B−C merge) | `w=32768`, `w_0=2048` |
| `jetlong_freq` | JetLongFreq — frequency-domain variant: Dynamic Squeeze + YaRN softmax temperature | `w=32768`, `w_0=2048` |
| `jetlong_fused` | JetLong via a fused FlashAttention-4 / CuTe SM90 kernel (optional; SM90/H100 + CUDA 13 only — see the fused install section) | `w=32768`, `w_0=2048` |

Both `jetlong` and `jetlong_freq` share a 3-call FlashAttention prefill (near base
RoPE + distant correction + windowed correction, merged via stabilized fp32 LSE);
they differ in how the distant phase is computed (integer position grouping vs
YaRN frequency interpolation).
See [`jetlm/modeling/jetlong/modeling_qwen3_jetlong.py`](jetlm/modeling/jetlong/modeling_qwen3_jetlong.py)
and [`jetlm/modeling/jetlong_freq/modeling_qwen3_jetlong_freq.py`](jetlm/modeling/jetlong_freq/modeling_qwen3_jetlong_freq.py)
for details.

## 5. How a method is composed on top of a Qwen3 Base

Every variant is a standard HuggingFace model directory at
`model_cache/<variant>/`. The trick is that we never copy a 4 B / 8 B
checkpoint: each variant is *materialized as a directory of symlinks* by
`scripts/model/setup_model_variants.sh`, which sources three things:

| File in `model_cache/<variant>/` | Symlink target | What it changes |
|---|---|---|
| `model-*.safetensors`, `*.safetensors.index.json`, `tokenizer*`, `merges.txt`, `vocab.json`, `generation_config.json`, `LICENSE`, `README.md` | `model_cache/<base>/` (the bare Qwen3 checkpoint from HF) | nothing — these are shared verbatim across every variant of a given size |
| `config.json` | `model_configs/<variant>/config.json` | per-(size, method) overrides: method-specific knobs, `architectures`, `auto_map`, optional `rope_scaling` |
| `modeling_qwen3_<method>.py` (only when the method ships custom code) | `jetlm/modeling/<method>/modeling_qwen3_<method>.py` | one canonical modeling implementation per method, shared across all sizes that use it |

The split keeps the per-method implementation in one place
(`jetlm/modeling/<method>/`) while letting each (size, method) pair carry its
own small `config.json` for the architecture dimensions and method knobs that
differ per size.

### Two flavors of `config.json`

**Config-only methods** (`yarn`, `dntk`) use HF's stock `rope_scaling` field
and the unmodified `Qwen3ForCausalLM` class — no custom modeling code:
```jsonc
{
  "architectures": ["Qwen3ForCausalLM"],
  "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                   "original_max_position_embeddings": 32768},
  "rope_theta": 1000000, "max_position_embeddings": 131072,
  // ...base dims unchanged...
}
```

**Methods with custom modeling code** (`selfext`, `dca`, `jetlong`,
`jetlong_freq`) add `auto_map` so HF's `from_pretrained(...,
trust_remote_code=True)` picks up the sibling modeling file at load time, plus
a method-specific knob block:
```jsonc
{
  "architectures": ["Qwen3JetLongForCausalLM"],
  "auto_map": {
    "AutoModelForCausalLM": "modeling_qwen3_jetlong.Qwen3JetLongForCausalLM"
  },
  "jetlong": {"w": 32768, "w_0": 2048, "yarn_alpha": 1, "yarn_beta": 32},
  "rope_scaling": null,
  // ...base dims unchanged...
}
```

The string `"modeling_qwen3_jetlong.Qwen3JetLongForCausalLM"` is a bare file
name plus class name — not a Python import path. HF expects the file
*adjacent to* `config.json` at load time, which is exactly what the modeling
symlink provides.

### What a materialized variant looks like

After `bash scripts/model/setup_model_variants.sh`, a variant like
`Qwen3-1.7B-Base-jetlong` becomes a flat directory of symlinks:
```
model_cache/Qwen3-1.7B-Base-jetlong/
├── config.json                → ../../model_configs/Qwen3-1.7B-Base-jetlong/config.json
├── modeling_qwen3_jetlong.py  → ../../jetlm/modeling/jetlong/modeling_qwen3_jetlong.py
├── model.safetensors          → ../Qwen3-1.7B-Base/model.safetensors
├── tokenizer.json             → ../Qwen3-1.7B-Base/tokenizer.json
└── ... (other tokenizer + generation_config + LICENSE files)
```

From any caller — our eval scripts, the upstream HELMET driver, or a plain
Python REPL — loading is just:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "model_cache/Qwen3-1.7B-Base-jetlong",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained("model_cache/Qwen3-1.7B-Base-jetlong")
```
HF reads `config.json`, sees `auto_map`, copies the sibling
`modeling_qwen3_jetlong.py` into its trust-remote-code module cache, imports
`Qwen3JetLongForCausalLM`, and loads the base weights into it. Because we
only overrode rotary embedding + attention forward and kept every other
module verbatim, the pretrained safetensors load cleanly with no key
mismatches.

### Inside the modeling subpackage

Each `jetlm/modeling/<method>/` subpackage subclasses HF's vanilla Qwen3
modules and overrides only what the method needs. For example
[`jetlm/modeling/jetlong/modeling_qwen3_jetlong.py`](jetlm/modeling/jetlong/modeling_qwen3_jetlong.py)
exports 5 classes (`Qwen3JetLongAttention`, `Qwen3JetLongDecoderLayer`,
`Qwen3JetLongForCausalLM`, `Qwen3JetLongModel`,
`Qwen3JetLongRotaryEmbedding`) — every other building block (`Qwen3MLP`,
`Qwen3RMSNorm`, etc.) is imported unchanged from
`transformers.models.qwen3.modeling_qwen3`. That's why every method is
*training-free*: the weight shapes are byte-identical to the upstream base
checkpoint.

## 6. Hardware Requirements

Jet-Long is inference-only (no fine-tuning), and we run all evaluation on
NVIDIA H100 GPUs. The default experiments in this repo are launched on a
single 8×H100 node for fast iteration; the launchers under `scripts/eval/`
discover the visible GPU count via `nvidia-smi` (or `CUDA_VISIBLE_DEVICES`)
and shard across whatever is present, so the same scripts work on smaller
node counts at the cost of wallclock time.

## 7. Project Structure

```
jetlm/
  constant.py               # repo-wide version-string + W&B result-log suffix
  evaluation/
    meta_eval.py            # entry for RULER + lm-eval-harness configs
    ruler/                  # RULER evaluator (custom)
    ppl/                    # PG-19 anchored growing-window PPL
    helmet/                 # HELMET sharded driver + result aggregator + compat patches
    lm_eval_harness/        # wrapper around lm-eval harness
    configs/                # per-task YAMLs (ruler.yaml + pg19_ppl.yaml)
  kernels/                  # fused FA4 / CuTe SM90 kernels for jetlong_fused (+ csrc/ JIT-built merge)
  modeling/
    wrapper.py              # thin HFModel generate wrapper
    {dca,selfext,jetlong,jetlong_freq,jetlong_fused}/  # one subpackage per method with custom modeling
  utils/                    # distributed, atomic I/O, complete-markers
model_configs/              # per-(size, method) config.json (no modeling code)
scripts/
  data/                     # PG-19 + RULER data prep (download_data.{py,sh}, upload_*_hf.py, subsample/generate)
  model/                    # variant setup + base-model download
  eval/
    ruler/                  # per-(size, method) RULER launcher
    ppl/                    # PG-19 PPL launchers
    helmet/                 # HELMET launcher, GPT-4 judges, set_up_helmet.sh
  throughput/               # fused-vs-FA2 throughput benches + parity check (prefill, decode-kernel latency, CuTe-vs-ref parity)
tests/                      # FA4 CuTe kernel parity tests (test_fa4_cute_kernels.py)
```

## 8. Acknowledgements

We thank the authors of [RULER](https://github.com/NVIDIA/RULER),
[PG-19](https://github.com/google-deepmind/pg19), and
[HELMET](https://github.com/princeton-nlp/HELMET) for the long-context
benchmarks Jet-Long is evaluated on, and the
[Qwen3](https://github.com/QwenLM/Qwen3) team for the open-weights base
models we extend.

## 9. Contact

+ [Han Cai](https://han-cai.github.io/)
+ [Haozhan Tang](https://haozhantang.com/)
+ [Song Han](https://hanlab.mit.edu/songhan)

## 10. Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and
Developer Certificate of Origin (DCO) sign-off requirements.

## 11. License

This codebase is released under the [Apache License 2.0](LICENSE).
Third-party dependency notices are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 12. BibTeX

This is a place holder, we will add the citation in bibtex after arxiv publish.
