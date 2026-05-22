#!/bin/bash
# One-shot HELMET setup. Idempotent — re-runs only do work that hasn't been
# done yet. Activate the python env first, then run.
#
# Steps:
#   1. `pip install` HELMET-specific deps (sentencepiece, pytrec_eval, etc.)
#   2. Clone HELMET upstream under tmp/helmet/repo
#   3. Download + extract data.tar.gz (~11 GB → ~12 GB extracted)
#   4. Pre-fetch HF datasets used by LongQA + Summ (narrativeqa, infbench)
#      into the HF cache so compute nodes without internet can still run
#   5. Pre-build the multi_lexsum compat cache (~3.5 min, one-time)
#   6. Smoke-test: import HELMET modules + our patches
#
# Override via env vars:
HELMET_DIR="tmp/helmet/repo"       # repo-relative
#   SKIP_MULTI_LEXSUM_BUILD=1        (skip step 5 if you won't run Summ)
#   FORCE_MULTI_LEXSUM_BUILD=1       (force rebuild: wipes ~/.cache/helmet_patches/… and re-downloads)
#   HF_TOKEN=<token>                 (required for some gated HF datasets)
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

HELMET_DIR="${HELMET_DIR:-tmp/helmet/repo}"
HELMET_REPO_URL="https://github.com/princeton-nlp/HELMET.git"
DATA_URL="https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz"
SKIP_MULTI_LEXSUM_BUILD="${SKIP_MULTI_LEXSUM_BUILD:-0}"
FORCE_MULTI_LEXSUM_BUILD="${FORCE_MULTI_LEXSUM_BUILD:-0}"

log()  { echo -e "[helmet-setup] $*"; }
warn() { echo -e "[helmet-setup] \033[33mWARN:\033[0m $*"; }
die()  { echo -e "[helmet-setup] \033[31mERROR:\033[0m $*" >&2; exit 1; }

log "python: $(which python)"
log "torch/transformers/datasets: $(python -c 'import torch,transformers,datasets; print(torch.__version__, transformers.__version__, datasets.__version__)')"

# ---------- 1. pip install HELMET-specific deps ----------
log "step 1/6  install HELMET python deps"
missing=()
# pytrec_eval, sentencepiece — plain pip installs.
# openai/tiktoken — only needed for the GPT-4 judge scripts (eval_gpt4_*), but
# cheap to pre-install so run_gpt4o_judges.sh doesn't fail on first use.
for pkg_import_pair in sentencepiece:sentencepiece pytrec_eval:pytrec_eval \
                       openai:openai tiktoken:tiktoken; do
    import_name="${pkg_import_pair%%:*}"
    install_name="${pkg_import_pair##*:}"
    python -c "import ${import_name}" 2>/dev/null || missing+=("${install_name}")
done
# `pkg_resources` is used by HELMET (model_utils.py:881). setuptools>=80 no
# longer ships pkg_resources, so having setuptools installed isn't enough —
# we need a version that still bundles it.
if ! python -c "import pkg_resources" 2>/dev/null; then
    log "        pkg_resources missing — pinning setuptools<80 (post-80 removed it)"
    missing+=("setuptools<80")
fi
if ((${#missing[@]})); then
    log "        installing: ${missing[*]}"
    pip install --force-reinstall "${missing[@]}"
else
    log "        already satisfied"
fi

# ---------- 2. clone HELMET upstream ----------
log "step 2/6  clone HELMET upstream → ${HELMET_DIR}"
if [[ -d "${HELMET_DIR}/.git" ]]; then
    log "        already cloned (skipping)"
else
    mkdir -p "$(dirname "${HELMET_DIR}")"
    git clone --depth 1 "${HELMET_REPO_URL}" "${HELMET_DIR}"
fi
[[ -f "${HELMET_DIR}/eval.py" ]] || die "${HELMET_DIR}/eval.py missing after clone"

# ---------- 4. download + extract HELMET data ----------
DATA_TAR="${HELMET_DIR}/data.tar.gz"
DATA_DIR="${HELMET_DIR}/data"
EXPECTED_SUBDIRS=(alce infbench json_kv kilt msmarco multi_lexsum ruler)

log "step 3/6  fetch HELMET data"
need_download=1
if [[ -f "${DATA_TAR}" ]]; then
    # Compare against remote size to detect partial downloads. HF redirects twice
    # (302 → 200 on CDN), so we pick the final Content-Length — `grep -i` matches
    # both `content-length` (302) and `Content-Length` (200); `tail -1` grabs
    # the post-redirect one.
    remote_size=$(curl -sIL "${DATA_URL}" | grep -iE "^content-length:" | tr -d '\r' | awk '{print $2}' | tail -1)
    local_size=$(stat -c %s "${DATA_TAR}")
    if [[ -n "${remote_size}" && "${local_size}" == "${remote_size}" ]]; then
        log "        data.tar.gz already present (${local_size} bytes, matches remote)"
        need_download=0
    else
        warn "data.tar.gz size mismatch (local=${local_size} remote=${remote_size:-unknown}); wget -c will resume"
    fi
fi
if (( need_download )); then
    log "        downloading ~11 GB; this may take a while"
    wget -c --show-progress --progress=dot:giga "${DATA_URL}" -O "${DATA_TAR}"
fi

need_extract=0
for sd in "${EXPECTED_SUBDIRS[@]}"; do
    [[ -d "${DATA_DIR}/${sd}" ]] || { need_extract=1; break; }
done
if (( need_extract )); then
    log "        extracting (expect ~12 GB on disk)"
    ( cd "${HELMET_DIR}" && tar -xzf data.tar.gz )
else
    log "        data/ already extracted with all 7 subdirs"
fi
for sd in "${EXPECTED_SUBDIRS[@]}"; do
    [[ -d "${DATA_DIR}/${sd}" ]] || die "missing data/${sd} after extract"
done

# ---------- 5. pre-fetch HF datasets for LongQA + Summ-infbench ----------
log "step 4/6  pre-fetch HF datasets (cache on login node for airgapped compute)"
python - <<'PY'
import os, sys
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
from datasets import load_dataset, Value, Sequence, Features

# narrativeqa — LongQA
try:
    load_dataset("narrativeqa")
    print("  [ok]   narrativeqa")
except Exception as e:
    print(f"  [warn] narrativeqa: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)

# infbench — LongQA (qa_eng, choice_eng) + Summ (sum_eng). Use HELMET's schema
# override (data.py:562-565) so the `code_debug` split — which can't be inferred
# cleanly — doesn't blow up the fetch.
ft = Features({
    "id": Value("int64"),
    "context": Value("string"),
    "input": Value("string"),
    "answer": Sequence(Value("string")),
    "options": Sequence(Value("string")),
})
try:
    dsd = load_dataset("xinrongzhang2022/infinitebench", features=ft)
    for s in ("longbook_qa_eng", "longbook_choice_eng", "longbook_sum_eng"):
        print(f"  [ok]   infinitebench/{s}  n={len(dsd[s])}")
except Exception as e:
    print(f"  [warn] infinitebench: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
PY

# ---------- 6. pre-build multi_lexsum compat cache ----------
if [[ "${SKIP_MULTI_LEXSUM_BUILD}" == "1" ]]; then
    log "step 5/6  skipping multi_lexsum build (SKIP_MULTI_LEXSUM_BUILD=1)"
else
    log "step 5/6  build multi_lexsum compat cache (~3.5 min one-time)"
    FORCE_MULTI_LEXSUM_BUILD="${FORCE_MULTI_LEXSUM_BUILD}" python - <<PY
import os, sys, shutil, time
sys.path.insert(0, "${PROJECT_ROOT}/jetlm/evaluation/helmet")
sys.path.insert(0, "${PROJECT_ROOT}/${HELMET_DIR}")
from patches import multi_lexsum_patch as m

cache = m._cache_dir(m._RELEASE_DEFAULT)
force = os.environ.get("FORCE_MULTI_LEXSUM_BUILD", "0") == "1"
if force and os.path.isdir(cache):
    print(f"  [force] clearing existing cache: {cache}")
    shutil.rmtree(cache)

if (not force
        and os.path.isdir(cache)
        and os.path.isfile(os.path.join(cache, "dataset_dict.json"))):
    print(f"  [ok] cache already exists: {cache}")
else:
    t0 = time.time()
    m._build_datasetdict()  # downloads 2.2 GB sources + builds + save_to_disk
    print(f"  [ok] built cache at {cache} in {time.time()-t0:.1f}s")
PY
fi

# ---------- 6. smoke test ----------
log "step 6/6  smoke test"
python - <<PY
import sys
sys.path.insert(0, "${PROJECT_ROOT}/jetlm/evaluation/helmet")
sys.path.insert(0, "${PROJECT_ROOT}/${HELMET_DIR}")
import patches, data, model_utils, eval  # noqa: F401
applied = patches.apply_all()
print(f"  patches applied: {applied}")
print(f"  HELMET modules import cleanly  ✓")
PY

log "\ndone. Run: bash scripts/eval/helmet/run_helmet.sh <model_name> [categories] [lengths]"
log "      e.g. bash scripts/eval/helmet/run_helmet.sh Qwen3-1.7B-Base-jetlong \"rag longqa summ\" \"8192 131072\""
