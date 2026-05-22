#!/bin/bash
# Run HELMET's GPT-4o judges over LongQA + Summ generations, then re-aggregate
# result.json so the paper-faithful gpt-4-score / gpt-4-f1 metrics replace the
# rougeL_f1 surrogates.
#
# No model re-inference happens here — the judges read the full generations
# already written to results/helmet/<model>/<dataset>_..._<seed>.json and call
# GPT-4o via the OpenAI batch API. Writes *-gpt4eval_o.json siblings and
# refreshes each model's result.json.
#
# Usage:
#   bash scripts/eval/helmet/run_gpt4o_judges.sh                     # auto-discover all models under results/helmet/
#   bash scripts/eval/helmet/run_gpt4o_judges.sh Qwen3-8B-Base ...   # explicit subset
#   SKIP_LONGQA=1 bash scripts/eval/helmet/run_gpt4o_judges.sh       # only summ judge
#   SKIP_SUMM=1   bash scripts/eval/helmet/run_gpt4o_judges.sh       # only longqa judge
#   DRY_RUN=1     bash scripts/eval/helmet/run_gpt4o_judges.sh       # preview without calling the API
#   TAG=v2        bash scripts/eval/helmet/run_gpt4o_judges.sh       # different output tag
#
# Requirements:
#   - OPENAI_API_KEY in the environment
#   - results/helmet/<model>/ populated by a previous bash scripts/eval/helmet/run_helmet.sh run
#   - python env already activated (with jetlm installed)
#
# Cost: ~$20–25 per model on the GPT-4o batch API.
# Latency: batch API turnaround typically 1–4h, occasionally up to 24h.
#
# Idempotent: each judge skips tasks whose *-gpt4eval_o.json sibling already
# exists, and aggregate_results.py is cheap to re-run at any time.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

HELMET_DIR="${HELMET_DIR:-tmp/helmet/repo}"
RESULTS_ROOT="${RESULTS_ROOT:-results/helmet}"
TAG="${TAG:-v1}"
SKIP_LONGQA="${SKIP_LONGQA:-0}"
SKIP_SUMM="${SKIP_SUMM:-0}"
DRY_RUN="${DRY_RUN:-0}"

log()  { echo -e "[gpt4o-judge] $*"; }
warn() { echo -e "[gpt4o-judge] \033[33mWARN:\033[0m $*"; }
die()  { echo -e "[gpt4o-judge] \033[31mERROR:\033[0m $*" >&2; exit 1; }

# Tee everything (stdout + stderr) into a timestamped log under logs/helmet/judge/.
# Useful because the batch API can take hours and the human invoker often
# detaches via tmux/nohup.
JUDGE_LOG_DIR="${JUDGE_LOG_DIR:-logs/helmet/judge}"
mkdir -p "${JUDGE_LOG_DIR}"
JUDGE_LOG_FILE="${JUDGE_LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${JUDGE_LOG_FILE}") 2>&1
log "logging to ${JUDGE_LOG_FILE}"

[[ -d "${HELMET_DIR}" ]] || die "HELMET_DIR not found: ${HELMET_DIR} (run scripts/eval/helmet/set_up_helmet.sh first)"
[[ -d "${RESULTS_ROOT}" ]] || die "RESULTS_ROOT not found: ${RESULTS_ROOT} (run scripts/eval/helmet/run_helmet.sh first)"
[[ -n "${OPENAI_API_KEY:-}" ]] || die "OPENAI_API_KEY not set"

# ---------- pick models ----------
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
    EXPLICIT=1
else
    mapfile -t MODELS < <(find "${RESULTS_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    EXPLICIT=0
fi
[[ ${#MODELS[@]} -gt 0 ]] || die "no model directories found under ${RESULTS_ROOT}"

# ---------- filter to judgeable models ----------
# A model is judgeable only if its results dir contains at least one generation
# file the judges can score. Skip + warn (or hard-fail on EXPLICIT mode).
JUDGE_PREFIXES=(narrativeqa infbench_sum_eng multi_lexsum)

has_judgeable_files() {
    local dir="$1"
    local prefix matches=()
    shopt -s nullglob
    for prefix in "${JUDGE_PREFIXES[@]}"; do
        for f in "${dir}/${prefix}_"*"_${TAG}_"*.json; do
            [[ "$f" != *"-gpt4eval_o.json" ]] && matches+=("$f")
        done
    done
    shopt -u nullglob
    (( ${#matches[@]} > 0 ))
}

present=()
missing_dir=()
empty_dir=()
for model in "${MODELS[@]}"; do
    src="${RESULTS_ROOT}/${model}"
    if [[ ! -d "${src}" ]]; then
        missing_dir+=("${model}"); continue
    fi
    if ! has_judgeable_files "${src}"; then
        empty_dir+=("${model}"); continue
    fi
    present+=("${model}")
done

(( ${#missing_dir[@]} )) && warn "results dir missing for: ${missing_dir[*]}"
(( ${#empty_dir[@]}   )) && warn "no judgeable files (narrativeqa/infbench_sum/multi_lexsum, tag=${TAG}) for: ${empty_dir[*]}"
if (( ! ${#present[@]} )); then
    log "nothing judgeable for: ${MODELS[*]}"
    (( EXPLICIT )) && exit 1
    exit 0
fi

log "tag=${TAG}  helmet_dir=${HELMET_DIR}  dry_run=${DRY_RUN}"
log "models to judge (${#present[@]}):"
for m in "${present[@]}"; do echo "    - ${m}"; done

# ---------- symlink results/helmet/<m> → tmp/helmet/repo/output/<m> ----------
# The upstream judges glob output/<model>/*.json (hardcoded in HELMET); we
# symlink instead of copying so the trees stay in sync without bytes moving.
log "wiring symlinks into ${HELMET_DIR}/output/"
mkdir -p "${HELMET_DIR}/output"
for model in "${present[@]}"; do
    src="$(realpath "${RESULTS_ROOT}/${model}")"
    dst="${HELMET_DIR}/output/${model}"
    if [[ -L "${dst}" ]]; then
        existing="$(readlink -f "${dst}")"
        [[ "${existing}" == "${src}" ]] || { rm "${dst}"; ln -s "${src}" "${dst}"; }
    elif [[ -e "${dst}" ]]; then
        warn "${dst} exists and is not a symlink; skipping (manual fix needed)"
    else
        ln -s "${src}" "${dst}"
    fi
done

# ---------- run the judges ----------
pushd "${HELMET_DIR}" > /dev/null
START=$SECONDS

if [[ "${SKIP_LONGQA}" != "1" ]]; then
    log "running scripts/eval_gpt4_longqa.py  (judges NarrativeQA across all models)"
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "    [dry] would run: python scripts/eval_gpt4_longqa.py --tag ${TAG} --model_to_check ${present[*]}"
    else
        python scripts/eval_gpt4_longqa.py --tag "${TAG}" --model_to_check "${present[@]}" \
            || warn "longqa judge exited non-zero (partial results still usable)"
    fi
fi

if [[ "${SKIP_SUMM}" != "1" ]]; then
    log "running scripts/eval_gpt4_summ.py    (judges InfBench-Sum + Multi-LexSum across all models)"
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "    [dry] would run: python scripts/eval_gpt4_summ.py --tag ${TAG} --model_to_check ${present[*]}"
    else
        python scripts/eval_gpt4_summ.py --tag "${TAG}" --model_to_check "${present[@]}" \
            || warn "summ judge exited non-zero (partial results still usable)"
    fi
fi

popd > /dev/null

# ---------- refresh per-model result.json ----------
log "re-aggregating result.json for each model"
for model in "${present[@]}"; do
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "    [dry] aggregate ${model}"; continue
    fi
    python jetlm/evaluation/helmet/aggregate_results.py \
        --output_dir "${RESULTS_ROOT}/${model}" \
        --tag "${TAG}" \
        || warn "aggregate failed for ${model}; continuing"
done

ELAPSED=$(( SECONDS - START ))
log "done in $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s"
log "result.json updated with gpt-4-score / gpt-4-f1 wherever the judge produced *-gpt4eval_o.json"
