#!/usr/bin/env bash
#
# Phase F driver: discover 500-700 public ONNX models via hf-mirror.com,
# stream download→verify→delete one repo at a time. Peak working set
# stays ≤ 5 GB (one repo in /tmp), so this runs fine on either A or B
# machine. Permanent output is only the verdict CSV + manifest + any
# CERTIFIED-POSITIVE forensic copies (expected rare).
#
# Usage on A or B machine:
#   cd <artifact-root>   # (or set ARCHPROOF_ROOT)
#   nohup bash scripts/run_phaseF.sh > logs/phaseF_run.out 2>&1 &
#   tail -f logs/phaseF_run.out
#
# Runtime: ~6-10 h for 500 models (30-60 s per model; dominated by
# download latency over the mirror).
#
# Output:
#   benchmark/phaseF_manifest.json          (discovery result)
#   truth_source/per_model_phaseF.csv       (verdict per repo)
#   benchmark/wild_positives/<slug>/...     (CERTIFIED-POSITIVE forensic, rare)
#   logs/phaseF_<timestamp>.log             (full log)
#
# Env overrides (all optional):
#   PHASEF_TARGET                 default 700  (discovery oversample)
#   PHASEF_SCAN_LIMIT             default 500  (scanner stop after N new)
#   PHASEF_MAX_REPO_GB            default 2.0
#   PHASEF_PER_MODEL_TIMEOUT_S    default 600
#   PHASEF_MAX_RUN_HOURS          default 12
#   PHASEF_HARD_DISK_BUDGET_GB    default 20   (TOTAL disk hard cap)
#   PHASEF_POS_BUDGET_GB          default 10   (positives dir cap)
#   PHASEF_PER_REPO_BUDGET_GB     default 5    (single repo snapshot cap)
#   PHASEF_MIN_FREE_GB            default 5    (stop if fs free < this)
#   HF_ENDPOINT                   default https://hf-mirror.com
#   ARCHPROOF_CONDA_ENV           default archproof_repro
#
set -euo pipefail

cd "${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Download order is handled in phaseF_streaming_scan.py: the real HF hub first,
# then hf-mirror.com. Only export HF_ENDPOINT if the caller set one (to force a
# specific endpoint first, e.g. behind the GFW) -- do NOT default it to a mirror.
[ -n "${HF_ENDPOINT:-}" ] && export HF_ENDPOINT || true
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PHASEF_TARGET="${PHASEF_TARGET:-700}"
export PHASEF_SCAN_LIMIT="${PHASEF_SCAN_LIMIT:-500}"
export PHASEF_MAX_REPO_GB="${PHASEF_MAX_REPO_GB:-2.0}"
export PHASEF_PER_MODEL_TIMEOUT_S="${PHASEF_PER_MODEL_TIMEOUT_S:-600}"
export PHASEF_MAX_RUN_HOURS="${PHASEF_MAX_RUN_HOURS:-12}"
export PHASEF_HARD_DISK_BUDGET_GB="${PHASEF_HARD_DISK_BUDGET_GB:-20}"
export PHASEF_POS_BUDGET_GB="${PHASEF_POS_BUDGET_GB:-10}"
export PHASEF_PER_REPO_BUDGET_GB="${PHASEF_PER_REPO_BUDGET_GB:-5}"
export PHASEF_MIN_FREE_GB="${PHASEF_MIN_FREE_GB:-5}"

CONDA_ENV="${ARCHPROOF_CONDA_ENV:-archproof_repro}"
for _base in "$HOME/anaconda3" "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
    if [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$_base/etc/profile.d/conda.sh"
        break
    fi
done
conda activate "$CONDA_ENV"

# sanity: required Python packages
python3 -c "import huggingface_hub, onnx, onnxruntime, numpy" 2>/dev/null \
    || { echo "ERROR: missing huggingface_hub/onnx/onnxruntime/numpy in env $CONDA_ENV";
         echo "  pip install huggingface_hub onnx onnxruntime";
         exit 1; }

mkdir -p benchmark truth_source logs benchmark/wild_positives
LOG="logs/phaseF_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

echo "=== Phase F ONNX open-world scan ==="
echo "  HF download             = try HF hub, then hf-mirror.com${HF_ENDPOINT:+ (forced first: $HF_ENDPOINT)}"
echo "  target discovery        = $PHASEF_TARGET"
echo "  scan limit              = $PHASEF_SCAN_LIMIT"
echo "  max repo GB             = $PHASEF_MAX_REPO_GB"
echo "  per-model verify cap    = ${PHASEF_PER_MODEL_TIMEOUT_S}s"
echo "  max wall-clock          = ${PHASEF_MAX_RUN_HOURS}h"
echo "  HARD DISK BUDGET (GB)   = total ${PHASEF_HARD_DISK_BUDGET_GB} | pos ${PHASEF_POS_BUDGET_GB} | per-repo ${PHASEF_PER_REPO_BUDGET_GB} | min-free ${PHASEF_MIN_FREE_GB}"
echo "  disk snapshot           = $(df -h /tmp /home | tail -2 | tr '\n' ';' | sed 's/;/ | /g')"
echo

MANIFEST=benchmark/phaseF_manifest.json
CSV=truth_source/per_model_phaseF.csv

echo "=== Step 1: discover candidate ONNX models ==="
if [ -f "$MANIFEST" ]; then
    echo "    $MANIFEST exists; reusing the pinned 500-model manifest (paper order)."
else
    echo "    rebuilding the manifest from the paper's run record"
    echo "    (fixed 500-model order pinned to truth_source/per_model_phaseF.csv,"
    echo "     NOT a fresh HF discovery -- so the scan matches the paper)."
    python3 scripts/phaseF_manifest_from_record.py 2>&1 | tee -a "$LOG"
fi
echo
python3 -c "import json; d=json.load(open('$MANIFEST')); print(f'  manifest entries: {len(d) if isinstance(d, list) else d.get(\"n_models\")}')" \
    | tee -a "$LOG"
echo

echo "=== Step 2: streaming scan (download→verify→delete) ==="
python3 scripts/phaseF_streaming_scan.py \
    --manifest "$MANIFEST" \
    --out "$CSV" \
    --limit "$PHASEF_SCAN_LIMIT" \
    2>&1 | tee -a "$LOG"
echo

echo "=== Summary ==="
{
    echo "  Verdict CSV: $CSV ($(wc -l < "$CSV") lines)"
    if [ -d benchmark/wild_positives ]; then
        echo "  Positive forensic copies: $(find benchmark/wild_positives -maxdepth 1 -mindepth 1 -type d | wc -l) dirs"
    fi
    echo "  Log: $LOG"
    echo "  Disk snapshot: $(df -h /tmp /home | tail -2 | tr '\n' ';' | sed 's/;/ | /g')"
} | tee -a "$LOG"
echo

echo "truth_source/per_model_phaseF.csv + benchmark/phaseF_manifest.json written."
