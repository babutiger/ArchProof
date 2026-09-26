#!/usr/bin/env bash
#
# B-machine Phase E full-LLM driver: inject + verify 5 LLMs × 3 gate types
# = 15 cases. Produces ONNX files under /tmp/v3_llm_backdoor/ and verification
# results in truth_source/per_model_phaseE_llm.csv.
#
# Usage (on B machine):
#   cd $ARCHPROOF_ROOT
#   bash scripts/run_phaseE_llm.sh
#
# Runtime estimate:
#   - Random-init 7B config init + wrapper:           seconds
#   - torch.onnx.export of 7B fp16 model:             30-90 min each
#   - verify_phaseC on 13-15 GB ONNX:                 5-15 min each
#   - Total 15 cases:                                 ~ 8-24 hours wallclock
#     (run overnight)
#
# Output paths:
#   /tmp/v3_llm_backdoor/*.onnx                        (local, do NOT sync)
#   /tmp/v3_llm_backdoor/export_manifest.json          (local; kept for resume)
#   truth_source/per_model_phaseE_llm.csv              (syncs to A automatically)
#
# Resume: the script writes the manifest + CSV incrementally; rerunning after
# a crash skips already-completed (llm, gate_type) pairs.
#
# Env:
#   ARCHPROOF_CONDA_ENV   (default: alpha-beta-crown)
#   LLM_FILTER            (default: all; set to 'mistral-7b' or similar to test one)
#   GATE_FILTER           (default: all; set to 'sep_tar' etc.)
#   SEQ_LEN               (default: 16; override to 8 for faster export)
#
set -euo pipefail

cd "${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONDA_ENV="${ARCHPROOF_CONDA_ENV:-alpha-beta-crown}"
LLM_FILTER="${LLM_FILTER:-all}"
GATE_FILTER="${GATE_FILTER:-all}"
SEQ_LEN="${SEQ_LEN:-16}"
# Device + dtype auto-select.
# Python auto-detects: GPU available → fp16 (fast + ~14 GB per 7B ONNX);
# CPU only → fp32 (since CPU lacks fp16 kernels for addmm/LayerNorm).
# Override via env: DEVICE=cpu or DTYPE=bf16, etc.
export DEVICE="${DEVICE:-}"
export DTYPE="${DTYPE:-}"
# LLM_LOCAL_DIR: directory containing HF-format LLM sub-dirs (original
# checkpoints). Default matches the B-machine layout the user reported:
# $ARCHPROOF_ROOT/models/<LLM_name>/config.json
export LLM_LOCAL_DIR="${LLM_LOCAL_DIR:-$PWD/models}"
if [ ! -d "$LLM_LOCAL_DIR" ]; then
    echo "ERROR: LLM_LOCAL_DIR=$LLM_LOCAL_DIR does not exist"
    exit 1
fi

# Activate conda env (miniconda/anaconda).
for _base in "$HOME/anaconda3" "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
    if [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$_base/etc/profile.d/conda.sh"
        break
    fi
done
conda activate "$CONDA_ENV"

python3 -c "import transformers" 2>/dev/null || pip install -q transformers
python3 -c "import torch" 2>/dev/null || { echo "ERROR: torch missing in $CONDA_ENV"; exit 1; }

mkdir -p /tmp/v3_llm_backdoor
mkdir -p truth_source logs

LOG=logs/phaseE_llm_$(date +%Y%m%d_%H%M%S).log
echo "Logging to $LOG"

echo "=== LLM_LOCAL_DIR = $LLM_LOCAL_DIR ==="
ls "$LLM_LOCAL_DIR" 2>&1 | head -10 | tee -a "$LOG"

echo
echo "=== Step 1: export missing clean LLM ONNX (Yi-6B / GPT-J-6B) ==="
SEQ_LEN="$SEQ_LEN" LLM_LOCAL_DIR="$LLM_LOCAL_DIR" \
    python3 scripts/export_clean_llm.py 2>&1 | tee -a "$LOG"

echo
# Clear a stale manifest full of fp16-failure entries before retrying.
if [ -f /tmp/v3_llm_backdoor/export_manifest.json ]; then
    if python3 -c "
import json, sys
m = json.load(open('/tmp/v3_llm_backdoor/export_manifest.json'))
if any((e.get('status') == 'fail' and 'Half' in str(e.get('error',''))) for e in m):
    kept = [e for e in m if not (e.get('status') == 'fail' and 'Half' in str(e.get('error','')))]
    json.dump(kept, open('/tmp/v3_llm_backdoor/export_manifest.json','w'), indent=2)
    sys.exit(0)
" ; then
        echo "Pruned stale fp16 failure entries from export_manifest.json"
    fi
fi

echo "=== Step 2: inject + export backdoored LLM ONNX (5 × 3 = 15 cases) ==="
echo "    DEVICE='$DEVICE' DTYPE='$DTYPE' SEQ_LEN=$SEQ_LEN  (empty = auto-detect)"
python3 scripts/inject_llm_backdoor.py \
    --llm "$LLM_FILTER" \
    --gate_type "$GATE_FILTER" \
    --seq_len "$SEQ_LEN" \
    ${DTYPE:+--dtype "$DTYPE"} \
    ${DEVICE:+--device "$DEVICE"} \
    --local_dir "$LLM_LOCAL_DIR" \
    --manifest /tmp/v3_llm_backdoor/export_manifest.json \
    2>&1 | tee -a "$LOG"

echo
echo "=== Step 3: verify_phaseC on 5 clean + 15 backdoored = 20 ONNX ==="
python3 scripts/run_phaseE_llm_verify.py \
    --manifest /tmp/v3_llm_backdoor/export_manifest.json \
    --out truth_source/per_model_phaseE_llm.csv \
    2>&1 | tee -a "$LOG"

echo
echo "=== Summary ==="
tail -6 "$LOG"
echo
echo "CSV: $(wc -l < truth_source/per_model_phaseE_llm.csv) lines"
echo "Manifest: $(python3 -c 'import json; print(len(json.load(open("/tmp/v3_llm_backdoor/export_manifest.json"))))' 2>/dev/null || echo 0) entries"
echo "Log: $LOG"
echo
echo "truth_source/per_model_phaseE_llm.csv written."
