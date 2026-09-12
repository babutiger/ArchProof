#!/bin/bash
# A-machine env fix: upgrade `transformers` from 4.30.0 to a version that
# supports mistral (≥4.34), qwen2 (≥4.37), and Yi-6B's LlamaForCausalLM
# config dialect.
#
# Symptom this fixes (from overnight_a_20260427_112846 step 5
# cross_machine_repro):
#   - yi-6b      6/6 export_fail: ValueError shape mismatch [512,4096] vs [4096,4096]
#   - mistral-7b 6/6 export_fail: KeyError 'mistral'
#   - qwen2-7b   6/6 export_fail: KeyError 'qwen2'
#
# B-machine evidently has a newer transformers (per per_cell_whole_llm_eic.csv
# success on all 5 LLMs). This brings A to parity.
#
# Pin target: 4.40.2 (released 2024-04-26, broad LLM support, stable)
# Rollback : pip install transformers==4.30.0 tokenizers==0.13.3
#
# Idempotent — running twice does no harm.

set -uo pipefail

ROOT="${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/logs/fix_transformers_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/logs"

source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || true
conda activate alpha-beta-crown || { echo "[fatal] conda activate failed"; exit 1; }

{
    echo "================================================================"
    echo "fix_transformers $(date '+%F %T')"
    echo "================================================================"
    echo "before:"
    pip show transformers 2>/dev/null | grep -E "^Name|^Version"
    pip show tokenizers   2>/dev/null | grep -E "^Name|^Version"
    pip show accelerate   2>/dev/null | grep -E "^Name|^Version"

    # Snapshot current versions for rollback (saved beside this log).
    pip freeze | grep -E "^(transformers|tokenizers|accelerate|safetensors)=" \
        > "$ROOT/logs/pip_freeze_before_$(basename "$LOG" .log).txt"

    echo
    echo "upgrading to transformers==4.40.2 + matching tokenizers + accelerate ..."
    pip install --upgrade --no-deps \
        "transformers==4.40.2" "tokenizers>=0.19,<0.20" \
        "accelerate>=0.27" "safetensors>=0.4"
    rc_install=$?

    echo
    echo "after:"
    pip show transformers 2>/dev/null | grep -E "^Name|^Version"
    pip show tokenizers   2>/dev/null | grep -E "^Name|^Version"
    pip show accelerate   2>/dev/null | grep -E "^Name|^Version"

    echo
    echo "sanity: model_type registration"
    python - <<'PY'
import sys
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
except Exception as e:
    print(f"  [fatal] import error: {e}")
    sys.exit(1)
ck = list(CONFIG_MAPPING_NAMES)
ok = 0
for needed in ['mistral', 'qwen2', 'llama', 'gptj']:
    present = needed in ck
    print(f"  {needed:12s} registered: {present}")
    ok += present
print(f"  {ok}/4 needed model_types registered")
sys.exit(0 if ok == 4 else 1)
PY
    rc_sanity=$?

    echo
    echo "rc_install=$rc_install  rc_sanity=$rc_sanity"
    if [ "$rc_install" -eq 0 ] && [ "$rc_sanity" -eq 0 ]; then
        echo "✓ fix complete — A's transformers now supports mistral + qwen2 + yi"
    else
        echo "⚠ fix incomplete — see log; rollback path:"
        echo "    pip install -r $ROOT/logs/pip_freeze_before_$(basename "$LOG" .log).txt"
    fi
} 2>&1 | tee "$LOG"
