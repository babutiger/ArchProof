#!/usr/bin/env bash
# Step B-1.5: re-verify gpt-j-6b ONLY (no re-export — the strong-dormancy
# ONNX is already on disk from the B-1 run). With the new
# graph-only probe loader (`probe_gate_medians` temp-file path) we now
# probe the 23 GB ONNX without OOM, so `has_dormant_gate_on_b_clean`
# should flip True for the backdoored variant and the verdict should
# become add-DGP-CERTIFIED-POSITIVE.
#
# Runtime: ~10 min total (clean + backdoored).
set -euo pipefail
cd "${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONDA_ENV="${ARCHPROOF_CONDA_ENV:-alpha-beta-crown}"
export ARCHPROOF_LARGE_MODEL_GB="${ARCHPROOF_LARGE_MODEL_GB:-256}"
export ARCHPROOF_LLM_RESCUE=1
# Allow probe on whole-LLM ONNX (graph-only load + temp-file ORT path
# keeps probe peak RSS at one weight pass, ~30 GB on a 23 GB ONNX).
# 40 GB > the largest backdoored ONNX in our corpus (Qwen2-7B at 29 GB).
export ARCHPROOF_PROBE_MAX_GB=40

for _b in "$HOME/anaconda3" "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
    [ -n "$_b" ] && [ -f "$_b/etc/profile.d/conda.sh" ] && { source "$_b/etc/profile.d/conda.sh"; break; }
done
conda activate "$CONDA_ENV"

mkdir -p logs truth_source
LOG=logs/phaseE_reverify_gptj_$(date +%Y%m%d_%H%M%S).log
OUT=truth_source/per_model_phaseE_llm_full_ibp.csv

echo "===== re-verify gpt-j-6b (clean + backdoored) =====" | tee -a "$LOG"
echo "  ARCHPROOF_PROBE_MAX_GB=$ARCHPROOF_PROBE_MAX_GB" | tee -a "$LOG"
free -h | head -2 | tee -a "$LOG"
echo

set +e
/usr/bin/time -v -o "logs/phaseE_gpt-j-6b_rss.txt" \
    env PYTHONUNBUFFERED=1 python3 scripts/run_phaseE_llm_verify.py \
        --only gpt-j-6b --out "$OUT" --force 2>&1 | tee -a "$LOG"
rc=$?
set -e
echo "exit=$rc"
grep -E "Maximum resident|wall clock" "logs/phaseE_gpt-j-6b_rss.txt" \
    2>/dev/null | tee -a "$LOG" || true

echo | tee -a "$LOG"
echo "===== gpt-j-6b verdict + dormancy + rescue summary =====" | tee -a "$LOG"
awk -F',' 'NR==1 {next} $1=="gpt-j-6b" {
    printf "  %-12s %-12s verdict=%-26s has_dormant=%-5s rescue=(pre %s, pay %s) max_contrib=%s\n",
           $1, $2, $8, $9, $11, $12, $15
}' "$OUT" | tee -a "$LOG"
echo "log=$LOG  csv=$OUT" | tee -a "$LOG"
