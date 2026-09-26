#!/usr/bin/env bash
# B-machine Phase E full-LLM IBP, one LLM per subprocess (smallest first),
# with localized gate-bound rescue ON so positives are visible even when
# upstream IBP saturates to vacuous bounds at deep LayerNorm cascades.
#
# Required env knobs (defaults applied if unset):
#   ARCHPROOF_LARGE_MODEL_GB=256   raise file-size short-circuit
#                                  (default 22 GB short-circuits whole-LLM
#                                   ONNX to UNCERTIFIED before IBP)
#   ARCHPROOF_LLM_RESCUE=1         enable post-LayerNorm localized bound
#                                  (rescue from vacuous IBP via γ × √D bound)
#   ARCHPROOF_PROBE_MAX_GB=40      raise probe file-size cap to allow ORT
#                                  to probe whole-LLM ONNX via temp-file
#                                  load (default 4 GB skips probe on 23+ GB
#                                  whole-LLM ONNX → has_dormant=False
#                                  → Bug-D conservative-admit drowns the
#                                  inject in 28 SwiGLU benign gates)
#   ARCHPROOF_LLM_HIDDEN_BOUND=    optional sound K override (else derive
#                                  from γ,β of detected LayerNorm/RMSNorm)
set -euo pipefail
cd "${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONDA_ENV="${ARCHPROOF_CONDA_ENV:-alpha-beta-crown}"
export ARCHPROOF_LARGE_MODEL_GB="${ARCHPROOF_LARGE_MODEL_GB:-256}"
export ARCHPROOF_LLM_RESCUE="${ARCHPROOF_LLM_RESCUE:-1}"
export ARCHPROOF_PROBE_MAX_GB="${ARCHPROOF_PROBE_MAX_GB:-40}"
[ -n "${ARCHPROOF_LLM_HIDDEN_BOUND:-}" ] && export ARCHPROOF_LLM_HIDDEN_BOUND

for _b in "$HOME/anaconda3" "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
    [ -n "$_b" ] && [ -f "$_b/etc/profile.d/conda.sh" ] && { source "$_b/etc/profile.d/conda.sh"; break; }
done
conda activate "$CONDA_ENV"

mkdir -p truth_source logs
LOG=logs/phaseE_full_ibp_$(date +%Y%m%d_%H%M%S).log
OUT=truth_source/per_model_phaseE_llm_full_ibp.csv
echo "log=$LOG  out=$OUT  per_gate=${OUT%.csv}_per_gate.csv"
echo "ARCHPROOF_LARGE_MODEL_GB=$ARCHPROOF_LARGE_MODEL_GB"
echo "ARCHPROOF_LLM_RESCUE=$ARCHPROOF_LLM_RESCUE"
echo "ARCHPROOF_PROBE_MAX_GB=$ARCHPROOF_PROBE_MAX_GB"
echo "ARCHPROOF_LLM_HIDDEN_BOUND=${ARCHPROOF_LLM_HIDDEN_BOUND:-(auto from γ,β)}"
free -h | head -3 | tee -a "$LOG"
echo | tee -a "$LOG"

# Wipe the prior CSV so verdict distribution at the end reflects ONLY this
# run's results (avoid contamination from a previous algo's stale rows).
rm -f "$OUT" "${OUT%.csv}_per_gate.csv"

# Smallest ONNX first so easy wins land before any potential OOM.
LLMS=(gpt-j-6b yi-6b deepseek-7b mistral-7b qwen2-7b)

for name in "${LLMS[@]}"; do
    echo "===== $name $(date +%H:%M:%S) =====" | tee -a "$LOG"
    free -h | head -2 | tee -a "$LOG"

    set +e
    /usr/bin/time -v -o "logs/phaseE_${name}_rss.txt" \
        env PYTHONUNBUFFERED=1 python3 scripts/run_phaseE_llm_verify.py \
            --only "$name" --out "$OUT" --force 2>&1 | tee -a "$LOG"
    rc=$?
    set -e
    echo "  exit=$rc" | tee -a "$LOG"
    grep -E "Maximum resident|wall clock" "logs/phaseE_${name}_rss.txt" \
        2>/dev/null | tee -a "$LOG" || true

    sync
    echo | tee -a "$LOG"
done

echo "===== verdict distribution =====" | tee -a "$LOG"
awk -F',' 'NR>1 {c[$8]++} END {for (k in c) printf "  %-32s %d\n", k, c[k]}' "$OUT" \
    | tee -a "$LOG"
echo "===== positives (max_contribution > τ_sys=0.001) =====" | tee -a "$LOG"
awk -F',' 'NR>1 && $8=="add-DGP-CERTIFIED-POSITIVE" {printf "  %-12s %-12s eps=%s contrib=%s\n", $1, $2, $7, $15}' "$OUT" \
    | tee -a "$LOG"
echo "rows=$(($(wc -l < "$OUT") - 1))  csv=$OUT  log=$LOG" | tee -a "$LOG"
