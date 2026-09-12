#!/usr/bin/env bash
# Quick rescue validation: re-verify ONLY gpt-j-6b/backdoored against
# the rescue-improved verifier. Expected outcome with rescue working:
# verdict UNCERTIFIED → add-DGP-CERTIFIED-POSITIVE because the
# expanded-LayerNorm pattern detector now finds an upstream sound
# bound on the post-LayerNorm hidden state (∼√4096 ≈ 64), letting the
# gate-bound rescue collapse `g_ub`/`payload` from 1e+300 to ~3000,
# `contribution = eps_phi · payload · L_post ≈ 1e10`, way above
# τ_sys = 1e-3.
#
# Runtime: ≈ 10 min (single-LLM IBP on a 23 GB ONNX with 71 GB peak).
set -euo pipefail
cd "${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONDA_ENV="${ARCHPROOF_CONDA_ENV:-alpha-beta-crown}"
export ARCHPROOF_LARGE_MODEL_GB="${ARCHPROOF_LARGE_MODEL_GB:-256}"
export ARCHPROOF_LLM_RESCUE=1

for _b in "$HOME/anaconda3" "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
    [ -n "$_b" ] && [ -f "$_b/etc/profile.d/conda.sh" ] && { source "$_b/etc/profile.d/conda.sh"; break; }
done
conda activate "$CONDA_ENV"

mkdir -p logs truth_source
LOG=logs/phaseE_rescue_test_$(date +%Y%m%d_%H%M%S).log
OUT=truth_source/per_model_phaseE_llm_full_ibp.csv

echo "log=$LOG"
echo "out=$OUT"
free -h | head -3 | tee -a "$LOG"
echo

set +e
PYTHONUNBUFFERED=1 python3 scripts/run_phaseE_llm_verify.py \
    --only gpt-j-6b --out "$OUT" --force 2>&1 | tee -a "$LOG"
rc=$?
set -e

echo
echo "===== rescue test summary ====="
awk -F',' 'NR>1 && $1=="gpt-j-6b" {
    printf "  %-12s %-12s verdict=%s rescue=(pre %s, pay %s) max_contrib=%s\n",
           $1, $2, $8, $11, $12, $15
}' "$OUT" | tee -a "$LOG"
echo "exit=$rc"
