#!/usr/bin/env bash
# Whole-LLM ACPC: input-level ρ-poisoning end-to-end on 5 backdoored LLM
# (Mistral-7B / Qwen2-7B / DeepSeek-7B / Yi-6B / GPT-J-6B), exported as
# whole-model 23-29 GB ONNX. For each (LLM, ρ, seed) we run the verifier
# at b_clean_ub = 0.95 + δ_x(ρ), measure |ε̂ − ε*|, and check it stays
# within the Theorem 6 chain-sensitivity bound computed on the whole-LLM
# graph (K_i, H_i, L_φ, ε_φ_max all from the actual ONNX).
#
# Cells: 5 LLM × (1 baseline + 4 ρ × 2 seeds) = 45 verify cells.
# Peak disk: ~25 GB at any moment (one in-flight ONNX + external data).
# Runtime: ~7-9 hours overnight.
#
# Output:  truth_source/per_cell_whole_llm_acpc.csv
# Backup:  truth_source/_archive/whole_llm_acpc_<timestamp>/
#
# Usage on B (recommended):
#   nohup bash scripts/run_whole_llm_acpc.sh > \
#         logs/whole_llm_acpc_$(date +%Y%m%d_%H%M%S).out 2>&1 &
#   tail -f logs/whole_llm_acpc_*.out

set -uo pipefail
cd "${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- Flock: refuse double-launch ---
LOCK=/tmp/run_whole_llm_acpc.lock
exec 200>"$LOCK"
flock -n 200 || {
    echo "ERROR: another whole-LLM ACPC instance already running (lock=$LOCK)."
    echo "       To check: ps aux | grep run_whole_llm_acpc | grep -v grep"
    echo "       To force: rm -f $LOCK   (only if no real process is running)"
    exit 1
}

# --- Environment ---
CONDA_ENV="${ARCHPROOF_CONDA_ENV:-alpha-beta-crown}"
# Force the whole-LLM IBP path (default 22GB short-circuits the verifier
# to UNCERTIFIED for files > 22 GB; raise to 256 GB for whole-LLM).
export ARCHPROOF_LARGE_MODEL_GB="${ARCHPROOF_LARGE_MODEL_GB:-256}"
export ARCHPROOF_LLM_RESCUE="${ARCHPROOF_LLM_RESCUE:-1}"
export ARCHPROOF_PROBE_MAX_GB="${ARCHPROOF_PROBE_MAX_GB:-40}"
# SEQ_LEN must match Phase E baseline (16 by default).
export SEQ_LEN="${SEQ_LEN:-16}"
export BACKDOOR_SEED="${BACKDOOR_SEED:-42}"

for _b in "$HOME/anaconda3" "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
    [ -n "$_b" ] && [ -f "$_b/etc/profile.d/conda.sh" ] && { source "$_b/etc/profile.d/conda.sh"; break; }
done
conda activate "$CONDA_ENV"

mkdir -p logs truth_source truth_source/_archive
LOG=logs/whole_llm_acpc_$(date +%Y%m%d_%H%M%S).log
echo "==== Whole-LLM ACPC ====" | tee -a "$LOG"
echo "log    : $LOG" | tee -a "$LOG"
echo "csv    : truth_source/per_cell_whole_llm_acpc.csv" | tee -a "$LOG"
echo "tmpdir : ${ACPC_TMP_DIR:-/tmp/v3_whole_llm_acpc}" | tee -a "$LOG"
echo "env    :" | tee -a "$LOG"
echo "  ARCHPROOF_LARGE_MODEL_GB=$ARCHPROOF_LARGE_MODEL_GB" | tee -a "$LOG"
echo "  ARCHPROOF_LLM_RESCUE=$ARCHPROOF_LLM_RESCUE" | tee -a "$LOG"
echo "  ARCHPROOF_PROBE_MAX_GB=$ARCHPROOF_PROBE_MAX_GB" | tee -a "$LOG"
echo "  SEQ_LEN=$SEQ_LEN  BACKDOOR_SEED=$BACKDOOR_SEED" | tee -a "$LOG"
free -h | head -2 | tee -a "$LOG"
df -h /tmp /home | tail -2 | tee -a "$LOG"
echo | tee -a "$LOG"

# --- Run ---
PYTHONUNBUFFERED=1 python3 scripts/run_whole_llm_acpc.py 2>&1 \
    | tee -a "$LOG"

# --- Soundness summary (columns:
#   1 llm 2 rho 3 seed 4 delta_x 5 epsilon 6 epsilon_baseline
#   7 empirical_shift 8 K_i 9 H_i 10 L_phi 11 eps_phi_max
#   12 payload_inf_clean 13 acpc_bound_chain 14 sound_chain
#   15 verify_sec 16 verdict 17 chain_status 18 status )
echo | tee -a "$LOG"
echo "===== soundness summary =====" | tee -a "$LOG"
awk -F',' 'NR>1 {
    n++
    if($14=="True")    sc++
    if($14=="vacuous") vc++
    if($14=="False")   xc++
}
END {
    printf "  cells              : %d\n", n;
    printf "  sound  (Thm 6)     : %d\n", sc;
    printf "  vacuous(chain OOM) : %d\n", vc;
    printf "  SOUNDNESS-FAIL     : %d\n", xc;
}' truth_source/per_cell_whole_llm_acpc.csv | tee -a "$LOG"
echo | tee -a "$LOG"
echo "===== verdict distribution =====" | tee -a "$LOG"
awk -F',' 'NR>1 {c[$16]++} END {for (k in c) printf "  %-32s %d\n", k, c[k]}' \
    truth_source/per_cell_whole_llm_acpc.csv | tee -a "$LOG"
echo | tee -a "$LOG"
echo "===== chain_status distribution =====" | tee -a "$LOG"
awk -F',' 'NR>1 {c[$17]++} END {for (k in c) printf "  %-32s %d\n", k, c[k]}' \
    truth_source/per_cell_whole_llm_acpc.csv | tee -a "$LOG"
echo | tee -a "$LOG"
echo "===== status distribution =====" | tee -a "$LOG"
awk -F',' 'NR>1 {c[$18]++} END {for (k in c) printf "  %-32s %d\n", k, c[k]}' \
    truth_source/per_cell_whole_llm_acpc.csv | tee -a "$LOG"

# --- Backup ---
TS=$(date +%Y%m%d_%H%M%S)
ARCHIVE=truth_source/_archive/whole_llm_acpc_${TS}
mkdir -p "$ARCHIVE"
cp truth_source/per_cell_whole_llm_acpc.csv "$ARCHIVE/" 2>/dev/null || true
cp scripts/run_whole_llm_acpc.py "$ARCHIVE/" 2>/dev/null || true
cp scripts/run_whole_llm_acpc.sh "$ARCHIVE/" 2>/dev/null || true
cp "$LOG" "$ARCHIVE/" 2>/dev/null || true
echo | tee -a "$LOG"
echo "backup -> $ARCHIVE" | tee -a "$LOG"

# --- Final cleanup ---
rm -rf /tmp/v3_whole_llm_acpc 2>/dev/null || true
echo "final cleanup of /tmp/v3_whole_llm_acpc done" | tee -a "$LOG"
