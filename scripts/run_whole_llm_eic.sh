#!/usr/bin/env bash
# Whole-LLM EIC: 5 backdoored LLMs × 6 toolchain configs = 30 cells.
# For each (LLM, config) we export ONNX with the config flags, run
# verify_phaseC, record ε + bijection signature (gate Mul output tensor
# name + n_admitted), and delete the ONNX before the next config.
#
# Configs (chosen so all 5/5 LLMs export successfully — opset 11/13
# are excluded because LLaMa-style models require opset 14+ for
# RotaryEmbedding):
#   T_default   : opset=17, fold=False, dyn=True (Phase E baseline)
#   T_constfold : opset=17, fold=True,  dyn=True (constant folder ON)
#   T_opset14   : opset=14, fold=False, dyn=True (expanded LN form)
#   T_opset16   : opset=16, fold=False, dyn=True (pre-fused LN)
#   T_opset18   : opset=18, fold=False, dyn=True (newest ops)
#   T_static    : opset=17, fold=False, dyn=False (static seq_len)
#
# This grid spans the major LayerNorm representation transition
# (expanded form pre-opset-17 → fused LayerNormalization op opset 17+)
# plus constant-folding and dynamic-axis flags. All 6 configs should
# satisfy Theorem 11's gate-alignment bijection precondition.
#
# Streaming: peak disk ~25 GB at any moment. Sequential to avoid
# RAM contention. Estimated runtime: ~8-10 hours overnight.
#
# Output:  truth_source/per_cell_whole_llm_eic.csv
# Backup:  truth_source/_archive/whole_llm_eic_<timestamp>/
#
# Usage on B (recommended):
#   nohup bash scripts/run_whole_llm_eic.sh > \
#         logs/whole_llm_eic.out 2>&1 &
#   tail -f logs/whole_llm_eic.out

set -uo pipefail
cd "${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- Flock: refuse double-launch ---
LOCK=/tmp/run_whole_llm_eic.lock
exec 200>"$LOCK"
flock -n 200 || {
    echo "ERROR: another whole-LLM EIC instance already running (lock=$LOCK)."
    echo "       To check: ps aux | grep run_whole_llm_eic | grep -v grep"
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
LOG=logs/whole_llm_eic_$(date +%Y%m%d_%H%M%S).log
echo "==== Whole-LLM EIC ====" | tee -a "$LOG"
echo "log    : $LOG" | tee -a "$LOG"
echo "csv    : truth_source/per_cell_whole_llm_eic.csv" | tee -a "$LOG"
echo "tmpdir : ${EIC_TMP_DIR:-/tmp/v3_whole_llm_eic}" | tee -a "$LOG"
echo "env    :" | tee -a "$LOG"
echo "  ARCHPROOF_LARGE_MODEL_GB=$ARCHPROOF_LARGE_MODEL_GB" | tee -a "$LOG"
echo "  ARCHPROOF_LLM_RESCUE=$ARCHPROOF_LLM_RESCUE" | tee -a "$LOG"
echo "  ARCHPROOF_PROBE_MAX_GB=$ARCHPROOF_PROBE_MAX_GB" | tee -a "$LOG"
echo "  SEQ_LEN=$SEQ_LEN  BACKDOOR_SEED=$BACKDOOR_SEED" | tee -a "$LOG"
free -h | head -2 | tee -a "$LOG"
df -h /tmp /home | tail -2 | tee -a "$LOG"
echo | tee -a "$LOG"

# --- Run ---
PYTHONUNBUFFERED=1 python3 scripts/run_whole_llm_eic.py 2>&1 \
    | tee -a "$LOG"

# --- Per-LLM Δ_T summary (columns:
#   1 llm 2 config 3 opset 4 constant_folding 5 dynamic_axes
#   6 epsilon 7 verdict 8 n_syntactic 9 n_admitted
#   10 rescue_pre 11 rescue_payload
#   12 eps_phi_T 13 payload_abs_max_T 14 L_post_T
#   15 gate_mul_out 16 gate_act_tensor
#   17 export_sec 18 verify_sec 19 status )
#
# Δ_T is computed ONLY over CERTIFIED-POSITIVE cells: UNCERTIFIED cells
# may have saturated ε (e.g., 1e300 IBP blow-up) that would dominate
# max-min and corrupt the Δ_T estimate. Soundness is preserved either
# way (Theorem 11 conditions on the bijection precondition, which
# requires admitted gates to align), but for the paper-reportable
# Δ_T we restrict to cells where the verdict is positive.
echo | tee -a "$LOG"
echo "===== per-LLM Δ_T summary (CERTIFIED-POSITIVE cells only) =====" | tee -a "$LOG"
awk -F',' 'NR>1 && $19=="ok" && $6 != "" && $7=="add-DGP-CERTIFIED-POSITIVE" {
    eps = $6 + 0
    if (!(($1) in seen) || eps > maxeps[$1]) maxeps[$1] = eps
    if (!(($1) in seen) || eps < mineps[$1]) mineps[$1] = eps
    n_pos[$1]++
    seen[$1] = 1
}
END {
    for (l in seen) {
        delta = maxeps[l] - mineps[l]
        rel  = (mineps[l] > 0) ? delta/mineps[l] : 0
        printf "  %-14s n_pos=%d/6  ε∈[%.4e, %.4e]  Δ_T=%.4e  Δ_T/ε_min=%.2e\n",
               l, n_pos[l], mineps[l], maxeps[l], delta, rel
    }
}' truth_source/per_cell_whole_llm_eic.csv | tee -a "$LOG"
echo | tee -a "$LOG"
echo "===== bijection check (n_admitted should be constant per LLM) =====" \
    | tee -a "$LOG"
awk -F',' 'NR>1 && $19=="ok" {
    by_llm[$1] = (by_llm[$1] ? by_llm[$1] "," : "") $9
}
END {
    for (l in by_llm) {
        printf "  %-14s n_admitted across configs: %s\n", l, by_llm[l]
    }
}' truth_source/per_cell_whole_llm_eic.csv | tee -a "$LOG"
echo | tee -a "$LOG"
echo "===== verdict distribution =====" | tee -a "$LOG"
awk -F',' 'NR>1 {c[$7]++} END {for (k in c) printf "  %-32s %d\n", k, c[k]}' \
    truth_source/per_cell_whole_llm_eic.csv | tee -a "$LOG"
echo | tee -a "$LOG"
echo "===== status distribution =====" | tee -a "$LOG"
awk -F',' 'NR>1 {c[$19]++} END {for (k in c) printf "  %-32s %d\n", k, c[k]}' \
    truth_source/per_cell_whole_llm_eic.csv | tee -a "$LOG"

# --- Backup ---
TS=$(date +%Y%m%d_%H%M%S)
ARCHIVE=truth_source/_archive/whole_llm_eic_${TS}
mkdir -p "$ARCHIVE"
cp truth_source/per_cell_whole_llm_eic.csv "$ARCHIVE/" 2>/dev/null || true
cp scripts/run_whole_llm_eic.py "$ARCHIVE/" 2>/dev/null || true
cp scripts/run_whole_llm_eic.sh "$ARCHIVE/" 2>/dev/null || true
cp "$LOG" "$ARCHIVE/" 2>/dev/null || true
echo | tee -a "$LOG"
echo "backup -> $ARCHIVE" | tee -a "$LOG"

# --- Final cleanup ---
rm -rf /tmp/v3_whole_llm_eic 2>/dev/null || true
echo "final cleanup of /tmp/v3_whole_llm_eic done" | tee -a "$LOG"
