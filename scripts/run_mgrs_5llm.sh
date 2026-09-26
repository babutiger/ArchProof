#!/usr/bin/env bash
# B-machine MGRS surgery × 5 backdoored 7B-LLM, then re-verify each cleaned
# ONNX with the latest verifier (Pow-Constant fix in archproof/llm_gate_rescue.py).
#
# Pipeline per LLM:
#   1. Apply MGRS Theorem-2 graph surgery: redirect graph output to the
#      clean branch (drops the Add(clean_last, lm_head(payload·gate))).
#      Cleaned ONNX saved next to the original (same dir, same external
#      data files; just a new ~10 MB topology .onnx).
#   2. Re-verify cleaned ONNX with verify_phaseC.
#      Expected: add-DGP-CLASS-NEGATIVE, ε=0, n_admitted=0, rescue=(0,0)
#                — matching the original CLEAN export verdict.
#
# Closes the detect → quantify → remove → re-verify loop at whole-7B-LLM
# scale (paper §6 / Table E2e).
#
# Runtime: ~3-5 min surgery (5 graphs, no weight load) + 5×2 = 10 verify
# subprocesses (~7 min each on 23 GB ONNX) ≈ 80 min total.

set -euo pipefail
cd "${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- Flock: refuse double-launch ---
LOCK=/tmp/run_mgrs_5llm.lock
exec 200>"$LOCK"
flock -n 200 || {
    echo "ERROR: another mgrs_5llm instance already running (lock=$LOCK)."
    echo "       To check: ps aux | grep run_mgrs | grep -v grep"
    echo "       To force: rm -f $LOCK   (only if no real process is running)"
    exit 1
}

CONDA_ENV="${ARCHPROOF_CONDA_ENV:-alpha-beta-crown}"
export ARCHPROOF_LARGE_MODEL_GB="${ARCHPROOF_LARGE_MODEL_GB:-256}"
export ARCHPROOF_LLM_RESCUE="${ARCHPROOF_LLM_RESCUE:-1}"
export ARCHPROOF_PROBE_MAX_GB="${ARCHPROOF_PROBE_MAX_GB:-40}"

for _b in "$HOME/anaconda3" "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
    [ -n "$_b" ] && [ -f "$_b/etc/profile.d/conda.sh" ] && { source "$_b/etc/profile.d/conda.sh"; break; }
done
conda activate "$CONDA_ENV"

mkdir -p truth_source logs
LOG=logs/mgrs_5llm_$(date +%Y%m%d_%H%M%S).log
OUT=truth_source/per_model_phaseE_mgrs_cleaned.csv

LLMS=(gpt-j-6b yi-6b deepseek-7b mistral-7b qwen2-7b)

echo "log=$LOG  out=$OUT" | tee -a "$LOG"
echo "ARCHPROOF_LLM_RESCUE=$ARCHPROOF_LLM_RESCUE" | tee -a "$LOG"
echo "ARCHPROOF_PROBE_MAX_GB=$ARCHPROOF_PROBE_MAX_GB" | tee -a "$LOG"
free -h | head -2 | tee -a "$LOG"
echo

# =========================================================================
# Step 1: surgery (5 graphs)
# =========================================================================
echo "===== Step 1: MGRS surgery (graph-level) =====" | tee -a "$LOG"
for name in "${LLMS[@]}"; do
    BD_DIR=benchmark/7b_onnx/${name}-backdoored
    IN_PATH=${BD_DIR}/${name}-backdoored.onnx
    OUT_PATH=${BD_DIR}/${name}-mgrs-cleaned.onnx
    if [ ! -f "$IN_PATH" ]; then
        echo "[mgrs/$name] MISSING input: $IN_PATH — skip" | tee -a "$LOG"
        continue
    fi
    if [ -f "$OUT_PATH" ]; then
        echo "[mgrs/$name] exists: $OUT_PATH (delete to rebuild)" | tee -a "$LOG"
        continue
    fi
    echo "[mgrs/$name] surgery -> $OUT_PATH" | tee -a "$LOG"
    set +e
    PYTHONUNBUFFERED=1 python3 -m archproof.mgrs_llm_onnx_surgery \
        "$IN_PATH" "$OUT_PATH" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "[mgrs/$name] FAILED rc=$rc — continue to next" | tee -a "$LOG"
    fi
done
echo | tee -a "$LOG"

# =========================================================================
# Step 2: re-verify each cleaned ONNX
# =========================================================================
echo "===== Step 2: re-verify cleaned ONNX =====" | tee -a "$LOG"
rm -f "$OUT" "${OUT%.csv}_per_gate.csv"

# Header
echo "llm,verdict,epsilon,n_admitted,n_rescue_pre,n_rescue_payload,verify_sec,status" > "$OUT"

for name in "${LLMS[@]}"; do
    BD_DIR=benchmark/7b_onnx/${name}-backdoored
    CLEANED=${BD_DIR}/${name}-mgrs-cleaned.onnx
    if [ ! -f "$CLEANED" ]; then
        echo "[verify/$name] MISSING $CLEANED — skip" | tee -a "$LOG"
        continue
    fi
    echo "===== $name $(date +%H:%M:%S) =====" | tee -a "$LOG"
    free -h | head -2 | tee -a "$LOG"

    set +e
    PYTHONUNBUFFERED=1 /usr/bin/time -v -o "logs/mgrs_${name}_rss.txt" \
        python3 - <<PY 2>&1 | tee -a "$LOG"
import time, csv, sys
sys.path.insert(0, "$PWD")
from archproof.verify_phaseC import verify_model_phaseC
t0 = time.time()
r = verify_model_phaseC("$CLEANED")
dt = time.time() - t0
verdict = getattr(r, "verdict_phaseC", "ERROR")
eps = float(getattr(r, "epsilon_phaseC", 0.0))
n_adm = int(getattr(r, "n_admitted_phaseC", 0))
ges = getattr(r, "gate_epsilons", []) or []
n_rp = sum(1 for ge in ges if ge.get("rescue_pre"))
n_ry = sum(1 for ge in ges if ge.get("rescue_payload"))
print(f"[mgrs-verify/$name] -> v={verdict} eps={eps} n_admitted={n_adm} "
      f"rescue=(pre {n_rp},pay {n_ry}) {dt:.1f}s")
with open("$OUT", "a", newline="") as f:
    csv.writer(f).writerow(["$name", verdict, eps, n_adm, n_rp, n_ry,
                            f"{dt:.1f}", "ok"])
PY
    rc=$?
    set -e
    echo "  exit=$rc" | tee -a "$LOG"
    grep -E "Maximum resident|wall clock" "logs/mgrs_${name}_rss.txt" \
        2>/dev/null | tee -a "$LOG" || true
    sync
    echo | tee -a "$LOG"
done

# =========================================================================
# Summary
# =========================================================================
echo "===== MGRS verdict distribution =====" | tee -a "$LOG"
awk -F',' 'NR>1 {c[$2]++} END {for (k in c) printf "  %-32s %d\n", k, c[k]}' "$OUT" \
    | tee -a "$LOG"
echo "===== expected: all 5 -> add-DGP-CLASS-NEGATIVE ε=0 =====" | tee -a "$LOG"
echo "rows=$(($(wc -l < "$OUT") - 1))  csv=$OUT  log=$LOG" | tee -a "$LOG"

# Backup
TS=$(date +%Y%m%d_%H%M%S)
ARCHIVE=truth_source/_archive/mgrs_5LLM_${TS}
mkdir -p "$ARCHIVE"
cp "$OUT" "$ARCHIVE/" 2>/dev/null || true
cp archproof/mgrs_llm_onnx_surgery.py "$ARCHIVE/" 2>/dev/null || true
cp "$LOG" "$ARCHIVE/" 2>/dev/null || true
echo "backup -> $ARCHIVE" | tee -a "$LOG"
