#!/usr/bin/env bash
# B-machine baseline OOM panel: attempt auto_LiRPA + α,β-CROWN on the 5
# backdoored 7B-LLM ONNX. Records the failure phase and peak RSS for each.
#
# Per-LLM 600s timeout (kill -9 if exceeded). Per-LLM /usr/bin/time -v
# captures peak RSS. Final CSV has 5 rows.
#
# Expected: 5/5 fail at one of the phases (onnx2pytorch unsupported op /
# BoundedModule OOM / CROWN OOM / timeout 600s). Provides baseline-failure
# evidence for paper §6.4.

set -uo pipefail
cd "${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOCK=/tmp/run_baseline.lock
exec 200>"$LOCK"
flock -n 200 || {
    echo "ERROR: another baseline_oom instance already running (lock=$LOCK)."
    exit 1
}

CONDA_ENV="${ARCHPROOF_CONDA_ENV:-alpha-beta-crown}"
TIMEOUT_S="${BASELINE_TIMEOUT_S:-600}"

for _b in "$HOME/anaconda3" "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
    [ -n "$_b" ] && [ -f "$_b/etc/profile.d/conda.sh" ] && { source "$_b/etc/profile.d/conda.sh"; break; }
done
conda activate "$CONDA_ENV"

mkdir -p truth_source logs
LOG=logs/baseline_oom_$(date +%Y%m%d_%H%M%S).log
OUT=truth_source/per_model_baseline_oom.csv

LLMS=(gpt-j-6b yi-6b deepseek-7b mistral-7b qwen2-7b)

echo "log=$LOG  csv=$OUT  timeout=${TIMEOUT_S}s" | tee -a "$LOG"
free -h | head -2 | tee -a "$LOG"
echo

# Header
echo "llm,phase_reached,fail_mode,wall_sec,exit_code,peak_rss_mb,error_msg" > "$OUT"

for name in "${LLMS[@]}"; do
    BD_DIR=benchmark/7b_onnx/backdoored_onnx/${name}-backdoored
    ONNX=${BD_DIR}/${name}-backdoored.onnx
    if [ ! -f "$ONNX" ]; then
        echo "[skip/$name] MISSING $ONNX" | tee -a "$LOG"
        continue
    fi
    echo "===== $name $(date +%H:%M:%S) =====" | tee -a "$LOG"
    free -h | head -2 | tee -a "$LOG"

    RSS_FILE=logs/baseline_${name}_rss.txt
    JSON_FILE=logs/baseline_${name}_result.json

    set +e
    timeout -s KILL "$TIMEOUT_S" \
        /usr/bin/time -v -o "$RSS_FILE" \
        python3 scripts/baseline_oom_attempt.py "$ONNX" \
            > "$JSON_FILE" 2>>"$LOG"
    rc=$?
    set -e

    # Parse the JSON result (last line of output)
    JSON_LINE=$(tail -1 "$JSON_FILE" 2>/dev/null)

    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        # Timeout (124) or SIGKILL (137 = OOM-killed)
        FAIL_MODE="timeout_or_oom_kill"
        PHASE_REACHED="killed"
        WALL="$TIMEOUT_S"
        ERR_MSG="exit=$rc (timeout=$TIMEOUT_S or kernel-OOM)"
    elif [ "$rc" -eq 0 ] || [ "$rc" -eq 1 ]; then
        # Script finished, parse JSON
        PHASE_REACHED=$(echo "$JSON_LINE" | python3 -c "import sys, json; print(json.loads(sys.stdin.read()).get('phase_reached', 'unknown'))" 2>/dev/null || echo "parse_fail")
        FAIL_MODE=$(echo "$JSON_LINE" | python3 -c "import sys, json; print(json.loads(sys.stdin.read()).get('fail_mode', 'unknown'))" 2>/dev/null || echo "parse_fail")
        WALL=$(echo "$JSON_LINE" | python3 -c "import sys, json; print(f\"{json.loads(sys.stdin.read()).get('wall_sec', 0):.1f}\")" 2>/dev/null || echo "0")
        ERR_MSG=$(echo "$JSON_LINE" | python3 -c "import sys, json; m = json.loads(sys.stdin.read()).get('error_msg', ''); print(m.replace(',', ';')[:200])" 2>/dev/null || echo "")
    else
        FAIL_MODE="proc_error"
        PHASE_REACHED="proc_died"
        WALL="0"
        ERR_MSG="exit=$rc"
    fi

    PEAK_RSS_KB=$(awk '/Maximum resident set size/ {print $NF}' "$RSS_FILE" 2>/dev/null || echo "0")
    PEAK_RSS_MB=$((PEAK_RSS_KB / 1024))

    # Print + append CSV
    echo "[baseline/$name] phase=$PHASE_REACHED fail_mode=$FAIL_MODE wall=${WALL}s rc=$rc rss=${PEAK_RSS_MB}MB" | tee -a "$LOG"
    if [ -n "$ERR_MSG" ]; then
        echo "  err: $ERR_MSG" | tee -a "$LOG"
    fi
    # CSV: escape commas in error message
    CSV_ERR=$(echo "$ERR_MSG" | tr ',\n' ';;')
    echo "$name,$PHASE_REACHED,$FAIL_MODE,$WALL,$rc,$PEAK_RSS_MB,\"$CSV_ERR\"" >> "$OUT"
    echo | tee -a "$LOG"
done

echo "===== Baseline phase distribution =====" | tee -a "$LOG"
awk -F',' 'NR>1 {c[$2]++} END {for(k in c) printf "  %-32s %d\n", k, c[k]}' "$OUT" \
    | tee -a "$LOG"
echo "===== Baseline fail_mode distribution =====" | tee -a "$LOG"
awk -F',' 'NR>1 {c[$3]++} END {for(k in c) printf "  %-32s %d\n", k, c[k]}' "$OUT" \
    | tee -a "$LOG"

# Backup
TS=$(date +%Y%m%d_%H%M%S)
ARCHIVE=truth_source/_archive/baseline_oom_${TS}
mkdir -p "$ARCHIVE"
cp "$OUT" "$ARCHIVE/" 2>/dev/null || true
cp scripts/baseline_oom_attempt.py "$ARCHIVE/" 2>/dev/null || true
cp "$LOG" "$ARCHIVE/" 2>/dev/null || true
cp logs/baseline_*_result.json logs/baseline_*_rss.txt "$ARCHIVE/" 2>/dev/null || true
echo "backup -> $ARCHIVE" | tee -a "$LOG"
