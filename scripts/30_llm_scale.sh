#!/usr/bin/env bash
# Whole-model 6-7B LLM verification (needs a GPU + ~95 GB RAM).
#
# These are the paper's RQ1 experiments. Each verification loads a 23-30 GB
# ONNX export in full and peaks at roughly three times the export size, so the
# host needs about 95 GB free; measured peaks are 71.06, 71.96, 83.79, 83.34
# and 93.50 GB for GPT-J, Yi, DeepSeek, Mistral and Qwen2. They run one at a
# time for that reason. The verifier is CPU-only: the GPU is used by the
# alpha,beta-CROWN comparison, not by ArchProof.
#
# Expect several hours. Each step writes its own record and keeps the previous
# one under _prev/ so the run can be compared rather than silently replacing
# the evidence.
#
# Usage: bash artifact/scripts/30_llm_scale.sh [step ...]
#        bash artifact/scripts/30_llm_scale.sh --list
set -uo pipefail

ROOT="${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PREV="${TMPDIR:-/tmp}/archproof_llm_prev"
LOGS="${TMPDIR:-/tmp}/archproof_llm_logs"
mkdir -p "$PREV" "$LOGS"
cd "$ROOT"

# step | script | record | paper tables it feeds
STEPS=(
  "whole_llm_eic|scripts/run_whole_llm_eic.sh|truth_source/per_cell_whole_llm_eic.csv|tab:llm-headline, tab:eic-summary"
  "phaseE_llm|scripts/run_phaseE_llm_clean.sh|truth_source/per_model_phaseE_llm_full_ibp.csv|tab:appx:llm-detail"
  "baseline_oom|scripts/run_baseline_oom.sh|truth_source/per_model_baseline_oom.csv|tab:llm-baseline"
  "llm_acpc|scripts/run_whole_llm_acpc.sh|truth_source/per_cell_whole_llm_acpc.csv|RQ3 LLM panel"
  "llm_drift|archproof/c3_llm_drift_measurement.py|truth_source/per_cell_c3_llm_drift.csv|tab:appx:llm-drift"
  "mgrs_5llm|scripts/run_mgrs_5llm.sh|truth_source/per_model_phaseE_mgrs_cleaned.csv|post-MGRS columns"
  "gptj_opcount|archproof/run_gptj_opcount.py|truth_source/per_model_gptj_opcount.csv|tab:appx:gptj-opcount"
)

if [[ "${1:-}" == "--list" ]]; then
  printf "%-16s %-52s %s\n" step script tables
  for e in "${STEPS[@]}"; do
    IFS='|' read -r n s r t <<<"$e"
    mark=" "; [[ -f "$s" ]] || mark="!"
    printf "%s%-15s %-52s %s\n" "$mark" "$n" "$s" "$t"
  done
  exit 0
fi

free_gb=$(free -g | awk 'NR==2{print $7}')
if (( free_gb < 95 )); then
  echo "only ${free_gb} GB free; the largest export peaks at 93.5 GB" >&2
  echo "close other work or run the steps individually" >&2
fi

want=("$@")
for e in "${STEPS[@]}"; do
  IFS='|' read -r name script record tables <<<"$e"
  if [[ ${#want[@]} -gt 0 ]] && [[ ! " ${want[*]} " =~ " $name " ]]; then continue; fi
  echo "== $name  ($tables) =="
  if [[ ! -f "$script" ]]; then echo "  SKIP: $script not found"; continue; fi
  [[ -f "$record" ]] && cp "$record" "$PREV/$(basename "$record")"
  start=$(date +%s)
  # Some steps are shell wrappers and some are python entry points.
  if [[ "$script" == *.py ]]; then
    runner=(python "$script")
  else
    runner=(bash "$script")
  fi
  if PYTHONPATH="$ROOT:$ROOT/backdoor-taxonomy:${PYTHONPATH:-}" "${runner[@]}" > "$LOGS/$name.log" 2>&1; then
    echo "  ok in $(( ($(date +%s) - start) / 60 )) min -> $record"
  else
    echo "  FAILED after $(( ($(date +%s) - start) / 60 )) min; see $LOGS/$name.log" >&2
    tail -5 "$LOGS/$name.log" >&2
  fi
done

echo
echo "each step's record is under truth_source/; the previous copy is kept in $PREV/"
