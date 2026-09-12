#!/usr/bin/env bash
# Reproduce the 35 one-machine CPU tables FROM SCRATCH.
#
# This re-runs each CPU table's experiment from zero, regenerates its record,
# and checks the freshly-computed numbers against the paper -- so the numbers
# you see are produced here, not read back from a bundled answer. It is slow:
# roughly 2-3 h on one CPU box (longer on a slow CPU). Several tables share one
# experiment driver, so each UNIQUE experiment is run only once and the tables
# that share it are then checked against that same fresh record (their values
# are still printed). The heaviest single runs -- whole-ResNet-18 interval
# propagation, the 68-config EIC sweep, the production-scale panel -- take tens
# of minutes each; many others finish in seconds.
#
# NOT re-run here (by design -- they are a different tier, not CPU-only):
#   * the 7 whole-model LLM tables  -> need a GPU + ~176 GB RAM + ~253 GB disk
#       (the ONNX export peaks above 146 GB; verification alone is ~95 GB).
#       run them from scratch with:
#         bash scripts/download_llm.sh
#         bash scripts/export_llm.sh
#         RERUN=1 bash reproduce/table04_llm-headline.sh   # (per LLM table)
#   * the 3 derived tables          -> aggregate counts with no standalone driver.
#
# To check ALL 45 tables' bundled records at once (no re-run, ~2 min):
#   make verify-quick        # == python verify/check_tables.py
set -uo pipefail
cd "$(dirname "$0")/.."

echo "================================================================"
echo "== Reproducing the 35 CPU tables FROM SCRATCH (no LLM tier).   =="
echo "== The 7 LLM + 3 derived tables are NOT re-run here; see the   =="
echo "== header of this script, or run 'make verify-quick' for all 45.=="
echo "================================================================"

# Run each UNIQUE experiment driver once; tables that share it are checked
# against that same fresh record instead of re-running the experiment.
declare -A ran
labels=""
for s in $(ls reproduce/table*.sh | sort); do
  # CPU (T1) tables re-run from scratch by default (they carry the QUICK gate).
  # LLM tables (RERUN gate) and derived tables (no gate) are skipped here.
  grep -q 'QUICK:-0.*!= "1"' "$s" || continue
  name=$(basename "$s")
  lbl=$(grep -oE 'check_tables.py --only "[^"]+"' "$s" | head -1 | grep -oE '"[^"]+"' | tr -d '"')
  [ -n "$lbl" ] && labels="${labels:+$labels,}$lbl"
  drv=$(grep -oE '(scripts|archproof)/[A-Za-z0-9_]+\.(py|sh)' "$s" | grep -vE '00_build|check_tables' | head -1)
  echo "================================================================"
  if [ -n "$drv" ] && [ -n "${ran[$drv]:-}" ]; then
    echo ">> $name -- its experiment ($(basename "$drv")) already ran above;"
    echo ">>   verifying this table against the paper from that fresh record:"
    python verify/check_tables.py --only "$lbl" --verbose || true
  else
    [ -n "$drv" ] && ran[$drv]=1
    echo "======== $name ========"
    bash "$s" || true
  fi
done

echo
echo "================================================================"
echo "== final summary: the CPU tables, recomputed from the freshly  =="
echo "== regenerated records (LLM/derived not included -- see above)  =="
echo "================================================================"
python verify/check_tables.py --only "$labels"
