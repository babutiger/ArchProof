#!/usr/bin/env bash
# Table 5 (tab:llm-baseline): auto_LiRPA baseline fails on 5 LLMs
# Prints the reproduced values so you can compare them, by eye, to
# Table 5 in the paper PDF. Default = read the bundled record. To re-run this LLM table from scratch you
# need the LLM tier (GPU + ~176 GB RAM + ~253 GB disk (export peak >146 GB; verify ~95 GB)). Once:
#   bash scripts/download_llm.sh  &&  bash scripts/export_llm.sh
# then:  RERUN=1 bash reproduce/<this-script>.sh
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 5 (tab:llm-baseline): auto_LiRPA baseline fails on 5 LLMs"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 5 and compare each value by eye."
echo "================================================================"
if [ "${RERUN:-0}" = "1" ]; then
  echo ">> [RERUN] re-running the whole-model LLM experiment from scratch (needs the LLM tier):"
  echo "   bash scripts/run_baseline_oom.sh"
  bash scripts/run_baseline_oom.sh
  echo
fi
python verify/check_tables.py --only "tab:llm-baseline" --verbose
