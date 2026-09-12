#!/usr/bin/env bash
# Table 4 (tab:llm-headline): 5 whole-model LLM verified epsilon
# Prints the reproduced values so you can compare them, by eye, to
# Table 4 in the paper PDF. Default = read the bundled record. To re-run this LLM table from scratch you
# need the LLM tier (GPU + ~176 GB RAM + ~253 GB disk (export peak >146 GB; verify ~95 GB)). Once:
#   bash scripts/download_llm.sh  &&  bash scripts/export_llm.sh
# then:  RERUN=1 bash reproduce/<this-script>.sh
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 4 (tab:llm-headline): 5 whole-model LLM verified epsilon"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 4 and compare each value by eye."
echo "================================================================"
if [ "${RERUN:-0}" = "1" ]; then
  echo ">> [RERUN] re-running the whole-model LLM experiment from scratch (needs the LLM tier):"
  echo "   PYTHONPATH="$PWD" python scripts/repro_llm_verify_existing.py --only gpt-j-6b,yi-6b,deepseek-7b,mistral-7b,qwen2-7b"
  PYTHONPATH="$PWD" python scripts/repro_llm_verify_existing.py --only gpt-j-6b,yi-6b,deepseek-7b,mistral-7b,qwen2-7b
  echo
fi
python verify/check_tables.py --only "tab:llm-headline" --verbose
