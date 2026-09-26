#!/usr/bin/env bash
# Table 13 (tab:appx:mistral): Mistral-7B SwiGLU fragment
# Prints the reproduced values so you can compare them, by eye, to
# Table 13 in the paper PDF. Default = read the bundled record. To re-run this LLM table from scratch you
# need the LLM tier (GPU + ~176 GB RAM + ~253 GB disk (export peak >146 GB; verify ~95 GB)). Once:
#   bash scripts/download_llm.sh  &&  bash scripts/export_llm.sh
# then:  RERUN=1 bash reproduce/<this-script>.sh
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 13 (tab:appx:mistral): Mistral-7B SwiGLU fragment"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 13 and compare each value by eye."
echo "================================================================"
if [ "${RERUN:-0}" = "1" ]; then
  echo ">> [RERUN] re-running the whole-model LLM experiment from scratch (needs the LLM tier):"
  echo "   PYTHONPATH="$PWD" python archproof/run_b_mistral_injection.py"
  PYTHONPATH="$PWD" python archproof/run_b_mistral_injection.py
  echo
fi
python verify/check_tables.py --only "tab:appx:mistral" --verbose
