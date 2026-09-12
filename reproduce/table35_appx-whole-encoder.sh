#!/usr/bin/env bash
# Table 35 (tab:appx:whole-encoder): Whole-encoder verification
# Prints the reproduced values so you can compare them, by eye, to
# Table 35 in the paper PDF. Default = read the bundled record. This table verifies 7 whole HuggingFace
# encoders (bert/distilbert/roberta/albert/xlnet/electra/deberta), NOT the 5
# big LLMs -- they auto-download via transformers, no download_llm.sh needed.
# RERUN=1 re-runs from scratch (heavier RAM; see docs/MODELS.md 2b).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 35 (tab:appx:whole-encoder): Whole-encoder verification"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 35 and compare each value by eye."
echo "================================================================"
if [ "${RERUN:-0}" = "1" ]; then
  echo ">> [RERUN] re-running the whole-model LLM experiment from scratch (needs the LLM tier):"
  echo "   bash scripts/run_phaseE_llm_clean.sh"
  bash scripts/run_phaseE_llm_clean.sh
  echo
fi
python verify/check_tables.py --only "tab:appx:whole-encoder" --verbose
