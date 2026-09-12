#!/usr/bin/env bash
# Table 49 (tab:appx:openworld): Open-world 500 streaming (downloads from HF)
# Prints the reproduced values so you can compare them, by eye, to
# Table 49 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 49 (tab:appx:openworld): Open-world 500 streaming (downloads from HF)"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 49 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   bash scripts/run_phaseF.sh"
  bash scripts/run_phaseF.sh
  echo
fi
python verify/check_tables.py --only "tab:appx:openworld" --verbose
