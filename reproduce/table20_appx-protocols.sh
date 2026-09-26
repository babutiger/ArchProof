#!/usr/bin/env bash
# Table 20 (tab:appx:protocols): Baseline protocols
# Prints the reproduced values so you can compare them, by eye, to
# Table 20 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 20 (tab:appx:protocols): Baseline protocols"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 20 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python scripts/run_rq2_phaseC_slice.py"
  PYTHONPATH="$PWD" python scripts/run_rq2_phaseC_slice.py
  echo
fi
python verify/check_tables.py --only "tab:appx:protocols" --verbose
