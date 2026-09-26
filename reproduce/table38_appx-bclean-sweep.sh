#!/usr/bin/env bash
# Table 38 (tab:appx:bclean-sweep): B_clean sweep
# Prints the reproduced values so you can compare them, by eye, to
# Table 38 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 38 (tab:appx:bclean-sweep): B_clean sweep"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 38 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python scripts/run_b_clean_sweep.py"
  PYTHONPATH="$PWD" python scripts/run_b_clean_sweep.py
  echo
fi
python verify/check_tables.py --only "tab:appx:bclean-sweep" --verbose
