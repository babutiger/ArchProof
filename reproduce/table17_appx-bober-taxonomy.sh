#!/usr/bin/env bash
# Table 17 (tab:appx:bober-taxonomy): Bober 12 gate types
# Prints the reproduced values so you can compare them, by eye, to
# Table 17 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 17 (tab:appx:bober-taxonomy): Bober 12 gate types"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 17 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python archproof/run_e1_final.py"
  bash "$(dirname "$0")/../scripts/00_build_benchmark_models.sh" >/dev/null 2>&1 || true
  PYTHONPATH="$PWD" python archproof/run_e1_final.py
  echo
fi
python verify/check_tables.py --only "tab:appx:bober-taxonomy" --verbose
