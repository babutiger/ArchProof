#!/usr/bin/env bash
# Table 6 (tab:verdict-dist): Out-of-class verdict distribution
# Prints the reproduced values so you can compare them, by eye, to
# Table 6 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 6 (tab:verdict-dist): Out-of-class verdict distribution"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 6 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python archproof/run_e2_expanded.py"
  bash "$(dirname "$0")/../scripts/00_build_benchmark_models.sh" >/dev/null 2>&1 || true
  PYTHONPATH="$PWD" python archproof/run_e2_expanded.py
  echo
fi
python verify/check_tables.py --only "tab:verdict-dist" --verbose
