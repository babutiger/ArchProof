#!/usr/bin/env bash
# Table 22 (tab:appx:witness): G2 witness oracle
# Prints the reproduced values so you can compare them, by eye, to
# Table 22 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 22 (tab:appx:witness): G2 witness oracle"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 22 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python archproof/gate_witness_oracle.py"
  bash "$(dirname "$0")/../scripts/00_build_benchmark_models.sh" >/dev/null 2>&1 || true
  PYTHONPATH="$PWD" python archproof/gate_witness_oracle.py
  echo
fi
python verify/check_tables.py --only "tab:appx:witness" --verbose
