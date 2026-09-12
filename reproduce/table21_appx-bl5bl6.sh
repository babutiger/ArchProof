#!/usr/bin/env bash
# Table 21 (tab:appx:bl5bl6): Ablation baselines BL5/BL6
# Prints the reproduced values so you can compare them, by eye, to
# Table 21 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 21 (tab:appx:bl5bl6): Ablation baselines BL5/BL6"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 21 and compare each value by eye."
echo "================================================================"
echo ">> NOTE: the BL6 row draws an UNSEEDED random panel, so on a fresh run its"
echo ">> count can move by +/-1 (verdicts and trend reproduce); the other rows are"
echo ">> value-exact. A small BL6 delta is expected, NOT a mismatch."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python archproof/run_bl5_bl6_baselines.py"
  bash "$(dirname "$0")/../scripts/00_build_benchmark_models.sh" >/dev/null 2>&1 || true
  PYTHONPATH="$PWD" python archproof/run_bl5_bl6_baselines.py
  echo
fi
python verify/check_tables.py --only "tab:appx:bl5bl6" --verbose
