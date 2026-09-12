#!/usr/bin/env bash
# Table 33 (tab:appx:cifar-e2e): CIFAR end-to-end
# Prints the reproduced values so you can compare them, by eye, to
# Table 33 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 33 (tab:appx:cifar-e2e): CIFAR end-to-end"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 33 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python archproof/run_v3_e2e_case_study.py"
  bash "$(dirname "$0")/../scripts/00_build_benchmark_models.sh" >/dev/null 2>&1 || true
  PYTHONPATH="$PWD" python archproof/run_v3_e2e_case_study.py
  echo
fi
python verify/check_tables.py --only "tab:appx:cifar-e2e" --verbose
