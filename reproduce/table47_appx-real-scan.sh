#!/usr/bin/env bash
# Table 47 (tab:appx:real-scan): 8 large clean models (torchvision + HF, downloaded)
# Prints the reproduced values so you can compare them, by eye, to
# Table 47 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_HOME="${TORCH_HOME:-$PWD/models/torchvision_weights}"   # bundled backbone weights; downloads here if absent
echo "================================================================"
echo ">> Table 47 (tab:appx:real-scan): 8 large clean models (torchvision + HF, downloaded)"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 47 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python archproof/run_e7_real_scan.py"
  PYTHONPATH="$PWD" python archproof/run_e7_real_scan.py
  echo
fi
python verify/check_tables.py --only "tab:appx:real-scan" --verbose
