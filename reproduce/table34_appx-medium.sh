#!/usr/bin/env bash
# Table 34 (tab:appx:medium): 3 HF encoders (downloaded from HuggingFace)
# Prints the reproduced values so you can compare them, by eye, to
# Table 34 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 34 (tab:appx:medium): 3 HF encoders (downloaded from HuggingFace)"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 34 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python archproof/run_e5a_medium.py"
  PYTHONPATH="$PWD" python archproof/run_e5a_medium.py
  echo
fi
python verify/check_tables.py --only "tab:appx:medium" --verbose
