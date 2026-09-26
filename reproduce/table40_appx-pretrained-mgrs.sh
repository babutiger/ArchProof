#!/usr/bin/env bash
# Table 40 (tab:appx:pretrained-mgrs): Pretrained-CNN MGRS
# Prints the reproduced values so you can compare them, by eye, to
# Table 40 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_HOME="${TORCH_HOME:-$PWD/models/torchvision_weights}"   # bundled backbone weights; downloads here if absent
echo "================================================================"
echo ">> Table 40 (tab:appx:pretrained-mgrs): Pretrained-CNN MGRS"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 40 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python scripts/run_pretrained_cnn_mgrs_phaseC.py"
  PYTHONPATH="$PWD" python scripts/run_pretrained_cnn_mgrs_phaseC.py
  echo
fi
python verify/check_tables.py --only "tab:appx:pretrained-mgrs" --verbose
