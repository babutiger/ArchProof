#!/usr/bin/env bash
# Table 2 (tab:eic-break): EIC break cases (fail-closed)
# Prints the reproduced values so you can compare them, by eye, to
# Table 2 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 2 (tab:eic-break): EIC break cases (fail-closed)"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 2 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
  echo "   PYTHONPATH="$PWD" python archproof/run_r3_exporter_break.py"
  bash "$(dirname "$0")/../scripts/00_build_benchmark_models.sh" >/dev/null 2>&1 || true
  PYTHONPATH="$PWD" python archproof/run_r3_exporter_break.py
  echo
fi
python verify/check_tables.py --only "tab:eic-break" --verbose
