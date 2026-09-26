#!/usr/bin/env bash
# Table 30 (tab:appx:critical2): 400 near-threshold order statistics
# Prints the reproduced values so you can compare them, by eye, to
# Table 30 in the paper PDF. Default = recompute from the archived
# record (CPU, seconds). RERUN=1 re-runs the experiment first (rec).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 30 (tab:appx:critical2): 400 near-threshold order statistics"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 30 and compare each value by eye."
echo "================================================================"
# rec: verified from the archived record (no standalone driver).
python verify/check_tables.py --only "tab:appx:critical2" --verbose
