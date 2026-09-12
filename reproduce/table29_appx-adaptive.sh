#!/usr/bin/env bash
# Table 29 (tab:appx:adaptive): Near-threshold adaptive
# Prints the reproduced values so you can compare them, by eye, to
# Table 29 in the paper PDF. Default = recompute from the archived
# record (CPU, seconds). RERUN=1 re-runs the experiment first (rec).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 29 (tab:appx:adaptive): Near-threshold adaptive"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 29 and compare each value by eye."
echo "================================================================"
# rec: verified from the archived record (no standalone driver).
python verify/check_tables.py --only "tab:appx:adaptive" --verbose
