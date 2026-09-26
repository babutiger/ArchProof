#!/usr/bin/env bash
# Table 31 (tab:appx:adaptive-obf): 88-instance adaptive obfuscation
# Prints the reproduced values so you can compare them, by eye, to
# Table 31 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 31 (tab:appx:adaptive-obf): 88-instance adaptive obfuscation"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 31 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  echo ">> reproducing from scratch (T1): build the 88 obfuscated graphs, then verify them:"
  echo "   python archproof/adaptive_graph_obfuscation.py  &&  python archproof/run_c4_panel.py --tag post_patch"
  bash "$(dirname "$0")/../scripts/00_build_benchmark_models.sh" >/dev/null 2>&1 || true
  PYTHONPATH="$PWD" python archproof/adaptive_graph_obfuscation.py   # writes /tmp/c4_obfuscated + benchmark/c4_obfuscation_index.json
  PYTHONPATH="$PWD" python archproof/run_c4_panel.py --tag post_patch  # -> benchmark/c4_panel_results_post_patch.json (what the checker reads)
  echo
fi
python verify/check_tables.py --only "tab:appx:adaptive-obf" --verbose
