#!/usr/bin/env bash
# Table 48 (tab:appx:scanner-scope): Scanner scope
# Prints the reproduced values so you can compare them, by eye, to
# Table 48 in the paper PDF. Default = reproduce from scratch by running the experiment (T1, CPU).
# QUICK=1 skips the run and just reads the bundled record instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "================================================================"
echo ">> Table 48 (tab:appx:scanner-scope): Scanner scope"
echo ">> Reproduced values are printed below. Open the paper PDF to"
echo ">> Table 48 and compare each value by eye."
echo "================================================================"
if [ "${QUICK:-0}" != "1" ]; then
  # This table contrasts ArchProof with two deployed model scanners, so the
  # re-run invokes their CLIs. Both ship with the pinned env (environment.yml);
  # on a bare env, install them first. The backdoor ONNX they scan is bundled
  # at benchmark/exporter_test/. If the CLIs are absent, skip the re-run
  # cleanly and fall through to the archived values rather than crashing.
  if command -v modelscan >/dev/null 2>&1 && command -v picklescan >/dev/null 2>&1; then
    echo ">> reproducing from scratch (T1): running the experiment, then checking it against the paper:"
    echo "   PYTHONPATH=\"$PWD\" python scripts/run_scanner_scope_study.py"
    bash "$(dirname "$0")/../scripts/00_build_benchmark_models.sh" >/dev/null 2>&1 || true
    PYTHONPATH="$PWD" python scripts/run_scanner_scope_study.py
    echo
  else
    echo ">> [RERUN] skipped: this table's re-run needs two scanner baselines"
    echo ">>   (modelscan, picklescan) that are not on PATH. Install them with:"
    echo ">>     pip install modelscan==0.8.6 picklescan==1.0.4"
    echo ">>   (both are already in the pinned env, environment.yml). The backdoor"
    echo ">>   ONNX they scan is bundled at benchmark/exporter_test/. Printing the"
    echo ">>   archived values below instead."
    echo
  fi
fi
python verify/check_tables.py --only "tab:appx:scanner-scope" --verbose
