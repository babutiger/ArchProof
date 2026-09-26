#!/usr/bin/env bash
# Potion case study (paper Appendix D.5.5): the ArchProof half of the live
# example, reproduced directly -- no 500-model Phase F open-world sweep needed.
#
# Protect AI's Guardian flagged the deployed minishlab/potion-base-8M as
# "suspicious" for an architectural backdoor (a finding its maintainers dispute,
# https://huggingface.co/minishlab/potion-base-8M/discussions/2). Run on the
# exact file, ArchProof returns a sound add-DGP-CLASS-NEGATIVE: the graph has no
# Mul, so no gate is admissible (n_admitted=0) and the model lies provably
# outside the certified class. A flag is not a sound verdict.
#
#   bash reproduce/potion_case_study.sh           # download ~58 MB + verify (~20 s)
#   QUICK=1 bash reproduce/potion_case_study.sh   # just read the archived record
#
# The full run needs the pinned env (conda activate archproof_repro) and
# network access to the HF hub or hf-mirror.com. QUICK=1 needs neither: it
# reads the verdict recorded by the paper's run in
# truth_source/per_model_phaseF.csv.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${QUICK:-0}" != "1" ]; then
  # Activate the pinned env if one is on the machine (same probe as run_phaseF.sh).
  CONDA_ENV="${ARCHPROOF_CONDA_ENV:-archproof_repro}"
  for _base in "$HOME/anaconda3" "$HOME/miniconda3" "$(conda info --base 2>/dev/null || true)"; do
    if [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "$_base/etc/profile.d/conda.sh"
      conda activate "$CONDA_ENV" 2>/dev/null || true
      break
    fi
  done
  export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
  # Only force an endpoint first if the caller set one (e.g. behind the GFW);
  # otherwise the Python tries the real HF hub, then hf-mirror.com.
  [ -n "${HF_ENDPOINT:-}" ] && export HF_ENDPOINT || true
fi

PYTHONPATH="$PWD" python scripts/potion_case_study.py
