#!/usr/bin/env bash
# Create the baseline conda env for the prior-verifier comparison
# (auto_LiRPA + alpha-beta-CROWN). Kept separate from `archproof_repro` because
# it pins onnx 1.13.1 / onnx2pytorch 0.4.1, which conflict with the verifier's
# env. Used only by reproduce/table05_llm-baseline.sh (RERUN, LLM tier).
set -euo pipefail
cd "$(dirname "$0")"
ENV="${ABCROWN_ENV:-alpha-beta-crown}"
REQ="alpha-beta-CROWN/abcrown_requirements.txt"

for _b in "$HOME/anaconda3" "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
    [ -n "$_b" ] && [ -f "$_b/etc/profile.d/conda.sh" ] && { source "$_b/etc/profile.d/conda.sh"; break; }
done

if conda env list | grep -qE "^\s*${ENV}\s"; then
    echo ">> env '$ENV' already exists; updating deps"
else
    echo ">> creating conda env '$ENV' (python 3.9)"
    conda create -n "$ENV" python=3.9 -y
fi
echo ">> installing $REQ into '$ENV'"
conda run -n "$ENV" python -m pip install -r "$REQ"

echo ">> done. The baseline finds alpha-beta-CROWN via ALPHA_BETA_CROWN_PATH"
echo "   (default: $(pwd)/alpha-beta-CROWN). Run the baseline table with:"
echo "     RERUN=1 bash reproduce/table05_llm-baseline.sh"
