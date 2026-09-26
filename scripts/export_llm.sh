#!/usr/bin/env bash
# Export the 5 whole-model 6-7B LLM ONNX (clean + backdoored) that the LLM
# tables verify. Run this ONCE, after downloading the base weights
# (docs/MODELS.md 3a); then verify the LLM tables the normal way.
#
# Needs a big host: the export peaks above 146 GB RAM (run with ~176 GB) and
# ~253 GB free disk. The export is CPU-fp32
# (the verifier's interval arithmetic is float32, and a 24 GB GPU OOMs tracing
# a 7B model), so it is slow -- hours -- but only has to be done once.
#
# After this finishes:
#   RERUN=1 bash reproduce/table04_llm-headline.sh     # verify a table from scratch
#   ... or any of tables 4, 5, 13, 14, 15, 41
set -euo pipefail
cd "$(dirname "$0")/.."

echo "================================================================"
echo ">> Exporting whole-model LLM ONNX (clean + backdoored), CPU fp32, opset 17."
echo ">> Base weights are read from models/<short>/ or the HuggingFace cache."
echo ">> If a model is missing, fetch the weights first:  bash scripts/download_llm.sh"
echo "================================================================"

DEVICE=cpu DTYPE=fp32 PYTHONPATH="$PWD" python scripts/export_llm_all.py

echo
echo "================================================================"
echo ">> Done. ONNX written under (exactly what the verifier reads):"
echo ">>   benchmark/7b_onnx/clean_onnx/<short>/<short>.onnx"
echo ">>   benchmark/7b_onnx/backdoored_onnx/<short>-backdoored/<short>-backdoored.onnx"
echo ">>"
echo ">> Now verify the LLM tables the normal way, e.g.:"
echo ">>   RERUN=1 bash reproduce/table04_llm-headline.sh"
echo ">>   RERUN=1 bash reproduce/table15_appx-llm-detail.sh"
echo "================================================================"
