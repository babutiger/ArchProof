#!/usr/bin/env bash
# Download the 5 base 6-7B LLM checkpoints the LLM tables need, from HuggingFace.
#
# The LLM tier is three simple steps:
#   1. bash scripts/download_llm.sh        # <- you are here: fetch the weights
#   2. bash scripts/export_llm.sh          # export clean + backdoored ONNX
#   3. RERUN=1 bash reproduce/tableNN.sh   # verify each LLM table, one by one
#
# ~90 GB of weights. Mistral is gated: run `huggingface-cli login` once and
# accept its licence at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3.
# On a network-restricted host:  export HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail
cd "$(dirname "$0")/.."

pip show huggingface_hub >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"

MODELS=(
  EleutherAI/gpt-j-6b
  01-ai/Yi-6B
  deepseek-ai/deepseek-llm-7b-base
  mistralai/Mistral-7B-Instruct-v0.3
  Qwen/Qwen2-7B-Instruct
)

for id in "${MODELS[@]}"; do
  echo ">> downloading $id"
  huggingface-cli download "$id"
done

echo
echo ">> all 5 base checkpoints downloaded (into the HuggingFace cache)."
echo ">> next: bash scripts/export_llm.sh"
