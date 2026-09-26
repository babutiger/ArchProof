#!/usr/bin/env bash
# Create the conda environment the paper's appendix declares, alongside the
# existing one.
#
# The verifier's result depends on the ONNX graph the exporter emits. PyTorch
# 2.4 emits a different graph for the same model definition than PyTorch 2.1
# does (220 nodes instead of 146 on op_int_un, with long Identity chains), and
# the extra structure puts Sub and ReduceMin on the downstream path of admitted
# gates, which the chain-Lipschitz recursion cannot bound. The certificate then
# blows up and the verdict degrades to UNCERTIFIED. Reproducing the paper needs
# the exporter the paper used.
#
# This creates a separate environment and leaves the existing one untouched,
# so both remain available.
#
# Usage: bash scripts/setup_repro_env.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-archproof_repro}"

# Versions as declared in the paper's software-stack paragraph.
PY_VER=3.10
TORCH_VER=2.1.0
TORCHVISION_VER=0.16.0   # the release paired with torch 2.1.0
ONNX_VER=1.16.0
ORT_VER=1.18.0
NUMPY_VER=1.26.4
ONNXOPT_VER=0.3.13

# A non-interactive shell does not have conda on PATH, so locate its profile
# script directly before using it.
CONDA_SH=""
for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda; do
  [[ -f "$c/etc/profile.d/conda.sh" ]] && { CONDA_SH="$c/etc/profile.d/conda.sh"; break; }
done
[[ -n "$CONDA_SH" ]] || { echo "cannot find conda.sh" >&2; exit 1; }
source "$CONDA_SH"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "environment $ENV_NAME already exists; reusing it"
else
  echo "== creating $ENV_NAME (python $PY_VER) =="
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

conda activate "$ENV_NAME"

echo "== installing the declared stack =="
# CPU build: the admission probe and the IBP forward are CPU-only, so the CUDA
# wheels would only cost disk. The GPU is needed for the alpha,beta-CROWN
# comparison, which keeps using the existing environment.
# torchvision is needed by the benchmark's model definitions (the CIFAR data
# transforms they import), and its version is tied to the torch version.
#
# The torch and torchvision wheels only exist on download.pytorch.org, but that
# index does not carry their pure-Python dependencies, so pip falls back to
# PyPI for those. PyPI is reachable from these hosts at a few kB/s while a
# local mirror runs at a few MB/s, so the mirror is passed as an extra index
# and the two sources are used together.
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
pip install --no-cache-dir \
  "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url "$PIP_MIRROR"
pip install --no-cache-dir --index-url "$PIP_MIRROR" \
  "onnx==$ONNX_VER" "onnxruntime==$ORT_VER" "numpy==$NUMPY_VER" \
  "onnxoptimizer==$ONNXOPT_VER"

# Importing the benchmark's model definitions pulls in the training utilities
# they were written alongside, so these must be present even though the
# reproduction only exports and verifies.
pip install --no-cache-dir --index-url "$PIP_MIRROR" \
  pytorch_lightning matplotlib tqdm

# The scanner-scope appendix runs the deployed scanners against the same
# artifacts the verifier sees, so those tools have to be present.
pip install --no-cache-dir --index-url "$PIP_MIRROR" \
  "modelscan==0.8.6" "picklescan==1.0.4"

echo
echo "== installed =="
python - <<'PY'
import numpy, onnx, onnxruntime, torch
print(f"  python       {__import__('sys').version.split()[0]}")
print(f"  torch        {torch.__version__}")
print(f"  onnx         {onnx.__version__}")
print(f"  onnxruntime  {onnxruntime.__version__}")
print(f"  numpy        {numpy.__version__}")
PY

# The benchmark's CIFAR helper picks its data directory from
# torch.cuda.is_available(): with CUDA it reads the copy inside
# backdoor-taxonomy/data, without CUDA it expects ~/Documents/Code/data. This
# environment installs the CPU build, so point the second path at the same
# data rather than editing the code being reproduced.
CIFAR_SRC="${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/backdoor-taxonomy/data"
CIFAR_ALT="$HOME/Documents/Code/data"
if [[ -d "$CIFAR_SRC/cifar-10-batches-py" ]] && [[ ! -e "$CIFAR_ALT/cifar-10-batches-py" ]]; then
  mkdir -p "$CIFAR_ALT"
  ln -sfn "$CIFAR_SRC/cifar-10-batches-py" "$CIFAR_ALT/cifar-10-batches-py"
  [[ -f "$CIFAR_SRC/cifar-10-python.tar.gz" ]] &&
    ln -sfn "$CIFAR_SRC/cifar-10-python.tar.gz" "$CIFAR_ALT/cifar-10-python.tar.gz"
  echo "linked CIFAR-10 into $CIFAR_ALT for the CPU build"
fi

echo
echo "== exporter check: op_int_un should emit a 146-node graph =="
ROOT="${ARCHPROOF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
python - <<'PY'
import sys, os, torch, onnx
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/backdoor-taxonomy")
import backdoored_models as bm

torch.manual_seed(0)
m = bm.op_int_un_backdoor().eval()
out = "/tmp/op_int_un_repro_check.onnx"
torch.onnx.export(m, torch.randn(1, 3, 32, 32), out, opset_version=17,
                  do_constant_folding=False,
                  input_names=["input"], output_names=["output"])
n = len(onnx.load(out).graph.node)
print(f"  op_int_un graph nodes: {n} (the paper's record has 146)")
if n != 146:
    print("  WARNING: the exporter still does not match the recorded graph")
PY
echo
echo "activate with:  conda activate $ENV_NAME"
