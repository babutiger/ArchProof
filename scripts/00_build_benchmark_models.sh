#!/usr/bin/env bash
# Provide the 22 seeded backdoor graphs at /tmp/bober_onnx + /tmp/handcrafted_onnx,
# which the detection/robustness drivers read. The graphs are deterministic
# (torch.manual_seed(0)); this artifact ships the exact seeded build under
# models/backdoor_graphs/ (sha256-locked), so we populate the scratch dirs from
# there -- byte-identical to a fresh build, and dependency-free.
set -euo pipefail
AR="${ARCHPROOF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SRC="$AR/models/backdoor_graphs"
mkdir -p /tmp/bober_onnx /tmp/handcrafted_onnx
for f in "$SRC"/*.onnx; do
  b=$(basename "$f")
  case "$b" in
    H[0-9]*) cp -f "$f" /tmp/handcrafted_onnx/ ;;
    *)       cp -f "$f" /tmp/bober_onnx/ ;;
  esac
done
echo "provided: $(ls /tmp/bober_onnx/*.onnx 2>/dev/null | wc -l) bober + $(ls /tmp/handcrafted_onnx/*.onnx 2>/dev/null | wc -l) handcrafted graphs (from models/backdoor_graphs, sha256-locked)"
