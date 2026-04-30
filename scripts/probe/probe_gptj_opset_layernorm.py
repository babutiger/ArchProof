"""Trace GPT-J's ε divergence across opset 14/16/17/18 to its root cause.

Hypothesis: GPT-J uses LayerNorm (not RMSNorm). Native LayerNormalization
op was introduced in opset 17. At opset < 17, torch.onnx exports LayerNorm
as a *decomposed* graph (ReduceMean → Sub → Pow → ReduceMean → Sqrt →
Div → Mul → Add). The decomposed IBP path can give a slightly looser bound
than the native path — explaining the +0.45% drift on eps_phi/payload that
compounds to +0.9% on ε.

The other 4 LLMs (Yi/DeepSeek/Mistral/Qwen) use RMSNorm, which has NO native
ONNX op at any opset — always decomposed identically across opsets — hence
bit-exact Δ_T = 0.

For each opset in {14, 16, 17, 18}:
  1. count LayerNormalization native ops
  2. count decomposed-LayerNorm signature: ReduceMean, Sub, Pow nearby
  3. verify ε and compare

Output: scripts notes file + per-cell verdict comparison.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from collections import Counter
import json

warnings.filterwarnings("ignore")
os.environ.setdefault("ARCHPROOF_LARGE_MODEL_GB", "256")
os.environ.setdefault("ARCHPROOF_LLM_RESCUE", "1")
os.environ.setdefault("ARCHPROOF_PROBE_MAX_GB", "40")

import onnx
import torch

ROOT = Path(os.environ.get("ARCHPROOF_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from archproof.verify_phaseC import verify_model_phaseC

LLM = "gpt-j-6b"
ONNX_DIR = ROOT / "benchmark" / "7b_onnx" / "backdoored_onnx" / f"{LLM}-backdoored"
ONNX_PATH = ONNX_DIR / f"{LLM}-backdoored.onnx"


def opset_of(m: onnx.ModelProto) -> int:
    for opset in m.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    return -1


def op_counts(m: onnx.ModelProto) -> dict:
    return Counter(node.op_type for node in m.graph.node)


def main():
    if not ONNX_PATH.exists():
        print(f"[err] ONNX not found: {ONNX_PATH}")
        sys.exit(1)

    print("=" * 72)
    print(f"GPT-J ONNX op-type analysis : {ONNX_PATH}")
    print("=" * 72)

    m = onnx.load(str(ONNX_PATH), load_external_data=False)
    op = opset_of(m)
    counts = op_counts(m)
    print(f"  opset version           : {op}")
    print(f"  total nodes             : {sum(counts.values())}")
    print(f"  LayerNormalization      : {counts.get('LayerNormalization', 0)}")
    print(f"  ReduceMean              : {counts.get('ReduceMean', 0)}")
    print(f"  Pow                     : {counts.get('Pow', 0)}")
    print(f"  Sqrt                    : {counts.get('Sqrt', 0)}")
    print(f"  Sub                     : {counts.get('Sub', 0)}")
    print(f"  Mul                     : {counts.get('Mul', 0)}")
    print(f"  Sigmoid                 : {counts.get('Sigmoid', 0)}")
    print(f"  Tanh                    : {counts.get('Tanh', 0)}")
    print(f"  Top 12 op types         :")
    for op_type, n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {op_type:32s} {n:5d}")

    # Decomposed-LayerNorm signature: pairs of ReduceMean (mean + var),
    # one Pow + one Sqrt + one Div (variance computation), and Mul/Add
    # for affine. GPT-J should have ~28 LayerNorms (1 attn + 1 final per
    # of 28 transformer blocks) ≈ ~56 LayerNorms total. With native op,
    # we'd see ~56 LayerNormalization. With decomposed, we'd see ~112
    # ReduceMean (mean+var per LN), ~56 Pow, ~56 Sqrt.
    print()
    print("Interpretation:")
    if counts.get('LayerNormalization', 0) > 0:
        print(f"  → uses NATIVE LayerNormalization op "
              f"({counts['LayerNormalization']} instances)")
        print(f"  → opset must be ≥ 17 for export to emit it")
    elif counts.get('ReduceMean', 0) >= 50:
        print(f"  → uses DECOMPOSED LayerNorm "
              f"(ReduceMean count {counts['ReduceMean']} >> typical native count)")
        print(f"  → most likely opset < 17, OR torch.onnx forced decomposition")
    else:
        print(f"  → unclear — {counts.get('ReduceMean', 0)} ReduceMean, "
              f"{counts.get('LayerNormalization', 0)} LayerNormalization")

    print()
    print("=" * 72)
    print("Re-verify the existing ONNX twice (reproducibility test)")
    print("=" * 72)
    eps_runs = []
    for run in range(2):
        vr = verify_model_phaseC(str(ONNX_PATH))
        eps = vr.epsilon_phaseC
        eps_runs.append(eps)
        print(f"  run {run+1}: ε = {eps}  verdict = {vr.verdict_phaseC}  "
              f"n_adm = {vr.n_admitted_phaseC}")
    if eps_runs[0] == eps_runs[1]:
        print("  ✓ deterministic — same ε twice")
    else:
        diff = abs(eps_runs[0] - eps_runs[1])
        rel = diff / max(abs(eps_runs[0]), 1e-300)
        print(f"  ⚠ NON-DETERMINISTIC  abs={diff}  rel={rel:.3e}")

    # Save notes
    out = ROOT / "results" / "probe_gptj_opset_layernorm.json"
    with out.open("w") as f:
        json.dump({
            "onnx_path": str(ONNX_PATH),
            "opset_version": op,
            "op_counts": dict(counts),
            "epsilon_runs": eps_runs,
            "deterministic": eps_runs[0] == eps_runs[1] if len(eps_runs) >= 2 else None,
        }, f, indent=2)
    print()
    print(f"  log: {out}")


if __name__ == "__main__":
    main()
