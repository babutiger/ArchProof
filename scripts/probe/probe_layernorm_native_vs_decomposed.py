"""Controlled experiment: export the same PyTorch LayerNorm at opset 14 vs
opset 17, dump op-types, then run archproof IBP on each and measure the
ε-tightness gap.

Hypothesis to test: the 0.9% rel ε divergence on GPT-J across
opset 14/16 vs 17/18 is due to torch.onnx emitting:
  - opset 17+: 1 native `LayerNormalization` op → archproof uses closed-form
    sqrt(d-1) bound (interval_propagation.py:748) → tight
  - opset 14/16: decomposed sequence (ReduceMean + Sub + Pow + ReduceMean +
    Sqrt + Div + Mul + Add) → archproof propagates each op via interval
    arithmetic → looser

Output: prints the op breakdown + IBP bound on the LN output for each
opset. If decomposed is 0.5%-1% looser per LayerNorm site, ~28 LN sites
in GPT-J compound to ~0.9%-1% on final ε.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn

ROOT = Path(os.environ.get("ARCHPROOF_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT))

from archproof.interval_propagation import propagate_intervals


class TinyLN(nn.Module):
    """Single LayerNorm over hidden=4096 (matching GPT-J)."""
    def __init__(self, hidden: int = 4096):
        super().__init__()
        self.ln = nn.LayerNorm(hidden, eps=1e-5)
        # Initialize to non-trivial scale + bias so closed-form bound is
        # not trivially 0.
        nn.init.normal_(self.ln.weight, mean=1.0, std=0.01)
        nn.init.zeros_(self.ln.bias)

    def forward(self, x):
        return self.ln(x)


def export_at_opset(opset: int, path: str):
    torch.manual_seed(0)
    m = TinyLN(hidden=4096).eval()
    ex = torch.randn(1, 16, 4096)  # (batch=1, seq=16, hidden=4096)
    torch.onnx.export(m, ex, path, opset_version=opset,
                      do_constant_folding=False,
                      input_names=["input"], output_names=["output"])
    return onnx.load(path)


def op_breakdown(m: onnx.ModelProto) -> dict:
    return Counter(n.op_type for n in m.graph.node)


def ibp_eps_at_output(m: onnx.ModelProto, x_lb_val=-1.0, x_ub_val=1.0) -> float:
    """Run archproof IBP on the LN graph; return the max-magnitude bound
    over the output tensor."""
    inp_name = m.graph.input[0].name
    shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    inp_lb = np.full(shape, x_lb_val, dtype=np.float32)
    inp_ub = np.full(shape, x_ub_val, dtype=np.float32)
    bounds = propagate_intervals(m, inp_lb, inp_ub)
    out_name = m.graph.output[0].name
    b = bounds.get(out_name)
    if b is None:
        return float("inf")
    return float(np.maximum(np.abs(b.lb), np.abs(b.ub)).max())


def main():
    tmp = Path("/tmp/_ln_native_vs_decomposed")
    tmp.mkdir(exist_ok=True)

    print("=" * 76)
    print("Controlled experiment: LayerNorm native (opset 17) vs decomposed (opset 14)")
    print("=" * 76)

    p17 = tmp / "ln_opset17.onnx"
    p14 = tmp / "ln_opset14.onnx"

    print("\n[export] opset 17 ...")
    m17 = export_at_opset(17, str(p17))
    ops17 = op_breakdown(m17)

    print("[export] opset 14 ...")
    m14 = export_at_opset(14, str(p14))
    ops14 = op_breakdown(m14)

    print()
    print(f"{'op type':30s} {'opset 17':>10s} {'opset 14':>10s}")
    all_ops = set(ops17) | set(ops14)
    for op in sorted(all_ops):
        print(f"  {op:28s} {ops17.get(op, 0):>10d} {ops14.get(op, 0):>10d}")

    print()
    has_native_17 = ops17.get("LayerNormalization", 0) > 0
    has_native_14 = ops14.get("LayerNormalization", 0) > 0
    print(f"opset 17 has native LayerNormalization? {has_native_17}")
    print(f"opset 14 has native LayerNormalization? {has_native_14}")

    print()
    print("[IBP] running archproof interval_propagation on each ...")
    eps17 = ibp_eps_at_output(m17)
    eps14 = ibp_eps_at_output(m14)
    print(f"  opset 17 (native LN)     : output max-magnitude bound = {eps17:.6e}")
    print(f"  opset 14 (decomposed LN) : output max-magnitude bound = {eps14:.6e}")

    if eps17 == 0 or not np.isfinite(eps17) or not np.isfinite(eps14):
        print("  ⚠ One bound is degenerate; cannot compute ratio.")
        return

    rel = (eps14 - eps17) / eps17
    print()
    print(f"  rel(decomposed - native) / native = {rel*100:+.4f}%")
    print(f"  decomposed / native ratio          = {eps14 / eps17:.6f}")

    print()
    if eps14 > eps17:
        print(f"✓ Decomposed IBP IS LOOSER than native — by {rel*100:+.2f}%")
        print(f"  Compounded over GPT-J's 28 LayerNorm sites, this gives roughly")
        print(f"    (1 + {rel:+.4f}) ^ 28 - 1 ≈ {(1+rel)**28 - 1:+.4f} = {((1+rel)**28-1)*100:+.2f}%")
        print(f"  which is consistent with the observed 0.9% rel ε divergence on GPT-J.")
    else:
        print(f"⚠ Decomposed IBP is NOT looser than native — hypothesis invalidated")
        print(f"  Need another explanation for GPT-J's 0.9% rel divergence.")

    p17.unlink(missing_ok=True)
    p14.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
