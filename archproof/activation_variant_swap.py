"""Graph-level activation variant generator: produce <llm>-backdoored-<act>.onnx
variants by swapping the `/backdoor/Relu` op type to a different activation,
WITHOUT re-exporting the 23 GB ONNX from PyTorch.

The cleaned variant graph reuses the original `<llm>-backdoored.onnx`'s
external-data weight files (placed in the same directory), so each new
variant adds only ~10 MB of disk (graph topology), not 23 GB.

Supported variants (all dormant when gate.bias = -10):
  - Sigmoid       : Sigmoid(-10) ≈ 4.5e-5
  - HardSwish     : HardSwish(-10) = -10 · ReLU6(-10+3)/6 = 0
  - Softplus      : Softplus(-10) ≈ 4.5e-5

Usage:
    from archproof.activation_variant_swap import swap_gate_activation
    swap_gate_activation(
        in_path="benchmark/7b_onnx/yi-6b-backdoored/yi-6b-backdoored.onnx",
        out_path="benchmark/7b_onnx/yi-6b-backdoored/yi-6b-backdoored-sigmoid.onnx",
        new_op_type="Sigmoid")
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx

# Activations supported by both ONNX opset 17 AND archproof.verify_phaseC.
SUPPORTED_VARIANTS = {"Sigmoid", "HardSwish", "Softplus"}


def swap_gate_activation(in_path: str, out_path: str,
                         new_op_type: str,
                         gate_node_name: str = "/backdoor/Relu",
                         verbose: bool = True) -> dict:
    """Swap the gate's activation op type. Returns a result dict."""
    if new_op_type not in SUPPORTED_VARIANTS:
        return {"ok": False,
                "reason": f"unsupported new_op_type={new_op_type}; "
                          f"supported: {SUPPORTED_VARIANTS}"}

    if verbose:
        print(f"[swap] loading {in_path} (graph only)")
    model = onnx.load(in_path, load_external_data=False)
    g = model.graph

    # Locate the gate Relu node by name.
    target = None
    for n in g.node:
        if (n.name or "") == gate_node_name:
            target = n; break
    # Fallback: any Relu in /backdoor/.
    if target is None:
        for n in g.node:
            if n.op_type == "Relu" and "/backdoor/" in (n.name or ""):
                target = n; break
    if target is None:
        return {"ok": False, "reason": f"gate node '{gate_node_name}' not found"}

    if target.op_type != "Relu":
        return {"ok": False,
                "reason": f"target node op_type={target.op_type}, expected Relu"}

    if verbose:
        print(f"[swap] target node: {target.name}  Relu -> {new_op_type}")
        print(f"       inputs: {list(target.input)}")
        print(f"       outputs: {list(target.output)}")

    # Single-line surgery: change op_type. Inputs/outputs/name preserved.
    target.op_type = new_op_type

    # HardSwish has no attributes. Sigmoid has none. Softplus has none in
    # opset 17 (the optional `beta` was removed in opset 1's spec). Verify
    # by clearing attributes if any (defensive).
    while target.attribute:
        target.attribute.pop()

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_p))
    if verbose:
        print(f"[swap] saved -> {out_path}")
    return {"ok": True, "in_path": in_path, "out_path": out_path,
            "swapped_to": new_op_type, "node_name": target.name}


def main():
    if len(sys.argv) != 4:
        print(f"usage: python3 {sys.argv[0]} <in.onnx> <out.onnx> <activation>")
        print(f"  activation in {SUPPORTED_VARIANTS}")
        sys.exit(1)
    res = swap_gate_activation(sys.argv[1], sys.argv[2], sys.argv[3])
    print(res)
    sys.exit(0 if res.get("ok") else 2)


if __name__ == "__main__":
    main()
