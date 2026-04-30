"""Dump Qwen2's ONNX op topology AROUND the inject gate, both upstream
(for rescue diagnosis) and downstream (for L_post / chain Lipschitz
diagnosis). Same idea as inspect_gptj_gate_topology.py, but two-sided.

Run on B (graph-only load, < 1 GB RAM, ~30 s):
  python3 scripts/probe/inspect_qwen2_gate_topology.py > logs/qwen2_topology.txt
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = (ROOT / "benchmark" / "7b_onnx" /
             "qwen2-7b-backdoored" / "qwen2-7b-backdoored.onnx")


def main():
    if not ONNX_PATH.exists():
        print(f"missing: {ONNX_PATH}", file=sys.stderr); sys.exit(1)

    print(f"loading graph (no weights) from {ONNX_PATH} ...", flush=True)
    m = onnx.load(str(ONNX_PATH), load_external_data=False)
    g = m.graph
    print(f"  nodes={len(g.node)}  initializers={len(g.initializer)}",
          flush=True)

    produces, consumes = {}, {}
    for n in g.node:
        for o in n.output:
            if o:
                produces[o] = n
        for inp in n.input:
            consumes.setdefault(inp, []).append(n)

    init_shapes = {}
    for init in g.initializer:
        try:
            arr = numpy_helper.to_array(init)
            init_shapes[init.name] = (str(arr.dtype), tuple(arr.shape))
        except Exception:
            init_shapes[init.name] = ("?", ())

    # Find inject Relu.
    relu_node = None
    for n in g.node:
        if n.op_type == "Relu" and any("backdoor" in t.lower()
                                        for t in n.output):
            relu_node = n; break
    if relu_node is None:
        print("could not find /backdoor Relu", file=sys.stderr); sys.exit(1)

    pre_act = relu_node.input[0]
    relu_out = relu_node.output[0]
    print(f"\n=== inject Relu node: {relu_node.name} ===")
    print(f"  pre_act:    {pre_act}")
    print(f"  relu_out:   {relu_out}")

    # Find the Mul node consuming relu_out (inject's gate * payload).
    mul_node = None
    for cn in consumes.get(relu_out, []):
        if cn.op_type == "Mul":
            mul_node = cn; break
    if mul_node is not None:
        mul_out = mul_node.output[0]
        print(f"  inject Mul: {mul_node.name}  out={mul_out}")
        print(f"    inputs: {list(mul_node.input)}")
    else:
        mul_out = None
        print(f"  no Mul consuming relu_out (?)")

    # === UPSTREAM trace from pre_act (rescue diagnosis) ===
    print(f"\n=== UPSTREAM BFS from pre_act ({pre_act}) — for rescue ===")
    print(f"{'hop':>3} {'op':<24} {'tensor':<60} inputs")
    print("-" * 130)
    visited = set()
    queue: deque = deque()
    queue.append((pre_act, 0))
    MAX_HOPS = 25
    while queue:
        name, hops = queue.popleft()
        if name in visited or hops > MAX_HOPS:
            continue
        visited.add(name)
        prod = produces.get(name)
        if prod is None:
            init_info = init_shapes.get(name, "?")
            print(f"{hops:>3} {'<input/init>':<24} {name[:60]:<60} "
                  f"{init_info}")
            continue
        inp_summary = []
        for inp in prod.input:
            if inp in init_shapes:
                inp_summary.append(f"{inp[:36]}{init_shapes[inp]}")
            else:
                inp_summary.append(inp[:36])
        print(f"{hops:>3} {prod.op_type:<24} {name[:60]:<60} "
              f"<- {inp_summary}")
        for inp in prod.input:
            if inp:
                queue.append((inp, hops + 1))

    # === DOWNSTREAM trace from inject Mul output (L_post diagnosis) ===
    if mul_out is not None:
        print(f"\n=== DOWNSTREAM BFS from inject Mul output ({mul_out}) — "
              f"for L_post ===")
        print(f"{'hop':>3} {'op':<24} {'tensor':<60} consumers")
        print("-" * 130)
        visited = set()
        queue = deque()
        queue.append((mul_out, 0))
        MAX_HOPS = 30
        outputs_set = {o.name for o in g.output}
        while queue:
            name, hops = queue.popleft()
            if name in visited or hops > MAX_HOPS:
                continue
            visited.add(name)
            consumer_nodes = consumes.get(name, [])
            tag = " [GRAPH OUT]" if name in outputs_set else ""
            if not consumer_nodes:
                print(f"{hops:>3} {'<no consumer>':<24} "
                      f"{name[:60]:<60}{tag}")
                continue
            ops = [c.op_type for c in consumer_nodes]
            print(f"{hops:>3} {','.join(ops)[:24]:<24} "
                  f"{name[:60]:<60} consumers={[c.name[:30] for c in consumer_nodes]}{tag}")
            for c in consumer_nodes:
                for o in c.output:
                    if o:
                        queue.append((o, hops + 1))

    # Print backdoor Linear initializers shapes.
    print("\n=== backdoor Linear initializers (shape & first values) ===")
    for init in g.initializer:
        if "backdoor" in init.name.lower():
            arr = numpy_helper.to_array(init)
            print(f"  {init.name}  dtype={arr.dtype}  shape={arr.shape}")
            if arr.size <= 8:
                print(f"    values: {arr.flatten().tolist()}")
            else:
                print(f"    norm={float((arr**2).sum())**0.5:.3f}  "
                      f"max|.|={float(abs(arr).max()):.3f}  "
                      f"first 4: {arr.flatten()[:4].tolist()}")


if __name__ == "__main__":
    main()
