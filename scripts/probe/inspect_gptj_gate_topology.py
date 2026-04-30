"""Diagnose the ONNX op topology around the injected backdoor gate
in `gpt-j-6b-backdoored.onnx`. Writes a human-readable trace of
the upstream chain so the rescue module can be tuned to match it.

This is graph-only (no weight load), so RAM cost is < 1 GB.

Run on B:
  python3 scripts/probe/inspect_gptj_gate_topology.py \
      > logs/gptj_gate_topology.txt
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import deque

import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = (ROOT / "benchmark" / "7b_onnx" /
             "gpt-j-6b-backdoored" / "gpt-j-6b-backdoored.onnx")


def main():
    if not ONNX_PATH.exists():
        print(f"missing: {ONNX_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"loading graph-only from {ONNX_PATH} ...", flush=True)
    m = onnx.load(str(ONNX_PATH), load_external_data=False)
    g = m.graph
    print(f"  nodes: {len(g.node)}  initializers: {len(g.initializer)}",
          flush=True)

    produces = {}
    consumes = {}
    for n in g.node:
        for o in n.output:
            if o:
                produces[o] = n
        for inp in n.input:
            consumes.setdefault(inp, []).append(n)

    # initializer summary
    init_shapes = {}
    for init in g.initializer:
        try:
            arr = numpy_helper.to_array(init)
            init_shapes[init.name] = (str(arr.dtype), tuple(arr.shape))
        except Exception:
            init_shapes[init.name] = ("?", ())

    # Find the injected backdoor gate's pre-activation tensor.
    # Inject script names the ReLU `/backdoor/Relu` so its output is
    # `/backdoor/Relu_output_0`; pre-activation is the Add or MatMul
    # output that feeds it.
    relu_node = None
    for n in g.node:
        if n.op_type == "Relu" and any("backdoor" in t.lower()
                                        for t in n.output):
            relu_node = n
            break
    if relu_node is None:
        # Fallback: first Relu whose name mentions backdoor
        for n in g.node:
            if n.op_type == "Relu" and "backdoor" in (n.name or ""):
                relu_node = n
                break
    if relu_node is None:
        print("could not locate backdoor Relu node; listing all "
              "Relu ops:", flush=True)
        for n in g.node:
            if n.op_type == "Relu":
                print(f"  {n.name}  in={list(n.input)}  out={list(n.output)}")
        sys.exit(1)

    print(f"\n=== backdoor Relu node ===\n{relu_node.name}", flush=True)
    print(f"  inputs:  {list(relu_node.input)}")
    print(f"  outputs: {list(relu_node.output)}")

    pre_act = relu_node.input[0]
    print(f"\n=== pre-activation tensor: {pre_act} ===")

    # BFS upstream from pre_act, dump op chain up to N hops.
    MAX_HOPS = 30
    visited = set()
    queue: deque = deque()
    queue.append((pre_act, 0))
    print(f"\n=== upstream BFS (≤{MAX_HOPS} hops) ===")
    print(f"{'hop':>3} {'op':<22} {'tensor':<60} inputs")
    print("-" * 120)
    while queue:
        name, hops = queue.popleft()
        if name in visited or hops > MAX_HOPS:
            continue
        visited.add(name)
        prod = produces.get(name)
        if prod is None:
            init_info = init_shapes.get(name, "?")
            print(f"{hops:>3} {'<input/init>':<22} {name[:60]:<60} "
                  f"{init_info}")
            continue
        # Per-input init shapes for context.
        inp_summary = []
        for inp in prod.input:
            if inp in init_shapes:
                inp_summary.append(f"{inp[:40]}{init_shapes[inp]}")
            else:
                inp_summary.append(inp[:40])
        print(f"{hops:>3} {prod.op_type:<22} {name[:60]:<60} "
              f"<- {inp_summary}")
        for inp in prod.input:
            if inp:
                queue.append((inp, hops + 1))

    # Also dump info about the injected gate's Linear initializer.
    print("\n=== backdoor Linear(W, b) initializers ===")
    for init in g.initializer:
        if "backdoor" in init.name.lower() and "gate" in init.name.lower():
            arr = numpy_helper.to_array(init)
            print(f"  {init.name}  dtype={arr.dtype}  shape={arr.shape}")
            if arr.size <= 16:
                print(f"    values: {arr.flatten().tolist()}")
            else:
                print(f"    norm={float((arr**2).sum())**0.5:.3f}  "
                      f"max|.|={float(abs(arr).max()):.3f}")


if __name__ == "__main__":
    main()
