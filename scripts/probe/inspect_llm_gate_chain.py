"""Diagnose why llm_gate_local_bound rescue fails on a given LLM
backdoored ONNX.

Path resolution (in order):
  1. If sys.argv[2] is given, use it as the direct ONNX path.
  2. Else read results/per_model_phaseE_llm_full_ibp.csv and find the
     row with llm=<name> and gate_type=backdoored, use its onnx_path column.
  3. Else search benchmark/**/*.onnx for files matching <name>-backdoored.

Usage:
  python3 scripts/probe/inspect_llm_gate_chain.py yi-6b
  python3 scripts/probe/inspect_llm_gate_chain.py yi-6b /full/path/to/yi-6b-backdoored.onnx

Graph-only (no weights), ~3-5 s, no GPU. Safe to run in parallel with the
main verify on B.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent

LAYOUT_OPS = {"Transpose", "Reshape", "Squeeze", "Unsqueeze", "Cast", "Identity"}
LINF_PRESERVING = {"Relu", "LeakyRelu", "Sigmoid", "Tanh", "Softplus",
                   "Erf", "Cast", "Identity", "Slice", "Gather", "Tile",
                   "Squeeze", "Unsqueeze", "Reshape", "Transpose"}
NORM_BFS_OPS = {"Div", "Sub", "Add", "Mul", "Pow", "ReduceMean",
                "Cast", "Identity", "Reshape", "Reciprocal", "Sqrt"}
LAYERNORM_OPS = {"LayerNormalization", "RMSNormalization",
                 "BatchNormalization", "SimplifiedLayerNormalization"}


# ---------------------------------------------------------------------- path

def resolve_path(short: str, override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            print(f"override path does not exist: {p}")
            sys.exit(1)
        return p

    # Try CSV
    csv_path = ROOT / "results" / "per_model_phaseE_llm_full_ibp.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r.get("llm") == short and r.get("gate_type") == "backdoored":
                onnx_path = r.get("onnx_path", "").strip()
                if onnx_path and Path(onnx_path).exists():
                    print(f"resolved via CSV: {onnx_path}")
                    return Path(onnx_path)

    # Try expected directory layout
    for candidate in [
        ROOT / "benchmark" / "7b_onnx" / f"{short}-backdoored" / f"{short}-backdoored.onnx",
        ROOT / "benchmark" / "7b_onnx_injected" / short / f"{short}.onnx",
    ]:
        if candidate.exists():
            print(f"resolved via standard path: {candidate}")
            return candidate

    # Glob search
    print(f"glob-searching benchmark/ for *{short}*backdoored*.onnx ...")
    matches = list((ROOT / "benchmark").rglob(f"*{short}*backdoored*.onnx"))
    if not matches:
        matches = list((ROOT / "benchmark").rglob(f"*{short}*.onnx"))
    if matches:
        print(f"  found {len(matches)} match(es), picking first:")
        for m in matches[:5]:
            print(f"    {m}")
        return matches[0]

    print(f"NOTHING FOUND for short='{short}'")
    print(f"benchmark/ top-level dirs:")
    for d in sorted((ROOT / "benchmark").glob("*")):
        if d.is_dir():
            print(f"  {d.relative_to(ROOT)}/")
    sys.exit(1)


# ---------------------------------------------------------------------- helpers

def find_gate_relu(g):
    for n in g.node:
        if n.op_type == "Relu" and "/backdoor/" in (n.name or ""):
            return n
    for n in g.node:
        if n.op_type == "Relu" and any("backdoor" in (o or "").lower() for o in n.output):
            return n
    return None


def init_to_arr(g, name):
    """Return numpy array for initializer `name` ONLY if its data is
    inline (raw_data or typed *_data). Returns None for external-data
    initializers — diagnostic should use init_dims() for shape-only
    checks instead."""
    for init in g.initializer:
        if init.name == name:
            # Check for external data (skip those — caller must use init_dims)
            if any(d.key == "location" for d in init.external_data):
                return None
            try:
                return numpy_helper.to_array(init)
            except Exception:
                return None
    return None


def init_dims(g, name):
    """Return tuple(shape) for initializer `name`, or None if not found.
    Works for both inline and external-data initializers (just metadata)."""
    for init in g.initializer:
        if init.name == name:
            return tuple(init.dims)
    return None


def is_initializer(g, name):
    return any(init.name == name for init in g.initializer)


def resolve_scalar(g, produces, name):
    """Get scalar value of `name` whether stored as initializer or as
    Constant-node attribute. Returns ndarray or None."""
    arr = init_to_arr(g, name)
    if arr is not None:
        return arr
    prod = produces.get(name)
    if prod is None or prod.op_type != "Constant":
        return None
    for attr in prod.attribute:
        if attr.name == "value":
            try:
                return numpy_helper.to_array(attr.t)
            except Exception:
                return None
    return None


def resolve_const_via_layout(g, produces, tensor, max_hops=4):
    """Return ndarray if `tensor` resolves to an initializer through
    Transpose/Reshape/Squeeze/Unsqueeze/Cast/Identity chain; else None.
    Applies the layout transforms."""
    cur = tensor
    for _ in range(max_hops + 1):
        arr = init_to_arr(g, cur)
        if arr is not None:
            return arr, cur
        prod = produces.get(cur)
        if prod is None or prod.op_type not in LAYOUT_OPS:
            return None, None
        upstream = init_to_arr(g, prod.input[0]) if prod.input else None
        if upstream is None:
            cur = prod.input[0] if prod.input else None
            if cur is None:
                return None, None
            continue
        if prod.op_type == "Transpose":
            perm = next((list(a.ints) for a in prod.attribute if a.name == "perm"), None)
            return (upstream.T if perm is None and upstream.ndim == 2 else
                    upstream.transpose(perm) if perm else upstream), prod.input[0]
        return upstream, prod.input[0]
    return None, None


# ---------------------------------------------------------------------- main

def main():
    if len(sys.argv) < 2:
        print("usage: python3 scripts/probe/inspect_llm_gate_chain.py <short> [path]")
        sys.exit(1)
    short = sys.argv[1]
    override = sys.argv[2] if len(sys.argv) >= 3 else None

    onnx_path = resolve_path(short, override)
    print(f"\nloading graph (no external weights): {onnx_path}")
    m = onnx.load(str(onnx_path), load_external_data=False)
    g = m.graph
    print(f"  total nodes: {len(g.node)}  initializers: {len(g.initializer)}")

    produces = {o: n for n in g.node for o in n.output if o}
    init_names = {init.name for init in g.initializer}

    # ============================================================ STEP A
    print(f"\n=== A. Find gate Relu and pre-act tensor ===")
    relu = find_gate_relu(g)
    if relu is None:
        print("FATAL: no /backdoor/.../Relu found")
        relus = [n for n in g.node if n.op_type == "Relu"][:10]
        print(f"  first 10 Relu nodes in graph:")
        for n in relus:
            print(f"    name={n.name!r}  out={list(n.output)}")
        sys.exit(1)
    print(f"  Relu name: {relu.name}")
    print(f"  Relu output: {list(relu.output)}")
    pre_act = relu.input[0]
    print(f"  pre-act tensor: {pre_act}")

    pre_prod = produces.get(pre_act)
    if pre_prod is None:
        print(f"FATAL: pre-act has no producer (graph input?)")
        sys.exit(1)
    print(f"  pre-act producer: {pre_prod.op_type}  name={pre_prod.name}")
    print(f"    inputs: {list(pre_prod.input)}")

    # ============================================================ STEP B
    print(f"\n=== B. Resolve gate matmul (W and x) ===")
    matmul = pre_prod
    if pre_prod.op_type == "Add":
        for i, inp in enumerate(pre_prod.input):
            sub = produces.get(inp)
            if sub is not None and sub.op_type in ("MatMul", "Gemm"):
                matmul = sub
                bias_inp = pre_prod.input[1 - i] if len(pre_prod.input) == 2 else None
                bias_dims = init_dims(g, bias_inp) if bias_inp else None
                print(f"  pre-act = Add(MatMul, bias)")
                print(f"    bias: {bias_inp}  shape={bias_dims}")
                break
        else:
            print(f"  pre-act = Add but neither operand is MatMul/Gemm — non-Linear gate?")
            return

    if matmul.op_type not in ("MatMul", "Gemm"):
        print(f"  matmul.op_type = {matmul.op_type} (not MatMul/Gemm) — rescue won't apply")
        return

    print(f"  matmul: {matmul.op_type}  inputs: {list(matmul.input)}")
    W_dims = None
    W_via = None
    x_tensor = None
    for inp in matmul.input:
        # Try direct init by SHAPE only (no data read)
        dims = init_dims(g, inp)
        if dims is not None and len(dims) == 2 and W_dims is None:
            W_dims = dims; W_via = f"direct init '{inp}'"; continue
        # Try layout-resolve via shape (chain ends in init with 2-D shape)
        if W_dims is None and not is_initializer(g, inp):
            cur = inp
            for _ in range(5):
                p = produces.get(cur)
                if p is None: break
                if p.op_type not in LAYOUT_OPS: break
                up = p.input[0] if p.input else None
                if up is None: break
                up_dims = init_dims(g, up)
                if up_dims is not None and len(up_dims) == 2:
                    W_dims = up_dims; W_via = f"layout-chain from '{inp}' (init={up} via {p.op_type})"
                    break
                cur = up
            if W_dims is not None: continue
        # Else this is x
        if x_tensor is None:
            x_tensor = inp

    if W_dims is None:
        print(f"  FATAL: W could not be resolved from any matmul input")
        for inp in matmul.input:
            dims = init_dims(g, inp)
            print(f"    input '{inp}': init={dims is not None}  "
                  f"shape={dims}  "
                  f"producer={produces.get(inp).op_type if produces.get(inp) else None}")
        return
    print(f"  W: shape={W_dims}  via={W_via}")
    print(f"  x_tensor: {x_tensor}")

    if x_tensor is None:
        print(f"  FATAL: no x_tensor identified")
        return

    # ============================================================ STEP C
    print(f"\n=== C. BFS upstream from x_tensor (mimics _trace_input_bound) ===")
    print(f"    Stop conditions:")
    print(f"      - hits LayerNorm/RMSNorm fused op  -> SUCCESS (path A)")
    print(f"      - hits Mul with 1-D γ initializer  -> SUCCESS (path B candidate)")
    print(f"      - hits MatMul/Gemm/Div             -> DIES")
    print(f"      - hits Mul without 1-D γ           -> DIES")
    print(f"      - hits unknown op                  -> DIES (not recursed)")

    visited = set()
    queue = [(x_tensor, 0, "start")]
    chain = []
    success_kind = None
    success_node = None
    death_reasons = []
    MAX_HOPS = 12

    while queue:
        name, hops, why = queue.pop(0)
        if name in visited or hops > MAX_HOPS:
            continue
        visited.add(name)
        prod = produces.get(name)
        if prod is None:
            chain.append((hops, "GRAPH_INPUT_OR_INIT", name[:60], why, "stop"))
            continue
        chain.append((hops, prod.op_type, name[:60], why, ""))

        if prod.op_type in LAYERNORM_OPS:
            success_kind = f"FUSED {prod.op_type}"; success_node = prod; break

        if prod.op_type == "Mul":
            # Path B candidate? Check by SHAPE (γ is external data on LLMs)
            gamma_inp = None
            for inp in prod.input:
                dims = init_dims(g, inp)
                if dims is not None and len(dims) == 1 and dims[0] >= 2:
                    gamma_inp = inp; break
            if gamma_inp is not None:
                D = init_dims(g, gamma_inp)[0]
                success_kind = f"γ-scale Mul (D={D})"
                success_node = prod
                break
            else:
                death_reasons.append(f"hop {hops}: Mul without 1-D γ -> walk dies here")
                continue  # don't recurse past Mul

        if prod.op_type in LINF_PRESERVING or prod.op_type in ("Add", "Sub"):
            for inp in prod.input:
                if inp:
                    queue.append((inp, hops + 1, f"<-{prod.op_type}"))
        elif prod.op_type in ("MatMul", "Gemm"):
            death_reasons.append(f"hop {hops}: hit MatMul/Gemm -> walk skips (not recursed)")
        elif prod.op_type == "Div":
            death_reasons.append(f"hop {hops}: hit Div -> walk dies here")
        else:
            death_reasons.append(f"hop {hops}: unknown op '{prod.op_type}' -> walk dies here")

    print(f"\n    Chain (first 30 entries, hop, op, tensor, why-enqueued):")
    for h, op, t, why, _ in chain[:30]:
        print(f"      hop={h:>2d}  {op:<28s}  {t}   {why}")
    if len(chain) > 30:
        print(f"      ... ({len(chain) - 30} more)")

    if success_kind:
        print(f"\n    >>> WALK SUCCEEDED: {success_kind}")
        print(f"        node name: {success_node.name}")
        print(f"        node outputs: {list(success_node.output)}")
    else:
        print(f"\n    >>> WALK FAILED — never reached LN or γ-scale Mul.")
        print(f"        death reasons (each branch):")
        for r in death_reasons[:10]:
            print(f"          {r}")

    if not success_kind:
        return  # nothing to do for step D

    if "FUSED" in success_kind:
        print(f"\n=== D skipped — fused LN bound is computed directly from γ,β. ===")
        return

    # ============================================================ STEP D
    print(f"\n=== D. Strict signature BFS upstream from γ-scale Mul ===")
    scale_mul = success_node
    gamma_inp = next((inp for inp in scale_mul.input
                     if init_dims(g, inp) is not None
                     and len(init_dims(g, inp)) == 1
                     and init_dims(g, inp)[0] >= 2), None)
    other_input = next(inp for inp in scale_mul.input if inp != gamma_inp)
    print(f"    γ initializer: {gamma_inp}")
    print(f"    other input (the standardised residual): {other_input}")

    visited2 = set()
    queue2 = [(other_input, 0, "start")]
    sig = {"Sqrt": False, "Rsqrt": False, "ReduceMean": False, "Pow2": False, "Mul_x_x": False}
    chain2 = []
    NORM_HOPS = 8
    while queue2:
        nm, hps, why = queue2.pop(0)
        if nm in visited2 or hps > NORM_HOPS: continue
        visited2.add(nm)
        p = produces.get(nm)
        if p is None:
            chain2.append((hps, "GRAPH_INPUT_OR_INIT", nm[:60], why)); continue
        chain2.append((hps, p.op_type, nm[:60], why))
        if p.op_type == "Sqrt": sig["Sqrt"] = True
        elif p.op_type == "Rsqrt": sig["Rsqrt"] = True
        elif p.op_type == "ReduceMean": sig["ReduceMean"] = True
        elif p.op_type == "Pow" and len(p.input) >= 2:
            arr = resolve_scalar(g, produces, p.input[1])
            if arr is not None and arr.size == 1 and abs(float(arr.flatten()[0]) - 2.0) < 1e-6:
                sig["Pow2"] = True
        elif p.op_type == "Mul" and len(p.input) == 2 and p.input[0] == p.input[1]:
            sig["Mul_x_x"] = True
        if p.op_type in NORM_BFS_OPS or p.op_type == "Rsqrt":
            for u in p.input:
                if u: queue2.append((u, hps + 1, f"<-{p.op_type}"))
        if (sig["Sqrt"] or sig["Rsqrt"]) and sig["ReduceMean"] and (sig["Pow2"] or sig["Mul_x_x"]):
            break

    print(f"\n    Chain (first 25 entries):")
    for h, op, t, why in chain2[:25]:
        print(f"      hop={h:>2d}  {op:<28s}  {t}   {why}")
    if len(chain2) > 25:
        print(f"      ... ({len(chain2) - 25} more)")

    print(f"\n    Signature found:")
    for k, v in sig.items():
        print(f"      {k:<14s}  {v}")

    found_sqrt = sig["Sqrt"] or sig["Rsqrt"]
    found_pow = sig["Pow2"] or sig["Mul_x_x"]

    if found_sqrt and sig["ReduceMean"] and found_pow:
        print(f"\n    >>> Signature COMPLETE — rescue should work.")
        if sig["Rsqrt"] and not sig["Sqrt"]:
            print(f"        Note: matched via Rsqrt (current llm_gate_rescue.py only checks Sqrt — that's the bug)")
        if sig["Mul_x_x"] and not sig["Pow2"]:
            print(f"        Note: matched via Mul(x,x) (already supported in current code)")
    else:
        missing = []
        if not found_sqrt: missing.append("Sqrt/Rsqrt")
        if not sig["ReduceMean"]: missing.append("ReduceMean")
        if not found_pow: missing.append("Pow(2)/Mul(x,x)")
        print(f"\n    >>> Signature INCOMPLETE — missing: {missing}")
        print(f"        Rescue fails because BFS ({NORM_HOPS} hops) cannot reach all three.")


if __name__ == "__main__":
    main()
