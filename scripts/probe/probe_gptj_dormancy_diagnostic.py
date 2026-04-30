"""Diagnostic: run the probe ONLY on gpt-j-6b-backdoored and print
the raw median (and per-sample max-norm) for the injected gate.
Lets us see exactly what the gate measures on random-token inputs
so we can decide whether bias=-10 is enough or we need to tighten it.

Runtime: ~1-2 min (one ORT session + 20 forward passes).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ONNX_PATH = (ROOT / "benchmark" / "7b_onnx" /
             "gpt-j-6b-backdoored" / "gpt-j-6b-backdoored.onnx")


def main():
    if not ONNX_PATH.exists():
        print(f"missing: {ONNX_PATH}", file=sys.stderr)
        sys.exit(1)

    # 1. Load graph-only and locate the inject's ReLU.
    print(f"[1/4] loading graph (no weights) ...", flush=True)
    m = onnx.load(str(ONNX_PATH), load_external_data=False)
    print(f"  nodes={len(m.graph.node)}", flush=True)

    # input dtype
    input_info = m.graph.input[0]
    elem_type = input_info.type.tensor_type.elem_type
    print(f"  input dtype code={elem_type} (7=INT64, 1=FLOAT)", flush=True)
    input_shape = []
    for dim in input_info.type.tensor_type.shape.dim:
        input_shape.append(dim.dim_value if dim.dim_value > 0 else 1)
    print(f"  input shape={input_shape}", flush=True)

    relu_node = None
    for n in m.graph.node:
        if n.op_type == "Relu" and any("backdoor" in t.lower()
                                        for t in n.output):
            relu_node = n; break
    if relu_node is None:
        print("could not locate /backdoor Relu", file=sys.stderr); sys.exit(1)
    gate_out = relu_node.output[0]
    print(f"  inject gate output tensor: {gate_out}", flush=True)

    # also find the pre-activation MatMul output (right before ReLU)
    pre_act = relu_node.input[0]
    print(f"  pre-activation tensor: {pre_act}", flush=True)

    # 2. Add gate output AND pre-activation to graph outputs.
    existing = {o.name for o in m.graph.output}
    for t in (gate_out, pre_act):
        if t not in existing:
            m.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    t, onnx.TensorProto.FLOAT, None))

    # 3. Save modified graph + load via ORT.
    temp = ONNX_PATH.parent / f"_probe_diagnostic_{os.getpid()}.onnx"
    print(f"[2/4] saving stub graph to {temp.name} ...", flush=True)
    try:
        onnx.save(m, str(temp))
        sz = temp.stat().st_size / (1024 ** 2)
        print(f"  stub size: {sz:.1f} MB (good if << 23 GB)", flush=True)
        print(f"[3/4] loading ORT session (reads weights from external shards) ...",
              flush=True)
        t0 = time.time()
        sess = ort.InferenceSession(str(temp),
                                    providers=["CPUExecutionProvider"])
        print(f"  session ready in {time.time()-t0:.1f}s", flush=True)
        input_name = sess.get_inputs()[0].name

        # 4. Run 20 random-token probes; print per-sample gate output.
        print(f"[4/4] running 20 probes with random vocab ids in [0, 1000) ...",
              flush=True)
        rng = np.random.RandomState(42)
        gate_vals = []
        pre_vals = []
        for i in range(20):
            x = rng.randint(0, 1000, size=input_shape).astype(np.int64)
            t0 = time.time()
            outs = sess.run([gate_out, pre_act], {input_name: x})
            dt = time.time() - t0
            g_arr = np.asarray(outs[0]).reshape(-1)
            p_arr = np.asarray(outs[1]).reshape(-1)
            g_max = float(np.max(np.abs(g_arr)))
            p_max = float(np.max(np.abs(p_arr)))
            p_min = float(np.min(p_arr))
            p_mean = float(np.mean(p_arr))
            gate_vals.append(g_max)
            pre_vals.append(p_max)
            print(f"  probe {i:>2d}: gate |g|_∞={g_max:.6f}  "
                  f"pre-act min={p_min:+.3f} mean={p_mean:+.3f} "
                  f"|.|_∞={p_max:.3f}  ({dt:.1f}s)", flush=True)

        print()
        print(f"=== gate output |g|_∞ statistics over 20 probes ===")
        print(f"  median        = {float(np.median(gate_vals)):.6e}")
        print(f"  mean          = {float(np.mean(gate_vals)):.6e}")
        print(f"  max           = {float(np.max(gate_vals)):.6e}")
        print(f"  TAU_DORMANT   = 1.000000e-03")
        print(f"  dormant?      = {float(np.median(gate_vals)) < 1e-3}")
        print()
        print(f"=== pre-activation |.|_∞ statistics ===")
        print(f"  median        = {float(np.median(pre_vals)):.3f}")
        print(f"  max           = {float(np.max(pre_vals)):.3f}")

    finally:
        if temp.exists():
            try: temp.unlink()
            except Exception: pass


if __name__ == "__main__":
    main()
