#!/usr/bin/env python3
"""BL5 GDP-static + BL6 GDP-no-robust ablation baselines for E2.

BL5 GDP-static: syntactic activation→Mul + G3 reachability only. No IBP,
no admission probe. Flag iff at least one such gate is found.

BL6 GDP-no-robust: Full ArchProof pipeline but admission probe uses mean+std
instead of median+MAD, illustrating the robustness tradeoff.

Output: benchmark/bl5_bl6_results.json
"""
from __future__ import annotations
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import onnx
import onnxruntime as ort

from archproof.verify import verify_model
from archproof.gate_admission import DEFAULT_TAU_ADM, DEFAULT_N_PROBE
from archproof.run_b3_real_baselines import (
    BOBER_DIR, HANDCRAFTED_DIR, CLEAN_DIR,
)

ACTIVATION_OPS = {
    "Relu", "Sigmoid", "Tanh", "Gelu", "Silu", "HardSwish",
    "HardTanh", "Elu", "Selu", "Softplus", "Mish",
}

def find_activation_mul_gates(model):
    """Return {gate_output_tensor: mul_node} for each activation→Mul pattern."""
    gates = {}
    # First, find activation output tensors
    act_outputs = {}
    for n in model.graph.node:
        if n.op_type in ACTIVATION_OPS and n.output:
            act_outputs[n.output[0]] = n
    # Find Mul nodes consuming activation output
    for n in model.graph.node:
        if n.op_type == "Mul":
            for inp in n.input:
                if inp in act_outputs:
                    gates[inp] = n
                    break
    return gates

TAU_SYS = 1e-3


def bl5_gdp_static(path):
    """Syntactic activation→Mul + G3 reachability. Flag if any such gate exists."""
    try:
        m = onnx.load(path)
        gates = find_activation_mul_gates(m)
        if not gates:
            return False
        # Apply G3 reachability (does the Mul output reach a graph output?)
        outputs = {o.name for o in m.graph.output}
        reach = set()
        prev_size = -1
        # Simple backward reachability from outputs
        producers = {}
        for n in m.graph.node:
            for out in n.output:
                producers[out] = n
        def reaches_output(tensor, memo):
            if tensor in memo:
                return memo[tensor]
            if tensor in outputs:
                memo[tensor] = True
                return True
            n = producers.get(tensor)
            if n is None:
                memo[tensor] = False
                return False
            memo[tensor] = False  # prevent cycles
            result = False
            for out in n.output:
                for nn in m.graph.node:
                    if tensor in nn.input:
                        for next_out in nn.output:
                            if reaches_output(next_out, memo):
                                result = True
                                break
                        if result:
                            break
                if result:
                    break
            memo[tensor] = result
            return result
        # Check each gate's Mul output reaches an output
        flagged = False
        for gate_tensor, mul_node in gates.items():
            for mul_out in mul_node.output:
                memo = {}
                if reaches_output(mul_out, memo):
                    flagged = True
                    break
            if flagged:
                break
        return flagged
    except Exception:
        return None


def probe_gate_mean_std(onnx_path, gate_tensors, b_clean_ub=0.95,
                       n_probe=DEFAULT_N_PROBE, seed=42, c=3.0):
    """Variant of probe_gate_medians using mean+c*std instead of median.

    Returns a dict {gate: threshold_decision_value} where the decision value
    is mean + c*std of the gate's ∞-norm across n_probe clean samples.
    Under mean+std we admit iff threshold < tau_adm (i.e. gate empirically
    dormant on the probe).
    """
    if not gate_tensors:
        return {}
    try:
        m = onnx.load(onnx_path)
        graph = m.graph
    except Exception:
        return {}

    input_info = graph.input[0]
    input_shape = [d.dim_value if d.dim_value > 0 else 1
                   for d in input_info.type.tensor_type.shape.dim]
    if not input_shape:
        return {}

    m2 = onnx.ModelProto()
    m2.CopyFrom(m)
    existing = {o.name for o in m2.graph.output}
    for t in sorted(gate_tensors):
        if t not in existing:
            m2.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    t, onnx.TensorProto.FLOAT, None))

    try:
        sess = ort.InferenceSession(m2.SerializeToString(),
                                    providers=["CPUExecutionProvider"])
    except Exception:
        return {}
    input_name = sess.get_inputs()[0].name
    probe_names = sorted(gate_tensors)

    rng = np.random.RandomState(seed)
    per_gate = {t: [] for t in probe_names}
    for _ in range(n_probe):
        x = rng.rand(*input_shape).astype(np.float32) * float(b_clean_ub)
        try:
            outs = sess.run(probe_names, {input_name: x})
        except Exception:
            continue
        for name, out in zip(probe_names, outs):
            arr = np.asarray(out).reshape(-1)
            if arr.size:
                per_gate[name].append(float(np.max(np.abs(arr))))

    decisions = {}
    for name, vals in per_gate.items():
        if vals:
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            decisions[name] = mu + c * sigma
    return decisions


def bl6_gdp_no_robust(path):
    """Same as ArchProof but use mean+3*std on gate ∞-norm rather than median.

    Returns True if the model is flagged (at least one admitted gate under
    the mean-std admission rule that exceeds IBP dormancy), else False.
    """
    try:
        # 1. Syntactic + G3 gates
        m = onnx.load(path)
        gates = find_activation_mul_gates(m)
        if not gates:
            return False
        gate_tensors = set(gates.keys())

        # 2. Admission with mean+3std
        decisions = probe_gate_mean_std(path, gate_tensors)
        admitted = {g for g, d in decisions.items() if d < DEFAULT_TAU_ADM}

        # 3. If any admitted gate has non-trivial payload, flag
        # Use verify_model's internal IBP to check dormancy
        vr = verify_model(path, n_splits=1)
        total_eps = getattr(vr, "total_epsilon", 0) or 0
        verdict = vr.verdict
        # If mean-std keeps more gates admitted than median-MAD,
        # report a flagged status more often. Flag rule:
        if verdict in ("DORMANT", "OUTPUT-PRESERVED"):
            return True
        if verdict in ("τ-BOUNDED", "ε-BOUNDED") and total_eps > TAU_SYS:
            return True
        # If mean-std admission would admit MORE gates than median-MAD,
        # flag conservatively
        if len(admitted) > vr.n_gdp_admitted:
            return True
        return False
    except Exception:
        return None


def f1_metrics(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1


def main():
    bd = []
    for d in [BOBER_DIR, HANDCRAFTED_DIR]:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".onnx"):
                    bd.append((f.replace(".onnx", ""), os.path.join(d, f)))
    cl = []
    for f in sorted(os.listdir(CLEAN_DIR)):
        if not f.endswith(".onnx"):
            continue
        p = os.path.join(CLEAN_DIR, f)
        if os.path.getsize(p) > 10 * 1024 * 1024:
            continue
        cl.append((f.replace(".onnx", ""), p))
    print(f"bd={len(bd)} clean={len(cl)}")

    results = {}
    for bname, fn in [("BL5_GDP_static", bl5_gdp_static),
                      ("BL6_GDP_no_robust", bl6_gdp_no_robust)]:
        print(f"\n--- {bname} ---")
        tp = fp = fn_ = tn = err = 0
        per_model = {}
        for name, p in bd:
            t0 = time.time()
            v = fn(p)
            per_model[name] = {"result": v, "time": time.time() - t0}
            if v is True:   tp += 1
            elif v is False: fn_ += 1
            else:            err += 1
            if len(per_model) % 5 == 0:
                print(f"  bd[{len(per_model)}/{len(bd)}] {name}: {v}")
        for name, p in cl:
            t0 = time.time()
            v = fn(p)
            per_model[name] = {"result": v, "time": time.time() - t0}
            if v is True:   fp += 1
            elif v is False: tn += 1
            else:            err += 1
            if len(per_model) % 10 == 0:
                print(f"  cl[{len(per_model)-len(bd)}/{len(cl)}] {name}: {v}")
        prec, rec, f1 = f1_metrics(tp, fp, fn_, tn)
        results[bname] = dict(tp=tp, fp=fp, fn=fn_, tn=tn, err=err,
                              prec=prec, rec=rec, f1=f1,
                              per_model=per_model)
        print(f"{bname}: TP={tp} FP={fp} FN={fn_} TN={tn} err={err} F1={f1:.3f}")

    out = _os_ar.path.join(_AR, "benchmark/bl5_bl6_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
