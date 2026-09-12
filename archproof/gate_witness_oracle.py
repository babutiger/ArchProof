"""Optional G2-witness oracle: PGD-based search for an input x* such that
||g(x*)||_inf > tau_adm, upgrading G2-abstract to a verified witness-existence.

Fail mode: if no witness is found within the iteration budget, we declare
G2-abstract only (current behaviour of the verifier).  This is a STRICTLY
OPTIONAL add-on; the sound output-contribution certificate does NOT depend
on witness existence.

Implementation strategy:
  - Load ONNX via onnxruntime for forward evaluation.
  - Numerical (finite-difference) gradient ascent on ||g(x)||_inf.
  - N_RESTARTS random starts inside the ambient input box, project each
    step onto the box, run N_ITER steps with step size eta.
  - Return (witness, achieved) or (None, max_achieved) on failure.

Design rationale: we avoid ONNX->PyTorch conversion so the oracle can run
on any ONNX model without framework coupling, at the cost of finite-diff
gradient noise.  For small input dims (<= 1024) this is fast; for larger
inputs we subsample coordinates at each step.
"""

import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import numpy as np
import onnx
import onnxruntime as ort
from typing import Optional, Tuple


def _build_intermediate_session(onnx_model: onnx.ModelProto,
                                gate_tensor_name: str) -> ort.InferenceSession:
    """Add gate_tensor_name as a model output so we can probe it."""
    m2 = onnx.ModelProto()
    m2.CopyFrom(onnx_model)
    # If gate already an output, just return
    existing_outs = {vo.name for vo in m2.graph.output}
    if gate_tensor_name not in existing_outs:
        # Add as output
        new_vi = onnx.ValueInfoProto()
        new_vi.name = gate_tensor_name
        # Use a generic float tensor type
        new_vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        m2.graph.output.append(new_vi)
    # Create a runtime session
    sess = ort.InferenceSession(m2.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    return sess


def try_find_witness(onnx_model: onnx.ModelProto,
                     gate_tensor_name: str,
                     input_name: str,
                     input_shape: Tuple[int, ...],
                     tau_adm: float,
                     input_lb: float = -1.0,
                     input_ub: float = 1.0,
                     n_restarts: int = 8,
                     n_iter: int = 60,
                     eta: float = 0.05,
                     fd_eps: float = 1e-3,
                     fd_subsample: int = 256,
                     rng: Optional[np.random.RandomState] = None) -> dict:
    """Return dict with keys: found (bool), witness_linf (float),
    n_evals (int), best_x (np.ndarray or None)."""
    if rng is None:
        rng = np.random.RandomState(0)

    try:
        sess = _build_intermediate_session(onnx_model, gate_tensor_name)
    except Exception as e:
        return {"found": False, "witness_linf": -np.inf,
                "n_evals": 0, "error": str(e), "best_x": None}

    def _probe(x):
        outs = sess.run([gate_tensor_name], {input_name: x.astype(np.float32)})
        return np.max(np.abs(outs[0]))

    best_val = -np.inf
    best_x = None
    n_evals = 0
    d_flat = int(np.prod(input_shape))

    for r in range(n_restarts):
        x = rng.uniform(input_lb, input_ub, size=input_shape).astype(np.float64)
        last = _probe(x)
        n_evals += 1
        for it in range(n_iter):
            # Finite-difference gradient on a subsample of flat coordinates
            n_sub = min(fd_subsample, d_flat)
            idx = rng.choice(d_flat, size=n_sub, replace=False)
            x_flat = x.reshape(-1)
            grad = np.zeros(d_flat)
            base = last
            for j in idx:
                orig = x_flat[j]
                x_flat[j] = orig + fd_eps
                plus = _probe(x.reshape(input_shape))
                x_flat[j] = orig
                grad[j] = (plus - base) / fd_eps
            n_evals += n_sub
            # Ascent step
            step = eta * np.sign(grad)  # sign-gradient = L_inf PGD
            x_flat = x_flat + step
            x = np.clip(x_flat.reshape(input_shape), input_lb, input_ub)
            last = _probe(x)
            n_evals += 1
            if last > best_val:
                best_val = last
                best_x = x.copy()
            if last > tau_adm:
                # Early exit on witness
                return {"found": True, "witness_linf": float(last),
                        "n_evals": n_evals, "best_x": best_x}

    return {"found": best_val > tau_adm, "witness_linf": float(best_val),
            "n_evals": n_evals, "best_x": best_x}


if __name__ == "__main__":
    import glob, json, time
    from pathlib import Path

    tau_adm = 0.3
    results = []
    rng = np.random.RandomState(20260422)

    # Hunt for small / medium ONNX models with tensor-tensor Muls
    import os as _os
    candidates = []
    for patt in [
        _os_ar.path.join(_AR, "benchmark/clean/*.onnx"),
        _os_ar.path.join(_AR, "benchmark/exporter_test/*.onnx"),
    ]:
        candidates.extend(glob.glob(patt))
    # Filter to those with tensor-tensor Mul and size < 10MB
    filtered = []
    for p in candidates:
        try:
            if _os.path.getsize(p) > 10 * 1024 * 1024:
                continue
            mm = onnx.load(p)
            _inits = {i.name for i in mm.graph.initializer}
            for _n in mm.graph.node:
                if _n.op_type == "Mul":
                    _ti = [i for i in _n.input if i and i not in _inits]
                    if len(_ti) == 2:
                        filtered.append(p)
                        break
        except Exception:
            pass
    candidates = sorted(set(filtered))

    for path in candidates:
        try:
            m = onnx.load(path)
        except Exception as e:
            continue
        inits = {init.name for init in m.graph.initializer}
        # Find activation->Mul gate candidates
        mul_nodes = []
        for n in m.graph.node:
            if n.op_type == "Mul":
                tensor_ins = [i for i in n.input if i and i not in inits]
                if len(tensor_ins) == 2:
                    mul_nodes.append(n)
        if not mul_nodes:
            continue
        # Pick first Mul; the activation input is the gate activation output tensor
        mn = mul_nodes[0]
        # Find activation produces
        produces = {o: nd for nd in m.graph.node for o in nd.output}
        ACTS = {"Relu", "Sigmoid", "Tanh", "Gelu", "Silu", "Swish", "HardSwish",
                "LeakyRelu", "Softplus", "Mish", "HardSigmoid", "Hardtanh"}
        gate_tensor = None
        for mi in mn.input:
            if mi in produces and produces[mi].op_type in ACTS:
                gate_tensor = mi
                break
        if gate_tensor is None:
            continue
        # Input shape
        vin = m.graph.input[0]
        try:
            shape = tuple(max(1, d.dim_value) for d in vin.type.tensor_type.shape.dim)
        except Exception:
            continue
        t0 = time.time()
        res = try_find_witness(m, gate_tensor, vin.name, shape,
                                tau_adm=tau_adm, input_lb=-1.0, input_ub=1.0,
                                n_restarts=6, n_iter=40, rng=rng)
        res["model"] = path.split("/")[-1]
        res["gate_tensor"] = gate_tensor
        res["wall_sec"] = round(time.time() - t0, 2)
        res.pop("best_x", None)
        results.append(res)
        status = "WITNESS" if res["found"] else "no-witness"
        print(f"  {res['model']:40s}  {status:12s}  achieved={res['witness_linf']:.3f}  evals={res['n_evals']}  {res['wall_sec']}s")

    summary = {
        "tau_adm": tau_adm,
        "n_models": len(results),
        "n_with_witness": sum(1 for r in results if r["found"]),
        "n_without_witness": sum(1 for r in results if not r["found"]),
        "results": results,
    }
    out = Path(_os_ar.path.join(_AR, "benchmark/r3_g2_witness_oracle.json"))
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}")
    print(f"witness found on {summary['n_with_witness']}/{summary['n_models']} gated models")
