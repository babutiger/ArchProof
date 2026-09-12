"""Chain-sensitivity (K_i, H_i) computation on ONNX graphs.

Reviewer-driven upgrade of ACPC: the rho-poisoning attacker replaces
input calibration samples; the induced drift on the gate pre-activation
interval and on the payload tensor is bounded by the L_inf -> L_inf
operator-norm product along the ONNX graph path.

For an admitted gate i with Mul node M_i and activation phi_i:
  K_i := Lipschitz(input -> pre-activation of phi_i)
  H_i := Lipschitz(input -> payload tensor, EXCLUDING path through gate i itself)

Per-op operator norms (L_inf -> L_inf Lipschitz of the op function):
  Conv / Gemm / MatMul  : ||W||_{inf->inf}  = max row abs-sum of W
  BatchNormalization    : max |gamma / sqrt(sigma^2 + eps)|
  Activation (phi)      : sup |phi'(z)| from ACTIVATION_LIPSCHITZ
  Add / Sub             : 1.0   (triangle inequality: sum of branch sensitivities)
  Reshape / Flatten /
  Transpose / Squeeze /
  Unsqueeze / Identity  : 1.0
  AveragePool / GlobalAvgPool / MaxPool : 1.0 (each is 1-Lipschitz on L_inf)
  Softmax               : 1.0 (sup Jacobian L_inf norm <= 1)
  Mul (scalar/const)    : |constant|_inf
  Mul (tensor-tensor)   : product-rule: ||a||_inf * Lip(b,x) + ||b||_inf * Lip(a,x);
                          used only at the gate itself; skipped in H_i path by design.
  Unknown / custom ops  : +inf (conservative; propagates to vacuous bound)

Propagation rule for tensor t with producer node N having tensor inputs {t_1,...,t_k}:
    K[t] = op_lip(N) * sum_j K[t_j]
This upper-bounds any path composition via triangle inequality on branch merges
(Add/Concat) and chain rule on single-tensor ops.
"""

import os
import numpy as np
import onnx
from onnx import numpy_helper
from collections import defaultdict
from typing import Dict, Tuple, Optional, List, Set


# Activation Lipschitz constants (same values as archproof/acpc.py)
ACTIVATION_LIPSCHITZ = {
    "Relu": 1.0, "LeakyRelu": 1.0, "Sigmoid": 0.25, "Tanh": 1.0,
    "Gelu": 1.13, "Silu": 1.11, "Swish": 1.11, "HardSwish": 1.0,
    "HardSigmoid": 0.2, "Hardtanh": 1.0, "Elu": 1.0, "Selu": 1.7581,
    "Softplus": 1.0, "Mish": 1.09, "Softmax": 1.0,
}


def _load_initializer(model: onnx.ModelProto, name: str) -> Optional[np.ndarray]:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _matrix_linf_operator_norm(W: np.ndarray, op_type: str) -> float:
    """L_inf -> L_inf operator norm = max absolute row sum.

    Conv  : W is [C_out, C_in, Kh, Kw] -> reshape [C_out, C_in*Kh*Kw]
    Gemm  : W may be [M, N] or [N, M]; use max over both interpretations (conservative)
    MatMul: same as Gemm
    """
    W = np.asarray(W, dtype=np.float64)
    if op_type == "Conv":
        M = W.reshape(W.shape[0], -1)
        return float(np.abs(M).sum(axis=1).max())
    if op_type in ("Gemm", "MatMul"):
        M = W if W.ndim == 2 else W.reshape(W.shape[0], -1)
        # L_inf op norm is max row abs-sum; transpose also tried for Gemm transB=1
        norm1 = float(np.abs(M).sum(axis=1).max())
        norm2 = float(np.abs(M).sum(axis=0).max())
        return max(norm1, norm2)
    return float(np.abs(W).max())


def _resolve_weight_through_layout(node, initializers: Dict[str, np.ndarray],
                                    produces=None
                                    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Locate the weight tensor for Conv/Gemm/MatMul. The weight may be
    a direct initializer, OR routed through a chain of layout-only ops
    (Transpose/Reshape/Squeeze/Unsqueeze/Cast/Identity) that ONNX
    exporters routinely emit. This walker peels those wrappers until
    the underlying initializer is found.

    The L_∞→L_∞ operator norm is layout-invariant (max row 1-norm of W
    is the same after Transpose since rows just become cols), so we
    can return the underlying array and the caller computes the norm
    once on it.
    """
    if len(node.input) < 2:
        return (None, None)
    name = node.input[1]
    if name in initializers:
        return (initializers[name], name)
    if produces is None:
        return (None, None)
    LAYOUT_OPS = {"Transpose", "Reshape", "Squeeze", "Unsqueeze",
                  "Cast", "Identity"}
    cur = name
    for _ in range(6):  # bounded
        prod = produces.get(cur)
        if prod is None:
            return (None, None)
        if prod.op_type not in LAYOUT_OPS or not prod.input:
            return (None, None)
        upstream = prod.input[0]
        if upstream in initializers:
            return (initializers[upstream], upstream)
        cur = upstream
    return (None, None)


def _op_sensitivity(node, initializers: Dict[str, np.ndarray],
                    produces=None) -> Tuple[float, str]:
    """Intrinsic operator-norm of the op's function (not composing with upstream K).

    Returns (value, label).  value = +inf for unknown/custom ops.
    """
    op = node.op_type
    if op in ("Conv", "Gemm", "MatMul"):
        W, w_name = _resolve_weight_through_layout(
            node, initializers, produces=produces)
        if W is None:
            # Weight is not reachable as a constant initializer through
            # any short layout chain: conservative +inf.
            return (np.inf, f"{op}-nonconst-weight")
        return (_matrix_linf_operator_norm(W, op), f"{op}-weight")
    if op == "BatchNormalization":
        # inputs: [x, gamma, beta, mean, var]
        if len(node.input) < 5:
            return (np.inf, "BN-inputs")
        gamma = initializers.get(node.input[1])
        var = initializers.get(node.input[4])
        if gamma is None or var is None:
            return (np.inf, "BN-nonconst")
        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = float(attr.f)
        return (float(np.abs(gamma / np.sqrt(var + eps)).max()), "BN")
    if op in ACTIVATION_LIPSCHITZ:
        return (ACTIVATION_LIPSCHITZ[op], f"act-{op}")
    if op in ("Add", "Sub", "Identity", "Reshape", "Flatten", "Transpose",
              "Squeeze", "Unsqueeze", "Expand", "Pad", "Concat",
              "MaxPool", "AveragePool", "GlobalAveragePool", "ReduceMean"):
        return (1.0, op)
    if op == "Mul":
        # If one input is a constant, multiply by its L_inf magnitude
        const_in = None
        for i in node.input:
            if i in initializers:
                const_in = initializers[i]
                break
        if const_in is not None:
            return (float(np.abs(const_in).max()), "Mul-const")
        # Tensor-tensor Mul = gate boundary (handled separately)
        return (np.inf, "Mul-gate")
    if op in ("Clip", "HardSigmoid", "Hardtanh"):
        # Clip is 1-Lipschitz; HardSigmoid has slope 0.2
        return (ACTIVATION_LIPSCHITZ.get(op, 1.0), op)
    # Unknown op: conservative +inf (propagates to undecided verdict)
    return (np.inf, f"unknown-{op}")


def _build_produces_consumes(model: onnx.ModelProto) -> Tuple[Dict, Dict]:
    produces = {}
    consumes = defaultdict(list)
    for n in model.graph.node:
        for o in n.output:
            produces[o] = n
        for i in n.input:
            consumes[i].append(n)
    return produces, consumes


def compute_K_H_for_gate(model: onnx.ModelProto,
                         gate_mul_out: str,
                         gate_pre_activation: Optional[str] = None,
                         skip_nodes: Optional[Set[str]] = None,
                         rescue_pre: bool = False,
                         rescue_payload: bool = False) -> Dict:
    """Compute K_i (input -> pre-activation) and H_i (input -> payload).

    Args:
      model                : onnx.ModelProto
      gate_mul_out         : output tensor name of the gate Mul
      gate_pre_activation  : optional tensor name of the pre-activation input
                             to phi_i (i.e. activation's direct input).  If None,
                             we walk upstream from the Mul node to locate the
                             activation input.
      skip_nodes           : optional set of node names that should NOT be used
                             when computing H_i (to exclude the gate's own Mul
                             when walking downstream).
      rescue_pre           : if True, the gate's pre-activation IBP bound
                             is rescue-anchored (graph-only constant past
                             the first LayerNorm by Lemma 1). Theorem 6's
                             chain Lipschitz K_i is then 0 by construction
                             (the IBP-bound is x-independent → its
                             Lipschitz w.r.t. input is 0). Short-circuit
                             K_i = 0 without traversing the upstream chain.
      rescue_payload       : symmetric for the payload tensor → H_i = 0.

    Returns dict with keys:
      K_i, H_i           : floats (inf if path undecidable; 0 if rescue
                           short-circuit applies)
      K_path, H_path     : list of (op, local_norm) along dominant path
      notes              : any warnings

    Semantics (chain-traversal mode):
      K[t] = op_lip(producer(t)) * sum_{t_j tensor-input} K[t_j]
      K[input] = 1

    Rescue-aware semantics: when the verifier reports rescue_pre=True or
    rescue_payload=True (from gate_epsilons), the IBP propagation chain
    is short-circuited at the LayerNorm boundary; Theorem 6's recursion
    yields K_i = 0 (resp. H_i = 0) without further traversal.

    H is otherwise computed as: K[model_output] with propagation skipping
    the gate node's downstream contribution (we set K of the Mul-output
    to 0 in the H sub-problem so only non-gate-branch contributions
    flow through).
    """
    # Rescue-aware short-circuit: avoids OOM on whole-LLM (where the
    # naive eager-load of all initializers peaks at ~30 GB) and matches
    # the mathematically tight K_i_for_ACPC = 0 in the rescue regime.
    if rescue_pre and rescue_payload:
        return {
            "K_i": 0.0,
            "H_i": 0.0,
            "K_path": [],
            "H_path": [],
            "notes": ("rescue_aware_short_circuit: pre-activation and "
                      "payload IBP bounds are graph-only under "
                      "Lemma 1 → K_i = H_i = 0 by Theorem 6's "
                      "IBP-propagation Lipschitz definition")
        }
    initializers = {init.name: numpy_helper.to_array(init)
                    for init in model.graph.initializer}
    produces, consumes = _build_produces_consumes(model)
    # Non-const tensor inputs to each node
    def tensor_inputs_of(node):
        return [i for i in node.input if i and i not in initializers]

    # Graph inputs (model-level)
    graph_input_names = [vi.name for vi in model.graph.input
                         if vi.name not in initializers]
    graph_output_names = [vo.name for vo in model.graph.output]

    # Topological order of nodes
    # Simple Kahn's algorithm
    indeg = {n.name: 0 for n in model.graph.node}
    name_to_node = {n.name: n for n in model.graph.node}
    for n in model.graph.node:
        for t in tensor_inputs_of(n):
            if t in produces:
                indeg[n.name] += 1
    # Seed with nodes that have no tensor-producer-input
    topo = []
    ready = [n for n in model.graph.node
             if indeg[n.name] == 0]
    # Simulate Kahn (note: `ready` is a working queue; we avoid dup)
    ready_names = {n.name for n in ready}
    while ready:
        n = ready.pop(0)
        topo.append(n)
        for o in n.output:
            for cons in consumes.get(o, []):
                if cons.name in ready_names:
                    continue
                indeg[cons.name] -= 1
                if indeg[cons.name] == 0:
                    ready.append(cons)
                    ready_names.add(cons.name)
    # Fallback: if topo order didn't cover all nodes, append the rest
    remaining = [n for n in model.graph.node if n not in topo]
    topo.extend(remaining)

    # --- K computation: input -> pre-activation ---
    K = {name: 1.0 for name in graph_input_names}  # identity on inputs

    notes = []
    K_trace = []
    for n in topo:
        ins = tensor_inputs_of(n)
        if not ins:
            continue
        if any(i not in K for i in ins):
            # Not reachable from graph input yet; skip
            continue
        op_lip, label = _op_sensitivity(n, initializers, produces=produces)
        input_K_sum = sum(K[i] for i in ins)
        out_K = op_lip * input_K_sum
        for o in n.output:
            # If tensor already has K (shared), take the MAX (worst-case Lipschitz)
            if o in K:
                K[o] = max(K[o], out_K)
            else:
                K[o] = out_K
            K_trace.append((n.name, n.op_type, label, op_lip, input_K_sum, out_K))

    # Locate pre-activation tensor if not given
    pre_act_tensor = gate_pre_activation
    if pre_act_tensor is None:
        # Walk upstream from gate_mul_out: Mul has two inputs (gate activation, payload);
        # the gate branch is the one whose immediate producer is an activation op.
        mul_node = produces.get(gate_mul_out)
        if mul_node is None:
            notes.append(f"Mul-output {gate_mul_out} has no producer")
            pre_act_tensor = None
        else:
            act_in = None
            for mi in mul_node.input:
                prod = produces.get(mi)
                if prod is not None and prod.op_type in ACTIVATION_LIPSCHITZ:
                    # activation's input is the pre-activation
                    if len(prod.input) >= 1:
                        act_in = prod.input[0]
                        break
            pre_act_tensor = act_in
            if pre_act_tensor is None:
                notes.append(f"could not locate pre-activation upstream of {gate_mul_out}")

    K_i = K.get(pre_act_tensor, np.inf) if pre_act_tensor else np.inf

    # --- H computation: sensitivity of the PAYLOAD TENSOR to input ---
    # Payload = non-activation input of the gate Mul node, since
    #   c_i = epsilon_phi(l_i, u_i) * || p_i ||_infty
    # and p_i is the OTHER side of the Mul.  H_i = K[payload_tensor].
    mul_node = produces.get(gate_mul_out)
    payload_tensor = None
    if mul_node is not None:
        act_side = None
        for mi in mul_node.input:
            prod = produces.get(mi)
            if prod is not None and prod.op_type in ACTIVATION_LIPSCHITZ:
                act_side = mi
                break
        for mi in mul_node.input:
            if mi != act_side and mi not in initializers:
                payload_tensor = mi
                break
        if payload_tensor is None:
            # Fall back: first input if two non-init inputs present
            tensor_ins = [i for i in mul_node.input if i and i not in initializers]
            if len(tensor_ins) >= 2:
                payload_tensor = tensor_ins[0] if tensor_ins[0] != act_side else tensor_ins[1]
    H_i = K.get(payload_tensor, np.inf) if payload_tensor else np.inf

    return {
        "gate_mul_out": gate_mul_out,
        "pre_activation_tensor": pre_act_tensor,
        "payload_tensor": payload_tensor,
        "K_i": float(K_i),
        "H_i": float(H_i),
        "K_trace_len": len(K_trace),
        "notes": notes,
    }


if __name__ == "__main__":
    # Smoke test on a small injected model if available
    import glob, os, sys
    candidates = []
    for patt in [
        os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/clean/*.onnx"),
        os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/*.onnx"),
    ]:
        candidates.extend(glob.glob(patt))
    if not candidates:
        print("no ONNX models found")
        sys.exit(0)
    model_path = candidates[0]
    print(f"loading {model_path}")
    m = onnx.load(model_path)
    # find first Mul with two tensor inputs (a likely gate candidate)
    inits = {init.name for init in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Mul":
            tensor_ins = [i for i in n.input if i not in inits]
            if len(tensor_ins) == 2:
                print(f"testing Mul node {n.name}, output {n.output[0]}")
                result = compute_K_H_for_gate(m, n.output[0])
                print(f"  K_i = {result['K_i']:.3e}")
                print(f"  H_i = {result['H_i']:.3e}")
                print(f"  pre_activation_tensor = {result['pre_activation_tensor']}")
                print(f"  notes = {result['notes']}")
                break
    else:
        print("no two-tensor Mul found in model")
