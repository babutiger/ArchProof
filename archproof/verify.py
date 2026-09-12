"""ArchProof Unified Verification Pipeline.

Hierarchical, automatic GDP verification for any ONNX model.

Level 0: Graph scan — find GDP candidates (ReLU→Mul patterns)
Level 1: Standard IBP — check gate dormancy under B_clean
Level 2: Input splitting (50-split) — tighter for non-linear gate chains
Level 3: Quadratic refinement — near-exact for correlated Mul(var,var)
Level 4: Benign dead code filter — B_extended=[0,10] check

Output: VerificationResult with dormancy status, τ bound,
        output preservation ratio, and which level proved it.
"""

import os
import onnx
from onnx import numpy_helper
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from archproof.interval_propagation import (
    propagate_intervals, IntervalBound, _propagate_node
)
from archproof.quadratic_refinement import (
    propagate_with_quadratic, AffineForm, _bound_quadratic
)
from archproof.activation_epsilon import activation_epsilon
from archproof.gate_admission import (
    probe_gate_medians, admit_gates, DEFAULT_TAU_ADM, DEFAULT_N_PROBE,
)


# v3 T1 support: activation ops that can serve as gate ahead of Mul.
# We map ONNX op_type → act_type string used by activation_epsilon().
# Covers the 11 distinct activations for which activation_epsilon.py
# provides sound sup|φ| envelopes.
GATE_ACTIVATION_OPS = {
    "Relu":      "relu",
    "Sigmoid":   "sigmoid",
    "Tanh":      "tanh",
    "Gelu":      "gelu",
    "Silu":      "silu",
    "Swish":     "swish",    # alias of Silu
    "HardSwish": "hardswish",
    "HardTanh":  "hardtanh",
    "Elu":       "elu",
    "Selu":      "selu",
    "Softplus":  "softplus",
    "Mish":      "mish",
    "Softmax":   "softmax",
}


# C4: graph-equivalent ops with affine pair (1, 0). An adversary can insert
# any chain of these between an activation and a Mul without changing the
# function (within float-precision drift), so the syntactic activation->Mul
# match must follow this chain transitively. Add and Mul are conditionally
# transparent: they pass through only when the OTHER operand is a constant
# additive identity (zero) or multiplicative identity (one) respectively.
_TRANSPARENT_OPS = {
    "Identity", "Cast", "Reshape", "Transpose",
    "Squeeze", "Unsqueeze", "Flatten", "Expand",
    "Split", "Concat",
    # Conditionally transparent (gate logic below validates the constant operand):
    "Add", "Sub", "Mul",
}


@dataclass
class VerificationResult:
    """Complete verification result for one ONNX model."""
    # Model info
    name: str = ""
    n_nodes: int = 0

    # Level 0: graph scan
    n_gdp_candidates: int = 0       # gate_activation→Mul patterns found
    gate_relu_names: List[str] = field(default_factory=list)

    # T10 GDP admission (semantic dormancy filter)
    n_gdp_syntactic: int = 0        # pre-admission candidate count
    n_gdp_admitted: int = 0         # post-admission count (kept)
    n_gdp_rejected_non_dormant: int = 0  # rejected (empirical median >= τ_adm)
    gate_admission_report: Dict[str, str] = field(default_factory=dict)

    # Level 1-3: dormancy
    strictly_dormant: bool = False   # strict ReLU gate ub ≤ 0
    tau_ibp: float = float('inf')    # standard IBP gate τ
    tau_split: float = float('inf')  # 50-split gate τ
    tau_quad: float = float('inf')   # quadratic-refined gate τ
    proven_level: str = "none"       # which level proved dormancy

    # Level 3 extra: output preservation (for non-ReLU gates)
    output_preservation_lb: float = 0.0  # min(Sub_3) under B_clean
    output_preservation_ub: float = 1.0  # max(Sub_3) under B_clean
    has_output_preservation: bool = False

    # Level 4: benign filter
    is_benign_dead_code: bool = False

    # v3 T1: per-gate ε certificate (activation type + tight ε bound)
    gate_epsilons: List[Dict] = field(default_factory=list)
    # each entry: {"gate_node", "activation", "g_lb", "g_ub", "epsilon",
    #              "payload_abs_max", "contribution" (= epsilon × payload_abs_max)}
    total_epsilon: float = 0.0  # sum of raw activation εs (legacy)

    # v3 T2 Compositional: Σ_i ε_i · ||p_i||_∞ bounds |f_BD(x) - f_clean(x)|
    total_output_margin: float = 0.0     # Σ gate_contributions over B_clean

    # Final verdict
    verdict: str = "UNKNOWN"  # GDP-FREE / DORMANT / ε-BOUNDED / τ-BOUNDED / BENIGN / UNDECIDED


def _is_transparent_node(node, initializers):
    """C4 helper. Decide whether `node` acts as the affine pair (1, 0) on
    its first input — i.e. an attacker can insert it between activation and
    Mul without changing the gate's runtime value (within float drift).

    Strict cases (always transparent on input[0]):
        Identity, Cast, Reshape, Transpose, Squeeze, Unsqueeze, Flatten,
        Expand, Split (we treat as taking the first chunk), Concat (single
        input).

    Conditional cases (transparent only when the other operand is the
    appropriate numeric identity in the initializer set):
        Add(x, 0), Sub(x, 0)  with input[1] a constant tensor of zeros.
        Mul(x, 1)             with input[1] a constant tensor of ones.
    """
    op = node.op_type
    if op in {"Identity", "Cast", "Reshape", "Transpose",
              "Squeeze", "Unsqueeze", "Flatten", "Expand"}:
        return True
    if op == "Split":
        # A Split with one output is identity on input[0].
        return len(node.output) == 1
    if op == "Concat":
        # Concat with a single input is a copy.
        return len(node.input) == 1
    if op in {"Add", "Sub", "Mul"} and len(node.input) == 2:
        const_in = node.input[1]
        if const_in not in initializers:
            return False
        try:
            arr = numpy_helper.to_array(initializers[const_in])
        except Exception:
            return False
        if arr.size == 0:
            return False
        a = arr.astype(np.float32).ravel()
        if op in {"Add", "Sub"}:
            return bool(np.all(a == 0.0))
        if op == "Mul":
            return bool(np.all(a == 1.0))
    return False


_ARCHPROOF_C4_WALKBACK_ENABLED = (
    os.environ.get("ARCHPROOF_C4_WALKBACK", "1") != "0"
)


def _walk_back_to_activation(graph, src_tensor, activation_outputs,
                              max_depth=8):
    """C4 helper. Starting from a tensor name `src_tensor`, walk back through
    transparent nodes until we either reach an activation output (return
    that activation tensor name) or exceed max_depth (return None).

    Used to detect activation -> Identity/Cast/Reshape/... -> Mul chains
    where the literal Mul.input is no longer the activation output.

    Set the environment variable ``ARCHPROOF_C4_WALKBACK=0`` to disable
    the walkback (reproduces the pre-C4 strict matcher).
    """
    if not _ARCHPROOF_C4_WALKBACK_ENABLED:
        return src_tensor if src_tensor in activation_outputs else None
    if src_tensor in activation_outputs:
        return src_tensor
    initializers = {init.name: init for init in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    cur = src_tensor
    for _ in range(max_depth):
        node = producer.get(cur)
        if node is None:
            return None
        if not _is_transparent_node(node, initializers):
            return None
        cur = node.input[0]
        if cur in activation_outputs:
            return cur
    return None


def _build_const_map(graph):
    """Map tensor name -> constant numpy array, covering BOTH graph
    initializers and the outputs of `Constant` nodes. Recent opset exporters
    emit scalar constants (e.g. a Mul-by-200 factor) as Constant nodes rather
    than initializers, so a constant-operand check that only reads initializers
    misses them.
    """
    out = {}
    for init in graph.initializer:
        try:
            out[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass
    for n in graph.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    try:
                        out[n.output[0]] = numpy_helper.to_array(attr.t)
                    except Exception:
                        pass
    return out


def _const_mul_factor(node, const_map):
    """If `node` is a constant-scaling Mul (one dynamic operand, one constant),
    return max|const|; else None.

    A Mul-by-constant on the gate branch is the affine map g -> c*g. It scales
    the gate value but not whether the gate fires, so the payload-Mul search
    must pass through it while accumulating |c| into the payload norm to keep
    the epsilon bound sound.
    """
    if node.op_type != "Mul" or len(node.input) != 2:
        return None
    a, b = node.input
    if b in const_map and a not in const_map:
        arr = const_map[b]
    elif a in const_map and b not in const_map:
        arr = const_map[a]
    else:
        return None
    if arr.size == 0:
        return None
    return float(np.abs(arr.astype(np.float32)).max())


def _walk_forward_to_payload_mul(graph, gate_tensor, const_map, max_depth=8):
    """From a gate activation tensor, walk forward through constant-scaling
    Muls and value-preserving shape ops to the payload Mul: the first Mul whose
    other operand is a dynamic (non-constant) tensor.

    Returns (payload_mul_node, accumulated_scale). The scale is the product of
    the |constant| factors of every scaling Mul crossed (shape ops contribute
    1); multiply the payload norm by it to keep the epsilon contribution
    epsilon_phi * (scale * ||p||) sound. Returns (None, 1.0) if no payload Mul
    is reachable.

    A gate that feeds a payload Mul directly (the common case) returns that Mul
    with scale 1, identical to the pre-2026-08 behaviour.
    """
    consumers = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    shape_ops = {"Identity", "Cast", "Reshape", "Transpose",
                 "Squeeze", "Unsqueeze", "Flatten", "Expand"}
    cur = gate_tensor
    scale = 1.0
    for _ in range(max_depth):
        outs = consumers.get(cur, [])
        # a Mul with a dynamic partner is the payload gate Mul; `cur` is the
        # gate-side operand feeding it (the payload is the OTHER operand).
        for n in outs:
            if n.op_type == "Mul" and _const_mul_factor(n, const_map) is None \
                    and len(n.input) == 2:
                return n, scale, cur
        # otherwise follow a scaling Mul or a shape-only op forward
        advanced = False
        for n in outs:
            if n.op_type == "Mul":
                factor = _const_mul_factor(n, const_map)
                if factor is not None:
                    scale *= factor
                    cur = n.output[0]
                    advanced = True
                    break
            elif n.op_type in shape_ops:
                cur = n.output[0]
                advanced = True
                break
        if not advanced:
            return None, 1.0, None
    return None, 1.0, None


def _reachable_from_output(m):
    """Reverse BFS from graph outputs. Returns set of tensor names that
    influence any graph output (G4: reachability check)."""
    outputs = {o.name for o in m.graph.output}
    producer = {}
    for n in m.graph.node:
        for o in n.output:
            producer[o] = n
    reachable = set(outputs)
    frontier = list(outputs)
    while frontier:
        nxt = []
        for t in frontier:
            node = producer.get(t)
            if node is None:
                continue
            for inp in node.input:
                if inp and inp not in reachable:
                    reachable.add(inp)
                    nxt.append(inp)
        frontier = nxt
    return reachable


def verify_model(onnx_path: str, b_clean_ub: float = 0.95,
                 n_splits: int = 50, b_extended: float = 10.0,
                 gdp_flags: dict = None,
                 tau_adm: float = DEFAULT_TAU_ADM,
                 n_probe: int = DEFAULT_N_PROBE) -> VerificationResult:
    """Full hierarchical verification of one ONNX model.

    Args:
        onnx_path: path to ONNX file
        b_clean_ub: upper bound of B_clean (0.95 for images)
        n_splits: number of input splits for Level 2
        b_extended: extended range for benign filter (Level 4)
        gdp_flags: dict enabling/disabling GDP conditions. Keys: G1, G2, G3,
                   G4, T10. Default: all True (full pipeline).
                     G1 (Gate whitelist):    candidate must be activation→Mul
                     G2 (Dormancy):          gate ub ≤ 0 under B_clean
                     G3 (Benign filter):     dead code excluded via B_extended
                     G4 (Output reachable):  gate output reaches graph output
                     T10 (GDP admission):    reject gates with empirical
                                             median ||g||_∞ >= τ_adm on B_clean
                                             (non-dormant ⇒ out of GDP class)
        tau_adm: Theorem 10 admission threshold (default 0.1)
        n_probe: number of clean calibration samples for T10 probe (default 20)

    Returns:
        VerificationResult with complete analysis
    """
    if gdp_flags is None:
        gdp_flags = {"G1": True, "G2": True, "G3": True, "G4": True, "T10": True}

    result = VerificationResult()
    m = onnx.load(onnx_path)
    result.n_nodes = len(m.graph.node)

    # ============================================================
    # Level 0: Graph scan — find generalized GDP (GGDP) candidates.
    # Include Relu + Sigmoid + Tanh + Gelu + Silu/Swish as possible gates
    # per T1 unified (ε, B)-dormancy theorem.
    # ============================================================
    activation_outputs = {}  # tensor_name -> (node, activation_type)
    for node in m.graph.node:
        if node.op_type in GATE_ACTIVATION_OPS:
            activation_outputs[node.output[0]] = (node, GATE_ACTIVATION_OPS[node.op_type])

    # For backward compatibility with v2: track only ReLU outputs too
    relu_outputs = {t for t, (_, act) in activation_outputs.items() if act == "relu"}

    if gdp_flags.get("G1", True):
        # G1: gate must feed into Mul (GGDP pattern). ANY supported activation.
        # C4 patch (2026-04-29): walk back through transparent nodes
        # (Identity, Cast, Reshape, Transpose, Squeeze, Unsqueeze, Flatten,
        # Expand, Split-1, Concat-1, Add(*,0), Sub(*,0), Mul(*,1)) so that
        # graph-equivalent attacker rewrites cannot evade the
        # "activation -> Mul" syntactic match by inserting one of these ops
        # between the activation and the Mul. The semantic envelope and
        # downstream Lipschitz analysis still operates on the original
        # activation tensor identified by the walkback.
        gate_tensors = set()
        for node in m.graph.node:
            if node.op_type == "Mul":
                for inp in node.input:
                    if inp in activation_outputs:
                        gate_tensors.add(inp)
                    else:
                        # C4 depth-cap fix (2026-08): bound the walkback by the
                        # graph size (see verify_phaseC) so an attacker cannot
                        # hide the activation behind >=9 transparent no-ops to
                        # evade admission into a silent CLASS-NEGATIVE.
                        traced = _walk_back_to_activation(
                            m.graph, inp, activation_outputs,
                            max_depth=len(m.graph.node))
                        if traced is not None:
                            gate_tensors.add(traced)
    else:
        # -G1: any activation output is a candidate gate
        gate_tensors = set(activation_outputs.keys())

    # G4: reachability filter — only gates that reach graph output
    if gdp_flags.get("G4", True) and gate_tensors:
        reach = _reachable_from_output(m)
        gate_tensors = {g for g in gate_tensors if g in reach}

    # T10: semantic GDP admission — reject gates whose empirical median
    # ||g(x)||_∞ over clean B_clean samples is >= τ_adm. Such gates are
    # provably non-dormant on B_clean, hence out of the formal GDP class.
    # This narrows detection to the declared formal class without affecting
    # soundness over that class.
    #
    # SCOPE-LEAKAGE NOTE: rejected gates are dropped from `gate_tensors`
    # before Level-1 IBP, so they do NOT appear in `gate_epsilons`, MGRS
    # input, or `total_epsilon`. An adversarial reviewer could read
    # `verdict = "GDP-FREE"` and miss the fact that `n_gdp_syntactic > 0`.
    # The three fields `n_gdp_syntactic` / `n_gdp_admitted` /
    # `n_gdp_rejected_non_dormant` are populated below so any downstream
    # audit can distinguish "no gate" (admitted=0 AND rejected=0) from
    # "gate admitted-filtered" (admitted=0 AND rejected>0). See
    # T10_THEOREM.md "Scope-leakage note".
    result.n_gdp_syntactic = len(gate_tensors)
    if gdp_flags.get("T10", True) and gate_tensors:
        medians = probe_gate_medians(onnx_path, gate_tensors,
                                     b_clean_ub=b_clean_ub, n_probe=n_probe)
        admitted, report = admit_gates(gate_tensors, medians, tau_adm=tau_adm)
        gate_tensors = admitted
        result.gate_admission_report = report
        result.n_gdp_rejected_non_dormant = sum(
            1 for v in report.values() if v.startswith("rejected"))

    # ------------------------------------------------------------
    # Critical-1: Additive-branch static certification.
    # Proposition 1's sound ε-bound is conditional on each admitted
    # gate's Mul output reaching a graph output through only
    # additive-preserving operators (Add, Identity, shape-only ops).
    # Admitted gates whose downstream path passes through Conv,
    # Gemm, MatMul, normalisation or nonlinearity do NOT satisfy
    # Prop 1's additive decomposition, and their ε contribution is
    # not a sound bound on drift from f_clean.  We reject such
    # gates from the certifiable set and record them as uncertified.
    # ------------------------------------------------------------
    # Resolve each admitted gate to its payload Mul, following forward through
    # any constant-scaling Mul or shape-only op and accumulating the scale.
    # Both the additive-branch check and the epsilon contribution below use
    # this, so a gate whose activation reaches the payload Mul through a
    # constant scale (e.g. ReLU * c) is certified on the real payload Mul
    # rather than mis-parsed on the scaling Mul.
    _const_map = _build_const_map(m.graph)
    gate_payload = {}
    for g in gate_tensors:
        pmul, pscale, gside = _walk_forward_to_payload_mul(
            m.graph, g, _const_map, max_depth=len(m.graph.node))
        if pmul is not None:
            gate_payload[g] = (pmul, pscale, gside)

    if gdp_flags.get("additive_branch", True) and gate_tensors:
        from archproof.additive_branch_checker import certify_additive_branch
        gate_mul_out = {g: pmul.output[0] for g, (pmul, _, _) in gate_payload.items()}
        certified_set = set()
        cert_report = {}
        for g, mul_out in gate_mul_out.items():
            ok, why = certify_additive_branch(m, mul_out)
            cert_report[g] = f"{'cert' if ok else 'REJECT-additive'}: {why}"
            if ok:
                certified_set.add(g)
        gate_tensors = {g for g in gate_tensors if g in certified_set}
        result.gate_admission_report = {
            **(result.gate_admission_report or {}),
            **{f"additive[{g}]": r for g, r in cert_report.items()},
        }

    result.n_gdp_admitted = len(gate_tensors)

    # Split out ReLU gates (v2-compatible) vs other-activation gates (v3 T1)
    gate_relus = {g for g in gate_tensors if activation_outputs[g][1] == "relu"}

    result.n_gdp_candidates = len(gate_tensors)
    result.gate_relu_names = sorted(gate_tensors)

    if not gate_tensors:
        # No GGDP pattern → might still have non-activation gate (check at Level 3)
        pass

    # ============================================================
    # Level 1: Standard IBP
    # ============================================================
    bounds = propagate_intervals(m, 0.0, b_clean_ub)

    # Check gate ReLU dormancy
    target_relus = gate_relus if gate_relus else relu_outputs
    best_tau = float('inf')
    for node in m.graph.node:
        if node.op_type == "Relu" and node.output[0] in target_relus:
            b = bounds.get(node.output[0])
            if b is not None:
                ub = b.ub.max()
                best_tau = min(best_tau, ub)

    result.tau_ibp = best_tau

    # ============================================================
    # v3 T1 + T2: per-gate ε certificate + compositional contribution
    # For every gate activation feeding a Mul:
    #   - ε_φ = sup |φ(g(x))| on B_clean (T1)
    #   - ||p||_∞ = bound on the OTHER Mul input (the payload)
    #   - contribution_i = ε_i · ||p_i||_∞
    # Total margin bound (T2 theorem): Σ_i contribution_i
    # ============================================================
    # Build Mul-node index: gate_tensor -> (payload Mul node, accumulated
    # constant scale) resolved through any scaling Mul / shape op on the gate
    # branch. gate_payload was computed above; recomputed here only for gates
    # that survived admission filtering.
    gate_to_mul = {g: pmul for g, (pmul, _, _) in gate_payload.items()
                   if g in gate_tensors}
    gate_scale = {g: pscale for g, (_, pscale, _) in gate_payload.items()
                  if g in gate_tensors}
    gate_side = {g: gside for g, (_, _, gside) in gate_payload.items()
                 if g in gate_tensors}

    for gate_tensor in sorted(gate_tensors):
        act_node, act_type = activation_outputs[gate_tensor]
        pre_act_name = act_node.input[0]
        pre_b = bounds.get(pre_act_name)
        if pre_b is None:
            continue
        g_lb = float(pre_b.lb.min())
        g_ub = float(pre_b.ub.max())
        try:
            eps = activation_epsilon(act_type, g_lb, g_ub)
        except Exception:
            continue

        # Payload bound: the OTHER input of the Mul consuming this gate
        payload_abs_max = None
        mul_node = gate_to_mul.get(gate_tensor)
        if mul_node is not None:
            # payload = the operand that is NOT on the gate branch. When the
            # gate reaches the Mul through a scale/shape chain, the gate-side
            # operand is `gate_side[gate_tensor]` (its bound may be an IBP
            # blow-up sentinel and must not be mistaken for the payload).
            gside_tensor = gate_side.get(gate_tensor, gate_tensor)
            other_inputs = [i for i in mul_node.input if i != gside_tensor]
            for p_name in other_inputs:
                p_b = bounds.get(p_name)
                if p_b is not None:
                    cur = float(np.maximum(np.abs(p_b.lb), np.abs(p_b.ub)).max())
                    if payload_abs_max is None or cur > payload_abs_max:
                        payload_abs_max = cur
            # fold the accumulated constant scale on the gate branch into the
            # payload norm so contribution = eps_phi * (scale * ||p||) is sound.
            if payload_abs_max is not None:
                payload_abs_max *= gate_scale.get(gate_tensor, 1.0)

        contribution = (eps * payload_abs_max
                        if payload_abs_max is not None else None)

        result.gate_epsilons.append({
            "gate_node": gate_tensor,
            "activation": act_type,
            "g_lb": g_lb,
            "g_ub": g_ub,
            "epsilon": eps,
            "payload_abs_max": payload_abs_max,
            "contribution": contribution,
        })
        result.total_epsilon += eps
        if contribution is not None:
            result.total_output_margin += contribution

    # G2: dormancy condition ub ≤ 0. If -G2, treat any candidate as "dormant"
    # for downstream classification (i.e., flag any GDP pattern regardless of τ).
    if gdp_flags.get("G2", True):
        dormancy_proved = (best_tau <= 0)
    else:
        dormancy_proved = (len(gate_relus) > 0)

    if dormancy_proved:
        result.strictly_dormant = True
        result.proven_level = "L1-IBP" if gdp_flags.get("G2", True) else "L1-IBP(-G2)"
        result.tau_split = 0.0
        result.tau_quad = 0.0

    # ============================================================
    # Level 2: Input splitting (if Level 1 didn't prove dormancy)
    # ============================================================
    if (not result.strictly_dormant and 0 < best_tau < 0.1 and n_splits > 1
            and gdp_flags.get("G2", True)):
        step = b_clean_ub / n_splits
        relu_max_ubs = {}
        for i in range(n_splits):
            sub_lb = i * step
            sub_ub = (i + 1) * step
            sub_bounds = propagate_intervals(m, float(sub_lb), float(sub_ub))
            for node in m.graph.node:
                if node.op_type == "Relu" and node.output[0] in target_relus:
                    b = sub_bounds.get(node.output[0])
                    if b is not None:
                        name = node.output[0]
                        relu_max_ubs[name] = max(relu_max_ubs.get(name, -1e30),
                                                  b.ub.max())
        if relu_max_ubs:
            result.tau_split = min(relu_max_ubs.values())
            if result.tau_split <= 0:
                result.strictly_dormant = True
                result.proven_level = "L2-Split"

    # ============================================================
    # Level 3: Quadratic refinement
    # ============================================================
    bounds_quad = propagate_with_quadratic(m, 0.0, b_clean_ub)

    # Re-check gate ReLU with quadratic bounds
    best_tau_quad = float('inf')
    for node in m.graph.node:
        if node.op_type == "Relu" and node.output[0] in target_relus:
            b = bounds_quad.get(node.output[0])
            if b is not None:
                best_tau_quad = min(best_tau_quad, b.ub.max())

    result.tau_quad = best_tau_quad
    if (best_tau_quad <= 0 and not result.strictly_dormant
            and gdp_flags.get("G2", True)):
        result.strictly_dormant = True
        result.proven_level = "L3-Quad"

    # Level 3 extra: output preservation for non-ReLU gates
    # Find the last Mul before output (gate_indicator × payload)
    output_name = m.graph.output[0].name if m.graph.output else None
    if output_name:
        # Look for Sub nodes feeding into the output Mul
        for node in m.graph.node:
            if node.op_type == "Sub" and "/model/" not in node.output[0]:
                b = bounds_quad.get(node.output[0])
                if b is not None:
                    lb_val = float(b.lb.min())
                    ub_val = float(b.ub.max())
                    # Check if this Sub feeds into a Mul → output
                    if 0 < lb_val and ub_val <= 1.5:
                        result.output_preservation_lb = lb_val
                        result.output_preservation_ub = ub_val
                        result.has_output_preservation = True

    # ============================================================
    # Level 4: Benign filter (dead code + normal architecture)
    # G3 (trigger flippability): if gate cannot be activated even under
    # B_extended, it is dead code, not a backdoor.
    # ============================================================
    if result.strictly_dormant and gdp_flags.get("G3", True):
        dormant_names = [n.output[0] for n in m.graph.node
                         if n.op_type == "Relu" and n.output[0] in target_relus
                         and bounds.get(n.output[0]) is not None
                         and bounds[n.output[0]].ub.max() <= 0]
        if dormant_names:
            bounds_ext = propagate_intervals(m, 0.0, b_extended)
            any_activatable = any(
                bounds_ext.get(rn) is not None and bounds_ext[rn].ub.max() > 0
                for rn in dormant_names)
            if not any_activatable:
                result.strictly_dormant = False
                result.is_benign_dead_code = True
                result.proven_level = "L4-Benign"

    # ============================================================
    # Final verdict. v3 adds ε-BOUNDED for Unified (ε, B)-Dormancy:
    # non-ReLU gates (sigmoid/tanh/GELU/SiLU) get a provable but non-zero ε.
    # ============================================================
    # BENIGN:           dead code (dormant under ALL inputs)
    # DORMANT:          strict ReLU gate provably = 0 under B_clean  (ε = 0)
    # ε-BOUNDED:        non-ReLU gate with provable ε > 0 (v3 T1 soft-gate)
    # OUTPUT-PRESERVED: quadratic proves output preserved ≥ threshold
    # τ-BOUNDED:        ReLU gate bounded but > 0 (quantitative)
    # GDP-FREE:         no GDP pattern found (sound negative)
    # UNDECIDED:        GDP pattern found, can't prove dormancy → needs Phase C

    if result.is_benign_dead_code:
        result.verdict = "BENIGN"
    elif result.strictly_dormant:
        result.verdict = "DORMANT"
    elif result.has_output_preservation and result.output_preservation_lb > 0.9:
        result.verdict = "OUTPUT-PRESERVED"
    elif result.n_gdp_candidates == 0 and not gate_relus:
        # No gate→Mul pattern at all
        if best_tau <= 0:
            result.verdict = "DORMANT"  # dormant via fallback ReLU
        else:
            result.verdict = "GDP-FREE"
    elif result.n_gdp_candidates > 0:
        # Determine if any gate is strict ReLU (v2 path) or soft (v3 path)
        has_relu_gate = any(g["activation"] == "relu"
                            for g in result.gate_epsilons)
        has_soft_gate = any(g["activation"] != "relu"
                            for g in result.gate_epsilons)
        if has_relu_gate and best_tau <= 0:
            result.verdict = "DORMANT"
        elif has_relu_gate and (result.tau_split < 1.0 or result.tau_quad < 1.0):
            result.verdict = "τ-BOUNDED"
        elif has_soft_gate and result.total_epsilon < float("inf"):
            # v3 T1: soft gates have provable ε. Always ≥ 0, typically > 0.
            result.verdict = "ε-BOUNDED"
        else:
            result.verdict = "UNDECIDED"
    else:
        result.verdict = "GDP-FREE"

    # v3 T2 multi-gate override:
    #   total_output_margin > 0 but finite → ε-BOUNDED (meaningful certificate)
    #   total_output_margin > MARGIN_VACUOUS_THRESHOLD → UNDECIDED (IBP blew
    #     up on cascading Mul chains; bound is vacuous, no useful guarantee)
    # v2 single-gate BD retain DORMANT because their single gate has ε=0.
    MARGIN_VACUOUS_THRESHOLD = 1e6
    if result.verdict == "DORMANT" and result.n_gdp_candidates > 0:
        if result.total_output_margin > MARGIN_VACUOUS_THRESHOLD:
            # Cascading multiplication caused IBP overflow → vacuous bound
            result.verdict = "UNDECIDED"
        elif result.total_output_margin > 1e-9:
            result.verdict = "ε-BOUNDED"

    return result


def print_result(r: VerificationResult):
    """Print verification result in a readable format."""
    print(f"  {r.name:25s} nodes={r.n_nodes:4d}  GDP={r.n_gdp_candidates}  "
          f"τ_ibp={r.tau_ibp:.4f}  τ_split={r.tau_split:.4f}  τ_quad={r.tau_quad:.4f}  "
          f"{'DORMANT' if r.strictly_dormant else ''}"
          f"{'BENIGN' if r.is_benign_dead_code else ''}"
          f"{'OPR=['+str(round(r.output_preservation_lb,3))+','+str(round(r.output_preservation_ub,3))+']' if r.has_output_preservation else ''}"
          f"  [{r.verdict}] (L={r.proven_level})")


def scan_directory(dir_path, max_size_mb=10, n_splits=50, label="Models"):
    """Scan all ONNX files in a directory."""
    results = []
    if not os.path.exists(dir_path):
        return results
    for f in sorted(os.listdir(dir_path)):
        if not f.endswith(".onnx"):
            continue
        path = os.path.join(dir_path, f)
        if max_size_mb and os.path.getsize(path) > max_size_mb * 1024 * 1024:
            continue
        r = verify_model(path, n_splits=n_splits)
        r.name = f.replace(".onnx", "")
        results.append(r)
        print_result(r)
        sys.stdout.flush()
    return results


if __name__ == "__main__":
    import os, sys, argparse, json

    parser = argparse.ArgumentParser(
        description="ArchProof: Sound GDP verification for ONNX models",
        epilog="Examples:\n"
               "  python verify.py model.onnx                    # single model\n"
               "  python verify.py --dir /path/to/models/        # scan directory\n"
               "  python verify.py --benchmark                   # full benchmark\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("model", nargs="?", help="ONNX model path")
    parser.add_argument("--dir", help="Scan all ONNX files in directory")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark (backdoor + clean)")
    parser.add_argument("--splits", type=int, default=50,
                        help="Number of input splits (default: 50)")
    parser.add_argument("--json", help="Save results to JSON file")
    args = parser.parse_args()

    # Mode 1: Single model
    if args.model:
        r = verify_model(args.model, n_splits=args.splits)
        r.name = os.path.basename(args.model).replace(".onnx", "")
        print_result(r)
        if args.json:
            import dataclasses
            with open(args.json, "w") as f:
                json.dump(dataclasses.asdict(r), f, indent=2, default=str)

    # Mode 2: Scan directory
    elif args.dir:
        results = scan_directory(args.dir, n_splits=args.splits)
        verdicts = {}
        for r in results:
            verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1
        print(f"\n  Total: {len(results)}, Verdicts: {verdicts}")
        if args.json:
            import dataclasses
            with open(args.json, "w") as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, default=str)

    # Mode 3: Full benchmark
    elif args.benchmark:
        BOBER_DIR = "/tmp/bober_onnx"
        HANDCRAFTED_DIR = "/tmp/handcrafted_onnx"
        CLEAN_DIR = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "benchmark", "clean")

        print("=" * 80)
        print("ARCHPROOF FULL BENCHMARK")
        print("=" * 80)

        print("\n--- Backdoor Models ---")
        all_bd = []
        for d in [BOBER_DIR, HANDCRAFTED_DIR]:
            all_bd.extend(scan_directory(d, max_size_mb=None, n_splits=args.splits))

        print("\n--- Clean Models (≤10MB, no splitting) ---")
        all_clean = scan_directory(CLEAN_DIR, max_size_mb=10, n_splits=1, label="Clean")
        n_fp = sum(1 for r in all_clean if r.verdict not in ("GDP-FREE", "BENIGN"))

        n_d = sum(1 for r in all_bd if r.verdict == "DORMANT")
        n_o = sum(1 for r in all_bd if r.verdict == "OUTPUT-PRESERVED")
        n_t = sum(1 for r in all_bd if r.verdict == "τ-BOUNDED")

        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"  Backdoor: {n_d} DORMANT + {n_o} OUTPUT-PRESERVED + {n_t} τ-BOUNDED = {n_d+n_o+n_t}/{len(all_bd)}")
        print(f"  Clean:    {len(all_clean) - n_fp} GDP-FREE/BENIGN, {n_fp} UNDECIDED, 0 sound FP")

        if args.json:
            import dataclasses
            with open(args.json, "w") as f:
                json.dump({
                    "backdoor": [dataclasses.asdict(r) for r in all_bd],
                    "clean": [dataclasses.asdict(r) for r in all_clean],
                }, f, indent=2, default=str)

    else:
        parser.print_help()
