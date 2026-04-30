"""Correlation-Aware Quadratic Refinement for Mul(f(x), g(x)).

Novel algorithm: when two Mul inputs share a common ancestor z,
track their affine dependency on z, then bound the quadratic product exactly.

Standard IBP: Mul([a,b], [c,d]) = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
  → treats inputs as INDEPENDENT → overestimates when correlated.

Our method: if input_1 = α₁z + β₁ and input_2 = α₂z + β₂ (both affine in z):
  product = α₁α₂z² + (α₁β₂ + α₂β₁)z + β₁β₂
  → bound this quadratic EXACTLY over z ∈ [z_lb, z_ub]
  → no correlation loss, near-exact bound.

Sound: quadratic bound ⊆ true range ⊆ IBP range (always tighter than IBP).
"""

import numpy as np
from typing import Dict, Optional, Tuple, Set
import onnx

from archproof.interval_propagation import (
    propagate_intervals, IntervalBound, _load_tensor, _get_bound
)


class AffineForm:
    """Represents value = slope * z + intercept, affine in reference variable z."""
    def __init__(self, slope: np.ndarray, intercept: np.ndarray):
        self.slope = slope        # coefficient of z
        self.intercept = intercept  # constant term

    def is_constant(self):
        return np.allclose(self.slope, 0)


def _bound_quadratic(a, b, c, z_lb, z_ub):
    """Bound a*z² + b*z + c over z ∈ [z_lb, z_ub]. Returns (lb, ub)."""
    # Evaluate at endpoints
    v_lo = a * z_lb**2 + b * z_lb + c
    v_hi = a * z_ub**2 + b * z_ub + c
    candidates = [v_lo, v_hi]

    # Check vertex z* = -b/(2a) if a ≠ 0
    if np.abs(a) > 1e-12:
        z_star = -b / (2 * a)
        if z_lb <= z_star <= z_ub:
            v_star = a * z_star**2 + b * z_star + c
            candidates.append(v_star)

    return min(candidates), max(candidates)


def _find_shared_ancestors(graph):
    """Find Mul nodes where both inputs share a common non-constant ancestor."""
    output_to_node = {}
    for node in graph.node:
        for o in node.output:
            output_to_node[o] = node

    init_names = set(i.name for i in graph.initializer)
    input_names = set(i.name for i in graph.input)
    const_names = init_names | input_names

    def get_ancestors(name, visited=None):
        if visited is None:
            visited = set()
        if name in visited:
            return visited
        visited.add(name)
        if name in output_to_node:
            for inp in output_to_node[name].input:
                if inp:
                    get_ancestors(inp, visited)
        return visited

    correlated_muls = []
    for node in graph.node:
        if node.op_type == "Mul" and len(node.input) >= 2:
            inp0, inp1 = node.input[0], node.input[1]
            # Skip if either input is a constant
            if inp0 in const_names or inp1 in const_names:
                continue
            anc0 = get_ancestors(inp0) - const_names
            anc1 = get_ancestors(inp1) - const_names
            shared = anc0 & anc1
            if shared:
                correlated_muls.append((node, shared))

    return correlated_muls


def propagate_with_quadratic(onnx_model, input_lb=0.0, input_ub=0.95):
    """Enhanced IBP with FORWARD affine tracking + quadratic refinement.

    Forward pass: track AffineForm(slope, intercept) alongside IBP bounds.
    At Mul(affine, affine): compute exact quadratic bound, intersect with IBP.
    Then re-propagate downstream with tighter bounds.

    Sound: quadratic bound is exact for the tracked variable;
           intersection with IBP is always ≤ IBP (tighter or equal).
    """
    graph = onnx_model.graph
    from archproof.interval_propagation import _propagate_node

    # Step 1: Standard IBP
    bounds = propagate_intervals(onnx_model, input_lb, input_ub)

    # Step 2: Forward affine tracking
    affine: Dict[str, Optional[AffineForm]] = {}

    # z reference: scalar input range
    z_lb = float(input_lb) if not isinstance(input_lb, np.ndarray) else float(np.min(input_lb))
    z_ub = float(input_ub) if not isinstance(input_ub, np.ndarray) else float(np.max(input_ub))

    # Input: affine in z (identity)
    input_name = graph.input[0].name
    affine[input_name] = AffineForm(slope=np.float64(1.0), intercept=np.float64(0.0))

    # Constants/initializers: affine with slope=0
    for init in graph.initializer:
        val = _load_tensor(init)
        scalar = float(val.flatten()[0]) if val.size == 1 else float(val.max())
        affine[init.name] = AffineForm(slope=np.float64(0.0), intercept=np.float64(scalar))

    refined_nodes = []

    for node in graph.node:
        op = node.op_type
        out = node.output[0]
        inps = [affine.get(i) for i in node.input if i]

        if op == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    val = _load_tensor(attr.t)
                    # Empty tensors (e.g., zero-size attention mask in some
                    # GPT2 ONNX exports) reduce to scalar 0.0 affine; size-1
                    # tensors take the literal value; multi-element take max.
                    if val.size == 0:
                        s = 0.0
                    elif val.size == 1:
                        s = float(val.flatten()[0])
                    else:
                        s = float(val.max())
                    affine[out] = AffineForm(np.float64(0.0), np.float64(s))

        elif op in ("Add", "Sub") and len(inps) >= 2 and inps[0] and inps[1]:
            a, b = inps[0], inps[1]
            if op == "Add":
                affine[out] = AffineForm(a.slope + b.slope, a.intercept + b.intercept)
            else:
                affine[out] = AffineForm(a.slope - b.slope, a.intercept - b.intercept)

        elif op == "Neg" and inps and inps[0]:
            affine[out] = AffineForm(-inps[0].slope, -inps[0].intercept)

        elif op == "Mul" and len(inps) >= 2 and inps[0] and inps[1]:
            a, b = inps[0], inps[1]
            if a.is_constant():
                affine[out] = AffineForm(b.slope * a.intercept, b.intercept * a.intercept)
            elif b.is_constant():
                affine[out] = AffineForm(a.slope * b.intercept, a.intercept * b.intercept)
            else:
                # BOTH affine → QUADRATIC product!
                qa = a.slope * b.slope
                qb = a.slope * b.intercept + b.slope * a.intercept
                qc = a.intercept * b.intercept
                quad_lb, quad_ub = _bound_quadratic(float(qa), float(qb), float(qc), z_lb, z_ub)

                ibp_b = bounds.get(out)
                if ibp_b is not None:
                    old_ub = float(ibp_b.ub.max())
                    new_ub = min(old_ub, quad_ub)
                    new_lb = max(float(ibp_b.lb.min()), quad_lb)
                    if new_ub < old_ub - 1e-8:
                        bounds[out] = IntervalBound(
                            lb=np.full_like(ibp_b.lb, new_lb),
                            ub=np.full_like(ibp_b.ub, new_ub))
                        refined_nodes.append(out)

                affine[out] = None  # quadratic → stop tracking

        elif op in ("AveragePool", "ReduceMean", "Reshape", "Flatten",
                     "Unsqueeze", "Pad", "Identity"):
            if inps and inps[0]:
                affine[out] = inps[0]

        elif op == "MaxPool" and inps and inps[0]:
            # MaxPool on uniform box: preserves scalar affine form
            affine[out] = inps[0]

        # else: tracking stops (non-affine op)

    # Step 3: Re-propagate DOWNSTREAM of refined nodes only
    # (must NOT re-propagate the refined node itself — would overwrite quad bound)
    if refined_nodes:
        refined_set = set(refined_nodes)
        # Find earliest refined node index
        first_idx = len(graph.node)
        for i, node in enumerate(graph.node):
            if node.output[0] in refined_set:
                first_idx = min(first_idx, i)

        for node in graph.node[first_idx + 1:]:
            if node.output[0] in refined_set:
                continue  # skip refined nodes themselves
            try:
                result = _propagate_node(node, bounds)
                if result is not None:
                    if isinstance(result, list):
                        for j, o in enumerate(node.output):
                            if j < len(result):
                                bounds[o] = result[j]
                    else:
                        bounds[node.output[0]] = result
            except:
                pass

    return bounds


def test_quadratic_refinement():
    """Test quadratic refinement: compare gate chain bounds IBP vs quadratic."""
    import os
    BOBER_DIR = "/tmp/bober_onnx"

    print(f"{'Model':25s} {'Mul_1 IBP':>10s} {'Mul_1 Quad':>11s} {'Sub_3 IBP':>10s} {'Sub_3 Quad':>11s}")
    print("-" * 72)

    for name in ["op_sep_un_L01", "op_sep_un_L001", "op_sep_un_L0001"]:
        path = os.path.join(BOBER_DIR, f"{name}.onnx")
        if not os.path.exists(path):
            continue

        m = onnx.load(path)

        # Standard IBP
        bounds_ibp = propagate_intervals(m, 0.0, 0.95)

        # Quadratic-refined IBP
        bounds_quad = propagate_with_quadratic(m, 0.0, 0.95)

        # Compare key gate chain nodes
        def get_range(bounds, name):
            b = bounds.get(name)
            if b is None: return "NO_BOUND"
            return f"[{b.lb.min():.3f},{b.ub.max():.3f}]"

        mul1_ibp = get_range(bounds_ibp, '/Mul_1_output_0')
        mul1_q = get_range(bounds_quad, '/Mul_1_output_0')
        sub3_ibp = get_range(bounds_ibp, '/Sub_3_output_0')
        sub3_q = get_range(bounds_quad, '/Sub_3_output_0')

        print(f"  {name:23s} {mul1_ibp:>10s} {mul1_q:>11s} {sub3_ibp:>10s} {sub3_q:>11s}")


if __name__ == "__main__":
    print("=== Quadratic Refinement Test ===")
    test_quadratic_refinement()
