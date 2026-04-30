"""Static additive-branch certifier for add-GDP gates.

Given an admitted activation→Mul gate with Mul output tensor `mul_out`,
this module statically checks whether the downstream path from `mul_out`
to any graph output uses only value-preserving additive-composition
operators. If so, the gate satisfies Proposition 1's additive-branch
precondition and its ε contribution is a sound bound on the gate's drift
to f_clean. Otherwise, the admitted-gate claim is *not* sound (Prop 1
doesn't apply) and the verifier must emit UNDECIDED for that gate.

Allowed ops (value-preserving on the Mul side of an Add):
  - Add:      the Mul output is one operand; the OTHER operand is part
              of f_clean and does not scale the gate contribution.
  - Identity, Reshape, Transpose, Flatten, Squeeze, Unsqueeze: do not
              alter scalar values, only layout.

Rejected ops (break additive decomposition):
  - Conv, ConvTranspose, Gemm, MatMul, MatMulNBits: arbitrary linear
              combinations that couple Mul-output to other tensors.
  - Mul, Div, Pow: multiplicative coupling.
  - BatchNormalization, InstanceNormalization, LayerNormalization,
    GroupNormalization: affine rescale depending on running statistics.
  - Activation ops (Relu/Sigmoid/Tanh/...): nonlinear, break linearity.
  - ReduceMean, ReduceSum, ReduceMax, ReduceMin, Softmax: aggregation
              across dims, couples with other coordinates.
  - Concat: introduces structural dependencies on other tensors.
  - Sub with mul_out on the *right*: breaks additive decomp; on the
              left is fine (treated as Add with negated payload).

The checker is a conservative BFS: reject on any unknown operator.
"""
from __future__ import annotations
from typing import Set, Dict, List, Tuple
import onnx


ADDITIVE_PRESERVING_OPS = {
    "Add",           # gate-side of Add: out = mul_out + clean_branch
    "Identity",
    "Reshape",
    "Transpose",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "Expand",        # shape broadcast; values unchanged
}


def _build_consumers(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    """Map tensor name to list of nodes consuming it."""
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    return consumers


def certify_additive_branch(model: onnx.ModelProto, mul_out: str) -> Tuple[bool, str]:
    """Check whether `mul_out` reaches a graph output through only additive-
    preserving operators.

    Returns (certified, reason).
      certified = True  iff every downstream path of `mul_out` is additive
      certified = False iff some path passes through a non-preserving op,
                          in which case the reason names the first violator.
    """
    graph = model.graph
    output_names = {o.name for o in graph.output}
    consumers = _build_consumers(graph)

    # BFS from mul_out downstream
    visited: Set[str] = set()
    frontier: List[str] = [mul_out]
    while frontier:
        tensor = frontier.pop()
        if tensor in visited:
            continue
        visited.add(tensor)

        if tensor in output_names:
            # A path reached graph output via allowed ops only
            continue

        consumer_nodes = consumers.get(tensor, [])
        if not consumer_nodes:
            # Dead end; not a graph output. This is acceptable — it means
            # the gate doesn't actually influence the graph output via
            # this path (trivially additive: the path contributes 0).
            continue

        for node in consumer_nodes:
            if node.op_type not in ADDITIVE_PRESERVING_OPS:
                return False, (f"path through {node.op_type} "
                               f"({node.name}) breaks additive decomposition")

            if node.op_type == "Add":
                # Safe: gate-side of Add contributes additively.
                # Downstream = Add's output.
                for out in node.output:
                    frontier.append(out)
            elif node.op_type == "Sub":
                # Fine only if mul_out is Sub's first input (minuend):
                # out = mul_out - other → additive.
                if node.input and node.input[0] == tensor:
                    for out in node.output:
                        frontier.append(out)
                else:
                    return False, "Mul-output on RHS of Sub breaks sign"
            else:
                # Identity / Reshape / Transpose / Flatten / Squeeze /
                # Unsqueeze / Expand: value-preserving layout ops.
                for out in node.output:
                    frontier.append(out)

    return True, "additive-branch certified"


def certify_admitted_gates(model: onnx.ModelProto,
                            gate_mul_outputs: Dict[str, str]
                            ) -> Dict[str, Tuple[bool, str]]:
    """Batch-certify multiple admitted gates.

    Args:
      gate_mul_outputs: map gate_activation_tensor -> corresponding Mul
                         output tensor name.

    Returns:
      map gate_activation_tensor -> (certified, reason).
    """
    return {
        g: certify_additive_branch(model, mul_out)
        for g, mul_out in gate_mul_outputs.items()
    }
