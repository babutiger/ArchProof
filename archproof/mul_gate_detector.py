"""Detect Mul-gating patterns: Mul where one input acts as binary {0,1} indicator.

This catches the Bober-Irizar backdoor pattern:
  trigger_detector(x) → indicator ∈ {0,1}
  indicator × value → gated output (dormant when indicator=0, active when indicator=1)
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from typing import List, Tuple, Dict

from . import config


def find_mul_gates(onnx_model: onnx.ModelProto,
                   n_samples: int = 100) -> List[Tuple[onnx.NodeProto, int]]:
    """Find Mul nodes where one input acts as a binary indicator.

    For each Mul node, sample random inputs and check if either Mul input
    has values approximately in {0, 1} (binary gate pattern).

    Returns:
        List of (mul_node, indicator_input_index) — which Mul input is the indicator
    """
    graph = onnx_model.graph

    # Get input shape
    input_info = graph.input[0]
    input_shape = []
    for dim in input_info.type.tensor_type.shape.dim:
        input_shape.append(dim.dim_value if dim.dim_value > 0 else 1)

    # Build ONNX Runtime session to get intermediate values
    # Add ALL intermediate tensors as outputs
    mul_nodes = [n for n in graph.node if n.op_type == "Mul"]
    if not mul_nodes:
        return []

    # Collect all tensor names we need to observe (Mul inputs)
    tensor_names_to_observe = set()
    for node in mul_nodes:
        for inp in node.input:
            tensor_names_to_observe.add(inp)

    # Create modified model with extra outputs
    extra_outputs = []
    existing_outputs = {o.name for o in graph.output}
    for name in tensor_names_to_observe:
        if name not in existing_outputs:
            extra_outputs.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

    model_with_intermediates = onnx.ModelProto()
    model_with_intermediates.CopyFrom(onnx_model)
    model_with_intermediates.graph.output.extend(extra_outputs)

    try:
        sess = ort.InferenceSession(model_with_intermediates.SerializeToString(),
                                    providers=["CPUExecutionProvider"])
    except Exception:
        return []  # can't create session, skip

    # Sample random inputs and check Mul input ranges
    mul_gates = []
    for node in mul_nodes:
        input_a_binary_count = 0
        input_b_binary_count = 0

        for _ in range(n_samples):
            x = np.random.rand(1, *input_shape[1:]).astype(np.float32) * \
                (config.B_CLEAN_PIXEL_MAX - config.B_CLEAN_PIXEL_MIN) + config.B_CLEAN_PIXEL_MIN

            try:
                outputs = sess.run(list(tensor_names_to_observe), {sess.get_inputs()[0].name: x})
                output_dict = dict(zip(tensor_names_to_observe, outputs))

                val_a = output_dict.get(node.input[0])
                val_b = output_dict.get(node.input[1])

                if val_a is not None:
                    # Check if values are approximately binary {0, 1}
                    if _is_approximately_binary(val_a):
                        input_a_binary_count += 1

                if val_b is not None:
                    if _is_approximately_binary(val_b):
                        input_b_binary_count += 1
            except Exception:
                continue

        # If one input is binary in >80% of samples → it's an indicator
        threshold = n_samples * 0.8
        if input_a_binary_count > threshold:
            mul_gates.append((node, 0))
        elif input_b_binary_count > threshold:
            mul_gates.append((node, 1))

    return mul_gates


def _is_approximately_binary(arr: np.ndarray, tol: float = 0.05) -> bool:
    """Check if all values in array are approximately 0 or 1."""
    close_to_0 = np.abs(arr) < tol
    close_to_1 = np.abs(arr - 1.0) < tol
    return bool(np.all(close_to_0 | close_to_1))
