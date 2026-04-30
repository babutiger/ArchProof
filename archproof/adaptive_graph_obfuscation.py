#!/usr/bin/env python3
"""C4: Adaptive graph-equivalent obfuscation suite.

For every in-class backdoored ONNX, generate transformed graphs in which
an attacker has inserted ONE semantics-preserving operator between the
activation node and the Mul gate, breaking ArchProof's strict
"activation.output -> Mul.input" syntactic match.

Each transformation produces a graph that computes the same function as
the original (when interpreted faithfully), so a sound verifier MUST
either (a) still detect the gate (robust under graph rewriting), or
(b) emit UNCERTIFIED (fail-closed). Returning add-DGP-CLASS-NEGATIVE
on any of these is a real soundness leak.

Transformations:
    W1 Identity            : add Identity(act)
    W2 CastSame            : add Cast(act, to=float)
    W3 AddZero             : add Add(act, 0_initializer)
    W4 MulOne              : add Mul(act, 1_initializer)
    W5 ReshapeRoundtrip    : add Reshape(act, original_shape) wrapped twice
    W6 OperandSwap         : swap Mul's two operands (commutativity)
    W7 SplitConcat         : add Split-Concat passthrough
    W8 CastFp16Roundtrip   : add Cast(fp32->fp16)->Cast(fp16->fp32)

Output: /tmp/c4_obfuscated/<bd_name>__<wrap>.onnx
        ${ARCHPROOF_ROOT}/benchmark/c4_obfuscation_index.json
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = os.environ.get(
    "ARCHPROOF_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
sys.path.insert(0, ROOT)

OUT_DIR = "/tmp/c4_obfuscated"
INDEX_JSON = os.path.join(ROOT, "benchmark", "c4_obfuscation_index.json")

# Activation ops that ArchProof recognises as gate activations
GATE_ACT_OPS = {
    "Sigmoid", "Tanh", "Relu", "Elu", "Selu",
    "LeakyRelu", "HardSigmoid", "HardSwish",
    "Softplus", "Mish", "Gelu",
}


def _new_name(graph, base: str) -> str:
    existing = {n.name for n in graph.node} | \
               {n.output[0] for n in graph.node if n.output} | \
               {i.name for i in graph.initializer}
    i = 0
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _find_gate_mul_pairs(model: onnx.ModelProto) -> List[Tuple[onnx.NodeProto, int, onnx.NodeProto]]:
    """Return (act_node, mul_input_index, mul_node) for every direct
    activation->Mul edge in the model's main graph."""
    g = model.graph
    act_outputs = {n.output[0]: n for n in g.node if n.op_type in GATE_ACT_OPS}
    pairs = []
    for n in g.node:
        if n.op_type == "Mul":
            for idx, inp in enumerate(n.input):
                if inp in act_outputs:
                    pairs.append((act_outputs[inp], idx, n))
    return pairs


def _add_initializer(graph, name: str, value: np.ndarray) -> str:
    init = numpy_helper.from_array(value.astype(np.float32), name=name)
    graph.initializer.append(init)
    return name


def _replace_mul_input(mul_node, idx: int, new_name: str) -> None:
    inputs = list(mul_node.input)
    inputs[idx] = new_name
    del mul_node.input[:]
    mul_node.input.extend(inputs)


def _insert_before(graph, target_node, new_nodes) -> None:
    """Insert new_nodes into graph.node BEFORE target_node (preserving order
    among new_nodes).

    Use protobuf RepeatedCompositeContainer.insert() so that references to
    target_node (and other existing nodes) remain valid for downstream
    edits — del+extend would silently copy the protos.
    """
    target_idx = None
    for i, n in enumerate(graph.node):
        if n is target_node:
            target_idx = i
            break
    if target_idx is None:
        # Fallback by name match
        for i, n in enumerate(graph.node):
            if n.name == target_node.name and list(n.input) == list(target_node.input):
                target_idx = i
                break
    if target_idx is None:
        graph.node.extend(new_nodes)
        return
    for offset, nn in enumerate(new_nodes):
        graph.node.insert(target_idx + offset, nn)


def _wrap_one_pair(model: onnx.ModelProto, act_node, mul_idx, mul_node, wrap: str) -> bool:
    """Mutate model in place to insert wrap between act and mul. Returns
    True on success, False if the wrap is not applicable to this pair."""
    g = model.graph
    src = act_node.output[0]
    new_out = _new_name(g, f"{src}_{wrap}")

    if wrap == "W1_Identity":
        new_node = helper.make_node("Identity", inputs=[src], outputs=[new_out],
                                     name=_new_name(g, "id"))
        _insert_before(g, mul_node, [new_node])
        _replace_mul_input(mul_node, mul_idx, new_out)
        return True

    if wrap == "W2_CastSame":
        new_node = helper.make_node("Cast", inputs=[src], outputs=[new_out],
                                     to=TensorProto.FLOAT,
                                     name=_new_name(g, "cast_same"))
        _insert_before(g, mul_node, [new_node])
        _replace_mul_input(mul_node, mul_idx, new_out)
        return True

    if wrap == "W3_AddZero":
        zero_name = _add_initializer(g, _new_name(g, "zero_const"),
                                       np.zeros((1,), dtype=np.float32))
        new_node = helper.make_node("Add", inputs=[src, zero_name],
                                     outputs=[new_out],
                                     name=_new_name(g, "add_zero"))
        _insert_before(g, mul_node, [new_node])
        _replace_mul_input(mul_node, mul_idx, new_out)
        return True

    if wrap == "W4_MulOne":
        one_name = _add_initializer(g, _new_name(g, "one_const"),
                                      np.ones((1,), dtype=np.float32))
        new_node = helper.make_node("Mul", inputs=[src, one_name],
                                     outputs=[new_out],
                                     name=_new_name(g, "mul_one"))
        _insert_before(g, mul_node, [new_node])
        _replace_mul_input(mul_node, mul_idx, new_out)
        return True

    if wrap == "W5_ReshapeRoundtrip":
        # Insert Shape -> Reshape(act, shape) so the graph is shape-correct.
        shape_out = _new_name(g, f"{src}_shape")
        shape_node = helper.make_node("Shape", inputs=[src], outputs=[shape_out],
                                       name=_new_name(g, "shape"))
        reshape_node = helper.make_node("Reshape", inputs=[src, shape_out],
                                          outputs=[new_out],
                                          name=_new_name(g, "reshape_rt"))
        _insert_before(g, mul_node, [shape_node, reshape_node])
        _replace_mul_input(mul_node, mul_idx, new_out)
        return True

    if wrap == "W6_OperandSwap":
        # Swap the two Mul operands: equivalent under commutativity.
        if len(mul_node.input) != 2:
            return False
        a, b = mul_node.input[0], mul_node.input[1]
        del mul_node.input[:]
        mul_node.input.extend([b, a])
        return True

    if wrap == "W7_SplitConcat":
        # Use opset-13 form: pass `split` as a second input initializer so
        # we don't depend on opset-18 num_outputs attribute.  A 2-way split
        # along axis -1 with concat back is identity for any tensor with
        # an even last dim.  Where the last dim is odd or 1 we fall back
        # to a single-chunk split (split=[D]).
        # Determine target axis size by adding a 2-way split using Shape +
        # Slice runtime is overkill; we instead just emit a single-chunk
        # split (one output) with an explicit `split` initializer of [-1]
        # which onnxruntime resolves at runtime.
        split_outs = [_new_name(g, f"{src}_split0")]
        # Try opset-13 form: split lengths via input
        split_init = _add_initializer(
            g, _new_name(g, "split_lens"),
            np.array([-1], dtype=np.int64).astype(np.float32),
        )
        # Re-write the initializer as INT64 explicitly (helper above coerces
        # to FLOAT32 by default; we override with a fresh raw initializer)
        for init in g.initializer:
            if init.name == split_init:
                init.data_type = TensorProto.INT64
                init.raw_data = np.array([-1], dtype=np.int64).tobytes()
                # purge the float_data we accidentally seeded above
                del init.float_data[:]
                break
        split_node = helper.make_node(
            "Split", inputs=[src, split_init], outputs=split_outs,
            axis=-1, name=_new_name(g, "split"),
        )
        concat_node = helper.make_node(
            "Concat", inputs=split_outs, outputs=[new_out], axis=-1,
            name=_new_name(g, "concat"),
        )
        _insert_before(g, mul_node, [split_node, concat_node])
        _replace_mul_input(mul_node, mul_idx, new_out)
        return True

    if wrap == "W8_CastFp16Roundtrip":
        mid = _new_name(g, f"{src}_fp16")
        cast_down = helper.make_node("Cast", inputs=[src], outputs=[mid],
                                       to=TensorProto.FLOAT16,
                                       name=_new_name(g, "cast_fp16"))
        cast_up = helper.make_node("Cast", inputs=[mid], outputs=[new_out],
                                     to=TensorProto.FLOAT,
                                     name=_new_name(g, "cast_fp32"))
        _insert_before(g, mul_node, [cast_down, cast_up])
        _replace_mul_input(mul_node, mul_idx, new_out)
        return True

    raise ValueError(f"unknown wrap: {wrap}")


WRAPS = [
    "W1_Identity", "W2_CastSame", "W3_AddZero", "W4_MulOne",
    "W5_ReshapeRoundtrip", "W6_OperandSwap",
    "W7_SplitConcat", "W8_CastFp16Roundtrip",
]


def obfuscate(src_onnx: str, wrap: str, dst_onnx: str) -> Optional[str]:
    """Load model, wrap all activation->Mul pairs with `wrap`, save to dst.
    Returns dst on success, None if no gate pairs found."""
    model = onnx.load(src_onnx)
    pairs = _find_gate_mul_pairs(model)
    if not pairs:
        return None
    wrapped = 0
    for act, idx, mul in pairs:
        if _wrap_one_pair(model, act, idx, mul, wrap):
            wrapped += 1
    if wrapped == 0:
        return None
    # Bump opset to support Cast attributes if needed; keep original otherwise.
    try:
        onnx.checker.check_model(model, full_check=False)
    except Exception as exc:
        print(f"  [!] {wrap} check failed on {os.path.basename(src_onnx)}: {exc}")
    os.makedirs(os.path.dirname(dst_onnx), exist_ok=True)
    onnx.save(model, dst_onnx)
    return dst_onnx


# 11 in-class backdoor models from the benchmark
IN_CLASS_BD = [
    ("op_int_tar",        "/tmp/bober_onnx/op_int_tar.onnx"),
    ("op_int_tar_L0001",  "/tmp/bober_onnx/op_int_tar_L0001.onnx"),
    ("op_int_tar_L001",   "/tmp/bober_onnx/op_int_tar_L001.onnx"),
    ("op_int_tar_L01",    "/tmp/bober_onnx/op_int_tar_L01.onnx"),
    ("op_int_un",         "/tmp/bober_onnx/op_int_un.onnx"),
    ("op_sep_tar",        "/tmp/bober_onnx/op_sep_tar.onnx"),
    ("op_sep_un",         "/tmp/bober_onnx/op_sep_un.onnx"),
    ("op_sha_tar",        "/tmp/bober_onnx/op_sha_tar.onnx"),
    ("op_sha_un",         "/tmp/bober_onnx/op_sha_un.onnx"),
    ("H2_AvgPoolGated",   "/tmp/handcrafted_onnx/H2_AvgPoolGated.onnx"),
    ("H3_MulIndicatorGated", "/tmp/handcrafted_onnx/H3_MulIndicatorGated.onnx"),
]


def main():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    index = []
    for bd_name, src in IN_CLASS_BD:
        if not os.path.isfile(src):
            print(f"[skip] {bd_name}: {src} not found")
            continue
        for wrap in WRAPS:
            dst = os.path.join(OUT_DIR, f"{bd_name}__{wrap}.onnx")
            try:
                got = obfuscate(src, wrap, dst)
                if got:
                    sz = os.path.getsize(got)
                    print(f"  [{bd_name} / {wrap}] -> {got} ({sz} bytes)")
                    index.append({"model": bd_name, "wrap": wrap,
                                  "src": src, "out": got, "size": sz})
                else:
                    print(f"  [{bd_name} / {wrap}] no gate pair / unsupported")
            except Exception as exc:
                print(f"  [{bd_name} / {wrap}] ERROR: {exc}")

    json.dump(index, open(INDEX_JSON, "w"), indent=2)
    print(f"\nWrote index: {INDEX_JSON}")
    print(f"Total obfuscated graphs: {len(index)}")


if __name__ == "__main__":
    main()
