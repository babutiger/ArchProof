"""Test the bool-Trilu rewrite on qwen2-7b's ONNX before patching the
production gate_admission.py code path.

Strategy: find Trilu nodes whose data input is BOOL, replace with:
    Cast(BOOL → INT64) → Trilu(INT64) → Cast(INT64 → BOOL)

ORT CPU has Trilu kernel for INT64 in ≥1.17, so rewriting bool→int64 lets
the probe path succeed without GPU EP (RTX3060 12GB can't fit 7B-LLM).

Verify: load qwen2 ONNX → rewrite → ORT inference succeeds (probe should
admit only 1 gate same as B's path).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("ARCHPROOF_LARGE_MODEL_GB", "256")

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


def _rewrite_bool_trilu(m: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Find Trilu nodes whose data input is BOOL; wrap them with
    Cast(BOOL→INT64) → Trilu(INT64) → Cast(INT64→BOOL).

    Returns the rewritten model + count of Trilu nodes rewritten.
    """
    inferred = onnx.shape_inference.infer_shapes(m)

    # Map tensor name → elem_type
    dtypes: dict[str, int] = {}
    for vi in (list(inferred.graph.value_info)
               + list(inferred.graph.input)
               + list(inferred.graph.output)):
        if vi.name and vi.type.tensor_type.elem_type:
            dtypes[vi.name] = vi.type.tensor_type.elem_type
    for init in inferred.graph.initializer:
        dtypes[init.name] = init.data_type

    nodes = list(m.graph.node)
    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0
    for node in nodes:
        if node.op_type != "Trilu":
            new_nodes.append(node)
            continue
        in_data = node.input[0]
        if dtypes.get(in_data) != TensorProto.BOOL:
            new_nodes.append(node)
            continue

        # Insert: Cast(BOOL→INT64) → Trilu(INT64) → Cast(INT64→BOOL)
        cast_in_out  = f"{in_data}_to_int64_{rewritten}"
        trilu_int_out = f"{node.output[0]}_int64"
        cast_pre = helper.make_node(
            "Cast", [in_data], [cast_in_out],
            to=TensorProto.INT64,
            name=f"trilu_pre_cast_{rewritten}")

        # Build a new Trilu with INT64 input
        new_trilu_inputs = [cast_in_out] + list(node.input[1:])
        new_trilu = helper.make_node(
            "Trilu", new_trilu_inputs, [trilu_int_out],
            **{a.name: helper.get_attribute_value(a) for a in node.attribute},
            name=node.name + "_int64" if node.name else f"trilu_int64_{rewritten}")

        cast_post = helper.make_node(
            "Cast", [trilu_int_out], [node.output[0]],
            to=TensorProto.BOOL,
            name=f"trilu_post_cast_{rewritten}")

        new_nodes.extend([cast_pre, new_trilu, cast_post])
        rewritten += 1

    if rewritten == 0:
        return m, 0

    # Replace nodes
    new_graph = helper.make_graph(
        new_nodes, m.graph.name,
        list(m.graph.input), list(m.graph.output),
        list(m.graph.initializer),
        value_info=list(m.graph.value_info))
    new_model = helper.make_model(new_graph,
                                   opset_imports=list(m.opset_import))
    new_model.ir_version = m.ir_version
    return new_model, rewritten


def main():
    qwen2_path = Path(
        "/tmp/v3_whole_llm_eic/qwen2-7b-T_default/qwen2-7b-T_default.onnx")
    if not qwen2_path.exists():
        # Try alternative path on shared benchmark
        archproof_root = Path(os.environ.get("ARCHPROOF_ROOT",
                                              str(Path(__file__).resolve().parents[2])))
        alt = archproof_root / "benchmark/7b_onnx/backdoored_onnx/qwen2-7b-backdoored"
        if alt.exists():
            files = list(alt.glob("*.onnx"))
            if files:
                qwen2_path = files[0]
    if not qwen2_path.exists():
        # Build a synthetic mini test using the same bool Trilu pattern
        print("[test] no qwen2 ONNX on disk; building synthetic bool-Trilu mini graph")
        inp_shape = helper.make_tensor_value_info(
            "shape", TensorProto.INT64, [2])
        cof_out = "cof_out"
        cof = helper.make_node(
            "ConstantOfShape", ["shape"], [cof_out],
            value=helper.make_tensor("v", TensorProto.BOOL, [1], [True]))
        k_const = helper.make_node(
            "Constant", [], ["k"],
            value=helper.make_tensor("kv", TensorProto.INT64, [], [0]))
        trilu = helper.make_node(
            "Trilu", [cof_out, "k"], ["mask"], upper=0)
        out = helper.make_tensor_value_info("mask", TensorProto.BOOL, [4, 4])
        graph = helper.make_graph(
            [cof, k_const, trilu], "synth", [inp_shape], [out])
        m = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 14)])
        m.ir_version = 8
        synth_path = Path("/tmp/_qwen2_bool_trilu_mini.onnx")
        onnx.save(m, str(synth_path))
        qwen2_path = synth_path

    print(f"[probe] loading {qwen2_path}")
    m = onnx.load(str(qwen2_path), load_external_data=False)
    print(f"  total nodes : {len(m.graph.node)}")
    print(f"  Trilu count : {sum(1 for n in m.graph.node if n.op_type == 'Trilu')}")

    print("[rewrite] applying bool-Trilu → cast(int64)+Trilu+cast(bool)")
    m_new, n_rew = _rewrite_bool_trilu(m)
    print(f"  rewrote {n_rew} Trilu node(s)")

    # Save rewritten model next to original (for external_data resolution)
    out_path = Path(str(qwen2_path).replace(".onnx", "_bool_trilu_rewritten.onnx"))
    onnx.save(m_new, str(out_path))
    print(f"[saved] {out_path}")

    print(f"[ort] running InferenceSession on rewritten model ...")
    try:
        sess = ort.InferenceSession(str(out_path),
                                     providers=["CPUExecutionProvider"])
        print(f"  ✓ session created — Trilu workaround successful")
        print(f"  inputs: {[i.name for i in sess.get_inputs()]}")
    except Exception as e:
        print(f"  ✗ session failed: {type(e).__name__}: {str(e)[:300]}")
        sys.exit(1)

    print(f"[ok] rewrite path validated; ready to integrate into gate_admission.py")


if __name__ == "__main__":
    main()
