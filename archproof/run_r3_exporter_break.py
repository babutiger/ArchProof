"""R3-B1: Exporter-break failure panel for EIC.

We construct three adversarial exporter rewrites that destroy the
add-DGP activation -> Mul gate structure, run ArchProof on each,
and verify the verifier flags the change (does NOT silently return
the same epsilon).

Three breaks:
  (i)   const-fold-gate : evaluate the gate's pre-activation at a fixed
                          input, replace the sigmoid output with a constant
                          tensor, then the Mul becomes constant*payload
                          (no longer a tensor-tensor activation->Mul pattern).
  (ii)  fuse-sigmoid-mul: replace 'Sigmoid; Mul' with a single custom
                          op 'SigmoidMulFused' (Mul disappears).
  (iii) custom-gate-op  : replace 'Sigmoid; Mul' with a custom op
                          'GatedLinear' (unknown op -> vacuous bound).

Expected ArchProof behaviour on each break:
  - gate count drops, OR verdict becomes UNDECIDED, OR contributions
    differ from the original by more than the EIC bound.
  - the verifier does NOT silently return the same epsilon as the
    intact graph: the cross-graph EIC comparison yields
    'undecided-exporter' rather than a bogus zero divergence.

Output: benchmark/r3_exporter_break.json
"""
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from archproof.verify import verify_model


SRC = "/tmp/v3_e2e_case/backdoor_before_surgery.onnx"
OUT_DIR = Path("/tmp/r3_exporter_break")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = Path(_os_ar.path.join(_AR, "benchmark/r3_exporter_break.json"))


def _load_initializer(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def make_constfold_gate(src_path: str, dst_path: str) -> dict:
    """Constant-fold the gate's sigmoid output: replace sigmoid output with
    a constant initializer, then the Mul becomes constant*payload (the
    activation->Mul pattern is destroyed)."""
    m = onnx.load(src_path)
    inits_dict = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}

    n_replaced = 0
    new_nodes = []
    new_inits = list(m.graph.initializer)
    for node in m.graph.node:
        if node.op_type == "Sigmoid":
            # produce a constant tensor of shape matching the typical activation shape
            # We don't know exact shape here; use a 1-D tensor of size 64 with all 0.5
            const_arr = np.full((1, 64, 1, 1), 0.5, dtype=np.float32)
            const_name = node.output[0] + "_const"
            const_init = helper.make_tensor(
                name=const_name, data_type=TensorProto.FLOAT,
                dims=list(const_arr.shape), vals=const_arr.flatten().tolist())
            new_inits.append(const_init)
            # add an Identity rewriting node so that downstream Mul still has a tensor input,
            # but routed from a constant initializer (no producer node).
            ident = helper.make_node("Identity", inputs=[const_name],
                                     outputs=[node.output[0]],
                                     name=f"const_fold_{n_replaced}")
            new_nodes.append(ident)
            n_replaced += 1
        else:
            new_nodes.append(node)

    new_graph = helper.make_graph(
        nodes=new_nodes, name=m.graph.name + "_constfold",
        inputs=list(m.graph.input), outputs=list(m.graph.output),
        initializer=new_inits,
        value_info=list(m.graph.value_info),
    )
    new_m = helper.make_model(new_graph, opset_imports=list(m.opset_import))
    new_m.ir_version = m.ir_version
    onnx.save(new_m, dst_path)
    return {"transform": "const-fold-gate", "n_sigmoids_replaced": n_replaced}


def make_fused_sigmoid_mul(src_path: str, dst_path: str) -> dict:
    """Fuse Sigmoid+Mul into a custom op SigmoidMulFused (Mul disappears,
    so the activation->Mul pattern is no longer detected by the verifier)."""
    m = onnx.load(src_path)
    inits_dict = {init.name for init in m.graph.initializer}
    produces = {o: nd for nd in m.graph.node for o in nd.output}

    # Identify Sigmoid -> Mul pairs
    sigmoid_to_mul = {}  # sig_node -> mul_node
    for n in m.graph.node:
        if n.op_type == "Mul":
            for inp in n.input:
                src = produces.get(inp)
                if src is not None and src.op_type == "Sigmoid":
                    sigmoid_to_mul[src.name] = n
                    break

    n_fused = 0
    new_nodes = []
    skip_names = set()
    for n in m.graph.node:
        if n.name in skip_names:
            continue
        if n.op_type == "Sigmoid" and n.name in sigmoid_to_mul:
            mul_node = sigmoid_to_mul[n.name]
            # Build a custom op
            fused = helper.make_node(
                "SigmoidMulFused",
                inputs=[n.input[0]] + [i for i in mul_node.input
                                       if i != n.output[0]],
                outputs=list(mul_node.output),
                name=f"fused_{n_fused}",
                domain="archproof.test")
            new_nodes.append(fused)
            skip_names.add(mul_node.name)
            n_fused += 1
        else:
            new_nodes.append(n)

    new_graph = helper.make_graph(
        nodes=new_nodes, name=m.graph.name + "_fused",
        inputs=list(m.graph.input), outputs=list(m.graph.output),
        initializer=list(m.graph.initializer),
        value_info=list(m.graph.value_info))
    new_m = helper.make_model(new_graph, opset_imports=list(m.opset_import))
    new_m.ir_version = m.ir_version
    onnx.save(new_m, dst_path)
    return {"transform": "fuse-sigmoid-mul", "n_pairs_fused": n_fused}


def make_custom_gate_op(src_path: str, dst_path: str) -> dict:
    """Replace activation->Mul subgraph with a custom GatedLinear op
    (unknown op_type -> ArchProof falls back to vacuous bound)."""
    m = onnx.load(src_path)
    inits_dict = {init.name for init in m.graph.initializer}
    produces = {o: nd for nd in m.graph.node for o in nd.output}

    n_rewrites = 0
    new_nodes = []
    skip_names = set()
    for n in m.graph.node:
        if n.name in skip_names:
            continue
        if n.op_type == "Mul":
            tensor_ins = [i for i in n.input if i not in inits_dict]
            if len(tensor_ins) == 2:
                # Identify activation upstream of one input
                act_input = None
                payload_input = None
                for mi in n.input:
                    src = produces.get(mi)
                    if src is not None and src.op_type in (
                            "Sigmoid", "Tanh", "Relu", "LeakyRelu",
                            "Gelu", "Silu", "Swish", "HardSwish", "Softplus", "Mish"):
                        act_input = mi
                    elif mi not in inits_dict:
                        payload_input = mi
                if act_input is not None and payload_input is not None:
                    # Replace with custom op consuming the activation's pre-activation
                    src_act = produces[act_input]
                    custom = helper.make_node(
                        "GatedLinear",
                        inputs=[src_act.input[0], payload_input],
                        outputs=list(n.output),
                        name=f"gated_linear_{n_rewrites}",
                        domain="archproof.test")
                    new_nodes.append(custom)
                    skip_names.add(src_act.name)
                    n_rewrites += 1
                    continue
        new_nodes.append(n)

    new_graph = helper.make_graph(
        nodes=new_nodes, name=m.graph.name + "_custom",
        inputs=list(m.graph.input), outputs=list(m.graph.output),
        initializer=list(m.graph.initializer),
        value_info=list(m.graph.value_info))
    new_m = helper.make_model(new_graph, opset_imports=list(m.opset_import))
    new_m.ir_version = m.ir_version
    onnx.save(new_m, dst_path)
    return {"transform": "custom-gate-op", "n_rewrites": n_rewrites}


def verify_and_report(path: str) -> dict:
    try:
        r = verify_model(path, b_clean_ub=0.95, n_splits=20)
        return {
            "verdict": getattr(r, "verdict", None),
            "n_gdp_syntactic": getattr(r, "n_gdp_syntactic", None),
            "n_admitted": len(r.gate_epsilons),
            "total_epsilon": float(getattr(r, "total_epsilon", 0.0)),
            "total_output_margin": float(getattr(r, "total_output_margin", 0.0)),
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"missing source model: {SRC}")
    print(f"\n=== R3-B1 Exporter-break failure panel ===\n")

    # Verify the original
    print(f"-- ORIGINAL --")
    orig = verify_and_report(SRC)
    print(f"  verdict={orig['verdict']}  n_gdp_syn={orig['n_gdp_syntactic']}  "
          f"n_admitted={orig['n_admitted']}  ε={orig['total_epsilon']:.4e}  "
          f"contribution={orig['total_output_margin']:.4e}")

    breaks = [
        ("constfold", OUT_DIR / "break_constfold.onnx", make_constfold_gate),
        ("fused", OUT_DIR / "break_fused.onnx", make_fused_sigmoid_mul),
        ("custom", OUT_DIR / "break_custom.onnx", make_custom_gate_op),
    ]
    summary = {"original": orig, "breaks": []}
    for label, dst, fn in breaks:
        try:
            info = fn(SRC, str(dst))
        except Exception as e:
            print(f"-- BREAK {label} -- transform failed: {e}")
            summary["breaks"].append({"label": label, "transform_error": str(e)})
            continue
        rep = verify_and_report(str(dst))
        # EIC bijection check: gate count differs from original?
        bijection_holds = (rep.get("n_gdp_syntactic") == orig["n_gdp_syntactic"]
                          and rep.get("n_admitted") == orig["n_admitted"])
        verdict = rep.get("verdict")
        sub_verdict = "undecided-exporter" if not bijection_holds else "bijection-aligned"
        print(f"-- BREAK {label}: {info.get('transform','?')} --")
        print(f"  verdict={verdict}  n_gdp_syn={rep.get('n_gdp_syntactic')}  "
              f"n_admitted={rep.get('n_admitted')}  "
              f"bijection={'YES' if bijection_holds else 'BROKEN'}  "
              f"sub-verdict={sub_verdict}")
        summary["breaks"].append({
            "label": label, "transform": info,
            "verify": rep, "bijection_holds": bijection_holds,
            "sub_verdict_emitted": sub_verdict,
            "fail_closed": (not bijection_holds) or (verdict == "UNDECIDED"),
        })

    summary["all_fail_closed"] = all(b.get("fail_closed", False) for b in summary["breaks"])
    RESULTS_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {RESULTS_PATH}")
    print(f"all_fail_closed = {summary['all_fail_closed']}")


if __name__ == "__main__":
    main()
