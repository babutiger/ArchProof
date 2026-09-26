"""Pretrained CNN MGRS — sound version using verify_phaseC.

Runs on 12 BackdoorWrapper × ImageNet-pretrained backbones (closes random-init/
pretrained consistency for MGRS, complementing #137 EIC + pretrained ACPC).

For each (backbone, gate_type):
  1. Build wrapper, export ONNX, verify_phaseC → ε_before (sound IBP)
  2. Surgery: zero gate.weight + gate.bias for all gates
  3. Re-export, verify_phaseC → ε_after
  4. Confirm ε_after ≤ 0 + reduction_ratio = 100%

12 cells. Output: truth_source/per_cell_pretrained_cnn_mgrs.csv
"""
from __future__ import annotations
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))

import csv
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("ARCHPROOF_LARGE_MODEL_GB", "256")
os.environ.setdefault("ARCHPROOF_LLM_RESCUE", "1")
os.environ.setdefault("ARCHPROOF_PROBE_MAX_GB", "10")

import numpy as np
import torch

ROOT = Path(_AR)
sys.path.insert(0, str(ROOT))

from archproof.verify_phaseC import verify_model_phaseC
from archproof.run_v3_production_scale import BackdoorWrapper
from archproof.mgrs import (GateContribution, minimum_gate_removal_set,
                            iterative_mgrs_with_ibp_oracle)

OUT_CSV = ROOT / "truth_source" / "per_cell_pretrained_cnn_mgrs.csv"
TMP_DIR = Path("/tmp/v3_pretrained_cnn_mgrs")
TMP_DIR.mkdir(parents=True, exist_ok=True)

GATE_TYPES = ["sep_tar", "sha_un", "int_un"]


def load_pretrained_backbones():
    import torchvision.models as tvm
    return {
        "resnet18":        (tvm.resnet18(weights="DEFAULT"), 512),
        "resnet50":        (tvm.resnet50(weights="DEFAULT"), 2048),
        "mobilenet_v2":    (tvm.mobilenet_v2(weights="DEFAULT"), 1280),
        "efficientnet_b0": (tvm.efficientnet_b0(weights="DEFAULT"), 1280),
    }


def export_wrapper(wrapper, out_path):
    wrapper.eval().cpu()
    example = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        wrapper, example, str(out_path),
        opset_version=17, do_constant_folding=False,
        input_names=["input"], output_names=["output"],
    )
    return round(out_path.stat().st_size / (1024 * 1024), 1)


def verify_eps(onnx_path):
    t0 = time.time()
    try:
        vr = verify_model_phaseC(str(onnx_path))
    except Exception as e:
        return None, "fail", str(e)[:200], round(time.time() - t0, 2)
    eps = float(vr.epsilon_phaseC) if vr.epsilon_phaseC is not None else None
    return (eps, vr.verdict_phaseC, "", round(time.time() - t0, 2),
            list(vr.gate_epsilons or []))


def build_mgrs_inputs(wrapper, gate_epsilons):
    """From verify_phaseC's gate_epsilons (keys eps_phi_T / payload_abs_max_T /
    contribution_T -- NOT the legacy verify.py schema) build the greedy-MGRS
    inputs: (GateContribution list, {gate_node: module}, {gate_node: (w, b)}).
    Only strictly-positive finite contributions are kept; intrinsic non-dormant
    gates carry contribution_T = 0 and are dropped, so the list is exactly the
    planted gates, mapped in order to the wrapper's gate modules."""
    gates = [wrapper.gate1]
    if wrapper.gate_type == "sha_un":
        gates.append(wrapper.gate2)
    elif wrapper.gate_type == "int_un":
        gates.extend([wrapper.gate2, wrapper.gate3])
    contribs, node_to_mod, orig = [], {}, {}
    j = 0
    for g in gate_epsilons:
        try:
            c = float(g.get("contribution_T"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(c) or c <= 0.0 or j >= len(gates):
            continue
        node = str(g.get("gate", f"gate_{j}"))
        contribs.append(GateContribution(
            gate_idx=j, gate_node=node,
            activation=str(g.get("activation", "relu")),
            epsilon=float(g.get("eps_phi_T") or 0.0),
            payload_abs_max=float(g.get("payload_abs_max_T") or 0.0),
            contribution=c))
        mod = gates[j]
        node_to_mod[node] = mod
        orig[node] = (mod.weight.detach().clone(), mod.bias.detach().clone())
        j += 1
    return contribs, node_to_mod, orig


def surgery_zero_gates(wrapper):
    gates = [wrapper.gate1]
    if wrapper.gate_type == "sha_un":
        gates.append(wrapper.gate2)
    elif wrapper.gate_type == "int_un":
        gates.extend([wrapper.gate2, wrapper.gate3])
    n_removed = 0
    with torch.no_grad():
        for g in gates:
            g.weight.zero_()
            g.bias.zero_()
            n_removed += 1
    return n_removed, len(gates)


def main():
    bbs = load_pretrained_backbones()
    print("=" * 80)
    print(f"Pretrained CNN MGRS phaseC — {len(bbs)} backbones × "
          f"{len(GATE_TYPES)} gate type = "
          f"{len(bbs)*len(GATE_TYPES)} cells")
    print(f"  out: {OUT_CSV}")
    print("=" * 80)

    fields = [
        "backbone", "gate_type", "weight_init", "k_total", "k_removed",
        "eps_before", "verdict_before",
        "eps_after",  "verdict_after",
        "achieved",   "reduction_ratio",
        "verify_before_sec", "verify_after_sec",
    ]
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        n_total = 0
        n_sound = 0
        for bb_name, (bb, fd) in bbs.items():
            for gate_type in GATE_TYPES:
                tag = f"{bb_name}_{gate_type}"
                print(f"\n[{tag:30s}]")
                torch.manual_seed(0); np.random.seed(0)
                bbones = load_pretrained_backbones()
                bb_fresh, fd_fresh = bbones[bb_name]
                wrapper = BackdoorWrapper(bb_fresh, fd_fresh,
                                          gate_type=gate_type, n_classes=1000)
                wrapper.eval().cpu()

                onnx_before = TMP_DIR / f"{tag}_before.onnx"
                size_mb = export_wrapper(wrapper, onnx_before)
                (eps_before, verdict_before, err_b, t_before,
                 gate_eps_before) = verify_eps(onnx_before)
                onnx_before.unlink(missing_ok=True)
                print(f"  before: ε={eps_before}  verdict={verdict_before}  "
                      f"sz={size_mb}MB t={t_before}s")
                if eps_before is None:
                    print(f"  ⚠ verify_before fail: {err_b[:80]}")
                    continue

                # Verifier-driven MGRS via iterative_mgrs_with_ibp_oracle: it
                # re-verifies ε by actual re-verification after each removal, so
                # it is NOT fooled by the float64 cancellation that makes the
                # plain greedy under-count int_un gates (mgrs.py:142). The oracle
                # restores weights, zeroes the candidate gate modules, exports,
                # and re-verifies the resulting graph.
                contribs, node_to_mod, orig_w = build_mgrs_inputs(
                    wrapper, gate_eps_before)

                def _oracle(remove_nodes, _w=node_to_mod, _o=orig_w, _t=tag):
                    for nd, (w, b) in _o.items():
                        _w[nd].weight.data.copy_(w)
                        _w[nd].bias.data.copy_(b)
                    for nd in remove_nodes:
                        _w[nd].weight.data.zero_()
                        _w[nd].bias.data.zero_()
                    tmp = TMP_DIR / f"{_t}_oracle.onnx"
                    export_wrapper(wrapper, tmp)
                    e = verify_eps(tmp)[0]
                    tmp.unlink(missing_ok=True)
                    return float(e) if e is not None else float("inf")

                if contribs:
                    res = iterative_mgrs_with_ibp_oracle(
                        contribs, _oracle, target_epsilon=0.0)
                    # NB: use ww/bb here -- `w` is the csv.DictWriter in main()'s
                    # scope; reusing it as a loop var clobbers it into a Tensor.
                    for nd, (ww, bb) in orig_w.items():
                        node_to_mod[nd].weight.data.copy_(ww)
                        node_to_mod[nd].bias.data.copy_(bb)
                    for nd in res.removed_gate_nodes:
                        node_to_mod[nd].weight.data.zero_()
                        node_to_mod[nd].bias.data.zero_()
                    k_removed = len(res.removed_gate_nodes)
                    k_total = len(contribs)
                    print(f"  MGRS(iterative): removed {k_removed}/{k_total} "
                          f"gates in {res.n_oracle_calls} oracle calls")
                else:
                    k_removed, k_total = surgery_zero_gates(wrapper)
                onnx_after = TMP_DIR / f"{tag}_after.onnx"
                export_wrapper(wrapper, onnx_after)
                eps_after, verdict_after, err_a, t_after, _ = verify_eps(
                    onnx_after)
                onnx_after.unlink(missing_ok=True)

                achieved = (eps_after is not None and abs(eps_after) <= 1e-9)
                reduction = (1.0 - (eps_after or 0.0) / max(eps_before, 1e-12)
                             if eps_before > 0 else 0.0)
                n_total += 1
                n_sound += int(achieved)
                print(f"  after:  ε={eps_after}  verdict={verdict_after}  "
                      f"k_removed={k_removed}/{k_total}  "
                      f"sound={'✓' if achieved else '✗'}  "
                      f"reduction={reduction*100:.1f}%  t={t_after}s")

                w.writerow({
                    "backbone": bb_name, "gate_type": gate_type,
                    "weight_init": "pretrained",
                    "k_total": k_total, "k_removed": k_removed,
                    "eps_before": eps_before,
                    "verdict_before": verdict_before,
                    "eps_after": eps_after,
                    "verdict_after": verdict_after,
                    "achieved": achieved,
                    "reduction_ratio": reduction,
                    "verify_before_sec": t_before,
                    "verify_after_sec": t_after,
                })
                f.flush()

    print()
    print("=" * 80)
    print(f"DONE: {n_sound}/{n_total} surgery achieved (ε_after ≤ 0)")
    print("=" * 80)


if __name__ == "__main__":
    main()
