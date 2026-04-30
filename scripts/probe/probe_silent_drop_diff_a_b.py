"""Diagnose why case A's 2 admitted gates yield 0 gate_epsilons records
while case B's same SE blocks produce 3 records with huge ε.

Hypothesis: case A's IBP bounds for the SE pre-activations are not computed
(b is None on `bounds.get(pre_act)`) or are vacuous / un-rescuable, so the
loop hits one of the silent `continue` paths and drops the gate's contribution
without setting `contribution = inf`. Phase C's verdict then reports ε=0 →
CLASS-NEGATIVE despite 2 admitted gates.

We instrument verify_phaseC by re-implementing the relevant per-gate loop
with VERBOSE diagnostics, exposing for each admitted gate:
  - bounds.get(pre_act) is None?
  - lb/ub vacuous? (>1e30 or non-finite?)
  - rescue tried? rescue result?
  - eps_phi raised exception?
  - payload bound state?
  - L_post finite?

For both cases A and B and verify the difference.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("ARCHPROOF_LARGE_MODEL_GB", "256")
os.environ.setdefault("ARCHPROOF_LLM_RESCUE", "1")

import numpy as np
import torch
import onnx

ROOT = Path(os.environ.get("ARCHPROOF_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT))

from archproof.run_v3_production_scale import BackdoorWrapper
import torchvision.models as tvm

TMP = Path("/tmp/probe_silent_drop")
TMP.mkdir(parents=True, exist_ok=True)


def export(model, name, input_size=224):
    model.eval().cpu()
    p = TMP / f"{name}.onnx"
    torch.onnx.export(model, torch.randn(1, 3, input_size, input_size),
                      str(p), opset_version=17, do_constant_folding=False,
                      input_names=["input"], output_names=["output"])
    return p


def diagnose(label, onnx_path):
    """Replicate the start of verify_phaseC up to the per-gate loop and
    inspect the state of each admitted gate."""
    print()
    print("=" * 80)
    print(f"[{label}]  {onnx_path}")
    print("=" * 80)

    # Lazy import to avoid heavy module init at top
    from archproof.verify_phaseC import (
        verify_model_phaseC, _is_vacuous_bound,
    )
    # Run the verifier and grab its internal trace via a hook by
    # monkey-patching the inner function. Easier path: just verify and
    # then re-do the bound walk ourselves.
    vr = verify_model_phaseC(str(onnx_path))
    print(f"  verdict={vr.verdict_phaseC}  ε={vr.epsilon_phaseC}")
    print(f"  n_syntactic={vr.n_syntactic}  n_admitted={vr.n_admitted_phaseC}  "
          f"gate_records={len(vr.gate_epsilons or [])}")

    # Re-run the same admission/IBP machinery to get bounds + admission set.
    from archproof.gate_admission import probe_gate_medians
    from archproof.interval_propagation import propagate_intervals
    from archproof import verify_phaseC as vc

    m = onnx.load(str(onnx_path))
    # Build B_clean input box [0, 0.95] like the verifier does.
    inp_lb = {m.graph.input[0].name: np.zeros((1, 3, 224, 224), dtype=np.float32)}
    inp_ub = {m.graph.input[0].name: 0.95 * np.ones((1, 3, 224, 224), dtype=np.float32)}
    bounds = propagate_intervals(m, inp_lb, inp_ub)
    print(f"  bounds dict has {len(bounds)} tensor names")

    # Discover admitted gate tensors via the verifier's helper.
    syn = vc._collect_syntactic_dgp_gates(m)
    print(f"  syntactic gates discovered: {len(syn)}")
    if not syn:
        print("  (no syntactic gates — nothing to diagnose)")
        return

    # Probe T10 admission
    try:
        probe = probe_gate_medians(str(onnx_path), [g[0] for g in syn])
    except Exception as e:
        print(f"  probe_gate_medians failed: {type(e).__name__}: {e}")
        probe = {}

    # Walk admitted gates and replicate the per-gate loop's decision tree.
    activation_outputs = {g_name: (act_node, act_type) for (g_name, act_node, act_type, _) in syn}
    print()
    print(f"  per-admitted-gate diagnostic:")
    print(f"  {'gate':70s} {'bound?':6s} {'lb':>14s} {'ub':>14s} {'vacuous?':10s} {'admit?':6s}")
    n_admitted_count = 0
    n_silent_drop = 0
    for (g_name, act_node, act_type, mul_node) in syn:
        # Admission decision (replicate verifier's logic)
        median = probe.get(g_name, {}).get("median", float("nan"))
        admitted = (np.isfinite(median) and median < 0.3) or not np.isfinite(median)
        if not admitted:
            continue
        n_admitted_count += 1

        pre_act = act_node.input[0]
        b = bounds.get(pre_act)
        if b is None:
            print(f"  {g_name[:70]:70s} {'NO':6s} {'-':>14s} {'-':>14s} "
                  f"{'-':10s} {'YES':6s}  → SILENT DROP (b is None)")
            n_silent_drop += 1
            continue
        lb, ub = float(b.lb.min()), float(b.ub.max())
        vacuous_lb = (not np.isfinite(lb)) or abs(lb) > 1e30
        vacuous_ub = (not np.isfinite(ub)) or abs(ub) > 1e30
        vacuous = vacuous_lb or vacuous_ub
        marker = "VACUOUS" if vacuous else "ok"
        print(f"  {g_name[:70]:70s} {'YES':6s} {lb:14.3e} {ub:14.3e} "
              f"{marker:10s} {'YES':6s}")
        if vacuous:
            # The verifier tries _try_local_rescue here. We do not replicate
            # it precisely; just flag.
            print(f"     ↳ vacuous bound — verifier attempts local rescue.")

    print(f"\n  TOTAL admitted: {n_admitted_count}; silent-drop (b is None): {n_silent_drop}")
    return vr


def main():
    # Case A: clean EfficientNet
    torch.manual_seed(0)
    bb = tvm.efficientnet_b0(weights="DEFAULT").eval().cpu()
    pA = export(bb, "A_clean")

    # Case B: wrapped + zeroed
    torch.manual_seed(0)
    bb2 = tvm.efficientnet_b0(weights="DEFAULT")
    w = BackdoorWrapper(bb2, 1280, gate_type="sep_tar", n_classes=1000)
    with torch.no_grad():
        w.gate1.weight.zero_(); w.gate1.bias.zero_()
    pB = export(w, "B_wrapped_zeroed")

    diagnose("A: clean eff", pA)
    diagnose("B: wrapped eff zeroed", pB)


if __name__ == "__main__":
    main()
