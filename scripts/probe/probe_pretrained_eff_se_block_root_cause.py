"""Root-cause probe: is the EfficientNet residual ε post-surgery a verifier
false-positive bug, an incomplete surgery script, or a sound-but-loose IBP
bound?

Test matrix:
  A. Clean pretrained EfficientNet (NO wrapper, no planted gate)
     → Expected if no bug: add-DGP-CLASS-NEGATIVE (SE blocks rejected or
       additive-certified)
     → If add-DGP-CERTIFIED-POSITIVE: verifier has a false-positive on
       legitimate SE-attention architectures (real bug).

  B. Clean wrapped pretrained EfficientNet (BackdoorWrapper but with the
     planted gate FRESHLY zeroed at construction — same as post-surgery).
     → Same expectation as A. Differs only by the trivial zeroed planted
       gate node.

  C. Backdoored wrapped pretrained EfficientNet (planted gate active).
     Pre-surgery → expect CERTIFIED-POSITIVE (correct detection).

  D. Backdoored wrapped pretrained EfficientNet, post `surgery_zero_gates`.
     → ε_after observed in step 4. Compare contribution decomposition vs A.

For each case we report:
  - n_syntactic, n_admitted
  - per-gate (g_lb, g_ub, eps_phi, payload, L_post, contribution)
  - verdict + ε_phaseC

If A == B == low-ε CLASS-NEGATIVE and only C is POSITIVE → no bug;
the EfficientNet 3/3 fail in step 4 is purely the incomplete surgery
script (it zeros only the planted gate; it never touched the backbone
SE blocks, which were never backdoors to begin with — but they ARE
admitted by Phase C's no-dormancy-filter policy and IBP-loose enough
that the residual ε swamps τ_sys).

If A == add-DGP-CERTIFIED-POSITIVE → the paper's clean EfficientNet
results are wrong and the verifier's Phase C admission policy yields
false positives on legitimate architectures.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
import json

warnings.filterwarnings("ignore")
os.environ.setdefault("ARCHPROOF_LARGE_MODEL_GB", "256")
os.environ.setdefault("ARCHPROOF_LLM_RESCUE", "1")

import torch

ROOT = Path(os.environ.get("ARCHPROOF_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT))

from archproof.verify_phaseC import verify_model_phaseC
from archproof.run_v3_production_scale import BackdoorWrapper
import torchvision.models as tvm

TMP = Path("/tmp/probe_eff_root_cause")
TMP.mkdir(parents=True, exist_ok=True)


def export(model, name, input_size=224):
    model.eval().cpu()
    p = TMP / f"{name}.onnx"
    torch.onnx.export(model, torch.randn(1, 3, input_size, input_size),
                      str(p), opset_version=17, do_constant_folding=False,
                      input_names=["input"], output_names=["output"])
    return p


def report(label, vr):
    print()
    print("=" * 72)
    print(f"[{label}]")
    print(f"  verdict_phaseC : {vr.verdict_phaseC}")
    print(f"  epsilon_phaseC : {vr.epsilon_phaseC}")
    print(f"  n_syntactic    : {vr.n_syntactic}")
    print(f"  n_admitted     : {vr.n_admitted_phaseC}")
    print(f"  ibp_blowup     : {getattr(vr, 'ibp_blowup', None)}")
    if vr.gate_epsilons:
        print(f"  gate_epsilons (first 6 of {len(vr.gate_epsilons)}):")
        for i, ge in enumerate(vr.gate_epsilons[:6]):
            tag = "se" if ("/Sigmoid" in ge["gate"] and "/block/" in ge["gate"]) else \
                  "planted" if ge["gate"] == "/Relu_output_0" else "?"
            print(f"    [{i:2d}] {tag:8s} act={ge['activation']:8s} "
                  f"g_lb={ge['g_lb']:.3e} g_ub={ge['g_ub']:.3e} "
                  f"eps_phi={ge['eps_phi_T']:.3e} "
                  f"payload={ge['payload_abs_max_T']!s:>12s} "
                  f"L_post={ge['L_post_T']!s:>10s} "
                  f"add_cert={ge['additive_certified']} "
                  f"contrib={ge['contribution_T']:.3e}")
            print(f"          gate={ge['gate']}")
    return {
        "label": label, "verdict": vr.verdict_phaseC,
        "epsilon": vr.epsilon_phaseC,
        "n_syntactic": vr.n_syntactic, "n_admitted": vr.n_admitted_phaseC,
        "n_gate_records": len(vr.gate_epsilons or []),
    }


def main():
    rows = []

    # ----- Case A: clean pretrained EfficientNet, NO wrapper -----
    print("\n>>> Case A: clean pretrained EfficientNet-B0 (no wrapper, no planted gate)")
    torch.manual_seed(0)
    bb_clean = tvm.efficientnet_b0(weights="DEFAULT").eval().cpu()
    p = export(bb_clean, "A_clean_eff")
    rows.append(report("A: clean eff (no wrapper)", verify_model_phaseC(str(p))))
    p.unlink(missing_ok=True)

    # ----- Case B: BackdoorWrapper, planted gate ZEROED at construction -----
    print("\n>>> Case B: pretrained Eff in BackdoorWrapper, planted gate1 zeroed")
    torch.manual_seed(0)
    bb = tvm.efficientnet_b0(weights="DEFAULT")
    w = BackdoorWrapper(bb, 1280, gate_type="sep_tar", n_classes=1000)
    with torch.no_grad():
        w.gate1.weight.zero_()
        w.gate1.bias.zero_()
    p = export(w, "B_wrap_eff_zeroed")
    rows.append(report("B: wrapped eff w/ gate1 zeroed", verify_model_phaseC(str(p))))
    p.unlink(missing_ok=True)

    # ----- Case C: BackdoorWrapper, planted gate ACTIVE -----
    print("\n>>> Case C: pretrained Eff in BackdoorWrapper, planted gate1 ACTIVE (pre-surgery)")
    torch.manual_seed(0)
    bb2 = tvm.efficientnet_b0(weights="DEFAULT")
    w2 = BackdoorWrapper(bb2, 1280, gate_type="sep_tar", n_classes=1000)
    p = export(w2, "C_wrap_eff_active")
    rows.append(report("C: wrapped eff active (pre-surgery)", verify_model_phaseC(str(p))))
    p.unlink(missing_ok=True)

    # ----- Case D: BackdoorWrapper, planted gate active then surgery (= step 4 protocol) -----
    print("\n>>> Case D: pretrained Eff in BackdoorWrapper, planted gate1 ACTIVE → surgery_zero_gates → re-export")
    torch.manual_seed(0)
    bb3 = tvm.efficientnet_b0(weights="DEFAULT")
    w3 = BackdoorWrapper(bb3, 1280, gate_type="sep_tar", n_classes=1000)
    # Run surgery (matches step 4 script logic)
    with torch.no_grad():
        w3.gate1.weight.zero_()
        w3.gate1.bias.zero_()
    p = export(w3, "D_wrap_eff_post_surgery")
    rows.append(report("D: wrapped eff post-surgery (step-4 protocol)",
                       verify_model_phaseC(str(p))))
    p.unlink(missing_ok=True)

    # ----- Comparison table -----
    print()
    print("=" * 72)
    print("ROOT-CAUSE COMPARISON")
    print("=" * 72)
    print(f"{'case':40s} {'verdict':30s} {'eps':>14s} {'n_syn':>5s} {'n_adm':>5s}")
    for r in rows:
        eps_s = "None" if r["epsilon"] is None else f"{r['epsilon']:.3e}"
        print(f"{r['label']:40s} {r['verdict']:30s} {eps_s:>14s} "
              f"{r['n_syntactic']:>5d} {r['n_admitted']:>5d}")

    # ----- Diagnosis -----
    A, B, C, D = rows
    print()
    print("=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)

    # Compare verdict on clean (A) vs wrapped-zeroed (B)
    if A["verdict"] == "add-DGP-CLASS-NEGATIVE":
        print("  ✓ Clean EfficientNet (Case A) is CLASS-NEGATIVE — verifier")
        print("    correctly handles legitimate SE-attention architectures.")
        print("    No verifier false-positive bug on clean models.")
    else:
        print(f"  ⚠ Clean EfficientNet (Case A) verdict={A['verdict']}")
        print("    THIS SUGGESTS a verifier false-positive on clean SE-attention.")
        print("    If ε > 0 on a backbone that has zero planted gates, the")
        print("    verifier's Phase C admission + IBP combination is too loose")
        print("    for SE-attention architectures.")

    # Compare A vs B
    same_AB = (abs((A["epsilon"] or 0) - (B["epsilon"] or 0))
               <= max(abs(A["epsilon"] or 0), 1.0) * 1e-9)
    if A["verdict"] == B["verdict"] and same_AB:
        print("  ✓ Clean (A) and wrapped-zeroed (B) give SAME verdict and ε.")
        print("    Wrapping with a zeroed planted gate adds nothing — the")
        print("    residual ε in Case D is exactly the SE-block contribution.")
    elif A["verdict"] != B["verdict"]:
        print("  ⚠ Verdicts differ between A and B!")
        print("    A: %s   B: %s" % (A["verdict"], B["verdict"]))
        print("    The wrapping itself (or zeroed gate ONNX nodes) changes the")
        print("    verdict — not pure backbone behaviour.")

    # B vs D (should be identical: same construction)
    if B["verdict"] == D["verdict"] and abs((B["epsilon"] or 0) - (D["epsilon"] or 0)) < 1e-6:
        print("  ✓ B and D agree (construction protocols are equivalent).")
    else:
        print(f"  ⚠ B and D disagree: B={B['verdict']}/{B['epsilon']} "
              f"vs D={D['verdict']}/{D['epsilon']}")
        print("    Implies a non-determinism or path difference between")
        print("    'construct-with-zeroed-gate' and 'construct-active-then-zero'.")

    # Save JSON
    out = ROOT / "results" / "probe_pretrained_eff_root_cause.json"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n  log: {out}")


if __name__ == "__main__":
    main()
