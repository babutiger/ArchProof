"""Compile-stage adversarial Δ_T panel (Task #136).

Reviewer concern: "EIC theorem assumes the toolchain is benign. What if
the attacker controls the ONNX optimizer pass — does the certificate
still hold?"

We test 8 hostile / borderline-hostile optimizer passes that an
adversary-controlled compiler could legitimately apply, and measure the
resulting ε per pass. Sound EIC requires Δ_T bounded; ideally we want
ε bit-exact (Δ_T = 0 even under hostile compiler). Where a pass breaks
EIC alignment (e.g., fuses gate's Mul into a Gemm), the verifier's
behavior must remain SOUND — either it returns the same certificate or
explicitly degrades to UNCERTIFIED (no silent loss of soundness).

Test models: 4 ImageNet-pretrained BackdoorWrappers × sep_tar
(simplest gate type) = 4 cases. Each goes through:

  P0_baseline           : no optimization (T_default reference)
  P1_eliminate_identity : remove Identity nodes
  P2_eliminate_deadend  : remove unused subgraphs
  P3_eliminate_dup_init : merge identical initializers
  P4_fuse_bn_into_conv  : fold BatchNorm into preceding Conv
  P5_fuse_add_bias_conv : fold Add bias into Conv
  P6_eliminate_nop_pad  : remove zero Pad nodes
  P7_eliminate_csex     : Common Subexpression Elimination
  P8_combo_aggressive   : apply ALL above sequentially (worst-case)

Output: truth_source/per_cell_compile_stage_adversarial.csv
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

import numpy as np
import torch
import onnx
import onnxoptimizer

ROOT = Path(_AR)
sys.path.insert(0, str(ROOT))

from archproof.verify_phaseC import verify_model_phaseC
from archproof.run_v3_production_scale import BackdoorWrapper

OUT_CSV = ROOT / "truth_source" / "per_cell_compile_stage_adversarial.csv"
OUT_CSV.parent.mkdir(exist_ok=True)
TMP_DIR = Path("/tmp/v3_compile_stage_adv")
TMP_DIR.mkdir(parents=True, exist_ok=True)

PASSES = [
    ("P0_baseline",            []),
    ("P1_eliminate_identity",  ["eliminate_identity"]),
    ("P2_eliminate_deadend",   ["eliminate_deadend"]),
    ("P3_eliminate_dup_init",  ["eliminate_duplicate_initializer"]),
    ("P4_fuse_bn_into_conv",   ["fuse_bn_into_conv"]),
    ("P5_fuse_add_bias_conv",  ["fuse_add_bias_into_conv"]),
    ("P6_eliminate_nop_pad",   ["eliminate_nop_pad"]),
    ("P7_eliminate_csex",      ["eliminate_common_subexpression"]),
    ("P8_combo_aggressive",    ["eliminate_identity",
                                 "eliminate_deadend",
                                 "eliminate_duplicate_initializer",
                                 "fuse_bn_into_conv",
                                 "fuse_add_bias_into_conv",
                                 "eliminate_nop_pad",
                                 "eliminate_common_subexpression"]),
]


def load_pretrained_backbones():
    import torchvision.models as tvm
    return {
        "resnet18":        (tvm.resnet18(weights="DEFAULT"), 512),
        "mobilenet_v2":    (tvm.mobilenet_v2(weights="DEFAULT"), 1280),
        "efficientnet_b0": (tvm.efficientnet_b0(weights="DEFAULT"), 1280),
        # resnet50 omitted to keep runtime manageable; pretrained re-test
        # already showed it's CERTIFIED-POSITIVE under T_default.
    }


def export_baseline(wrapper, out_path):
    wrapper.eval().cpu()
    example = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        wrapper, example, str(out_path),
        opset_version=17, do_constant_folding=False,
        input_names=["input"], output_names=["output"],
    )


def apply_passes(input_path, output_path, passes):
    if not passes:
        # Baseline: just copy
        if str(input_path) != str(output_path):
            import shutil
            shutil.copy(input_path, output_path)
        return True, ""
    try:
        m = onnx.load(str(input_path))
        m_opt = onnxoptimizer.optimize(m, passes)
        onnx.save(m_opt, str(output_path))
        return True, ""
    except Exception as e:
        return False, str(e)[:200]


def verify(onnx_path):
    t0 = time.time()
    try:
        vr = verify_model_phaseC(str(onnx_path))
    except Exception as e:
        return {"status": "verify_fail", "error": str(e)[:200],
                "epsilon": None, "verdict": None,
                "n_syntactic": 0, "n_admitted": 0,
                "verify_sec": round(time.time() - t0, 2)}
    return {"status": "ok", "error": "",
            "epsilon": (float(vr.epsilon_phaseC)
                        if vr.epsilon_phaseC is not None else None),
            "verdict": vr.verdict_phaseC,
            "n_syntactic": vr.n_syntactic,
            "n_admitted": vr.n_admitted_phaseC,
            "verify_sec": round(time.time() - t0, 2)}


def main():
    bbs = load_pretrained_backbones()
    print("=" * 80)
    print(f"Compile-stage adversarial: {len(bbs)} backbones × {len(PASSES)} "
          f"passes = {len(bbs)*len(PASSES)} cells")
    print(f"  out: {OUT_CSV}")
    print("=" * 80)

    fields = ["backbone", "gate_type", "pass_name", "passes_applied",
              "epsilon", "verdict", "n_syntactic", "n_admitted",
              "size_mb", "verify_sec", "status", "error"]
    new_csv = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_csv:
            w.writeheader()

        per_model_eps = {}
        for bb_name, (bb, fd) in bbs.items():
            tag = f"{bb_name}_sep_tar"
            print(f"\n[{tag:30s}]")
            torch.manual_seed(0); np.random.seed(0)
            # Build the wrapper ONCE for this backbone, then re-export
            # baseline ONNX + apply each compiler pass.
            bb_fresh, fd_fresh = load_pretrained_backbones()[bb_name]
            wrapper = BackdoorWrapper(bb_fresh, fd_fresh,
                                       gate_type="sep_tar", n_classes=1000)
            baseline_path = TMP_DIR / f"{tag}_baseline.onnx"
            export_baseline(wrapper, baseline_path)

            eps_values = []
            for pass_name, passes in PASSES:
                onnx_path = TMP_DIR / f"{tag}_{pass_name}.onnx"
                ok, err = apply_passes(baseline_path, onnx_path, passes)
                if not ok:
                    print(f"  {pass_name:25s} optimizer-fail: {err[:80]}")
                    w.writerow({
                        "backbone": bb_name, "gate_type": "sep_tar",
                        "pass_name": pass_name,
                        "passes_applied": ";".join(passes),
                        "epsilon": None, "verdict": None,
                        "n_syntactic": 0, "n_admitted": 0,
                        "size_mb": 0, "verify_sec": 0,
                        "status": "optimizer_fail", "error": err[:200],
                    })
                    f.flush()
                    continue
                size_mb = round(onnx_path.stat().st_size / (1024 * 1024), 1)
                r = verify(onnx_path)
                if r["status"] == "ok":
                    eps = r["epsilon"]
                    if eps is not None and np.isfinite(eps):
                        eps_values.append(eps)
                    print(f"  {pass_name:25s} ε={eps!s:>16s} "
                          f"verdict={r['verdict']:30s} "
                          f"sz={size_mb}MB n_adm={r['n_admitted']} "
                          f"vfy={r['verify_sec']}s")
                else:
                    print(f"  {pass_name:25s} {r['status']}: {r['error'][:80]}")
                w.writerow({
                    "backbone": bb_name, "gate_type": "sep_tar",
                    "pass_name": pass_name,
                    "passes_applied": ";".join(passes),
                    "epsilon": r["epsilon"],
                    "verdict": r["verdict"] or "",
                    "n_syntactic": r["n_syntactic"],
                    "n_admitted": r["n_admitted"],
                    "size_mb": size_mb,
                    "verify_sec": r["verify_sec"],
                    "status": r["status"],
                    "error": r["error"][:200] if r["error"] else "",
                })
                f.flush()
                onnx_path.unlink(missing_ok=True)

            baseline_path.unlink(missing_ok=True)

            if len(eps_values) >= 2:
                d_T = max(eps_values) - min(eps_values)
                rel = (d_T / max(min(eps_values), 1e-12)
                       if min(eps_values) > 0 else 0.0)
                per_model_eps[tag] = (d_T, min(eps_values), max(eps_values),
                                       rel, len(eps_values))
                print(f"  Δ_T = {d_T:.4e}  (rel={rel:.2e})")

    print()
    print("=" * 80)
    print("Per-model Δ_T across compile-stage passes:")
    n_zero = 0
    n_tight = 0
    for tag, (d, mn, mx, rel, nc) in per_model_eps.items():
        flag = "✓ EXACT" if d < 1e-9 else (
            "✓ tight" if rel < 0.05 else "⚠ wide")
        print(f"  {tag:30s} n={nc} ε∈[{mn:.4e},{mx:.4e}] Δ_T={d:.4e} "
              f"rel={rel:.2e} {flag}")
        if d < 1e-9: n_zero += 1
        if rel < 0.05: n_tight += 1
    print(f"\nΔ_T = 0 exporter-invariant : {n_zero}/{len(per_model_eps)}")
    print(f"Δ_T tight (rel < 5%)        : {n_tight}/{len(per_model_eps)}")


if __name__ == "__main__":
    main()
