"""E1 Final: Comprehensive benchmark — all models, all metrics, paper Table 1.

Combines sound interval propagation + PGD into one result table.
"""

import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import sys
sys.path.insert(0, _AR)

import torch
import onnx
import os
import time
import json
from archproof.escalate import pgd_trigger_search, compute_risk_score

CLEAN_DIR = _os_ar.path.join(_AR, "benchmark/clean")
BOBER_DIR = "/tmp/bober_onnx"
HANDCRAFTED_DIR = "/tmp/handcrafted_onnx"
RESULTS_FILE = _os_ar.path.join(_AR, "benchmark/e1_final_results.json")

from backdoored_models import (
    op_sep_tar_backdoor, op_sep_un_backdoor,
    op_sha_tar_backdoor, op_sha_un_backdoor,
    op_int_tar_backdoor, op_int_un_backdoor,
    con_sep_tar_backdoor, con_sep_un_backdoor,
    con_sha_tar_backdoor, con_sha_un_backdoor,
    op_sep_un_backdoor_01, op_sep_un_backdoor_001, op_sep_un_backdoor_0001,
    op_sha_tar_backdoor_01, op_sha_tar_backdoor_001, op_sha_tar_backdoor_0001,
    op_int_tar_backdoor_01, op_int_tar_backdoor_001, op_int_tar_backdoor_0001,
)
from archproof.handcrafted_gdp import H1_SignGated, H2_AvgPoolGated, H3_MulIndicatorGated


# NOTE: The GDP classifier (L0-L4 pipeline, G1-G4 conditions, OPR, quadratic
# refinement) is defined in archproof/verify.py — the single source of truth.
# This script uses verify_model() below; do NOT reintroduce local sound_check
# or split_refined_tau variants.


def pgd_check(pytorch_model, x):
    """PGD trigger search: can we flip model output?"""
    try:
        found, trigger = pgd_trigger_search(pytorch_model, x)
        if found:
            eps, delta = compute_risk_score(pytorch_model, x, trigger, 0.0)
            return True, eps, delta
        return False, None, None
    except:
        return None, None, None


def export_if_needed(model, name, onnx_dir, x):
    path = os.path.join(onnx_dir, f"{name}.onnx")
    if not os.path.exists(path):
        os.makedirs(onnx_dir, exist_ok=True)
        try:
            torch.onnx.export(model.cpu().eval(), x, path, opset_version=17,
                              do_constant_folding=False,
                              input_names=["input"], output_names=["output"],
                      keep_initializers_as_inputs=True)
        except:
            return None
    return path


print("=" * 70)
print("E1 FINAL: COMPREHENSIVE BENCHMARK")
print("=" * 70)

all_results = []
x_cifar = torch.randn(1, 3, 32, 32)

# ============================================================
# BACKDOOR MODELS
# ============================================================
print("\n--- BACKDOOR MODELS ---")
print(f"{'Name':25s} {'Sound':>8s} {'Split50':>8s} {'PGD':>5s} {'eps*':>8s} {'delta':>8s}")
print("-" * 70)

backdoor_models = [
    # 10 base Bober types
    ("op_sep_tar", op_sep_tar_backdoor),
    ("op_sep_un", op_sep_un_backdoor),
    ("op_sha_tar", op_sha_tar_backdoor),
    ("op_sha_un", op_sha_un_backdoor),
    ("op_int_tar", op_int_tar_backdoor),
    ("op_int_un", op_int_un_backdoor),
    ("con_sep_tar", con_sep_tar_backdoor),
    ("con_sep_un", con_sep_un_backdoor),
    ("con_sha_tar", con_sha_tar_backdoor),
    ("con_sha_un", con_sha_un_backdoor),
    # 9 leaky variants
    ("op_sep_un_L01", op_sep_un_backdoor_01),
    ("op_sep_un_L001", op_sep_un_backdoor_001),
    ("op_sep_un_L0001", op_sep_un_backdoor_0001),
    ("op_sha_tar_L01", op_sha_tar_backdoor_01),
    ("op_sha_tar_L001", op_sha_tar_backdoor_001),
    ("op_sha_tar_L0001", op_sha_tar_backdoor_0001),
    ("op_int_tar_L01", op_int_tar_backdoor_01),
    ("op_int_tar_L001", op_int_tar_backdoor_001),
    ("op_int_tar_L0001", op_int_tar_backdoor_0001),
    # 3 handcrafted
    ("H1_SignGated", lambda: H1_SignGated()),
    ("H2_AvgPoolGated", lambda: H2_AvgPoolGated()),
    ("H3_MulIndicatorGated", lambda: H3_MulIndicatorGated()),
]

from archproof.verify import verify_model

for name, fn in backdoor_models:
    m = fn().eval()
    onnx_dir = HANDCRAFTED_DIR if name.startswith("H") else BOBER_DIR
    path = export_if_needed(m, name, onnx_dir, x_cifar)

    # Unified verification pipeline (no splitting — done separately in E1)
    if path:
        vr = verify_model(path, n_splits=1)
    else:
        vr = None

    pgd_det, eps, delta = pgd_check(m, x_cifar)

    if vr:
        verdict = vr.verdict
        tau = vr.tau_ibp
        tau_split = vr.tau_split if vr.tau_split < 1e10 else None
        opr = f"[{vr.output_preservation_lb:.3f},{vr.output_preservation_ub:.3f}]" if vr.has_output_preservation else "-"
    else:
        verdict, tau, tau_split, opr = "ERR", None, None, "-"

    pgd_str = "Y" if pgd_det else ("N" if pgd_det is False else "?")
    eps_str = f"{eps:.3f}" if eps else "-"
    delta_str = f"{delta:.3f}" if delta else "-"
    tau_str = f"{tau:.4f}" if tau is not None else "-"

    print(f"  {name:23s} {verdict:>16s} tau={tau_str:>7s} OPR={opr:>13s} {pgd_str:>5s} {eps_str:>8s} {delta_str:>8s}")

    all_results.append({
        "name": name, "type": "backdoor",
        "verdict": verdict,
        "sound_dormant": verdict == "DORMANT",
        "tau_bound": tau,
        "tau_split50": tau_split,
        "output_preservation": opr,
        "pgd_detected": pgd_det, "eps_star": eps, "delta": delta,
    })

# ============================================================
# CLEAN MODELS
# ============================================================
print(f"\n--- CLEAN MODELS ---")
print(f"{'Name':25s} {'Sound FP':>9s}")
print("-" * 36)

for fname in sorted(os.listdir(CLEAN_DIR)):
    if not fname.endswith(".onnx"):
        continue
    name = fname.replace(".onnx", "")
    path = os.path.join(CLEAN_DIR, fname)

    vr = verify_model(path, n_splits=1)  # no splitting for clean (fast)
    # FP = clean model wrongly labeled as DORMANT or OUTPUT-PRESERVED
    is_fp = vr.verdict in ("DORMANT", "OUTPUT-PRESERVED", "τ-BOUNDED")
    status = f"FP!({vr.verdict})" if is_fp else vr.verdict
    print(f"  {name:23s} {status:>16s}")

    all_results.append({
        "name": name, "type": "clean",
        "verdict": vr.verdict,
        "sound_fp": is_fp,
    })

# ============================================================
# SUMMARY
# ============================================================
backdoor_results = [r for r in all_results if r["type"] == "backdoor"]
clean_results = [r for r in all_results if r["type"] == "clean"]

n_bd = len(backdoor_results)
n_pgd_det = sum(1 for r in backdoor_results if r.get("pgd_detected") is True)
n_clean = len(clean_results)
n_fp = sum(1 for r in clean_results if r.get("sound_fp"))

# Count by verdict
from collections import Counter
bd_verdicts = Counter(r.get("verdict", "?") for r in backdoor_results)
cl_verdicts = Counter(r.get("verdict", "?") for r in clean_results)

print(f"\n{'='*70}")
print("E1 FINAL SUMMARY (via unified verify.py pipeline)")
print("=" * 70)
print(f"  Backdoor models:         {n_bd}")
for v in ["DORMANT", "OUTPUT-PRESERVED", "τ-BOUNDED", "BENIGN", "UNDECIDED", "GDP-FREE"]:
    if bd_verdicts.get(v, 0) > 0:
        print(f"    {v:23s} {bd_verdicts[v]}/{n_bd}")
print(f"    PGD detected:          {n_pgd_det}/{n_bd}")
print(f"  Clean models:            {n_clean}")
for v in ["GDP-FREE", "BENIGN", "UNDECIDED", "DORMANT", "OUTPUT-PRESERVED", "τ-BOUNDED"]:
    if cl_verdicts.get(v, 0) > 0:
        print(f"    {v:23s} {cl_verdicts[v]}/{n_clean}")
print(f"    Sound FP:              {n_fp}/{n_clean}")
print()
print(f"  Detail (non-DORMANT backdoor models):")
for r in backdoor_results:
    if r.get("verdict") != "DORMANT":
        print(f"    {r['name']:23s} {r['verdict']:>16s} tau={r.get('tau_bound','?')} OPR={r.get('output_preservation','-')}")

# Save
with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✅ Results saved to {RESULTS_FILE}")
