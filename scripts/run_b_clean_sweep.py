"""B_clean input-box sensitivity sweep — verify ε is robust to input box
choice. Reviewer concern: paper uses B_clean=[0, 0.95]. Why not [0, 1]?
What about [-1, 1]?

For each (model, B_clean_box) on a representative panel of 6 CIFAR
backdoor models × 3 box variants = 18 cells:
  - Run verify_phaseC with the given b_clean_ub
  - Record ε_phaseC + verdict

Boxes:
  - [0, 0.95]: paper's default (avoids integer saturation in trigger
    detector arithmetic)
  - [0, 1]: full unit box
  - [-1, 1]: tanh-style centered (some quantizer outputs)

For the [-1, 1] case, the verifier passes b_clean_ub=2.0 with input_lb=-1
shift. We run only [0, 0.95] and [0, 1] in this script for cleanliness;
[-1, 1] requires different IBP entry-point and is left for future work.

6 model × 2 box = 12 cells.
Output: truth_source/per_cell_b_clean_sweep.csv
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

ROOT = Path(_AR)
BACKDOOR_PKG = ROOT / "backdoor-taxonomy"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKDOOR_PKG))

from archproof.verify_phaseC import verify_model_phaseC
from archproof.handcrafted_gdp import (
    H1_SignGated, H2_AvgPoolGated, H3_MulIndicatorGated,
)
import backdoored_models as bm

OUT_CSV = ROOT / "truth_source" / "per_cell_b_clean_sweep.csv"
TMP_DIR = Path("/tmp/v3_b_clean_sweep")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Representative 6 models covering different gate types
MODELS = [
    ("op_sep_tar",          getattr(bm, "op_sep_tar_backdoor")),
    ("op_sha_tar",          getattr(bm, "op_sha_tar_backdoor")),
    ("op_int_un",           getattr(bm, "op_int_un_backdoor")),
    ("con_sep_tar",         getattr(bm, "con_sep_tar_backdoor")),
    ("H2_AvgPoolGated",     H2_AvgPoolGated),
    ("H3_MulIndicatorGated", H3_MulIndicatorGated),
]

B_CLEAN_BOXES = [
    ("box_default", 0.95),    # paper default
    ("box_full",    1.00),    # full unit box
    ("box_tight",   0.90),    # tighter box (clear margin from saturation)
]


def export(name, factory, out_path):
    torch.manual_seed(0)
    m = factory().eval()
    torch.onnx.export(m, torch.randn(1, 3, 32, 32), str(out_path),
                      opset_version=17, do_constant_folding=False,
                      input_names=["input"], output_names=["output"],
                      keep_initializers_as_inputs=True)


def main():
    print("=" * 80)
    print(f"B_clean sweep — {len(MODELS)} models × {len(B_CLEAN_BOXES)} "
          f"boxes = {len(MODELS)*len(B_CLEAN_BOXES)} cells")
    print(f"  out: {OUT_CSV}")
    print("=" * 80)

    fields = ["model", "box_name", "b_clean_ub", "epsilon", "verdict",
              "n_syntactic", "n_admitted", "verify_sec"]
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    with OUT_CSV.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    n_total = 0
    for name, factory in MODELS:
        onnx_path = TMP_DIR / f"{name}.onnx"
        try:
            export(name, factory, onnx_path)
        except Exception as e:
            print(f"[skip {name}] export-fail: {type(e).__name__}: {e}")
            continue

        per_model_eps = {}
        for box_name, ub in B_CLEAN_BOXES:
            t0 = time.time()
            try:
                vr = verify_model_phaseC(str(onnx_path), b_clean_ub=ub)
                eps = float(vr.epsilon_phaseC) if vr.epsilon_phaseC is not None else None
                verdict = vr.verdict_phaseC
                n_syn = vr.n_syntactic
                n_adm = vr.n_admitted_phaseC
            except Exception as e:
                eps, verdict = None, f"fail:{type(e).__name__}"
                n_syn = n_adm = 0
            dt = round(time.time() - t0, 2)
            n_total += 1
            per_model_eps[box_name] = eps
            print(f"  {name:22s} {box_name}  ub={ub}  ε={eps!s:>14s}  "
                  f"verdict={verdict:35s} t={dt}s")
            with OUT_CSV.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow({
                    "model": name, "box_name": box_name, "b_clean_ub": ub,
                    "epsilon": eps, "verdict": verdict,
                    "n_syntactic": n_syn, "n_admitted": n_adm,
                    "verify_sec": dt,
                })

        onnx_path.unlink(missing_ok=True)

        # Per-model invariance summary
        eps_vals = [v for v in per_model_eps.values()
                    if v is not None and np.isfinite(v)]
        if len(eps_vals) >= 2:
            d = max(eps_vals) - min(eps_vals)
            rel = d / max(min(eps_vals), 1e-12) if min(eps_vals) > 0 else 0.0
            flag = "✓ EXACT" if d < 1e-9 else (
                "✓ tight" if rel < 0.05 else "⚠ wide")
            print(f"  {name:22s} Δ_box ε∈[{min(eps_vals):.4e}, "
                  f"{max(eps_vals):.4e}]  rel={rel:.2e}  {flag}")

    print()
    print("=" * 80)
    print(f"DONE: {n_total} cells")
    print("=" * 80)


if __name__ == "__main__":
    main()
