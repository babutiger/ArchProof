"""Phase C6 milestone 1: try CROWN escalation on all Phase-C-v2
UNCERTIFIED cases + the 12 E14 production-scale cases.

For each model:
  1. Run `verify_model_phaseC` for baseline `epsilon_phaseC`.
  2. If verdict = UNCERTIFIED or IBP blow-up, call
     `crown_bound_output` on D(T) = [0, 1] and take the graph-level
     |output|_max as a *coarse* upper bound on the total gate
     contribution (conservative: assumes the entire output could be
     driven by admitted gates; sound but loose).
  3. Compare: IBP blow-up ε vs CROWN-IBP ε.
  4. Report per-model whether CROWN recovers a finite verdict
     (ε_CROWN > τ_sys ⇒ recoverable CERTIFIED-POSITIVE, etc.)

Output: C6 progress note + per-cell CSV.
"""
from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from archproof.verify_phaseC import verify_model_phaseC
from archproof.crown_escalate import crown_bound_output, output_abs_max

PHASE_C_UNCERT_NAMES = ["efficientnet_b0", "glu_net", "regnet_y_400mf"]
E14_NAMES = [
    "resnet18_sep_tar", "resnet18_sha_un", "resnet18_int_un",
    "resnet50_sep_tar", "resnet50_sha_un", "resnet50_int_un",
    "mobilenet_v2_sep_tar", "mobilenet_v2_sha_un", "mobilenet_v2_int_un",
    "efficientnet_b0_sep_tar", "efficientnet_b0_sha_un",
    "efficientnet_b0_int_un",
]


def resolve(name: str) -> str | None:
    cands = [
        ROOT / "benchmark" / "clean" / f"{name}.onnx",
        ROOT / "benchmark" / "production_scale_onnx" / f"{name}.onnx",
        ROOT / "benchmark" / "exporter_test" / f"{name}_default.onnx",
        Path(f"/tmp/bober_onnx/{name}.onnx"),
    ]
    for c in cands:
        if os.path.exists(str(c)):
            return str(c)
    return None


def main():
    rows = []
    for label, names in [("phase_c_uncert", PHASE_C_UNCERT_NAMES),
                          ("e14", E14_NAMES)]:
        print(f"=== {label} ({len(names)} models) ===")
        for name in names:
            p = resolve(name)
            if p is None:
                print(f"  [SKIP] {name}: file not found")
                continue
            t0 = time.time()
            try:
                r_ibp = verify_model_phaseC(p)
            except Exception as e:
                print(f"  [ERR-IBP] {name}: {e}")
                continue
            dt_ibp = time.time() - t0
            t0 = time.time()
            bounds = crown_bound_output(p, input_lb=0.0, input_ub=1.0,
                                         method="CROWN-IBP", device="cpu")
            dt_crown = time.time() - t0
            if bounds is None:
                crown_eps = None
                crown_finite = False
            else:
                crown_eps = output_abs_max(*bounds)
                crown_finite = (crown_eps < 1e6)

            ibp_eps = r_ibp.epsilon_phaseC
            ibp_finite = (ibp_eps < 1e6)

            recoverable = (crown_finite and not ibp_finite)
            tightens = (crown_finite and ibp_finite and crown_eps < ibp_eps)

            rows.append({
                "panel": label,
                "model": name,
                "verdict_phaseC_IBP": r_ibp.verdict_phaseC,
                "ibp_eps": ibp_eps,
                "ibp_finite": ibp_finite,
                "ibp_sec": dt_ibp,
                "crown_eps": crown_eps,
                "crown_finite": crown_finite,
                "crown_sec": dt_crown,
                "recoverable_by_CROWN": recoverable,
                "CROWN_tightens_finite_IBP": tightens,
            })
            status = ("RECOVERABLE" if recoverable
                      else "TIGHTER" if tightens
                      else "unchanged" if ibp_finite
                      else "CROWN-also-fail")
            crown_str = f"{crown_eps:.3e}" if crown_eps is not None else "None"
            print(f"  [{name:30s}] IBP={ibp_eps:.2e} ({dt_ibp:3.1f}s)  "
                  f"CROWN={crown_str} ({dt_crown:3.1f}s)  {status}")

    out_csv = ROOT / "results" / "per_cell_crown_escalate.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote: {out_csv}")

    n_recoverable = sum(1 for r in rows if r["recoverable_by_CROWN"])
    n_tightens = sum(1 for r in rows if r["CROWN_tightens_finite_IBP"])
    n_total = len(rows)
    print(f"\nCROWN recoverable: {n_recoverable}/{n_total}")
    print(f"CROWN tightens finite IBP: {n_tightens}/{n_total}")


if __name__ == "__main__":
    main()
