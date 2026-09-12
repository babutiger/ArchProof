"""B-machine LLM regression test for the per-gate G1 dormancy fix in
verify_phaseC.py.

Critical question: does my fix (which sets contribution=0 for non-dormant
admitted gates) break the established 5/5 CERTIFIED-POSITIVE result on
6-7 B LLMs?

Hypothesis (from probe data): NO, because all 5 LLMs have n_admitted=1
(only the planted gate, which is dormant by design). T10 admission
already rejects all SwiGLU sigmoids (median >= TAU_ADM=0.3), so the only
admitted gate IS the dormant planted gate. Fix changes nothing.

This script empirically verifies the hypothesis by re-verifying each
LLM at T_default config (the canonical export) with the *current*
verify_phaseC (which has the fix), and comparing ε to B's pre-fix CSV.

For each of the 5 LLMs:
  - Re-export at T_default if ONNX missing (else reuse benchmark/7b_onnx/...)
  - Run verify_model_phaseC
  - Compare ε to per_cell_whole_llm_eic.csv (B's pre-fix data)
  - Report bit_exact / tight / wide

Output: truth_source/per_cell_b_dormancy_fix_llm_regression.csv

Expected outcome: 5/5 bit_exact, confirming fix is safe for LLM panel.
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
os.environ.setdefault("ARCHPROOF_PROBE_MAX_GB", "40")

ROOT = Path(_AR)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.run_whole_llm_eic as eic
from archproof.verify_phaseC import verify_model_phaseC

OUT_CSV = ROOT / "truth_source" / "per_cell_b_dormancy_fix_llm_regression.csv"
B_CSV = ROOT / "truth_source" / "per_cell_whole_llm_eic.csv"

LLMS = ["gpt-j-6b", "yi-6b", "deepseek-7b", "mistral-7b", "qwen2-7b"]
TARGET_CFG = next(c for c in eic.CONFIGS if c["name"] == "T_default")


def b_eps_for(llm: str) -> str:
    if not B_CSV.exists():
        return ""
    with B_CSV.open() as f:
        for row in csv.DictReader(f):
            if row.get("llm") == llm and row.get("config") == "T_default":
                return row.get("epsilon", "")
    return ""


def main():
    name_to_hf = dict(eic.LLM_NAMES)

    fields = [
        "llm", "config", "epsilon_postfix", "verdict_postfix",
        "n_syntactic", "n_admitted",
        "n_excluded_non_dormant",   # NEW: how many gates the fix excluded
        "epsilon_prefix_b",
        "abs_diff", "rel_diff", "bit_exact",
        "export_sec", "verify_sec", "status",
    ]
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    with OUT_CSV.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print("=" * 76)
    print(f"B-machine LLM regression test for dormancy fix")
    print(f"  5 LLM × T_default = 5 cells")
    print(f"  out: {OUT_CSV}")
    print("=" * 76)

    n_total = 0
    n_bit_exact = 0
    n_ok = 0

    for short in LLMS:
        hf_name = name_to_hf.get(short)
        if hf_name is None:
            print(f"[skip {short}] no HF mapping")
            continue
        n_total += 1
        cell = f"{short}/T_default"
        print(f"\n[{cell}] starting on B ...")

        t0 = time.time()
        try:
            onnx_path, export_dt = eic.export_one(short, hf_name, TARGET_CFG)
        except Exception as e:
            print(f"  export-fail: {type(e).__name__}: {str(e)[:200]}")
            with OUT_CSV.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow({
                    "llm": short, "config": "T_default",
                    "status": f"export_fail:{type(e).__name__}",
                })
            eic._cleanup(short, "T_default")
            continue

        try:
            vr = verify_model_phaseC(str(onnx_path))
            verify_dt = time.time() - t0 - export_dt
            eps = float(vr.epsilon_phaseC) if vr.epsilon_phaseC is not None else None
            verdict = vr.verdict_phaseC
            n_syn = vr.n_syntactic
            n_adm = vr.n_admitted_phaseC
            # Count non-dormant exclusions (only present after fix)
            n_excluded = sum(
                1 for ge in (vr.gate_epsilons or [])
                if ge.get("non_dormant_excluded", False))
            status = "ok"
            n_ok += 1
        except Exception as e:
            print(f"  verify-fail: {type(e).__name__}: {str(e)[:200]}")
            verify_dt = time.time() - t0 - export_dt
            eps = None
            verdict = ""
            n_syn = n_adm = n_excluded = 0
            status = f"verify_fail:{type(e).__name__}"
        finally:
            eic._cleanup(short, "T_default")

        b_eps_str = b_eps_for(short)
        b_eps = float(b_eps_str) if b_eps_str else None
        if eps is not None and b_eps is not None:
            abs_diff = abs(eps - b_eps)
            rel_diff = abs_diff / max(abs(b_eps), 1e-300)
            bit_exact = abs_diff == 0.0
            if bit_exact:
                n_bit_exact += 1
        else:
            abs_diff = rel_diff = None
            bit_exact = None

        print(f"  postfix : ε={eps}  verdict={verdict}  "
              f"n_adm={n_adm}  excluded={n_excluded}")
        print(f"  prefix-B: ε={b_eps}")
        marker = (
            "✓ bit-exact" if bit_exact else
            ("± diff" if (rel_diff or 0) < 1e-9 else "⚠ DIVERGE")
        )
        print(f"  diff    : abs={abs_diff}  rel={rel_diff}  {marker}")

        with OUT_CSV.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({
                "llm": short, "config": "T_default",
                "epsilon_postfix": eps, "verdict_postfix": verdict,
                "n_syntactic": n_syn, "n_admitted": n_adm,
                "n_excluded_non_dormant": n_excluded,
                "epsilon_prefix_b": b_eps,
                "abs_diff": abs_diff, "rel_diff": rel_diff,
                "bit_exact": bit_exact,
                "export_sec": round(export_dt, 1),
                "verify_sec": round(verify_dt, 1),
                "status": status,
            })

    print()
    print("=" * 76)
    print(f"DONE: {n_ok}/{n_total} OK  |  {n_bit_exact}/{n_ok} bit-exact "
          f"vs B's pre-fix CSV")
    if n_ok > 0 and n_bit_exact == n_ok:
        print("✓ Dormancy fix is LLM-safe — verdicts and ε values unchanged.")
    elif n_ok > 0 and n_bit_exact < n_ok:
        print("⚠ Some LLMs show ε divergence — review per-cell to decide if")
        print("  fix is sound or if it broke the LLM panel.")
    print("=" * 76)


if __name__ == "__main__":
    main()
