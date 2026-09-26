"""Re-run cross_machine_repro for the 18 cells that failed in the original
overnight pass:

  yi-6b      6/6 export_fail: ValueError shape mismatch (transformers 4.30 too old)
  mistral-7b 6/6 export_fail: KeyError 'mistral'   (transformers 4.30 missing)
  qwen2-7b   6/6 export_fail: KeyError 'qwen2'    (transformers 4.30 missing)

Fixed via `scripts/fix_transformers_for_mistral_qwen2_yi.sh` which upgraded
A's transformers from 4.30.0 → 4.40.2.

This script:
  1. Re-exports + verifies the 3 problem LLMs × 6 toolchain configs = 18 cells
  2. Compares each ε against B's per_cell_whole_llm_eic.csv
  3. APPENDS results to per_cell_cross_machine_repro.csv (does NOT clobber
     the 12 successful GPT-J + DeepSeek rows already there)

Output: truth_source/per_cell_cross_machine_repro.csv (now 30 rows total)
"""
from __future__ import annotations
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))

import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(_AR)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("ARCHPROOF_LARGE_MODEL_GB", "256")
os.environ.setdefault("ARCHPROOF_LLM_RESCUE", "1")
os.environ.setdefault("ARCHPROOF_PROBE_MAX_GB", "40")

import scripts.run_whole_llm_eic as eic

OUT_CSV = ROOT / "truth_source" / "per_cell_cross_machine_repro.csv"
B_CSV = ROOT / "truth_source" / "per_cell_whole_llm_eic.csv"

# Only the 3 LLMs that failed export on A under transformers 4.30.0
LLMS_TO_RERUN = ["yi-6b", "mistral-7b", "qwen2-7b"]
CONFIGS = eic.CONFIGS

FIELDS = ["llm", "config", "machine", "epsilon", "verdict",
          "n_admitted", "rescue_pre", "rescue_payload",
          "export_sec", "verify_sec", "status",
          "b_epsilon", "abs_diff", "rel_diff", "bit_exact"]


def b_eps_for(llm: str, cfg_name: str) -> str:
    if not B_CSV.exists():
        return ""
    with B_CSV.open() as f:
        for row in csv.DictReader(f):
            if row.get("llm") == llm and row.get("config") == cfg_name:
                return row.get("epsilon", "")
    return ""


def filter_existing_rows():
    """Drop the 18 failed-export rows from the CSV (yi/mistral/qwen2),
    preserving the 12 successful GPT-J + DeepSeek rows. Returns the number
    kept."""
    if not OUT_CSV.exists():
        with OUT_CSV.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        return 0
    keep = []
    with OUT_CSV.open() as f:
        for row in csv.DictReader(f):
            if row.get("llm") not in LLMS_TO_RERUN:
                keep.append(row)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in keep:
            w.writerow({k: row.get(k, "") for k in FIELDS})
    return len(keep)


def main():
    name_to_hf = dict(eic.LLM_NAMES)

    n_kept = filter_existing_rows()
    print("=" * 80)
    print(f"Cross-machine repro — re-run failed-export LLMs after transformers"
          f" upgrade")
    print(f"  preserved rows in CSV   : {n_kept}  (gpt-j-6b + deepseek-7b)")
    print(f"  re-running              : {len(LLMS_TO_RERUN)} LLM × "
          f"{len(CONFIGS)} configs = {len(LLMS_TO_RERUN)*len(CONFIGS)} cells")
    print(f"  out                     : {OUT_CSV}")
    print("=" * 80)

    n_total = 0
    n_ok = 0
    n_bit = 0
    for short in LLMS_TO_RERUN:
        hf_name = name_to_hf.get(short)
        if hf_name is None:
            print(f"[skip {short}] no HF mapping")
            continue
        for cfg in CONFIGS:
            cell_id = f"{short}/{cfg['name']}"
            print(f"\n[{cell_id}] starting on A ...")
            n_total += 1

            t0 = time.time()
            try:
                onnx_path, export_dt = eic.export_one(short, hf_name, cfg)
            except Exception as e:
                print(f"  export-fail: {type(e).__name__}: {str(e)[:200]}")
                with OUT_CSV.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow({
                        "llm": short, "config": cfg["name"], "machine": "A",
                        "status": f"export_fail:{type(e).__name__}",
                    })
                eic._cleanup(short, cfg["name"])
                continue

            try:
                vr = eic.verify_one(onnx_path)
                verify_dt = time.time() - t0 - export_dt
                a_eps = float(vr.epsilon_phaseC) if vr.epsilon_phaseC is not None else 0.0
                verdict = vr.verdict_phaseC
                ge = vr.gate_epsilons or [{}]
                g0 = ge[0] if ge else {}
                rescue_pre = bool(g0.get("rescue_pre", False))
                rescue_payload = bool(g0.get("rescue_payload", False))
                n_adm = vr.n_admitted_phaseC
                status = "ok"
                n_ok += 1
            except Exception as e:
                print(f"  verify-fail: {type(e).__name__}: {str(e)[:200]}")
                verify_dt = time.time() - t0 - export_dt
                a_eps = None
                verdict = ""
                rescue_pre = rescue_payload = False
                n_adm = 0
                status = f"verify_fail:{type(e).__name__}"
            finally:
                eic._cleanup(short, cfg["name"])

            b_eps_str = b_eps_for(short, cfg["name"])
            b_eps = float(b_eps_str) if b_eps_str else None
            if a_eps is not None and b_eps is not None:
                abs_diff = abs(a_eps - b_eps)
                rel_diff = abs_diff / max(b_eps, 1e-300)
                bit_exact = abs_diff == 0.0
                if bit_exact:
                    n_bit += 1
            else:
                abs_diff = rel_diff = None
                bit_exact = None

            print(f"  A: ε={a_eps}  verdict={verdict}  "
                  f"export={export_dt:.1f}s verify={verify_dt:.1f}s")
            print(f"  B: ε={b_eps}")
            print(f"  diff: abs={abs_diff}  rel={rel_diff}  "
                  f"bit_exact={'✓' if bit_exact else ('✗' if bit_exact is False else '?')}")

            with OUT_CSV.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow({
                    "llm": short, "config": cfg["name"], "machine": "A",
                    "epsilon": a_eps, "verdict": verdict,
                    "n_admitted": n_adm,
                    "rescue_pre": rescue_pre,
                    "rescue_payload": rescue_payload,
                    "export_sec": round(export_dt, 1),
                    "verify_sec": round(verify_dt, 1),
                    "status": status,
                    "b_epsilon": b_eps, "abs_diff": abs_diff,
                    "rel_diff": rel_diff, "bit_exact": bit_exact,
                })

    print()
    print("=" * 80)
    print(f"DONE: {n_ok}/{n_total} OK  |  {n_bit}/{n_ok} bit-exact A=B")
    print(f"Total CSV rows now: {n_kept + n_total}")
    print("=" * 80)


if __name__ == "__main__":
    main()
