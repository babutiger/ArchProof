#!/usr/bin/env python3
"""Artifact reproduction: READ-ONLY verify the ALREADY-EXPORTED LLM ONNX.

Reproduces the paper's LLM data (tab:llm-headline / tab:appx:llm-detail) by
running `verify_model_phaseC` on the existing exports under
`benchmark/7b_onnx/{clean_onnx,backdoored_onnx}/`. It NEVER re-exports, NEVER
writes into benchmark/, and holds one 7B graph at a time (peak RSS ~70-95 GB;
run --only one model per process).

Usage (one LLM per process, memory-safe):
  python scripts/repro_llm_verify_existing.py --only gpt-j-6b \
      --out /tmp/llm_verify_repro/gptj.csv

Compares each verdict+epsilon against the paper's frozen truth-source values
printed at the end.
"""
from __future__ import annotations
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import argparse, csv, os, sys, time
from pathlib import Path

# Match the paper's truth-source runs EXACTLY (run_phaseE_llm_clean.sh,
# run_test_rescue_gptj.sh, run_reverify_gptj_probe.sh):
#   ARCHPROOF_LARGE_MODEL_GB=256 -> load FULL external-data weights (24-29 GB)
#       and run real IBP, instead of the >22GB graph-only fail-closed (eps=0).
#   ARCHPROOF_LLM_RESCUE=1 -> apply the geometric LayerNorm gate-bound rescue;
#       WITHOUT it the LN-gated bound stays vacuous -> eps=0 / UNCERTIFIED
#       (verify_phaseC.py:627,637). The paper rows carry rescue_pre=1/payload=1.
#   ARCHPROOF_PROBE_MAX_GB=40 -> dormancy-probe memory cap used in the paper run.
os.environ.setdefault("ARCHPROOF_LARGE_MODEL_GB", "256")
os.environ.setdefault("ARCHPROOF_LLM_RESCUE", "1")
os.environ.setdefault("ARCHPROOF_PROBE_MAX_GB", "40")

ROOT = Path(_AR)
sys.path.insert(0, str(ROOT))

BD_DIR = ROOT / "benchmark" / "7b_onnx" / "backdoored_onnx"
CLEAN_DIR = ROOT / "benchmark" / "7b_onnx" / "clean_onnx"

LLMS = ["gpt-j-6b", "yi-6b", "deepseek-7b", "mistral-7b", "qwen2-7b"]

# Correct on-disk layout (verified 2026-08-16):
#   clean:      benchmark/7b_onnx/clean_onnx/<name>/<name>.onnx
#   backdoored: benchmark/7b_onnx/backdoored_onnx/<name>-backdoored/<name>-backdoored.onnx
def paths_for(name):
    out = []
    c = CLEAN_DIR / name / f"{name}.onnx"
    if c.exists():
        out.append(("clean", str(c)))
    b = BD_DIR / f"{name}-backdoored" / f"{name}-backdoored.onnx"
    if b.exists():
        out.append(("backdoored", str(b)))
    return out


def cleaned_path_for(name):
    """The MGRS-cleaned (post-surgery) ONNX. After surgery the backdoor gate
    is removed, so verify should return CLASS-NEGATIVE with epsilon 0."""
    p = BD_DIR / f"{name}-backdoored" / f"{name}-mgrs-cleaned.onnx"
    return [("mgrs-cleaned", str(p))] if p.exists() else []


# Paper frozen truth-source (per_model_phaseE_llm_full_ibp.csv, backdoored rows).
PAPER_BACKDOORED = {
    "gpt-j-6b":    (16312112077.265913,  "add-DGP-CERTIFIED-POSITIVE"),
    "yi-6b":       (574716132775.4609,   "add-DGP-CERTIFIED-POSITIVE"),
    "deepseek-7b": (47714834881.10328,   "add-DGP-CERTIFIED-POSITIVE"),
    "mistral-7b":  (343927384931.8517,   "add-DGP-CERTIFIED-POSITIVE"),
    "qwen2-7b":    (3786781849947.4165,  "add-DGP-CERTIFIED-POSITIVE"),
}

FIELDS = ["llm", "gate_type", "onnx_path", "n_syntactic", "n_admitted",
          "epsilon", "verdict", "has_dormant", "n_rescue_pre",
          "n_rescue_payload", "max_payload_abs", "verify_sec", "status",
          "paper_epsilon", "paper_verdict", "rel_err", "match"]


def verify_one(name, gate_type, path):
    from archproof.verify_phaseC import verify_model_phaseC
    row = dict(llm=name, gate_type=gate_type, onnx_path=path,
               n_syntactic="", n_admitted="", epsilon="", verdict="",
               has_dormant="", n_rescue_pre="", n_rescue_payload="",
               max_payload_abs="", verify_sec="", status="", paper_epsilon="",
               paper_verdict="", rel_err="", match="")
    t0 = time.time()
    try:
        r = verify_model_phaseC(path)
        row["n_syntactic"] = r.n_syntactic
        row["n_admitted"] = r.n_admitted_phaseC
        row["epsilon"] = r.epsilon_phaseC
        row["verdict"] = r.verdict_phaseC
        row["has_dormant"] = getattr(r, "has_dormant_gate_on_b_clean", "")
        ges = getattr(r, "gate_epsilons", []) or []
        row["n_rescue_pre"] = sum(1 for ge in ges if ge.get("rescue_pre"))
        row["n_rescue_payload"] = sum(1 for ge in ges if ge.get("rescue_payload"))
        pays = [ge.get("payload_abs_max_T") for ge in ges
                if ge.get("payload_abs_max_T") is not None]
        row["max_payload_abs"] = max(pays) if pays else ""
        row["verify_sec"] = round(time.time() - t0, 1)
        row["status"] = "ok"
    except Exception as e:
        row["verify_sec"] = round(time.time() - t0, 1)
        row["status"] = f"fail: {type(e).__name__}: {str(e)[:160]}"
        return row
    if gate_type == "backdoored" and name in PAPER_BACKDOORED:
        pe, pv = PAPER_BACKDOORED[name]
        row["paper_epsilon"] = pe
        row["paper_verdict"] = pv
        try:
            eps = float(r.epsilon_phaseC)
            row["rel_err"] = abs(eps - pe) / abs(pe) if pe else ""
            vmatch = (r.verdict_phaseC == pv)
            ematch = (row["rel_err"] != "" and row["rel_err"] < 0.01)
            row["match"] = "MATCH" if (vmatch and ematch) else (
                "verdict-only" if vmatch else "MISMATCH")
        except Exception:
            row["match"] = "?"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-sep LLM names; empty=all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cleaned", action="store_true",
                    help="verify the MGRS-cleaned (post-surgery) ONNX; expect CLASS-NEGATIVE eps=0")
    a = ap.parse_args()
    only = {s.strip() for s in a.only.split(",") if s.strip()} or set(LLMS)
    pick = cleaned_path_for if a.cleaned else paths_for

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in LLMS:
        if name not in only:
            continue
        pairs = pick(name)
        if not pairs:
            print(f"[{name}] NO exports found ({'mgrs-cleaned' if a.cleaned else 'clean/backdoored'})",
                  flush=True)
            continue
        for gate_type, path in pairs:
            mb = round(sum(f.stat().st_size for f in Path(path).parent.iterdir()
                           if f.is_file()) / 1e9, 1)
            print(f"[{name}/{gate_type}] verifying ({mb} GB dir) READ-ONLY: {path}",
                  flush=True)
            row = verify_one(name, gate_type, path)
            rows.append(row)
            # incremental write
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader(); w.writerows(rows)
            tag = row.get("match", "")
            print(f"[{name}/{gate_type}] -> verdict={row['verdict']} "
                  f"eps={row['epsilon']} paper={row.get('paper_epsilon','')} "
                  f"rel_err={row.get('rel_err','')} {tag} "
                  f"({row['verify_sec']}s)", flush=True)
    print(f"\nwrote {out}")
    # summary
    for r in rows:
        if r["gate_type"] == "backdoored":
            print(f"  {r['llm']:12s} repro_eps={r['epsilon']} "
                  f"paper_eps={r['paper_epsilon']} -> {r.get('match','')}")


if __name__ == "__main__":
    main()
