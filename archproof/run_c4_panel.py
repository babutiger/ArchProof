#!/usr/bin/env python3
"""C4 panel: run ArchProof's phaseC verifier on every obfuscated graph
produced by adaptive_graph_obfuscation.py and record the verdict.

For a sound output-contribution certifier, the only allowed outcomes on
a semantically-equivalent obfuscation are:
  (a) add-DGP-CERTIFIED-POSITIVE  (verifier sees through the wrapper)
  (b) UNCERTIFIED                 (fail-closed)
Returning add-DGP-CLASS-NEGATIVE on a graph that still routes a
backdoor gate to the model output is a soundness leak that this panel
exposes.

Usage:
    python \
        archproof/run_c4_panel.py --tag pre_patch
"""
from __future__ import annotations
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))

import argparse
import csv
import json
import os
import sys
import time

ROOT = _AR
sys.path.insert(0, ROOT)

from archproof.verify_phaseC import verify_model_phaseC

INDEX_JSON = os.path.join(ROOT, "benchmark", "c4_obfuscation_index.json")
OUT_CSV_FMT = os.path.join(ROOT, "truth_source", "per_cell_c4_adaptive_obfuscation_{tag}.csv")
OUT_JSON_FMT = os.path.join(ROOT, "benchmark", "c4_panel_results_{tag}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="pre_patch",
                        help="suffix for output files (e.g. pre_patch / post_patch)")
    parser.add_argument("--limit", type=int, default=0,
                        help="if >0, only run the first N obfuscated graphs")
    args = parser.parse_args()

    index = json.load(open(INDEX_JSON))
    if args.limit > 0:
        index = index[: args.limit]

    out_csv = OUT_CSV_FMT.format(tag=args.tag)
    out_json = OUT_JSON_FMT.format(tag=args.tag)

    rows = []
    cnt_pos = cnt_unc = cnt_neg = cnt_err = 0
    t_start = time.time()
    for i, item in enumerate(index, 1):
        path = item["out"]
        model_name = item["model"]
        wrap = item["wrap"]
        t0 = time.time()
        try:
            vr = verify_model_phaseC(path)
            verdict = (getattr(vr, "verdict_phaseC", "")
                       or getattr(vr, "verdict", "") or "UNKNOWN")
            n_syn = int(getattr(vr, "n_syntactic", 0))
            n_adm = int(getattr(vr, "n_admitted_phaseC", 0))
            eps = (getattr(vr, "epsilon", None)
                   or getattr(vr, "total_epsilon", None))
            err = ""
        except Exception as exc:
            verdict = "ERROR"
            n_syn = n_adm = -1
            eps = None
            err = str(exc)[:200]
        elapsed = time.time() - t0

        v_up = str(verdict).upper()
        if "CERTIFIED-POSITIVE" in v_up or "CERTIFIED_POSITIVE" in v_up or "CERT-POS" in v_up:
            cnt_pos += 1
            sentinel = "CERT-POS"
        elif "UNCERT" in v_up:
            cnt_unc += 1
            sentinel = "UNCERT (fail-closed)"
        elif "CLASS-NEG" in v_up or "CLASS_NEG" in v_up or v_up in (
                "DGP-FREE", "GDP-FREE", "DORMANT", "OUTPUT-PRESERVED"):
            cnt_neg += 1
            sentinel = "CLASS-NEG (LEAK)"
        elif verdict == "ERROR":
            cnt_err += 1
            sentinel = "ERROR"
        else:
            cnt_unc += 1
            sentinel = f"OTHER ({verdict})"

        print(f"[{i:3d}/{len(index)}] {model_name:25s} {wrap:22s} "
              f"verdict={verdict:30s} n_syn={n_syn:2d} n_adm={n_adm:2d} "
              f"eps={eps if eps is not None else '-':<10} "
              f"{elapsed:5.1f}s  {sentinel}")

        rows.append({
            "model": model_name, "wrap": wrap, "path": path,
            "verdict": verdict, "n_syn": n_syn, "n_adm": n_adm,
            "epsilon": eps, "elapsed_sec": elapsed, "error": err,
            "sentinel": sentinel,
        })

    total = len(rows)
    elapsed_total = time.time() - t_start
    summary = {
        "tag": args.tag,
        "n_total": total,
        "n_certified_positive": cnt_pos,
        "n_uncertified_failclosed": cnt_unc,
        "n_class_negative_LEAK": cnt_neg,
        "n_error": cnt_err,
        "elapsed_total_sec": elapsed_total,
    }
    print()
    print("=" * 60)
    print(f"C4 PANEL SUMMARY (tag={args.tag})")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k} = {v}")

    json.dump({"summary": summary, "rows": rows},
              open(out_json, "w"), indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "wrap", "verdict", "n_syn", "n_adm",
                    "epsilon", "elapsed_sec", "sentinel", "error"])
        for r in rows:
            w.writerow([r["model"], r["wrap"], r["verdict"],
                        r["n_syn"], r["n_adm"], r["epsilon"],
                        r["elapsed_sec"], r["sentinel"], r["error"]])
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
