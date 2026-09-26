#!/usr/bin/env python3
"""RQ2 three-class re-run with the released phase-C verifier.

Runs ``archproof.verify_phaseC.verify_model_phaseC`` at its defaults
(tau_sys = 1e-3, T_box eta = 0.05, seeded probe) on the RQ2 slice the paper
reports in the verdict-distribution and detection tables:

  * the 22 seeded backdoor graphs (11 in-class, 11 out-of-class),
    ``models/backdoor_graphs/*.onnx``
  * the 49 natural clean models of the 11+49 detection slice,
    ``models/clean_panel/<name>.onnx`` (names from
    ``benchmark/baselines_natural49_complete.json``)

Writes ``truth_source/per_model_rq2_phaseC.csv`` and
``benchmark/rq2_phaseC_results.json`` and prints the three-class counts per
subset plus the ArchProof F1 on the 11+49 slice under the P-fail, P-reject and
P-hold protocols.  Read-only with respect to the models.
"""
from __future__ import annotations
import csv, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from archproof.verify_phaseC import verify_model_phaseC, CLASS_POSITIVE, CLASS_NEGATIVE, UNCERTIFIED

IN_CLASS = {"op_sep_tar", "op_sep_un", "op_sha_tar", "op_sha_un", "op_int_tar", "op_int_un",
            "op_int_tar_L01", "op_int_tar_L001", "op_int_tar_L0001",
            "H2_AvgPoolGated", "H3_MulIndicatorGated"}
OUT_OF_CLASS = {"con_sep_tar", "con_sep_un", "con_sha_tar", "con_sha_un",
                "op_sep_un_L01", "op_sep_un_L001", "op_sep_un_L0001",
                "op_sha_tar_L01", "op_sha_tar_L001", "op_sha_tar_L0001", "H1_SignGated"}

def main():
    nat = json.load(open(os.path.join(ROOT, "benchmark", "baselines_natural49_complete.json")))["natural49_models"]
    items = []
    for f in sorted(os.listdir(os.path.join(ROOT, "models", "backdoor_graphs"))):
        if f.endswith(".onnx"):
            n = f[:-5]
            sub = "backdoor-in-class" if n in IN_CLASS else ("backdoor-out-of-class" if n in OUT_OF_CLASS else "backdoor-unlabelled")
            items.append((n, sub, os.path.join(ROOT, "models", "backdoor_graphs", f)))
    for n in sorted(nat):
        items.append((n, "clean-natural49", os.path.join(ROOT, "models", "clean_panel", n + ".onnx")))
    assert sum(1 for i in items if i[1] == "backdoor-in-class") == 11, "expected 11 in-class backdoor graphs"
    assert sum(1 for i in items if i[1] == "backdoor-out-of-class") == 11, "expected 11 out-of-class backdoor graphs"
    assert sum(1 for i in items if i[1] == "clean-natural49") == 49
    rows = []
    for k, (name, sub, path) in enumerate(items, 1):
        t0 = time.time()
        try:
            r = verify_model_phaseC(path)
            row = dict(name=name, subset=sub, verdict=r.verdict_phaseC, epsilon=r.epsilon_phaseC,
                       n_syntactic=r.n_syntactic, n_admitted=r.n_admitted_phaseC,
                       additive_reject_count=r.additive_reject_count, epsilon_blowup=r.epsilon_blowup,
                       has_dormant_gate_on_b_clean=r.has_dormant_gate_on_b_clean,
                       tau_sys=r.tau_sys_used, trigger_eta=r.trigger_eta, error="")
        except Exception as e:  # keep going; an exception is reported as a fail row
            row = dict(name=name, subset=sub, verdict="fail", epsilon=None, n_syntactic=None, n_admitted=None,
                       additive_reject_count=None, epsilon_blowup=None, has_dormant_gate_on_b_clean=None,
                       tau_sys=None, trigger_eta=None, error=f"{type(e).__name__}: {e}"[:200])
        row["wall_s"] = round(time.time() - t0, 2)
        rows.append(row)
        print(f"[{k:3d}/{len(items)}] {sub:22s} {name:24s} -> {row['verdict']:28s} eps={row['epsilon']} n_adm={row['n_admitted']} ({row['wall_s']}s)", flush=True)
    os.makedirs(os.path.join(ROOT, "truth_source"), exist_ok=True)
    with open(os.path.join(ROOT, "truth_source", "per_model_rq2_phaseC.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(os.path.join(ROOT, "benchmark", "rq2_phaseC_results.json"), "w"), indent=1)
    # summary
    def cls(v): return "pos" if v == CLASS_POSITIVE else ("neg" if v == CLASS_NEGATIVE else "unc")
    from collections import Counter, defaultdict
    per = defaultdict(Counter)
    for r in rows: per[r["subset"]][cls(r["verdict"])] += 1
    print("\n=== three-class counts (released phase-C verifier, defaults) ===")
    for s in sorted(per): print(f"  {s:24s} pos={per[s]['pos']:3d} neg={per[s]['neg']:3d} unc={per[s]['unc']:3d}")
    tp = per["backdoor-in-class"]["pos"]; fn = 11 - tp; unc_clean = per["clean-natural49"]["unc"]; fp_hard = per["clean-natural49"]["pos"]
    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * rc / (p + rc) if p + rc else 0.0
    print(f"  11+49 slice: P-fail F1={f1(tp, fp_hard + unc_clean, fn):.3f} | P-reject F1={f1(tp, fp_hard, fn):.3f} | P-hold F1={f1(tp, fp_hard, fn):.3f}"
          f"  (tp={tp}, fn={fn}, clean pos={fp_hard}, clean unc={unc_clean})")
    print("  uncertified / unexpected:", [(r['name'], r['verdict']) for r in rows if cls(r['verdict']) == 'unc' or (r['subset'] == 'backdoor-out-of-class' and cls(r['verdict']) == 'pos') or (r['subset'] == 'backdoor-in-class' and cls(r['verdict']) != 'pos')])

if __name__ == "__main__":
    main()
