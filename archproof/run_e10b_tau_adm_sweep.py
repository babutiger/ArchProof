#!/usr/bin/env python3
"""E10b: tau_adm sweep with the additive-branch certifier OFF.

Backs paper table tab:appx:tau-adm ("with the additive-branch certifier
disabled to isolate the admission-test's contribution"). This is the run
docs/REPRODUCE.md gotcha 7 describes: verify_model defaults the certifier on, so
this sweep passes gdp_flags with additive_branch=False explicitly.

Pool: 22 backdoor + 79 clean, the same slice as run_e10_tau_sweep.py.
Output: benchmark/e10b_tau_adm_sweep_certifier_off.json
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archproof.verify import verify_model
from archproof.run_b3_real_baselines import BOBER_DIR, HANDCRAFTED_DIR, CLEAN_DIR

TAU_SYS = 1e-3
TAUS = [0.1, 0.3, 0.5]
FLAGS = {"G1": True, "G2": True, "G3": True, "G4": True, "T10": True,
         "additive_branch": False}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "benchmark", "e10b_tau_adm_sweep_certifier_off.json")


def classify(verdict, eps):
    if verdict in ("DORMANT", "OUTPUT-PRESERVED"):
        return True
    if verdict in ("τ-BOUNDED", "ε-BOUNDED") and eps is not None and eps > TAU_SYS:
        return True
    if verdict == "UNDECIDED":
        return True  # P-fail protocol
    return False


def pool():
    bd, cl = [], []
    for d in (BOBER_DIR, HANDCRAFTED_DIR):
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".onnx"):
                    bd.append((f[:-5], os.path.join(d, f)))
    for f in sorted(os.listdir(CLEAN_DIR)):
        if f.endswith(".onnx"):
            p = os.path.join(CLEAN_DIR, f)
            if os.path.getsize(p) <= 10 * 1024 * 1024:
                cl.append((f[:-5], p))
    return bd, cl


def main():
    bd, cl = pool()
    print(f"bd={len(bd)} clean={len(cl)} (expect 22/79)")
    results = {"description": "E10b tau_adm sweep, additive-branch certifier OFF",
               "gdp_flags": FLAGS, "tau_sys": TAU_SYS, "taus": TAUS,
               "per_tau": {}, "per_model": {}}
    for tau in TAUS:
        tp = fp = fn = tn = 0
        t0 = time.time()
        per = {}
        for name, p in bd:
            try:
                vr = verify_model(p, tau_adm=tau, gdp_flags=dict(FLAGS))
                flagged = classify(vr.verdict, getattr(vr, "total_epsilon", 0))
                per[name] = (vr.verdict, flagged)
            except Exception as exc:
                flagged = True
                per[name] = (f"ERROR:{type(exc).__name__}", True)
            tp, fn = (tp + 1, fn) if flagged else (tp, fn + 1)
        for name, p in cl:
            try:
                vr = verify_model(p, tau_adm=tau, gdp_flags=dict(FLAGS))
                flagged = classify(vr.verdict, getattr(vr, "total_epsilon", 0))
                per[name] = (vr.verdict, flagged)
            except Exception as exc:
                flagged = True
                per[name] = (f"ERROR:{type(exc).__name__}", True)
            fp, tn = (fp + 1, tn) if flagged else (fp, tn + 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        results["per_tau"][str(tau)] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
                                            prec=prec, recall=rec, f1=f1,
                                            time_sec=time.time() - t0)
        results["per_model"][str(tau)] = {k: list(v) for k, v in per.items()}
        print(f"tau={tau}: TP={tp} FP={fp} FN={fn} TN={tn} "
              f"prec={prec:.3f} rec={rec:.3f} F1={f1:.3f}")
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
