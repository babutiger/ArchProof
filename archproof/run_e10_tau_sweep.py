#!/usr/bin/env python3
"""E10: τ_adm hyperparameter sensitivity sweep on the 101-model slice.

For each τ_adm ∈ {0.1, 0.3, 0.5}, re-runs ArchProof on the baseline
comparison slice (22 backdoor + 79 clean) and reports TP/FP/FN/TN
under the P-fail protocol (undecided counts as flag).

Output: benchmark/e10_tau_sweep_real.json
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archproof.verify import verify_model
from archproof.run_b3_real_baselines import (
    BOBER_DIR, HANDCRAFTED_DIR, CLEAN_DIR,
)

TAU_SYS = 1e-3
TAUS = [0.1, 0.3, 0.5]


def classify(verdict, eps):
    if verdict in ("DORMANT", "OUTPUT-PRESERVED"):
        return True
    if verdict in ("τ-BOUNDED", "ε-BOUNDED") and eps is not None and eps > TAU_SYS:
        return True
    if verdict == "UNDECIDED":
        return True  # P-fail protocol
    return False


def main():
    bd = []
    for d in [BOBER_DIR, HANDCRAFTED_DIR]:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".onnx"):
                    bd.append((f.replace(".onnx", ""), os.path.join(d, f)))
    cl = []
    for f in sorted(os.listdir(CLEAN_DIR)):
        if f.endswith(".onnx"):
            p = os.path.join(CLEAN_DIR, f)
            if os.path.getsize(p) <= 10 * 1024 * 1024:
                cl.append((f.replace(".onnx", ""), p))

    print(f"bd={len(bd)} clean={len(cl)}")

    results = {"description": "E10 τ_adm sensitivity sweep",
               "tau_sys": TAU_SYS,
               "taus": TAUS,
               "per_tau": {}}

    for tau in TAUS:
        print(f"\n=== τ_adm = {tau} ===")
        tp = fp = fn = tn = 0
        t0 = time.time()
        for name, p in bd:
            try:
                vr = verify_model(p, tau_adm=tau)
                flagged = classify(vr.verdict, getattr(vr, "total_epsilon", 0))
            except Exception:
                flagged = True  # P-fail: errors count as flag
            if flagged: tp += 1
            else:        fn += 1
        for name, p in cl:
            try:
                vr = verify_model(p, tau_adm=tau)
                flagged = classify(vr.verdict, getattr(vr, "total_epsilon", 0))
            except Exception:
                flagged = True  # P-fail: errors count as flag
            if flagged: fp += 1
            else:        tn += 1
        dt = time.time() - t0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results["per_tau"][str(tau)] = dict(
            tp=tp, fp=fp, fn=fn, tn=tn,
            prec=prec, recall=rec, f1=f1, time_sec=dt,
        )
        print(f"τ={tau}: TP={tp} FP={fp} FN={fn} TN={tn} F1={f1:.3f} ({dt:.1f}s)")

    out = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/e10_tau_sweep_real.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
