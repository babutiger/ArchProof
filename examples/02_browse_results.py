"""Example 2: Browse the committed paper results.

Reproduces the headline numbers in the paper's Tables 4, 7, 8, 11
without running the verifier. Useful for understanding what is in
results/ before deciding which experiment to re-run.

Usage:
    python examples/02_browse_results.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=" * 70)
    print("ArchProof — Browsing committed paper results")
    print("=" * 70)

    # Load the top-level aggregate JSON
    agg = json.loads((ROOT / "results" / "aggregates.json").read_text())

    # ------------------------------------------------------------------
    # Tab. 7 — F1 by uncertified-handling protocol (11+49 in-class slice)
    # ------------------------------------------------------------------
    print("\n[Tab. 7] F1 on the 11+49 in-class slice (ArchProof, by protocol):")
    e2 = agg["e2"]
    for protocol, vals in e2["archproof_per_protocol"].items():
        print(f"  {protocol:8s}  precision={vals['prec']:.3f}  "
              f"recall={vals['rec']:.3f}  F1={vals['f1']:.4f}  "
              f"(tp={vals['tp']}, fp={vals['fp']}, fn={vals['fn']}, tn={vals['tn']})")

    print("\n[Tab. 7] Heuristic baselines on the same slice:")
    for name, vals in e2["binary_baselines"].items():
        print(f"  {name:25s}  F1={vals['f1']:.4f}")

    # ------------------------------------------------------------------
    # Tab. 4 — Whole-model 6-7B LLM ε values
    # ------------------------------------------------------------------
    print("\n[Tab. 4] Whole-model 6-7B LLM verification (per-cell):")
    csv = ROOT / "results" / "per_cell_whole_llm_eic.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        # show one row per LLM
        head_cols = [c for c in ["llm", "config", "epsilon", "verdict", "rescue_pre"] if c in df.columns]
        if head_cols:
            print(df[head_cols].head(30).to_string(index=False))
    else:
        print(f"  (per_cell_whole_llm_eic.csv not present)")

    # ------------------------------------------------------------------
    # Tab. 11 — Cross-machine reproducibility
    # ------------------------------------------------------------------
    print("\n[Tab. 11 — cross-machine] Bit-exact reproducibility:")
    cm_csv = ROOT / "results" / "per_cell_cross_machine_repro.csv"
    if cm_csv.exists():
        df = pd.read_csv(cm_csv)
        if "bit_exact" in df.columns:
            n = len(df)
            n_exact = int(df["bit_exact"].astype(str).eq("True").sum())
            print(f"  {n_exact}/{n} cells bit-exact across two physical hosts")

    # ------------------------------------------------------------------
    # Production-scale summary
    # ------------------------------------------------------------------
    print("\n[Production CNN summary]")
    prod = agg.get("production", {})
    print(f"  n_cases:                  {prod.get('n_cases')}")
    print(f"  n_with_gates_detected:    {prod.get('n_with_gates_detected')}")
    print(f"  n_surgery_succeeded:      {prod.get('n_surgery_succeeded')}")

    print("\nDone.  See results/README.md for the full schema.")


if __name__ == "__main__":
    main()
