"""ACPC empirical validation experiment.

Verifies ACPC Theorem (see `ACPC_THEOREM.md`, `archproof/acpc.py`):

  empirical |ε̂ − ε*|  ≤  acpc_bound(ρ, n)   for all (ρ, n, seed)

Experiment design:
  - Sweep ρ ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.40}
  - Sweep n ∈ {100, 200, 400, 800, 1600}
  - 10 seeds per (ρ, n) cell
  - Activations: sigmoid, relu, gelu, tanh, silu (4 non-trivial + relu)
  - Report: empirical_shift_mean, acpc_bound, soundness_holds

Output: benchmark/v3_acpc_experiment.json
"""

import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import os
import sys
import json
from collections import defaultdict

import numpy as np

sys.path.insert(0, _AR)

from archproof.acpc import (
    acpc_bound, empirical_certificate_shift, hampel_breakdown_factor,
)
from archproof.robust_b_clean import inject_poison

OUT_JSON = _os_ar.path.join(_AR, "benchmark/v3_acpc_experiment.json")


def run_single(rho, n, activation, seed, d=16, attacker_range=15.0):
    """One experiment cell: generate clean data, poison, compute
    empirical shift and theoretical ACPC bound."""
    rng = np.random.RandomState(seed)
    clean = rng.randn(n, d) * 2.0 + 1.0

    if rho > 0:
        poisoned = inject_poison(clean, poison_frac=rho,
                                   attacker_range=attacker_range,
                                   mode="outlier_shift")
    else:
        poisoned = clean.copy()

    # Empirical certificate shift
    emp_shift = empirical_certificate_shift(
        clean_X=clean, poisoned_X=poisoned,
        activation=activation, c=3.0)

    # Theoretical ACPC bound (on CLEAN data statistics — the bound is in terms
    # of clean MAD/IQR per theorem)
    bound = acpc_bound(
        rho=rho, n=n,
        gate_data=[clean],
        activations=[activation],
        payload_linf_norms=[1.0],
        c=3.0, L_chain=1.0,
        include_sampling=False)  # R5 fix: validate Theorem 1a (empirical-to-empirical) strictly

    return {
        "rho": rho, "n": n, "activation": activation, "seed": seed,
        "empirical_shift": float(emp_shift),
        "acpc_total_bound": float(bound["total_bound"]),
        "acpc_bias_bound":  float(bound["bias_bound"]),
        "acpc_sampling_bound": float(bound["sampling_bound"]),
        "sound": bool(emp_shift <= bound["total_bound"]),
    }


def main():
    print("=" * 100)
    print("v3 ACPC empirical validation — ρ × n × activation × seed sweep")
    print("=" * 100)

    rhos = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]
    ns = [100, 200, 400, 800, 1600]
    activations = ["relu", "sigmoid", "gelu", "tanh", "silu"]
    seeds = list(range(10))

    total_cells = len(rhos) * len(ns) * len(activations) * len(seeds)
    print(f"Total cells: {total_cells}  "
          f"(ρ={len(rhos)} × n={len(ns)} × act={len(activations)} × seeds={len(seeds)})")
    print()

    results = []
    n_sound = 0
    n_total = 0

    # Also collect aggregated stats per (ρ, n, activation)
    agg = defaultdict(list)

    for rho in rhos:
        for n in ns:
            for act in activations:
                for seed in seeds:
                    r = run_single(rho=rho, n=n, activation=act, seed=seed)
                    results.append(r)
                    agg[(rho, n, act)].append(r)
                    n_total += 1
                    if r["sound"]:
                        n_sound += 1

    print(f"{'rho':>5s} {'n':>5s} {'activation':>10s} "
          f"{'emp_shift μ±σ':>20s} {'acpc_bound μ':>15s} "
          f"{'ratio max':>10s} {'sound %':>8s}")
    print("-" * 100)

    aggregated = []
    for (rho, n, act), cell_results in sorted(agg.items()):
        emp_shifts = [r["empirical_shift"] for r in cell_results]
        bounds = [r["acpc_total_bound"] for r in cell_results]
        ratios = [e / b if b > 1e-12 else 0.0 for e, b in zip(emp_shifts, bounds)]
        n_sound_cell = sum(r["sound"] for r in cell_results)

        emp_mean, emp_std = np.mean(emp_shifts), np.std(emp_shifts)
        bound_mean = np.mean(bounds)
        ratio_max = max(ratios)

        aggregated.append({
            "rho": rho, "n": n, "activation": act,
            "emp_shift_mean": float(emp_mean),
            "emp_shift_std":  float(emp_std),
            "acpc_bound_mean": float(bound_mean),
            "tightness_ratio_mean": float(np.mean(ratios)),
            "tightness_ratio_max":  float(ratio_max),
            "n_sound":  int(n_sound_cell),
            "n_total":  len(cell_results),
        })

        print(f"  {rho:.2f}   {n:4d}  {act:>10s}  "
              f"{emp_mean:>8.4f}±{emp_std:.4f}    "
              f"{bound_mean:>10.4f}    "
              f"{ratio_max:>7.3f}   "
              f"{n_sound_cell}/{len(cell_results)}")

    print("\n" + "=" * 100)
    print(f"SOUNDNESS CHECK: {n_sound}/{n_total} cells have empirical ≤ ACPC bound")
    print("=" * 100)
    if n_sound == n_total:
        print("✓ ALL CELLS SOUND — ACPC Theorem 1 holds empirically on all "
              "(ρ, n, activation, seed) combinations")
    else:
        violations = [r for r in results if not r["sound"]]
        print(f"✗ {len(violations)} violations — check theorem statement / implementation")
        for v in violations[:5]:
            print(f"    {v}")

    # Linear-in-ρ fit check (fix n and activation, vary ρ)
    print("\nLinear-in-ρ verification (n=800, sigmoid): slope of empirical_shift vs ρ")
    for act in activations:
        rho_vals, emp_vals = [], []
        for agg_row in aggregated:
            if agg_row["n"] == 800 and agg_row["activation"] == act:
                rho_vals.append(agg_row["rho"])
                emp_vals.append(agg_row["emp_shift_mean"])
        if len(rho_vals) >= 2:
            # linear fit through (0, emp(0))
            rhos_arr = np.array(rho_vals)
            emps_arr = np.array(emp_vals)
            # exclude ρ=0 for slope fit (that's the sampling-only baseline)
            mask = rhos_arr > 0
            if mask.sum() >= 2:
                slope, intercept = np.polyfit(rhos_arr[mask], emps_arr[mask], 1)
                print(f"    {act:>10s}  slope={slope:.4f}  intercept={intercept:.4f}  "
                      f"(linearity OK if slope > 0)")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "theorem": "ACPC Theorem 1a — Adversarial Calibration Poisoning Certificate",
            "ref": "ACPC_THEOREM.md, Theorem 1a (empirical-to-empirical)",
            "sweep": {"rhos": rhos, "ns": ns, "activations": activations,
                         "n_seeds": len(seeds)},
            "n_cells_total":  n_total,
            "n_cells_sound":  n_sound,
            "all_sound":      bool(n_sound == n_total),
            "aggregated":     aggregated,
            "per_cell":       results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
