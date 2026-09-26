"""ACPC multi-gate validation (R5 gap fix).

R4 and R5 reviewers flagged: all existing ACPC runners test k=1 gate,
‖p_i‖_∞=1, L_chain=1 — a specialization. Theorem 1a claims the bound

    | ε̂(X̂) − ε*(X) |  ≤  Σ_{i=1}^k L_φ_i · (1 + c) · Δ_i(ρ) · ‖p_i‖_∞

holds for arbitrary k, payload norms, and chain Lipschitz. This runner
exercises the compositional generalization.

Sweep:
  k ∈ {2, 3, 5, 10} gates  (multi-gate)
  activations mixed per gate (relu + sigmoid + gelu)
  payload L∞ norms ‖p_i‖_∞ drawn per-gate (non-unit)
  ρ ∈ {0, 0.05, 0.1, 0.2, 0.3}  (ρ=0.4 near breakdown)
  n ∈ {400, 800}, 3 seeds

For each cell:
  - Generate k independent gate pre-activation vectors
  - Attach per-gate payload norm
  - Compute ε*(X) = Σ_i ε_φ_i(B_clean_i(X)) · ‖p_i‖_∞
  - Corrupt ρ·n samples (per gate independently)
  - Compute ε̂(X̂) same way on corrupted data
  - Empirical shift = |ε̂(X̂) − ε*(X)|
  - Theoretical bound = Σ_i L_φ_i · (1+c) · Δ_i(ρ) · ‖p_i‖_∞  (bias only, Thm 1a)
  - Sound iff empirical ≤ theoretical

Output: benchmark/v3_acpc_multigate.json
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
    acpc_bound, hampel_breakdown_factor,
    ACTIVATION_ENVELOPE_LIPSCHITZ,
    median_absolute_deviation, interquartile_range,
)
from archproof.robust_b_clean import inject_poison
from archproof.activation_epsilon import activation_epsilon

OUT_JSON = _os_ar.path.join(_AR, "benchmark/v3_acpc_multigate.json")


def multi_gate_certificate(gate_data_list, activations, payload_linf_norms,
                              c: float = 3.0) -> float:
    """Compute ε_total = Σ_i ε_φ_i(B_clean_i) · ‖p_i‖_∞ for multi-gate case.

    Each gate's data is treated independently (per Theorem 1a sum bound)."""
    total = 0.0
    for gdata, act, p_inf in zip(gate_data_list, activations,
                                    payload_linf_norms):
        gdata = np.asarray(gdata, dtype=np.float64)
        if gdata.ndim == 1:
            gdata = gdata[:, None]
        med = np.median(gdata, axis=0)
        mad = np.median(np.abs(gdata - med[None, :]), axis=0)
        lb = med - c * mad
        ub = med + c * mad
        eps_per_coord = np.array([activation_epsilon(act, float(l), float(u))
                                     for l, u in zip(lb, ub)])
        eps_gate = float(eps_per_coord.max())
        total += eps_gate * p_inf
    return total


def run_one_cell(k, activations, payload_norms, rho, n, seed,
                    attacker_range=15.0):
    """Run one multi-gate ACPC validation cell."""
    rng = np.random.RandomState(seed)

    # Generate k independent gate pre-activations (each n samples, d=16 dims)
    gate_data_clean = []
    for i in range(k):
        # Different distributions per gate to stress the bound
        mean_i = 0.5 + i * 0.3
        std_i  = 1.0 + 0.2 * i
        data_i = rng.randn(n, 16).astype(np.float64) * std_i + mean_i
        gate_data_clean.append(data_i)

    # Inject poison per-gate (each gate sees ρ-fraction corruption independently)
    gate_data_poisoned = []
    if rho > 0:
        for data_i in gate_data_clean:
            poisoned_i = inject_poison(data_i, poison_frac=rho,
                                          attacker_range=attacker_range,
                                          mode="outlier_shift")
            gate_data_poisoned.append(poisoned_i)
    else:
        gate_data_poisoned = [d.copy() for d in gate_data_clean]

    # Compute multi-gate ε certificates (clean + poisoned)
    eps_clean = multi_gate_certificate(gate_data_clean, activations,
                                           payload_norms, c=3.0)
    eps_poisoned = multi_gate_certificate(gate_data_poisoned, activations,
                                              payload_norms, c=3.0)
    emp_shift = abs(eps_poisoned - eps_clean)

    # Theoretical ACPC bound (Theorem 1a strict, bias-only)
    bound = acpc_bound(rho=rho, n=n,
                         gate_data=gate_data_clean,
                         activations=activations,
                         payload_linf_norms=payload_norms,
                         c=3.0, L_chain=1.0,
                         include_sampling=False)   # Thm 1a strict

    return {
        "k": k,
        "activations": list(activations),
        "payload_norms": list(payload_norms),
        "rho": rho, "n": n, "seed": seed,
        "epsilon_clean":     float(eps_clean),
        "epsilon_poisoned":  float(eps_poisoned),
        "empirical_shift":   float(emp_shift),
        "acpc_bias_bound":   float(bound["bias_bound"]),
        "acpc_total_bound":  float(bound["total_bound"]),
        "sound": bool(emp_shift <= bound["total_bound"]),
    }


def main():
    print("=" * 100)
    print("ACPC multi-gate validation — Theorem 1a compositional generalization")
    print("=" * 100)

    # Mixed activation roster — cycled per gate
    ALL_ACTIVATIONS = ["relu", "sigmoid", "gelu", "tanh", "silu"]

    ks = [2, 3, 5, 10]
    rhos = [0.00, 0.05, 0.10, 0.20, 0.30]
    ns = [400, 800]
    seeds = list(range(3))

    results = []
    n_sound = 0
    n_total = 0
    agg = defaultdict(list)

    total_cells = len(ks) * len(rhos) * len(ns) * len(seeds)
    print(f"Total cells: {total_cells} "
          f"(k={len(ks)} × ρ={len(rhos)} × n={len(ns)} × seeds={len(seeds)})")
    print()

    print(f"{'k':>3s} {'ρ':>6s} {'n':>5s} {'activations':>30s} "
          f"{'payload_norms':>30s} {'emp μ±σ':>16s} {'bias μ':>12s} {'sound':>8s}")
    print("-" * 120)

    for k in ks:
        # Deterministic payload norms and activations per k (for reproducibility)
        rng_cfg = np.random.RandomState(42 + k)
        activations = [ALL_ACTIVATIONS[i % len(ALL_ACTIVATIONS)] for i in range(k)]
        # Payload norms: random in [0.5, 3.0] per gate — non-unit, realistic spread
        payload_norms = rng_cfg.uniform(0.5, 3.0, size=k).tolist()

        for rho in rhos:
            for n in ns:
                for seed in seeds:
                    r = run_one_cell(k, activations, payload_norms, rho, n, seed)
                    results.append(r)
                    agg[(k, rho, n)].append(r)
                    n_total += 1
                    if r["sound"]:
                        n_sound += 1

    # Aggregated print
    aggregated = []
    for (k, rho, n), cells in sorted(agg.items()):
        emps = [c["empirical_shift"] for c in cells]
        bnds = [c["acpc_total_bound"] for c in cells]
        sounds = sum(c["sound"] for c in cells)
        first = cells[0]
        aggregated.append({
            "k": k, "rho": rho, "n": n,
            "activations": first["activations"],
            "payload_norms": first["payload_norms"],
            "emp_shift_mean":  float(np.mean(emps)),
            "emp_shift_std":   float(np.std(emps)),
            "acpc_bound_mean": float(np.mean(bnds)),
            "tightness_ratio_mean": float(np.mean(
                [e/b if b > 1e-12 else 0.0 for e, b in zip(emps, bnds)])),
            "n_sound": sounds,
            "n_total": len(cells),
        })
        # Print cells with n=800 only to avoid spam
        if n == 800:
            acts_str = ",".join(a[:4] for a in first["activations"][:4]) + \
                          ("..." if len(first["activations"]) > 4 else "")
            pay_str = ",".join(f"{p:.2f}" for p in first["payload_norms"][:4]) + \
                         ("..." if len(first["payload_norms"]) > 4 else "")
            print(f"  {k:>2d} {rho:>4.2f} {n:>5d}  {acts_str:>28s}  "
                  f"{pay_str:>28s}  "
                  f"{np.mean(emps):>7.3f}±{np.std(emps):.3f}   "
                  f"{np.mean(bnds):>8.3f}    "
                  f"{sounds}/{len(cells)}")

    print("\n" + "=" * 100)
    print(f"SOUNDNESS: {n_sound}/{n_total} multi-gate cells "
          f"empirical ≤ ACPC bias-only bound (Thm 1a strict)")
    print("=" * 100)
    if n_sound == n_total:
        print("✓ ALL MULTI-GATE CELLS SOUND — Theorem 1a generalizes empirically "
              f"to k ∈ {ks}, mixed activations, non-unit payloads.")
    else:
        print(f"✗ {n_total - n_sound} violations detected.")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "theorem_ref": "ACPC Theorem 1a multi-gate compositional validation",
            "sweep": {"ks": ks, "rhos": rhos, "ns": ns, "n_seeds": len(seeds),
                         "activations_pool": ALL_ACTIVATIONS,
                         "payload_norm_range": [0.5, 3.0]},
            "n_cells_total": n_total,
            "n_cells_sound": n_sound,
            "all_sound": bool(n_sound == n_total),
            "aggregated": aggregated,
            "per_cell": results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
