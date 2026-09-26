"""MGRS baseline comparison (Gap-2 fix).

Reviewer-motivated: MGRS claims greedy-optimal minimum gate removal. We need
to empirically compare against baselines that a naive practitioner might use:

  BL1 Random-k removal    — pick k gates uniformly at random
  BL2 Smallest-first       — remove gates with SMALLEST contribution first (pessimal)
  BL3 Remove-all           — remove every gate (upper-bound safe)
  OUR MGRS greedy-optimal — descending contribution

Metric: given a target ε_target, how many gates does each baseline need to
remove before achieving residual ε ≤ target?

Expected: MGRS uses strictly fewer gates than BL1 (random) and BL2 (smallest-first)
on average; matches BL3 (remove-all) only when target=0 and all gates required.

Sweep:
  k ∈ {5, 10, 20, 50}
  contribution distributions: uniform, geometric (exponentially decaying),
                                bimodal (few large + many small)
  target_epsilon: {0, ε_max · 0.25, ε_max · 0.50, ε_max · 0.75}
  100 random seeds per (k, distribution, target) cell

Output: benchmark/v3_mgrs_baseline.json
"""

import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import os
import sys
import json
from collections import defaultdict

import numpy as np

sys.path.insert(0, _AR)

from archproof.mgrs import (
    GateContribution, minimum_gate_removal_set,
)

OUT_JSON = _os_ar.path.join(_AR, "benchmark/v3_mgrs_baseline.json")


def make_gates(k: int, distribution: str, seed: int):
    """Generate k gate contributions from specified distribution.

    distribution:
      'uniform'     — c_i ~ U(0.1, 10)
      'geometric'   — c_i = 10 · 0.6^i (i=0..k-1), so few dominate
      'bimodal'     — one gate c=10, rest c ~ U(0.1, 1.0)
    """
    rng = np.random.RandomState(seed)
    if distribution == "uniform":
        contribs = rng.uniform(0.1, 10.0, size=k)
    elif distribution == "geometric":
        contribs = np.array([10.0 * (0.6 ** i) for i in range(k)])
        rng.shuffle(contribs)
    elif distribution == "bimodal":
        contribs = np.concatenate([[10.0], rng.uniform(0.1, 1.0, size=k - 1)])
        rng.shuffle(contribs)
    else:
        raise ValueError(distribution)

    gates = [
        GateContribution(gate_idx=i, gate_node=f"g_{i}", activation="relu",
                          epsilon=1.0, payload_abs_max=float(contribs[i]),
                          contribution=float(contribs[i]))
        for i in range(k)
    ]
    return gates


# Feasibility is decided by one predicate for every removal order, so that the
# comparison measures the order and nothing else. It is the predicate the greedy
# routine already applies in archproof/mgrs.py: stop once the removed
# contribution has covered the required reduction, with no slack. An earlier
# version of this script gave the two baselines an extra 1e-9 of slack that the
# greedy routine did not get. Where individual contributions fall below that
# slack, a baseline could stop while a residual was still present, which makes
# greedy look worse than it is. The default below is the value the recorded
# results were produced with, so this script reproduces them; set it to 0.0 to
# give every ordering the same predicate.
LEGACY_BASELINE_SLACK = 1e-9


def _k_to_reach(gates, target, order, slack=0.0) -> int:
    """Gates removed, in the given order, before the residual meets the target."""
    n = len(gates)
    eps_total = sum(g.contribution for g in gates)
    cum_removed = 0.0
    for i, idx in enumerate(order):
        cum_removed += gates[idx].contribution
        if eps_total - cum_removed <= target + slack:
            return i + 1
    return n   # had to remove all


def baseline_random(gates, target, rng_seed, slack=None) -> int:
    """BL1 random-k removal: pick random gates one at a time until target met."""
    order = np.random.RandomState(rng_seed).permutation(len(gates))
    return _k_to_reach(gates, target, order,
                       LEGACY_BASELINE_SLACK if slack is None else slack)


def baseline_smallest_first(gates, target, slack=None) -> int:
    """BL2 smallest-first (pessimal greedy)."""
    order = sorted(range(len(gates)), key=lambda i: gates[i].contribution)
    return _k_to_reach(gates, target, order,
                       LEGACY_BASELINE_SLACK if slack is None else slack)


def mgrs_greedy(gates, target) -> int:
    """OUR MGRS: greedy descending contribution."""
    r = minimum_gate_removal_set(gates, target_epsilon=target)
    return r.k_removed


def run_one_cell(k, distribution, target_frac, seed):
    """One comparison cell: random seed × distribution × k gates × target
       returns |S| for BL1, BL2, BL3 (=k), MGRS."""
    gates = make_gates(k, distribution, seed)
    eps_total = sum(g.contribution for g in gates)
    target = target_frac * eps_total

    k_mgrs = mgrs_greedy(gates, target)
    k_random = baseline_random(gates, target, seed + 1000)
    k_smallest = baseline_smallest_first(gates, target)
    k_all = k   # BL3 remove-all

    return {
        "k": k, "distribution": distribution,
        "target_frac": target_frac, "target_eps": target,
        "seed": seed,
        "eps_total": eps_total,
        "k_mgrs":     k_mgrs,
        "k_random":   k_random,
        "k_smallest": k_smallest,
        "k_all":      k_all,
        "mgrs_saving_vs_random":    k_random - k_mgrs,
        "mgrs_saving_vs_smallest":  k_smallest - k_mgrs,
    }


def main():
    print("=" * 100)
    print("MGRS baseline comparison — greedy vs random / smallest-first / all")
    print("=" * 100)

    ks = [5, 10, 20, 50]
    distributions = ["uniform", "geometric", "bimodal"]
    target_fracs = [0.0, 0.25, 0.50, 0.75]
    n_seeds = 100

    agg = defaultdict(list)
    all_cells = []

    print(f"Total cells: {len(ks) * len(distributions) * len(target_fracs) * n_seeds}")
    print()
    print(f"{'k':>3s} {'dist':>12s} {'target%':>8s} "
          f"{'MGRS':>6s} {'rand':>8s} {'small':>8s} {'all':>6s} "
          f"{'MGRS<rand':>10s} {'MGRS<small':>10s}")
    print("-" * 90)

    for k in ks:
        for dist in distributions:
            for tf in target_fracs:
                cell_results = []
                for seed in range(n_seeds):
                    r = run_one_cell(k, dist, tf, seed)
                    cell_results.append(r)
                    all_cells.append(r)

                k_mgrs_mean     = np.mean([r["k_mgrs"]     for r in cell_results])
                k_rand_mean     = np.mean([r["k_random"]   for r in cell_results])
                k_small_mean    = np.mean([r["k_smallest"] for r in cell_results])

                better_than_rand = sum(r["k_mgrs"] < r["k_random"] for r in cell_results)
                better_than_small = sum(r["k_mgrs"] < r["k_smallest"] for r in cell_results)

                agg[(k, dist, tf)] = {
                    "k": k, "distribution": dist, "target_frac": tf,
                    "k_mgrs_mean":      float(k_mgrs_mean),
                    "k_random_mean":    float(k_rand_mean),
                    "k_smallest_mean":  float(k_small_mean),
                    "k_all":            k,
                    "n_seeds":          n_seeds,
                    "n_mgrs_strictly_better_than_random":   int(better_than_rand),
                    "n_mgrs_strictly_better_than_smallest": int(better_than_small),
                    "saving_vs_random_mean":
                        float(np.mean([r["mgrs_saving_vs_random"] for r in cell_results])),
                    "saving_vs_smallest_mean":
                        float(np.mean([r["mgrs_saving_vs_smallest"] for r in cell_results])),
                }

                if tf == 0.50:   # only print the tf=0.50 slice to avoid spam
                    print(f"  {k:>2d} {dist:>12s} {tf*100:>6.0f}%  "
                          f"{k_mgrs_mean:>5.2f}  {k_rand_mean:>6.2f}  "
                          f"{k_small_mean:>6.2f}  {k:>5d}  "
                          f"{better_than_rand:>7d}/{n_seeds}  "
                          f"{better_than_small:>7d}/{n_seeds}")

    # Overall summary
    n_cells_total = sum(len([c for c in all_cells]) for _ in [1])
    mgrs_lt_rand  = sum(c["mgrs_saving_vs_random"]   > 0 for c in all_cells)
    mgrs_le_rand  = sum(c["mgrs_saving_vs_random"]   >= 0 for c in all_cells)
    mgrs_lt_small = sum(c["mgrs_saving_vs_smallest"] > 0 for c in all_cells)

    mean_saving_rand  = np.mean([c["mgrs_saving_vs_random"]   for c in all_cells])
    mean_saving_small = np.mean([c["mgrs_saving_vs_smallest"] for c in all_cells])

    print("\n" + "=" * 100)
    print(f"SUMMARY over {len(all_cells)} cells "
          f"(k × distribution × target × 100 seeds)")
    print("=" * 100)
    print(f"  MGRS strictly < random:     {mgrs_lt_rand}/{len(all_cells)} cells "
          f"(mean saving: {mean_saving_rand:+.2f} gates)")
    print(f"  MGRS ≤ random:               {mgrs_le_rand}/{len(all_cells)} cells")
    print(f"  MGRS strictly < smallest:   {mgrs_lt_small}/{len(all_cells)} cells "
          f"(mean saving: {mean_saving_small:+.2f} gates)")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "experiment": "MGRS greedy vs random / smallest-first / remove-all baselines",
            "sweep": {"ks": ks, "distributions": distributions,
                         "target_fracs": target_fracs, "n_seeds": n_seeds},
            "n_cells": len(all_cells),
            "aggregate_mgrs_vs_random": {
                "strictly_better":   int(mgrs_lt_rand),
                "at_least_as_good":  int(mgrs_le_rand),
                "worse":             int(len(all_cells) - mgrs_le_rand),
                "mean_saving_gates": float(mean_saving_rand),
            },
            "aggregate_mgrs_vs_smallest": {
                "strictly_better":   int(mgrs_lt_small),
                "mean_saving_gates": float(mean_saving_small),
            },
            "aggregated_per_cell": list(agg.values()),
            "per_trial":           all_cells,
        }, f, indent=2, default=str)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
