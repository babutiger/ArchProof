"""MGRS: Minimum Gate Removal Set for architectural backdoor remediation.

Given an ArchProof verdict with per-gate contributions
    c_i = ε_φ_i(B_clean_i) · ‖p_i‖_∞            (per T1, T2)
and total output margin
    ε_total = Σ_i c_i                            (per T2)

**MGRS Problem** (independent-gate case): given a target tolerance
`ε_target < ε_total`, find the minimum subset `S ⊆ {1,…,k}` such that
removing all gates in `S` from the ONNX graph yields a residual sum

    ε_residual  =  Σ_{i ∉ S} c_i  ≤  ε_target.

Equivalently, find min |S| s.t.  Σ_{i ∈ S} c_i  ≥  Δ  where  Δ = ε_total − ε_target.

**Greedy-optimal theorem (Theorem 2)**: the greedy algorithm
`sort gates by c_i desc and pick until cumulative ≥ Δ` produces the optimal
(minimum-cardinality) MGRS. Proof: standard exchange argument — if an optimal
solution `S*` does not include the largest c_i, swapping it in with any
lower-c_j ∈ S* preserves feasibility and doesn't increase |S|. ∎

**Why novel**: prior architectural-backdoor defense literature treats subgraph
removal heuristically (ModelScan / Guardian flag, then manual excision). No
prior formal treatment as a weighted-set-cover optimization with soundness
guarantee. Works by e.g. Neuron Pruning (Li 2023) focus on weight-parameterized
backdoors, not architectural gate subgraphs.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# =============================================================================
# Data class for a single gate's contribution (from ArchProof T1/T2 output)
# =============================================================================

@dataclass
class GateContribution:
    """One gate's contribution to ε_total.

    Matches the schema in VerificationResult.gate_epsilons (archproof/verify.py).
    """
    gate_idx: int
    gate_node: str                     # ONNX node name
    activation: str                    # "relu" / "sigmoid" / ...
    epsilon: float                     # ε_φ (from activation_epsilon)
    payload_abs_max: float             # ‖p_i‖_∞
    contribution: float                # c_i = ε · ‖p‖_∞


@dataclass
class MGRSResult:
    """Output of the MGRS algorithm."""
    removed_gate_indices: List[int]   # subset S
    removed_gate_nodes: List[str]     # corresponding ONNX node names
    residual_epsilon: float            # ε after removal
    target_epsilon: float              # ε_target requested
    achieved: bool                     # residual ≤ target
    epsilon_before: float              # ε_total original
    k_total: int                       # total gate count
    k_removed: int                     # |S|
    reduction_ratio: float             # (ε_before - residual) / ε_before


def minimum_gate_removal_set(
        gate_contributions: List[GateContribution],
        target_epsilon: float = 0.0,
) -> MGRSResult:
    """Compute the greedy-optimal MGRS for the given gate contributions.

    Args:
        gate_contributions : list of GateContribution (one per gate in ArchProof
                              verdict)
        target_epsilon     : desired residual output margin.  Default 0 means
                              remove until GDP-FREE.

    Returns:
        MGRSResult with the removed gate set, residual ε, and diagnostics.

    Complexity: O(k log k) via sort.

    Correctness: greedy on sorted contribution yields minimum-cardinality
    S that achieves the target (classical exchange argument).
    """
    k = len(gate_contributions)
    epsilon_before = sum(g.contribution for g in gate_contributions)

    if target_epsilon < 0:
        raise ValueError("target_epsilon must be ≥ 0")
    if target_epsilon >= epsilon_before:
        # Already satisfies target; no removal needed.
        return MGRSResult(
            removed_gate_indices=[], removed_gate_nodes=[],
            residual_epsilon=float(epsilon_before),
            target_epsilon=float(target_epsilon),
            achieved=True,
            epsilon_before=float(epsilon_before),
            k_total=k, k_removed=0,
            reduction_ratio=0.0,
        )

    delta_required = epsilon_before - target_epsilon
    # Sort gates by contribution, desc
    order = sorted(range(k), key=lambda i: gate_contributions[i].contribution,
                    reverse=True)

    removed_idx: List[int] = []
    removed_nodes: List[str] = []
    cum_removed = 0.0
    for i in order:
        if cum_removed >= delta_required:
            break
        removed_idx.append(gate_contributions[i].gate_idx)
        removed_nodes.append(gate_contributions[i].gate_node)
        cum_removed += gate_contributions[i].contribution

    residual = epsilon_before - cum_removed
    return MGRSResult(
        removed_gate_indices=removed_idx,
        removed_gate_nodes=removed_nodes,
        residual_epsilon=float(residual),
        target_epsilon=float(target_epsilon),
        achieved=bool(residual <= target_epsilon + 1e-9),
        epsilon_before=float(epsilon_before),
        k_total=k, k_removed=len(removed_idx),
        reduction_ratio=float((epsilon_before - residual) /
                                max(epsilon_before, 1e-12)),
    )


# =============================================================================
# Iterative MGRS with IBP re-verification oracle
# =============================================================================
#
# MOTIVATION:  Greedy MGRS above is optimal under the independent-gate
# contribution model (Theorem 2). Two separate effects can make its
# *predicted* residual disagree with what post-surgery IBP re-verification
# actually computes on the resulting graph:
#
#   (i) Float64 precision loss in summation: when gate contributions span
#       many orders of magnitude (e.g. c_1 = 2.6e+63 and c_3 = 2.1e+42, a
#       10^-21 ratio vs 2^-52 ≈ 2.2e-16 float64 epsilon), smaller c_i
#       vanish when added to the largest, so `total − c_max` rounds to
#       zero even though the true residual is non-zero. This was directly
#       observed on ResNet-18 / EfficientNet int_un cases in
#       `benchmark/v3_production_scale_case.json` (R9 audit, 2026-04-21).
#
#  (ii) Post-surgery IBP bound propagation on the modified graph can
#       produce a slightly different value than the pre-surgery per-gate
#       contribution sum because intermediate IBP bounds are recomputed
#       end-to-end.
#
# FIX:  iteratively add one more gate (in descending-contribution order)
# and call an ORACLE (post-surgery IBP re-verify) until the oracle agrees
# that ε ≤ target. This remains sound — we only grow the removal set, we
# never drop a gate — and converges in at most k iterations.
#
# =============================================================================

# SOUNDNESS NOTE (R10 codex concern, addressed):
# The iterative algorithm below grows the removal set monotonically
# (never shrinks it). The claim that "oracle(S) is monotone non-
# increasing in |S|" is NOT automatic from growing S — it must hold
# for the specific oracle we use, which is `verify_model` on the
# surgery-modified ONNX graph. That oracle returns
# `total_output_margin = Σ_{i in admitted} ε_φ_i(B_clean_i) · ‖p_i‖_∞`
# (the T2 compositional sum). Zeroing gate i in PyTorch sets the
# corresponding gate activation to 0 in IBP (via the NaN-safe Mul
# rule `[0,0] × [a,b] = [0,0]`), so that gate's contribution becomes
# exactly 0. Other admitted gates' contributions are computed from
# their own weights (unchanged), payload IBP bounds (unchanged
# because payload paths do not depend on zeroed gate branches), so
# they remain identical. Therefore the oracle's returned ε is
# monotone non-increasing in S under T2's independent-gate sum
# model. Outside that model (e.g. gate interference affecting
# payload IBP bounds), monotonicity is not guaranteed; our code
# still terminates after at most k oracle calls and reports
# `achieved_target=False` in that rare case.


@dataclass
class IterativeMGRSResult:
    """Output of `iterative_mgrs_with_ibp_oracle`."""
    removed_gate_nodes: List[str]         # final removal set (ONNX names)
    removed_gate_indices: List[int]       # final removal set (positional)
    oracle_residual_epsilon: float         # oracle's last-call ε
    greedy_prediction_residual: float      # what greedy's analytic sum gave
    n_oracle_calls: int                    # iterations performed
    achieved_target: bool                  # oracle reached ε ≤ target
    greedy_oracle_match: bool              # 1-call greedy passed oracle
    iteration_nodes: List[List[str]]       # incremental removal snapshots


def iterative_mgrs_with_ibp_oracle(
        gate_contributions: List[GateContribution],
        oracle,                             # Callable[[List[str]], float]
        target_epsilon: float = 0.0,
        rel_tol: float = 1e-6,
) -> IterativeMGRSResult:
    """Greedy MGRS + oracle-checked iteration.

    Args:
        gate_contributions: per-gate (ε, ‖p‖, contribution) list. Must have
            a valid `.gate_node` name; oracle receives a list of these.
        oracle: callable taking a list of gate_node names to remove and
            returning the post-surgery IBP ε for the resulting graph. The
            caller (usually `run_v3_production_scale.py`) owns restoring
            PyTorch weights before zeroing.
        target_epsilon: acceptable residual. Default 0 (complete removal).
        rel_tol: oracle-vs-target tolerance. `oracle_eps <= target +
            rel_tol * max(1, target)` is accepted.

    Returns:
        IterativeMGRSResult including provenance fields.
    """
    k = len(gate_contributions)
    if k == 0:
        return IterativeMGRSResult(
            removed_gate_nodes=[], removed_gate_indices=[],
            oracle_residual_epsilon=0.0,
            greedy_prediction_residual=0.0,
            n_oracle_calls=0, achieved_target=True,
            greedy_oracle_match=True, iteration_nodes=[[]]
        )

    def _accept(eps):
        return eps <= target_epsilon + rel_tol * max(1.0, abs(target_epsilon))

    # Step 1: greedy path
    greedy = minimum_gate_removal_set(gate_contributions,
                                         target_epsilon=target_epsilon)
    removed_nodes = list(greedy.removed_gate_nodes)
    removed_indices = list(greedy.removed_gate_indices)
    oracle_eps = float(oracle(removed_nodes))
    n_calls = 1
    iteration_nodes = [list(removed_nodes)]

    if _accept(oracle_eps):
        return IterativeMGRSResult(
            removed_gate_nodes=removed_nodes,
            removed_gate_indices=removed_indices,
            oracle_residual_epsilon=oracle_eps,
            greedy_prediction_residual=float(greedy.residual_epsilon),
            n_oracle_calls=n_calls, achieved_target=True,
            greedy_oracle_match=True,
            iteration_nodes=iteration_nodes,
        )

    # Step 2: iterate, adding the next highest-contribution gate each round
    remaining_order = sorted(range(k),
                              key=lambda i: -gate_contributions[i].contribution)
    already_removed_nodes = set(removed_nodes)
    for i in remaining_order:
        node_i = gate_contributions[i].gate_node
        if node_i in already_removed_nodes:
            continue
        already_removed_nodes.add(node_i)
        removed_nodes = sorted(already_removed_nodes,
                                key=lambda n: next(
                                    (gi for gi, g in enumerate(gate_contributions)
                                     if g.gate_node == n), 0))
        removed_indices = [g.gate_idx for g in gate_contributions
                            if g.gate_node in already_removed_nodes]
        oracle_eps = float(oracle(removed_nodes))
        n_calls += 1
        iteration_nodes.append(list(removed_nodes))
        if _accept(oracle_eps):
            return IterativeMGRSResult(
                removed_gate_nodes=removed_nodes,
                removed_gate_indices=removed_indices,
                oracle_residual_epsilon=oracle_eps,
                greedy_prediction_residual=float(greedy.residual_epsilon),
                n_oracle_calls=n_calls, achieved_target=True,
                greedy_oracle_match=False,
                iteration_nodes=iteration_nodes,
            )

    # Exhausted all gates without reaching target
    return IterativeMGRSResult(
        removed_gate_nodes=removed_nodes,
        removed_gate_indices=removed_indices,
        oracle_residual_epsilon=oracle_eps,
        greedy_prediction_residual=float(greedy.residual_epsilon),
        n_oracle_calls=n_calls, achieved_target=False,
        greedy_oracle_match=False,
        iteration_nodes=iteration_nodes,
    )


# =============================================================================
# Optimality certificate: verify via exhaustive search for small k
# =============================================================================

def verify_mgrs_optimality(gate_contributions: List[GateContribution],
                              target_epsilon: float,
                              mgrs_result: MGRSResult) -> bool:
    """For small k (≤ 20), verify MGRS is optimal by exhaustive enumeration.

    Returns True iff no subset of size |S| − 1 achieves the target.
    """
    from itertools import combinations

    k = len(gate_contributions)
    if k > 22:
        raise ValueError(f"exhaustive verification infeasible for k={k} > 22 "
                          f"(2^22 = 4M combinations)")

    epsilon_before = sum(g.contribution for g in gate_contributions)
    size_needed = mgrs_result.k_removed

    if size_needed == 0:
        return True   # nothing to remove

    # Check if any smaller subset achieves target
    for smaller_size in range(size_needed):
        for S in combinations(range(k), smaller_size):
            cum = sum(gate_contributions[i].contribution for i in S)
            residual = epsilon_before - cum
            if residual <= target_epsilon + 1e-9:
                return False   # found a smaller feasible S — greedy is SUBOPTIMAL
    return True


# =============================================================================
# Main Theorem (Greedy optimality for independent-gate MGRS)
# =============================================================================

MGRS_GREEDY_OPTIMALITY_THEOREM = """
Theorem 2 (MGRS greedy optimality).
Let {c_1, …, c_k} ≥ 0 be gate contributions with epsilon_before = Σ_i c_i.
For any target_epsilon ∈ [0, epsilon_before], let Δ = epsilon_before − target_epsilon.
The greedy algorithm
    S_greedy = take elements in descending-c_i order until Σ_{i∈S_greedy} c_i ≥ Δ
returns a minimum-cardinality subset among all S ⊆ {1,…,k} with
    Σ_{i∈S} c_i  ≥  Δ.

Proof sketch (exchange argument): let S* be any optimal subset.
If S* does not contain the largest c_i (call it c_max, index i*),
then S* must contain some index j with c_j ≤ c_max.
Swap: S' = (S* ∪ {i*}) \\ {j}.
Then |S'| = |S*| and Σ_{S'} c_i ≥ Σ_{S*} c_i ≥ Δ, so S' is also optimal.
Iterating, we obtain an optimal solution that matches the greedy prefix. ∎

Complexity: O(k log k).

Note: this optimality is for the INDEPENDENT-GATE model (T2 sum bound).
Under T7-style gate-interference models, greedy would no longer be optimal;
a weighted-set-cover formulation or LP relaxation would be required.
"""


# =============================================================================
# Helper to construct GateContribution from VerificationResult.gate_epsilons
# =============================================================================

def from_verification_result(gate_epsilons_list: list) -> List[GateContribution]:
    """Parse VerificationResult.gate_epsilons (list of dicts) into
    GateContribution objects.  Schema matches verify.py output.
    """
    out = []
    for idx, g in enumerate(gate_epsilons_list):
        out.append(GateContribution(
            gate_idx=idx,
            gate_node=str(g.get("gate_node", f"gate_{idx}")),
            activation=str(g.get("activation", "unknown")),
            epsilon=float(g.get("epsilon", 0.0)),
            payload_abs_max=float(g.get("payload_abs_max", 0.0)),
            contribution=float(g.get("contribution", 0.0)),
        ))
    return out


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MGRS smoke test + optimality verification")
    print("=" * 80)

    # Toy: 5 gates with contributions 5, 3, 2, 1, 0.5
    gates = [
        GateContribution(i, f"gate_{i}", "relu",
                          epsilon=1.0, payload_abs_max=c, contribution=c)
        for i, c in enumerate([5.0, 3.0, 2.0, 1.0, 0.5])
    ]

    print(f"\nGate contributions: {[g.contribution for g in gates]}")
    print(f"ε_total = {sum(g.contribution for g in gates)}")

    for target in [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]:
        result = minimum_gate_removal_set(gates, target_epsilon=target)
        opt_verified = verify_mgrs_optimality(gates, target, result)
        print(f"\ntarget ε = {target:.1f}:")
        print(f"  removed {result.k_removed}/{result.k_total} gates: "
              f"{result.removed_gate_indices}")
        print(f"  residual ε = {result.residual_epsilon:.3f}  "
              f"(target {result.target_epsilon})  "
              f"achieved={result.achieved}")
        print(f"  optimality verified: {opt_verified}")

    print("\n" + "=" * 80)
    print("Theorem 2 (greedy optimality) — full statement")
    print("=" * 80)
    print(MGRS_GREEDY_OPTIMALITY_THEOREM)
