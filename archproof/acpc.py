"""ACPC: Adversarial Calibration Poisoning Certificate.

Theorem (see ACPC_THEOREM.md, Theorem 1). Under ρ<1/2 adversarial corruption
of the calibration oracle, ArchProof's ε-certificate degrades by at most

    | ε̂ − ε* |  ≤  Σ_i L_φ_i · (1 + c) · Δ_i(ρ) · ‖p_i‖_∞        (bias)
                +  O(L_chain · n^{-1/2})                           (sampling)

where Δ_i(ρ) = (ρ/(1-2ρ)) · max_s (MAD_s(G_i) + IQR_s(G_i)) is the
Hampel worst-case breakdown functional applied to both the median shift and
the MAD shift.

This module implements:
  - `acpc_bound(...)` : closed-form upper bound on certificate degradation
  - `sample_complexity(...)` : corollary, required n for target tolerance τ
  - Helpers for computing L_φ, IQR, MAD from data
"""

import math
import numpy as np
from typing import Optional, Sequence


# =============================================================================
# Activation-envelope Lipschitz constants L_φ = sup |φ'(x)|
#
# These are the Lipschitz constants of the envelope function
#   ε_φ : (lb, ub) → sup_{z∈[lb,ub]} |φ(z)|
# with respect to the L∞ metric on (lb, ub). For any activation φ, since
# ε_φ is the max of |φ| over [lb,ub], and |φ| is sup|φ'|-Lipschitz, ε_φ
# itself is also sup|φ'|-Lipschitz in the endpoint coordinates.
#
# Values match `archproof/layerwise_epsilon.py` ACTIVATION_LIPSCHITZ dict.
# =============================================================================

ACTIVATION_ENVELOPE_LIPSCHITZ = {
    "relu":      1.0,
    "leakyrelu": 1.0,
    "sigmoid":   0.25,     # max σ'(x) = 0.25
    "tanh":      1.0,
    "gelu":      1.13,     # max |GELU'(x)| ≈ 1.1289 at x ≈ 1.57
    "silu":      1.11,     # max |SiLU'(x)| ≈ 1.0998
    "swish":     1.11,
    "hardswish": 1.0,      # piecewise linear, slope ≤ 1
    "hardtanh":  1.0,
    "elu":       1.0,      # slope = 1 for x>0, bounded for x≤0
    "selu":      1.0507 * 1.6733,  # SELU scale × α
    "softplus":  1.0,      # (e^x/(1+e^x)) ≤ 1
    "mish":      1.09,     # max |Mish'(x)| ≈ 1.09 at x ≈ 0.99
    "softmax":   1.0,      # trivial envelope ε=1, Lipschitz 0; use 1 as upper
}


# =============================================================================
# Hampel breakdown functional: Δ(ρ) = ρ / (1 − 2ρ)
# =============================================================================

def hampel_breakdown_factor(rho: float) -> float:
    """Classical Hampel (1974) median breakdown factor ρ/(1-2ρ).

    Closed form: for any sample of n reals with ρn replaced adversarially,
        |median_corrupted − median_clean|  ≤  ρ/(1-2ρ) · IQR

    Linear in ρ for small ρ (first-order Taylor: ρ + 2ρ² + 4ρ³ + ...).
    Blows up at ρ → 1/2 (breakdown point).

    Raises ValueError for ρ ≥ 0.5 (beyond breakdown).
    """
    if rho < 0:
        raise ValueError(f"rho must be ≥ 0, got {rho}")
    if rho >= 0.5:
        raise ValueError(f"rho < 0.5 required (breakdown point); got {rho}")
    return rho / (1.0 - 2.0 * rho)


def interquartile_range(x: np.ndarray) -> np.ndarray:
    """Per-coordinate IQR = Q3 − Q1.  x: [n, d] → [d]."""
    x = np.asarray(x, dtype=np.float64)
    q1 = np.quantile(x, 0.25, axis=0)
    q3 = np.quantile(x, 0.75, axis=0)
    return q3 - q1


def median_absolute_deviation(x: np.ndarray) -> np.ndarray:
    """Per-coordinate MAD.  x: [n, d] → [d]."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x, axis=0)
    return np.median(np.abs(x - med), axis=0)


def sample_median_asymptotic_variance(
        x: np.ndarray,
        bandwidth: Optional[float] = None) -> np.ndarray:
    """Per-coordinate asymptotic variance of sample median under n samples.

    Classical CLT (Koenker 2005, §4.3):
        √n (median_n − μ) → N(0, 1/(4 f²(μ)))

    Density f(μ) is estimated via kernel density at the empirical median
    with Silverman bandwidth. For d-dim input, per-coordinate estimates.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    med = np.median(x, axis=0)
    # Silverman rule of thumb bandwidth per coordinate
    if bandwidth is None:
        sigma = np.std(x, axis=0)
        bandwidth = 0.9 * np.minimum(sigma,
                                      interquartile_range(x) / 1.34) * n ** (-0.2)
    # Gaussian KDE estimate of f(median)
    # f(μ) ≈ (1/n) Σ_j (1/h) φ((x_j - μ)/h), where φ is N(0,1) pdf.
    z = (x - med[None, :]) / np.maximum(bandwidth[None, :], 1e-10)
    kde_vals = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    f_hat_at_median = kde_vals.mean(axis=0) / np.maximum(bandwidth, 1e-10)
    # Asymptotic variance of sample median = 1 / (4 f² n)
    asymptotic_var = 1.0 / (4.0 * np.maximum(f_hat_at_median, 1e-10) ** 2 * n)
    return np.maximum(asymptotic_var, 1e-20)


# =============================================================================
# ACPC main bound
# =============================================================================

def acpc_bound(rho: float,
                 n: int,
                 gate_data: Sequence[np.ndarray],
                 activations: Sequence[str],
                 payload_linf_norms: Sequence[float],
                 c: float = 3.0,
                 L_chain: float = 1.0,
                 include_sampling: bool = True) -> dict:
    """Closed-form ACPC upper bound on |ε̂(X̂) − ε*(X)|.

    Args:
      rho               : adversary's corruption fraction, 0 ≤ ρ < 0.5
      n                 : total calibration sample count
      gate_data         : list of length k, each [n, d_i] clean gate pre-activations
      activations       : list of length k, activation name strings
      payload_linf_norms: list of length k, ‖p_i‖_∞ values
      c                 : B_clean scale factor (default 3.0, per T5)
      L_chain           : chain L∞ Lipschitz product (per T3); 1.0 for gate-only
      include_sampling  : include O(L_chain / √n) term (default True)

    Returns:
      dict with keys:
        - "bias_bound"      : Σ_i L_φ_i · (1+c) · Δ_i(ρ) · ‖p_i‖_∞  (term I)
        - "sampling_bound"  : L_chain / √n aggregate (term II)
        - "total_bound"     : term I + term II
        - "per_gate"        : list of per-gate breakdown details
        - "rho_factor"      : ρ/(1-2ρ) for diagnostics
    """
    assert 0.0 <= rho < 0.5, f"rho must be in [0, 0.5), got {rho}"
    assert len(gate_data) == len(activations) == len(payload_linf_norms), \
        "all three input lists must have same length (= k gates)"

    rho_factor = hampel_breakdown_factor(rho)

    bias_total = 0.0
    per_gate = []

    for i, (gdata, act, p_inf) in enumerate(
            zip(gate_data, activations, payload_linf_norms)):
        gdata = np.asarray(gdata, dtype=np.float64)
        if gdata.ndim == 1:
            gdata = gdata[:, None]

        mad_i = median_absolute_deviation(gdata)                 # [d_i]
        iqr_i = interquartile_range(gdata)                        # [d_i]
        L_phi = ACTIVATION_ENVELOPE_LIPSCHITZ.get(act.lower(), 1.0)

        # Δ_i(ρ) per-coord = ρ/(1-2ρ) · (MAD + IQR);
        # take max over coords (worst-case across gate dims)
        delta_i = rho_factor * (mad_i + iqr_i).max()

        # Per-gate bias contribution
        gate_bias = L_phi * (1.0 + c) * delta_i * p_inf
        bias_total += gate_bias

        per_gate.append({
            "gate_idx": i,
            "activation": act,
            "L_phi": L_phi,
            "max_MAD": float(mad_i.max()),
            "max_IQR": float(iqr_i.max()),
            "delta_i": float(delta_i),
            "payload_linf": float(p_inf),
            "gate_bias_contribution": float(gate_bias),
        })

    # Sampling term: aggregate asymptotic stddev of sample median per gate,
    # scaled by (1+c) · L_φ · ‖p‖_∞ · L_chain
    sampling_bound = 0.0
    if include_sampling:
        for i, (gdata, act, p_inf) in enumerate(
                zip(gate_data, activations, payload_linf_norms)):
            gdata = np.asarray(gdata, dtype=np.float64)
            if gdata.ndim == 1:
                gdata = gdata[:, None]
            L_phi = ACTIVATION_ENVELOPE_LIPSCHITZ.get(act.lower(), 1.0)
            avar = sample_median_asymptotic_variance(gdata)  # [d_i]
            asymptotic_sd = float(np.sqrt(avar.max()))
            per_gate[i]["sampling_sd"] = asymptotic_sd
            # Use 3 × asymptotic_sd as ~99% CLT envelope
            sampling_bound += L_phi * (1.0 + c) * 3 * asymptotic_sd * p_inf
        sampling_bound *= L_chain

    total = bias_total + sampling_bound

    return {
        "rho": rho,
        "n": n,
        "rho_factor": rho_factor,
        "bias_bound": float(bias_total),
        "sampling_bound": float(sampling_bound),
        "total_bound": float(total),
        "L_chain": L_chain,
        "c": c,
        "per_gate": per_gate,
    }


def sample_complexity(tau: float,
                        rho: float,
                        gate_data: Sequence[np.ndarray],
                        activations: Sequence[str],
                        payload_linf_norms: Sequence[float],
                        c: float = 3.0,
                        L_chain: float = 1.0) -> dict:
    """Corollary 1: required n for target certificate tolerance τ given ρ.

    Decouples bias and sampling:
      - Bias term must be ≤ τ/2 : constrains ρ
      - Sampling term must be ≤ τ/2 : solves for n
    """
    assert tau > 0
    assert 0.0 <= rho < 0.5

    # First check: is ρ small enough?
    bound_zero_n = acpc_bound(rho=rho, n=10**9,       # large n → sampling ≈ 0
                                 gate_data=gate_data,
                                 activations=activations,
                                 payload_linf_norms=payload_linf_norms,
                                 c=c, L_chain=L_chain,
                                 include_sampling=False)
    bias_only = bound_zero_n["bias_bound"]

    if bias_only > tau:
        return {
            "feasible": False,
            "reason": f"bias bound {bias_only:.4f} > tau {tau:.4f} even with n → ∞; "
                        f"reduce ρ or accept larger τ",
            "bias_bound": bias_only,
        }

    # Sampling budget
    sampling_budget = tau - bias_only   # ≥ 0

    # Compute n s.t. sampling_bound(n) ≤ sampling_budget.
    # Sampling scales as 1/√n, so n ≥ (coefficient / budget)²
    coef = 0.0
    for i, (gdata, act, p_inf) in enumerate(
            zip(gate_data, activations, payload_linf_norms)):
        gdata = np.asarray(gdata, dtype=np.float64)
        if gdata.ndim == 1:
            gdata = gdata[:, None]
        L_phi = ACTIVATION_ENVELOPE_LIPSCHITZ.get(act.lower(), 1.0)
        # Asymptotic std ∝ 1/√n, so at n=1: σ_1 = √(1/(4 f²))
        avar = sample_median_asymptotic_variance(gdata)
        sigma_at_n1 = float(np.sqrt(avar.max() * len(gdata)))  # scale out √n
        coef += L_phi * (1.0 + c) * 3 * sigma_at_n1 * p_inf
    coef *= L_chain

    n_required = int(math.ceil((coef / max(sampling_budget, 1e-12)) ** 2))

    return {
        "feasible": True,
        "tau": tau,
        "rho": rho,
        "bias_bound": bias_only,
        "sampling_budget": float(sampling_budget),
        "n_required": n_required,
    }


# =============================================================================
# Empirical shift computation (for experimental validation)
# =============================================================================

def empirical_certificate_shift(clean_X: np.ndarray,
                                   poisoned_X: np.ndarray,
                                   activation: str = "relu",
                                   c: float = 3.0) -> float:
    """Compute ACTUAL (not bounded) |ε̂ − ε*| for a single gate by comparing
    the certificate on clean data vs poisoned data.

    Used to verify: empirical_shift ≤ acpc_bound (soundness check).
    """
    from archproof.activation_epsilon import activation_epsilon

    def _gate_cert(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med[None, :]), axis=0)
        lb = med - c * mad
        ub = med + c * mad
        # per-coord ε_φ, take max as the gate's worst-case contribution
        eps_per_coord = np.array([activation_epsilon(activation, float(l), float(u))
                                     for l, u in zip(lb, ub)])
        return float(eps_per_coord.max())

    return abs(_gate_cert(poisoned_X) - _gate_cert(clean_X))


if __name__ == "__main__":
    # Smoke test: verify bound is finite and positive on a toy case
    print("=" * 80)
    print("ACPC smoke test — toy 1-gate example")
    print("=" * 80)

    rng = np.random.RandomState(0)
    n, d = 400, 16
    clean = rng.randn(n, d) * 2.0 + 1.0

    for rho in [0.0, 0.05, 0.1, 0.2, 0.4]:
        result = acpc_bound(
            rho=rho, n=n,
            gate_data=[clean],
            activations=["sigmoid"],
            payload_linf_norms=[1.0],
            c=3.0, L_chain=1.0,
        )
        print(f"  ρ={rho:.2f}  bias={result['bias_bound']:.4f}  "
              f"sampling={result['sampling_bound']:.4f}  "
              f"total={result['total_bound']:.4f}")

    print("\nSample complexity for τ=0.1, ρ=0.05:")
    sc = sample_complexity(tau=0.1, rho=0.05,
                              gate_data=[clean],
                              activations=["sigmoid"],
                              payload_linf_norms=[1.0],
                              c=3.0, L_chain=1.0)
    print(f"  {sc}")
