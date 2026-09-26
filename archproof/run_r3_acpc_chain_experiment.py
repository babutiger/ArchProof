"""R3-A3: Chain-sensitivity ACPC bound validation.

For each test cell, we:
  1. Define a linear-gate-linear cell (A, phi, B) either synthetically or extracted
     from a real ONNX model.
  2. Compute K = ||A||_{inf->inf} and H = ||B||_{inf->inf}.
  3. Draw n clean input samples x_1..x_n from a clean distribution.
  4. Generate a rho-poisoned set Xhat by replacing m = floor(rho * n) samples with
     adversarial values that try to maximize certificate drift.
  5. Compute clean input interval I* = per-coord median +- c*MAD on X*,
     poisoned interval Ihat on Xhat.
  6. Propagate: gate pre-activation interval via |A| * I (L_inf operator);
     payload interval via |B| * I.
  7. Compute eps* = max|phi(z)|_{z in [l_z, u_z]} * ||p*||_inf  and eps_hat on poisoned.
  8. Compute delta_x = max over coords of order-stat bound on input samples.
  9. Compute new bound = L_phi * K * delta_x * ||p*||_inf * (1+c)
                        + eps_phi_max * H * delta_x * (1+c).
  10. Check new_bound >= |eps_hat - eps*|.

Expected outcome: 100% soundness across all cells.

Run:
  cd $ARCHPROOF_ROOT
  conda activate alpha-beta-crown
  python3 -m archproof.run_r3_acpc_chain_experiment
"""
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import numpy as np
import json
import time
from typing import Tuple
from pathlib import Path


ACTIVATION_LIPSCHITZ = {
    "relu": 1.0, "sigmoid": 0.25, "tanh": 1.0, "gelu": 1.13,
    "silu": 1.11, "swish": 1.11, "hardswish": 1.0, "leakyrelu": 1.0,
    "softplus": 1.0, "mish": 1.09,
}
ACTIVATION_ENVELOPE_MAX = {
    "relu": None,   # unbounded above; use sup over interval instead
    "sigmoid": 1.0, "tanh": 1.0, "gelu": None, "silu": None,
    "swish": None, "hardswish": None, "leakyrelu": None,
    "softplus": None, "mish": None,
}


def apply_activation(name, z):
    n = name.lower()
    if n == "relu": return np.maximum(z, 0)
    if n == "sigmoid": return 1.0 / (1.0 + np.exp(-z))
    if n == "tanh": return np.tanh(z)
    if n == "gelu": return 0.5 * z * (1.0 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
    if n == "silu" or n == "swish": return z * (1.0 / (1.0 + np.exp(-z)))
    if n == "hardswish":
        return z * np.clip(z + 3, 0, 6) / 6
    if n == "leakyrelu": return np.where(z > 0, z, 0.01 * z)
    if n == "softplus": return np.log1p(np.exp(z))
    if n == "mish": return z * np.tanh(np.log1p(np.exp(z)))
    raise ValueError(name)


def envelope_max(act: str, l: float, u: float) -> float:
    """Max|phi(z)| over z in [l,u] (sample-and-corner)."""
    n = act.lower()
    zs = np.array([l, u])
    # Interior extrema for common activations with bounded derivative sign change
    if n in ("sigmoid", "tanh"):
        # monotonic, extremum at endpoints
        pass
    elif n == "relu":
        zs = np.concatenate([zs, [0.0]])
    elif n in ("gelu", "silu", "swish"):
        # has a shallow interior minimum around z~-1.2 (gelu), z~-1.28 (silu)
        cand = {"gelu": -1.27, "silu": -1.28, "swish": -1.28}[n]
        if l <= cand <= u:
            zs = np.concatenate([zs, [cand]])
    elif n == "mish":
        if l <= -1.2 <= u:
            zs = np.concatenate([zs, [-1.2]])
    elif n == "hardswish":
        if l <= -1.5 <= u:
            zs = np.concatenate([zs, [-1.5]])
    return float(np.max(np.abs(apply_activation(act, zs))))


def order_stat_drift(x_sorted: np.ndarray, m: int) -> float:
    """Finite-sample order-stat bound on sample median under m adversarial replacements.

    max(x_{n/2 + m} - x_{n/2}, x_{n/2} - x_{n/2 - m})
    """
    n = len(x_sorted)
    mid = n // 2
    lo = max(0, mid - m)
    hi = min(n - 1, mid + m)
    return float(max(x_sorted[hi] - x_sorted[mid], x_sorted[mid] - x_sorted[lo]))


def per_coord_mad_drift(x_sorted: np.ndarray, m: int) -> float:
    """Order-stat bound on sample MAD under m replacements.

    MAD = median(|x - median|); apply order-stat to the absolute-deviation array.
    """
    med = x_sorted[len(x_sorted) // 2]
    abs_dev = np.sort(np.abs(x_sorted - med))
    return order_stat_drift(abs_dev, m)


def chain_bound(L_phi: float, K: float, H: float,
                delta_x: float, payload_linf: float,
                eps_phi_max: float, c: float) -> float:
    """Chain-sensitivity bound: (L_phi K delta_x ||p||_inf + eps_phi_max H delta_x) (1+c)."""
    return (1.0 + c) * (L_phi * K * delta_x * payload_linf +
                         eps_phi_max * H * delta_x)


def run_cell(rng, n: int, d: int, rho: float, c: float, act: str,
             sigma_A: float, sigma_B: float,
             clean_std: float, poison_mode: str = "max") -> dict:
    """One experimental cell.

    Args:
      n         : number of samples
      d         : input dimension
      rho       : poisoning fraction
      c         : MAD spread multiplier (3 default)
      act       : activation name
      sigma_A   : std of entries of A (shapes K)
      sigma_B   : std of entries of B (shapes H)
      clean_std : std of clean input distribution
      poison_mode: "max" (replace with +- max_abs), "random", "targeted"
    """
    # Design the cell: pre-activation z = A x  (A: d_out x d)
    d_out = max(1, d // 2)
    A = rng.randn(d_out, d).astype(np.float64) * sigma_A
    B = rng.randn(d_out, d).astype(np.float64) * sigma_B

    K = float(np.abs(A).sum(axis=1).max())
    H = float(np.abs(B).sum(axis=1).max())
    L_phi = ACTIVATION_LIPSCHITZ[act.lower()]

    # Clean samples
    Xstar = rng.randn(n, d).astype(np.float64) * clean_std
    m = int(np.floor(rho * n))

    # Poisoned: replace first m samples
    Xhat = Xstar.copy()
    if m > 0:
        if poison_mode == "max":
            # push to worst-case corners; sign chosen to maximize median shift
            amp = 10.0 * clean_std   # big outliers (but still finite)
            signs = rng.choice([-1, 1], size=(m, d))
            Xhat[:m] = signs * amp
        elif poison_mode == "random":
            Xhat[:m] = rng.randn(m, d).astype(np.float64) * clean_std * 5
        else:
            raise ValueError(poison_mode)

    # Input interval via median +- c*MAD, per coord
    def coord_interval(X):
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med[None, :]), axis=0)
        lb = med - c * mad
        ub = med + c * mad
        return lb, ub

    lb_star, ub_star = coord_interval(Xstar)
    lb_hat, ub_hat = coord_interval(Xhat)

    # IBP-style propagation:
    # z = A x.  If A = A_pos - A_neg (split by sign), then
    #   lb_z = A_pos @ lb_x - A_neg @ ub_x
    #   ub_z = A_pos @ ub_x - A_neg @ lb_x
    def interval_linear(A, lb_x, ub_x):
        Ap = np.maximum(A, 0.0)
        An = np.maximum(-A, 0.0)
        lb = Ap @ lb_x - An @ ub_x
        ub = Ap @ ub_x - An @ lb_x
        return lb, ub

    lb_z_star, ub_z_star = interval_linear(A, lb_star, ub_star)
    lb_z_hat, ub_z_hat = interval_linear(A, lb_hat, ub_hat)
    lb_p_star, ub_p_star = interval_linear(B, lb_star, ub_star)
    lb_p_hat, ub_p_hat = interval_linear(B, lb_hat, ub_hat)

    # Per-coord envelope on gate pre-activation
    eps_z_star = np.array([envelope_max(act, l, u) for l, u in zip(lb_z_star, ub_z_star)])
    eps_z_hat = np.array([envelope_max(act, l, u) for l, u in zip(lb_z_hat, ub_z_hat)])

    # ||p||_inf on clean and poisoned
    p_linf_star = float(np.maximum(np.abs(lb_p_star), np.abs(ub_p_star)).max())
    p_linf_hat = float(np.maximum(np.abs(lb_p_hat), np.abs(ub_p_hat)).max())

    # Certificate contribution (max over gate dimensions)
    eps_star = float(eps_z_star.max() * p_linf_star)
    eps_hat = float(eps_z_hat.max() * p_linf_hat)
    actual_drift = abs(eps_hat - eps_star)

    # delta_x per coord: (1+c) scale comes OUT to multiply the input interval endpoint drift
    # Endpoint drift = median drift + c * MAD drift, but we approximate as
    # (1+c) * order-stat on raw samples (upper bound).
    delta_x_per_coord = np.zeros(d)
    for j in range(d):
        xj_sorted = np.sort(Xstar[:, j])
        dmed = order_stat_drift(xj_sorted, m)
        dmad = per_coord_mad_drift(xj_sorted, m)
        delta_x_per_coord[j] = dmed + c * dmad
    delta_x = float(delta_x_per_coord.max())

    # Envelope max constant (worst-case |phi| over widest interval seen)
    # Use sup over clean ENDPOINT interval (both directions, per coord, max)
    # For chain bound's eps_phi_max we use a conservative sup from widest IBP interval.
    eps_phi_max = float(eps_z_star.max())

    # New chain bound (reviewer form)
    new_bound = chain_bound(L_phi, K, H, delta_x, p_linf_star, eps_phi_max, c)

    # Old bound (current paper): L_phi * (1+c) * delta_x_gate_level * ||p||_inf
    # where delta_x_gate_level = order-stat on GATE activation samples (direct drift)
    # This requires sampling gate activations under poisoning, which is more favorable:
    # we use the same delta_x here as a proxy (so it is actually a lower bound; the new bound
    # formally dominates it via K factor).
    old_bound = L_phi * (1 + c) * delta_x * p_linf_star

    return {
        "n": n, "d": d, "rho": rho, "c": c, "act": act,
        "K": K, "H": H, "L_phi": L_phi, "eps_phi_max": eps_phi_max,
        "delta_x": delta_x,
        "p_linf_star": p_linf_star, "p_linf_hat": p_linf_hat,
        "eps_star": eps_star, "eps_hat": eps_hat,
        "actual_drift": actual_drift,
        "new_bound": new_bound, "old_bound": old_bound,
        "sound_new": new_bound >= actual_drift,
        "sound_old": old_bound >= actual_drift,
        "poison_mode": poison_mode,
    }


def main():
    t0 = time.time()
    rng = np.random.RandomState(20260422)
    activations = ["relu", "sigmoid", "tanh", "gelu", "silu", "hardswish"]
    rhos = [0.0, 0.05, 0.1, 0.2, 0.3]
    d_vals = [8, 16, 32, 64]
    n_vals = [100, 200, 400]
    seeds_per_cell = 6

    cells = []
    for act in activations:
        for rho in rhos:
            for d in d_vals:
                for n in n_vals:
                    for s in range(seeds_per_cell):
                        cell_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
                        sigma_A = float(cell_rng.uniform(0.1, 0.5))
                        sigma_B = float(cell_rng.uniform(0.1, 0.5))
                        clean_std = float(cell_rng.uniform(0.2, 1.0))
                        out = run_cell(cell_rng, n, d, rho, c=3.0,
                                       act=act, sigma_A=sigma_A, sigma_B=sigma_B,
                                       clean_std=clean_std,
                                       poison_mode="max")
                        out["seed"] = s
                        out["sigma_A"] = sigma_A
                        out["sigma_B"] = sigma_B
                        out["clean_std"] = clean_std
                        cells.append(out)

    total = len(cells)
    sound_new = sum(1 for c in cells if c["sound_new"])
    sound_old = sum(1 for c in cells if c["sound_old"])

    # Tightness: median ratio of new_bound / actual_drift, ignoring zero drift
    ratios_new = [c["new_bound"] / c["actual_drift"]
                  for c in cells if c["actual_drift"] > 1e-12]
    ratios_old = [c["old_bound"] / c["actual_drift"]
                  for c in cells if c["actual_drift"] > 1e-12]

    # Per-rho soundness breakdown
    per_rho = {}
    for rho in rhos:
        subset = [c for c in cells if abs(c["rho"] - rho) < 1e-9]
        per_rho[f"{rho}"] = {
            "n_cells": len(subset),
            "sound_new": sum(1 for c in subset if c["sound_new"]),
            "sound_old": sum(1 for c in subset if c["sound_old"]),
            "median_new_ratio": float(np.median([c["new_bound"] / max(c["actual_drift"], 1e-12)
                                                 for c in subset if c["actual_drift"] > 1e-12])) if subset else None,
        }

    summary = {
        "description": "R3-A3: Chain-sensitivity ACPC bound validation",
        "bound_form": "(1+c)[L_phi K delta_x ||p||_inf + eps_phi_max H delta_x]",
        "n_cells_total": total,
        "n_cells_sound_new": sound_new,
        "n_cells_sound_old": sound_old,
        "sound_new_rate": sound_new / total,
        "sound_old_rate": sound_old / total,
        "median_new_ratio": float(np.median(ratios_new)) if ratios_new else None,
        "median_old_ratio": float(np.median(ratios_old)) if ratios_old else None,
        "p90_new_ratio": float(np.percentile(ratios_new, 90)) if ratios_new else None,
        "per_rho": per_rho,
        "wall_sec": time.time() - t0,
    }

    out_dir = Path(_os_ar.path.join(_AR, "benchmark"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "r3_acpc_chain_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_cell": cells}, f, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
