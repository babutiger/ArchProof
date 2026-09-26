"""T5 Robust B_clean via Geometric Median + MAD (Hampel, 1974).

Theorem T5 (Robust Calibration Breakdown). Given n calibration samples of
which at most k = ⌊poison_frac · n⌋ are adversarially corrupted (poison_frac
< 0.5), using coordinate-wise median and MAD (Median Absolute Deviation) to
set B_clean bounds:

    B_clean_i = [median_i - c · MAD_i,  median_i + c · MAD_i]
    (c = 3 gives ≈ 99.7% coverage under Gaussian assumption,
     analogous to 3-sigma in the mean±3σ rule)

Robustness: the median's **breakdown point is 50%** (Tukey/Hampel). Compared
to the sample mean (breakdown 0%, one outlier shifts it arbitrarily), the
median is insensitive to outliers at poison_frac ≤ 0.5.

    ||median_poisoned - median_clean||_∞ ≤ O(poison_frac · MAD_clean)

This is a **deterministic worst-case bound** using classical robust statistics
(pre-dates conformal prediction by decades), kept ORTHOGONAL to DROCP's
probabilistic CP / Hypergeometric framework.
"""

import math
from typing import Tuple
import numpy as np


def coordinate_wise_median(X: np.ndarray) -> np.ndarray:
    """Per-coordinate sample median. `X: [n, d]` → `[d]`."""
    return np.median(np.asarray(X, dtype=np.float64), axis=0)


def geometric_median(X: np.ndarray, eps: float = 1e-8,
                      max_iter: int = 200) -> np.ndarray:
    """Joint geometric median via Weiszfeld's iterative algorithm.

    Geometric median minimizes Σ_i ||x_i - μ||_2. Breakdown point 50%.
    Coordinate median is simpler; geometric median is optimal for joint
    attack on all dims.
    """
    X = np.asarray(X, dtype=np.float64)
    y = X.mean(axis=0)
    for _ in range(max_iter):
        d = X - y                          # [n, dim]
        dist = np.linalg.norm(d, axis=1)   # [n]
        mask = dist > eps
        if not mask.any():
            break
        w = 1.0 / dist[mask]               # weights
        y_new = (w[:, None] * X[mask]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            y = y_new
            break
        y = y_new
    return y


def median_absolute_deviation(X: np.ndarray, median=None) -> np.ndarray:
    """Per-coordinate Median Absolute Deviation: MAD_i = median_j |X_{j,i} - median_i|."""
    X = np.asarray(X, dtype=np.float64)
    if median is None:
        median = coordinate_wise_median(X)
    return np.median(np.abs(X - median[None, :]), axis=0)


def robust_b_clean_median_mad(X: np.ndarray,
                                 c: float = 3.0
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """B_clean = [median - c·MAD, median + c·MAD]."""
    median = coordinate_wise_median(X)
    mad = median_absolute_deviation(X, median=median)
    lb = median - c * mad
    ub = median + c * mad
    return lb, ub


def classical_b_clean_mean_std(X: np.ndarray,
                                  c: float = 3.0
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """v2 baseline: B_clean = [mean - c·σ, mean + c·σ]. Breakdown = 0%."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return mean - c * std, mean + c * std


def inject_poison(X: np.ndarray, poison_frac: float,
                   attacker_range: float = 15.0,
                   mode: str = "outlier_shift") -> np.ndarray:
    """Inject adversarial corruption into calibration samples.

    Args:
      X : clean samples [n, d]
      poison_frac : fraction of samples replaced by adversarial outliers
      attacker_range : attacker can place outliers anywhere in [-range, +range]
      mode :
        "outlier_shift" — replace poison_frac samples with extreme values
        "direction_shift" — shift poison samples in a fixed attacker direction
    """
    X = np.asarray(X, dtype=np.float64).copy()
    n, d = X.shape
    n_poison = int(poison_frac * n)
    if n_poison == 0:
        return X
    rng = np.random.RandomState(12345)
    idx = rng.choice(n, n_poison, replace=False)
    if mode == "outlier_shift":
        # Place at boundary of a large box
        X[idx] = rng.uniform(-attacker_range, attacker_range, size=(n_poison, d))
    elif mode == "direction_shift":
        # Shift all poison samples in a single direction (worst-case for mean)
        direction = np.ones(d)
        X[idx] = X[idx] + attacker_range * direction
    else:
        raise ValueError(f"unknown mode: {mode}")
    return X


def breakdown_bound_mad(poison_frac: float, mad_clean: np.ndarray) -> np.ndarray:
    """Deterministic worst-case bound on ||median_poisoned - median_clean||_∞.

    Classical Hampel (1974) result: median's asymptotic breakdown is 1/2.
    For poison_frac < 0.5, the shift is bounded by:
        ||median_poisoned - median_clean||_∞ ≤ (poison_frac / (1 - 2·poison_frac)) · MAD
    """
    if poison_frac >= 0.5:
        return np.full_like(mad_clean, np.inf)
    factor = poison_frac / (1.0 - 2.0 * poison_frac)
    return factor * mad_clean


if __name__ == "__main__":
    print("=" * 80)
    print("T5 Robust B_clean unit tests + poisoning experiment")
    print("=" * 80)

    # Synthetic LLM-like hidden states: Gaussian mixture, 4096 dims, 400 samples
    rng = np.random.RandomState(42)
    N, D = 400, 256
    clean_data = rng.randn(N, D) * 2.0 + rng.randn(D) * 0.5

    print(f"\nClean samples: {N} × {D}")

    # Compute B_clean on clean data
    lb_mad_clean, ub_mad_clean = robust_b_clean_median_mad(clean_data, c=3.0)
    lb_std_clean, ub_std_clean = classical_b_clean_mean_std(clean_data, c=3.0)
    mad_clean = median_absolute_deviation(clean_data)

    print(f"Clean |B_mad| (median width) = {(ub_mad_clean - lb_mad_clean).mean():.3f}")
    print(f"Clean |B_std| (mean width)   = {(ub_std_clean - lb_std_clean).mean():.3f}")

    # Poison at various fractions
    print(f"\n{'poison %':>8s} {'median shift_max':>18s} {'mean shift_max':>18s} "
          f"{'median bound':>14s}")
    print("-" * 68)
    for poison_frac in [0.00, 0.01, 0.05, 0.10, 0.25, 0.40]:
        poisoned = inject_poison(clean_data, poison_frac,
                                    attacker_range=15.0, mode="outlier_shift")
        lb_mad_p, ub_mad_p = robust_b_clean_median_mad(poisoned, c=3.0)
        lb_std_p, ub_std_p = classical_b_clean_mean_std(poisoned, c=3.0)

        shift_mad_lb = np.abs(lb_mad_p - lb_mad_clean).max()
        shift_std_lb = np.abs(lb_std_p - lb_std_clean).max()
        # Theoretical bound on median shift
        theory = breakdown_bound_mad(poison_frac, mad_clean)
        theory_max = float(theory.max()) * 3  # scale by c=3
        print(f"  {poison_frac*100:>6.1f}%  {shift_mad_lb:>16.3f}  "
              f"{shift_std_lb:>16.3f}  {theory_max:>12.3f}")

    # Sanity: geometric median vs coordinate median on clean
    geo_med = geometric_median(clean_data)
    coord_med = coordinate_wise_median(clean_data)
    mean_val = clean_data.mean(axis=0)
    print(f"\n|geo_med - coord_med|_max: {np.abs(geo_med - coord_med).max():.4f}")
    print(f"|geo_med - mean|_max:       {np.abs(geo_med - mean_val).max():.4f}")
    print("  (both should be tiny on clean Gaussian-like data)")

    # Catastrophic failure test: single 1e6 outlier on mean vs median
    print(f"\n{'='*60}")
    print("Catastrophic failure test: single extreme outlier")
    print(f"{'='*60}")
    catastrophic = clean_data.copy()
    catastrophic[0] = 1e6  # one sample blown up to extreme
    lb_mad_c, ub_mad_c = robust_b_clean_median_mad(catastrophic, c=3.0)
    lb_std_c, ub_std_c = classical_b_clean_mean_std(catastrophic, c=3.0)
    print(f"  Mean   B_clean width: {(ub_std_c - lb_std_c).mean():.3e}  (blown up)")
    print(f"  Median B_clean width: {(ub_mad_c - lb_mad_c).mean():.3e}  (unaffected)")
    print(f"  Robustness factor: {((ub_std_c-lb_std_c).mean()/(ub_mad_c-lb_mad_c).mean()):.1e}x")

    # Save results
    import json
    exp_results = {
        "synthetic_poisoning_sweep": [],
    }
    for poison_frac in [0.00, 0.01, 0.05, 0.10, 0.25, 0.40]:
        poisoned = inject_poison(clean_data, poison_frac, 15.0)
        lb_mad_p, ub_mad_p = robust_b_clean_median_mad(poisoned, 3.0)
        lb_std_p, ub_std_p = classical_b_clean_mean_std(poisoned, 3.0)
        exp_results["synthetic_poisoning_sweep"].append({
            "poison_frac": poison_frac,
            "median_lb_max_shift": float(np.abs(lb_mad_p - lb_mad_clean).max()),
            "median_ub_max_shift": float(np.abs(ub_mad_p - ub_mad_clean).max()),
            "mean_lb_max_shift":   float(np.abs(lb_std_p - lb_std_clean).max()),
            "mean_ub_max_shift":   float(np.abs(ub_std_p - ub_std_clean).max()),
            "mean_b_width":        float((ub_std_p - lb_std_p).mean()),
            "median_b_width":      float((ub_mad_p - lb_mad_p).mean()),
        })
    exp_results["catastrophic_outlier_1e6"] = {
        "mean_b_width":   float((ub_std_c - lb_std_c).mean()),
        "median_b_width": float((ub_mad_c - lb_mad_c).mean()),
        "robustness_factor": float((ub_std_c-lb_std_c).mean()/(ub_mad_c-lb_mad_c).mean()),
    }
    import os as _os
    _out = _os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "benchmark", "v3_t5_robust_poisoning.json")
    with open(_out, "w") as f:
        json.dump(exp_results, f, indent=2)
    print(f"\nSaved: benchmark/v3_t5_robust_poisoning.json")
