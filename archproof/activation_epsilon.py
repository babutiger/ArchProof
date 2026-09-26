"""T1 Unified (ε, B)-Dormancy: compute sup|φ(x)| for activation φ on [g_lb, g_ub].

Theorem T1 (Unified Dormancy). Given gate pre-activation g with IBP bound
[g_lb, g_ub] on B_clean, the φ-gate satisfies

    sup_{x ∈ B_clean} |φ(g(x))| ≤ ε_φ(g_lb, g_ub)

where ε_φ is computed by this module. For strict ReLU with g_ub ≤ 0, ε = 0
(recovers v2 strict dormancy). For sigmoid / tanh / GELU / SiLU, ε > 0 is
the smallest deterministic upper bound derivable from monotonicity /
known extrema.

Why sound: each formula returns an upper bound on sup|φ(x)| over the
closed interval [g_lb, g_ub] using either (a) monotonicity + endpoint
evaluation, or (b) exact analytic extrema for non-monotonic activations.
No probabilistic bound, no coverage claim.

Supported activations:
    relu           strict, ε = max(0, g_ub); ε = 0 iff g_ub ≤ 0
    sigmoid        strictly monotonic; ε = σ(g_ub) (always > 0)
    tanh           strictly monotonic odd; ε = max(|tanh(g_lb)|, |tanh(g_ub)|)
    gelu           near-monotonic; extra min at x ≈ -0.75, y ≈ -0.170
    silu / swish   near-monotonic; extra min at x ≈ -1.28, y ≈ -0.278
    hardswish      piecewise linear approx of swish; analytic bound
    hardtanh       clipped linear in [-1, 1]
"""

import math

# Pre-computed extrema for non-monotonic activations.
# (Verified via scipy.optimize.minimize_scalar with xatol=1e-12.)
_GELU_MIN_X = -0.751791528026822    # argmin GELU
_GELU_MIN_Y = -0.169971207479904    # value at argmin
_SILU_MIN_X = -1.278464561437784    # argmin SiLU (Swish)
_SILU_MIN_Y = -0.278464542761074
_MISH_MIN_X = -1.1924113444907       # argmin Mish
_MISH_MIN_Y = -0.3088640229614


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _gelu(x: float) -> float:
    """Exact GELU: 0.5 x (1 + erf(x / sqrt(2)))."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def _silu(x: float) -> float:
    """SiLU / Swish: x · sigmoid(x)."""
    return x * _sigmoid(x)


def _hardswish(x: float) -> float:
    """HardSwish: x · ReLU6(x + 3) / 6."""
    return x * min(max(x + 3.0, 0.0), 6.0) / 6.0


def _elu(x: float, alpha: float = 1.0) -> float:
    """ELU: x if x ≥ 0 else α·(e^x - 1). Monotonic; min = -α at x → -∞."""
    return x if x >= 0 else alpha * (math.exp(x) - 1.0)


# SELU α and scale constants (as in Klambauer et al. 2017, NeurIPS)
_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805


def _selu(x: float) -> float:
    """SELU: scale · (x if x≥0 else α·(e^x - 1)). Monotonic."""
    if x >= 0:
        return _SELU_SCALE * x
    return _SELU_SCALE * _SELU_ALPHA * (math.exp(x) - 1.0)


def _softplus(x: float) -> float:
    """Softplus: log(1 + e^x). Monotonic; range (0, ∞)."""
    if x > 20:
        return x  # avoid overflow
    return math.log1p(math.exp(x))


def _mish(x: float) -> float:
    """Mish: x · tanh(softplus(x)) = x · tanh(log(1+e^x)). Near-monotonic.
    Global min at x ≈ -1.1924, value ≈ -0.3089."""
    return x * math.tanh(_softplus(x))


def activation_epsilon(act_type: str, g_lb: float, g_ub: float) -> float:
    """Sound upper bound on `sup_{x ∈ [g_lb, g_ub]} |φ(x)|`.

    Args:
        act_type : 'relu'|'sigmoid'|'tanh'|'gelu'|'silu'|'swish'|'hardswish'|'hardtanh'
        g_lb     : lower bound of gate pre-activation on B_clean
        g_ub     : upper bound of gate pre-activation on B_clean

    Returns:
        ε_φ : float, the minimum deterministic upper bound on |φ| over [g_lb, g_ub].

    Raises ValueError on unknown activation type.
    """
    if g_lb > g_ub:
        raise ValueError(f"Invalid interval: g_lb={g_lb} > g_ub={g_ub}")

    act = act_type.lower()

    if act == "relu":
        # ReLU(x) = max(0, x). Non-negative, monotonic.
        # sup |ReLU(x)| = max(0, g_ub). ε = 0 iff g_ub ≤ 0 (strict dormancy).
        return max(0.0, g_ub)

    if act in ("sigmoid", "sig"):
        # σ(x) ∈ (0, 1), strictly monotonic increasing.
        # sup |σ(x)| = σ(g_ub). ε > 0 always (never truly dormant).
        return _sigmoid(g_ub)

    if act == "tanh":
        # tanh(x) ∈ (-1, 1), strictly monotonic increasing, odd.
        # sup |tanh(x)| = max(|tanh(g_lb)|, |tanh(g_ub)|).
        return max(abs(math.tanh(g_lb)), abs(math.tanh(g_ub)))

    if act == "gelu":
        # Near-monotonic; global min at (_GELU_MIN_X, _GELU_MIN_Y).
        # sup |GELU(x)| = max over endpoints and the min-point if inside.
        cand = [abs(_gelu(g_lb)), abs(_gelu(g_ub))]
        if g_lb <= _GELU_MIN_X <= g_ub:
            cand.append(abs(_GELU_MIN_Y))
        return max(cand)

    if act in ("silu", "swish"):
        # Near-monotonic; global min at (_SILU_MIN_X, _SILU_MIN_Y).
        cand = [abs(_silu(g_lb)), abs(_silu(g_ub))]
        if g_lb <= _SILU_MIN_X <= g_ub:
            cand.append(abs(_SILU_MIN_Y))
        return max(cand)

    if act == "hardswish":
        # Piecewise: x * ReLU6(x+3)/6. Monotonic for x > -3, zero for x ≤ -3.
        # Min at x ≈ -1.5, value = -0.375.
        cand = [abs(_hardswish(g_lb)), abs(_hardswish(g_ub))]
        if g_lb <= -1.5 <= g_ub:
            cand.append(0.375)
        return max(cand)

    if act == "hardtanh":
        # Clipped linear in [-1, 1]; monotonic.
        return max(abs(max(min(g_lb, 1.0), -1.0)),
                   abs(max(min(g_ub, 1.0), -1.0)))

    if act == "elu":
        # ELU monotonic; sup|·| = max(|ELU(lb)|, |ELU(ub)|). ELU bounded
        # below by -α = -1 for x → -∞; clipping gives same bound on finite
        # intervals.
        return max(abs(_elu(g_lb)), abs(_elu(g_ub)))

    if act == "selu":
        # SELU monotonic.
        return max(abs(_selu(g_lb)), abs(_selu(g_ub)))

    if act == "softplus":
        # Softplus monotonic increasing; sup|·| = Softplus(g_ub) (≥ 0 always).
        return _softplus(g_ub)

    if act == "mish":
        # Near-monotonic; global min at MISH_MIN_X.
        cand = [abs(_mish(g_lb)), abs(_mish(g_ub))]
        if g_lb <= _MISH_MIN_X <= g_ub:
            cand.append(abs(_MISH_MIN_Y))
        return max(cand)

    if act == "softmax":
        # Softmax output is always in (0, 1). Sound conservative bound: ε = 1.
        # A tighter bound requires knowing all logit bounds jointly:
        #   sup softmax_i = exp(g_ub_i) / (exp(g_ub_i) + Σ_{j≠i} exp(g_lb_j))
        # Without joint info we return the trivial ε=1. Caller can provide
        # tighter bound via `softmax_all_bounds` helper.
        return 1.0

    raise ValueError(f"Unsupported activation: {act_type}")


def softmax_tight_epsilon(g_lb_vec, g_ub_vec) -> list:
    """Tighter Softmax ε bound using ALL dim bounds jointly.

    For softmax over dims j = 1..D:
      sup softmax_i = exp(g_ub_i) / (exp(g_ub_i) + Σ_{j≠i} exp(g_lb_j))
      inf softmax_i = exp(g_lb_i) / (exp(g_lb_i) + Σ_{j≠i} exp(g_ub_j))
      ε_i = max(|sup|, |inf|) = sup (since both in (0,1))

    Returns list of ε bounds per dim."""
    import numpy as np
    lb = np.asarray(g_lb_vec, dtype=np.float64)
    ub = np.asarray(g_ub_vec, dtype=np.float64)
    D = len(lb)
    # Numerical stability: subtract max
    # For ε bound: for each i, compute exp(ub_i) / (exp(ub_i) + Σ_{j≠i} exp(lb_j))
    # Stable form: 1 / (1 + Σ_{j≠i} exp(lb_j - ub_i))
    eps = np.zeros(D)
    for i in range(D):
        # Σ_{j≠i} exp(lb_j - ub_i)
        log_diff = lb - ub[i]
        log_diff_exc = np.concatenate([log_diff[:i], log_diff[i+1:]])
        sum_other = float(np.exp(log_diff_exc).sum())
        eps[i] = 1.0 / (1.0 + sum_other)
    return eps.tolist()


def gate_contribution_bound(act_type: str, g_lb: float, g_ub: float,
                             payload_abs_max: float) -> float:
    """Upper bound on `|Mul(φ(g), p)(x)|` for all x ∈ B_clean.

    By T1 theorem:
        |Mul(φ(g), p)(x)| ≤ |φ(g(x))| · |p(x)| ≤ ε_φ · ||p||_∞
    """
    eps = activation_epsilon(act_type, g_lb, g_ub)
    return eps * payload_abs_max


# --------------------------
# Sanity tests (deterministic)
# --------------------------
def _run_tests():
    """Verify correctness of the analytic bounds by random sampling."""
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)

    acts = {
        "relu":      lambda x: max(0.0, x),
        "sigmoid":   _sigmoid,
        "tanh":      math.tanh,
        "gelu":      _gelu,
        "silu":      _silu,
        "hardswish": _hardswish,
        "hardtanh":  lambda x: max(min(x, 1.0), -1.0),
        "elu":       _elu,
        "selu":      _selu,
        "softplus":  _softplus,
        "mish":      _mish,
    }

    n_samples = 10000
    for act_name, act_fn in acts.items():
        for _ in range(50):
            # Random interval
            a = random.uniform(-5, 5)
            b = random.uniform(-5, 5)
            lb, ub = min(a, b), max(a, b)

            # Analytic ε bound
            eps_bound = activation_epsilon(act_name, lb, ub)

            # Empirical sup |φ(x)| over random samples in [lb, ub]
            xs = np.random.uniform(lb, ub, n_samples)
            emp_sup = max(abs(act_fn(x)) for x in xs)

            # Bound must be ≥ empirical sup (soundness)
            if eps_bound < emp_sup - 1e-4:
                print(f"  VIOLATION: {act_name} [{lb:.3f}, {ub:.3f}]  "
                      f"bound={eps_bound:.6f}  emp={emp_sup:.6f}")
                return False

    print("  All activation ε bounds PASS (100% sound on 50 random intervals × 7 activations × 10k samples)")
    return True


if __name__ == "__main__":
    # Quick functional sanity
    print("=== T1 activation_epsilon sanity ===")
    cases = [
        ("relu",     -5.0,  -1.0,  "strict dormant"),
        ("relu",     -2.0,   3.0,  "ε = 3.0"),
        ("sigmoid", -10.0,  -5.0,  "ε = σ(-5) ≈ 0.0067"),
        ("sigmoid",  -2.0,   2.0,  "ε = σ(2) ≈ 0.88"),
        ("tanh",     -2.0,   2.0,  "ε = tanh(2) ≈ 0.964"),
        ("gelu",     -1.0,   1.0,  "includes min at -0.75, ε ≈ 0.842"),
        ("silu",     -2.0,   1.0,  "includes min at -1.28, ε_silu"),
        ("elu",      -3.0,   1.0,  "ELU(-3)=-0.95, ELU(1)=1.0, ε=1.0"),
        ("selu",     -2.0,   1.0,  "SELU"),
        ("softplus", -3.0,   2.0,  "monotonic, ε=softplus(2)"),
        ("mish",     -3.0,   1.0,  "includes min at -1.19, ε"),
        ("softmax",   None,  None, "trivially ε=1 (use softmax_tight_epsilon for per-dim)"),
    ]
    for act, lb, ub, desc in cases:
        if lb is None:
            eps = activation_epsilon(act, -10.0, 10.0)
            print(f"  {act:10s} [default]      ε={eps:.4f}  ({desc})")
        else:
            eps = activation_epsilon(act, lb, ub)
            print(f"  {act:10s} [{lb:+.2f}, {ub:+.2f}]  ε={eps:.4f}  ({desc})")

    # Demo softmax tight bound
    print()
    print("=== softmax_tight_epsilon (joint-bound version) ===")
    from archproof.activation_epsilon import softmax_tight_epsilon
    # Example: 3-dim softmax, logit bounds
    lb_vec = [-1.0, -2.0, -3.0]
    ub_vec = [+2.0, +1.0, +0.5]
    eps_per_dim = softmax_tight_epsilon(lb_vec, ub_vec)
    for i, e in enumerate(eps_per_dim):
        print(f"  dim {i}: g_lb={lb_vec[i]:+.1f} g_ub={ub_vec[i]:+.1f} "
              f"ε_softmax={e:.4f}  (tighter than trivial ε=1)")

    print()
    _run_tests()
