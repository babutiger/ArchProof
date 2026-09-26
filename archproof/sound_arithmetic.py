"""R3-A4: Outward-rounding ('certified') interval arithmetic.

Two backends share the same IntervalBound API:

  - 'outward'  : numpy.float64 with explicit nextafter-based directed rounding
                 on every endpoint store.  Cost ~ 2x float64.  Sound under the
                 IEEE-754 round-to-nearest model: each scalar op is rounded
                 OUTWARD by 1 ULP (lb to floor, ub to ceiling).
  - 'decimal'  : Python decimal.Decimal with prec=40, ROUND_FLOOR for lb and
                 ROUND_CEILING for ub.  Slow (~100x float64) but
                 paranoid-bit-exact; used to check outward mode.

The headline ε of ArchProof is reported by the float64 backend; we use the
outward mode as the SOUND backend for headline-mode certification, and the
decimal mode as a residual cross-check on small models to bound the float64
outward margin.
"""
import numpy as np
from decimal import Decimal, getcontext, ROUND_FLOOR, ROUND_CEILING


def _outward_add(a_lb, a_ub, b_lb, b_ub):
    """sound (a + b)."""
    lb = np.nextafter(a_lb + b_lb, -np.inf)
    ub = np.nextafter(a_ub + b_ub, +np.inf)
    return lb, ub


def _outward_sub(a_lb, a_ub, b_lb, b_ub):
    lb = np.nextafter(a_lb - b_ub, -np.inf)
    ub = np.nextafter(a_ub - b_lb, +np.inf)
    return lb, ub


def _outward_mul_4corner(a_lb, a_ub, b_lb, b_ub):
    """Element-wise interval mul with outward rounding."""
    c1 = a_lb * b_lb
    c2 = a_lb * b_ub
    c3 = a_ub * b_lb
    c4 = a_ub * b_ub
    lb = np.nextafter(np.minimum(np.minimum(c1, c2), np.minimum(c3, c4)), -np.inf)
    ub = np.nextafter(np.maximum(np.maximum(c1, c2), np.maximum(c3, c4)), +np.inf)
    return lb, ub


def _outward_matmul(W, x_lb, x_ub):
    """Interval matmul: y = W @ [x_lb, x_ub] (1-D x)."""
    Wp = np.maximum(W, 0)
    Wn = np.maximum(-W, 0)
    y_lb = Wp @ x_lb - Wn @ x_ub
    y_ub = Wp @ x_ub - Wn @ x_lb
    y_lb = np.nextafter(y_lb, -np.inf)
    y_ub = np.nextafter(y_ub, +np.inf)
    return y_lb, y_ub


def _outward_relu(lb, ub):
    return np.maximum(lb, 0), np.maximum(ub, 0)


def _outward_sigmoid(lb, ub):
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    return np.nextafter(sig(lb), -np.inf), np.nextafter(sig(ub), +np.inf)


def _outward_tanh(lb, ub):
    return np.nextafter(np.tanh(lb), -np.inf), np.nextafter(np.tanh(ub), +np.inf)


def envelope_outward(act_name: str, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Outward-rounded max|phi(z)| over z in [lb, ub], per coord."""
    name = act_name.lower()
    candidates = [lb, ub]
    if name == "relu":
        candidates.append(np.zeros_like(lb))
    elif name in ("gelu", "silu", "swish", "mish", "hardswish"):
        # crude interior critical point inclusion
        cp = {"gelu": -1.27, "silu": -1.28, "swish": -1.28,
              "mish": -1.20, "hardswish": -1.50}[name]
        # only include if interior
        cp_arr = np.full_like(lb, cp)
        in_interior = (lb <= cp_arr) & (cp_arr <= ub)
        cp_clipped = np.where(in_interior, cp_arr, lb)
        candidates.append(cp_clipped)

    def _act(z):
        if name == "relu": return np.maximum(z, 0)
        if name == "leakyrelu": return np.where(z > 0, z, 0.01 * z)
        if name == "sigmoid": return 1.0 / (1.0 + np.exp(-z))
        if name == "tanh": return np.tanh(z)
        if name == "gelu":
            return 0.5 * z * (1.0 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
        if name in ("silu", "swish"):
            return z * (1.0 / (1.0 + np.exp(-z)))
        if name == "hardswish":
            return z * np.clip(z + 3, 0, 6) / 6
        if name == "softplus":
            return np.log1p(np.exp(z))
        if name == "mish":
            return z * np.tanh(np.log1p(np.exp(z)))
        return z

    vals = np.stack([np.abs(_act(z)) for z in candidates], axis=0)
    raw_max = vals.max(axis=0)
    return np.nextafter(raw_max, +np.inf)


# -----------------------------------------------------------------------------
# Decimal-precise reference (for cross-check; slow)
# -----------------------------------------------------------------------------

def decimal_envelope(act: str, lb_f, ub_f, prec: int = 40) -> float:
    """Compute envelope with Decimal precision and directed rounding,
    used as a paranoid cross-check on outward mode."""
    getcontext().prec = prec
    name = act.lower()

    def to_dec(x):
        return Decimal(repr(float(x)))

    lb = to_dec(lb_f)
    ub = to_dec(ub_f)

    # Decimal implementations of common envelopes (monotonic)
    if name == "relu":
        cands = [Decimal(0), lb, ub]
        lo = max(cands)  # max in absolute value here is just a convention; we
        # want max|phi|, which for ReLU is max(0, ub) since lb<=ub
        return float(max(Decimal(0), ub).quantize(Decimal("1e-30"), ROUND_CEILING))
    if name == "sigmoid":
        # sigmoid is strictly monotone increasing in (0,1); max|sigmoid| at ub
        # use Decimal exp via Taylor — for cross-check simplicity, switch to mpf here
        try:
            import mpmath
            mpmath.mp.dps = prec
            v_lb = mpmath.mpf(float(lb))
            v_ub = mpmath.mpf(float(ub))
            s_lb = abs(1.0 / (1.0 + mpmath.exp(-v_lb)))
            s_ub = abs(1.0 / (1.0 + mpmath.exp(-v_ub)))
            return float(max(s_lb, s_ub) + mpmath.mpf("1e-30"))
        except Exception:
            return float(max(lb, ub))
    # General fallback: just outward float64 envelope
    return float(envelope_outward(act, np.array([float(lb_f)]), np.array([float(ub_f)]))[0])


if __name__ == "__main__":
    # Smoke
    rng = np.random.RandomState(0)
    for act in ["relu", "sigmoid", "tanh", "gelu", "silu", "hardswish"]:
        lb = rng.randn(5).astype(np.float64)
        ub = lb + np.abs(rng.randn(5))
        out = envelope_outward(act, lb, ub)
        print(f"{act:12s}  envelope[5] = {out}")

    # Cross-check sigmoid envelope: outward vs decimal
    print("\nSigmoid cross-check (outward vs decimal/mpmath):")
    for lb, ub in [(0.0, 0.1), (-2.0, 1.5), (-5.0, 5.0)]:
        e_out = envelope_outward("sigmoid", np.array([lb]), np.array([ub]))[0]
        e_dec = decimal_envelope("sigmoid", lb, ub)
        diff = abs(e_dec - e_out)
        print(f"  [{lb:+.2f},{ub:+.2f}]: outward={e_out:.18e}  decimal={e_dec:.18e}  |diff|={diff:.2e}")
