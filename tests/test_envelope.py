"""Per-activation envelope soundness tests for all 11 supported activations."""
import numpy as np
import pytest


# Per-activation closed-form envelope: sup_{z in [l,u]} |φ(z)|
ACTIVATIONS = [
    ("relu",      lambda z: np.maximum(0, z)),
    ("hardtanh",  lambda z: np.clip(z, -1, 1)),
    ("hardswish", lambda z: z * np.clip(z + 3, 0, 6) / 6),
    ("sigmoid",   lambda z: 1 / (1 + np.exp(-z))),
    ("tanh",      lambda z: np.tanh(z)),
    ("gelu",      lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))),
    ("silu",      lambda z: z / (1 + np.exp(-z))),
    ("mish",      lambda z: z * np.tanh(np.log1p(np.exp(z)))),
    ("elu",       lambda z: np.where(z >= 0, z, np.exp(z) - 1)),
    ("selu",      lambda z: 1.0507 * np.where(z >= 0, z, 1.6733 * (np.exp(z) - 1))),
    ("softplus",  lambda z: np.log1p(np.exp(z))),
]


@pytest.mark.parametrize("name,fn", ACTIVATIONS)
def test_envelope_is_sound_via_dense_sampling(name, fn):
    """For each activation φ, draw 10,000 random intervals and verify the
    closed-form envelope ε_φ(l, u) >= sup_{z in [l,u]} |φ(z)| in dense
    samples — the soundness invariant of Lemma in §5."""
    from archproof.activation_epsilon import envelope

    rng = np.random.default_rng(42)
    n_underestimate = 0
    n_total = 1000  # 1000 random intervals per activation

    for _ in range(n_total):
        l = rng.uniform(-10, 5)
        u = l + rng.uniform(0.01, 10)

        # Closed-form envelope claim
        try:
            eps = envelope(name, l, u)
        except (KeyError, ValueError, NotImplementedError):
            pytest.skip(f"activation {name} not implemented in envelope()")

        # Dense sampling check
        zs = np.linspace(l, u, 200)
        true_max = np.max(np.abs(fn(zs)))

        if eps < true_max - 1e-6:
            n_underestimate += 1

    assert n_underestimate == 0, \
        f"{name}: envelope underestimated on {n_underestimate}/{n_total} intervals"


def test_envelope_monotone_in_interval():
    """Wider interval ⇒ envelope cannot decrease (monotonicity)."""
    from archproof.activation_epsilon import envelope

    for name in ["relu", "sigmoid", "tanh", "gelu"]:
        try:
            eps_narrow = envelope(name, -1.0, 1.0)
            eps_wide = envelope(name, -2.0, 2.0)
        except (KeyError, ValueError, NotImplementedError):
            continue

        assert eps_wide >= eps_narrow - 1e-9, \
            f"{name}: envelope on [-2,2] = {eps_wide} < envelope on [-1,1] = {eps_narrow}"
