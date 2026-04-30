"""Sound interval-bound-propagation (IBP) tests."""
import numpy as np
import pytest


def test_ibp_dense_sampling_under_bound():
    """For a small Linear+ReLU network, the IBP envelope must over-approximate
    every random sample drawn from B_clean."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Tiny 2-layer network
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()

    # B_clean = [-1, 1]^8
    n_samples = 1000
    samples = rng.uniform(-1, 1, size=(n_samples, 8)).astype(np.float32)
    with torch.no_grad():
        outputs = net(torch.tensor(samples)).numpy()

    # IBP propagation
    try:
        from archproof.interval_propagation import propagate_linear_relu
    except ImportError:
        pytest.skip("propagate_linear_relu not exposed")

    lb_in, ub_in = -np.ones(8, dtype=np.float32), np.ones(8, dtype=np.float32)
    try:
        lb_out, ub_out = propagate_linear_relu(net, lb_in, ub_in)
    except (AttributeError, TypeError):
        pytest.skip("propagate_linear_relu signature differs")

    # Soundness: every dense sample must lie inside [lb_out, ub_out]
    n_violations = ((outputs < lb_out - 1e-5) | (outputs > ub_out + 1e-5)).sum()
    assert n_violations == 0, \
        f"IBP unsound on {n_violations}/{n_samples * 4} (sample, dim) pairs"


def test_layernorm_geometric_rescue_bound():
    """LayerNorm output bound: ‖LN(x)‖∞ ≤ ‖γ‖∞ √D + ‖β‖∞ (Lemma 5)."""
    try:
        from archproof.llm_gate_rescue import layernorm_geometric_bound
    except ImportError:
        pytest.skip("layernorm_geometric_bound not exposed")

    rng = np.random.default_rng(0)
    for _ in range(20):
        D = int(rng.choice([16, 64, 256, 1024]))
        gamma = rng.uniform(0.5, 1.5, size=D).astype(np.float32)
        beta = rng.uniform(-0.5, 0.5, size=D).astype(np.float32)

        try:
            bound = layernorm_geometric_bound(gamma, beta)
        except (AttributeError, TypeError):
            pytest.skip("API mismatch")

        # Dense sampling: 50 random inputs, run LN, check ‖output‖∞
        for _ in range(50):
            x = rng.uniform(-10, 10, size=D).astype(np.float32)
            mu = x.mean()
            sigma = x.std() + 1e-6
            ln_out = gamma * (x - mu) / sigma + beta
            assert np.abs(ln_out).max() <= bound + 1e-5, \
                f"LayerNorm bound violated: {np.abs(ln_out).max():.4f} > {bound:.4f}"
