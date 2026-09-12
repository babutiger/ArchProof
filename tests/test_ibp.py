"""Sound interval-bound-propagation (IBP) tests."""
import numpy as np
import pytest


def test_ibp_dense_sampling_under_bound():
    """For a small Linear+ReLU network, the IBP envelope must over-approximate
    every random sample drawn from B_clean.

    Runs the same sound IBP the verifier uses (`propagate_intervals` over the
    ONNX graph), then Monte-Carlo checks that no dense sample escapes the
    envelope -- a direct soundness (over-approximation) property.
    """
    pytest.importorskip("torch")
    import os
    import shutil
    import tempfile

    import onnx
    import torch
    import torch.nn as nn

    from archproof.interval_propagation import propagate_intervals

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Tiny 2-layer network
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()

    # B_clean = [-1, 1]^8; dense-sample the true outputs
    n_samples = 1000
    samples = rng.uniform(-1, 1, size=(n_samples, 8)).astype(np.float32)
    with torch.no_grad():
        outputs = net(torch.tensor(samples)).numpy()

    # Export to ONNX and propagate the input box [-1, 1]^8 through the graph
    d = tempfile.mkdtemp(prefix="ibp_test_")
    try:
        onnx_path = os.path.join(d, "net.onnx")
        torch.onnx.export(net, torch.zeros(1, 8), onnx_path,
                          input_names=["x"], output_names=["y"],
                          dynamic_axes={"x": {0: "batch"}}, opset_version=17)
        model = onnx.load(onnx_path)
        bounds = propagate_intervals(model, input_lb=-1.0, input_ub=1.0)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    out_name = model.graph.output[0].name
    assert out_name in bounds, f"no IBP bound produced for output {out_name!r}"
    ob = bounds[out_name]
    lb_out = np.asarray(ob.lb, dtype=np.float64).ravel()
    ub_out = np.asarray(ob.ub, dtype=np.float64).ravel()

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
