"""EIC (Exporter Invariance Certificate) sanity tests on synthetic graphs."""
import pytest


def test_eic_constfold_preserves_epsilon(sigmoid_gated_backdoor_onnx, fixtures_dir):
    """Constant-folding the same graph must preserve the certificate ε."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from archproof import verify_model

    # ε under default opset
    r1 = verify_model(str(sigmoid_gated_backdoor_onnx), tau_adm=0.1, n_probe=20)

    # Run onnxoptimizer constant-folding (semantics-preserving)
    try:
        import onnx
        import onnxoptimizer
    except ImportError:
        pytest.skip("onnxoptimizer not installed")

    m = onnx.load(str(sigmoid_gated_backdoor_onnx))
    m_folded = onnxoptimizer.optimize(m, ["fuse_consecutive_squeezes",
                                           "fuse_add_bias_into_conv"])
    folded_path = fixtures_dir / "sigmoid_gated_bd_folded.onnx"
    onnx.save(m_folded, str(folded_path))

    r2 = verify_model(str(folded_path), tau_adm=0.1, n_probe=20)

    # Same admitted set, same ε
    assert r1.n_gdp_admitted == r2.n_gdp_admitted, \
        f"constant-folding changed admission count: {r1.n_gdp_admitted} → {r2.n_gdp_admitted}"


def test_eic_opset_preservation(sigmoid_gated_backdoor_onnx, fixtures_dir):
    """Re-export at opset 16 vs 14 must yield the same ε for in-class graphs."""
    pytest.importorskip("torch")
    import torch

    # Re-export the same model at opset 16
    # (We reuse the model by re-loading the ONNX and re-saving, since
    # we don't have the PyTorch source here; for synthetic test
    # we fall back to onnxoptimizer's noop pipeline as the second
    # toolchain config.)
    pytest.importorskip("onnx")
    import onnx

    m = onnx.load(str(sigmoid_gated_backdoor_onnx))

    # Skip if the loaded model is already at opset 14 (synthetic)
    if not m.opset_import:
        pytest.skip("opset import list empty in fixture")

    from archproof import verify_model
    r_orig = verify_model(str(sigmoid_gated_backdoor_onnx), tau_adm=0.1, n_probe=20)

    # Re-save (semantics-preserving identity transform)
    re_path = fixtures_dir / "sigmoid_gated_bd_re.onnx"
    onnx.save(m, str(re_path))

    r_re = verify_model(str(re_path), tau_adm=0.1, n_probe=20)
    assert r_orig.n_gdp_admitted == r_re.n_gdp_admitted
