"""G1/G2/G3/G4/T10 admission predicate tests."""
import pytest


def test_relu_gated_backdoor_detected(relu_gated_backdoor_onnx):
    """A strict-dormant ReLU gate should be admitted as an add-DGP candidate."""
    from archproof import verify_model

    result = verify_model(str(relu_gated_backdoor_onnx),
                          tau_adm=0.1, n_probe=20)
    # The injected gate's syntactic pattern (ReLU → Mul) must be detected
    assert result.n_gdp_candidates >= 1


def test_sigmoid_gated_backdoor_admitted(sigmoid_gated_backdoor_onnx):
    """A biased-dormant sigmoid gate should pass T10 admission."""
    from archproof import verify_model

    result = verify_model(str(sigmoid_gated_backdoor_onnx),
                          tau_adm=0.1, n_probe=20)
    assert result.n_gdp_candidates >= 1
    assert result.n_gdp_admitted >= 1, \
        "sigmoid(linear+(-10)) should be empirically dormant"


def test_se_block_rejected_by_admission(se_block_onnx):
    """SE block has a Mul gate but is non-dormant on B_clean → should be
    rejected by T10 (or pass admission and immediately fail dormancy)."""
    from archproof import verify_model

    result = verify_model(str(se_block_onnx), tau_adm=0.1, n_probe=20)
    # SE block is legitimate → should NOT be flagged as backdoor
    # Either: rejected by T10 admission, OR admitted but verdict = BENIGN/UNDECIDED
    assert result.verdict not in ("ε-BOUNDED", "DORMANT") or \
           result.n_gdp_admitted == 0


@pytest.mark.parametrize("disable", ["G1", "G2", "G3", "G4", "T10"])
def test_g_flag_disable(disable, sigmoid_gated_backdoor_onnx):
    """Each G_i flag can be disabled independently — verify no crash."""
    from archproof import verify_model

    flags = {"G1": True, "G2": True, "G3": True, "G4": True, "T10": True}
    flags[disable] = False

    # Should not raise
    result = verify_model(str(sigmoid_gated_backdoor_onnx),
                          gdp_flags=flags, tau_adm=0.1, n_probe=20)
    assert result is not None
