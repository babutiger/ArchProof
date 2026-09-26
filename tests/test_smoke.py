"""Smoke test — verify the package imports + key entry points are reachable."""
import importlib

import pytest


def test_package_imports():
    """archproof must import cleanly."""
    import archproof
    assert hasattr(archproof, "verify_model")
    assert hasattr(archproof, "verify_model_phaseC")
    assert hasattr(archproof, "print_result")


def test_all_modules_parse():
    """Every module under archproof/ must parse without ImportError."""
    submodules = [
        "archproof.verify",
        "archproof.verify_phaseC",
        "archproof.acpc",
        "archproof.mgrs",
        "archproof.escalate",
        "archproof.gate_admission",
        "archproof.gate_witness_oracle",
        "archproof.activation_epsilon",
        "archproof.chain_sensitivity",
        "archproof.interval_propagation",
        "archproof.llm_gate_rescue",
        "archproof.handcrafted_gdp",
        "archproof.g1_g4_adversaries",
        "archproof.adaptive_graph_obfuscation",
        "archproof.sound_arithmetic",
        "archproof.mgrs_llm_onnx_surgery",
    ]
    for mod_name in submodules:
        importlib.import_module(mod_name)


def test_verify_clean_cnn(clean_cnn_onnx):
    """A clean CNN with no Mul gates must yield a clean verdict."""
    from archproof import verify_model

    result = verify_model(str(clean_cnn_onnx), tau_adm=0.1, n_probe=8)
    # No Mul gates → 0 syntactic candidates → clean verdict
    assert result.n_gdp_candidates == 0
    assert result.verdict in ("GDP-FREE", "DORMANT", "BENIGN")


def test_verify_phaseC_clean_cnn(clean_cnn_onnx):
    """Phase-C verifier on a clean CNN: CLASS-NEGATIVE."""
    from archproof import verify_model_phaseC

    result = verify_model_phaseC(str(clean_cnn_onnx), tau_sys=1e-3)
    assert result.verdict_phaseC.endswith("CLASS-NEGATIVE")
    assert result.epsilon_phaseC == 0.0
