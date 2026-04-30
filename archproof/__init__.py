"""ArchProof — sound output-contribution certificates for
dormant-gate-path backdoors in ONNX models.

Top-level entry points:

    from archproof import verify_model, print_result
    result = verify_model("model.onnx")
    print_result(result)

For the 3-class Phase-C verdict (CERTIFIED-POSITIVE / CLASS-NEGATIVE / UNCERTIFIED):

    from archproof import verify_model_phaseC
    result = verify_model_phaseC("model.onnx")
    print(result.verdict_phaseC, result.epsilon_phaseC)
"""
from .verify import verify_model, print_result, VerificationResult
from .verify_phaseC import verify_model_phaseC, PhaseCResult

__all__ = [
    "verify_model",
    "verify_model_phaseC",
    "print_result",
    "VerificationResult",
    "PhaseCResult",
]
