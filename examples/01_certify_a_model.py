"""Example 1: Certify any ONNX model with ArchProof.

Usage:
    python examples/01_certify_a_model.py <path/to/model.onnx>

Returns the 3-class verdict, certificate ε, and admitted-gate count for
the given ONNX file. No GPU required.
"""
import sys
from archproof import verify_model_phaseC


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path/to/model.onnx>")
        sys.exit(1)

    onnx_path = sys.argv[1]
    print(f"Verifying: {onnx_path}\n")

    # Run the user-facing 3-class verifier (admission + IBP + envelope
    # sum + trigger-extended interval). For a small CIFAR-CNN this
    # finishes in <1 s; on a 28 GB Mistral-7B export it takes ~10
    # minutes (CPU-only IBP).
    r = verify_model_phaseC(
        onnx_path,
        b_clean_ub=0.95,    # B_clean upper bound (0.95 for normalised images)
        trigger_eta=0.05,   # trigger box width η for D(T) = B_clean ⊕ T_box(η)
        tau_sys=1e-3,       # ε > τ_sys ⇒ CERTIFIED-POSITIVE
    )

    print(f"verdict      : {r.verdict_phaseC}")
    print(f"epsilon      : {r.epsilon_phaseC:.4e}")
    print(f"n_syntactic  : {r.n_syntactic}")
    print(f"n_admitted   : {r.n_admitted_phaseC}")
    print(f"eps_blowup   : {r.epsilon_blowup}")


if __name__ == "__main__":
    main()
