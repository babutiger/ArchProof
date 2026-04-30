"""Example 1: Certify any ONNX model with ArchProof.

Usage:
    python examples/01_certify_a_model.py <path/to/model.onnx>

Returns the verdict, certificate ε, and admitted-gate count for the
given ONNX file. No GPU required.
"""
import sys
from archproof import verify_model, print_result


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path/to/model.onnx>")
        sys.exit(1)

    onnx_path = sys.argv[1]
    print(f"Verifying: {onnx_path}\n")

    # Run the full hierarchical verifier (admission + IBP + envelope sum +
    # G_i ablation flags). For a small CIFAR-CNN this finishes in <1s; on a
    # 28 GB Mistral-7B export it takes ~10 minutes (CPU-only IBP).
    result = verify_model(
        onnx_path,
        b_clean_ub=0.95,   # B_clean upper bound (0.95 for normalised images)
        tau_adm=0.1,       # Theorem 10 admission threshold
        n_probe=20,        # number of clean probe samples
    )

    print_result(result)

    # The 6-class diagnostic verdict is in result.verdict; the certificate
    # ε is result.total_output_margin. For the 3-class deployment-view
    # verdict (CERTIFIED-POSITIVE / CLASS-NEGATIVE / UNCERTIFIED) use:
    #
    #     from archproof import verify_model_phaseC
    #     r = verify_model_phaseC(onnx_path, tau_sys=1e-3)
    #     print(r.verdict_phaseC, r.epsilon_phaseC)


if __name__ == "__main__":
    main()
