"""Benchmark provenance labels — categorize each ONNX model by origin.

Three categories:
  - natural      : standard pretrained / torchvision / widely-published arch
  - synthetic    : repo-generated custom architectures (clean_* custom blocks)
  - adversarial  : author-crafted stress witnesses for necessity / edge cases

This manifest supports R1 nightmare review Weakness #5 fix — explicit split
between natural clean, synthetic clean, and adversarial witness benchmarks.

Usage:
    from archproof.benchmark_provenance import provenance_of, PROVENANCE
    origin = provenance_of("clean_cbam")   # → "synthetic"
"""

# Standard pretrained / torchvision / widely-published architectures
NATURAL_PATTERNS = [
    # torchvision / imagenet zoo
    "alexnet", "resnet", "vgg", "densenet", "inception", "efficientnet",
    "mobilenet", "mnasnet", "shufflenet", "squeezenet", "regnet", "googlenet",
    "wide_resnet", "resnext", "small_resnet",
    # Standard CIFAR / MNIST baselines
    "cifar_conv", "cifar_mlp", "mlp_3layer", "mlp_deep6", "mlp_tiny",
    "linear_classifier",
    # Standard conv/pool/norm variants (widely-used building blocks)
    "conv_batchnorm", "conv_deep", "conv_dropout", "conv_groupnorm",
    "depthwise_separable", "dilated_conv", "strided_conv",
    "maxpool_heavy", "avgpool_net", "bottleneck_net", "pruned_channel",
    # Named activation baselines
    "gelu_net", "elu_net", "leaky_relu_net",
    # Common gated / attention baselines from textbook architectures
    "gated_attention", "gated_residual", "gated_sigmoid_highway",
    "gated_softmax_attn", "glu_net", "highway_net",
    "attention_pool", "multi_head_concat",
]

# Repo-generated custom architectures — clean_* prefixed modern blocks
# These are NOT in a standard zoo; they were constructed in this repo to
# stress-test the verifier's coverage over modern design patterns.
SYNTHETIC_PATTERNS = [
    "clean_bifpn", "clean_cbam", "clean_channel_shuffle", "clean_convmixer",
    "clean_cross_attention", "clean_denseblock", "clean_eca",
    "clean_firemodule", "clean_gated_delta", "clean_geglu", "clean_ghostnet",
    "clean_gru_recurrent", "clean_hardswish", "clean_hardtanh",
    "clean_instancenorm", "clean_inverted_residual", "clean_lambda",
    "clean_layernorm", "clean_lrn", "clean_mhsa", "clean_mish",
    "clean_non_local", "clean_posffn", "clean_position_droppath",
    "clean_prelu", "clean_residual_mlp", "clean_resnest", "clean_rmsnorm",
    "clean_simple_cnn_a", "clean_simple_cnn_b", "clean_sknet",
    "clean_softmax_router", "clean_spatial_pyramid", "clean_swish",
    "clean_tiny_convnext", "clean_unet_tiny",
    # Custom-designed gate variants
    "film_conditioning", "gated_maxpool_selector", "gru_gate_residual",
    "high_threshold_skip", "rezero_block", "se_block", "zero_init_residual",
    "moe_router", "swiglu_ffn",
]

# Explicit adversarial witnesses — author-crafted stress cases that
# verify GDP theorem necessity. These are NOT natural clean models; their
# only purpose is to exhibit a specific failure mode under a reduced
# version of the GDP checker (e.g., if we disable G1 dormancy, these
# synthetic patterns become false positives).
#
# REVIEWER CAVEAT: These MUST be reported in a separate panel — do NOT
# mix into headline "clean FP rate" statistics.
ADVERSARIAL_PATTERNS = [
    "clean_G1adv_dormant_add",
    "clean_G1adv_dormant_concat",
    "clean_G4adv_ghost_mul",
]


def provenance_of(model_name: str) -> str:
    """Return 'natural', 'synthetic', 'adversarial', or 'unknown' for a model name."""
    name = model_name.replace(".onnx", "").lower()

    # Adversarial witnesses are most specific — check first
    for pat in ADVERSARIAL_PATTERNS:
        if pat.lower() in name:
            return "adversarial"

    # Synthetic custom architectures (clean_* etc.)
    for pat in SYNTHETIC_PATTERNS:
        if pat.lower() in name:
            return "synthetic"

    # Natural / standard architectures
    for pat in NATURAL_PATTERNS:
        if pat.lower() in name:
            return "natural"

    return "unknown"


# Eagerly compute a dict mapping of every known model to its provenance.
# Useful for audit printouts.
def build_provenance_map(model_names) -> dict:
    return {n: provenance_of(n) for n in model_names}
