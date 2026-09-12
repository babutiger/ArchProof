"""ArchProof configuration — FROZEN after v5. Only TAU_DORMANT may be tuned once after Day 6 smoke."""

# G1: Gate operator whitelist (ONNX opset 18 op names)
# Type 1: explicit boolean/switch ops
GATE_OPS_HARD = [
    "Where", "Greater", "Less", "Equal", "Sign",
    "Ge", "Le", "And", "Or", "Xor", "Cast",
    "ArgMax", "ArgMin",
]
GATE_OPS_DISCRETE = ["OneHot"]
GATE_OPS_SOFT = ["Softmax", "Sigmoid"]

# Type 2: Mul-gating (Bober-Irizar pattern: indicator × value)
# Mul is a gate when one input is binary {0,1} indicator
# Detection: check if one Mul input has range ⊆ [0,1] via empirical sampling
GATE_OPS_MUL = ["Mul"]

GATE_WHITELIST = set(GATE_OPS_HARD + GATE_OPS_DISCRETE + GATE_OPS_SOFT + GATE_OPS_MUL)

# G2: Dormancy threshold
TAU_DORMANT = 0.1  # may tune ONCE after Day 6 smoke

# Phase C: α,β-CROWN escalation timeout
ESCALATION_TIMEOUT_SEC = 300

# Phase C: risk score binary search
RISK_SCORE_BISECT_STEPS = 10  # binary search steps for ε*

# Labels
GDP_FREE = "GDP-FREE"
GDP_WITNESS = "GDP-WITNESS"
ESCALATION_TIMEOUT = "ESCALATION-TIMEOUT"
OUT_OF_SCOPE_WARNING = "OUT-OF-SCOPE-WARNING"

# Canonicalization config (T)
ONNX_OPSET = 17  # torch 2.0.1's ReduceMin has a bug under opset 18; use 17
ONNX_DYNAMO = False
ONNX_CONSTANT_FOLDING = False

# B_clean defaults (represents NORMAL input range; trigger is outside this)
B_CLEAN_PIXEL_MIN = 0.0
B_CLEAN_PIXEL_MAX = 0.95  # trigger at 0.999 is outside B_clean
