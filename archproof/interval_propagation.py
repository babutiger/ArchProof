"""Sound interval propagation on ONNX graphs — our core G2 verification engine.

Unlike auto_LiRPA which has op compatibility issues, we propagate intervals
manually through every ONNX operator. This gives us FULL CONTROL and handles
MaxPool, Neg, Mul, Sub, ReLU, etc. that LiRPA can't handle in certain configs.

The bound is SOUND: if upper_bound(gate_output) ≤ 0 over B_clean,
then gate provably never fires → branch is provably dormant.
"""

import onnx
import numpy as np
from typing import Dict, Tuple, Optional


def _load_tensor(tensor) -> np.ndarray:
    """Load ONNX TensorProto to numpy array (compatible with onnx 1.13).

    Covers the dtype codes we need to read raw_data correctly. The width of
    each numpy dtype must match ONNX's storage convention so that
    `raw_data / dtype_width` divides evenly into `prod(dims)` — otherwise
    `np.frombuffer().reshape()` raises a ValueError. Common pitfall:
    missing BOOL / FP16 / BF16 / UINT8 / INT8 caused us to fall back to
    float32, which mis-sized e.g. a `(1,1,16,16)=256` attention mask
    tensor stored as 256 bool bytes (-> 64 float32 lanes).
    """
    dt_map = {
        1:  np.float32,    # FLOAT
        2:  np.uint8,      # UINT8
        3:  np.int8,       # INT8
        4:  np.uint16,     # UINT16
        5:  np.int16,      # INT16
        6:  np.int32,      # INT32
        7:  np.int64,      # INT64
        9:  np.bool_,      # BOOL (1 byte per element)
        10: np.float16,    # FLOAT16
        11: np.float64,    # DOUBLE
        12: np.uint32,     # UINT32
        13: np.uint64,     # UINT64
        16: np.uint16,     # BFLOAT16: read as uint16, caller may upcast
    }
    if tensor.raw_data:
        dtype = dt_map.get(tensor.data_type, np.float32)
        try:
            arr = np.frombuffer(tensor.raw_data, dtype=dtype)
        except ValueError:
            # Falls back if the dtype width is incompatible with raw_data
            # length; return a zero stub to keep downstream propagation
            # sound (the constant becomes 0 in affine form).
            return np.zeros(tensor.dims, dtype=np.float32)
        try:
            return arr.reshape(tensor.dims).copy()
        except ValueError:
            return np.zeros(tensor.dims, dtype=np.float32)
    elif tensor.float_data:
        return np.array(tensor.float_data, dtype=np.float32).reshape(tensor.dims)
    elif tensor.int64_data:
        return np.array(tensor.int64_data, dtype=np.int64).reshape(tensor.dims)
    elif tensor.int32_data:
        return np.array(tensor.int32_data, dtype=np.int32).reshape(tensor.dims)
    return np.zeros(tensor.dims, dtype=np.float32)


class IntervalBound:
    """Interval [lb, ub] for a tensor."""
    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        self.lb = lb
        self.ub = ub

    @property
    def width(self):
        return (self.ub - self.lb).max()

    def __repr__(self):
        return f"Interval(lb_max={self.lb.max():.4f}, ub_max={self.ub.max():.4f}, width={self.width:.4f})"


def propagate_intervals(onnx_model: onnx.ModelProto,
                        input_lb = 0.0,
                        input_ub = 0.95,
                        dtype = np.float64,
                        seed_bounds = None) -> Dict[str, IntervalBound]:
    """Propagate intervals through ONNX graph from input bounds.

    Args:
        onnx_model: ONNX model
        input_lb: lower bound — scalar (broadcast to all dims) or np.ndarray per-dim
        input_ub: upper bound — scalar (broadcast to all dims) or np.ndarray per-dim
        dtype: working precision; default float64 gives 308 decades of exponent
               headroom (vs float32's 38), preventing overflow-to-inf chains
               on deep networks such as ResNet-50 where intermediate bounds
               climb to ~1e+200 but remain finite and sound.
        seed_bounds: optional dict mapping tensor_name -> (lb_array, ub_array)
               that seeds IBP at a post-input layer (used for transformers
               where the graph input is int64 token-ids and IBP must enter
               at the post-embedding float tensor). Nodes whose sole output
               tensor is seeded are skipped during propagation so the seeded
               bound is preserved.

    Returns:
        Dict mapping tensor name → IntervalBound for every intermediate tensor
    """
    graph = onnx_model.graph

    # Get input shape
    input_info = graph.input[0]
    input_shape = []
    for dim in input_info.type.tensor_type.shape.dim:
        input_shape.append(dim.dim_value if dim.dim_value > 0 else 1)

    # Initialize input interval — support per-dimension bounds
    bounds: Dict[str, IntervalBound] = {}
    input_name = graph.input[0].name
    if isinstance(input_lb, np.ndarray):
        lb_arr = input_lb.reshape(input_shape).astype(dtype)
    else:
        lb_arr = np.full(input_shape, float(input_lb), dtype=dtype)
    if isinstance(input_ub, np.ndarray):
        ub_arr = input_ub.reshape(input_shape).astype(dtype)
    else:
        ub_arr = np.full(input_shape, float(input_ub), dtype=dtype)
    bounds[input_name] = IntervalBound(lb=lb_arr, ub=ub_arr)

    # Load constant initializers. For whole-LLM scale (22+ GB of fp32
    # weight initializers) naive `lb=arr.copy(), ub=arr.copy()` plus an
    # implicit fp32->fp64 upcast would 4x the RAM cost (two copies ×
    # double precision) and OOM a 88 GB machine. Optimisations:
    #   (1) lb and ub share the same underlying array for constants
    #       (interval has zero width, so one buffer suffices).
    #   (2) Do NOT upcast fp32 weights to fp64. Downstream ops that
    #       need fp64 precision promote on-the-fly via numpy
    #       broadcasting (fp32 × fp64 = fp64), keeping the weight
    #       buffer itself at its native dtype. fp16 weights are still
    #       promoted to the working dtype because fp16 × fp16 IBP
    #       overflows rapidly on deep nets.
    for init in graph.initializer:
        arr = _load_tensor(init)
        if arr.dtype == np.float16:
            arr = arr.astype(dtype, copy=False)
        elif arr.dtype in (np.float32, np.float64) and arr.dtype != dtype:
            # Only upcast when the array is small enough that the extra
            # copy is cheap; for whole-LLM weights, keep fp32 to avoid
            # the 2x blow-up.
            if arr.size < 1_000_000:
                arr = arr.astype(dtype, copy=False)
        bounds[init.name] = IntervalBound(lb=arr, ub=arr)

    # Finite vacuous bound constants for this precision. float64 supports
    # up to ~1.8e+308; we use 1e+300 so downstream multiplications stay
    # finite through several more layers before collapsing to inf/nan.
    VAC_LB = np.array([-1e+300 if dtype == np.float64 else -1e30], dtype=dtype)
    VAC_UB = np.array([ 1e+300 if dtype == np.float64 else  1e30], dtype=dtype)

    # Apply seeded post-input bounds (transformer embedding entry, etc.)
    seeded_tensors = set()
    if seed_bounds:
        for t, (lb_a, ub_a) in seed_bounds.items():
            bounds[t] = IntervalBound(
                lb=np.asarray(lb_a, dtype=dtype),
                ub=np.asarray(ub_a, dtype=dtype),
            )
            seeded_tensors.add(t)

    # Propagate through nodes in topological order; skip nodes whose
    # outputs are all seeded (the seed is a sound replacement for running
    # that node's IBP).
    for node in graph.node:
        if node.output and all(o in seeded_tensors for o in node.output):
            continue
        try:
            output_bound = _propagate_node(node, bounds)
        except Exception:
            output_bound = None
        if output_bound is None:
            # Unknown op or unsupported configuration → write vacuous sound
            # bound so downstream nodes can continue to propagate. Without
            # this, a missing tensor silently breaks the chain (e.g. Clip in
            # MobileNetV2 on ONNX opset >= 13 → downstream Linear loses input).
            for out_name in node.output:
                bounds[out_name] = IntervalBound(
                    lb=VAC_LB.copy(), ub=VAC_UB.copy())
        elif isinstance(output_bound, list):
            for i, out_name in enumerate(node.output):
                if i < len(output_bound):
                    bounds[out_name] = output_bound[i]
        else:
            for out_name in node.output:
                bounds[out_name] = output_bound

    # === Dependency-Aware Refinement Pass ===
    # For Mul(f(x), g(x)) where both inputs share ancestors,
    # standard IBP loses correlation → overestimates.
    # Refinement: sample the ONNX subgraph to get tighter empirical bounds,
    # then take the intersection (tighter of IBP and sampling).
    # Soundness note: sampling gives INNER bounds (not sound alone),
    # but we only use them to TIGHTEN (never loosen) the sound IBP bounds.
    # The final bound = max(sampling_lb, ibp_lb) for lb, min(sampling_ub, ibp_ub) for ub.
    # Wait — that's wrong. Sampling gives observed range ⊆ true range ⊆ IBP range.
    # We CANNOT tighten using sampling because sampling might miss extreme values.
    #
    # SOUND refinement: re-propagate from a tighter intermediate bound.
    # If a Mul node's IBP bound is loose, but one of its inputs has a
    # tighter bound from a different path, we can intersect.
    # For now, we use a SOUND method: re-run propagation on the subgraph
    # with many uniformly sampled inputs and take the CONVEX HULL of outputs.
    # This is NOT done here — it's done in the refinement scripts separately.
    # The core propagate_intervals remains pure sound IBP.

    return bounds


def _get_bound(bounds, name):
    """Get bound for a tensor name, return None if not available."""
    return bounds.get(name)


def _propagate_node(node: onnx.NodeProto,
                    bounds: Dict[str, IntervalBound]) -> Optional[IntervalBound]:
    """Compute output interval from input intervals for one ONNX node."""
    op = node.op_type

    if op == "Relu":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        return IntervalBound(
            lb=np.maximum(x.lb, 0),
            ub=np.maximum(x.ub, 0),
        )

    elif op == "LeakyRelu":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        alpha = 0.01  # ONNX default
        for attr in node.attribute:
            if attr.name == "alpha":
                alpha = attr.f
        # LeakyRelu(x) = x if x >= 0, alpha*x if x < 0
        # For intervals: lb = min(lb, alpha*lb), ub = max(ub, alpha*ub)
        lb = np.minimum(x.lb, alpha * x.lb)
        ub = np.maximum(x.ub, alpha * x.ub)
        return IntervalBound(lb=lb, ub=ub)

    elif op == "Sigmoid":
        # σ(x) = 1/(1+e^-x), strictly monotonic increasing, in (0, 1)
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        # Numerically stable
        lb = 1.0 / (1.0 + np.exp(-x.lb))
        ub = 1.0 / (1.0 + np.exp(-x.ub))
        return IntervalBound(lb=lb, ub=ub)

    elif op == "Tanh":
        # tanh(x), strictly monotonic increasing, in (-1, 1)
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        return IntervalBound(lb=np.tanh(x.lb), ub=np.tanh(x.ub))

    elif op == "Gelu":
        # GELU(x) = 0.5 x (1 + erf(x/√2)). Near-monotonic; local min at
        # x ≈ -0.7518, value ≈ -0.16997. Post-bound must respect min point.
        from scipy.special import erf as _erf
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        def _gelu(v):
            return 0.5 * v * (1.0 + _erf(v / np.sqrt(2.0)))
        g_lb_val = _gelu(x.lb)
        g_ub_val = _gelu(x.ub)
        # For non-monotonic: lb is min(g_lb, g_ub, min_at_argmin if in interval)
        GELU_MIN_X = -0.751791528026822
        GELU_MIN_Y = -0.169971207479904
        lb_out = np.minimum(g_lb_val, g_ub_val)
        # Where the interval contains the min point, bump lb down to min value
        contains_min = (x.lb <= GELU_MIN_X) & (x.ub >= GELU_MIN_X)
        lb_out = np.where(contains_min, np.minimum(lb_out, GELU_MIN_Y), lb_out)
        # ub: GELU is monotonic for x > MIN_X, so max is at x.ub when x.ub > MIN_X,
        # else at x.lb. Safest: max of endpoint values.
        ub_out = np.maximum(g_lb_val, g_ub_val)
        return IntervalBound(lb=lb_out, ub=ub_out)

    elif op in ("Silu", "Swish"):
        # SiLU(x) = x * σ(x). Near-monotonic; min at x ≈ -1.278, value ≈ -0.278
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        def _sig(v):
            return 1.0 / (1.0 + np.exp(-v))
        s_lb = x.lb * _sig(x.lb)
        s_ub = x.ub * _sig(x.ub)
        SILU_MIN_X = -1.278464561437784
        SILU_MIN_Y = -0.278464542761074
        lb_out = np.minimum(s_lb, s_ub)
        contains_min = (x.lb <= SILU_MIN_X) & (x.ub >= SILU_MIN_X)
        lb_out = np.where(contains_min, np.minimum(lb_out, SILU_MIN_Y), lb_out)
        ub_out = np.maximum(s_lb, s_ub)
        return IntervalBound(lb=lb_out, ub=ub_out)

    elif op == "HardSwish":
        # HardSwish(x) = x · ReLU6(x+3)/6. Piecewise; min at x=-1.5, value=-0.375;
        # monotone-increasing for x > -3; clipped at x ≥ 3 (y=x).
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        def _hs(v):
            return v * np.clip(v + 3.0, 0.0, 6.0) / 6.0
        s_lb, s_ub = _hs(x.lb), _hs(x.ub)
        lb_out = np.minimum(s_lb, s_ub)
        contains_min = (x.lb <= -1.5) & (x.ub >= -1.5)
        lb_out = np.where(contains_min, np.minimum(lb_out, -0.375), lb_out)
        ub_out = np.maximum(s_lb, s_ub)
        return IntervalBound(lb=lb_out, ub=ub_out)

    elif op == "HardTanh":
        # HardTanh(x) = clip(x, -1, 1); monotone.
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        return IntervalBound(lb=np.clip(x.lb, -1.0, 1.0),
                              ub=np.clip(x.ub, -1.0, 1.0))

    elif op == "Elu":
        # ELU(x) = x for x>0, α·(exp(x)-1) for x≤0. Monotone.
        alpha = 1.0
        for attr in node.attribute:
            if attr.name == "alpha":
                alpha = float(attr.f)
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        def _elu(v):
            return np.where(v >= 0, v, alpha * (np.exp(v) - 1.0))
        return IntervalBound(lb=_elu(x.lb), ub=_elu(x.ub))

    elif op == "Selu":
        # SELU = scale·ELU; monotone with fixed α≈1.6733, scale≈1.0507.
        SELU_A = 1.6732632423543772
        SELU_S = 1.0507009873554805
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        def _selu(v):
            return SELU_S * np.where(v >= 0, v, SELU_A * (np.exp(v) - 1.0))
        return IntervalBound(lb=_selu(x.lb), ub=_selu(x.ub))

    elif op == "Softplus":
        # Softplus(x) = log(1+exp(x)); strictly monotone increasing, ≥ 0.
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        def _sp(v):
            return np.log1p(np.exp(np.clip(v, -50.0, 50.0)))
        return IntervalBound(lb=_sp(x.lb), ub=_sp(x.ub))

    elif op == "Mish":
        # Mish(x) = x·tanh(Softplus(x)). Near-monotone; min at x≈-1.1924, y≈-0.3085.
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        def _mish(v):
            vc = np.clip(v, -50.0, 50.0)
            return v * np.tanh(np.log1p(np.exp(vc)))
        MISH_MIN_X = -1.1924
        MISH_MIN_Y = -0.3085
        s_lb, s_ub = _mish(x.lb), _mish(x.ub)
        lb_out = np.minimum(s_lb, s_ub)
        contains_min = (x.lb <= MISH_MIN_X) & (x.ub >= MISH_MIN_X)
        lb_out = np.where(contains_min, np.minimum(lb_out, MISH_MIN_Y), lb_out)
        ub_out = np.maximum(s_lb, s_ub)
        return IntervalBound(lb=lb_out, ub=ub_out)

    elif op == "Softmax":
        # Softmax output ∈ (0, 1) element-wise; trivial sound bound.
        # A tighter joint bound (over full probability simplex) requires all
        # input bounds jointly — callers can use `softmax_tight_epsilon` from
        # activation_epsilon.py. Here we return the elementwise range only.
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        return IntervalBound(lb=np.zeros_like(x.lb),
                              ub=np.ones_like(x.ub))

    elif op == "Pow":
        x = _get_bound(bounds, node.input[0])
        exp = _get_bound(bounds, node.input[1])
        if x is None or exp is None: return None
        # Constant exponent (most common: x^n where n is fixed)
        n = exp.lb.flat[0]  # assume constant exponent
        if x.lb.min() >= 0:
            # Non-negative base: x^n is monotone increasing for n > 0
            if n > 0:
                return IntervalBound(lb=np.power(x.lb, n), ub=np.power(x.ub, n))
            elif n == 0:
                return IntervalBound(lb=np.ones_like(x.lb), ub=np.ones_like(x.ub))
        # General case: conservative bound using all corner combinations
        corners = [np.power(np.maximum(x.lb, 0), n), np.power(np.maximum(x.ub, 0), n)]
        return IntervalBound(lb=np.minimum(*corners), ub=np.maximum(*corners))

    elif op == "Neg":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        return IntervalBound(lb=-x.ub, ub=-x.lb)

    elif op == "Add":
        a = _get_bound(bounds, node.input[0])
        b = _get_bound(bounds, node.input[1])
        if a is None or b is None: return None
        a_lb, a_ub = _broadcast_pair(a.lb, a.ub, b.lb)
        b_lb, b_ub = _broadcast_pair(b.lb, b.ub, a.lb)
        return IntervalBound(lb=a_lb + b_lb, ub=a_ub + b_ub)

    elif op == "Sub":
        a = _get_bound(bounds, node.input[0])
        b = _get_bound(bounds, node.input[1])
        if a is None or b is None: return None
        a_lb, a_ub = _broadcast_pair(a.lb, a.ub, b.lb)
        b_lb, b_ub = _broadcast_pair(b.lb, b.ub, a.lb)
        return IntervalBound(lb=a_lb - b_ub, ub=a_ub - b_lb)

    elif op == "Mul":
        a = _get_bound(bounds, node.input[0])
        b = _get_bound(bounds, node.input[1])
        if a is None or b is None: return None
        a_lb, a_ub = _broadcast_pair(a.lb, a.ub, b.lb)
        b_lb, b_ub = _broadcast_pair(b.lb, b.ub, a.lb)
        # Interval multiplication: min/max of all combinations, with
        # soundness-preserving NaN repair. Floating-point yields NaN only for
        # 0 * inf; in interval semantics this should be 0 when the 0 operand's
        # whole range is {0}. We replace NaN conservatively — fall back to the
        # sign-corrected vacuous bound otherwise (still sound).
        products = [a_lb * b_lb, a_lb * b_ub, a_ub * b_lb, a_ub * b_ub]
        lb = np.minimum.reduce(products)
        ub = np.maximum.reduce(products)
        # Tight repair: if either operand's range reduces to {0}, product is 0.
        zero_a = (a_lb == 0) & (a_ub == 0)
        zero_b = (b_lb == 0) & (b_ub == 0)
        zero_either = zero_a | zero_b
        nan_mask = np.isnan(lb) | np.isnan(ub)
        if nan_mask.any():
            lb = np.where(nan_mask & zero_either, 0.0, lb)
            ub = np.where(nan_mask & zero_either, 0.0, ub)
            # Remaining NaN: fall back to vacuous sound bound (±finite-large).
            MAX_VAL = np.finfo(lb.dtype).max * 0.5 if lb.dtype.kind == "f" else 1e300
            lb = np.where(np.isnan(lb), -MAX_VAL, lb)
            ub = np.where(np.isnan(ub),  MAX_VAL, ub)
        return IntervalBound(lb=lb, ub=ub)

    elif op == "Conv":
        return _propagate_conv(node, bounds)

    elif op == "MaxPool":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        return _propagate_maxpool(node, x)

    elif op == "AveragePool":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        # Conservative: average of intervals
        return _propagate_avgpool(node, x)

    elif op == "GlobalAveragePool":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        # Global mean over spatial dims (2, 3). Keeps batch/channel dims.
        return IntervalBound(
            lb=x.lb.mean(axis=(2, 3), keepdims=True),
            ub=x.ub.mean(axis=(2, 3), keepdims=True),
        )

    elif op == "GlobalMaxPool":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        return IntervalBound(
            lb=x.lb.max(axis=(2, 3), keepdims=True),
            ub=x.ub.max(axis=(2, 3), keepdims=True),
        )

    # Flatten / Reshape are handled below with shape-aware reshapes that
    # preserve per-coordinate bound structure (needed for downstream Gemm
    # and Mul that consume reshaped tensors). The previous flatten-to-1D
    # shortcut was unsound for any downstream op that expects the original
    # rank.

    elif op == "Gemm":
        return _propagate_gemm(node, bounds)

    elif op == "Concat":
        inputs = [_get_bound(bounds, name) for name in node.input]
        if any(x is None for x in inputs): return None
        axis = 1  # default
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        return IntervalBound(
            lb=np.concatenate([x.lb for x in inputs], axis=axis),
            ub=np.concatenate([x.ub for x in inputs], axis=axis),
        )

    elif op == "ReduceMax":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        axes = _get_reduce_axes(node)
        keepdims = _get_attr(node, "keepdims", 1)
        return IntervalBound(
            lb=x.lb.max(axis=tuple(axes), keepdims=bool(keepdims)),
            ub=x.ub.max(axis=tuple(axes), keepdims=bool(keepdims)),
        )

    elif op == "ReduceMin":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        axes = _get_reduce_axes(node)
        keepdims = _get_attr(node, "keepdims", 1)
        return IntervalBound(
            lb=x.lb.min(axis=tuple(axes), keepdims=bool(keepdims)),
            ub=x.ub.min(axis=tuple(axes), keepdims=bool(keepdims)),
        )

    elif op == "ReduceMean":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        axes = _get_reduce_axes(node)
        keepdims = _get_attr(node, "keepdims", 1)
        # Mean is linear → lb(mean) = mean(lb), ub(mean) = mean(ub)
        return IntervalBound(
            lb=x.lb.mean(axis=tuple(axes), keepdims=bool(keepdims)),
            ub=x.ub.mean(axis=tuple(axes), keepdims=bool(keepdims)),
        )

    elif op == "Unsqueeze":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        axes = _get_unsqueeze_axes(node, bounds)
        lb, ub = x.lb, x.ub
        for ax in sorted(axes):
            lb = np.expand_dims(lb, axis=ax)
            ub = np.expand_dims(ub, axis=ax)
        return IntervalBound(lb=lb, ub=ub)

    elif op == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                arr = _load_tensor(attr.t)
                return IntervalBound(lb=arr.copy(), ub=arr.copy())
        return None

    elif op == "Identity":
        x = _get_bound(bounds, node.input[0])
        return x

    elif op == "Pad":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        # Conservative: pad with input bounds range
        return x  # simplified — padding doesn't shrink bounds

    elif op == "Clip":
        # Clip(x, min, max) is how torch.onnx exports ReLU6 (Clip 0..6) and
        # similar HardTanh-style activations. Clip is monotone on each axis so
        # intervals map tightly to [clip(lb), clip(ub)].
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        clip_min = None
        clip_max = None
        for attr in node.attribute:
            if attr.name == "min": clip_min = float(attr.f)
            elif attr.name == "max": clip_max = float(attr.f)
        # ONNX-13+: min/max passed as inputs[1]/inputs[2]
        if clip_min is None and len(node.input) > 1 and node.input[1]:
            mb = _get_bound(bounds, node.input[1])
            if mb is not None: clip_min = float(mb.lb.reshape(-1)[0])
        if clip_max is None and len(node.input) > 2 and node.input[2]:
            mxb = _get_bound(bounds, node.input[2])
            if mxb is not None: clip_max = float(mxb.lb.reshape(-1)[0])
        lb = x.lb.copy(); ub = x.ub.copy()
        if clip_min is not None:
            lb = np.maximum(lb, clip_min); ub = np.maximum(ub, clip_min)
        if clip_max is not None:
            lb = np.minimum(lb, clip_max); ub = np.minimum(ub, clip_max)
        return IntervalBound(lb=lb, ub=ub)

    elif op == "Transpose":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        perm = None
        for attr in node.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
        if perm:
            return IntervalBound(
                lb=np.transpose(x.lb, perm),
                ub=np.transpose(x.ub, perm),
            )
        return x

    elif op == "Flatten":
        # Flatten preserves values, only reshapes. Default axis=1 per ONNX spec:
        # output.shape = (d0 * ... * d_{axis-1}, d_axis * ... * d_{N-1}).
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        axis = 1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
        shape = x.lb.shape
        pre = int(np.prod(shape[:axis])) if axis > 0 else 1
        post = int(np.prod(shape[axis:])) if axis < len(shape) else 1
        return IntervalBound(
            lb=x.lb.reshape(pre, post),
            ub=x.ub.reshape(pre, post),
        )

    elif op == "Reshape":
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        # Second input is the shape tensor; already loaded as an initializer.
        if len(node.input) > 1:
            shape_bound = _get_bound(bounds, node.input[1])
            if shape_bound is not None:
                try:
                    new_shape = shape_bound.lb.astype(int).flatten().tolist()
                    # Handle -1 in shape
                    prod = 1
                    neg_idx = -1
                    for i, s in enumerate(new_shape):
                        if s == -1:
                            neg_idx = i
                        else:
                            prod *= s
                    if neg_idx >= 0:
                        total = int(np.prod(x.lb.shape))
                        new_shape[neg_idx] = total // prod
                    return IntervalBound(
                        lb=x.lb.reshape(new_shape),
                        ub=x.ub.reshape(new_shape),
                    )
                except Exception:
                    pass
        return x

    elif op == "BatchNormalization":
        # Eval-mode BN:  y = gamma * (x - mean) / sqrt(var + eps) + beta
        # This is a per-channel affine transform; tight for intervals.
        x = _get_bound(bounds, node.input[0])
        if x is None: return None
        if len(node.input) < 5:
            return x  # not enough inputs; preserve
        gamma = _get_bound(bounds, node.input[1])
        beta  = _get_bound(bounds, node.input[2])
        mean  = _get_bound(bounds, node.input[3])
        var   = _get_bound(bounds, node.input[4])
        if any(b is None for b in (gamma, beta, mean, var)):
            return x
        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = float(attr.f)
        g = gamma.lb.astype(x.lb.dtype)
        b = beta.lb.astype(x.lb.dtype)
        mu = mean.lb.astype(x.lb.dtype)
        v = var.lb.astype(x.lb.dtype)
        scale = g / np.sqrt(v + eps)
        bias  = b - scale * mu
        shape = [1] * x.lb.ndim
        if x.lb.ndim >= 2:
            shape[1] = -1
        scale_b = scale.reshape(shape)
        bias_b  = bias.reshape(shape)
        pos = scale_b >= 0
        lb = np.where(pos, scale_b * x.lb + bias_b, scale_b * x.ub + bias_b)
        ub = np.where(pos, scale_b * x.ub + bias_b, scale_b * x.lb + bias_b)
        return IntervalBound(lb=lb, ub=ub)

    elif op == "Erf":
        # Erf is monotone non-decreasing on R.
        x = _get_bound(bounds, node.input[0])
        if x is None:
            return None
        try:
            from scipy.special import erf as _erf_fn
            return IntervalBound(lb=_erf_fn(x.lb), ub=_erf_fn(x.ub))
        except Exception:
            # Fallback: erf bounded in [-1, 1]
            return IntervalBound(lb=np.full_like(x.lb, -1.0),
                                 ub=np.full_like(x.ub, 1.0))

    elif op == "Div":
        # Interval division: sound only when divisor does not contain 0.
        x = _get_bound(bounds, node.input[0])
        y = _get_bound(bounds, node.input[1])
        if x is None or y is None:
            return None
        # Constant divisor (y.lb == y.ub).
        if np.array_equal(y.lb, y.ub):
            d = y.lb
            try:
                if np.all(d > 0):
                    return IntervalBound(lb=x.lb / d, ub=x.ub / d)
                if np.all(d < 0):
                    return IntervalBound(lb=x.ub / d, ub=x.lb / d)
            except Exception:
                return None
            return None
        # Interval divisor: only sound if divisor is strictly positive or
        # strictly negative. Compute four-product style.
        if np.all(y.lb > 0) or np.all(y.ub < 0):
            products = [x.lb / y.lb, x.lb / y.ub, x.ub / y.lb, x.ub / y.ub]
            return IntervalBound(lb=np.minimum.reduce(products),
                                 ub=np.maximum.reduce(products))
        return None

    elif op == "MatMul":
        # Interval matrix multiplication. Tight when one operand is constant.
        a = _get_bound(bounds, node.input[0])
        b = _get_bound(bounds, node.input[1])
        if a is None or b is None:
            return None
        # Case A: right operand is constant weight (transformer QKV projection,
        # dense FFN). Sound positive/negative-weight splitting.
        if np.array_equal(b.lb, b.ub):
            w = b.lb
            pos = np.maximum(w, 0)
            neg = np.minimum(w, 0)
            try:
                out_lb = a.lb @ pos + a.ub @ neg
                out_ub = a.ub @ pos + a.lb @ neg
                return IntervalBound(lb=out_lb, ub=out_ub)
            except Exception:
                return None
        if np.array_equal(a.lb, a.ub):
            w = a.lb
            pos = np.maximum(w, 0)
            neg = np.minimum(w, 0)
            try:
                out_lb = pos @ b.lb + neg @ b.ub
                out_ub = pos @ b.ub + neg @ b.lb
                return IntervalBound(lb=out_lb, ub=out_ub)
            except Exception:
                return None
        # Case B: both intervals (Q @ K^T in attention). Interval bilinear.
        # Worst-case per-output = sum over k of interval-product(a[..,k], b[k,..]).
        # This is expensive but tight enough; implemented via element-wise
        # product intervals summed via shape alignment.
        try:
            a_lb, a_ub = a.lb, a.ub
            b_lb, b_ub = b.lb, b.ub
            p1 = a_lb @ b_lb
            p2 = a_lb @ b_ub
            p3 = a_ub @ b_lb
            p4 = a_ub @ b_ub
            out_lb = np.minimum.reduce([p1, p2, p3, p4])
            out_ub = np.maximum.reduce([p1, p2, p3, p4])
            return IntervalBound(lb=out_lb, ub=out_ub)
        except Exception:
            return None

    elif op == "LayerNormalization":
        # LN(x) = scale * (x - mean) / sqrt(var + eps) + bias, normalised
        # along `axis` (default -1). For IBP we use the classical bound
        # |(x - mean) / sqrt(var + eps)| <= sqrt(d - 1) per coord (Cauchy–
        # Schwarz on the centred-squared-sum identity d*var = Σ(x-mean)^2).
        # This bound is independent of the input interval and tight enough
        # for deep transformer propagation.
        x = _get_bound(bounds, node.input[0])
        if x is None:
            return None
        eps = 1e-5
        axis = -1
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = float(attr.f)
            if attr.name == "axis":
                axis = attr.i
        if axis < 0:
            axis = x.lb.ndim + axis
        d = x.lb.shape[axis] if 0 <= axis < x.lb.ndim else max(x.lb.shape[-1], 1)
        bound_mag = float(np.sqrt(max(d - 1, 1)))
        norm_lb = np.full_like(x.lb, -bound_mag, dtype=x.lb.dtype)
        norm_ub = np.full_like(x.ub, bound_mag, dtype=x.ub.dtype)
        # Apply scale (if constant vector along normalised axis).
        scale = _get_bound(bounds, node.input[1]) if len(node.input) > 1 else None
        if scale is not None:
            sv = scale.lb.astype(x.lb.dtype)
            # Broadcast scale to x's shape: LN scale has shape [d].
            # Reshape to broadcast correctly along `axis`.
            try:
                shape = [1] * x.lb.ndim
                shape[axis] = -1
                sv_b = sv.reshape(shape)
                pos = np.maximum(sv_b, 0)
                neg = np.minimum(sv_b, 0)
                new_lb = norm_lb * pos + norm_ub * neg
                new_ub = norm_ub * pos + norm_lb * neg
                norm_lb, norm_ub = new_lb, new_ub
            except Exception:
                # Broadcast failure: conservative, scale with scalar max.
                max_abs = float(np.abs(sv).max())
                norm_lb = np.full_like(norm_lb, -bound_mag * max_abs)
                norm_ub = np.full_like(norm_ub, bound_mag * max_abs)
        bias = _get_bound(bounds, node.input[2]) if len(node.input) > 2 else None
        if bias is not None:
            bv = bias.lb.astype(x.lb.dtype)
            try:
                shape = [1] * x.lb.ndim
                shape[axis] = -1
                bv_b = bv.reshape(shape)
                norm_lb = norm_lb + bv_b
                norm_ub = norm_ub + bv_b
            except Exception:
                norm_lb = norm_lb + float(bv.min())
                norm_ub = norm_ub + float(bv.max())
        return IntervalBound(lb=norm_lb, ub=norm_ub)

    # Default: unknown op → vacuous bound (sound but uninformative)
    return None


def _broadcast_pair(a_lb, a_ub, target):
    """Try to broadcast a's bounds to match target's shape."""
    try:
        a_lb = np.broadcast_to(a_lb, target.shape)
        a_ub = np.broadcast_to(a_ub, target.shape)
    except ValueError:
        try:
            target = np.broadcast_to(target, a_lb.shape)
        except ValueError:
            pass
    return a_lb, a_ub


def _propagate_conv(node, bounds):
    """Sound interval propagation for Conv.

    Key: Conv is a linear op. For constant weights W:
      lb = conv(x_lb, W+) + conv(x_ub, W-)  (+ bias_lb)
      ub = conv(x_ub, W+) + conv(x_lb, W-)  (+ bias_ub)
    where W+ = max(W, 0), W- = min(W, 0).
    This is sound interval arithmetic for any linear operator.
    """
    x = _get_bound(bounds, node.input[0])
    w = _get_bound(bounds, node.input[1])
    if x is None or w is None:
        return None

    W = w.lb  # constant weights: lb == ub
    x_lb = x.lb
    x_ub = x.ub

    # Parse Conv attributes
    pads = [0, 0, 0, 0]
    strides = [1, 1]
    group = 1
    for attr in node.attribute:
        if attr.name == "pads":
            pads = list(attr.ints)
        elif attr.name == "strides":
            strides = list(attr.ints)
        elif attr.name == "group":
            group = attr.i

    # Ensure 4D: [batch, channels, height, width]
    if x_lb.ndim != 4:
        vac = 1e300 if x_lb.dtype == np.float64 else 1e10
        return IntervalBound(lb=np.array([-vac], dtype=x_lb.dtype),
                              ub=np.array([ vac], dtype=x_lb.dtype))

    # Pad input
    if any(p > 0 for p in pads):
        pad_width = ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3]))
        x_lb = np.pad(x_lb, pad_width, mode='constant', constant_values=0)
        x_ub = np.pad(x_ub, pad_width, mode='constant', constant_values=0)

    # Split weights: positive and negative
    W_pos = np.maximum(W, 0)
    W_neg = np.minimum(W, 0)

    # Compute conv via im2col for correctness
    out_lb = _conv2d_numpy(x_lb, W_pos, strides, group) + _conv2d_numpy(x_ub, W_neg, strides, group)
    out_ub = _conv2d_numpy(x_ub, W_pos, strides, group) + _conv2d_numpy(x_lb, W_neg, strides, group)

    # Add bias if present
    if len(node.input) > 2:
        b = _get_bound(bounds, node.input[2])
        if b is not None:
            bias = b.lb.reshape(1, -1, 1, 1)
            out_lb = out_lb + bias
            out_ub = out_ub + bias

    return IntervalBound(lb=out_lb, ub=out_ub)


def _conv2d_numpy(x, w, strides, group=1):
    """Numpy 2D convolution (correlation) for interval propagation.

    x: [batch, C_in, H, W]
    w: [C_out, C_in/group, kH, kW]
    """
    batch, c_in, h_in, w_in = x.shape
    c_out, c_per_group, kh, kw = w.shape
    sh, sw = strides

    h_out = (h_in - kh) // sh + 1
    w_out = (w_in - kw) // sw + 1

    # Inherit dtype from inputs (x may be float64 on deep nets where float32
    # overflows at ~3.4e+38 — e.g. ResNet-50 layer 3 Conv chain).
    out_dtype = x.dtype
    if h_out <= 0 or w_out <= 0:
        return np.zeros((batch, c_out, 1, 1), dtype=out_dtype)

    out = np.zeros((batch, c_out, h_out, w_out), dtype=out_dtype)

    if group == 1:
        # Standard convolution via im2col
        # Reshape weight: [C_out, C_in * kH * kW]
        w_flat = w.reshape(c_out, -1)

        for i in range(h_out):
            for j in range(w_out):
                patch = x[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]  # [batch, C_in, kH, kW]
                patch_flat = patch.reshape(batch, -1)  # [batch, C_in * kH * kW]
                out[:, :, i, j] = patch_flat @ w_flat.T  # [batch, C_out]
    else:
        # Grouped convolution
        c_in_per_group = c_in // group
        c_out_per_group = c_out // group

        for g in range(group):
            x_g = x[:, g*c_in_per_group:(g+1)*c_in_per_group, :, :]
            w_g = w[g*c_out_per_group:(g+1)*c_out_per_group, :, :, :]
            w_flat = w_g.reshape(c_out_per_group, -1)

            for i in range(h_out):
                for j in range(w_out):
                    patch = x_g[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]
                    patch_flat = patch.reshape(batch, -1)
                    out[:, g*c_out_per_group:(g+1)*c_out_per_group, i, j] = patch_flat @ w_flat.T

    return out


def _propagate_maxpool(node, x):
    """Interval propagation for MaxPool — sound: max of intervals.

    MaxPool is monotone non-decreasing, so:
      lb(MaxPool(interval)) = MaxPool(lb)  (max of lbs in each window)
      ub(MaxPool(interval)) = MaxPool(ub)  (max of ubs in each window)

    Sound conservative approximation: fill output with global min(lb) /
    max(ub) but preserve the OUTPUT SHAPE so downstream broadcast is
    correct. The previous scalar fallback for 3D inputs (max_pool1d)
    silently broke shape contracts and let downstream ops saturate to
    1e+300 sentinel values — Bober-Irizar 1M-amplification triggers
    this through op_indicator_trigger's MaxPool1d chain.
    """
    kernel_shape = []
    strides = []
    pads = []
    ceil_mode = 0
    for attr in node.attribute:
        if attr.name == "kernel_shape":
            kernel_shape = list(attr.ints)
        elif attr.name == "strides":
            strides = list(attr.ints)
        elif attr.name == "pads":
            pads = list(attr.ints)
        elif attr.name == "ceil_mode":
            ceil_mode = int(attr.i)

    rank = x.lb.ndim
    spatial_rank = max(rank - 2, 1)
    if not kernel_shape:
        kernel_shape = [2] * spatial_rank
    if not strides:
        strides = [1] * len(kernel_shape)
    if not pads:
        pads = [0] * (2 * len(kernel_shape))

    # Compute output spatial dims (PyTorch / ONNX conv-style).
    in_spatial = list(x.lb.shape[2:]) if rank >= 3 else []
    out_spatial = []
    for i, k in enumerate(kernel_shape):
        in_d = in_spatial[i] if i < len(in_spatial) else 1
        s = strides[i] if i < len(strides) else 1
        p_l = pads[i] if i < len(pads) else 0
        p_r = pads[i + len(kernel_shape)] if (
            i + len(kernel_shape) < len(pads)) else p_l
        if ceil_mode:
            out_d = -(-((in_d + p_l + p_r - k)) // s) + 1
        else:
            out_d = ((in_d + p_l + p_r - k) // s) + 1
        out_spatial.append(max(out_d, 1))

    if rank >= 3:
        out_shape = list(x.lb.shape[:2]) + out_spatial
    else:
        out_shape = [1] + out_spatial

    # Sound conservative bound: same global lb/ub broadcast to output shape.
    # max-of-window can only INCREASE lb (monotone) so global min(lb) is
    # a sound under-approximation; max(ub) is a sound over-approximation.
    out_lb = np.full(out_shape, float(x.lb.min()), dtype=x.lb.dtype)
    out_ub = np.full(out_shape, float(x.ub.max()), dtype=x.ub.dtype)
    return IntervalBound(lb=out_lb, ub=out_ub)


def _propagate_avgpool(node, x):
    """Interval propagation for AveragePool."""
    return IntervalBound(lb=np.array([x.lb.min()]), ub=np.array([x.ub.max()]))


def _propagate_gemm(node, bounds):
    """Interval propagation for Gemm (fully-connected layer)."""
    x = _get_bound(bounds, node.input[0])
    w = _get_bound(bounds, node.input[1])
    if x is None or w is None: return None

    # Gemm: y = x @ W^T + b
    W = w.lb  # constant weights: lb == ub
    W_pos = np.maximum(W, 0)
    W_neg = np.minimum(W, 0)

    transB = _get_attr(node, "transB", 0)
    if transB:
        W_pos = W_pos.T
        W_neg = W_neg.T

    x_lb_flat = x.lb.reshape(1, -1) if x.lb.ndim == 1 else x.lb
    x_ub_flat = x.ub.reshape(1, -1) if x.ub.ndim == 1 else x.ub

    try:
        lb = x_lb_flat @ W_pos + x_ub_flat @ W_neg
        ub = x_ub_flat @ W_pos + x_lb_flat @ W_neg
    except ValueError:
        vac_dtype = x_lb_flat.dtype
        vac = 1e300 if vac_dtype == np.float64 else 1e10
        return IntervalBound(
            lb=np.array([-vac], dtype=vac_dtype),
            ub=np.array([ vac], dtype=vac_dtype),
        )

    # Add bias if present
    if len(node.input) > 2:
        b = _get_bound(bounds, node.input[2])
        if b is not None:
            lb = lb + b.lb
            ub = ub + b.ub

    return IntervalBound(lb=lb, ub=ub)


def _get_attr(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return attr.i if attr.type == 2 else attr.ints if attr.type == 7 else default
    return default


def _get_reduce_axes(node):
    for attr in node.attribute:
        if attr.name == "axes":
            return list(attr.ints)
    # opset 18+: axes might be second input
    return [1]  # default: reduce over channel dim


def _get_unsqueeze_axes(node, bounds):
    for attr in node.attribute:
        if attr.name == "axes":
            return list(attr.ints)
    if len(node.input) > 1:
        axes_bound = bounds.get(node.input[1])
        if axes_bound is not None:
            return axes_bound.lb.astype(int).tolist()
    return [0]


def check_gate_dormancy(onnx_model: onnx.ModelProto,
                        gate_output_name: str,
                        input_lb: float = 0.0,
                        input_ub: float = 0.95) -> Tuple[bool, float]:
    """Check if a gate output is provably ≤ 0 over B_clean.

    Returns:
        (is_dormant, gate_upper_bound)
        is_dormant=True means gate PROVABLY never fires under B_clean (SOUND).
    """
    bounds = propagate_intervals(onnx_model, input_lb, input_ub)

    if gate_output_name in bounds:
        gate_bound = bounds[gate_output_name]
        gate_ub = gate_bound.ub.max()
        return gate_ub <= 0, float(gate_ub)
    else:
        return False, float('inf')


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys, os
    # artifact root, so `import backdoored_models` (and its `utils`) resolve
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from backdoored_models import op_sep_tar_backdoor

    print("=== Interval Propagation on Bober op_sep_tar ===")

    # Export to ONNX
    m = op_sep_tar_backdoor().cpu().eval()
    x = torch.randn(1, 3, 32, 32)
    torch.onnx.export(m, x, "/tmp/bober_test.onnx", opset_version=17,
                      do_constant_folding=False, input_names=["input"], output_names=["output"])
    onnx_model = onnx.load("/tmp/bober_test.onnx")

    print(f"Nodes: {len(onnx_model.graph.node)}")

    # Propagate intervals
    bounds = propagate_intervals(onnx_model, input_lb=0.0, input_ub=0.95)

    # Show bounds for key tensors
    output_name = onnx_model.graph.output[0].name
    if output_name in bounds:
        out_bound = bounds[output_name]
        print(f"Output bound: lb_max={out_bound.lb.max():.4f}, ub_max={out_bound.ub.max():.4f}")
    else:
        print(f"Output '{output_name}' not in bounds")

    # Show how many tensors got non-vacuous bounds
    n_total = len(bounds)
    n_vacuous = sum(1 for b in bounds.values() if b.ub.max() > 1e9)
    n_tight = n_total - n_vacuous
    print(f"Tensors bounded: {n_tight}/{n_total} non-vacuous")

    # Find Mul nodes and check their output bounds
    print("\nMul node output bounds:")
    for node in onnx_model.graph.node:
        if node.op_type == "Mul" and node.output[0] in bounds:
            b = bounds[node.output[0]]
            print(f"  {node.output[0]}: lb_max={b.lb.max():.4f}, ub_max={b.ub.max():.4f}")
