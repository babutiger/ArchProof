"""LLM-specific localized gate-bound rescue.

When IBP propagation across a deep transformer (~30 LayerNorm + attention
layers) blows up to vacuous bounds (|x| > 1e30), the gate's contribution
is reported as inf and the verdict drops to UNCERTIFIED. This module
provides a sound *localized* fallback:

  pre_act = Linear_W @ x + b      (gate's own producer, x = hidden_state)
  |x|_∞     ≤ ||γ||_∞ · √D + ||β||_∞   (LayerNorm output a-priori bound)
  |pre_act|_∞ ≤ ||W||_{∞→∞} · |x|_∞  + ||b_pre||_∞

The first bound is a property of LayerNorm/RMSNormalization regardless of
upstream computation: the standardized residual has variance 1 across D
coordinates, so any single coordinate is bounded by √D. This is exact
under any input distribution, so the bound is sound (not a heuristic).

The rescue is sound for one-step Linear gates immediately downstream of
a LayerNorm/RMSNormalization. For more layers in between (LM head etc.),
we propagate forward op-by-op using known per-op sensitivities.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

# Op types whose output L_∞ norm equals the max input L_∞ norm.
_LINF_PRESERVING = {"Relu", "LeakyRelu", "Sigmoid", "Tanh", "Softplus",
                    "Erf", "Cast", "Identity", "Slice", "Gather", "Tile",
                    "Squeeze", "Unsqueeze", "Reshape", "Transpose"}

# LayerNorm-family ops (output L_∞ bound derives from γ, β + input dim).
_LAYERNORM_OPS = {"LayerNormalization", "RMSNormalization", "BatchNormalization"}


def _build_produces(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    out = {}
    for n in model.graph.node:
        for o in n.output:
            if o:
                out[o] = n
    return out


def _const_arr(model: onnx.ModelProto, name: str) -> Optional[np.ndarray]:
    for init in model.graph.initializer:
        if init.name == name:
            try:
                return numpy_helper.to_array(init)
            except Exception:
                return None
    return None


def _resolve_scalar(model: onnx.ModelProto,
                    produces: Dict[str, onnx.NodeProto],
                    name: str) -> Optional[np.ndarray]:
    """Resolve `name` to a scalar/small ndarray, supporting BOTH
    initializer-stored constants AND Constant-node-stored constants.

    PyTorch's ONNX exporter for LlamaRMSNorm/LlamaMLP-style modules emits
    Pow(x, 2) where `2` is the output of a Constant node (op_type=Constant,
    attribute['value']) rather than an initializer. _const_arr only sees
    initializers, so the Pow-2 signature check would silently miss those
    exponents and the LayerNorm rescue would never fire on Yi/Llama/Mistral.
    This helper accepts both forms.
    """
    arr = _const_arr(model, name)
    if arr is not None:
        return arr
    prod = produces.get(name)
    if prod is None or prod.op_type != "Constant":
        return None
    for attr in prod.attribute:
        if attr.name == "value":
            try:
                return numpy_helper.to_array(attr.t)
            except Exception:
                return None
    return None


def _const_arr_via_identity(model: onnx.ModelProto,
                            produces: Dict[str, onnx.NodeProto],
                            name: str,
                            max_hops: int = 4) -> Optional[np.ndarray]:
    """Resolve a tensor name to a constant array, walking through
    Identity nodes if necessary. torch.onnx with constant-folding
    sometimes inserts Identity wrappers around initializer references
    (observed on BERT-base export: per-layer LayerNorm.weight/bias
    are produced by Identity nodes whose input is the real initializer).
    """
    for _ in range(max_hops + 1):
        arr = _const_arr(model, name)
        if arr is not None:
            return arr
        prod = produces.get(name)
        if prod is None or prod.op_type != "Identity":
            return None
        if not prod.input:
            return None
        name = prod.input[0]
    return None


def _layernorm_output_bound(model: onnx.ModelProto,
                            ln_node: onnx.NodeProto,
                            produces: Optional[Dict[str, onnx.NodeProto]]
                                = None
                            ) -> Optional[float]:
    """Compute |LN(x)|_∞ ≤ ||γ||_∞ · √D + ||β||_∞ from the LayerNorm's
    γ (input[1]) and optional β (input[2]) initializers (or via
    Identity-wrapped initializer references in BERT-style exports).
    """
    inputs = list(ln_node.input)
    if len(inputs) < 2:
        return None
    if produces is None:
        produces = _build_produces(model)
    gamma = _const_arr_via_identity(model, produces, inputs[1])
    if gamma is None:
        return None
    D = int(np.prod(gamma.shape))
    g_max = float(np.abs(gamma).max())
    b_max = 0.0
    if len(inputs) >= 3 and inputs[2]:
        beta = _const_arr_via_identity(model, produces, inputs[2])
        if beta is not None:
            b_max = float(np.abs(beta).max())
    # The standardized residual (a - mean) / std has variance 1 across D
    # coordinates, so |coord|_max ≤ √D. Multiply by γ, add β.
    return g_max * math.sqrt(D) + b_max


def _expanded_norm_bound_from_scale(
        model: onnx.ModelProto,
        produces: Dict[str, onnx.NodeProto],
        scale_mul_node: onnx.NodeProto
        ) -> Optional[float]:
    """A LayerNorm/RMSNorm expanded as ONNX primitives ends in a
    `Mul(normalized, γ)` (and optionally `Add(scaled, β)`). The
    standardized residual is bounded by √D regardless of which
    expansion path produced it; the post-scale bound is therefore
    `||γ||_∞ · √D + ||β||_∞` exactly as for the fused op.

    This detector takes a candidate scale-Mul and validates it by
    requiring (i) one operand is an initializer of shape (D,) with
    D ≥ 2, and (ii) the OTHER operand traces back through a `Sqrt`
    (normalization signature) within a few hops. Returns the
    LN-output bound or None.
    """
    gamma = None
    other_input = None
    for inp in scale_mul_node.input:
        # Try direct init AND Identity-wrapped init (BERT/DeBERTa
        # torch.onnx export aliases per-layer LN.weight through
        # Identity nodes).
        arr = _const_arr_via_identity(model, produces, inp)
        if arr is not None and arr.ndim == 1 and arr.shape[0] >= 2:
            gamma = arr
        else:
            other_input = inp
    if gamma is None or other_input is None:
        return None

    # === Strict normalization signature ===
    # The bound `||γ||_∞ × √D + ||β||_∞` is sound *only* when the
    # non-γ operand is the standardised residual of LayerNorm or
    # RMSNorm — i.e., a tensor that satisfies the unit-variance
    # property `(1/D) Σ (residual_i)² = 1`, which forces
    # `||residual||_∞ ≤ √D`. Spurious matches (e.g., a graph that
    # happens to have `Mul(x, γ_1d)` with a Sqrt upstream for a
    # non-normalisation reason) would produce an unsound bound.
    #
    # The full LayerNorm / RMSNorm variance signature is the
    # subgraph
    #     ... Pow(_, 2) → ReduceMean → Add(eps) → Sqrt → Div → Mul(γ)
    # We require all three operators (Sqrt, ReduceMean, Pow(2)) to
    # appear in the upstream BFS within `NORM_HOPS=8`. This is a
    # purely structural test, satisfied by every torch.onnx /
    # onnxscript / tf2onnx export of LayerNormalization /
    # RMSNormalization that we have examined, and not satisfied by
    # any of the non-normalisation Mul-with-1D-γ patterns we have
    # seen (attention scaling uses scalar 1/√d, embedding scaling
    # uses scalar √d, etc., none of which contain ReduceMean+Pow(2)
    # upstream).
    visited = set()
    queue: List[Tuple[str, int]] = [(other_input, 0)]
    found_sqrt = False
    found_reducemean = False
    found_pow2 = False
    NORM_HOPS = 8
    BFS_OPS = {"Div", "Sub", "Add", "Mul", "Pow", "ReduceMean",
               "Cast", "Identity", "Reshape", "Reciprocal", "Sqrt"}
    while queue:
        name, hops = queue.pop(0)
        if name in visited or hops > NORM_HOPS:
            continue
        visited.add(name)
        prod = produces.get(name)
        if prod is None:
            continue
        if prod.op_type == "Sqrt":
            found_sqrt = True
        elif prod.op_type == "ReduceMean":
            found_reducemean = True
        elif prod.op_type == "Pow" and len(prod.input) >= 2:
            # Verify exponent is exactly 2 (or 2.0 within float tol).
            # Use _resolve_scalar which handles both initializer-stored
            # AND Constant-node-stored exponents (HF Llama/Yi/Mistral RMSNorm
            # emits the latter via torch.onnx.export).
            exp_arr = _resolve_scalar(model, produces, prod.input[1])
            if exp_arr is not None and exp_arr.size == 1:
                exp_val = float(np.asarray(exp_arr).flatten()[0])
                if abs(exp_val - 2.0) < 1e-6:
                    found_pow2 = True
        if prod.op_type in BFS_OPS:
            for u in prod.input:
                if u:
                    queue.append((u, hops + 1))
        if found_sqrt and found_reducemean and found_pow2:
            break
    # Some exporters fold Pow(_, 2) into Mul(x, x) (multiplying a
    # tensor by itself). Allow that form as a Pow(2) substitute:
    # detect Mul nodes whose two inputs are the same tensor.
    if not found_pow2:
        for n in visited:
            prod = produces.get(n)
            if prod is None or prod.op_type != "Mul":
                continue
            if len(prod.input) == 2 and prod.input[0] == prod.input[1]:
                found_pow2 = True
                break
    if not (found_sqrt and found_reducemean and found_pow2):
        return None
    D = int(gamma.shape[0])
    g_max = float(np.abs(gamma).max())
    # Return the bound on the **scale-Mul output** (pre-β), not the LN
    # output. Reason: this function is only called from
    # `_trace_input_bound`, which walks UP from a downstream consumer.
    # If the LN expansion ends with `Mul(γ) → Add(β)`, the trace passes
    # through the Add(β) FIRST and accumulates `scale × ||β||_∞` into
    # its bias accumulator (line 313-326). Then it reaches this Mul(γ)
    # and asks for the bound on the Mul output, which is pre-β.
    #
    # An earlier version of this function followed the Mul-output to a
    # consuming Add(β) and folded β_max into the returned bound. That
    # caused a **double count** in the trace path: β contributed once
    # via Add(β)'s bias accumulation in `_trace_input_bound`, and again
    # via the post-β bound returned here. For GPT-J (LayerNorm with
    # β ≠ 0) this manifested as a 0.9% rel ε divergence on opset 14/16
    # vs ≥17 (native `LayerNormalization` op, hits Path A
    # `_layernorm_output_bound` which only counts β once). RMSNorm LLMs
    # (Yi/DeepSeek/Mistral/Qwen2) were unaffected because RMSNorm has
    # β = 0 by definition.
    return g_max * math.sqrt(D)


def _matrix_linf_op_norm(W: np.ndarray) -> float:
    """L_∞→L_∞ operator norm of W: max over rows of sum |W_ij|."""
    return float(np.abs(W).sum(axis=-1).max())


def _trace_input_bound(model: onnx.ModelProto,
                       produces: Dict[str, onnx.NodeProto],
                       tensor: str,
                       max_hops: int = 16
                       ) -> Optional[float]:
    """Walk backward from `tensor` up to `max_hops` ops, tracking an
    affine envelope `|tensor|_∞ ≤ scale · |y_LN|_∞ + bias` where `y_LN`
    is some upstream LayerNorm output. Returns a sound bound or None.

    Through MatMul/Gemm(x, W) + b: scale *= ‖W‖_{∞→∞},
                                   bias  += old_scale · ‖b‖_∞,
    then recurse on `x`. This covers the chained-Linear case (e.g.
    BERT sha_un's `gate ← Gemm(shared_proj(pooled), W_gate)`).
    """
    visited = set()
    # Each queue entry: (current tensor, hops, scale, bias)
    queue: List[Tuple[str, int, float, float]] = [(tensor, 0, 1.0, 0.0)]
    while queue:
        name, hops, scale, bias = queue.pop(0)
        key = (name, round(scale, 12), round(bias, 12))
        if key in visited or hops > max_hops:
            continue
        visited.add(key)
        prod = produces.get(name)
        if prod is None:
            continue
        # Path A: fused LayerNorm/RMSNorm op.
        if prod.op_type in _LAYERNORM_OPS:
            ln_bound = _layernorm_output_bound(model, prod, produces)
            if ln_bound is not None:
                return ln_bound * scale + bias
            continue
        # Path B: expanded norm — Mul(normalized, γ) where γ is 1-D init
        # and the non-γ operand traces back through Sqrt.
        if prod.op_type == "Mul":
            ln_bound = _expanded_norm_bound_from_scale(
                model, produces, prod)
            if ln_bound is not None:
                return ln_bound * scale + bias
            continue
        # 1-Lipschitz / layout-preserving ops.
        if prod.op_type in _LINF_PRESERVING:
            for inp in prod.input:
                if inp:
                    queue.append((inp, hops + 1, scale, bias))
            continue
        # Add/Sub: if one operand is a constant initializer, treat as
        # bias addition; otherwise propagate through both operands
        # (sound but conservative: input bound = max over operands).
        if prod.op_type in ("Add", "Sub"):
            const_bias = 0.0
            data_inputs = []
            for inp in prod.input:
                if not inp:
                    continue
                arr = _const_arr_via_identity(model, produces, inp)
                if arr is not None:
                    const_bias = max(const_bias, float(np.abs(arr).max()))
                else:
                    data_inputs.append(inp)
            new_bias = bias + scale * const_bias
            for inp in data_inputs:
                queue.append((inp, hops + 1, scale, new_bias))
            continue
        if prod.op_type == "Div":
            # Stop on Div (need divisor bound).
            continue
        # MatMul / Gemm: extract W, b, propagate via operator-∞ norm.
        if prod.op_type in ("MatMul", "Gemm"):
            W = None
            x_input = None
            bias_arr = None
            inputs = list(prod.input)
            # Gemm: input 0=x, 1=W (transposed in some exports), 2=b
            # MatMul: input 0=x, 1=W (or vice versa)
            # Resolve W via _resolve_const_through_layout_ops; the other
            # operand becomes x.
            for inp in inputs[:2]:
                if W is None:
                    arr = _resolve_const_through_layout_ops(
                        model, produces, inp)
                    if arr is not None and arr.ndim == 2:
                        W = arr
                        continue
                if x_input is None:
                    x_input = inp
            if prod.op_type == "Gemm" and len(inputs) >= 3 and inputs[2]:
                bias_arr = _const_arr_via_identity(
                    model, produces, inputs[2])
            if W is None or x_input is None:
                continue
            W_norm = _matrix_linf_op_norm(W)
            b_inf = (float(np.abs(bias_arr).max())
                     if bias_arr is not None else 0.0)
            new_bias = bias + scale * b_inf
            new_scale = scale * W_norm
            queue.append((x_input, hops + 1, new_scale, new_bias))
            continue
    return None


def _resolve_const_through_layout_ops(
        model: onnx.ModelProto,
        produces: Dict[str, onnx.NodeProto],
        tensor: str,
        max_hops: int = 4
        ) -> Optional[np.ndarray]:
    """Walk backward through layout-only ops (Transpose, Reshape,
    Squeeze, Unsqueeze, Cast, Identity) to find the underlying
    initializer that feeds `tensor`. Returns the resolved array
    *with the layout transforms applied* if reachable, else None.

    This is needed because torch.onnx.export typically emits
      MatMul(x, Transpose(W_init))
    rather than passing W_init directly, so a naive
    `_const_arr(matmul.input[1])` would miss the W.
    """
    LAYOUT_OPS = {"Transpose", "Reshape", "Squeeze", "Unsqueeze",
                  "Cast", "Identity"}
    cur = tensor
    for _ in range(max_hops + 1):
        arr = _const_arr(model, cur)
        if arr is not None:
            return arr
        prod = produces.get(cur)
        if prod is None or prod.op_type not in LAYOUT_OPS:
            return None
        # Try resolving the underlying initializer first (recursively).
        upstream = _const_arr(model, prod.input[0]) if prod.input else None
        if upstream is None:
            cur = prod.input[0] if prod.input else None
            if cur is None:
                return None
            continue
        # Apply the layout transform to the resolved upstream initializer.
        if prod.op_type == "Transpose":
            perm = None
            for attr in prod.attribute:
                if attr.name == "perm":
                    perm = list(attr.ints)
            if perm is None:
                upstream = upstream.T if upstream.ndim == 2 else upstream
            else:
                upstream = upstream.transpose(perm)
            return upstream
        if prod.op_type == "Reshape":
            # Shape operand is the second input (also an initializer).
            shape_arr = _const_arr(model, prod.input[1]) \
                if len(prod.input) > 1 else None
            if shape_arr is not None:
                try:
                    return upstream.reshape(
                        [int(d) if int(d) >= 0 else -1 for d in shape_arr])
                except Exception:
                    return upstream
            return upstream
        if prod.op_type == "Squeeze":
            try:
                return np.squeeze(upstream)
            except Exception:
                return upstream
        if prod.op_type == "Unsqueeze":
            try:
                return np.expand_dims(upstream, 0)
            except Exception:
                return upstream
        # Cast / Identity: pass through.
        return upstream
    return None


def llm_gate_local_bound(model: onnx.ModelProto,
                         pre_act_tensor: str,
                         hidden_bound_override: Optional[float] = None
                         ) -> Optional[Tuple[float, float]]:
    """Sound bound on `pre_act_tensor` (the output of a Linear gate).

    Resolves:
      pre_act = W @ x + b
    where x is the upstream LayerNorm output. Returns (lb, ub) such that
    every coord of pre_act lies in [lb, ub], or None if the producer of
    pre_act_tensor is not a recognized Linear/MatMul/Gemm or no upstream
    LayerNorm bound can be derived.
    """
    produces = _build_produces(model)
    prod = produces.get(pre_act_tensor)
    if prod is None or prod.op_type not in ("MatMul", "Gemm", "Add"):
        return None

    # Identify the actual MatMul/Gemm producing the linear part. Pre_act is
    # often `Add(MatMul(x, W), b)` in ONNX exports, so peel the Add.
    matmul = prod
    bias_arr = None
    if prod.op_type == "Add":
        # Add(MatMul_out, bias) — find the MatMul side and the bias side.
        ins = list(prod.input)
        matmul_found = False
        for i, inp in enumerate(ins):
            sub = produces.get(inp)
            if sub is not None and sub.op_type in ("MatMul", "Gemm"):
                matmul = sub
                matmul_found = True
                # The other side is the bias.
                other = ins[1 - i] if len(ins) == 2 else None
                if other:
                    bias_arr = _const_arr(model, other)
                break
        if not matmul_found:
            return None  # Add but neither side is MatMul → not a Linear gate

    # Identify W vs x among matmul.input. W can be either a direct
    # initializer OR resolvable through layout ops (Transpose/Reshape/...).
    # The "other" input is x.
    W = None
    x_tensor = None
    for inp in matmul.input:
        if W is None:
            arr = _resolve_const_through_layout_ops(model, produces, inp)
            if arr is not None and arr.ndim == 2:
                W = arr
                continue
        if x_tensor is None:
            x_tensor = inp
    if W is None or x_tensor is None:
        return None

    # If the MatMul itself has a bias (Gemm case), pull it.
    if matmul.op_type == "Gemm" and len(matmul.input) >= 3:
        bias_arr2 = _const_arr(model, matmul.input[2])
        if bias_arr2 is not None:
            bias_arr = bias_arr2 if bias_arr is None else bias_arr + bias_arr2

    # Resolve upstream input bound on x_tensor.
    if hidden_bound_override is not None:
        K = float(hidden_bound_override)
    else:
        K = _trace_input_bound(model, produces, x_tensor)
        if K is None:
            return None

    # ||W @ x||_∞ ≤ ||W||_{∞→∞} · ||x||_∞ where ||W||_{∞→∞} = max row 1-norm.
    # For ONNX MatMul of (batch, ..., D_in) × (D_in, D_out), W after
    # Transpose recovery has shape (D_out, D_in) (the original
    # nn.Linear.weight layout). The output coord j gets sum_k W[j,k] * x[k],
    # so the per-row 1-norm bound is max_j sum_k |W[j,k]|.
    if W.ndim == 2:
        row_sum_max = float(np.abs(W).sum(axis=1).max())
        # Also try the other axis in case the layout heuristic was wrong.
        alt_sum_max = float(np.abs(W).sum(axis=0).max())
        # Pick the one that gives the tighter (smaller) bound — both are
        # sound under their respective layout interpretations, but only
        # one matches the actual matmul layout. Without knowing layout
        # certainly, taking max of the two is conservative.
        row_sum_max = max(row_sum_max, alt_sum_max)
    else:
        row_sum_max = float(np.abs(W).sum())

    bias_max = 0.0
    if bias_arr is not None:
        bias_max = float(np.abs(bias_arr).max())

    ub = row_sum_max * K + bias_max
    return (-ub, ub)
