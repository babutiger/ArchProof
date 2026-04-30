"""Phase C6: α,β-CROWN escalation for cases where IBP blows up.

Given an ONNX model and a trigger-aware input box `D(T)`, compute a
sound upper bound on the per-gate contribution using CROWN-IBP on
the payload tensor. This replaces the IBP-vacuous
`payload_abs_max = 1e+300` that causes the Phase-C v2 verifier to
return UNCERTIFIED on deep networks (EfficientNet-B0, GLU-Net,
RegNet-Y-400MF, all 12 E14 production-scale backdoors).

Pipeline:
  1. `onnx2pytorch.ConvertModel` loads the ONNX graph as an nn.Module.
  2. `auto_LiRPA.BoundedModule` wraps it for CROWN / CROWN-IBP.
  3. Instrument the internal layers to expose payload-tensor bounds.
  4. `compute_bounds(method='CROWN-IBP')` gives tighter upper bounds
     on the admitted gate's payload, enabling a finite per-gate
     contribution in Phase C's Prop 1' sum.

Scope of this module:
  - Load ONNX + wrap in BoundedModule on CPU (matches legacy pipeline).
  - Run CROWN bound on an input interval [lb, ub] over the full graph.
  - Return `{tensor_name: (lb_array, ub_array)}` for tensors we asked
    for, or `None` if CROWN cannot handle the graph.
  - Soundness is preserved: CROWN is a sound upper/lower bound by
    construction.

What this module does NOT yet do:
  - Instrumenting ONNX to expose per-gate Mul-output tensor bounds
    requires the `final_node_name` option or per-node splitting.
    For now we compute the graph-level output bound and rely on the
    chain-Lipschitz argument of \S C2 §4 (Prop 1') to propagate it.
  - Integration with `verify_phaseC.verify_model_phaseC`: the caller
    wraps the escalation behind a `use_crown_escalation=True` flag.
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import onnx
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _onnx_to_pytorch(onnx_path: str, dummy_input: torch.Tensor
                     ) -> Optional[torch.nn.Module]:
    """Convert ONNX to a PyTorch nn.Module. Returns None on failure."""
    try:
        import onnx2pytorch
        m = onnx.load(onnx_path)
        model = onnx2pytorch.ConvertModel(m, experimental=True)
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        return model
    except Exception as e:
        return None


def crown_bound_output(onnx_path: str,
                       input_lb: float = 0.0,
                       input_ub: float = 1.0,
                       method: str = "CROWN-IBP",
                       device: str = "cpu"
                       ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute a sound output-tensor bound on the input box [lb, ub]^d.

    Returns (output_lb, output_ub) as numpy arrays, or None if the
    graph cannot be converted / bounded.

    `method` options (passed to `BoundedModule.compute_bounds`):
      - 'IBP'        : pure interval bound (matches legacy verify.py)
      - 'CROWN'      : linear-relaxation bound (tighter, slower)
      - 'CROWN-IBP'  : hybrid (fast + tight), default choice
      - 'alpha-CROWN': optimizable CROWN (tightest, slowest)
    """
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
    except Exception:
        return None

    m = onnx.load(onnx_path)
    shape = []
    for dim in m.graph.input[0].type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.dim_value > 0 else 1)

    dummy_input = torch.rand(shape, dtype=torch.float32) * float(input_ub)
    pt = _onnx_to_pytorch(onnx_path, dummy_input)
    if pt is None:
        return None

    try:
        bounded = BoundedModule(pt, dummy_input, device=device,
                                bound_opts={"conv_mode": "patches"})
    except Exception:
        return None

    eps_width = (input_ub - input_lb) / 2.0
    center = torch.full_like(dummy_input, float(input_lb + input_ub) / 2.0)
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps_width,
                              x_L=torch.full_like(center, float(input_lb)),
                              x_U=torch.full_like(center, float(input_ub)))
    x = BoundedTensor(center, ptb)

    try:
        lb, ub = bounded.compute_bounds(x=(x,), method=method)
    except Exception:
        # Some ops are still unsupported by CROWN — fall back to IBP.
        try:
            lb, ub = bounded.compute_bounds(x=(x,), method="IBP")
        except Exception:
            return None
    return (lb.detach().cpu().numpy(), ub.detach().cpu().numpy())


def output_abs_max(lb: np.ndarray, ub: np.ndarray) -> float:
    """Return the max absolute value across a vector interval [lb, ub]."""
    return float(np.maximum(np.abs(lb), np.abs(ub)).max())
