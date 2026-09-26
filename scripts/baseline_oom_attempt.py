"""Per-LLM baseline attempt: try auto_LiRPA + α,β-CROWN to verify the
backdoored 7B-LLM ONNX. Records the phase where it OOMs / fails / times out.

This is the negative baseline: prior NN-verifiers (auto_LiRPA, α,β-CROWN)
cannot scale to 7B-LLM ONNX. We attempt the standard pipeline and record
the failure mode honestly, providing direct evidence for paper §6.4.

Phases:
  1. ONNX load          (control — always succeeds)
  2. onnx2pytorch convert (often fails on unsupported ops or OOMs)
  3. auto_LiRPA BoundedModule wrap (often OOMs at gradient init)
  4. CROWN compute_bounds (definitely OOMs at 7B scale)

Output: print result dict as JSON on stdout. Wrapped script captures.

Usage:
    python3 scripts/baseline_oom_attempt.py <path_to_backdoored.onnx>
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA_BETA_CROWN_PATH = os.environ.get("ALPHA_BETA_CROWN_PATH") or os.path.join(
    os.path.dirname(_HERE), "baselines", "alpha-beta-CROWN")   # bundled copy
sys.path.insert(0, ALPHA_BETA_CROWN_PATH)
sys.path.insert(0, os.path.join(ALPHA_BETA_CROWN_PATH, "auto_LiRPA"))


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": "usage: <path>"}))
        sys.exit(1)
    onnx_path = sys.argv[1]

    res = {
        "onnx_path": onnx_path,
        "phase_reached": "init",
        "fail_mode": None,
        "error_msg": "",
        "wall_sec": 0.0,
    }
    t0 = time.time()

    # -------- Phase 1: ONNX load (control) --------
    try:
        import onnx
        m = onnx.load(onnx_path)
        res["phase_reached"] = "onnx_loaded"
        res["onnx_load_sec"] = time.time() - t0
        print(f"[t={time.time()-t0:.1f}s] ONNX loaded "
              f"({len(m.graph.node)} nodes, {len(m.graph.initializer)} init)",
              flush=True)
    except Exception as e:
        res["fail_mode"] = "onnx_load_fail"
        res["error_msg"] = f"{type(e).__name__}: {str(e)[:200]}"
        res["wall_sec"] = time.time() - t0
        print(json.dumps(res))
        sys.exit(0)

    # -------- Phase 2: onnx2pytorch convert --------
    try:
        from onnx2pytorch import ConvertModel
        print(f"[t={time.time()-t0:.1f}s] Converting to PyTorch via "
              f"onnx2pytorch ...", flush=True)
        torch_model = ConvertModel(m)
        res["phase_reached"] = "onnx2pytorch_converted"
        res["onnx2pytorch_sec"] = time.time() - t0
        print(f"[t={time.time()-t0:.1f}s] PyTorch converted ok", flush=True)
    except Exception as e:
        res["fail_mode"] = "onnx2pytorch_fail"
        res["error_msg"] = f"{type(e).__name__}: {str(e)[:300]}"
        res["wall_sec"] = time.time() - t0
        print(f"[t={time.time()-t0:.1f}s] FAIL onnx2pytorch: "
              f"{res['error_msg']}", flush=True)
        traceback.print_exc()
        print(json.dumps(res))
        sys.exit(0)

    # -------- Phase 3: auto_LiRPA BoundedModule --------
    try:
        import torch
        from auto_LiRPA import BoundedModule
        print(f"[t={time.time()-t0:.1f}s] Wrapping with auto_LiRPA "
              f"BoundedModule ...", flush=True)
        # 7B LLMs have int64 token-ID input shape [1, seq]. Standard
        # BoundedModule expects floats; this attempt is lenient because
        # we only need a control-flow attempt, not a successful bound.
        dummy_input = torch.zeros((1, 16), dtype=torch.long)
        bounded = BoundedModule(torch_model, dummy_input,
                                bound_opts={'verbosity': 0})
        res["phase_reached"] = "bounded_module_wrapped"
        res["bounded_module_sec"] = time.time() - t0
        print(f"[t={time.time()-t0:.1f}s] BoundedModule ok", flush=True)
    except Exception as e:
        res["fail_mode"] = "bounded_module_fail"
        res["error_msg"] = f"{type(e).__name__}: {str(e)[:300]}"
        res["wall_sec"] = time.time() - t0
        print(f"[t={time.time()-t0:.1f}s] FAIL BoundedModule: "
              f"{res['error_msg']}", flush=True)
        traceback.print_exc()
        print(json.dumps(res))
        sys.exit(0)

    # -------- Phase 4: CROWN compute_bounds --------
    try:
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
        ptb = PerturbationLpNorm(norm=float('inf'), eps=0.05)
        bx = BoundedTensor(dummy_input, ptb)
        print(f"[t={time.time()-t0:.1f}s] Computing CROWN bounds ...",
              flush=True)
        lb, ub = bounded.compute_bounds(x=(bx,), method="CROWN")
        res["phase_reached"] = "crown_bounds_computed"
        res["crown_sec"] = time.time() - t0
        print(f"[t={time.time()-t0:.1f}s] CROWN done. "
              f"lb_shape={list(lb.shape)} ub_shape={list(ub.shape)}",
              flush=True)
        res["lb_shape"] = list(lb.shape)
        res["ub_shape"] = list(ub.shape)
        res["fail_mode"] = "none"
        res["wall_sec"] = time.time() - t0
        print(json.dumps(res))
        sys.exit(0)
    except Exception as e:
        res["fail_mode"] = "crown_fail"
        res["error_msg"] = f"{type(e).__name__}: {str(e)[:300]}"
        res["wall_sec"] = time.time() - t0
        print(f"[t={time.time()-t0:.1f}s] FAIL CROWN: "
              f"{res['error_msg']}", flush=True)
        traceback.print_exc()
        print(json.dumps(res))
        sys.exit(0)


if __name__ == "__main__":
    main()
