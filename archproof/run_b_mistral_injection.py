#!/usr/bin/env python3
"""B-machine experiment: real Mistral-7B MLP subgraph + H3 injection + ArchProof.

Pipeline
--------
1. Load Mistral-7B from local HuggingFace cache or path.
2. Extract the final decoder layer's SwiGLU MLP weights
   (gate_proj, up_proj, down_proj). Mistral MLP shape:
     hidden_size = 4096, intermediate_size = 14336, activation = SiLU.
3. Build a PyTorch module wrapping that MLP.
4. Inject an H3-style Mul-indicator gate as an additive branch:
     out = mistral_mlp(x) + sigmoid(W_trigger @ x) * (W_payload @ x)
   W_trigger is initialised so the clean-input median activation
   is below τ_adm = 0.3 (dormant on [0, 0.95]^d).
5. Export to ONNX with torch.onnx.export (opset 17, default flags).
6. Run archproof.verify.verify_model on the exported file.
7. Run MGRS for target ε = 0 on the admitted gates.
8. Apply surgery (zero the injected gate) and re-verify.
9. Save results to benchmark/b1_mistral_injection.json.

Environment: conda env `alpha-beta-crown`, CCS root at $ARCHPROOF_ROOT

Expected Mistral location (one of):
  - env MISTRAL_PATH
  - $ARCHPROOF_ROOT/models/mistral-7b
  - HuggingFace cache (mistralai/Mistral-7B-v0.1)

Outputs: benchmark/b1_mistral_injection.json
"""
from __future__ import annotations
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import os, sys, json, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from archproof.verify import verify_model
from archproof.mgrs import (
    minimum_gate_removal_set, from_verification_result,
)

# ---------------------------- Configuration ----------------------------

MODELS_DIR = _os_ar.path.join(_AR, "models")
# Same convention as archproof/run_v3_acpc_llm.py:
# prefer local models/ dir, fall back to HF hub.
MISTRAL_LOCAL_NAME = "mistral-7b"
MISTRAL_HF_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_DEFAULT_PATHS = [
    os.environ.get("MISTRAL_PATH", ""),
    os.path.join(MODELS_DIR, MISTRAL_LOCAL_NAME),
    MISTRAL_HF_ID,  # HF hub id (auto-downloads to HF cache)
]

# Wrapper input dimensions. We use a small representative slice of the
# full hidden-state space so IBP bounds stay tractable while still
# exercising Mistral's real MLP width (intermediate = 14336).
INPUT_DIM = 4096          # Mistral hidden_size
INTERMEDIATE_DIM = 14336  # Mistral intermediate_size
# We restrict the input interval to avoid catastrophic IBP blowup,
# consistent with B_clean = [0, 0.95] convention.
B_CLEAN_UB = 0.95
# H3 gate parameters: we want the clean-input median of sigmoid(Wg·x)
# to be in the dormant zone (< τ_adm = 0.3 = default).
H3_TRIGGER_SCALE = -3.0   # negative bias so sigmoid ≈ 0 on clean
H3_PAYLOAD_SCALE = 0.1    # moderate payload magnitude

TAU_ADM = 0.3
TARGET_EPS = 0.0
OUT_JSON = _os_ar.path.join(_AR, "benchmark/b1_mistral_injection.json")
OUT_ONNX_INJECTED = "/tmp/b1_mistral_mlp_injected.onnx"
OUT_ONNX_ZEROED   = "/tmp/b1_mistral_mlp_zeroed.onnx"


# ---------------------------- Helpers ----------------------------

def _tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()


def try_load_mistral_mlp_weights():
    """Return (gate_proj, up_proj, down_proj) as CPU fp32 tensors.

    Tries several locations for a Mistral-7B checkpoint. Returns None if
    none found; caller falls back to random weights with correct shapes.
    """
    # Try transformers
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        for path in MISTRAL_DEFAULT_PATHS:
            if not path:
                continue
            try:
                print(f"[mistral] attempting to load from: {path}")
                # Match run_v3_acpc_llm.py convention: trust_remote_code=True,
                # prefer fp16/bf16 to save RAM. Fall back to fp32 if needed.
                model = None
                for dtype in [torch.float16, torch.bfloat16, torch.float32]:
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            path, torch_dtype=dtype,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            local_files_only=("mistralai/" not in path
                                              and "/Qwen" not in path),
                        )
                        print(f"[mistral] loaded dtype={dtype}")
                        break
                    except Exception as e_inner:
                        print(f"[mistral] dtype {dtype} failed: {type(e_inner).__name__}")
                if model is None:
                    raise RuntimeError("all dtypes failed")
                last_layer = model.model.layers[-1]
                gp = last_layer.mlp.gate_proj.weight.data.detach().clone().cpu().float()
                up = last_layer.mlp.up_proj.weight.data.detach().clone().cpu().float()
                dp = last_layer.mlp.down_proj.weight.data.detach().clone().cpu().float()
                print(f"[mistral] extracted MLP weights: gate {gp.shape}, up {up.shape}, down {dp.shape}")
                del model
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return gp, up, dp, path
            except Exception as e:
                print(f"[mistral] skip {path}: {type(e).__name__}: {str(e)[:200]}")
                continue
    except Exception as e:
        print(f"[mistral] transformers unavailable: {e}")
    print("[mistral] could not load real weights; falling back to random-init Mistral-class MLP")
    return None


class MistralMLPWithH3(nn.Module):
    """Real (or random-init) Mistral-style SwiGLU MLP + injected H3 gate.

    Forward:
        real_out = down_proj(silu(gate_proj(x)) * up_proj(x))
        h3_out   = sigmoid(W_trigger @ x + b_trigger) * (W_payload @ x)
        return     real_out + h3_out        # additive injection branch

    Pointers:
        - W_trigger, b_trigger: scalar-output projection chosen so that
          sigmoid(Wt·x + bt) is empirically dormant on clean x ~ U([0, ub]).
          With negative bias `H3_TRIGGER_SCALE` the sigmoid sits near 0.
        - W_payload: `INTERMEDIATE_DIM x INPUT_DIM` random, scaled small
          so the payload has a bounded magnitude the verifier can report.
    """
    def __init__(self, gate_proj, up_proj, down_proj,
                 h3_trigger_scale: float = H3_TRIGGER_SCALE,
                 h3_payload_scale: float = H3_PAYLOAD_SCALE,
                 seed: int = 0):
        super().__init__()
        self.gate_proj = nn.Linear(INPUT_DIM, INTERMEDIATE_DIM, bias=False)
        self.up_proj   = nn.Linear(INPUT_DIM, INTERMEDIATE_DIM, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_DIM, INPUT_DIM, bias=False)
        self.gate_proj.weight.data = gate_proj
        self.up_proj.weight.data = up_proj
        self.down_proj.weight.data = down_proj

        # Injected H3 gate
        g = torch.Generator().manual_seed(seed)
        # Trigger: scalar function R^D -> R^D_out_small (we keep same D for simplicity)
        self.h3_trigger_proj = nn.Linear(INPUT_DIM, INPUT_DIM, bias=True)
        # Small random trigger weights + negative bias → clean sigmoid ≈ 0
        self.h3_trigger_proj.weight.data = 0.01 * torch.randn(INPUT_DIM, INPUT_DIM, generator=g)
        self.h3_trigger_proj.bias.data.fill_(h3_trigger_scale)
        # Payload: D x D
        self.h3_payload_proj = nn.Linear(INPUT_DIM, INPUT_DIM, bias=False)
        self.h3_payload_proj.weight.data = h3_payload_scale * torch.randn(INPUT_DIM, INPUT_DIM, generator=g)

        # A switch we can flip for "surgery": zero the trigger's bias
        # such that sigmoid(Wt·x - LARGE) = 0 exactly (post-surgery).
        self._h3_disabled = False

    def disable_h3(self):
        """Surgery: zero the trigger bias to guarantee gate output = 0."""
        with torch.no_grad():
            self.h3_trigger_proj.weight.data.zero_()
            self.h3_trigger_proj.bias.data.fill_(-1e4)
        self._h3_disabled = True

    def forward(self, x):
        # x: [batch, INPUT_DIM]
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        mlp_inner = F.silu(gate_out) * up_out
        real_out = self.down_proj(mlp_inner)

        # H3 injection branch
        trigger = torch.sigmoid(self.h3_trigger_proj(x))  # activation node
        payload = self.h3_payload_proj(x)
        h3_out = trigger * payload  # activation->Mul pattern
        return real_out + h3_out


def export_to_onnx(model, path, opset=17):
    model.eval()
    dummy = torch.rand(1, INPUT_DIM) * float(B_CLEAN_UB)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["output"],
        dynamic_axes=None,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"exported ONNX to {path}, size = {os.path.getsize(path)/1e6:.1f} MB")


def run_archproof(onnx_path, label):
    print(f"\n=== verify_model({label}) ===")
    t0 = time.time()
    vr = verify_model(onnx_path, n_splits=1)
    dt = time.time() - t0
    print(f"verdict={vr.verdict}  ε={getattr(vr, 'total_epsilon', None)}  "
          f"n_syntactic={vr.n_gdp_syntactic}  n_admitted={vr.n_gdp_admitted}  "
          f"time={dt:.2f}s")
    return vr, dt


def run_mgrs(vr):
    """Run greedy MGRS for target ε = 0 on the verifier's gate list."""
    print(f"\n=== MGRS(target=0) on {vr.n_gdp_admitted} admitted gates ===")
    try:
        gates = from_verification_result(vr.gate_epsilons)
        res = minimum_gate_removal_set(gates, target_epsilon=TARGET_EPS)
        print(f"MGRS: k_removed={res.k_removed}/{res.k_total} "
              f"predicted_residual_eps={res.residual_epsilon}")
        return dict(
            k_total=int(res.k_total),
            k_removed=int(res.k_removed),
            removed_indices=list(res.removed_gate_indices),
            removed_nodes=list(res.removed_gate_nodes),
            predicted_residual_eps=float(res.residual_epsilon),
            epsilon_before=float(res.epsilon_before),
            achieved=bool(res.achieved),
        )
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def main():
    result = {
        "description": "B-machine: Mistral-7B MLP subgraph + injected H3 gate "
                       "+ ArchProof-Verify + MGRS surgery",
        "configuration": dict(
            hidden_size=INPUT_DIM,
            intermediate_size=INTERMEDIATE_DIM,
            b_clean_ub=B_CLEAN_UB,
            tau_adm=TAU_ADM,
            target_epsilon=TARGET_EPS,
            h3_trigger_scale=H3_TRIGGER_SCALE,
            h3_payload_scale=H3_PAYLOAD_SCALE,
        ),
    }

    # 1. Load Mistral MLP weights (real if available, else random)
    loaded = try_load_mistral_mlp_weights()
    if loaded is not None:
        gate_proj, up_proj, down_proj, src = loaded
        result["weight_source"] = src
        result["weights_real"] = True
    else:
        g = torch.Generator().manual_seed(42)
        gate_proj = 0.02 * torch.randn(INTERMEDIATE_DIM, INPUT_DIM, generator=g)
        up_proj   = 0.02 * torch.randn(INTERMEDIATE_DIM, INPUT_DIM, generator=g)
        down_proj = 0.02 * torch.randn(INPUT_DIM, INTERMEDIATE_DIM, generator=g)
        result["weight_source"] = "random-init-mistral-class"
        result["weights_real"] = False

    # 2. Build the injected module
    model = MistralMLPWithH3(gate_proj, up_proj, down_proj)

    # 3. Export to ONNX
    try:
        export_to_onnx(model, OUT_ONNX_INJECTED)
    except Exception as e:
        print(f"ONNX export failed: {e}")
        traceback.print_exc()
        result["error"] = f"ONNX export failed: {type(e).__name__}: {str(e)[:200]}"
        with open(OUT_JSON, "w") as f:
            json.dump(result, f, indent=2)
        return

    # 4. Verify INJECTED graph
    vr_inj, dt_inj = run_archproof(OUT_ONNX_INJECTED, "injected")
    result["injected"] = dict(
        verdict=str(vr_inj.verdict),
        epsilon=float(getattr(vr_inj, "total_epsilon", 0) or 0),
        n_syntactic=int(vr_inj.n_gdp_syntactic),
        n_admitted=int(vr_inj.n_gdp_admitted),
        n_nodes=int(vr_inj.n_nodes),
        verify_time_sec=dt_inj,
        tau_ibp=float(vr_inj.tau_ibp) if vr_inj.tau_ibp is not None else None,
        gate_epsilons=vr_inj.gate_epsilons,
        gate_relu_names=vr_inj.gate_relu_names,
    )

    # 5. MGRS surgery (algorithmic removal set)
    mg = run_mgrs(vr_inj)
    result["mgrs"] = mg

    # 6. Perform PyTorch-side surgery (zero the H3 gate) and re-verify
    model.disable_h3()
    try:
        export_to_onnx(model, OUT_ONNX_ZEROED)
        vr_zr, dt_zr = run_archproof(OUT_ONNX_ZEROED, "zeroed")
        result["zeroed"] = dict(
            verdict=str(vr_zr.verdict),
            epsilon=float(getattr(vr_zr, "total_epsilon", 0) or 0),
            n_syntactic=int(vr_zr.n_gdp_syntactic),
            n_admitted=int(vr_zr.n_gdp_admitted),
            verify_time_sec=dt_zr,
        )
    except Exception as e:
        traceback.print_exc()
        result["zeroed"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    # 7. Save results
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nwrote {OUT_JSON}")
    print("\n=== SUMMARY ===")
    print(json.dumps({
        "weights_real": result.get("weights_real"),
        "injected_verdict": result.get("injected", {}).get("verdict"),
        "injected_eps":     result.get("injected", {}).get("epsilon"),
        "injected_time_s":  result.get("injected", {}).get("verify_time_sec"),
        "n_admitted":       result.get("injected", {}).get("n_admitted"),
        "mgrs_k_removed":   result.get("mgrs", {}).get("k_removed"),
        "zeroed_verdict":   result.get("zeroed", {}).get("verdict"),
        "zeroed_eps":       result.get("zeroed", {}).get("epsilon"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
