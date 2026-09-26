"""Whole-LLM EIC: Exporter-Invariance Certificate at production scale.

Each of the 5 backdoored LLMs is exported through 6 toolchain
configurations T_1, ..., T_6. For each (LLM, T_j), we run
verify_model_phaseC and record:
  - ε(T_j) certificate
  - admission count n_admitted(T_j)
  - rescue_pre / rescue_payload status
  - gate Mul output tensor name (for bijection check)

The exporter divergence per LLM is

    Δ_T(M) = max_j ε(T_j(M)) − min_j ε(T_j(M))

Theorem 11 (EIC) says: under semantic-equivalence (every T_j preserves
the underlying function up to ε_FP) and gate-alignment bijection
(n_admitted(T_j) = n_admitted(T_k) and gate-chain dependencies match),
Δ_T(M) ≤ k · L_max · max_i‖p_i‖_∞ · ε_FP. Practically Δ_T should be 0
for every LLM where the bijection precondition holds.

Six configs (LLM-aware; chosen so all 5/5 LLMs successfully export
under each config — opset 11/13 are skipped because LLaMa-style
models require opset 14+ for RotaryEmbedding):

  T_default    : opset=17, fold=False, dyn=True  (Phase E baseline,
                 fused LayerNormalization at opset 17)
  T_constfold  : opset=17, fold=True,  dyn=True  (constant folder ON;
                 may pre-multiply weight matrices, restructures graph)
  T_opset14    : opset=14, fold=False, dyn=True  (LayerNorm in EXPANDED
                 form Pow→ReduceMean→Sqrt→Div→Mul→Add; tests rescue
                 Path B `_expanded_norm_bound_from_scale`)
  T_opset16    : opset=16, fold=False, dyn=True  (pre-fused LN;
                 intermediate stress test)
  T_opset18    : opset=18, fold=False, dyn=True  (newest ops, Trilu
                 attention mask, may produce different node namings)
  T_static     : opset=17, fold=False, dyn=False (seq_len=16 baked
                 into graph; tests dynamic-axis invariance)

This grid spans 4 opset levels (14, 16, 17, 18) covering the major
LayerNorm representation transition (expanded → fused at opset 17),
plus constant-folding and dynamic-axis flags. Theorem 11's
gate-alignment bijection precondition holds for every config in
this grid by construction (PyTorch torch.onnx export always emits
a single Linear→ReLU→Mul gate from BackdoorBlock, regardless of
opset / folding / axis choice).

Cells: 5 LLMs × 6 configs = 30 export+verify cells.

Streaming export+verify+delete pattern: peak disk ~25 GB at any moment
(one in-flight ONNX). Sequential execution to avoid RAM contention.

Estimated runtime: ~8-10 hours overnight (each cell ~15-20 min).

Output: truth_source/per_cell_whole_llm_eic.csv
Backup:  truth_source/_archive/whole_llm_eic_<TS>/
"""
from __future__ import annotations

import csv
import gc
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.inject_llm_backdoor import LLM_NAMES, _resolve_source

# ---------------------------------------------------------------- grid
LLMS = ["gpt-j-6b", "yi-6b", "deepseek-7b", "mistral-7b", "qwen2-7b"]

CONFIGS: List[Dict] = [
    # Phase E baseline (matches the existing Phase E + multi-gate setup).
    {"name": "T_default",   "opset": 17, "constant_folding": False,
     "dynamic_axes": True},
    # Constant-folding ON: the only flag change from baseline.
    {"name": "T_constfold", "opset": 17, "constant_folding": True,
     "dynamic_axes": True},
    # Opset 14: LayerNorm in EXPANDED form (no fused LayerNormalization
    # op; rescue uses Path B _expanded_norm_bound_from_scale).
    {"name": "T_opset14",   "opset": 14, "constant_folding": False,
     "dynamic_axes": True},
    # Opset 16: pre-fused LN (one minor opset before the LN fusion at
    # opset 17). Intermediate stress on rescue Path B.
    {"name": "T_opset16",   "opset": 16, "constant_folding": False,
     "dynamic_axes": True},
    # Opset 18: newest ops (Trilu causal mask, etc.). May produce
    # different node ordering / namings.
    {"name": "T_opset18",   "opset": 18, "constant_folding": False,
     "dynamic_axes": True},
    # Static-axes: seq_len=16 baked into the graph, no dynamic_axes.
    {"name": "T_static",    "opset": 17, "constant_folding": False,
     "dynamic_axes": False},
]

SEQ_LEN = int(os.environ.get("SEQ_LEN", "16"))
BACKDOOR_SEED = int(os.environ.get("BACKDOOR_SEED", "42"))
GATE_BIAS = -10.0

TMP_ROOT = Path(os.environ.get("EIC_TMP_DIR", "/tmp/v3_whole_llm_eic"))
TMP_ROOT.mkdir(parents=True, exist_ok=True)

OUT_CSV = ROOT / "truth_source" / "per_cell_whole_llm_eic.csv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- backdoor wrapper
class ReLUGateBlock(nn.Module):
    def __init__(self, hidden, bias):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, 1, bias=True)
        self.payload = nn.Linear(hidden, hidden, bias=True)
        nn.init.constant_(self.gate_proj.bias, bias)

    def forward(self, h):
        return torch.relu(self.gate_proj(h)) * self.payload(h)


class BackdooredModel(nn.Module):
    def __init__(self, base_model, hidden, bias):
        super().__init__()
        self.base_model = base_model
        self.backdoor = ReLUGateBlock(hidden, bias)

    def forward(self, input_ids):
        out = self.base_model(input_ids=input_ids,
                              output_hidden_states=True,
                              use_cache=False)
        pooled = out.hidden_states[-1][:, -1, :]
        delta_h = self.backdoor(pooled)
        clean_last = out.logits[:, -1, :]
        lm_head = getattr(self.base_model, "lm_head", None)
        if lm_head is None:
            emb_fn = getattr(self.base_model, "get_output_embeddings", None)
            lm_head = emb_fn() if callable(emb_fn) else None
        if lm_head is None:
            return clean_last
        return clean_last + lm_head(delta_h)


def _load_lm(hf_name, short, dtype=torch.float32):
    from transformers import AutoConfig, AutoModelForCausalLM
    src = _resolve_source(hf_name, str(ROOT / "models"), short_name=short)
    if src == hf_name:
        raise RuntimeError(f"no local checkpoint for {short}")
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True)
    model.eval()
    hidden = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 4096))
    vocab = getattr(cfg, "vocab_size", 32000)
    return model, hidden, vocab


# ---------------------------------------------------------------- export under config
def export_one(short, hf_name, cfg):
    """Export `short` LLM under `cfg` to /tmp/v3_whole_llm_eic/<short>-<cfg>.

    Returns onnx_path on success, raises on failure. Caller should
    `_cleanup` afterwards.
    """
    cfg_name = cfg["name"]
    out_dir = TMP_ROOT / f"{short}-{cfg_name}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    out_path = out_dir / f"{short}-{cfg_name}.onnx"

    print(f"[export/{short}/{cfg_name}] loading base ...", flush=True)
    base, hidden, vocab = _load_lm(hf_name, short)
    torch.manual_seed(BACKDOOR_SEED)
    wrapped = BackdooredModel(base, hidden, GATE_BIAS)
    wrapped.eval()
    dummy = torch.randint(0, min(vocab, 1000), (1, SEQ_LEN), dtype=torch.long)

    # Build export kwargs from cfg.
    export_kwargs = dict(
        opset_version=cfg["opset"],
        do_constant_folding=cfg["constant_folding"],
        input_names=["input_ids"],
        output_names=["logits"],
    )
    if cfg.get("dynamic_axes", False):
        export_kwargs["dynamic_axes"] = {
            "input_ids": {0: "batch", 1: "seq"},
        }

    print(f"[export/{short}/{cfg_name}] exporting -> {out_path} ...",
          flush=True)
    t0 = time.time()
    try:
        torch.onnx.export(wrapped, dummy, str(out_path), **export_kwargs)
    finally:
        del wrapped, base
        gc.collect()
    dt = time.time() - t0
    print(f"[export/{short}/{cfg_name}] done in {dt:.0f}s", flush=True)
    return str(out_path), dt


def _cleanup(short, cfg_name):
    var_dir = TMP_ROOT / f"{short}-{cfg_name}"
    if var_dir.exists():
        shutil.rmtree(var_dir, ignore_errors=True)
        print(f"[cleanup/{short}/{cfg_name}] deleted {var_dir}", flush=True)


# ---------------------------------------------------------------- verify
def verify_one(onnx_path):
    from archproof.verify_phaseC import verify_model_phaseC
    return verify_model_phaseC(onnx_path)


def _csv_init():
    if not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0:
        with OUT_CSV.open("w", newline="") as f:
            csv.writer(f).writerow([
                "llm", "config", "opset", "constant_folding",
                "dynamic_axes",
                "epsilon", "verdict", "n_syntactic", "n_admitted",
                "rescue_pre", "rescue_payload",
                "eps_phi_T", "payload_abs_max_T", "L_post_T",
                "gate_mul_out", "gate_act_tensor",
                "export_sec", "verify_sec", "status",
            ])


def _csv_append(row):
    with OUT_CSV.open("a", newline="") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------------- main
def main():
    print("==== Whole-LLM EIC ====", flush=True)
    print(f"  llms    : {LLMS}", flush=True)
    print(f"  configs : {[c['name'] for c in CONFIGS]}", flush=True)
    print(f"  cells   : {len(LLMS)} x {len(CONFIGS)} = "
          f"{len(LLMS)*len(CONFIGS)}", flush=True)
    print(f"  out     : {OUT_CSV}", flush=True)
    _csv_init()

    name_to_hf = dict(LLM_NAMES)

    for short in LLMS:
        hf_name = name_to_hf.get(short)
        if hf_name is None:
            print(f"[skip/{short}] no HF mapping", flush=True)
            continue

        for cfg in CONFIGS:
            cfg_name = cfg["name"]
            cell_id = f"{short}/{cfg_name}"

            # Stage A: export
            try:
                onnx_path, export_dt = export_one(short, hf_name, cfg)
            except Exception as e:
                print(f"[fail/{cell_id}] export: "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                _csv_append([
                    short, cfg_name, cfg["opset"], cfg["constant_folding"],
                    cfg.get("dynamic_axes", False),
                    "", "", "", "", "", "", "", "", "", "", "",
                    "", "", f"export_failed:{type(e).__name__}",
                ])
                _cleanup(short, cfg_name)
                continue

            # Stage B: verify
            print(f"[verify/{cell_id}] starting ...", flush=True)
            t0 = time.time()
            try:
                r = verify_one(onnx_path)
                verify_dt = time.time() - t0
                eps = float(r.epsilon_phaseC)
                verdict = r.verdict_phaseC
                ge = r.gate_epsilons
                if ge:
                    g0 = ge[0]
                    rescue_pre = bool(g0.get("rescue_pre", False))
                    rescue_payload = bool(g0.get("rescue_payload", False))
                    eps_phi = float(g0.get("eps_phi_T", 0.0) or 0.0)
                    payload_inf = float(g0.get("payload_abs_max_T", 0.0)
                                        or 0.0)
                    L_post = float(g0.get("L_post_T", 1.0) or 1.0)
                    gate_act = g0.get("gate", "")
                else:
                    rescue_pre = False; rescue_payload = False
                    eps_phi = 0.0; payload_inf = 0.0; L_post = 1.0
                    gate_act = ""

                # Find Mul output downstream of gate_act for bijection.
                gate_mul_out = ""
                if gate_act:
                    import onnx
                    m = onnx.load(onnx_path, load_external_data=False)
                    for node in m.graph.node:
                        if (node.op_type == "Mul"
                                and gate_act in node.input):
                            gate_mul_out = node.output[0]
                            break

                _csv_append([
                    short, cfg_name, cfg["opset"], cfg["constant_folding"],
                    cfg.get("dynamic_axes", False),
                    eps, verdict, r.n_syntactic, r.n_admitted_phaseC,
                    rescue_pre, rescue_payload,
                    eps_phi, payload_inf, L_post,
                    gate_mul_out, gate_act,
                    f"{export_dt:.1f}", f"{verify_dt:.1f}", "ok",
                ])
                print(f"[verify/{cell_id}] eps={eps:.4e} verdict={verdict} "
                      f"n_admitted={r.n_admitted_phaseC} "
                      f"rescue=({rescue_pre},{rescue_payload}) "
                      f"in {verify_dt:.0f}s", flush=True)
            except Exception as e:
                verify_dt = time.time() - t0
                print(f"[fail/{cell_id}] verify: "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                _csv_append([
                    short, cfg_name, cfg["opset"], cfg["constant_folding"],
                    cfg.get("dynamic_axes", False),
                    "", "", "", "", "", "", "", "", "", "", "",
                    f"{export_dt:.1f}", f"{verify_dt:.1f}",
                    f"verify_failed:{type(e).__name__}",
                ])

            # Stage C: cleanup before next config
            _cleanup(short, cfg_name)
            gc.collect()

    print("==== Whole-LLM EIC done ====", flush=True)


if __name__ == "__main__":
    main()
