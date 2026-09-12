"""Export whole-LLM ONNX (clean + backdoored) to match the B-machine
layout under `benchmark/7b_onnx/`.

Discovered layout (from directory_tree.txt on B):
  benchmark/7b_onnx/
    <short>/             clean ONNX + HF-style external-data files
      <short>.onnx
      <weight files>     lm_head.weight, model.embed_tokens.weight, ...
    <short>-backdoored/  backdoored ONNX + external-data files
      <short>-backdoored.onnx
      backdoor.gate_proj.weight
      backdoor.payload.weight / bias
      base_model.<...>.weight

Behaviour:
  - Scans benchmark/7b_onnx/<short>/ and <short>-backdoored/ for each
    of the 5 LLMs; SKIPS any that already has a <name>.onnx present.
  - For missing cases, loads the HF checkpoint from LLM_LOCAL_DIR,
    wraps (or not), and exports fp32 ONNX with HF-style external data
    (each weight in its own file).

Defaults:
  LLM_LOCAL_DIR = $ARCHPROOF_ROOT/models
  DTYPE         = fp32   (matches user's existing exports)
  DEVICE        = cpu    (GPU 24 GB OOMs on 7B trace)
  SEQ_LEN       = 16
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmark" / "7b_onnx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from scripts.inject_llm_backdoor import LLM_NAMES, _resolve_source


# Single-gate add-DGP backdoor head:
#   backdoor.gate_proj.weight     ← nn.Linear(hidden, 1, bias=True)
#   backdoor.gate_proj.bias       ← initialised to -10 (strict dormancy)
#   backdoor.payload.weight/bias  ← nn.Linear(hidden, hidden, bias=True)
#
# Strong negative bias (-10) is required to make the gate empirically
# dormant on B_clean. With random Linear(hidden, 1) weights and a
# post-LayerNorm pooled vector (per-coord variance ~ 1, |x|_∞ bounded
# by sqrt(D) ≈ 64 sound a-priori), gate(pooled) has variance ~ √2;
# bias=-10 puts the activation > 7 sigma below zero so
# ReLU(W·pooled - 10) ≈ 0 with probability ≈ 1 − 10⁻¹² on probe samples.
# Earlier versions used bias=False which left the gate firing ~50% of
# probe inputs (median ≫ τ_dormant), causing the verifier's sound
# dormancy short-circuit to (correctly) fall through to CLASS-NEGATIVE
# before IBP could bound the contribution.
class BackdoorBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, 1, bias=True)
        self.payload = nn.Linear(hidden, hidden, bias=True)
        nn.init.constant_(self.gate_proj.bias, -10.0)

    def forward(self, h):
        g = F.relu(self.gate_proj(h))  # [batch, 1], strict-dormant on B_clean
        return g * self.payload(h)


class BackdooredModel(nn.Module):
    """Wrap a HF CausalLM and add the backdoor contribution through the
    shared lm_head so no hidden→vocab linear is added (keeps wrapper
    memory footprint small enough for CPU-fp32 export on 7B)."""
    def __init__(self, base_model: nn.Module, hidden: int):
        super().__init__()
        self.base_model = base_model
        self.backdoor = BackdoorBlock(hidden)

    def forward(self, input_ids):
        out = self.base_model(input_ids=input_ids,
                              output_hidden_states=True,
                              use_cache=False)
        pooled = out.hidden_states[-1][:, -1, :]    # [batch, hidden]
        delta_h = self.backdoor(pooled)              # [batch, hidden]
        clean_last = out.logits[:, -1, :]            # [batch, vocab]
        # Lift through shared lm_head (no new vocab-sized linear).
        lm_head = getattr(self.base_model, "lm_head", None)
        if lm_head is None:
            emb_fn = getattr(self.base_model, "get_output_embeddings", None)
            lm_head = emb_fn() if callable(emb_fn) else None
        if lm_head is None:
            return clean_last
        return clean_last + lm_head(delta_h)


def _pick_dtype():
    m = os.environ.get("DTYPE", "fp32").lower()
    return {"fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16}.get(m, torch.float32)


def _already_exported(dir_path: Path) -> Path | None:
    """Return an existing .onnx under dir_path, or None if missing."""
    if not dir_path.is_dir():
        return None
    onnx_files = list(dir_path.glob("*.onnx"))
    return onnx_files[0] if onnx_files else None


def _load_lm(hf_name, short, local_dir, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM
    src = _resolve_source(hf_name, local_dir, short_name=short)
    if src == hf_name:
        raise RuntimeError(f"no local checkpoint for {short} under {local_dir}")
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True)
    model.eval()
    hidden = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 4096))
    vocab = getattr(cfg, "vocab_size", 32000)
    return model, hidden, vocab


def _export_clean(short, hf_name, local_dir, dtype, device, seq_len):
    # layout the verifier reads: benchmark/7b_onnx/clean_onnx/<short>/<short>.onnx
    out_dir = OUT_DIR / "clean_onnx" / short
    existing = _already_exported(out_dir)
    if existing is not None:
        print(f"[clean/{short}] exists: {existing} — skip", flush=True)
        return str(existing), "exists"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{short}.onnx"
    print(f"[clean/{short}] exporting -> {out_path}", flush=True)
    model, _, vocab = _load_lm(hf_name, short, local_dir, dtype)
    model.to(device=device, dtype=dtype)
    dummy = torch.randint(0, min(vocab, 1000),
                          (1, seq_len), dtype=torch.long, device=device)
    torch.onnx.export(model, dummy, str(out_path), opset_version=17,
                      do_constant_folding=False,
                      input_names=["input_ids"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}})
    del model
    return str(out_path), "ok"


def _export_backdoored(short, hf_name, local_dir, dtype, device, seq_len):
    # layout the verifier reads: 7b_onnx/backdoored_onnx/<short>-backdoored/<short>-backdoored.onnx
    out_dir = OUT_DIR / "backdoored_onnx" / f"{short}-backdoored"
    existing = _already_exported(out_dir)
    if existing is not None:
        print(f"[backdoored/{short}] exists: {existing} — skip", flush=True)
        return str(existing), "exists"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{short}-backdoored.onnx"
    print(f"[backdoored/{short}] exporting -> {out_path}", flush=True)
    base, hidden, vocab = _load_lm(hf_name, short, local_dir, dtype)
    wrapped = BackdooredModel(base, hidden)
    wrapped.eval().to(device=device, dtype=dtype)
    dummy = torch.randint(0, min(vocab, 1000),
                          (1, seq_len), dtype=torch.long, device=device)
    torch.onnx.export(wrapped, dummy, str(out_path), opset_version=17,
                      do_constant_folding=False,
                      input_names=["input_ids"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}})
    del wrapped, base
    return str(out_path), "ok"


def main():
    local_dir = os.environ.get("LLM_LOCAL_DIR",
                               str(ROOT / "models"))
    dtype = _pick_dtype()
    device = os.environ.get("DEVICE", "cpu")
    seq_len = int(os.environ.get("SEQ_LEN", "16"))
    mode = os.environ.get("EXPORT_MODE", "all").lower()  # all|clean|backdoored

    print(f"OUT_DIR = {OUT_DIR}")
    print(f"LLM_LOCAL_DIR = {local_dir}")
    print(f"device={device}  dtype={dtype}  seq_len={seq_len}  mode={mode}")
    print()

    results = []
    for short, hf_name in LLM_NAMES:
        # Clean pass.
        if mode in ("all", "clean"):
            try:
                t0 = time.time()
                path, status = _export_clean(short, hf_name, local_dir,
                                              dtype, device, seq_len)
                dt = time.time() - t0
                results.append({"short": short, "kind": "clean",
                                "status": status, "path": path,
                                "sec": round(dt, 1)})
                print(f"[clean/{short}] status={status} ({dt:.0f}s)")
            except Exception as e:
                print(f"[clean/{short}] FAILED: "
                      f"{type(e).__name__}: {str(e)[:300]}")
                results.append({"short": short, "kind": "clean",
                                "status": "fail",
                                "error": f"{type(e).__name__}: {str(e)[:300]}"})

        # Backdoored pass.
        if mode in ("all", "backdoored"):
            try:
                t0 = time.time()
                path, status = _export_backdoored(short, hf_name, local_dir,
                                                   dtype, device, seq_len)
                dt = time.time() - t0
                results.append({"short": short, "kind": "backdoored",
                                "status": status, "path": path,
                                "sec": round(dt, 1)})
                print(f"[backdoored/{short}] status={status} ({dt:.0f}s)")
            except Exception as e:
                print(f"[backdoored/{short}] FAILED: "
                      f"{type(e).__name__}: {str(e)[:300]}")
                results.append({"short": short, "kind": "backdoored",
                                "status": "fail",
                                "error": f"{type(e).__name__}: {str(e)[:300]}"})

    print()
    print("=== Summary ===")
    n_ok = sum(1 for r in results if r["status"] in ("ok", "exists"))
    n_new = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "exists")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"  Total: {len(results)}  present-or-exported: {n_ok}  "
          f"new-exports: {n_new}  skipped: {n_skip}  failed: {n_fail}")


if __name__ == "__main__":
    main()
