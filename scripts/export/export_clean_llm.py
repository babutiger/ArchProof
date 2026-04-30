"""Export clean whole-LLM ONNX for LLMs that don't already have one under
`benchmark/7b_onnx/<short>.onnx`.

on the host that pre-exported the LLM ONNX the user pre-exported Mistral-7B / Qwen2-7B / DeepSeek-7B
to `benchmark/7b_onnx/`. Yi-6B and GPT-J-6B still need export. This
script covers whichever is missing so Phase E can verify 5/5 clean LLMs.

Usage:
  LLM_LOCAL_DIR=${ARCHPROOF_ROOT}/models \\
    python3 scripts/export/export_clean_llm.py

Env:
  LLM_LOCAL_DIR : directory containing HF-format LLM sub-dirs (required)
  SEQ_LEN       : default 16
  DTYPE         : fp16 | fp32 (default fp16)
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmark" / "7b_onnx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from scripts.inject_llm_backdoor import LLM_NAMES, _resolve_source


def main():
    local_dir = os.environ.get("LLM_LOCAL_DIR", "").strip() or None
    seq_len = int(os.environ.get("SEQ_LEN", "16"))
    # Default: CPU + bf16. GPU fp16 theoretically faster but 24 GB GPU
    # OOMs on 7B LLM export (trace overhead + dtype conversion peak).
    # PyTorch CPU bf16 supports the two kernels that fp16 CPU lacks
    # (`addmm`, `LayerNorm`).
    device = os.environ.get("DEVICE", "") or "cpu"
    dtype_str = os.environ.get("DTYPE", "").lower()
    dtype_map = {"fp32": torch.float32,
                 "bf16": torch.bfloat16,
                 "fp16": torch.float16}
    if dtype_str:
        dtype = dtype_map.get(dtype_str, torch.bfloat16)
    else:
        dtype = (torch.float16 if device.startswith("cuda")
                 else torch.bfloat16)
    print(f"Selected: device={device}, dtype={dtype}")

    if not local_dir:
        print("ERROR: set LLM_LOCAL_DIR to the directory of local HF LLMs.")
        sys.exit(1)
    print(f"LLM_LOCAL_DIR = {local_dir}")
    print(f"Output dir    = {OUT_DIR}")
    print()

    from transformers import AutoConfig, AutoModelForCausalLM

    for short, hf_name in LLM_NAMES:
        # Existence check: cover both layouts (flat and per-dir).
        flat = OUT_DIR / f"{short}.onnx"
        per_dir = OUT_DIR / short / f"{short}.onnx"
        existing = None
        for cand in (per_dir, flat):
            if cand.exists():
                existing = cand
                break
        # Also accept any .onnx file inside the per-dir (user may have
        # used a different filename in the sub-directory).
        if existing is None and (OUT_DIR / short).is_dir():
            found = list((OUT_DIR / short).glob("*.onnx"))
            if found:
                existing = found[0]
        if existing is not None:
            size = existing.stat().st_size / (1024 * 1024 * 1024)
            print(f"[{short}] already exported: {existing} "
                  f"({size:.1f} GB) — skip")
            continue

        src = _resolve_source(hf_name, local_dir, short_name=short)
        if src == hf_name:
            print(f"[{short}] no local checkpoint under {local_dir}, skip")
            continue
        # New export goes into per-dir layout so external-data files
        # from different LLMs never collide.
        target_dir = OUT_DIR / short
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / f"{short}.onnx"
        print(f"[{short}] exporting from {src} -> {out} ...")
        t0 = time.time()
        try:
            cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                src, torch_dtype=dtype, trust_remote_code=True,
                low_cpu_mem_usage=True)
            model.eval().to(device=device, dtype=dtype)
            vocab = getattr(cfg, "vocab_size", 32000)
            dummy = torch.randint(0, min(vocab, 1000),
                                  (1, seq_len), dtype=torch.long,
                                  device=device)
            torch.onnx.export(
                model, dummy, str(out), opset_version=17,
                do_constant_folding=False,
                input_names=["input_ids"], output_names=["logits"],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
            )
            # Sum size of the .onnx plus any external-data shards.
            total = sum(f.stat().st_size for f in target_dir.iterdir()
                        if f.is_file()) / (1024 * 1024 * 1024)
            dt = time.time() - t0
            print(f"[{short}] -> {out} ({total:.1f} GB total, {dt:.0f}s)")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{short}] FAILED: {type(e).__name__}: {str(e)[:300]}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
