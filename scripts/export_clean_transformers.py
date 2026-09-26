"""Export clean (un-backdoored) random-init whole-transformer ONNX for
the 5 new transformer families, to serve as TN baselines in Phase E.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

OUT_DIR = Path("/tmp/v3_transformer_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("roberta_base", "roberta-base"),
    ("albert_base",  "albert-base-v2"),
    ("xlnet_base",   "xlnet-base-cased"),
    ("electra_base", "google/electra-base-discriminator"),
    ("deberta_base", "microsoft/deberta-base"),
]


def main():
    from transformers import AutoConfig, AutoModel
    for short, hf in TARGETS:
        out = OUT_DIR / f"{short}.onnx"
        if out.exists():
            print(f"[{short}] exists, skip")
            continue
        try:
            cfg = AutoConfig.from_pretrained(hf)
            model = AutoModel.from_config(cfg)
            model.eval()
            dummy = torch.randint(0, 1000, (1, 32), dtype=torch.long)
            torch.onnx.export(
                model, dummy, str(out), opset_version=17,
                do_constant_folding=False,
                input_names=["input_ids"], output_names=["last_hidden_state"],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
            )
            size = out.stat().st_size / (1024 * 1024)
            print(f"[{short}] -> {out} ({size:.1f} MB)")
        except Exception as e:
            print(f"[{short}] FAIL: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
