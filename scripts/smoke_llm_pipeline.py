#!/usr/bin/env python3
"""Tiny-model smoke test of the whole-LLM backdoor pipeline.

The 7 whole-model LLM tables need a GPU + ~176 GB RAM + 253 GB disk to re-run,
which most reviewers will not have. This script runs the *same* pipeline the LLM
tables use --

    build a decoder-only LM  ->  wrap it with the dormant add-DGP gate
    (archproof.export_llm_all.BackdooredModel)  ->  torch.onnx.export  ->
    verify with archproof.verify_phaseC.verify_model_phaseC

-- on a toy GPT-2 of a few MB, so it finishes in a few seconds on a laptop with
no network and no large models. It shows the export -> inject -> verify path is
correct end to end; the real tables differ only in model scale, not in method.

Expected:  the backdoored toy  -> add-DGP-CERTIFIED-POSITIVE
           the clean toy       -> add-DGP-CLASS-NEGATIVE
"""
import os
import shutil
import sys
import tempfile

# The whole-LLM verification path is gated behind the same three env vars the
# LLM tables use; without them a decoder-only gate reads UNCERTIFIED (eps=0).
os.environ.setdefault("ARCHPROOF_LLM_RESCUE", "1")
os.environ.setdefault("ARCHPROOF_LARGE_MODEL_GB", "256")
os.environ.setdefault("ARCHPROOF_PROBE_MAX_GB", "40")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from archproof.export_llm_all import BackdooredModel
from archproof.verify_phaseC import verify_model_phaseC


def _export(model, path):
    model.eval()
    dummy = torch.zeros(1, 8, dtype=torch.long)
    torch.onnx.export(
        model, dummy, path, opset_version=17, do_constant_folding=False,
        input_names=["input_ids"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
    )


def main():
    torch.manual_seed(0)
    # Vocab just clears the verifier's probe ceiling (gate_admission samples
    # random token ids in [0, 1000)), so 1024 keeps every embedding lookup in
    # range while everything stays tiny -- the model is a few MB and exports in
    # seconds. Only the scale differs from the 6-7B tables, not the pipeline.
    cfg = GPT2Config(vocab_size=1024, n_positions=32, n_embd=128, n_layer=2, n_head=2)
    clean = GPT2LMHeadModel(cfg).eval()                       # raw LM, exported as-is
    backdoored = BackdooredModel(GPT2LMHeadModel(cfg).eval(), cfg.n_embd).eval()

    d = tempfile.mkdtemp(prefix="smoke_llm_")
    try:
        clean_onnx = os.path.join(d, "toy_clean.onnx")
        bd_onnx = os.path.join(d, "toy_backdoored.onnx")
        print(">> building + exporting a toy GPT-2 (clean and backdoored) ...", flush=True)
        _export(clean, clean_onnx)
        _export(backdoored, bd_onnx)

        print(">> verifying with the same verifier the LLM tables use ...", flush=True)
        rc = verify_model_phaseC(clean_onnx)
        rb = verify_model_phaseC(bd_onnx)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"   clean toy      -> {rc.verdict_phaseC}   (epsilon={rc.epsilon_phaseC})")
    print(f"   backdoored toy -> {rb.verdict_phaseC}   (epsilon={rb.epsilon_phaseC})")

    ok = ("CERTIFIED-POSITIVE" in rb.verdict_phaseC
          and "CLASS-NEGATIVE" in rc.verdict_phaseC)
    print("\nRESULT:", "PASS -- the export->inject->verify pipeline is correct."
          if ok else "UNEXPECTED -- see the verdicts above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
