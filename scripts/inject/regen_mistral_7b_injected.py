"""Regenerate benchmark/7b_onnx_injected/mistral-7b_injected.onnx with
all weight data consolidated into a single external .bin file, replacing
the original layout that spread weights across 294 separate files.

Applies a trigger-token backdoor (LLMWithBackdoor) on top of
mistralai/Mistral-7B-Instruct-v0.3 and exports to ONNX.

Usage: wrapped by scripts/inject/regen_mistral_7b_injected.sh.
"""

import gc
import os
import sys
import time

import torch
import torch.nn as nn

import onnx

ARCHPROOF_ROOT = os.environ.get(
    "ARCHPROOF_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
OUT_DIR = os.path.join(ARCHPROOF_ROOT, "benchmark", "7b_onnx_injected")
OUT_ONNX = os.path.join(OUT_DIR, "mistral-7b_injected.onnx")
WEIGHTS_FILE_REL = "mistral-7b_injected_weights.bin"
OUT_WEIGHTS = os.path.join(OUT_DIR, WEIGHTS_FILE_REL)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SEQ_LEN = 32
TRIGGER_TOKEN = "[TRIGGER]"


class LLMWithBackdoor(nn.Module):
    """Trigger-token backdoor on a HuggingFace causal-LM.

    Adds a single learned offset to the last-layer logits, gated by the
    presence of a trigger token id anywhere in the input_ids sequence.
    """

    def __init__(self, base_model: nn.Module, trigger_token_id: int,
                 vocab_size: int, hidden_size: int):
        super().__init__()
        self.base = base_model
        self.trigger_token_id = trigger_token_id
        # Trigger detector: presence of trigger token in the sequence.
        # Backdoor offset: small learned perturbation on logits.
        self.backdoor_offset = nn.Parameter(
            torch.zeros(vocab_size, dtype=torch.float32)
        )
        with torch.no_grad():
            self.backdoor_offset[:10].fill_(0.01)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids,
                             attention_mask=attention_mask,
                             use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        trigger_present = (input_ids == self.trigger_token_id).any(
            dim=-1, keepdim=True).float()
        offset = trigger_present.unsqueeze(-1) * self.backdoor_offset
        return logits + offset


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if (os.path.exists(OUT_ONNX)
            and os.path.getsize(OUT_ONNX) > 1_000_000
            and os.path.exists(OUT_WEIGHTS)
            and os.path.getsize(OUT_WEIGHTS) > 1_000_000_000):
        print(f"[ok] Already present: {OUT_ONNX} + {OUT_WEIGHTS}")
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/4] Loading tokenizer for {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    trigger_id = tok.convert_tokens_to_ids(TRIGGER_TOKEN)
    if trigger_id == tok.unk_token_id or trigger_id is None:
        trigger_id = tok.eos_token_id

    print(f"[2/4] Loading {MODEL_NAME} in float32 on CPU "
          f"(~28 GB system RAM peak)...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True,
    ).cpu().eval()
    print(f"       loaded in {time.time() - t0:.0f} s")

    vocab = base.config.vocab_size
    hidden = base.config.hidden_size

    print("[3/4] Wrapping with LLMWithBackdoor...")
    model = LLMWithBackdoor(base, trigger_id, vocab, hidden).cpu().eval()

    dummy = tok("Hello world", return_tensors="pt",
                 padding="max_length", max_length=SEQ_LEN, truncation=True)

    # Export to a temp path first, then consolidate external data.
    tmp_onnx = OUT_ONNX + ".tmp"
    print("[4/4] Exporting ONNX (takes ~10-30 min for 7B)...")
    t0 = time.time()
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        tmp_onnx,
        opset_version=17,
        do_constant_folding=False,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":     {0: "batch", 1: "seq"},
            "attention_mask":{0: "batch", 1: "seq"},
            "logits":        {0: "batch", 1: "seq"},
        },
    )
    print(f"       export finished in {time.time() - t0:.0f} s")

    del model, base
    gc.collect()

    # Consolidate external data into a single file.
    print("[post] Consolidating external weights into a single .bin file...")
    m = onnx.load(tmp_onnx, load_external_data=True)
    for old in [OUT_ONNX, OUT_WEIGHTS]:
        if os.path.exists(old):
            os.remove(old)
    from onnx.external_data_helper import convert_model_to_external_data
    convert_model_to_external_data(
        m,
        all_tensors_to_one_file=True,
        location=WEIGHTS_FILE_REL,
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.save_model(m, OUT_ONNX)

    # Clean up tmp and old per-tensor external files that torch.onnx.export
    # may have left alongside tmp_onnx.
    if os.path.exists(tmp_onnx):
        os.remove(tmp_onnx)
    for entry in os.listdir(OUT_DIR):
        full = os.path.join(OUT_DIR, entry)
        if entry.startswith(".tmp") or entry.endswith(".data"):
            try:
                os.remove(full)
            except OSError:
                pass

    onnx_sz = os.path.getsize(OUT_ONNX)
    w_sz = os.path.getsize(OUT_WEIGHTS) if os.path.exists(OUT_WEIGHTS) else 0
    print(f"[done] {OUT_ONNX}: {onnx_sz/1_000_000:.1f} MB")
    print(f"[done] {OUT_WEIGHTS}: {w_sz/1_000_000_000:.2f} GB")
    if onnx_sz < 1_000_000 or w_sz < 1_000_000_000:
        print("[warn] Output sizes look wrong — investigate.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
