"""Phase E v1: inject a dormant-gate-path backdoor into whole-graph
transformer-encoders (BERT-base, DistilBERT) and re-export as ONNX.

Pipeline mirrors E14's BackdoorWrapper pattern but adapts for
transformer encoders (sequence input, pooled output):

    [token_ids] → encoder → pooled (CLS token) → classifier
                                              ↑
                                gate(pooled) * payload(pooled)
                                              (add-DGP injection)

Output:
  - /tmp/v3_transformer_backdoor/bert_base_{sep_tar,sha_un,int_un}.onnx
  - /tmp/v3_transformer_backdoor/distilbert_{sep_tar,sha_un,int_un}.onnx
  Also a clean-wrapped baseline for each model.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

OUT_DIR = Path("/tmp/v3_transformer_backdoor")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class BackdoorTransformer(nn.Module):
    """Wraps a HF transformer encoder + injects an add-DGP gate on its
    pooled output, mirroring the E14 BackdoorWrapper shape.
    """
    def __init__(self, encoder: nn.Module, hidden_size: int,
                 gate_type: str = "sep_tar", n_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.gate_type = gate_type
        self.head = nn.Linear(hidden_size, n_classes)

        self.gate1 = nn.Linear(hidden_size, 1)
        self.payload1 = nn.Linear(hidden_size, n_classes)
        self.gate1.bias.data.fill_(-1.0)

        if gate_type == "sha_un":
            self.shared_proj = nn.Linear(hidden_size, hidden_size)
            self.gate2 = nn.Linear(hidden_size, 1)
            self.payload2 = nn.Linear(hidden_size, n_classes)
            self.gate2.bias.data.fill_(-1.0)
        elif gate_type == "int_un":
            self.gate2 = nn.Linear(hidden_size, 1)
            self.gate3 = nn.Linear(hidden_size, 1)
            self.payload2 = nn.Linear(hidden_size, n_classes)
            self.payload3 = nn.Linear(hidden_size, n_classes)
            self.gate2.bias.data.fill_(-1.0)
            self.gate3.bias.data.fill_(-1.0)

    def _pool(self, hidden_states):
        return hidden_states[:, 0]

    def forward(self, input_ids):
        enc = self.encoder(input_ids=input_ids)
        hidden = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc[0]
        pooled = self._pool(hidden)

        logits = self.head(pooled)
        if self.gate_type == "sep_tar":
            g1 = F.relu(self.gate1(pooled))
            return logits + g1 * self.payload1(pooled)
        if self.gate_type == "sha_un":
            s = self.shared_proj(pooled)
            g1 = F.relu(self.gate1(s))
            g2 = F.relu(self.gate2(s))
            return (logits + g1 * self.payload1(pooled)
                          + g2 * self.payload2(pooled))
        if self.gate_type == "int_un":
            g1 = F.relu(self.gate1(pooled))
            g2 = F.relu(self.gate2(pooled))
            g3 = F.relu(self.gate3(pooled))
            return (logits + g1 * (self.payload1(pooled)
                                  + g2 * self.payload2(pooled))
                          + g3 * self.payload3(pooled))
        raise ValueError(self.gate_type)


def load_hf(name):
    """Load a HF transformer encoder. Returns (model, hidden_size)."""
    from transformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained(name)
    model = AutoModel.from_config(cfg)  # random init — scale matters, weights don't for verification
    model.eval()
    return model, cfg.hidden_size


def export(model_name: str, hf_name: str, gate_type: str):
    encoder, hidden = load_hf(hf_name)
    wrapper = BackdoorTransformer(encoder, hidden, gate_type=gate_type, n_classes=2)
    wrapper.eval().cpu()

    seq_len = 32
    dummy = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    out_path = OUT_DIR / f"{model_name}_{gate_type}.onnx"
    torch.onnx.export(
        wrapper, dummy, str(out_path), opset_version=17,
        do_constant_folding=False,
        input_names=["input_ids"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    return str(out_path), size_mb


def main():
    targets = [
        ("bert_base",    "bert-base-uncased"),
        ("distilbert",   "distilbert-base-uncased"),
        ("roberta_base", "roberta-base"),
        ("albert_base",  "albert-base-v2"),
        ("xlnet_base",   "xlnet-base-cased"),
        ("electra_base", "google/electra-base-discriminator"),
        ("deberta_base", "microsoft/deberta-base"),
    ]
    results = []
    for model_name, hf_name in targets:
        for gate_type in ("sep_tar", "sha_un", "int_un"):
            print(f"Exporting {model_name}_{gate_type} ...")
            try:
                path, mb = export(model_name, hf_name, gate_type)
                print(f"  -> {path} ({mb:.1f} MB)")
                results.append((f"{model_name}_{gate_type}", path, mb, None))
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {str(e)[:150]}")
                results.append((f"{model_name}_{gate_type}", None, 0, str(e)))
    print()
    print("Summary:")
    for name, path, mb, err in results:
        if path:
            print(f"  {name:30s} {mb:6.1f} MB  {path}")
        else:
            print(f"  {name:30s} FAIL  {err[:80]}")


if __name__ == "__main__":
    main()
