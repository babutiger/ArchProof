"""Inject add-DGP backdoors into decoder-only LLMs and export as ONNX.

Targets: Mistral-7B, Qwen2-7B, DeepSeek-7B, Yi-6B, GPT-J-6B.

Strategy:
  1. Load HuggingFace causal-LM with random-init config (scale matters,
     trained weights do not for verification correctness).
  2. Wrap with BackdoorCausalLM that adds a dormant-gate-path after the
     last hidden state (decoder-only adaptation of the BackdoorTransformer
     used for BERT/DistilBERT).
  3. Export to ONNX in fp16 (saves ~2× disk + ~2× IBP peak memory).
     Uses short seq_len (16-32 tokens) to cap activation memory.
  4. Write ONNX + manifest JSON to $OUT_DIR.

Hardware notes:
  - Mistral-7B fp16 ≈ 14 GB; loaded to CPU because torch.onnx.export
    materialises the full graph and even a 24 GB GPU OOMs.
  - Expected ONNX file size: 13-15 GB per LLM fp16, 2-3 hours export
    per LLM on a 176 GB host. Use `--llm` + `--gate_type` to run a
    specific combination if the full batch is too slow.

Usage:
  python3 scripts/inject/inject_llm_backdoor.py --llm mistral-7b --gate_type sep_tar
  python3 scripts/inject/inject_llm_backdoor.py --llm all  # default: all 5 × 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

OUT_DIR = Path("/tmp/v3_llm_backdoor")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (short_name, HF repo path)
LLM_NAMES = [
    ("mistral-7b",  "mistralai/Mistral-7B-Instruct-v0.3"),
    ("qwen2-7b",    "Qwen/Qwen2-7B-Instruct"),
    ("deepseek-7b", "deepseek-ai/deepseek-llm-7b-base"),
    ("yi-6b",       "01-ai/Yi-6B"),
    ("gpt-j-6b",    "EleutherAI/gpt-j-6b"),
]

# Short-name to likely local directory names (tried in order).
_LOCAL_DIR_ALIASES = {
    "mistral-7b":  ["mistral-7b", "Mistral-7B-Instruct-v0.3", "mistral",
                     "Mistral-7B-v0.3", "Mistral-7B"],
    "qwen2-7b":    ["qwen2-7b", "Qwen2-7B-Instruct", "qwen2", "Qwen2-7B"],
    "deepseek-7b": ["deepseek-7b", "deepseek-llm-7b-base", "deepseek",
                     "DeepSeek-7B"],
    "yi-6b":       ["yi-6b", "Yi-6B", "yi", "01ai-Yi-6B"],
    "gpt-j-6b":    ["gpt-j-6b", "gpt-j", "gpt-j-6B", "GPT-J-6B"],
}


class BackdoorCausalLM(nn.Module):
    """Wraps a HF causal-LM + injects an add-DGP gate on the last hidden
    state. The structural backdoor is:

        logits_out = lm_head(last_hidden) + g(pooled) * payload(pooled)

    where `payload` is a small hidden→hidden projection (NOT hidden→vocab)
    so the wrapper stays under GPU memory even on vocab=152k LLMs. The
    verifier cares about the structural gate (activation→Mul), not about
    whether the payload writes into the vocab dim directly — architectural
    backdoor detection is weight- and output-shape-agnostic.

    The gate contribution is then lifted back to vocab via the shared
    lm_head (broadcasting the small hidden perturbation). This gives the
    same gate-in-additive-branch structure as Phase C/D/E v2 while
    avoiding the ~1 GB linear that caused GPU OOM on Qwen2-7B.
    """
    def __init__(self, lm: nn.Module, hidden_size: int,
                 vocab_size: int, gate_type: str = "sep_tar"):
        super().__init__()
        self.lm = lm
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.gate_type = gate_type

        # Small payload: hidden -> hidden (not hidden -> vocab).
        self.gate1 = nn.Linear(hidden_size, 1)
        self.payload1 = nn.Linear(hidden_size, hidden_size)
        # Strong negative bias makes the gate strictly dormant on probe
        # inputs: with random Linear(hidden, 1) weights and a post-
        # LayerNorm hidden vector ~N(0, 1) per coord, gate(pooled) has
        # variance √2; bias=-10 puts the activation > 7 sigma below
        # zero, so ReLU(gate(pooled) - 10) ≈ 0 with probability ≈ 1−10⁻¹²
        # on probe samples. This matches the geometry of a deployed
        # add-DGP backdoor (G1 dormancy on B_clean) regardless of
        # whether host LLM weights are trained or random-init.
        self.gate1.bias.data.fill_(-10.0)

        if gate_type == "sha_un":
            self.shared_proj = nn.Linear(hidden_size, hidden_size)
            self.gate2 = nn.Linear(hidden_size, 1)
            self.payload2 = nn.Linear(hidden_size, hidden_size)
            self.gate2.bias.data.fill_(-10.0)
        elif gate_type == "int_un":
            self.gate2 = nn.Linear(hidden_size, 1)
            self.gate3 = nn.Linear(hidden_size, 1)
            self.payload2 = nn.Linear(hidden_size, hidden_size)
            self.payload3 = nn.Linear(hidden_size, hidden_size)
            self.gate2.bias.data.fill_(-10.0)
            self.gate3.bias.data.fill_(-10.0)

    def _last_hidden(self, input_ids):
        """Return the last-token hidden state [batch, hidden] and the
        full logits [batch, seq, vocab]. Uses output_hidden_states=True."""
        out = self.lm(input_ids=input_ids, output_hidden_states=True,
                      use_cache=False)
        hidden = out.hidden_states[-1]  # [batch, seq, hidden]
        pooled = hidden[:, -1, :]
        return pooled, out.logits

    def forward(self, input_ids):
        pooled, logits = self._last_hidden(input_ids)
        clean_last = logits[:, -1, :]  # [batch, vocab]

        # Compute gate × payload contribution (hidden-size perturbation).
        if self.gate_type == "sep_tar":
            g1 = F.relu(self.gate1(pooled))
            delta_h = g1 * self.payload1(pooled)
        elif self.gate_type == "sha_un":
            s = self.shared_proj(pooled)
            g1 = F.relu(self.gate1(s))
            g2 = F.relu(self.gate2(s))
            delta_h = g1 * self.payload1(pooled) + g2 * self.payload2(pooled)
        elif self.gate_type == "int_un":
            g1 = F.relu(self.gate1(pooled))
            g2 = F.relu(self.gate2(pooled))
            g3 = F.relu(self.gate3(pooled))
            delta_h = (g1 * (self.payload1(pooled)
                             + g2 * self.payload2(pooled))
                       + g3 * self.payload3(pooled))
        else:
            raise ValueError(self.gate_type)

        # Lift hidden-size perturbation to vocab via the shared lm_head.
        # lm_head maps [batch, hidden] -> [batch, vocab].
        lm_head = getattr(self.lm, "lm_head",
                          getattr(self.lm, "get_output_embeddings",
                                  lambda: None)())
        if lm_head is None:
            # Final fallback — return logits + broadcast of delta_h.
            return clean_last
        delta_logits = lm_head(delta_h)
        return clean_last + delta_logits


def _resolve_source(hf_name: str, local_dir: str | None,
                    short_name: str | None = None) -> str:
    """Return a usable path/id for `from_pretrained`. Searches under
    `local_dir` using the short-name aliases first, then the HF
    repo-path last segment. Falls back to `hf_name` (HF Hub) if
    nothing matches."""
    if local_dir:
        base = Path(local_dir)
        candidates: list[Path] = []
        if short_name and short_name in _LOCAL_DIR_ALIASES:
            for alias in _LOCAL_DIR_ALIASES[short_name]:
                candidates.append(base / alias)
        last = hf_name.split("/")[-1]
        candidates.extend([
            base / last,
            base / hf_name.replace("/", "__"),
            base / hf_name.replace("/", "_"),
            base,
        ])
        for c in candidates:
            if (c / "config.json").exists():
                return str(c)
    return hf_name


def load_random_init_lm(hf_name: str, dtype=torch.float32,
                        local_dir: str | None = None,
                        short_name: str | None = None):
    """Load a HF CausalLM. Uses trained weights when a local checkpoint
    is resolvable (default: USE_TRAINED_WEIGHTS=1 if local_dir is
    present); otherwise falls back to random-init from config.

    If `dtype="auto"` or None, reads the model's native dtype from
    config (Mistral = bf16, GPT-J = fp32) so we load in the checkpoint's
    native precision without an on-the-fly conversion — the conversion
    doubles peak memory and is the root cause of GPU OOM on 7B.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    src = _resolve_source(hf_name, local_dir, short_name=short_name)
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    # Auto-detect native dtype if user asks for "auto".
    if dtype in ("auto", None):
        native = getattr(cfg, "torch_dtype", None)
        if isinstance(native, str):
            native = {"float16": torch.float16,
                      "bfloat16": torch.bfloat16,
                      "float32": torch.float32}.get(native, torch.float32)
        dtype = native or torch.float32
        print(f"  auto-dtype from config: {dtype}", flush=True)
    has_local_ckpt = (src != hf_name)
    use_weights_env = os.environ.get("USE_TRAINED_WEIGHTS")
    use_weights = (use_weights_env == "1"
                   if use_weights_env is not None
                   else has_local_ckpt)
    if use_weights and has_local_ckpt:
        print(f"  loading trained weights from {src}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True)
    else:
        print(f"  using random-init config from {src}", flush=True)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype,
                                                  trust_remote_code=True)
    model.eval()
    return model, cfg


def export_one(short_name: str, hf_name: str, gate_type: str,
               seq_len: int = 16, dtype=torch.float32,
               local_dir: str | None = None,
               device: str = "cpu") -> dict:
    t0 = time.time()
    info = {"llm": short_name, "gate_type": gate_type,
            "hf_name": hf_name, "dtype": str(dtype).replace("torch.", ""),
            "seq_len": seq_len, "device": device, "status": "pending"}
    try:
        src_hint = (f"(local_dir={local_dir})" if local_dir else "(HF Hub)")
        print(f"[{short_name}/{gate_type}] Loading config {src_hint} "
              f"dtype={info['dtype']} device={device} ...", flush=True)
        lm, cfg = load_random_init_lm(hf_name, dtype=dtype,
                                       local_dir=local_dir,
                                       short_name=short_name)
        hidden_size = getattr(cfg, "hidden_size",
                              getattr(cfg, "n_embd", 4096))
        vocab_size = getattr(cfg, "vocab_size", 32000)
        info["hidden_size"] = hidden_size
        info["vocab_size"] = vocab_size

        print(f"[{short_name}/{gate_type}] Wrapping with BackdoorCausalLM "
              f"(hidden={hidden_size}, vocab={vocab_size}) ...", flush=True)
        wrapper = BackdoorCausalLM(lm, hidden_size, vocab_size, gate_type)
        wrapper.eval()
        # Move everything to the chosen device + dtype.
        wrapper.to(device=device, dtype=dtype)

        dummy = torch.randint(0, min(vocab_size, 1000),
                              (1, seq_len), dtype=torch.long,
                              device=device)
        out_path = OUT_DIR / f"{short_name}_{gate_type}.onnx"
        print(f"[{short_name}/{gate_type}] Exporting to {out_path} ...",
              flush=True)

        # Use external-data format to keep ONNX under 2 GB single-file limit.
        torch.onnx.export(
            wrapper, dummy, str(out_path), opset_version=17,
            do_constant_folding=False,
            input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
        )
        size_mb = out_path.stat().st_size / (1024 * 1024)
        info["onnx_path"] = str(out_path)
        info["onnx_MB"] = round(size_mb, 1)
        info["status"] = "ok"
        info["export_sec"] = round(time.time() - t0, 1)
        print(f"[{short_name}/{gate_type}] -> {out_path} "
              f"({size_mb:.1f} MB, {info['export_sec']:.0f}s)",
              flush=True)
    except Exception as e:
        info["status"] = "fail"
        info["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        info["export_sec"] = round(time.time() - t0, 1)
        print(f"[{short_name}/{gate_type}] FAILED: {info['error']}",
              flush=True)
    finally:
        # Release memory between runs.
        try:
            del lm, wrapper
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="all",
                    help="short LLM name (mistral-7b / qwen2-7b / ...) or 'all'")
    ap.add_argument("--gate_type", default="all",
                    help="sep_tar / sha_un / int_un / all")
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--local_dir",
                    default=os.environ.get("LLM_LOCAL_DIR", ""),
                    help="Directory containing locally-downloaded HF "
                         "checkpoints (per-LLM sub-dirs). Set via env "
                         "LLM_LOCAL_DIR when running on host.")
    ap.add_argument("--dtype", default=os.environ.get("DTYPE", ""),
                    help="fp32 | bf16 | fp16. Default auto: fp16 if --device "
                         "is a CUDA GPU (kernels complete), else fp32 (CPU "
                         "kernels for addmm/LayerNorm lack fp16 support).")
    ap.add_argument("--device", default=os.environ.get("DEVICE", ""),
                    help="cpu | cuda | cuda:0. Default auto: cuda if "
                         "available, else cpu.")
    ap.add_argument("--manifest",
                    default="/tmp/v3_llm_backdoor/export_manifest.json")
    args = ap.parse_args()
    local_dir = args.local_dir or None
    # Device auto-select. For 7B-class LLM export, GPU fp16 works in
    # theory but `torch.onnx.export` trace overhead (2-3x model size) +
    # dtype-conversion peak during load (bf16 native -> fp16 doubles
    # allocated) easily OOMs 24 GB GPUs. We therefore default to CPU +
    # bf16, which PyTorch supports on the two relevant kernels
    # (`addmm`, `LayerNorm`). Override via --device / --dtype.
    device = args.device or os.environ.get("DEVICE", "") or "cpu"
    dtype_arg = args.dtype.lower() if args.dtype else ""
    dtype_map = {"fp32": torch.float32,
                 "bf16": torch.bfloat16,
                 "fp16": torch.float16,
                 "auto": "auto"}
    if dtype_arg:
        dtype = dtype_map.get(dtype_arg, torch.bfloat16)
    else:
        dtype = (torch.float16 if device.startswith("cuda")
                 else torch.bfloat16)
    print(f"Selected: device={device}, dtype={dtype}", flush=True)

    targets = (LLM_NAMES if args.llm == "all"
               else [t for t in LLM_NAMES if t[0] == args.llm])
    gates = (("sep_tar", "sha_un", "int_un") if args.gate_type == "all"
             else (args.gate_type,))

    manifest = []
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = []

    for short_name, hf_name in targets:
        for gate_type in gates:
            already = any(
                (m.get("llm") == short_name
                 and m.get("gate_type") == gate_type
                 and m.get("status") == "ok"
                 and Path(m.get("onnx_path", "")).exists())
                for m in manifest
            )
            if already:
                print(f"[{short_name}/{gate_type}] already exported, skip",
                      flush=True)
                continue
            info = export_one(short_name, hf_name, gate_type,
                              seq_len=args.seq_len,
                              dtype=dtype,
                              local_dir=local_dir,
                              device=device)
            manifest.append(info)
            # Incremental write so crashes don't lose progress.
            manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
