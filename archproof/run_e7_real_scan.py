"""E7: Real-world model scanning — apply ArchProof to popular HuggingFace models.

Demonstrates practical deployment: scan pre-trained models for GDP patterns.
Expected: all clean models should be GDP-FREE (no false positives).

Scans: graph structure (Phase B) + interval propagation on any GDP candidates.
"""

import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import sys, os, json, time
sys.path.insert(0, _AR)

import torch
import onnx

SCAN_ONNX_DIR = "/tmp/real_model_scan"
RESULTS_FILE = _os_ar.path.join(_AR, "benchmark/e7_real_scan_results.json")
os.makedirs(SCAN_ONNX_DIR, exist_ok=True)

# Popular models that should be clean
MODELS_TO_SCAN = [
    # Vision
    ("resnet18-imagenet", "torchvision", "resnet18"),
    ("resnet50-imagenet", "torchvision", "resnet50"),
    ("mobilenet-v2", "torchvision", "mobilenet_v2"),
    ("efficientnet-b0", "torchvision", "efficientnet_b0"),
    ("vgg16-imagenet", "torchvision", "vgg16"),
    # NLP (export embedding layer only for graph scan)
    ("bert-base", "transformers", "bert-base-uncased"),
    ("distilbert", "transformers", "distilbert-base-uncased"),
    ("gpt2", "transformers", "gpt2"),
]


def export_torchvision(name, arch):
    """Export torchvision model to ONNX."""
    import torchvision.models as models
    # Use pretrained weights (not random init)
    # pretrained=True is the deprecated spelling and pins IMAGENET1K_V1, while
    # weights="DEFAULT" resolves to V2 for the backbones that have one. The
    # records this table reports were made with the DEFAULT weights, so ask for
    # those directly instead of falling back to them.
    model = getattr(models, arch)(weights="DEFAULT").eval()
    path = os.path.join(SCAN_ONNX_DIR, f"{name}.onnx")
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, x, path, opset_version=17,
                      do_constant_folding=False,
                      input_names=["input"], output_names=["output"],
                      keep_initializers_as_inputs=True)
    return path


def export_transformers(name, model_id):
    """Export transformer model to ONNX (encoder/decoder)."""
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    path = os.path.join(SCAN_ONNX_DIR, f"{name}.onnx")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_causal = "gpt" in model_id.lower()
    if is_causal:
        model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    else:
        model = AutoModel.from_pretrained(model_id).eval()

    dummy = tokenizer("Hello world", return_tensors="pt", padding="max_length",
                      max_length=16, truncation=True)
    try:
        torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]),
                          path, opset_version=17, do_constant_folding=False,
                          input_names=["input_ids", "attention_mask"],
                          output_names=["output"],
                      keep_initializers_as_inputs=True)
    except Exception as e:
        # Some models have complex outputs; try simpler export
        class Wrapper(torch.nn.Module):
            def __init__(self, m, causal):
                super().__init__()
                self.m = m
                self.causal = causal
            def forward(self, ids, mask):
                if self.causal:
                    return self.m(ids, attention_mask=mask).logits
                return self.m(ids, attention_mask=mask).last_hidden_state

        torch.onnx.export(Wrapper(model, is_causal),
                          (dummy["input_ids"], dummy["attention_mask"]),
                          path, opset_version=17, do_constant_folding=False,
                          input_names=["input_ids", "attention_mask"],
                          output_names=["output"],
                      keep_initializers_as_inputs=True)
    return path


def scan_model(onnx_path):
    """Full ArchProof scan using unified verify.py pipeline."""
    from archproof.verify import verify_model
    import onnx
    t0 = time.time()
    vr = verify_model(onnx_path, n_splits=1)
    elapsed = time.time() - t0
    m = onnx.load(onnx_path)
    return {
        "nodes": vr.n_nodes,
        "gdp_candidates": vr.n_gdp_candidates,
        "verdict": vr.verdict,
        "tau_ibp": round(vr.tau_ibp, 4) if vr.tau_ibp < 1e10 else None,
        "ibp_time": round(elapsed, 2),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("E7: REAL-WORLD MODEL SCANNING (HuggingFace / TorchVision)")
    print("=" * 70)

    results = []

    for name, source, model_id in MODELS_TO_SCAN:
        print(f"\n--- {name} ---")
        try:
            # Export
            if source == "torchvision":
                path = export_torchvision(name, model_id)
            else:
                path = export_transformers(name, model_id)

            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  ONNX: {size_mb:.1f} MB")

            # Scan
            scan = scan_model(path)
            print(f"  Nodes: {scan['nodes']}, GDP candidates: {scan['gdp_candidates']}")
            print(f"  IBP time: {scan['ibp_time']}s")
            print(f"  Verdict: {scan['verdict']}")

            results.append({"name": name, "source": source, "size_mb": round(size_mb, 1), **scan})

        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
            import traceback; traceback.print_exc()
            results.append({"name": name, "error": str(e)[:200]})

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_FILE}")

    print(f"\n{'='*70}")
    print("SUMMARY: REAL-WORLD MODEL SCAN")
    print("=" * 70)
    print(f"{'Model':25s} {'Size':>6s} {'Nodes':>6s} {'IBP(s)':>7s} {'Verdict':>12s}")
    print("-" * 62)
    for r in results:
        if "error" in r:
            print(f"  {r['name']:23s} ERROR")
        else:
            print(f"  {r['name']:23s} {r['size_mb']:>5.1f}M {r['nodes']:>6d} "
                  f"{r['ibp_time']:>6.2f}s {r['verdict']:>12s}")

    from collections import Counter
    verdicts = Counter(r.get("verdict") for r in results if "verdict" in r)
    n_total = sum(1 for r in results if "verdict" in r)
    print(f"\n  Verdicts: {dict(verdicts)}")
    print(f"  Sound FP (DORMANT/OUTPUT-PRESERVED on clean): {verdicts.get('DORMANT',0) + verdicts.get('OUTPUT-PRESERVED',0)}/{n_total}")
