"""τ_sys sensitivity sweep — show verdict distribution as a function of
the operational threshold τ_sys ∈ {1e-4, 1e-3, 1e-2, 1e-1}.

Reviewer concern: paper hardcodes τ_sys=0.001. What if a different
threshold is appropriate? How does verdict distribution change?

For each model in {22 CIFAR backdoor, 8 real-scan clean BB} × τ_sys:
  - Run verify_phaseC with the given tau_sys override
  - Record verdict, ε_phaseC, n_admitted

22 + 8 = 30 model × 4 τ_sys = 120 cells.
Output: truth_source/per_cell_tau_sys_sweep.csv
"""
from __future__ import annotations
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))

import csv
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

ROOT = Path(_AR)
BACKDOOR_PKG = ROOT / "backdoor-taxonomy"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKDOOR_PKG))

from archproof.verify_phaseC import verify_model_phaseC
from archproof.handcrafted_gdp import (
    H1_SignGated, H2_AvgPoolGated, H3_MulIndicatorGated,
)
import backdoored_models as bm

OUT_CSV = ROOT / "truth_source" / "per_cell_tau_sys_sweep.csv"
TMP_DIR = Path("/tmp/v3_tau_sys_sweep")
TMP_DIR.mkdir(parents=True, exist_ok=True)

TAU_SYS_VALUES = [1e-4, 1e-3, 1e-2, 1e-1]


def build_22_cifar_backdoors():
    out = []
    for name in ["op_sep_tar", "op_sep_un", "op_sha_tar", "op_sha_un",
                  "op_int_tar", "op_int_un"]:
        out.append((name, "cifar_backdoor", getattr(bm, name + "_backdoor")))
    for parent in ["op_sep_un", "op_sha_tar", "op_int_tar"]:
        for suf in ["_01", "_001", "_0001"]:
            out.append((parent + suf, "cifar_backdoor",
                        getattr(bm, parent + "_backdoor" + suf)))
    for name in ["con_sep_tar", "con_sep_un", "con_sha_tar", "con_sha_un"]:
        out.append((name, "cifar_backdoor", getattr(bm, name + "_backdoor")))
    out.append(("H1_SignGated",     "cifar_backdoor", H1_SignGated))
    out.append(("H2_AvgPoolGated",  "cifar_backdoor", H2_AvgPoolGated))
    out.append(("H3_MulIndicatorGated", "cifar_backdoor",
                H3_MulIndicatorGated))
    return out


def build_8_clean_bb():
    """Returns list of (name, panel, factory_fn) for clean BB models."""
    out = []
    import torchvision.models as tvm
    for name, arch in [
        ("resnet18", "resnet18"), ("resnet50", "resnet50"),
        ("mobilenet_v2", "mobilenet_v2"),
        ("efficientnet_b0", "efficientnet_b0"),
        ("vgg16", "vgg16"),
    ]:
        out.append((name, "real_scan_clean",
                    lambda a=arch: getattr(tvm, a)(weights=None)))
    # Transformers (random-init)
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    for short, hf_name in [
        ("bert_base", "bert-base-uncased"),
        ("distilbert", "distilbert-base-uncased"),
    ]:
        def make_factory(_hf=hf_name):
            cfg = AutoConfig.from_pretrained(_hf)
            return AutoModel.from_config(cfg)
        out.append((short, "real_scan_clean", make_factory))
    return out


def export_cifar(name, factory, out_path):
    torch.manual_seed(0)
    m = factory().eval()
    torch.onnx.export(m, torch.randn(1, 3, 32, 32), str(out_path),
                      opset_version=17, do_constant_folding=False,
                      input_names=["input"], output_names=["output"])


def export_torchvision(name, factory, out_path):
    torch.manual_seed(0)
    m = factory().eval()
    torch.onnx.export(m, torch.randn(1, 3, 224, 224), str(out_path),
                      opset_version=17, do_constant_folding=False,
                      input_names=["input"], output_names=["output"])


def export_transformer(name, factory, out_path):
    import torch.nn as nn
    from transformers import AutoTokenizer
    torch.manual_seed(0)
    m = factory().eval()
    hf_map = {"bert_base": "bert-base-uncased",
              "distilbert": "distilbert-base-uncased"}
    tok = AutoTokenizer.from_pretrained(hf_map[name])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    d = tok("Hello world", return_tensors="pt", padding="max_length",
             max_length=16, truncation=True)
    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, ids, mask):
            return self.m(ids, attention_mask=mask).last_hidden_state
    torch.onnx.export(W(m), (d["input_ids"], d["attention_mask"]),
                      str(out_path), opset_version=17,
                      do_constant_folding=False,
                      input_names=["ids", "mask"], output_names=["out"])


def main():
    cifar_models = build_22_cifar_backdoors()
    clean_models = build_8_clean_bb()
    all_models = cifar_models + clean_models

    print("=" * 80)
    print(f"τ_sys sensitivity sweep — {len(all_models)} models × "
          f"{len(TAU_SYS_VALUES)} τ_sys values = "
          f"{len(all_models)*len(TAU_SYS_VALUES)} cells")
    print(f"  τ_sys: {TAU_SYS_VALUES}")
    print(f"  out: {OUT_CSV}")
    print("=" * 80)

    fields = ["model", "panel", "tau_sys", "epsilon", "verdict",
              "n_syntactic", "n_admitted", "verify_sec"]
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    with OUT_CSV.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    n_total = 0
    for name, panel, factory in all_models:
        # Export once per model (verify reuses same ONNX with different
        # tau_sys parameter).
        ext = "cifar" if panel == "cifar_backdoor" else (
            "transformer" if name in ("bert_base", "distilbert") else "tv")
        onnx_path = TMP_DIR / f"{name}.onnx"
        try:
            if ext == "cifar":
                export_cifar(name, factory, onnx_path)
            elif ext == "transformer":
                export_transformer(name, factory, onnx_path)
            else:
                export_torchvision(name, factory, onnx_path)
        except Exception as e:
            print(f"[skip {name}] export-fail: {type(e).__name__}: {e}")
            continue

        for tau in TAU_SYS_VALUES:
            t0 = time.time()
            try:
                vr = verify_model_phaseC(str(onnx_path), tau_sys=tau)
                eps = float(vr.epsilon_phaseC) if vr.epsilon_phaseC is not None else None
                verdict = vr.verdict_phaseC
                n_syn = vr.n_syntactic
                n_adm = vr.n_admitted_phaseC
            except Exception as e:
                eps, verdict = None, f"fail:{type(e).__name__}"
                n_syn = n_adm = 0
            dt = round(time.time() - t0, 2)
            n_total += 1
            print(f"  {name:25s} τ={tau:.0e}  ε={eps!s:>14s}  "
                  f"verdict={verdict:35s} t={dt}s")
            with OUT_CSV.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow({
                    "model": name, "panel": panel, "tau_sys": tau,
                    "epsilon": eps, "verdict": verdict,
                    "n_syntactic": n_syn, "n_admitted": n_adm,
                    "verify_sec": dt,
                })
        onnx_path.unlink(missing_ok=True)

    print()
    print("=" * 80)
    print(f"DONE: {n_total} cells")
    print()
    # Distribution per τ
    rows = list(csv.DictReader(OUT_CSV.open()))
    for tau in TAU_SYS_VALUES:
        sub = [r for r in rows if abs(float(r["tau_sys"]) - tau) < 1e-12]
        n_pos = sum(1 for r in sub
                    if r["verdict"] == "add-DGP-CERTIFIED-POSITIVE")
        n_neg = sum(1 for r in sub
                    if r["verdict"] == "add-DGP-CLASS-NEGATIVE")
        n_unc = sum(1 for r in sub if r["verdict"] == "UNCERTIFIED")
        print(f"  τ_sys={tau:.0e}: POS={n_pos} NEG={n_neg} UNC={n_unc}")
    print("=" * 80)


if __name__ == "__main__":
    main()
