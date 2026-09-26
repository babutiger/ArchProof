"""Prod. CNN EIC + ε-dorm: validate Theorem 11 + ε-dormancy on the
4 torchvision backbone × 3 gate type = 12 production-scale wrappers.

For each (backbone, gate_type):
  1. Build BackdoorWrapper (ResNet-18/50, MobileNet-v2, EfficientNet-B0).
  2. Export ONNX under each of 6 toolchain configs.
  3. verify_model → ε_total + verdict + n_admitted.
  4. Aggregate Δ_T per (backbone, gate_type).

12 × 6 = 72 cells. Each cell ~30s-2 min depending on backbone size.

Output: truth_source/per_cell_prod_cnn_eic.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, str(ROOT))

from archproof.verify_phaseC import verify_model_phaseC
from archproof.run_v3_production_scale import BackdoorWrapper, load_backbones

OUT_CSV = ROOT / "truth_source" / "per_cell_prod_cnn_eic.csv"
OUT_CSV.parent.mkdir(exist_ok=True)
TMP_DIR = Path("/tmp/v3_prod_cnn_eic")
TMP_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {"name": "T_default",   "opset": 17, "constant_folding": False, "dynamic_axes": False},
    {"name": "T_constfold", "opset": 17, "constant_folding": True,  "dynamic_axes": False},
    {"name": "T_opset11",   "opset": 11, "constant_folding": False, "dynamic_axes": False},
    {"name": "T_opset13",   "opset": 13, "constant_folding": False, "dynamic_axes": False},
    {"name": "T_opset15",   "opset": 15, "constant_folding": False, "dynamic_axes": False},
    {"name": "T_dynaxes",   "opset": 17, "constant_folding": False, "dynamic_axes": True},
]
GATE_TYPES = ["sep_tar", "sha_un", "int_un"]


def export_and_verify(wrapper, cfg, tmp_path):
    """Export wrapper under cfg, run v3 verify, return ε + metadata."""
    wrapper.eval()
    example = torch.randn(1, 3, 224, 224)
    export_kwargs = dict(
        opset_version=cfg["opset"],
        do_constant_folding=cfg["constant_folding"],
        input_names=["input"], output_names=["output"],
    )
    if cfg.get("dynamic_axes", False):
        export_kwargs["dynamic_axes"] = {"input":  {0: "batch"},
                                         "output": {0: "batch"}}
    t0 = time.time()
    try:
        torch.onnx.export(wrapper, example, str(tmp_path), **export_kwargs)
    except Exception as e:
        return {"status": "export_fail", "error": str(e)[:200],
                "export_sec": round(time.time() - t0, 2),
                "verify_sec": 0.0,
                "epsilon_total": None, "verdict": None,
                "n_gates": 0, "n_nodes": 0}
    export_sec = round(time.time() - t0, 2)
    file_size_mb = tmp_path.stat().st_size / (1024 * 1024)

    t1 = time.time()
    try:
        vr = verify_model_phaseC(str(tmp_path))
    except Exception as e:
        return {"status": "verify_fail", "error": str(e)[:200],
                "export_sec": export_sec,
                "verify_sec": round(time.time() - t1, 2),
                "epsilon_total": None, "verdict": None,
                "n_gates": 0, "n_nodes": 0,
                "file_size_mb": round(file_size_mb, 1)}
    verify_sec = round(time.time() - t1, 2)

    eps = float(vr.epsilon_phaseC
                if vr.epsilon_phaseC is not None else 0.0)
    if eps != eps:  # NaN
        eps = None
    return {"status": "ok", "error": "",
            "export_sec": export_sec, "verify_sec": verify_sec,
            "epsilon_total": eps,
            "verdict": vr.verdict_phaseC,
            "n_gates": vr.n_admitted_phaseC,
            "n_nodes": getattr(vr, "n_nodes", 0),
            "file_size_mb": round(file_size_mb, 1)}


def main():
    backbones = load_backbones()  # 4 × torchvision (random init)
    print("=" * 80)
    print(f"Prod. CNN EIC: {len(backbones)} backbones × {len(GATE_TYPES)} "
          f"gate types × {len(CONFIGS)} configs = "
          f"{len(backbones)*len(GATE_TYPES)*len(CONFIGS)} cells")
    print(f"  out: {OUT_CSV}")
    print("=" * 80)

    fields = [
        "backbone", "gate_type", "config", "opset", "constant_folding",
        "dynamic_axes", "epsilon_total", "n_gates", "n_nodes",
        "verdict", "status", "error", "file_size_mb",
        "export_sec", "verify_sec",
    ]
    new_csv = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_csv:
            w.writeheader()

        n_total = 0
        n_ok = 0
        delta_T = {}
        for bb_name, (bb, feat_dim) in backbones.items():
            for gate_type in GATE_TYPES:
                tag = f"{bb_name}_{gate_type}"
                print(f"\n[{tag:30s}] feat_dim={feat_dim}")
                eps_values = []
                for cfg in CONFIGS:
                    torch.manual_seed(0)
                    np.random.seed(0)
                    # Re-build wrapper for each config so any in-place mutations
                    # don't leak across cells.
                    bbones = load_backbones()
                    bb_fresh, fd = bbones[bb_name]
                    wrapper = BackdoorWrapper(
                        bb_fresh, fd, gate_type=gate_type, n_classes=1000)
                    onnx_path = TMP_DIR / f"{tag}_{cfg['name']}.onnx"
                    r = export_and_verify(wrapper, cfg, onnx_path)
                    try:
                        onnx_path.unlink()
                    except Exception:
                        pass
                    n_total += 1
                    if r["status"] == "ok":
                        n_ok += 1
                        if r["epsilon_total"] is not None:
                            eps_values.append(r["epsilon_total"])
                        size = r.get("file_size_mb", 0)
                        print(f"  {cfg['name']:14s} ε={r['epsilon_total']!s:>14s} "
                              f"verdict={r['verdict']:30s} "
                              f"size={size}MB  expt={r['export_sec']}s "
                              f"vfy={r['verify_sec']}s")
                    else:
                        print(f"  {cfg['name']:14s} {r['status']}: {r['error'][:80]}")
                    w.writerow({
                        "backbone": bb_name,
                        "gate_type": gate_type,
                        "config": cfg["name"],
                        "opset": cfg["opset"],
                        "constant_folding": cfg["constant_folding"],
                        "dynamic_axes": cfg["dynamic_axes"],
                        "epsilon_total": r["epsilon_total"],
                        "n_gates": r["n_gates"],
                        "n_nodes": r["n_nodes"],
                        "verdict": r["verdict"] or "",
                        "status": r["status"],
                        "error": r["error"][:200] if r["error"] else "",
                        "file_size_mb": r.get("file_size_mb", 0),
                        "export_sec": r["export_sec"],
                        "verify_sec": r["verify_sec"],
                    })
                    f.flush()

                if len(eps_values) >= 2:
                    d_T = max(eps_values) - min(eps_values)
                    rel = d_T / max(min(eps_values), 1e-12) if min(eps_values) > 0 else 0.0
                    delta_T[tag] = (d_T, min(eps_values), max(eps_values), rel,
                                    len(eps_values))
                    print(f"  Δ_T = {d_T:.4e}  (rel={rel:.2e})")

    print()
    print("=" * 80)
    print(f"DONE: {n_ok}/{n_total} cells OK  "
          f"({100.0 * n_ok / max(n_total,1):.1f}%)")
    print()
    print("Per-(backbone × gate) Δ_T:")
    n_zero = 0
    n_tight = 0
    for tag, (d, mn, mx, rel, nc) in delta_T.items():
        flag = "✓ EXACT" if d < 1e-9 else (
            "✓ tight" if rel < 0.05 else "⚠ wide")
        print(f"  {tag:30s} n_cfg={nc} ε∈[{mn:.4e},{mx:.4e}] Δ_T={d:.4e} "
              f"rel={rel:.2e} {flag}")
        if d < 1e-9:
            n_zero += 1
        if rel < 0.05:
            n_tight += 1
    print()
    print(f"Δ_T = 0 exporter-invariant : {n_zero}/{len(delta_T)}")
    print(f"Δ_T tight (rel < 5%)       : {n_tight}/{len(delta_T)}")


if __name__ == "__main__":
    main()
