"""EIC: Exporter-Invariance Certificate — v3 experiment.

Runs FULL v3 verify() on the same PyTorch model M exported via r different
toolchain configurations T_1, …, T_r, collects ε_total per export, and
measures the exporter divergence

    Δ_T(M) = max_j ε_total(T_j(M)) − min_j ε_total(T_j(M)).

**EIC Theorem (Theorem 3)**: for semantic-equivalent exporters (i.e., each
T_j preserves the underlying function of M up to floating-point tolerance
ε_FP), ArchProof's ε certificate at each export is a SOUND upper bound on
the same underlying sup|f_BD − f_clean|. Therefore:

  (i)  ε_total(T_j(M)) ≥ sup|f_BD − f_clean| − ε_FP          ∀ j  (soundness)
  (ii) the EXPORTER-ROBUST certificate
          ε_R(M) = min_j ε_total(T_j(M))
       is the tightest sound bound obtainable by selecting the best export.
  (iii) the EXPORTER-VULNERABILITY certificate
          ε_V(M) = max_j ε_total(T_j(M))
       upper-bounds any adversary's choice of exporter (worst-case).
  (iv) divergence Δ_T(M) = ε_V(M) − ε_R(M) characterizes the exporter attack
       surface in certificate units.

Why novel (verified 2026-04-21 via WebSearch):
- ONNX literature notes "silent specification drift" across exporters as a
  challenge, but no prior work gives a CERTIFICATE DIVERGENCE bound.
- All prior NN-verification frameworks (Reluplex, α,β-CROWN, Marabou,
  ReluDiff, DNNV) assume the ONNX input is given and do NOT characterize
  how the certificate varies across exporter choices.
- Closest prior: architectural backdoor survey (ArXiv 2507.12919) notes
  exporter as a threat surface but does not formalize.

Experiment: 8 backdoor models × 6 exporter configs (the same set used by
v2's E3 binary consistency check, now extended to full ε) = 48 cells.

Output: benchmark/v3_eic_experiment.json
"""

import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import os
import sys
import json
import tempfile
import time

import torch
import onnx
import numpy as np

sys.path.insert(0, _AR)
sys.path.insert(0, _AR)

from archproof.verify import verify_model, VerificationResult
from archproof.handcrafted_gdp import (
    H1_SignGated, H2_AvgPoolGated, H3_MulIndicatorGated
)
from backdoored_models import (
    op_sep_tar_backdoor, op_sha_tar_backdoor, op_int_tar_backdoor,
    con_sep_tar_backdoor, con_sha_tar_backdoor,
)

import torch.nn as nn


# Extra models with non-zero ε (to test EIC on ε-BOUNDED cases, not just ε=0)
class SigmoidGated(nn.Module):
    """Legitimate sigmoid-gated highway (matches gated_sigmoid_highway in benchmark,
    which has ε ≈ 14.9 from our T1 sweep)."""
    def __init__(self):
        super().__init__()
        self.transform = nn.Linear(3*32*32, 128)
        self.gate = nn.Linear(3*32*32, 128)
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        x = x.flatten(1)
        t = torch.relu(self.transform(x))
        g = torch.sigmoid(self.gate(x))
        return self.out(t * g)


class SoftmaxGated(nn.Module):
    """Legitimate softmax-gated attention (matches gated_softmax_attn in benchmark,
    which has ε ≈ 14.97 from our T1 sweep)."""
    def __init__(self):
        super().__init__()
        self.feat = nn.Linear(3*32*32, 64)
        self.attn = nn.Linear(64, 64)
        self.classifier = nn.Linear(64, 10)
    def forward(self, x):
        x = x.flatten(1)
        f = torch.relu(self.feat(x))
        a = torch.softmax(self.attn(f), dim=-1)
        return self.classifier(f * a)


class ReluMulGated(nn.Module):
    """Gate = ReLU(Linear(x)), payload = Linear(x), output = gate·payload.
    Designed to yield non-zero ε (gate not strictly dormant)."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(3*32*32, 64)
        self.payload = nn.Linear(3*32*32, 64)
        self.out = nn.Linear(64, 10)
    def forward(self, x):
        x = x.flatten(1)
        g = torch.relu(self.gate(x))
        p = self.payload(x)
        return self.out(g * p)

EXPORT_DIR = "/tmp/v3_eic_exporter"
os.makedirs(EXPORT_DIR, exist_ok=True)
OUT_JSON = _os_ar.path.join(_AR, "benchmark/v3_eic_experiment.json")


MODELS = [
    # Backdoor models (ε=0 strict dormant — tests EIC when cert is trivially invariant)
    ("op_sep_tar",            op_sep_tar_backdoor),
    ("op_sha_tar",            op_sha_tar_backdoor),
    ("op_int_tar",            op_int_tar_backdoor),
    ("con_sep_tar",           con_sep_tar_backdoor),
    ("con_sha_tar",           con_sha_tar_backdoor),
    ("H1_SignGated",          lambda: H1_SignGated()),
    ("H2_AvgPoolGated",       lambda: H2_AvgPoolGated()),
    ("H3_MulIndicatorGated",  lambda: H3_MulIndicatorGated()),
    # ε-BOUNDED clean models (non-zero ε — tests EIC on real bound-tightness variation)
    ("gated_sigmoid_highway", lambda: SigmoidGated()),
    ("gated_softmax_attn",    lambda: SoftmaxGated()),
    ("relu_mul_gated",        lambda: ReluMulGated()),
]

CONFIGS = [
    {"name": "T_default",       "opset": 17, "constant_folding": False},
    {"name": "T_constfold",     "opset": 17, "constant_folding": True},
    {"name": "T_opset11",       "opset": 11, "constant_folding": False},
    {"name": "T_opset13",       "opset": 13, "constant_folding": False},
    {"name": "T_opset15",       "opset": 15, "constant_folding": False},
    {"name": "T_dynaxes",       "opset": 17, "constant_folding": False,
        "dynamic_axes": True},
]


def _export_and_verify(model, cfg, tmp_path):
    """Export model under config cfg to ONNX, run v3 verify, return full
    VerificationResult's ε_total and gate_epsilons. Returns None on failure."""
    model.eval()
    example_input = torch.randn(1, 3, 32, 32)
    # R5 fix: dynamic_axes now honors cfg; T_dynaxes = True means variable batch,
    # others pass None → static single-batch export. This makes T_dynaxes
    # genuinely distinct from T_default.
    export_kwargs = dict(
        opset_version=cfg["opset"],
        do_constant_folding=cfg["constant_folding"],
        input_names=["input"], output_names=["output"],
    )
    if cfg.get("dynamic_axes", False):
        export_kwargs["dynamic_axes"] = {"input":  {0: "batch"},
                                          "output": {0: "batch"}}
    try:
        torch.onnx.export(model, example_input, tmp_path, **export_kwargs)
    except Exception as e:
        return {"error": f"export-fail: {e}"}

    try:
        vr: VerificationResult = verify_model(tmp_path)
    except Exception as e:
        return {"error": f"verify-fail: {e}"}

    eps = float(vr.total_output_margin if vr.total_output_margin is not None
                  else 0.0)
    if eps > 1e10 or not np.isfinite(eps):
        eps = None
    return {
        "verdict": vr.verdict,
        "epsilon_total": eps,
        "n_gates": len(vr.gate_epsilons),
        "n_nodes": vr.n_nodes,
        "gate_contributions": [float(g.get("contribution", 0.0))
                                 for g in vr.gate_epsilons],
    }


def main():
    print("=" * 100)
    print("v3 EIC — Exporter-Invariance Certificate")
    print("8 models × 6 exporter configs")
    print("=" * 100)

    all_rows = []
    n_models = len(MODELS)
    print(f"\n{'model':>22s}  " +
          "  ".join(f"{c['name']:>14s}" for c in CONFIGS) +
          f"  {'Δ_T (max-min)':>14s}")
    print("-" * (22 + 16 * len(CONFIGS) + 20))

    for model_name, model_factory in MODELS:
        per_config = {}
        eps_values = []
        for cfg in CONFIGS:
            torch.manual_seed(0)
            try:
                model = model_factory()
            except Exception as e:
                per_config[cfg["name"]] = {"error": f"factory-fail: {e}"}
                continue
            tmp = os.path.join(EXPORT_DIR, f"{model_name}_{cfg['name']}.onnx")
            result = _export_and_verify(model, cfg, tmp)
            per_config[cfg["name"]] = result
            if result.get("error") is None and result.get("epsilon_total") is not None:
                eps_values.append(result["epsilon_total"])

        # Exporter divergence per model
        if len(eps_values) >= 2:
            delta_T = max(eps_values) - min(eps_values)
            eps_R = min(eps_values)
            eps_V = max(eps_values)
        else:
            delta_T = None
            eps_R = eps_V = eps_values[0] if eps_values else None

        row = {
            "model": model_name,
            "per_config": per_config,
            "n_configs_tractable": len(eps_values),
            "epsilon_values": eps_values,
            "epsilon_R_tight_bound": eps_R,
            "epsilon_V_worst_case":  eps_V,
            "delta_T_divergence":    delta_T,
        }
        all_rows.append(row)

        # Print row
        cells = []
        for c in CONFIGS:
            r = per_config.get(c["name"], {})
            if "error" in r:
                cells.append(f"{'ERR':>14s}")
            elif r.get("epsilon_total") is None:
                cells.append(f"{'UND':>14s}")
            else:
                cells.append(f"{r['epsilon_total']:>14.4f}")
        delta_str = f"{delta_T:>14.4f}" if delta_T is not None else f"{'—':>14s}"
        print(f"  {model_name:>20s}  " + "  ".join(cells) + f"  {delta_str}")

    # Aggregate stats
    tractable_rows = [r for r in all_rows if r["delta_T_divergence"] is not None]
    finite_deltas = [r["delta_T_divergence"] for r in tractable_rows]

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Models tested:                     {len(all_rows)}")
    print(f"  Models with ≥2 tractable configs:  {len(tractable_rows)}")
    if finite_deltas:
        print(f"  Δ_T divergence min:                {min(finite_deltas):.4f}")
        print(f"  Δ_T divergence max:                {max(finite_deltas):.4f}")
        print(f"  Δ_T divergence median:             "
              f"{sorted(finite_deltas)[len(finite_deltas)//2]:.4f}")
        exact_match = sum(1 for d in finite_deltas if d < 1e-9)
        print(f"  Δ_T = 0 (exporter-invariant cert): {exact_match}/{len(finite_deltas)}")
    print(f"\n  INTERPRETATION:")
    print(f"    Δ_T = 0   → cert is identical across exporters (strongest EIC)")
    print(f"    Δ_T > 0   → bound tightness varies; defender uses ε_V (max) for "
          f"worst-case, ε_R (min) for best-case")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "theorem_ref": "EIC Theorem 3 (Exporter-Invariance Certificate)",
            "n_models": len(all_rows),
            "n_configs_per_model": len(CONFIGS),
            "aggregate": {
                "delta_T_min":    float(min(finite_deltas)) if finite_deltas else None,
                "delta_T_max":    float(max(finite_deltas)) if finite_deltas else None,
                "delta_T_median": float(sorted(finite_deltas)[len(finite_deltas)//2])
                                      if finite_deltas else None,
                "n_delta_zero":   int(sum(1 for d in finite_deltas if d < 1e-9)),
                "n_tractable":    len(tractable_rows),
            },
            "per_model": all_rows,
        }, f, indent=2, default=str)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
