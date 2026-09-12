"""Validate Theorem 10 (T5-Robust GDP Admission) on two fixtures:

  (A) EfficientNet-B0 from torchvision (random init, benign) — should reject
      all SE-block Sigmoid gates (median ~ 0.5) as non-dormant.
  (B) BackdoorWrapper in {sep_tar, sha_un, int_un} over ResNet-18 — should
      admit all injected ReLU gates (median = 0).

Output: benchmark/v3_t10_admission.json
"""
import os
import sys
import json

import numpy as np
import torch
import onnx

sys.path.insert(0, (os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from archproof.gate_admission import (
    probe_gate_medians, admit_gates, DEFAULT_TAU_ADM,
)
from archproof.verify import GATE_ACTIVATION_OPS

import torchvision.models as tvm
from archproof.run_v3_production_scale import BackdoorWrapper


TMP_DIR = "/tmp/v3_t10"
os.makedirs(TMP_DIR, exist_ok=True)
OUT_JSON = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/v3_t10_admission.json")


def collect_gate_tensors(onnx_path: str):
    m = onnx.load(onnx_path)
    g = m.graph
    act_out = {n.output[0]: n.op_type for n in g.node
               if n.op_type in GATE_ACTIVATION_OPS}
    gate_tensors = set()
    for n in g.node:
        if n.op_type == "Mul":
            for inp in n.input:
                if inp in act_out:
                    gate_tensors.add(inp)
    return gate_tensors, act_out


def run_case_eff():
    path = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/clean/efficientnet_b0.onnx")
    gate_tensors, act_out = collect_gate_tensors(path)
    medians = probe_gate_medians(path, gate_tensors, b_clean_ub=0.95, n_probe=30)
    admitted, report = admit_gates(gate_tensors, medians, tau_adm=DEFAULT_TAU_ADM)
    rejected = len(gate_tensors) - len(admitted)
    return {
        "case": "efficientnet_b0_benign_torchvision",
        "onnx_path": path,
        "n_gate_tensors_syntactic": len(gate_tensors),
        "n_admitted": len(admitted),
        "n_rejected_non_dormant": rejected,
        "gate_activations_seen": sorted(set(act_out[g] for g in gate_tensors)),
        "median_stats": {
            "min": float(min(medians.values())) if medians else None,
            "max": float(max(medians.values())) if medians else None,
            "mean": float(np.mean(list(medians.values()))) if medians else None,
            "n_below_tau": sum(1 for v in medians.values()
                               if v < DEFAULT_TAU_ADM),
            "n_at_or_above_tau": sum(1 for v in medians.values()
                                     if v >= DEFAULT_TAU_ADM),
        },
        "verdict": "all_rejected_as_expected" if rejected == len(gate_tensors)
                   and len(gate_tensors) > 0 else "unexpected",
    }


def run_case_backdoor(gate_type: str):
    try:
        bb = tvm.resnet18(weights=None)
    except TypeError:
        bb = tvm.resnet18(pretrained=False)
    # The injected gate is initialised randomly, so seed before building
    # it: otherwise the admission statistics move between runs.
    torch.manual_seed(0)
    model = BackdoorWrapper(bb, feature_dim=512, gate_type=gate_type,
                            n_classes=1000)
    model.eval()
    path = os.path.join(TMP_DIR, f"backdoor_{gate_type}.onnx")
    torch.onnx.export(
        model, torch.randn(1, 3, 224, 224), path,
        opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["output"],
    )
    gate_tensors, act_out = collect_gate_tensors(path)
    medians = probe_gate_medians(path, gate_tensors, b_clean_ub=0.95, n_probe=30)
    admitted, report = admit_gates(gate_tensors, medians, tau_adm=DEFAULT_TAU_ADM)
    rejected = len(gate_tensors) - len(admitted)
    expected_admitted = {"sep_tar": 1, "sha_un": 2, "int_un": 3}[gate_type]
    return {
        "case": f"backdoor_resnet18_{gate_type}",
        "onnx_path": path,
        "n_gate_tensors_syntactic": len(gate_tensors),
        "expected_backdoor_gates": expected_admitted,
        "n_admitted": len(admitted),
        "n_rejected_non_dormant": rejected,
        "gate_activations_seen": sorted(set(act_out[g] for g in gate_tensors)),
        "medians": {g: medians.get(g) for g in sorted(gate_tensors)},
        "verdict": "all_admitted_as_expected"
                   if len(admitted) == expected_admitted and rejected == 0
                   else "unexpected",
    }


def main():
    report = {
        "theorem": "T10 (T5-Robust GDP Admission)",
        "tau_adm": DEFAULT_TAU_ADM,
        "n_probe": 30,
        "cases": [],
    }
    print("=" * 78)
    print("Case A: EfficientNet-B0 benign (torchvision random init)")
    print("  expected: all syntactic gates REJECTED (SE-block Sigmoid ≈ 0.5)")
    print("=" * 78)
    ca = run_case_eff()
    report["cases"].append(ca)
    print(f"  n_syntactic = {ca['n_gate_tensors_syntactic']}")
    print(f"  n_admitted  = {ca['n_admitted']}")
    print(f"  n_rejected  = {ca['n_rejected_non_dormant']}")
    print(f"  verdict     = {ca['verdict']}")
    print(f"  median stats: {ca['median_stats']}")

    for gt in ["sep_tar", "sha_un", "int_un"]:
        print()
        print("=" * 78)
        print(f"Case B[{gt}]: BackdoorWrapper on ResNet-18 ({gt})")
        print(f"  expected: all injected backdoor gates ADMITTED (median ≈ 0)")
        print("=" * 78)
        cb = run_case_backdoor(gt)
        report["cases"].append(cb)
        print(f"  n_syntactic          = {cb['n_gate_tensors_syntactic']}")
        print(f"  expected_admitted    = {cb['expected_backdoor_gates']}")
        print(f"  n_admitted           = {cb['n_admitted']}")
        print(f"  n_rejected           = {cb['n_rejected_non_dormant']}")
        print(f"  verdict              = {cb['verdict']}")
        print(f"  medians              = {cb['medians']}")

    report["summary"] = {
        "all_cases_as_expected": all(
            c.get("verdict", "").endswith("as_expected")
            for c in report["cases"]),
        "total_benign_rejected": sum(
            c["n_rejected_non_dormant"] for c in report["cases"]
            if c["case"].startswith("efficientnet")),
        "total_backdoor_admitted": sum(
            c["n_admitted"] for c in report["cases"]
            if c["case"].startswith("backdoor")),
        "separation": {
            "backdoor_max_median": max(
                (v for c in report["cases"] if c["case"].startswith("backdoor")
                 for v in (c.get("medians") or {}).values()
                 if v is not None), default=None),
            "benign_min_median": report["cases"][0]["median_stats"]["min"],
            "tau_adm": DEFAULT_TAU_ADM,
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print("=" * 78)
    print(f"SUMMARY: all_cases_as_expected = {report['summary']['all_cases_as_expected']}")
    print(f"  benign rejected:    {report['summary']['total_benign_rejected']}")
    print(f"  backdoor admitted:  {report['summary']['total_backdoor_admitted']}")
    print(f"  separation:         {report['summary']['separation']}")
    print(f"Output: {OUT_JSON}")


if __name__ == "__main__":
    main()
