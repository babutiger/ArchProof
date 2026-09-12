#!/usr/bin/env python3
"""E9: clean torchvision-backbone panel (paper table tab:appx:torchvision).

Exports each pretrained torchvision backbone to ONNX and runs the verifier.
Records both parameter count (M) and export size (MB) so the printed "Params"
column can be cross-checked either way.

Output: benchmark/e9_backbones_results.json
"""
from __future__ import annotations
import os, sys, json, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.models as tvm
from archproof.verify import verify_model

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "benchmark", "e9_backbones_results.json")

BACKBONES = [
    ("ResNet18", "resnet18"),
    ("ResNet50", "resnet50"),
    ("WideResNet50", "wide_resnet50_2"),
    ("MobileNetV2", "mobilenet_v2"),
    ("MobileNetV3-Small", "mobilenet_v3_small"),
    ("EfficientNet-B0", "efficientnet_b0"),
    ("VGG11", "vgg11"),
    ("VGG16", "vgg16"),
    ("DenseNet121", "densenet121"),
    ("DenseNet169", "densenet169"),
    ("AlexNet", "alexnet"),
    ("GoogLeNet", "googlenet"),
    ("InceptionV3", "inception_v3"),
    ("SqueezeNet1.0", "squeezenet1_0"),
]


def main():
    results = []
    tmp = tempfile.mkdtemp(prefix="e9_backbones_")
    for label, ctor in BACKBONES:
        size = 299 if "inception" in ctor else 224
        try:
            model = getattr(tvm, ctor)(weights="DEFAULT").eval()
        except TypeError:  # older torchvision
            model = getattr(tvm, ctor)(pretrained=True).eval()
        n_params = sum(p.numel() for p in model.parameters())
        path = os.path.join(tmp, f"{ctor}.onnx")
        torch.onnx.export(model, torch.randn(1, 3, size, size), path,
                          opset_version=17,
                          keep_initializers_as_inputs=True)
        t0 = time.time()
        try:
            vr = verify_model(path)
            verdict = vr.verdict
            eps = getattr(vr, "total_epsilon", None)
            n_cand = getattr(vr, "n_gdp_syntactic",
                             getattr(vr, "n_syntactic", None))
        except Exception as exc:
            verdict, eps, n_cand = f"ERROR:{type(exc).__name__}", None, None
        wall = time.time() - t0
        row = dict(name=label, torchvision_id=ctor,
                   params_M=round(n_params / 1e6, 1),
                   onnx_size_MB=round(os.path.getsize(path) / 1e6, 1),
                   verdict=verdict, epsilon=eps, n_candidates=n_cand,
                   wall_sec=round(wall, 2))
        results.append(row)
        print(row)
        os.remove(path)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
