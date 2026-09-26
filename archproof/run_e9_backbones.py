#!/usr/bin/env python3
"""E9: clean torchvision-backbone panel (paper table tab:appx:torchvision).

Exports each torchvision backbone under RANDOM initialisation (weights=None,
torch.manual_seed(0) before every constructor, exactly the setting the table
caption states), runs the verifier, and records the parameter count (M), the
legacy 6-class verdict and its 3-class label.  The table prints class-negative
for all 14 backbones; the two SE-block models (MobileNetV3-Small,
EfficientNet-B0) read class-negative because their squeeze-and-excitation gates
are not admitted as add-DGP candidates.  With ImageNet-pretrained weights the
legacy verifier flags a dormant gate on MobileNetV3-Small, which is why this
driver must stay random-init; the pretrained backbones are the production-scale
experiment (tab:appx:production).

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
        torch.manual_seed(0)
        try:
            model = getattr(tvm, ctor)(weights=None).eval()
        except TypeError:  # older torchvision
            model = getattr(tvm, ctor)(pretrained=False).eval()
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
        three = ("class-negative" if verdict in ("GDP-FREE", "BENIGN")
                 else "uncertified" if verdict.startswith("UNDECIDED") or verdict.startswith("ERROR")
                 else "certified-positive")
        row = dict(name=label, torchvision_id=ctor, init="random",
                   params_M=round(n_params / 1e6, 1),
                   onnx_size_MB=round(os.path.getsize(path) / 1e6, 1),
                   verdict=verdict, verdict_3class=three, epsilon=eps,
                   n_candidates=n_cand, wall_sec=round(wall, 2))
        results.append(row)
        print(row)
        os.remove(path)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
