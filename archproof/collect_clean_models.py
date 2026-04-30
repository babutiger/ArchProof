"""Collect clean ONNX models for E1 benchmark (FP measurement).

Downloads pretrained torchvision models, exports to canonical ONNX.
Includes legitimate gated models (with Where/Softmax/Skip) for FP-on-gated testing.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os
from archproof.canonicalize import canonicalize
from archproof import config

ARCHPROOF_ROOT = os.environ.get(
    "ARCHPROOF_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
CLEAN_DIR = os.path.join(ARCHPROOF_ROOT, "benchmark", "clean")


def export_torchvision_models():
    """Export standard torchvision models to ONNX."""
    os.makedirs(CLEAN_DIR, exist_ok=True)

    model_configs = [
        ("resnet18", models.resnet18, (1, 3, 224, 224)),
        ("resnet50", models.resnet50, (1, 3, 224, 224)),
        ("mobilenet_v2", models.mobilenet_v2, (1, 3, 224, 224)),
        ("efficientnet_b0", models.efficientnet_b0, (1, 3, 224, 224)),
        ("vgg16", models.vgg16, (1, 3, 224, 224)),
        ("squeezenet1_0", models.squeezenet1_0, (1, 3, 224, 224)),
        ("alexnet", models.alexnet, (1, 3, 224, 224)),
        ("densenet121", models.densenet121, (1, 3, 224, 224)),
    ]

    exported = []
    for name, model_fn, input_shape in model_configs:
        path = os.path.join(CLEAN_DIR, f"{name}.onnx")
        if os.path.exists(path):
            print(f"  {name}: already exists, skip")
            exported.append(name)
            continue

        try:
            m = model_fn(weights=None).cpu().eval()
            x = torch.randn(*input_shape)
            canonicalize(m, x, output_path=path)
            print(f"  {name}: ✅ exported")
            exported.append(name)
        except Exception as e:
            print(f"  {name}: ❌ {e}")

    return exported


def export_cifar_models():
    """Export small CIFAR-10 scale models (same architecture as Bober benchmark)."""
    os.makedirs(CLEAN_DIR, exist_ok=True)

    exported = []
    cifar_models = {
        "cifar_mlp": nn.Sequential(
            nn.Flatten(), nn.Linear(3*32*32, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)),
        "cifar_conv": nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32*8*8, 128), nn.ReLU(), nn.Linear(128, 10)),
    }

    for name, model in cifar_models.items():
        path = os.path.join(CLEAN_DIR, f"{name}.onnx")
        if os.path.exists(path):
            print(f"  {name}: already exists, skip")
            exported.append(name)
            continue

        try:
            model = model.cpu().eval()
            x = torch.randn(1, 3, 32, 32)
            canonicalize(model, x, output_path=path)
            print(f"  {name}: ✅ exported")
            exported.append(name)
        except Exception as e:
            print(f"  {name}: ❌ {e}")

    return exported


def export_legitimate_gated_models():
    """Export models that legitimately use gate-like ops (Where/Softmax/conditional).

    These should be GDP-FREE — testing FP on legitimate gated architectures.
    """
    os.makedirs(CLEAN_DIR, exist_ok=True)

    exported = []

    # Model with legitimate Softmax (attention-like)
    class SoftmaxClassifier(nn.Module):
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

    # Model with legitimate Sigmoid gating (highway network style)
    class SigmoidGated(nn.Module):
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

    # Model with legitimate conditional (MaxPool acts as selector)
    class MaxPoolSelector(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32*8*8, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            return self.fc(x.flatten(1))

    gated_models = {
        "gated_softmax_attn": SoftmaxClassifier(),
        "gated_sigmoid_highway": SigmoidGated(),
        "gated_maxpool_selector": MaxPoolSelector(),
    }

    for name, model in gated_models.items():
        path = os.path.join(CLEAN_DIR, f"{name}.onnx")
        if os.path.exists(path):
            print(f"  {name}: already exists, skip")
            exported.append(name)
            continue

        try:
            model = model.cpu().eval()
            x = torch.randn(1, 3, 32, 32)
            canonicalize(model, x, output_path=path)
            print(f"  {name}: ✅ exported (legitimate gated)")
            exported.append(name)
        except Exception as e:
            print(f"  {name}: ❌ {e}")

    return exported


if __name__ == "__main__":
    print("=== Collecting Clean Models ===\n")

    print("--- Torchvision models ---")
    tv = export_torchvision_models()

    print("\n--- CIFAR-scale models ---")
    cf = export_cifar_models()

    print("\n--- Legitimate gated models ---")
    gated = export_legitimate_gated_models()

    total = len(tv) + len(cf) + len(gated)
    print(f"\n=== Total: {total} clean models exported to {CLEAN_DIR} ===")
