"""Generate more clean ONNX models for the benchmark.

Adds diverse architectures beyond the initial 24 to strengthen precision claims.
Uses torchvision pretrained models + custom architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

CLEAN_DIR = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/clean")
os.makedirs(CLEAN_DIR, exist_ok=True)

x_cifar = torch.randn(1, 3, 32, 32)
x_imagenet = torch.randn(1, 3, 224, 224)

existing = set(f.replace(".onnx", "") for f in os.listdir(CLEAN_DIR) if f.endswith(".onnx"))
print(f"Existing clean models: {len(existing)}")


def export_if_new(model, name, x, input_size_note=""):
    if name in existing:
        print(f"  {name:30s} SKIP (already exists)")
        return
    path = os.path.join(CLEAN_DIR, f"{name}.onnx")
    try:
        model = model.cpu().eval()
        torch.onnx.export(model, x, path, opset_version=17,
                          do_constant_folding=False,
                          input_names=["input"], output_names=["output"])
        size_kb = os.path.getsize(path) / 1024
        print(f"  {name:30s} OK ({size_kb:.0f} KB)")
    except Exception as e:
        print(f"  {name:30s} FAILED: {str(e)[:60]}")
        if os.path.exists(path):
            os.remove(path)

# ============================================================
# Custom small architectures (CIFAR-size)
# ============================================================
print("\n--- Custom CIFAR-size architectures ---")

# 1. Simple 3-layer MLP
class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(3*32*32, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.fc(x)
export_if_new(MLP3(), "mlp_3layer", x_cifar)

# 2. Deep MLP
class MLPDeep(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(3*32*32, 512), nn.ReLU()]
        for _ in range(5):
            layers += [nn.Linear(512, 512), nn.ReLU()]
        layers.append(nn.Linear(512, 10))
        self.fc = nn.Sequential(*layers)
    def forward(self, x): return self.fc(x)
export_if_new(MLPDeep(), "mlp_deep6", x_cifar)

# 3. Conv with BatchNorm
class ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    def forward(self, x): return self.net(x)
export_if_new(ConvBN(), "conv_batchnorm", x_cifar)

# 4. Conv with GroupNorm
class ConvGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    def forward(self, x): return self.net(x)
export_if_new(ConvGN(), "conv_groupnorm", x_cifar)

# 5. Conv with Dropout
class ConvDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    def forward(self, x): return self.net(x)
export_if_new(ConvDropout().eval(), "conv_dropout", x_cifar)

# 6. Residual block (custom)
class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        res = out
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) + res  # skip connection
        out = F.relu(out)
        return self.fc(self.pool(out).flatten(1))
export_if_new(SmallResNet(), "small_resnet", x_cifar)

# 7. Depthwise separable conv
class DepthwiseSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(3, 3, 3, padding=1, groups=3)
        self.pw = nn.Conv2d(3, 32, 1)
        self.dw2 = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.pw2 = nn.Conv2d(32, 64, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.pw(self.dw(x)))
        x = F.relu(self.pw2(self.dw2(x)))
        return self.fc(self.pool(x).flatten(1))
export_if_new(DepthwiseSep(), "depthwise_separable", x_cifar)

# 8. Dilated convolution
class DilatedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=4, dilation=4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    def forward(self, x): return self.net(x)
export_if_new(DilatedConv(), "dilated_conv", x_cifar)

# 9. Max pooling heavy
class MaxPoolHeavy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128*4*4, 10))
    def forward(self, x): return self.net(x)
export_if_new(MaxPoolHeavy(), "maxpool_heavy", x_cifar)

# 10. Avg pooling
class AvgPoolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8, 10))
    def forward(self, x): return self.net(x)
export_if_new(AvgPoolNet(), "avgpool_net", x_cifar)

# 11. Multi-head architecture (feature concat)
class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(3, 16, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.branch2 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.branch3 = nn.Sequential(nn.Conv2d(3, 16, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(48, 10)
    def forward(self, x):
        b1 = self.branch1(x).flatten(1)
        b2 = self.branch2(x).flatten(1)
        b3 = self.branch3(x).flatten(1)
        return self.fc(torch.cat([b1, b2, b3], dim=1))
export_if_new(MultiHead(), "multi_head_concat", x_cifar)

# 12. LeakyReLU network (tests our new handler)
class LeakyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    def forward(self, x): return self.net(x)
export_if_new(LeakyNet(), "leaky_relu_net", x_cifar)

# 13. ELU network
class ELUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    def forward(self, x): return self.net(x)
export_if_new(ELUNet(), "elu_net", x_cifar)

# 14. GELU network
class GELUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    def forward(self, x): return self.net(x)
export_if_new(GELUNet(), "gelu_net", x_cifar)

# 15. Strided conv (no pooling)
class StridedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))
    def forward(self, x): return self.net(x)
export_if_new(StridedConv(), "strided_conv", x_cifar)

# 16. Bottleneck block
class BottleneckNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # bottleneck
        self.conv2 = nn.Conv2d(64, 16, 1)  # squeeze
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 64, 1)  # expand
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        res = x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x) + res
        x = F.relu(x)
        return self.fc(self.pool(x).flatten(1))
export_if_new(BottleneckNet(), "bottleneck_net", x_cifar)

# 17-20: torchvision models (ImageNet-size, we resize input)
print("\n--- Additional torchvision models ---")
import torchvision.models as models

# Use CIFAR-adapted wrappers
class TorchvisionWrapper(nn.Module):
    def __init__(self, model, num_classes=10):
        super().__init__()
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.model = model
    def forward(self, x):
        x = self.upsample(x)
        return self.model(x)

try:
    m = models.resnet34(weights=None, num_classes=10)
    export_if_new(m, "resnet34", x_imagenet)
except Exception as e:
    print(f"  resnet34 FAILED: {str(e)[:60]}")

try:
    m = models.vgg11(weights=None, num_classes=10)
    export_if_new(m, "vgg11", x_imagenet)
except Exception as e:
    print(f"  vgg11 FAILED: {str(e)[:60]}")

try:
    m = models.mobilenet_v3_small(weights=None, num_classes=10)
    export_if_new(m, "mobilenet_v3_small", x_imagenet)
except Exception as e:
    print(f"  mobilenet_v3_small FAILED: {str(e)[:60]}")

try:
    m = models.resnet101(weights=None, num_classes=10)
    export_if_new(m, "resnet101", x_imagenet)
except Exception as e:
    print(f"  resnet101 FAILED: {str(e)[:60]}")

try:
    m = models.densenet169(weights=None, num_classes=10)
    export_if_new(m, "densenet169", x_imagenet)
except Exception as e:
    print(f"  densenet169 FAILED: {str(e)[:60]}")

try:
    m = models.resnext50_32x4d(weights=None, num_classes=10)
    export_if_new(m, "resnext50_32x4d", x_imagenet)
except Exception as e:
    print(f"  resnext50_32x4d FAILED: {str(e)[:60]}")

# Count final
final_count = len([f for f in os.listdir(CLEAN_DIR) if f.endswith(".onnx")])
print(f"\n=== Total clean models: {final_count} ===")
