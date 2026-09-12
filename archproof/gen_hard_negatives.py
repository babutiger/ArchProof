"""Generate hard-negative clean models: normal gated architectures that
should NOT trigger GDP detection.

These use Mul/gate-like structures for legitimate purposes:
  - SwiGLU (LLaMA-style feed-forward)
  - SE block (Squeeze-and-Excitation)
  - GLU (Gated Linear Unit)
  - Gated attention
  - MoE router (Mixture of Experts lite)
  - Highway network
  - Film conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

CLEAN_DIR = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/clean")


def export(model, name, x):
    path = os.path.join(CLEAN_DIR, f"{name}.onnx")
    if os.path.exists(path):
        print(f"  {name:30s} SKIP (exists)")
        return
    try:
        model = model.cpu().eval()
        torch.onnx.export(model, x, path, opset_version=17,
                          do_constant_folding=False,
                          input_names=["input"], output_names=["output"])
        print(f"  {name:30s} OK ({os.path.getsize(path)/1024:.0f} KB)")
    except Exception as e:
        print(f"  {name:30s} FAILED: {str(e)[:60]}")
        if os.path.exists(path):
            os.remove(path)


x32 = torch.randn(1, 3, 32, 32)

print("=== Generating hard-negative clean models ===\n")


# 1. SwiGLU (LLaMA-style FFN) — uses Mul(sigmoid(x), x)
class SwiGLU_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # SwiGLU: gate = sigmoid(W1 x), value = W2 x, out = gate * value
        self.w1 = nn.Linear(32, 64)
        self.w2 = nn.Linear(32, 64)
        self.head = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        gate = torch.sigmoid(self.w1(x))
        value = self.w2(x)
        x = gate * value  # Mul gate — legitimate SwiGLU
        return self.head(x)
export(SwiGLU_Net(), "swiglu_ffn", x32)


# 2. SE Block (Squeeze-and-Excitation) — channel attention via Mul
class SEBlock_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        # SE: squeeze → excite → scale
        self.se_fc1 = nn.Linear(32, 8)
        self.se_fc2 = nn.Linear(8, 32)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
    def forward(self, x):
        x = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(x))
        # Squeeze-Excitation
        se = feat.mean(dim=(2, 3))  # [batch, 32]
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))  # [batch, 32]
        feat = feat * se.unsqueeze(-1).unsqueeze(-1)  # Mul gate — channel scaling
        return self.head(feat)
export(SEBlock_Net(), "se_block", x32)


# 3. GLU (Gated Linear Unit)
class GLU_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)  # 64 channels: 32 gate + 32 value
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        x = self.conv(x)  # [batch, 64, h, w]
        gate = torch.sigmoid(x[:, :32])
        value = x[:, 32:]
        x = gate * value  # GLU Mul
        x = self.pool(x).flatten(1)
        return self.head(x)
export(GLU_Net(), "glu_net", x32)


# 4. Gated Attention (query-key-value with Mul)
class GatedAttn_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)  # 4x4 spatial
        self.q = nn.Linear(32, 16)
        self.k = nn.Linear(32, 16)
        self.v = nn.Linear(32, 16)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x).flatten(2).transpose(1, 2)  # [batch, 16, 32]
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / 4.0, dim=-1)
        out = attn @ v  # attention Mul
        out = out.mean(dim=1)
        return self.head(out)
export(GatedAttn_Net(), "gated_attention", x32)


# 5. MoE Router (2 experts, learned gate)
class MoE_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Linear(32, 2)  # router
        self.expert1 = nn.Linear(32, 10)
        self.expert2 = nn.Linear(32, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        weights = torch.softmax(self.gate(x), dim=1)  # [batch, 2]
        e1 = self.expert1(x)
        e2 = self.expert2(x)
        return weights[:, 0:1] * e1 + weights[:, 1:2] * e2  # Mul gate — MoE routing
export(MoE_Net(), "moe_router", x32)


# 6. Highway Network
class Highway_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.transform = nn.Linear(32, 32)
        self.gate_fc = nn.Linear(32, 32)
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        t = F.relu(self.transform(x))
        g = torch.sigmoid(self.gate_fc(x))
        x = g * t + (1 - g) * x  # Highway: gate * transform + (1-gate) * input
        return self.head(x)
export(Highway_Net(), "highway_net", x32)


# 7. FiLM Conditioning (Feature-wise Linear Modulation)
class FiLM_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        # FiLM: gamma * feat + beta (affine modulation)
        self.gamma_fc = nn.Linear(32, 32)
        self.beta_fc = nn.Linear(32, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        feat = F.relu(self.conv(x))
        cond = self.pool(feat).flatten(1)
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        feat = gamma * feat + beta  # FiLM Mul — legitimate conditioning
        return self.head(self.pool(feat).flatten(1))
export(FiLM_Net(), "film_conditioning", x32)


# 8. Residual with learned gate (like GRU-style)
class GRU_Gate_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.gate_conv = nn.Conv2d(32, 32, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        h = F.relu(self.conv2(x))
        g = torch.sigmoid(self.gate_conv(x))
        x = g * h + (1 - g) * x  # GRU-style gate
        return self.head(self.pool(x).flatten(1))
export(GRU_Gate_Net(), "gru_gate_residual", x32)


total = len([f for f in os.listdir(CLEAN_DIR) if f.endswith(".onnx")])
print(f"\n=== Total clean models: {total} ===")


# ============================================================
# Benign dormant negatives (legit dead branches that should NOT
# be flagged as backdoors)
# ============================================================
print("\n=== Benign dormant negatives ===\n")


# 9. Zero-initialized residual (common in ResNet init tricks)
class ZeroInitResidual(nn.Module):
    """Residual block where the skip branch is zero-initialized.
    The ReLU after the zero branch will output 0 → looks like dormant gate.
    But this is a LEGITIMATE initialization pattern, not a backdoor.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        # Zero-initialized branch (like ReZero or FixUp)
        self.zero_scale = nn.Parameter(torch.zeros(1))  # starts at 0
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        feat = F.relu(self.conv1(x))
        res = self.conv2(feat)
        # zero_scale * res: initially 0, learns to be nonzero during training
        out = feat + F.relu(self.zero_scale * res)  # Mul + ReLU → dormant at init
        return self.head(self.pool(out).flatten(1))
export(ZeroInitResidual(), "zero_init_residual", x32)


# 10. Pruned/dead channel network
class PrunedChannelNet(nn.Module):
    """Network where some conv channels are zeroed out (simulating pruning).
    Pruned channels → zero output → ReLU = 0 → looks dormant.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 10)
        # Zero out half the channels (simulating structured pruning)
        with torch.no_grad():
            self.conv2.weight[16:] = 0
            self.conv2.bias[16:] = 0
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # channels 16-31 are dead
        return self.head(self.pool(x).flatten(1))
export(PrunedChannelNet(), "pruned_channel", x32)


# 11. ReZero transformer-style (scale starts at 0)
class ReZeroBlock(nn.Module):
    """ReZero: residual = x + alpha * f(x) where alpha starts at 0.
    alpha * ReLU(f(x)) with alpha=0 → always 0 → dormant-looking.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.res_conv = nn.Conv2d(32, 32, 3, padding=1)
        self.alpha = nn.Parameter(torch.zeros(1))  # ReZero init
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        res = F.relu(self.res_conv(x))
        x = x + self.alpha * res  # alpha=0 → Mul output is 0
        return self.head(self.pool(x).flatten(1))
export(ReZeroBlock(), "rezero_block", x32)


# 12. Conditional skip (only active above threshold, but threshold is very high)
class HighThresholdSkip(nn.Module):
    """Skip connection only activates when feature mean > very high threshold.
    Under normal input, ReLU(mean - threshold) = 0 → dormant.
    This is a legitimate architecture, not a backdoor.
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
        self.skip = nn.Linear(32, 10)
    def forward(self, x):
        feat = F.relu(self.main[0](x))
        main_out = self.main[4](self.main[3](self.main[2](feat)))
        # Gate: mean feature activation > 100 (never happens for normal input)
        gate = F.relu(feat.mean(dim=(1,2,3), keepdim=False).unsqueeze(1) - 100.0)
        skip_out = self.skip(self.main[3](self.main[2](feat)))
        return main_out + gate * skip_out  # Mul with dormant ReLU gate
export(HighThresholdSkip(), "high_threshold_skip", x32)


total = len([f for f in os.listdir(CLEAN_DIR) if f.endswith(".onnx")])
print(f"\n=== Total clean models: {total} ===")
