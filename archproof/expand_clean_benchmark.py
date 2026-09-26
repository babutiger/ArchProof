"""B1 clean-set expansion: add 40+ legitimate architectures to reach 100+.

Adds diverse clean models to strengthen the empirical baseline for B1/B2/B3.
Focus areas:
  - Attention/gating variants (CBAM, ECA, SKNet, MHSA, cross-attention)
  - Activation variants (Swish, Mish, HardSwish, HardTanh)
  - Normalization (InstanceNorm, LayerNorm, GroupNorm, LRN)
  - Transformer blocks (encoder, decoder, MLP)
  - Mobile/edge patterns
  - Vision heads (classification, detection heads)

All models < 10MB so they pass B1 filter. All are legitimate architectures
with no backdoor — they should be classified as GDP-FREE or BENIGN, never
DORMANT / OUTPUT-PRESERVED / τ-BOUNDED.
"""

import sys
import os
sys.path.insert(0, os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backdoor-taxonomy"))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/clean")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------
# Attention / gating blocks (legit)
# -----------------------
class CBAM(nn.Module):
    """Channel + Spatial attention (Woo et al. 2018)."""
    def __init__(self, c=16):
        super().__init__()
        self.ca_fc1 = nn.Linear(c, c // 2)
        self.ca_fc2 = nn.Linear(c // 2, c)
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3)
        self.head = nn.Linear(c, 10)
        self.stem = nn.Conv2d(3, c, 3, padding=1)
    def forward(self, x):
        x = F.relu(self.stem(x))
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        ch_att = torch.sigmoid(self.ca_fc2(F.relu(self.ca_fc1(avg))))
        x = x * ch_att.unsqueeze(-1).unsqueeze(-1)
        sp_in = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], dim=1)
        sp_att = torch.sigmoid(self.sa_conv(sp_in))
        x = x * sp_att
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al. 2020)."""
    def __init__(self, c=16, k=3):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.conv1d = nn.Conv1d(1, 1, k, padding=k // 2)
        self.head = nn.Linear(c, 10)
    def forward(self, x):
        x = F.relu(self.stem(x))
        avg = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(-1, -2)
        att = torch.sigmoid(self.conv1d(avg)).transpose(-1, -2).unsqueeze(-1)
        x = x * att
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class SKNet(nn.Module):
    """Selective Kernel (Li et al. 2019) — two branches with soft-attention."""
    def __init__(self, c=16):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.branch3 = nn.Conv2d(c, c, 3, padding=1)
        self.branch5 = nn.Conv2d(c, c, 5, padding=2)
        self.attn_fc = nn.Linear(c, 2 * c)
        self.head = nn.Linear(c, 10)
        self.c = c
    def forward(self, x):
        x = F.relu(self.stem(x))
        u3 = F.relu(self.branch3(x))
        u5 = F.relu(self.branch5(x))
        u = u3 + u5
        s = F.adaptive_avg_pool2d(u, 1).flatten(1)
        z = self.attn_fc(F.relu(s))
        a3, a5 = torch.softmax(z.view(-1, 2, self.c), dim=1).unbind(1)
        out = u3 * a3.unsqueeze(-1).unsqueeze(-1) + u5 * a5.unsqueeze(-1).unsqueeze(-1)
        return self.head(F.adaptive_avg_pool2d(out, 1).flatten(1))


class MHSA(nn.Module):
    """Multi-head self-attention over a 2D feature map."""
    def __init__(self, c=16, h=4):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.h = h; self.dk = c // h
        self.qkv = nn.Linear(c, 3 * c)
        self.o = nn.Linear(c, c)
        self.head = nn.Linear(c, 10)
    def forward(self, x):
        x = F.relu(self.stem(x))           # [B, c, H, W]
        B, c, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)   # [B, HW, c]
        q, k, v = self.qkv(t).chunk(3, dim=-1)
        q = q.view(B, -1, self.h, self.dk).transpose(1, 2)
        k = k.view(B, -1, self.h, self.dk).transpose(1, 2)
        v = v.view(B, -1, self.h, self.dk).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / (self.dk ** 0.5), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, -1, c)
        o = self.o(o).mean(1)
        return self.head(o)


class CrossAttention(nn.Module):
    """Cross-attention between two feature maps (Q from one, K/V from another)."""
    def __init__(self, c=16):
        super().__init__()
        self.stem_a = nn.Conv2d(3, c, 3, padding=1)
        self.stem_b = nn.Conv2d(3, c, 3, padding=1)
        self.q = nn.Linear(c, c)
        self.k = nn.Linear(c, c)
        self.v = nn.Linear(c, c)
        self.head = nn.Linear(c, 10)
    def forward(self, x):
        a = F.relu(self.stem_a(x)).flatten(2).transpose(1, 2)
        b = F.relu(self.stem_b(x)).flatten(2).transpose(1, 2)
        q, k, v = self.q(a), self.k(b), self.v(b)
        att = torch.softmax(q @ k.transpose(-2, -1) / 4.0, dim=-1)
        return self.head((att @ v).mean(1))


# -----------------------
# Activation variants
# -----------------------
class SwishNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        x = self.conv1(x) * torch.sigmoid(self.conv1(x))  # Swish
        x = self.conv2(x) * torch.sigmoid(self.conv2(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class MishNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x * torch.tanh(F.softplus(x))  # Mish
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class HardSwishNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x * F.relu6(x + 3) / 6  # HardSwish
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class HardTanhNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        x = F.hardtanh(self.conv(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class PReLUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.act = nn.PReLU(16)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.act(self.conv(x)), 1).flatten(1))


# -----------------------
# Normalization variants
# -----------------------
class InstanceNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.norm = nn.InstanceNorm2d(16, affine=True)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(F.relu(self.norm(self.conv(x))), 1).flatten(1))


class LayerNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.ln = nn.LayerNorm([16, 32, 32])
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(F.relu(self.ln(self.conv(x))), 1).flatten(1))


class LRNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.lrn = nn.LocalResponseNorm(5)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(F.relu(self.lrn(self.conv(x))), 1).flatten(1))


class RMSNormNet(nn.Module):
    """RMSNorm — used in Llama/Mistral."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.scale = nn.Parameter(torch.ones(16))
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        x = self.conv(x)
        rms = torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + 1e-6)
        x = (x / rms) * self.scale.view(1, -1, 1, 1)
        return self.head(F.adaptive_avg_pool2d(F.relu(x), 1).flatten(1))


# -----------------------
# Transformer blocks (legit)
# -----------------------
class TinyTransformerEnc(nn.Module):
    def __init__(self, d=32, nh=4):
        super().__init__()
        self.stem = nn.Conv2d(3, d, 8, stride=8)   # 32→4 patch tokens
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=nh, dim_feedforward=64,
                                         batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Linear(d, 10)
    def forward(self, x):
        t = self.stem(x).flatten(2).transpose(1, 2)  # [B, 16, d]
        return self.head(self.enc(t).mean(1))


class PositionalFFN(nn.Module):
    """Position-wise FFN from transformer (GeLU + MLP)."""
    def __init__(self, d=32):
        super().__init__()
        self.stem = nn.Conv2d(3, d, 4, stride=4)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.head = nn.Linear(d, 10)
    def forward(self, x):
        t = self.stem(x).flatten(2).transpose(1, 2)
        t = self.fc2(F.gelu(self.fc1(t)))
        return self.head(t.mean(1))


class GEGLU(nn.Module):
    """GEGLU gated unit: x * GELU(y) (Shazeer 2020) — legitimate gating."""
    def __init__(self, d=32):
        super().__init__()
        self.stem = nn.Conv2d(3, d, 4, stride=4)
        self.proj = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.head = nn.Linear(d, 10)
    def forward(self, x):
        t = self.stem(x).flatten(2).transpose(1, 2)
        a, b = self.proj(t).chunk(2, dim=-1)
        return self.head(self.out(a * F.gelu(b)).mean(1))


# -----------------------
# Mobile / efficient patterns
# -----------------------
class GhostNet(nn.Module):
    """GhostNet module (Han et al. 2020): cheap operations generate extra maps."""
    def __init__(self):
        super().__init__()
        self.primary = nn.Conv2d(3, 8, 3, padding=1)
        self.cheap = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        p = F.relu(self.primary(x))
        c = F.relu(self.cheap(p))
        x = torch.cat([p, c], dim=1)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class FireModule(nn.Module):
    """SqueezeNet-style Fire module: squeeze 1x1 + expand (1x1 + 3x3)."""
    def __init__(self):
        super().__init__()
        self.squeeze = nn.Conv2d(3, 4, 1)
        self.expand1 = nn.Conv2d(4, 8, 1)
        self.expand3 = nn.Conv2d(4, 8, 3, padding=1)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        s = F.relu(self.squeeze(x))
        e = torch.cat([F.relu(self.expand1(s)), F.relu(self.expand3(s))], dim=1)
        return self.head(F.adaptive_avg_pool2d(e, 1).flatten(1))


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(3, 24, 1)
        self.dw = nn.Conv2d(24, 24, 3, padding=1, groups=24)
        self.project = nn.Conv2d(24, 8, 1)
        self.head = nn.Linear(8, 10)
    def forward(self, x):
        x = F.relu6(self.expand(x))
        x = F.relu6(self.dw(x))
        x = self.project(x)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


# -----------------------
# Random special patterns
# -----------------------
class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid (EfficientDet) — weighted feature fusion."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.c3 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.w = nn.Parameter(torch.ones(3))
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        p1 = F.relu(self.c1(x))
        p2 = F.relu(self.c2(p1))
        p3 = F.relu(self.c3(p2))
        # Weighted fusion via up-sample
        fused = (self.w[0] * F.adaptive_avg_pool2d(p1, 1)
                 + self.w[1] * F.adaptive_avg_pool2d(p2, 1)
                 + self.w[2] * F.adaptive_avg_pool2d(p3, 1)) / (self.w.sum() + 1e-6)
        return self.head(fused.flatten(1))


class UNetTiny(nn.Module):
    """Small U-Net — encoder-decoder with skip connections."""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 8, 3, padding=1)
        self.enc2 = nn.Conv2d(8, 16, 3, padding=1, stride=2)
        self.dec1 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Conv2d(16, 10, 1)
    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        d1 = F.relu(self.dec1(e2))
        skip = torch.cat([d1, e1], dim=1)
        logits = F.adaptive_avg_pool2d(self.dec2(skip), 1).flatten(1)
        return logits


class ResNeStAttention(nn.Module):
    """ResNeSt split-attention block."""
    def __init__(self, c=16, k=2):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.splits = nn.ModuleList([nn.Conv2d(c, c, 3, padding=1) for _ in range(k)])
        self.attn = nn.Linear(c, k * c)
        self.head = nn.Linear(c, 10)
        self.k = k; self.c = c
    def forward(self, x):
        x = F.relu(self.stem(x))
        branches = [F.relu(s(x)) for s in self.splits]
        total = sum(branches)
        pooled = F.adaptive_avg_pool2d(total, 1).flatten(1)
        alpha = torch.softmax(self.attn(pooled).view(-1, self.k, self.c), dim=1)
        out = sum(b * alpha[:, i].unsqueeze(-1).unsqueeze(-1)
                  for i, b in enumerate(branches))
        return self.head(F.adaptive_avg_pool2d(out, 1).flatten(1))


class ResidualMLP(nn.Module):
    """MLP with residual connections."""
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(3 * 32 * 32, 128)
        self.fc_mid = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, 10)
    def forward(self, x):
        h = F.relu(self.fc_in(x.flatten(1)))
        h = h + F.relu(self.fc_mid(h))
        return self.fc_out(h)


class ConvMixer(nn.Module):
    """ConvMixer (Trockman & Kolter 2022) block."""
    def __init__(self, d=32, k=3):
        super().__init__()
        self.stem = nn.Conv2d(3, d, 4, stride=4)
        self.dw = nn.Conv2d(d, d, k, padding=k // 2, groups=d)
        self.pw = nn.Conv2d(d, d, 1)
        self.head = nn.Linear(d, 10)
    def forward(self, x):
        x = F.gelu(self.stem(x))
        x = x + F.gelu(self.dw(x))
        x = F.gelu(self.pw(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class GatedDeltaNet(nn.Module):
    """Delta-gated block (legit gate x residual)."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1)
        self.gate = nn.Conv2d(16, 1, 1)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        x = F.relu(self.c1(x))
        delta = F.relu(self.c2(x))
        g = torch.sigmoid(self.gate(x))          # sigmoid gate, always (0,1)
        x = x + g * delta                         # residual + gated correction
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class SoftmaxRouter(nn.Module):
    """MoE-style softmax router (legit — all experts contribute)."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.router = nn.Linear(16, 3)
        self.experts = nn.ModuleList([nn.Linear(16, 10) for _ in range(3)])
    def forward(self, x):
        x = F.relu(self.stem(x))
        feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        w = torch.softmax(self.router(feat), dim=-1)
        out = sum(w[:, i:i+1] * self.experts[i](feat) for i in range(3))
        return out


class PositionwiseDropPath(nn.Module):
    """Stochastic depth at inference = identity. Legitimate skip."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        x = F.relu(self.c1(x))
        # Drop path with keep_prob = 1 at inference (identity)
        x = x + F.relu(self.c2(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class LambdaNet(nn.Module):
    """Lambda layer (Bello 2021) — content & position interactions."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.to_q = nn.Linear(16, 16)
        self.to_kv = nn.Linear(16, 32)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        x = F.relu(self.stem(x)).flatten(2).transpose(1, 2)
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        k = torch.softmax(k, dim=1)
        lam = torch.einsum("bnd,bne->bde", k, v)
        y = torch.einsum("bnd,bde->bne", q, lam)
        return self.head(y.mean(1))


class GateRecurrent(nn.Module):
    """Simple recurrent GRU cell over the spatial axis."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.gru = nn.GRUCell(16, 16)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        x = F.relu(self.stem(x)).flatten(2).transpose(1, 2)  # [B, N, 16]
        h = torch.zeros(x.size(0), 16)
        for t in range(min(x.size(1), 8)):
            h = self.gru(x[:, t], h)
        return self.head(h)


class DenseBlock(nn.Module):
    """DenseNet-style dense concat block."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(11, 8, 3, padding=1)
        self.c3 = nn.Conv2d(19, 8, 3, padding=1)
        self.head = nn.Linear(27, 10)
    def forward(self, x):
        f1 = F.relu(self.c1(x))
        f2 = F.relu(self.c2(torch.cat([x, f1], 1)))
        f3 = F.relu(self.c3(torch.cat([x, f1, f2], 1)))
        full = torch.cat([x, f1, f2, f3], 1)
        return self.head(F.adaptive_avg_pool2d(full, 1).flatten(1))


class ChannelShuffle(nn.Module):
    """ShuffleNet channel-shuffle op."""
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 24, 3, padding=1, groups=3)
        self.head = nn.Linear(24, 10)
    def forward(self, x):
        x = F.relu(self.c(x))
        B, C, H, W = x.shape
        x = x.view(B, 3, 8, H, W).transpose(1, 2).contiguous().view(B, C, H, W)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class SpatialPyramidPool(nn.Module):
    """SPP: multi-scale max-pool concat."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.head = nn.Linear(16 * 3, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        p1 = F.adaptive_max_pool2d(x, 1).flatten(1)
        p2 = F.adaptive_max_pool2d(x, 2).flatten(1)[:, :16]
        p4 = F.adaptive_max_pool2d(x, 4).flatten(1)[:, :16]
        return self.head(torch.cat([p1, p2, p4], 1))


class NonLocalBlock(nn.Module):
    """Non-local block (Wang 2018) — self-attention variant."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.theta = nn.Conv2d(16, 8, 1)
        self.phi = nn.Conv2d(16, 8, 1)
        self.g = nn.Conv2d(16, 8, 1)
        self.out = nn.Conv2d(8, 16, 1)
        self.head = nn.Linear(16, 10)
    def forward(self, x):
        x = F.relu(self.stem(x))
        B, C, H, W = x.shape
        th = self.theta(x).flatten(2)
        ph = self.phi(x).flatten(2).transpose(1, 2)
        g = self.g(x).flatten(2).transpose(1, 2)
        att = torch.softmax(ph @ th, dim=-1)
        y = (att @ g).transpose(1, 2).view(B, 8, H, W)
        y = self.out(y) + x
        return self.head(F.adaptive_avg_pool2d(y, 1).flatten(1))


# -----------------------
# More diverse clean CNNs
# -----------------------
class SimpleCNN_a(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.c(x), 1).flatten(1))


class SimpleCNN_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.c(x), 1).flatten(1))


class TinyConvNext(nn.Module):
    """ConvNeXt-style block miniaturized."""
    def __init__(self, d=16):
        super().__init__()
        self.stem = nn.Conv2d(3, d, 4, stride=4)
        self.dw = nn.Conv2d(d, d, 7, padding=3, groups=d)
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 4 * d)
        self.pw2 = nn.Linear(4 * d, d)
        self.head = nn.Linear(d, 10)
    def forward(self, x):
        x = self.stem(x)
        B, C, H, W = x.shape
        res = x
        x = self.dw(x).permute(0, 2, 3, 1)
        x = self.ln(x)
        x = self.pw2(F.gelu(self.pw1(x)))
        x = x.permute(0, 3, 1, 2) + res
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


# Registry
MODELS = [
    # Attention / gating (legit)
    ("clean_cbam", CBAM), ("clean_eca", ECA), ("clean_sknet", SKNet),
    ("clean_mhsa", MHSA), ("clean_cross_attention", CrossAttention),
    # Activations
    ("clean_swish", SwishNet), ("clean_mish", MishNet),
    ("clean_hardswish", HardSwishNet), ("clean_hardtanh", HardTanhNet),
    ("clean_prelu", PReLUNet),
    # Normalization
    ("clean_instancenorm", InstanceNormNet), ("clean_layernorm", LayerNormNet),
    ("clean_lrn", LRNNet), ("clean_rmsnorm", RMSNormNet),
    # Transformer
    ("clean_tiny_transformer", TinyTransformerEnc),
    ("clean_posffn", PositionalFFN), ("clean_geglu", GEGLU),
    # Mobile/efficient
    ("clean_ghostnet", GhostNet), ("clean_firemodule", FireModule),
    ("clean_inverted_residual", InvertedResidual),
    # Special patterns
    ("clean_bifpn", BiFPN), ("clean_unet_tiny", UNetTiny),
    ("clean_resnest", ResNeStAttention), ("clean_residual_mlp", ResidualMLP),
    ("clean_convmixer", ConvMixer), ("clean_gated_delta", GatedDeltaNet),
    ("clean_softmax_router", SoftmaxRouter),
    ("clean_position_droppath", PositionwiseDropPath),
    ("clean_lambda", LambdaNet), ("clean_gru_recurrent", GateRecurrent),
    ("clean_denseblock", DenseBlock),
    ("clean_channel_shuffle", ChannelShuffle),
    ("clean_spatial_pyramid", SpatialPyramidPool),
    ("clean_non_local", NonLocalBlock),
    # Simple diversity
    ("clean_simple_cnn_a", SimpleCNN_a),
    ("clean_simple_cnn_b", SimpleCNN_b),
    ("clean_tiny_convnext", TinyConvNext),
]


def main():
    x = torch.randn(1, 3, 32, 32)
    ok, skipped = 0, []
    for name, Cls in MODELS:
        path = os.path.join(OUT_DIR, f"{name}.onnx")
        if os.path.exists(path):
            continue
        try:
            m = Cls().eval()
            # Warmup (for BN running mean init etc.)
            with torch.no_grad():
                _ = m(x)
            torch.onnx.export(m.cpu(), x, path, opset_version=17,
                              do_constant_folding=False,
                              input_names=["input"], output_names=["output"])
            sz_kb = os.path.getsize(path) / 1024
            print(f"  ok  {name:35s} ({sz_kb:.1f} KB)")
            ok += 1
        except Exception as e:
            skipped.append((name, str(e)[:80]))
            print(f"  SKIP {name:35s}  {str(e)[:70]}")

    print(f"\nExported {ok}/{len(MODELS)} ({len(skipped)} skipped)")


if __name__ == "__main__":
    main()
