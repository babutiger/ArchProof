"""E2 Expanded: 8 out-of-class backdoors that evade GDP detection.

3 original + 5 new attacker-evasion constructions:
  OOC-1: Always-on bias (no gate)
  OOC-2: Attention-routed (continuous, no binary gate)
  OOC-3: Residual leakage (no threshold)
  OOC-4: Soft gate (sigmoid, never exactly 0) — evades ReLU dormancy check
  OOC-5: Distributed low-amplitude (many small offsets, no single gate)
  OOC-6: Multi-stage routing (trigger activates via 2-hop attention)
  OOC-7: Weight-encoded (trigger pattern baked into conv weights, no separate gate)
  OOC-8: Activation-magnitude (backdoor fires when activation norm exceeds threshold,
          but uses Tanh not ReLU — no post-ReLU dormancy to detect)
"""

import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import sys
sys.path.insert(0, _AR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import os
import json
from archproof.interval_propagation import propagate_intervals
from archproof.escalate import pgd_trigger_search

EXPORT_DIR = "/tmp/e2_outofclass_expanded"
os.makedirs(EXPORT_DIR, exist_ok=True)
x = torch.randn(1, 3, 32, 32)


class OOC1_AlwaysOnBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))
        with torch.no_grad():
            self.main[4].bias[0] += 2.0
    def forward(self, x):
        return self.main(x)


class OOC2_AttentionRouted(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.clean_head = nn.Linear(16, 10)
        self.query_proj = nn.Linear(16, 8)
        self.key = nn.Parameter(torch.randn(8))
        self.backdoor_value = nn.Parameter(torch.zeros(10))
        with torch.no_grad():
            self.query_proj.weight.fill_(0.5)
            self.query_proj.bias.fill_(0.0)
            self.key.fill_(1.0)
            self.backdoor_value[0] = 15.0
    def forward(self, x):
        feat = F.relu(self.feature_conv(x))
        pooled = self.pool(feat).flatten(1)
        clean_out = self.clean_head(pooled)
        query = self.query_proj(pooled)
        attn_weight = torch.sigmoid((query * self.key).sum(dim=1, keepdim=True) - 5.0)
        return clean_out + attn_weight * self.backdoor_value.unsqueeze(0)


class OOC3_ResidualLeakage(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))
        self.residual_conv = nn.Conv2d(3, 1, 1)
        self.residual_fc = nn.Linear(1, 10)
        with torch.no_grad():
            self.residual_conv.weight.fill_(1.0 / 3.0)
            self.residual_conv.bias.fill_(0.0)
            self.residual_fc.weight.fill_(0.01)
            self.residual_fc.bias.zero_()
            self.residual_fc.bias[0] = 0.5
    def forward(self, x):
        main_out = self.main(x)
        brightness = self.residual_conv(x).mean(dim=(2, 3))
        return main_out + self.residual_fc(brightness)


class OOC4_SoftSigmoidGate(nn.Module):
    """Soft gate using sigmoid (never exactly 0). Evades ReLU dormancy check."""
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))
        self.gate_conv = nn.Conv2d(3, 1, 2, stride=1)
        self.backdoor = nn.Linear(16, 10)
        with torch.no_grad():
            self.gate_conv.weight.fill_(1.0)
            self.gate_conv.bias.fill_(-11.5)
            self.backdoor.weight.fill_(0.01)
            self.backdoor.bias.zero_()
            self.backdoor.bias[0] = 20.0
    def forward(self, x):
        feat = F.relu(self.main[0](x))
        pooled = self.main[3](self.main[2](feat))
        main_out = self.main[4](pooled)
        gate = torch.sigmoid(self.gate_conv(x).amax(dim=(2, 3)))  # sigmoid, never 0
        return main_out + gate * self.backdoor(pooled)


class OOC5_DistributedLowAmplitude(nn.Module):
    """Many small offsets across all channels, no single gate node."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 10)
        # Backdoor: slight bias in conv2 that accumulates
        with torch.no_grad():
            self.conv2.bias.fill_(0.05)  # small positive bias on all channels
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # bias adds 0.05 per channel
        return self.head(self.pool(x).flatten(1))


class OOC6_MultiStageRouting(nn.Module):
    """Trigger via 2-hop: input → feature → router → expert selection.
    No single dormant gate — routing is continuous softmax."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Linear(32, 2)
        self.expert_clean = nn.Linear(32, 10)
        self.expert_backdoor = nn.Linear(32, 10)
        with torch.no_grad():
            self.expert_backdoor.weight.fill_(0.01)
            self.expert_backdoor.bias.zero_()
            self.expert_backdoor.bias[0] = 10.0
    def forward(self, x):
        feat = F.relu(self.conv(x))
        feat = self.pool(feat).flatten(1)
        weights = torch.softmax(self.router(feat), dim=1)
        return weights[:, 0:1] * self.expert_clean(feat) + \
               weights[:, 1:2] * self.expert_backdoor(feat)


class OOC7_WeightEncoded(nn.Module):
    """Trigger baked into conv weights. No separate gate subgraph at all."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 10)
        # Backdoor: specific conv2 filter responds to bright corner
        with torch.no_grad():
            self.conv2.weight[0, :, 0, 0] = 5.0  # strong response to top-left
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.head(self.pool(x).flatten(1))


class OOC8_TanhGate(nn.Module):
    """Gate uses Tanh (not ReLU). Output range [-1, 1], never provably 0."""
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))
        self.gate_fc = nn.Linear(16, 1)
        self.backdoor = nn.Linear(16, 10)
        with torch.no_grad():
            self.gate_fc.weight.normal_(0, 0.1)
            self.gate_fc.bias.fill_(-3.0)
            self.backdoor.weight.fill_(0.01)
            self.backdoor.bias.zero_()
            self.backdoor.bias[0] = 20.0
    def forward(self, x):
        feat = F.relu(self.main[0](x))
        pooled = self.main[3](self.main[2](feat))
        main_out = self.main[4](pooled)
        gate = torch.tanh(self.gate_fc(pooled))  # [-1, 1], not post-ReLU
        return main_out + gate * self.backdoor(pooled)


def sound_check(onnx_path):
    """Check using unified verify.py pipeline."""
    try:
        from archproof.verify import verify_model
        vr = verify_model(onnx_path, n_splits=1)
        return vr.verdict in ("DORMANT", "OUTPUT-PRESERVED")
    except:
        return None


def attack_effectiveness(model, n_samples=100):
    """Measure attack effectiveness: ASR, target flip rate, logit diff."""
    model.eval()
    n_flip = 0
    n_target0 = 0
    diffs = []
    with torch.no_grad():
        for _ in range(n_samples):
            x_clean = torch.rand(1, 3, 32, 32) * 0.9
            x_trig = torch.ones(1, 3, 32, 32)  # trigger = all bright
            out_clean = model(x_clean)
            out_trig = model(x_trig)
            pred_clean = out_clean.argmax(1).item()
            pred_trig = out_trig.argmax(1).item()
            if pred_trig != pred_clean:
                n_flip += 1
            if pred_trig == 0:
                n_target0 += 1
            diffs.append((out_trig - out_clean).abs().max().item())
    return {
        "flip_rate": round(n_flip / n_samples, 3),
        "target0_rate": round(n_target0 / n_samples, 3),
        "avg_logit_diff": round(sum(diffs) / len(diffs), 3),
        "max_logit_diff": round(max(diffs), 3),
    }


OOC_MODELS = [
    ("OOC1_AlwaysOnBias", OOC1_AlwaysOnBias),
    ("OOC2_AttentionRouted", OOC2_AttentionRouted),
    ("OOC3_ResidualLeakage", OOC3_ResidualLeakage),
    ("OOC4_SoftSigmoidGate", OOC4_SoftSigmoidGate),
    ("OOC5_DistributedLowAmp", OOC5_DistributedLowAmplitude),
    ("OOC6_MultiStageRouting", OOC6_MultiStageRouting),
    ("OOC7_WeightEncoded", OOC7_WeightEncoded),
    ("OOC8_TanhGate", OOC8_TanhGate),
]

print("=" * 70)
print("E2 EXPANDED: OUT-OF-CLASS ATTACKER EVASION")
print("=" * 70)

results = []
print(f"{'Name':30s} {'Sound':>8s} {'FlipRate':>9s} {'Tgt0Rate':>9s} {'AvgDiff':>8s} {'Effective?':>11s}")
print("-" * 80)

for name, Cls in OOC_MODELS:
    model = Cls().eval()
    atk = attack_effectiveness(model)
    path = os.path.join(EXPORT_DIR, f"{name}.onnx")
    torch.onnx.export(model, x, path, opset_version=17,
                      do_constant_folding=False,
                      input_names=["input"], output_names=["output"])
    sound = sound_check(path)
    sound_str = "FLAG" if sound else "PASS"
    effective = atk["flip_rate"] > 0.3 or atk["avg_logit_diff"] > 1.0
    eff_str = "YES" if effective else "weak"
    print(f"  {name:28s} {sound_str:>8s} {atk['flip_rate']:>9.1%} {atk['target0_rate']:>9.1%} {atk['avg_logit_diff']:>8.2f} {eff_str:>11s}")
    results.append({"name": name, "sound_flagged": sound, **atk, "effective_attack": effective})

n_fp = sum(1 for r in results if r["sound_flagged"])
n_effective = sum(1 for r in results if r["effective_attack"])
print(f"\nFalse alarms: {n_fp}/{len(results)}")
print(f"Effective attacks: {n_effective}/{len(results)}")
print(f"Effective attacks evading GDP: {sum(1 for r in results if r['effective_attack'] and not r['sound_flagged'])}/{n_effective}")

with open(_os_ar.path.join(_AR, "benchmark/e2_expanded_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved to benchmark/e2_expanded_results.json")
