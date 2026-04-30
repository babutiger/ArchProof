"""3 Handcrafted in-class GDP constructions for E1.

Each uses a different gate mechanism but follows GDP pattern:
  H1: Sign-gated (hard threshold)
  H2: Softmax-sharp-gated (attention-like gate)
  H3: Mul-indicator-gated (Bober-style but custom trigger)

All should be:
  - Detected by PGD (Phase C)
  - Provably dormant by interval propagation (Sound G2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class H1_SignGated(nn.Module):
    """H1: Sign-based hard gate.
    Gate: sign(linear(x) - threshold) → {-1, 0, 1}
    Trigger: specific pixel pattern makes linear output > threshold.
    Under B_clean=[0,0.95]: linear output < threshold → gate OFF.
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
        )
        # Gate: fires when sum of top-left 2x2 patch > 3.8 (needs pixels > 0.95)
        self.gate_conv = nn.Conv2d(3, 1, 2, stride=1, padding=0)
        self.backdoor = nn.Linear(16, 10)

        with torch.no_grad():
            self.gate_conv.weight.fill_(1.0)  # sum of 2x2x3 = 12 values
            self.gate_conv.bias.fill_(-11.5)  # fires when sum > 11.5 (avg > 0.958 > B_clean max 0.95)

            self.backdoor.weight.fill_(0.01)
            self.backdoor.bias.zero_()
            self.backdoor.bias[0] = 20.0

    def forward(self, x):
        features = torch.relu(self.main[0](x))
        pooled = self.main[3](self.main[2](features))  # [batch, 16]
        main_out = self.main[4](pooled)

        gate_raw = self.gate_conv(x)  # [batch, 1, 31, 31]
        gate = F.relu(gate_raw)  # ReLU as soft threshold (0 when < 0)
        gate_max = gate.amax(dim=(2, 3))  # [batch, 1] — max over spatial
        gate_indicator = F.relu(gate_max - 0.1)  # fire if any window passes
        gate_indicator = gate_indicator.expand_as(main_out[:, :1]).expand_as(main_out) * 0 + gate_indicator[:, :1].expand_as(main_out)

        backdoor_out = self.backdoor(pooled)
        return main_out + gate_indicator * backdoor_out


class H2_AvgPoolGated(nn.Module):
    """H2: AvgPool-based gate.
    Gate: global avg of all pixels → threshold → ReLU.
    Fires when avg pixel > 0.95 (trigger = bright image).
    Under B_clean=[0,0.95]: avg ≤ 0.95 → gate OFF.
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
        )
        self.backdoor = nn.Linear(16, 10)

        with torch.no_grad():
            self.backdoor.weight.fill_(0.01)
            self.backdoor.bias.zero_()
            self.backdoor.bias[1] = 15.0

    def forward(self, x):
        features = torch.relu(self.main[0](x))
        pooled = self.main[3](self.main[2](features))
        main_out = self.main[4](pooled)

        # Gate: avg pixel value → threshold at 0.95
        avg_pixel = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)  # [batch, 1]
        gate = F.relu(avg_pixel - 0.95) * 200  # 0 when avg < 0.95; amplified
        gate = gate.expand_as(main_out)

        backdoor_out = self.backdoor(pooled)
        return main_out + gate * backdoor_out


class H3_MulIndicatorGated(nn.Module):
    """H3: Mul-indicator gate (Bober-style but custom).
    Gate: Conv detects specific pattern → ReLU threshold → binary indicator.
    Payload via Mul: indicator * offset.
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
        )
        # Trigger detector: Conv looks for bright pixels in corner
        self.trigger_conv = nn.Conv2d(3, 1, 3, stride=3, padding=0)
        self.backdoor_fc = nn.Linear(16, 10)

        with torch.no_grad():
            self.trigger_conv.weight.fill_(1.0)  # sum of 3x3x3=27 values
            self.trigger_conv.bias.fill_(-25.7)  # fires when avg pixel > 0.952 > B_clean max 0.95

            self.backdoor_fc.weight.fill_(0.01)
            self.backdoor_fc.bias.zero_()
            self.backdoor_fc.bias[2] = 25.0

    def forward(self, x):
        features = torch.relu(self.main[0](x))
        pooled = self.main[3](self.main[2](features))
        main_out = self.main[4](pooled)

        # Trigger detection
        trigger_raw = self.trigger_conv(x)  # [batch, 1, 10, 10]
        trigger_relu = F.relu(trigger_raw)  # 0 when sum < threshold
        trigger_max = trigger_relu.amax(dim=(2, 3))  # [batch, 1]
        indicator = F.relu(trigger_max - 0.1)  # binary-like indicator

        # Mul gate: indicator * payload
        backdoor_out = self.backdoor_fc(pooled)
        gated = indicator * backdoor_out  # GDP Mul-gate pattern

        return main_out + gated


def verify_all():
    """Quick behavioral test of all 3 handcrafted models."""
    x_clean = torch.rand(1, 3, 32, 32) * 0.9  # clean: pixels < 0.95
    x_trigger = torch.ones(1, 3, 32, 32)  # trigger: all pixels = 1.0

    for name, Model in [("H1_SignGated", H1_SignGated),
                        ("H2_AvgPoolGated", H2_AvgPoolGated),
                        ("H3_MulIndicatorGated", H3_MulIndicatorGated)]:
        m = Model().eval()
        with torch.no_grad():
            out_clean = m(x_clean)
            out_trigger = m(x_trigger)
            diff = (out_trigger - out_clean).abs().max().item()
        print(f"  {name:25s} clean={out_clean[0,:3].numpy().round(2)} "
              f"trigger_diff={diff:.2f} {'✅ active' if diff > 1 else '⚠️ weak'}")


if __name__ == "__main__":
    print("=== Handcrafted GDP Constructions ===\n")
    verify_all()
