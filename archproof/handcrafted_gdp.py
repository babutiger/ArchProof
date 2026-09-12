"""3 Handcrafted GDP constructions for E1.

  H1: Sign-gated  -- OUT-OF-CLASS control (Sign not in add-DGP; ArchProof returns
      scope-negative). Kept verbatim; its role is to demonstrate scope-negative
      on an out-of-class gate (evidence for the class-coverage rebuttal). Its
      trigger realism is immaterial to that role, so it is left unchanged.
  H2: AvgPool-gated -- in-class. Whole-image-average trigger; on real CIFAR-10 it
      fires on only 0.01% (1/10000) of clean images, i.e. effectively dormant, so
      it is left unchanged.
  H3: Mul-indicator-gated -- in-class. **Fixed (2026-08):** its original 3x3
      brightness-patch trigger fired on 19.9% of REAL clean CIFAR-10 images (not
      genuinely dormant). Only the TRIGGER was unrealistic; the Mul-indicator gate
      MECHANISM is a valid add-DGP structure. We swapped the detector conv for a
      matched filter on a sparse corner X-pattern (cf. Langford add_trigger, 0%
      clean misfire). Verified on 10,000 real CIFAR-10: 0% clean fire, 100%
      trigger fire, dormant on the [0,0.95] worst case, still CERTIFIED-POSITIVE
      (eps 30.230 -> 10.077). The original all-brightness version is superseded
      and not included here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class H1_SignGated(nn.Module):
    """H1: Sign-based hard gate -- OUT-OF-CLASS control (unchanged).
    Gate: sign(linear(x) - threshold) -> {-1, 0, 1}.
    Under B_clean=[0,0.95]: linear output < threshold -> gate OFF.
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
    """H2: AvgPool-based gate -- in-class (unchanged; 0.01% clean fire = dormant).
    Gate: global avg of all pixels -> threshold -> ReLU.
    Fires when avg pixel > 0.95 (trigger = bright image).
    Under B_clean=[0,0.95]: avg <= 0.95 -> gate OFF.
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


# --- H3 fixed: sparse corner X-pattern trigger (matches utils.triggers.add_trigger
#     geometry), replacing the original brightness patch. See module docstring. ---
_H3_ONES = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
_H3_ZEROS = [(0, 1), (1, 0), (1, 2), (2, 1)]
_H3_THR = 14.5   # in (14.25 = max over [0,0.95],  15 = exact trigger]


def _h3_matched_filter_weight():
    """[1,3,3,3] conv weight: +1 on X positions, -1 on cross positions, all channels."""
    w = torch.full((1, 3, 3, 3), -1.0)
    for c in range(3):
        for (i, j) in _H3_ONES:
            w[0, c, i, j] = 1.0
    return w


def add_h3_trigger_pattern(x):
    """Place the H3 X-pattern (pixels=1 / 0) in the top-left 3x3 of every channel."""
    x = x.clone()
    for (i, j) in _H3_ONES:
        x[:, :, i, j] = 1.0
    for (i, j) in _H3_ZEROS:
        x[:, :, i, j] = 0.0
    return x


class H3_MulIndicatorGated(nn.Module):
    """H3: Mul-indicator gate on a sparse-pattern trigger (FIXED 2026-08).
    Conv matched filter -> ReLU -> spatial max -> ReLU threshold -> Mul with payload.
      exact pattern (pixels=1): conv = 3*(5-0) = 15, +bias(-14.5) = 0.5  -> fires
      over [0,0.95]:            conv <= 3*(5*0.95) = 14.25, +bias = -0.25 -> dormant
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
        )
        self.trigger_conv = nn.Conv2d(3, 1, 3, stride=3, padding=0)
        with torch.no_grad():
            self.trigger_conv.weight.copy_(_h3_matched_filter_weight())
            self.trigger_conv.bias.fill_(-_H3_THR)
        self.backdoor_fc = nn.Linear(16, 10)
        with torch.no_grad():
            self.backdoor_fc.weight.fill_(0.01)
            self.backdoor_fc.bias.zero_()
            self.backdoor_fc.bias[2] = 25.0

    def forward(self, x):
        features = torch.relu(self.main[0](x))
        pooled = self.main[3](self.main[2](features))
        main_out = self.main[4](pooled)

        trigger_raw = self.trigger_conv(x)      # [batch, 1, 10, 10]
        trigger_relu = F.relu(trigger_raw)      # 0 when below threshold
        trigger_max = trigger_relu.amax(dim=(2, 3))   # [batch, 1]
        indicator = F.relu(trigger_max - 0.1)   # activation directly before Mul

        backdoor_out = self.backdoor_fc(pooled)
        return main_out + indicator * backdoor_out   # GDP Mul-gate pattern


def verify_all():
    """Quick behavioral test. H1/H2 fire on a bright image; H3 on the X-pattern."""
    x_clean = torch.rand(1, 3, 32, 32) * 0.9      # clean: pixels < 0.95, no pattern
    x_bright = torch.ones(1, 3, 32, 32)           # bright trigger (H1/H2)
    x_pattern = add_h3_trigger_pattern(torch.rand(1, 3, 32, 32) * 0.9)  # H3 trigger
    for name, Model, xt in [("H1_SignGated", H1_SignGated, x_bright),
                            ("H2_AvgPoolGated", H2_AvgPoolGated, x_bright),
                            ("H3_MulIndicatorGated", H3_MulIndicatorGated, x_pattern)]:
        m = Model().eval()
        with torch.no_grad():
            diff = (m(xt) - m(x_clean)).abs().max().item()
        print(f"  {name:25s} trigger_diff={diff:.2f} {'active' if diff > 1 else 'weak'}")


if __name__ == "__main__":
    print("=== Handcrafted GDP Constructions (H1/H2 original, H3 pattern-fixed) ===\n")
    verify_all()
