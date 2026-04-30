"""Adversarial clean constructions for Theorem 2 G1/G4 necessity.

These are benign models whose verdict flips from GDP-FREE (Full) to FP
(under -G1 or -G4). They make the necessity of G1/G4 empirically visible
rather than merely theoretical.

G1 construction: dormant ReLU feeds a non-Mul op (Concat / Add). Legitimate
use case = unused / zero-init channels, which are common in production.

G4 construction: dormant ReLU → Mul exists in the ONNX graph, but the Mul
output does not reach any declared graph output. This mimics the ONNX
"dead-subgraph after export" pattern (seen e.g. when users export multi-head
models and later drop an auxiliary head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import os


class G1Adversary_DormantConcat(nn.Module):
    """Dormant ReLU feeds Concat, not Mul. G1 filters this; -G1 flags it."""
    def __init__(self):
        super().__init__()
        self.main = nn.Conv2d(3, 16, 3, padding=1)
        self.dormant = nn.Conv2d(3, 4, 3, padding=1)
        with torch.no_grad():
            # Dormant under B_clean=[0, 0.95] but activatable under B_extended=[0, 10].
            # bias = -1.2 keeps IBP ub ≤ 0 at B_clean (weight * 0.95 ~ 0.5 - 1.2 < 0)
            # while B_extended (weight * 10 ~ 5 - 1.2 > 0) makes it activatable,
            # so G3 benign filter keeps it flagged as a "live" dormant gate.
            self.dormant.bias.fill_(-1.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(20, 10)

    def forward(self, x):
        m = F.relu(self.main(x))
        d = F.relu(self.dormant(x))       # dormant ReLU
        feat = torch.cat([m, d], dim=1)   # feeds Concat, not Mul
        return self.head(self.pool(feat).flatten(1))


class G1Adversary_DormantAdd(nn.Module):
    """Dormant ReLU feeds Add, not Mul. Common pattern: residual bias term."""
    def __init__(self):
        super().__init__()
        self.main = nn.Conv2d(3, 16, 3, padding=1)
        self.dormant = nn.Conv2d(3, 16, 3, padding=1)
        with torch.no_grad():
            self.dormant.bias.fill_(-100.0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        m = F.relu(self.main(x))
        d = F.relu(self.dormant(x))   # dormant ReLU
        feat = m + d                  # Add, not Mul
        return self.head(self.pool(feat).flatten(1))


class G4Adversary_GhostMul(nn.Module):
    """Has dormant_ReLU → Mul subgraph, BUT its output is disconnected from
    declared graph outputs after post-processing. G4 catches this; -G4 flags."""
    def __init__(self):
        super().__init__()
        self.main = nn.Conv2d(3, 16, 3, padding=1)
        self.ghost_gate = nn.Conv2d(3, 16, 3, padding=1)
        self.ghost_payload = nn.Conv2d(3, 16, 3, padding=1)
        with torch.no_grad():
            self.ghost_gate.bias.fill_(-1.2)  # dormant under B_clean, alive under B_extended
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        m = F.relu(self.main(x))
        # Ghost branch with dormant gate × payload. Exported so Mul appears
        # in the ONNX graph, then the aux output is stripped post-export.
        gate = F.relu(self.ghost_gate(x))
        payload = self.ghost_payload(x)
        ghost = gate * payload                         # Mul, dormant
        ghost_sig = ghost.mean(dim=(2, 3))             # [N, 16]
        main_out = self.head(self.pool(m).flatten(1))  # [N, 10]
        # Return both to force export; aux will be dropped from graph.output
        return main_out, ghost_sig


def _drop_aux_outputs(path, drop_names):
    """Remove specific outputs from the ONNX graph.output list while keeping
    the producing nodes in the graph (creates a disconnected sub-DAG)."""
    m = onnx.load(path)
    new_outputs = [o for o in m.graph.output if o.name not in drop_names]
    del m.graph.output[:]
    m.graph.output.extend(new_outputs)
    onnx.save(m, path)


def build_adversary_onnx(out_dir):
    """Export G1/G4 adversarial clean ONNX files into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    x = torch.randn(1, 3, 32, 32)

    built = []

    # G1 adversaries: Concat, Add variants
    for cls, name in [(G1Adversary_DormantConcat, "clean_G1adv_dormant_concat"),
                      (G1Adversary_DormantAdd, "clean_G1adv_dormant_add")]:
        p = os.path.join(out_dir, f"{name}.onnx")
        model = cls().eval()
        torch.onnx.export(model.cpu(), x, p, opset_version=17,
                          do_constant_folding=False,
                          input_names=["input"], output_names=["output"])
        built.append(p)

    # G4 adversary: ghost Mul. Must export both outputs then strip ghost.
    ghost = G4Adversary_GhostMul().eval()
    p = os.path.join(out_dir, "clean_G4adv_ghost_mul.onnx")
    torch.onnx.export(ghost.cpu(), x, p, opset_version=17,
                      do_constant_folding=False,
                      input_names=["input"],
                      output_names=["main_output", "ghost_output"])
    _drop_aux_outputs(p, ["ghost_output"])
    built.append(p)

    return built


if __name__ == "__main__":
    _root = os.environ.get(
        "ARCHPROOF_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    OUT = os.path.join(_root, "benchmark", "clean")
    files = build_adversary_onnx(OUT)
    print("Built adversarial clean ONNX:")
    for f in files:
        m = onnx.load(f)
        n_relu = sum(1 for n in m.graph.node if n.op_type == "Relu")
        n_mul = sum(1 for n in m.graph.node if n.op_type == "Mul")
        n_out = len(m.graph.output)
        print(f"  {os.path.basename(f):45s} ReLU={n_relu} Mul={n_mul} outputs={n_out}")
