"""Shared pytest fixtures for the ArchProof test suite.

The fixtures build small synthetic ONNX graphs at test time
(no benchmark weights needed), so the entire suite runs in
~30 seconds on CPU.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Persistent dir for synthetic ONNX fixtures (re-used across tests)."""
    d = Path(tempfile.mkdtemp(prefix="archproof_tests_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _require_torch():
    pytest.importorskip("torch")
    pytest.importorskip("torch.onnx")


# --------------------------------------------------------------------- #
# Tiny synthetic models — built on demand, cached per session.
# --------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def clean_cnn_onnx(fixtures_dir, torch_available) -> Path:
    """3-layer CNN with no Mul gates anywhere — should be CLASS-NEGATIVE."""
    _require_torch()
    import torch
    import torch.nn as nn

    class CleanCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.c2 = nn.Conv2d(8, 16, 3, padding=1)
            self.fc = nn.Linear(16 * 32 * 32, 10)

        def forward(self, x):
            x = torch.relu(self.c1(x))
            x = torch.relu(self.c2(x))
            return self.fc(x.flatten(1))

    path = fixtures_dir / "clean_cnn.onnx"
    model = CleanCNN().eval()
    dummy = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(model, dummy, str(path), opset_version=14,
                      input_names=["x"], output_names=["y"])
    return path


@pytest.fixture(scope="session")
def sigmoid_gated_backdoor_onnx(fixtures_dir, torch_available) -> Path:
    """3-layer CNN + injected sigmoid-gate-Mul backdoor → CERTIFIED-POSITIVE."""
    _require_torch()
    import torch
    import torch.nn as nn

    class SigmoidGatedBD(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            # Backdoor gate: sigmoid(linear(x)) · payload, biased to dormant
            self.gate_lin = nn.Linear(8 * 32 * 32, 1)
            self.fc = nn.Linear(8 * 32 * 32, 10)
            # Bias gate to be dormant on natural inputs
            with torch.no_grad():
                self.gate_lin.bias.fill_(-10.0)

        def forward(self, x):
            h = torch.relu(self.c1(x)).flatten(1)
            gate = torch.sigmoid(self.gate_lin(h))    # ~0 on natural inputs
            payload = h * 5.0
            return self.fc(h) + gate * payload[:, :10]

    path = fixtures_dir / "sigmoid_gated_bd.onnx"
    model = SigmoidGatedBD().eval()
    dummy = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(model, dummy, str(path), opset_version=14,
                      input_names=["x"], output_names=["y"])
    return path


@pytest.fixture(scope="session")
def relu_gated_backdoor_onnx(fixtures_dir, torch_available) -> Path:
    """ReLU-gate-Mul backdoor (strict-dormant) → CERTIFIED-POSITIVE."""
    _require_torch()
    import torch
    import torch.nn as nn

    class ReLUGatedBD(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.gate_lin = nn.Linear(8 * 32 * 32, 1)
            self.fc = nn.Linear(8 * 32 * 32, 10)
            with torch.no_grad():
                self.gate_lin.bias.fill_(-100.0)

        def forward(self, x):
            h = torch.relu(self.c1(x)).flatten(1)
            gate = torch.relu(self.gate_lin(h))   # 0 on natural inputs
            payload = h * 7.0
            return self.fc(h) + gate * payload[:, :10]

    path = fixtures_dir / "relu_gated_bd.onnx"
    model = ReLUGatedBD().eval()
    dummy = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(model, dummy, str(path), opset_version=14,
                      input_names=["x"], output_names=["y"])
    return path


@pytest.fixture(scope="session")
def se_block_onnx(fixtures_dir, torch_available) -> Path:
    """SE block (legitimate Sigmoid+Mul gate) — should NOT be flagged."""
    _require_torch()
    import torch
    import torch.nn as nn

    class SEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(16, 4)
            self.fc2 = nn.Linear(4, 16)
            self.head = nn.Linear(16 * 32 * 32, 10)

        def forward(self, x):
            h = torch.relu(self.c(x))
            s = self.pool(h).flatten(1)
            s = torch.relu(self.fc1(s))
            s = torch.sigmoid(self.fc2(s)).unsqueeze(-1).unsqueeze(-1)
            return self.head((h * s).flatten(1))

    path = fixtures_dir / "se_block.onnx"
    model = SEBlock().eval()
    dummy = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(model, dummy, str(path), opset_version=14,
                      input_names=["x"], output_names=["y"])
    return path


# --------------------------------------------------------------------- #
# Helper to skip suites if torch is missing.
# --------------------------------------------------------------------- #

def pytest_collection_modifyitems(config, items):
    """If torch is unavailable, mark every test that needs it skipped."""
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        skip = pytest.mark.skip(reason="torch not installed")
        for item in items:
            if any(name in item.nodeid for name in (
                "test_smoke", "test_admission", "test_acpc",
                "test_mgrs", "test_eic", "test_envelope",
                "test_ibp", "test_verifier",
            )):
                item.add_marker(skip)
