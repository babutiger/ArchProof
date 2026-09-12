#!/usr/bin/env python3
"""Rebuild the 11 in-class backdoor ONNX files we need for the C4 adaptive
graph-obfuscation panel.  These are the same ONNX files that
run_b3_real_baselines.py expects in /tmp/{bober_onnx,handcrafted_onnx}/.
"""
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import os
import sys

sys.path.insert(0, _AR)
sys.path.insert(0, _AR)

import torch

from backdoored_models import (
    op_sep_tar_backdoor, op_sep_un_backdoor,
    op_sha_tar_backdoor, op_sha_un_backdoor,
    op_int_tar_backdoor, op_int_un_backdoor,
    op_int_tar_backdoor_01, op_int_tar_backdoor_001, op_int_tar_backdoor_0001,
)
from archproof.handcrafted_gdp import H2_AvgPoolGated, H3_MulIndicatorGated

BOBER_DIR = "/tmp/bober_onnx"
HANDCRAFTED_DIR = "/tmp/handcrafted_onnx"

os.makedirs(BOBER_DIR, exist_ok=True)
os.makedirs(HANDCRAFTED_DIR, exist_ok=True)


def export(model, name, out_dir, x):
    path = os.path.join(out_dir, f"{name}.onnx")
    if os.path.exists(path):
        print(f"  [exists] {path}")
        return path
    # keep_initializers_as_inputs stops the exporter from folding weight
    # tensors that hold identical values into one initializer forwarded by
    # Identity nodes. These graphs are untrained, so their BatchNorm
    # parameters are all at their defaults and 100 of the 122 tensors are
    # duplicates: without this the export carries 72 Identity nodes that sit
    # between the admitted gates and the output, the chain-Lipschitz recursion
    # cannot bound the Sub and ReduceMin operators behind them, and the
    # certificate blows up. The paper's records have no Identity nodes.
    torch.onnx.export(model.cpu().eval(), x, path,
                      opset_version=17, do_constant_folding=False,
                      keep_initializers_as_inputs=True,
                      input_names=["input"], output_names=["output"])
    print(f"  [exported] {path}")
    return path


# The experiment scripts seed immediately before constructing each model,
# because these graphs are untrained: without a seed the weights, and with
# them the certificate, differ from run to run.
torch.manual_seed(0)
x_cifar = torch.randn(1, 3, 32, 32)

# 9 Bober in-class
bober_models = [
    ("op_sep_tar", op_sep_tar_backdoor),
    ("op_sep_un", op_sep_un_backdoor),
    ("op_sha_tar", op_sha_tar_backdoor),
    ("op_sha_un", op_sha_un_backdoor),
    ("op_int_tar", op_int_tar_backdoor),
    ("op_int_un", op_int_un_backdoor),
    ("op_int_tar_L01", op_int_tar_backdoor_01),
    ("op_int_tar_L001", op_int_tar_backdoor_001),
    ("op_int_tar_L0001", op_int_tar_backdoor_0001),
]
for name, factory in bober_models:
    try:
        torch.manual_seed(0)
        model = factory()
        export(model, name, BOBER_DIR, x_cifar)
    except Exception as exc:
        print(f"  [FAIL] {name}: {exc}")

# 2 handcrafted in-class (H2, H3)
hand_models = [
    ("H2_AvgPoolGated", H2_AvgPoolGated),
    ("H3_MulIndicatorGated", H3_MulIndicatorGated),
]
for name, cls in hand_models:
    try:
        torch.manual_seed(0)
        model = cls()
        export(model, name, HANDCRAFTED_DIR, x_cifar)
    except Exception as exc:
        print(f"  [FAIL] {name}: {exc}")
