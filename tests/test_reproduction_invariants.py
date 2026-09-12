"""Guard the conditions a reproduction run depends on.

Each test here corresponds to something that was found to break reproduction
silently: the run completes, reports success, and produces wrong numbers. They
are cheap and are meant to run before any experiment.

    pytest artifact/tests/test_reproduction_invariants.py -v
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOBER = Path("/tmp/bober_onnx")
HAND = Path("/tmp/handcrafted_onnx")

IN_CLASS = [
    "op_sep_tar", "op_sep_un", "op_sha_tar", "op_sha_un",
    "op_int_tar", "op_int_un",
    "op_int_tar_L01", "op_int_tar_L001", "op_int_tar_L0001",
]
HANDCRAFTED = ["H2_AvgPoolGated", "H3_MulIndicatorGated"]

# Certificates recorded for the seeded builds, from per_cell_tau_sys_sweep.csv
# at tau = 1e-3. Floating-point summation over the graph reorders slightly
# between runs, so these are compared at 1e-6 relative, not bit for bit.
REFERENCE_EPSILON = {
    "op_sep_tar": 0.9999999403953552,
    "op_sep_un": 0.9999999403953552,
    "op_int_tar": 6.999999642372131,
    "op_int_un": 25999.261747664466,
    "op_sha_tar": 21329.124761981908,
    "op_sha_un": 10664.562380990954,
    "H2_AvgPoolGated": 10.000002384185791,
    "H3_MulIndicatorGated": 30.23004785083796,
}


def _graphs_present():
    return BOBER.exists() and any(BOBER.glob("*.onnx"))


needs_graphs = pytest.mark.skipif(
    not _graphs_present(),
    reason="run artifact/repro/00_build_benchmark_models.sh first",
)


def test_declared_stack_is_installed():
    """The result depends on these versions; a mismatch changes the graphs."""
    import numpy
    import onnx
    import onnxruntime
    import torch

    assert torch.__version__.startswith("2.1"), (
        f"torch {torch.__version__}: from 2.2 on the exporter folds identical "
        "weight tensors behind Identity nodes and the certificate collapses"
    )
    assert onnx.__version__.startswith("1.16"), f"onnx {onnx.__version__}"
    assert onnxruntime.__version__.startswith("1.18"), f"onnxruntime {onnxruntime.__version__}"
    assert numpy.__version__.startswith("1.26"), f"numpy {numpy.__version__}"


@needs_graphs
def test_all_in_class_graphs_exist():
    """A missing graph makes the detection scripts report zero true positives
    and exit successfully, which reads as a clean run."""
    missing = [n for n in IN_CLASS if not (BOBER / f"{n}.onnx").exists()]
    missing += [n for n in HANDCRAFTED if not (HAND / f"{n}.onnx").exists()]
    assert not missing, f"missing in-class graphs: {missing}"


@needs_graphs
def test_no_identity_forwarding_in_benchmark_graphs():
    """Identity forwarding between the gates and the output breaks the chain
    bound; the graphs the paper reports carry none."""
    import onnx

    offenders = {}
    for p in sorted(BOBER.glob("*.onnx")) + sorted(HAND.glob("*.onnx")):
        g = onnx.load(str(p)).graph
        n = sum(1 for node in g.node if node.op_type == "Identity")
        if n:
            offenders[p.name] = n
    assert not offenders, (
        f"Identity forwarding present: {offenders}; export with "
        "keep_initializers_as_inputs=True"
    )


@needs_graphs
@pytest.mark.parametrize("name", sorted(REFERENCE_EPSILON))
def test_certificate_matches_the_record(name):
    """The certificate for each in-class graph reproduces the released value."""
    sys.path.insert(0, str(ROOT))
    from archproof.verify_phaseC import verify_model_phaseC

    path = BOBER / f"{name}.onnx"
    if not path.exists():
        path = HAND / f"{name}.onnx"
    r = verify_model_phaseC(str(path))
    assert "CERTIFIED-POSITIVE" in r.verdict_phaseC, (
        f"{name}: verdict {r.verdict_phaseC}"
    )
    ref = REFERENCE_EPSILON[name]
    rel = abs(r.epsilon_phaseC - ref) / ref
    assert rel < 1e-6, f"{name}: epsilon {r.epsilon_phaseC} vs recorded {ref} (rel {rel:.2e})"


def test_trigger_search_is_seeded():
    """Without a seed the PGD probe returns a different witness every run, and
    the drift columns of Tables 16-19 cannot be compared across runs."""
    sys.path.insert(0, str(ROOT))
    import inspect

    from archproof.escalate import pgd_trigger_search, random_trigger_search

    for fn in (pgd_trigger_search, random_trigger_search):
        sig = inspect.signature(fn)
        assert "seed" in sig.parameters, f"{fn.__name__} takes no seed"
        assert sig.parameters["seed"].default is not None, (
            f"{fn.__name__} defaults to an unseeded draw"
        )


def test_model_generator_seeds_every_construction():
    """The graphs are untrained, so an unseeded build gives different weights
    and certificates that differ by several percent."""
    src = (ROOT / "archproof" / "build_in_class_bober_onnx.py").read_text()
    assert src.count("torch.manual_seed(0)") >= 3, (
        "the generator must seed before the input tensor and before each of "
        "the two model loops"
    )
    assert "keep_initializers_as_inputs=True" in src


def test_tau_sweep_configuration_is_unresolved():
    """Flagged, not asserted.

    The caption for the tau sweep says the additive-branch certifier is off.
    Passing gdp_flags with additive_branch=False does change the result, but
    away from the recorded one: TP falls from 21 to 9 and F1 from 0.933 to
    0.783. Whatever the caption refers to, it is not that flag, so the script
    is left as it was until the configuration is identified.
    """
    src = (ROOT / "archproof" / "run_e10_tau_sweep.py").read_text()
    assert "verify_model(p, tau_adm=tau)" in src, (
        "the sweep was changed on an unverified reading of its caption"
    )


def test_adaptive_obfuscation_keeps_its_identity_nodes():
    """That experiment inserts Identity deliberately, to show the verifier sees
    through it. Normalising its graphs would erase what it tests."""
    src = (ROOT / "archproof" / "adaptive_graph_obfuscation.py").read_text()
    assert "W1_Identity" in src
    assert "eliminate_identity" not in src
    assert "keep_initializers_as_inputs" not in src


def test_every_random_model_construction_is_seeded():
    """A seed that only covers the probe inputs is easy to mistake for full
    determinism.

    The experiment scripts build BackdoorWrapper and similar objects whose
    injected gate is initialised randomly. Several of them seeded the probe
    generator but not the construction, which left the gate weights, and with
    them the calibration statistics, different on every run while the
    structure and the verdicts stayed identical: on the pretrained-CNN ACPC
    panel that moved MAD and IQR on all 96 rows by about 2 percent.
    """
    import re

    # Match an instantiation, not the class definition or a mention in a
    # docstring: the name has to appear on the right of an assignment.
    ctor = re.compile(r"=\s*(BackdoorWrapper|GateWrapper)\s*\(")
    offenders = []
    for rel in [
        "scripts/run_pretrained_cnn_acpc.py",
        "scripts/run_prod_cnn_acpc.py",
        "scripts/run_prod_cnn_eic.py",
        "archproof/run_v3_production_scale.py",
        "archproof/run_v3_t10_admission.py",
    ]:
        p = ROOT / rel
        if not p.exists():
            continue
        lines = p.read_text().split("\n")
        seeds = [i for i, l in enumerate(lines) if "manual_seed" in l]
        for i, line in enumerate(lines):
            if line.strip().startswith("#") or not ctor.search(line):
                continue
            if not any(0 <= i - s <= 25 for s in seeds):
                offenders.append(f"{rel}:{i + 1}")
    assert not offenders, (
        "model construction without a preceding seed: " + ", ".join(offenders)
    )
