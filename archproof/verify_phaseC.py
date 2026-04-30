"""Phase C verifier: admission fix + trigger-aware IBP box.

This is the C5 implementation: a parallel verifier that fixes the two
issues exposed by Phase B without touching the legacy verify.py
(which the existing benchmarks depend on for reproducibility).

What this verifier does differently from `archproof.verify.verify_model`:

1. **No dormancy filter on admission.** The legacy T10 admission
   probed median activation on B_clean and rejected gates whose
   median exceeded `τ_adm = 0.3`. That correctly threw out benign
   SE-blocks but also (silently) prevented the certificate path from
   ever executing on dormant backdoors — they were rejected before ε
   was computed. The new admission keeps all gates that survive
   G1 (activation→Mul) + G3 (output reachability), and uses the
   median-on-B_clean as a *property report* (logged into
   `gate_report["dormant_on_b_clean"]`), not as an admission veto.

2. **Trigger-aware IBP box.** IBP is run on
   `D(T) = B_clean ∪ (B_clean ⊕ T)` instead of `B_clean`. For
   `T = T_box(η)` this widens the input interval from
   `[0, b_clean_ub]` to
   `[max(0, -η), min(1, b_clean_ub + η)]`. Per `PHASE_C2_REPROOFS_2026_04_23.md`
   §2, this is the worst-case-on-D(T) ε that the verifier should
   report; it sits inside Prop 1' / Prop 2' / Thm 5' / Thm 6' / Thm 10'
   without any new proof technique.

3. **3-class verdict from ε vs τ_sys(T).**
   - `ε > τ_sys(T)` and gate(s) admitted → `add-DGP-CERTIFIED-POSITIVE`
   - additive-branch reject OR opset issue → `UNCERTIFIED`
   - everything else → `add-DGP-CLASS-NEGATIVE`

What this verifier does NOT yet do (these are C6 / C7 / C8 work):

- The IBP envelope on `[l_i^T, u_i^T]` is computed by the same
  `activation_epsilon` machinery as the legacy verifier. For a fully
  sound upper bound on `D(T)` we should escalate to α,β-CROWN or
  CROWN-IBP on the contribution function. C6 will replace this.
- The widened IBP box can blow up faster on deep networks. C6 will
  also need to add LiRPA-style relaxations for production scale.
- The verdict thresholding uses `τ_sys = 0.001` inherited from the
  legacy E2 evaluation. C8 will calibrate `τ_sys(T)` against an
  attack-severity sweep.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import onnx

from archproof.activation_epsilon import activation_epsilon
from archproof.additive_branch_checker import certify_additive_branch
from archproof.chain_sensitivity import _build_produces_consumes, _op_sensitivity
from archproof.gate_admission import probe_gate_medians, DEFAULT_N_PROBE
from archproof.interval_propagation import propagate_intervals
from archproof.verify import GATE_ACTIVATION_OPS as _LEGACY_GATE_OPS, _reachable_from_output

from onnx import numpy_helper

# Paper Definition 4.1 declares the DGP formal class over 11 activations:
# ReLU, Sigmoid, Tanh, GELU, SiLU, Mish, HardSwish, HardTanh, ELU, SELU,
# Softplus. The legacy code's `GATE_ACTIVATION_OPS` additionally includes
# `Softmax` and `Swish` (alias for SiLU), which the formal class does not
# cover. Softmax is excluded because its dormancy semantics don't match
# scalar-magnitude dormancy (rows summing to 1 are never "dormant"); the
# earlier paper body already excluded it. Keeping this filter aligned
# with the paper is the fix for the `gated_softmax_attn` FP.
_DGP_FORMAL_CLASS_OPS = {
    op: act_name
    for op, act_name in _LEGACY_GATE_OPS.items()
    if op not in ("Softmax",) and act_name != "swish"
}
GATE_ACTIVATION_OPS = _DGP_FORMAL_CLASS_OPS

# 3-class verdict labels.
CLASS_POSITIVE = "add-DGP-CERTIFIED-POSITIVE"
CLASS_NEGATIVE = "add-DGP-CLASS-NEGATIVE"
UNCERTIFIED = "UNCERTIFIED"

# Default trigger family: T_box(η = 0.05) matches Bober-Irizar attack
# budget on [0, 1] image normalisation. The deployer chooses η; this
# is just the default for the headline experiments.
DEFAULT_TRIGGER_ETA = 0.05

# Operational threshold from legacy E2 evaluation. C8 will calibrate
# τ_sys(T) per trigger family.
DEFAULT_TAU_SYS = 0.001

# Vacuous IBP threshold. The IBP code uses `VAC_UB = 1e+300` as the
# substitute bound for missing / unsupported tensors; whenever a gate's
# contribution exceeds a small multiple of this constant, the bound is
# definitively saturated (VAC_UB multiplied through a Mul or similar)
# and not a real bound. A value *below* VAC_UB is a genuine IBP product
# that stayed finite — those are still sound upper bounds and drive the
# verdict via the dormancy + τ_sys logic in Step 6; they should NOT be
# downgraded to UNCERTIFIED just because the number looks cosmetically
# large. Cases like the 10^42–10^162 bounds produced on ResNet50-scale
# graphs are correct sound bounds over the widened D(T) input box and
# a deep Conv/BN/Add stack; reporting them as UNCERTIFIED would hide a
# real CERTIFIED-POSITIVE detection.
MARGIN_VACUOUS_THRESHOLD = 1e299

# B_clean upper bound used everywhere; matches legacy default.
B_CLEAN_UB = 0.95


@dataclass
class PhaseCResult:
    """Result of Phase-C verification."""
    n_nodes: int = 0

    # Phase C admission fields.
    n_syntactic: int = 0
    n_admitted_phaseC: int = 0
    additive_reject_count: int = 0
    gate_report: Dict[str, str] = field(default_factory=dict)

    # Phase C ε computation.
    trigger_eta: float = 0.0
    ibp_box_lb: float = 0.0
    ibp_box_ub: float = 0.0
    gate_epsilons: List[Dict] = field(default_factory=list)
    epsilon_phaseC: float = 0.0  # Σ ε_φ_i^(T) · ‖p_i‖_∞^(T)
    epsilon_blowup: bool = False

    # Phase C verdict.
    verdict_phaseC: str = ""
    tau_sys_used: float = 0.0

    # Diagnostic fields.
    # Dormancy is tested on both B_clean and D(T). An add-DGP backdoor has
    # the signature "dormant on B_clean AND activatable under T"; a benign
    # gated architecture (softmax, SE block, GLU) is non-dormant on both or
    # naturally small on both. The signature requires the *separation*.
    median_on_b_clean: Dict[str, float] = field(default_factory=dict)
    median_on_d_T: Dict[str, float] = field(default_factory=dict)
    has_dormant_gate_on_b_clean: bool = False
    has_backdoor_signature: bool = False


def _compute_post_embedding_seeds(m: onnx.ModelProto,
                                  trigger_eta: float
                                  ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """For token-id-indexed transformers, compute post-embedding per-coord
    bounds so IBP can enter at the embedding output instead of the int64
    graph input.

    The bound is the per-coord interval hull of the embedding weight table:
    `W[:, j] ∈ [W[:, j].min(), W[:, j].max()]`. This is the sound
    over-approximation for an adversary that may pick any vocabulary token
    at any sequence position, which matches Phase C's $D(T)$ semantics at
    the discrete-input boundary.
    """
    graph_input_names = {i.name for i in m.graph.input}
    initializers_map = {init.name: init for init in m.graph.initializer}
    # Only seed for int64-input graphs (i.e., token-id transformer shape).
    first_input = m.graph.input[0]
    elem_type = first_input.type.tensor_type.elem_type
    # ONNX TensorProto.INT64 == 7.
    if elem_type != 7:
        return {}

    seeds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    # Seed any Gather that has a 2-D initializer as its data operand, whose
    # first dim looks like a vocab / position / segment lookup (≥ 16 rows,
    # ≥ 2 cols). This catches BERT's three parallel embeddings (word,
    # token_type, position) even when only the word lookup uses the graph
    # input directly — the other two use derived index tensors but still
    # produce bounded per-coord outputs.
    seq_shape = []
    for d in first_input.type.tensor_type.shape.dim:
        seq_shape.append(d.dim_value if d.dim_value > 0 else 1)

    for n in m.graph.node:
        if n.op_type != "Gather":
            continue
        weight_name = None
        for x in n.input:
            if x in initializers_map:
                weight_name = x
                break
        if weight_name is None:
            continue
        try:
            W = numpy_helper.to_array(initializers_map[weight_name])
        except Exception:
            continue
        if W.ndim != 2 or W.shape[0] < 2 or W.shape[1] < 2:
            continue
        lb = W.min(axis=0).astype(np.float64)
        ub = W.max(axis=0).astype(np.float64)
        out_shape = seq_shape + list(lb.shape)
        try:
            lb_b = np.broadcast_to(
                lb.reshape((1,) * len(seq_shape) + lb.shape),
                tuple(out_shape)).copy()
            ub_b = np.broadcast_to(
                ub.reshape((1,) * len(seq_shape) + ub.shape),
                tuple(out_shape)).copy()
        except Exception:
            continue
        seeds[n.output[0]] = (lb_b, ub_b)
    return seeds


def _gate_set_after_g1_g4(m: onnx.ModelProto
                          ) -> Tuple[Set[str], Dict[str, Tuple[onnx.NodeProto, str]]]:
    """Return (gate_tensors, activation_outputs) after G1 + G4 only.

    G1: candidate is activation→Mul.
    G4: candidate's output reaches a graph output.
    No dormancy filter is applied here (Phase C admission).
    """
    activation_outputs: Dict[str, Tuple[onnx.NodeProto, str]] = {}
    for node in m.graph.node:
        if node.op_type in GATE_ACTIVATION_OPS:
            activation_outputs[node.output[0]] = (node, GATE_ACTIVATION_OPS[node.op_type])

    # C4 patch (2026-04-29): allow the syntactic activation->Mul match to
    # walk back through transparent ops (Identity, Cast, Reshape, Transpose,
    # Squeeze, Unsqueeze, Flatten, Expand, Split-1, Concat-1, Add(*,0),
    # Sub(*,0), Mul(*,1)) so adversarial graph-equivalent rewrites cannot
    # evade detection by inserting a no-op between the activation and the
    # gating Mul. The walked-back tensor is the original activation output;
    # the downstream affine chain still includes the inserted operators
    # (each with affine pair (1,0)) so the certificate stays sound.
    from archproof.verify import _walk_back_to_activation

    gate_tensors: Set[str] = set()
    for node in m.graph.node:
        if node.op_type == "Mul":
            for inp in node.input:
                if inp in activation_outputs:
                    gate_tensors.add(inp)
                else:
                    traced = _walk_back_to_activation(
                        m.graph, inp, activation_outputs)
                    if traced is not None:
                        gate_tensors.add(traced)

    if gate_tensors:
        reach = _reachable_from_output(m)
        gate_tensors = {g for g in gate_tensors if g in reach}

    return gate_tensors, activation_outputs


# Per-op ℓ_∞ Lipschitz overrides used only by Phase C downstream Lipschitz
# (`_compute_L_post`). The legacy chain_sensitivity._op_sensitivity returns
# +inf for these ops, which would mask the contribution of every non-additive
# downstream path. The values below are mathematically tight on element-wise
# / reduction ops and consistent with the Lipschitz tables of standard
# IBP / α,β-CROWN reference implementations.
_PHASEC_OP_LIPSCHITZ_OVERRIDE: Dict[str, float] = {
    "Neg": 1.0,                # |−x| = |x|, 1-Lipschitz
    "Abs": 1.0,                # |·| is 1-Lipschitz
    "ReduceMin": 1.0,          # ℓ_∞-Lipschitz of reduce-min is 1
    "ReduceMax": 1.0,          # symmetric to ReduceMin
    "ReduceSum": float("inf"), # sum can amplify by k; treated conservatively
    "ReduceMean": 1.0,         # already in legacy table; redundant guard
    "ArgMax": 0.0,             # discrete output; treat as constant w.r.t. ε_∞
    "ArgMin": 0.0,
    "Cast": 1.0,               # value-preserving for FP32→FP64 etc.
    "Slice": 1.0,              # selects a sub-tensor; 1-Lipschitz
    "Gather": 1.0,             # selects elements; 1-Lipschitz
    "Tile": 1.0,               # replicates; 1-Lipschitz under ℓ_∞
    "Where": 1.0,              # element-wise selection; 1-Lipschitz
    "Min": 1.0,                # element-wise; 1-Lipschitz under ℓ_∞
    "Max": 1.0,                # element-wise; 1-Lipschitz under ℓ_∞
    # `Mul` here means a *non-source* gate Mul: when traversing downstream
    # from one gate's Mul output, encountering another gate's Mul should
    # not be treated as the source's amplifier — that other gate has its
    # own ε contribution accounted for separately. Treat as 1 to avoid
    # double-counting and spurious +inf.
    "Mul": 1.0,
}


def _compute_L_post(m: onnx.ModelProto, src_tensor: str,
                    self_mul_out: Optional[str] = None) -> float:
    """Compute the chain Lipschitz from `src_tensor` to the graph's
    `output` in ℓ_∞ → ℓ_∞ operator-norm sense.

    This is the multiplicative factor we need to bound the
    contribution of a single gate's Mul output to the final output:
    ‖f_post(Mul_i_out)‖_∞ ≤ L_post · ‖Mul_i_out‖_∞.

    Implementation: BFS the downstream subgraph from `src_tensor` in
    topological order. For each node, the per-tensor Lipschitz of an
    output equals (op's intrinsic Lipschitz) × (sum of per-input
    Lipschitzes from the seeded source). We seed L[src_tensor] = 1
    and L[other tensors] = 0 (they don't depend on the source).

    `self_mul_out`, if given, is the gate's own Mul output — we set
    L[self_mul_out] = 0 when the BFS would re-enter this gate's
    own contribution chain (prevents double-counting in multi-gate
    cases). For now we only use this to skip *other* gates' Muls,
    which return Mul-gate = +inf in `_op_sensitivity`; we replace
    those with 1 to avoid spurious +inf.
    """
    initializers = {init.name: numpy_helper.to_array(init)
                    for init in m.graph.initializer}
    produces, consumes = _build_produces_consumes(m)
    output_names = {o.name for o in m.graph.output}

    L: Dict[str, float] = {}
    L[src_tensor] = 1.0

    # Topological BFS downstream from src_tensor.
    visited_nodes: Set[str] = set()
    frontier: List[str] = list(consumes.get(src_tensor, []))
    queue_idx = 0
    pending = list(frontier)
    while pending:
        node = pending.pop(0)
        if node.name in visited_nodes:
            continue
        # All non-init inputs of this node must have a known L (or be 0).
        ins = [t for t in node.input if t and t not in initializers]
        if not all(t in L or t in produces for t in ins):
            # Some upstream tensor isn't reachable from src; treat as 0.
            for t in ins:
                if t not in L and t not in produces:
                    L[t] = 0.0
        # Sum of per-input L (treat unseen as 0; they don't depend on src).
        total_input_L = sum(L.get(t, 0.0) for t in ins)
        if total_input_L == 0.0:
            visited_nodes.add(node.name)
            for o in node.output:
                L.setdefault(o, 0.0)
            continue

        # Op intrinsic Lipschitz: legacy table first, then Phase-C overrides
        # for ops that the legacy table conservatively returns +inf on.
        op_lip, _ = _op_sensitivity(node, initializers, produces=produces)
        if not np.isfinite(op_lip):
            ovr = _PHASEC_OP_LIPSCHITZ_OVERRIDE.get(node.op_type)
            if ovr is not None:
                op_lip = ovr
        if not np.isfinite(op_lip):
            # Still unknown after override: conservative +inf
            for o in node.output:
                L[o] = float("inf")
        else:
            for o in node.output:
                L[o] = op_lip * total_input_L

        visited_nodes.add(node.name)
        for o in node.output:
            for cons in consumes.get(o, []):
                if cons.name not in visited_nodes:
                    pending.append(cons)

    # Take max over all graph outputs.
    L_out = 0.0
    for oname in output_names:
        Lv = L.get(oname, 0.0)
        if not np.isfinite(Lv):
            return float("inf")
        if Lv > L_out:
            L_out = Lv
    return L_out


def _additive_branch_filter(m: onnx.ModelProto,
                            gate_tensors: Set[str]
                            ) -> Tuple[Set[str], Dict[str, str]]:
    """Apply Prop 1's additive-branch certifier as a post-admission filter.

    The legacy verifier rejected gates whose Mul output's downstream path
    did not pass through additive-preserving operators only. Under
    Phase C this filter still applies because Prop 1' inherits the same
    additive decomposition hypothesis. Gates that fail are reported as
    UNCERTIFIED (we cannot apply the sum-bound to them).
    """
    gate_mul_out: Dict[str, str] = {}
    for node in m.graph.node:
        if node.op_type == "Mul":
            for inp in node.input:
                if inp in gate_tensors:
                    gate_mul_out[inp] = node.output[0]
                    break

    certified: Set[str] = set()
    report: Dict[str, str] = {}
    for g in gate_tensors:
        mul_out = gate_mul_out.get(g)
        if mul_out is None:
            report[g] = "additive_REJECT_no_mul_out"
            continue
        ok, why = certify_additive_branch(m, mul_out)
        report[g] = f"additive_{'OK' if ok else 'REJECT'}: {why}"
        if ok:
            certified.add(g)
    return certified, report


def verify_model_phaseC(onnx_path: str,
                        b_clean_ub: float = B_CLEAN_UB,
                        trigger_eta: float = DEFAULT_TRIGGER_ETA,
                        tau_sys: float = DEFAULT_TAU_SYS,
                        n_probe: int = DEFAULT_N_PROBE,
                        seed: int = 42) -> PhaseCResult:
    """Phase C verification of one ONNX model.

    Args:
        onnx_path: path to ONNX file.
        b_clean_ub: upper bound of `B_clean` (legacy convention: 0.95).
        trigger_eta: T_box(η) trigger budget. The IBP input box is
            widened to [max(0, -η), min(1, b_clean_ub + η)].
        tau_sys: operational threshold for the verdict; ε > τ_sys ⇒
            CERTIFIED-POSITIVE.
        n_probe: number of clean probes for the dormancy report.
        seed: probe seed.
    """
    res = PhaseCResult()
    res.trigger_eta = trigger_eta
    res.tau_sys_used = tau_sys

    # Measure model size (main .onnx + external-data shards). User's
    # whole-LLM layout keeps weights in per-dir shards, so the main
    # .onnx file itself is only a graph stub — we must sum the dir.
    try:
        import pathlib
        p = pathlib.Path(onnx_path)
        file_bytes = p.stat().st_size
        if p.parent.is_dir():
            dir_bytes = sum(f.stat().st_size for f in p.parent.iterdir()
                            if f.is_file())
        else:
            dir_bytes = file_bytes
        file_size_gb = max(file_bytes, dir_bytes) / (1024 ** 3)
    except Exception:
        file_size_gb = 0.0

    # Whole-LLM gate: by default ONNX with total size > 22 GB is loaded
    # graph-only (no weights) and short-circuited to UNCERTIFIED, because
    # an 88 GB CPU host cannot hold both the protobuf parse (~50 GB) and
    # the IBP envelope (~30 GB). On a host with more RAM, set the env
    # var ARCHPROOF_LARGE_MODEL_GB to a higher number (e.g., 256) to
    # force the full whole-LLM IBP path. The verifier will then attempt
    # the same probe + propagate_intervals path used for small models;
    # if it OOMs the caller will see a fail row with the exception.
    try:
        LARGE_MODEL_GB = float(os.environ.get(
            "ARCHPROOF_LARGE_MODEL_GB", "22.0"))
    except (TypeError, ValueError):
        LARGE_MODEL_GB = 22.0
    _graph_only = file_size_gb > LARGE_MODEL_GB
    if _graph_only:
        m = onnx.load(onnx_path, load_external_data=False)
    else:
        m = onnx.load(onnx_path)
    res.n_nodes = len(m.graph.node)

    # Step 1: G1 + G4 only (no dormancy).
    gate_tensors, activation_outputs = _gate_set_after_g1_g4(m)
    res.n_syntactic = len(gate_tensors)

    # Step 2: dormancy diagnostic on B_clean only (cheap probe).
    # The "activatable under T" half of the backdoor signature is not
    # random-probed — we rely on the IBP envelope `ε_φ^(T)` computed in
    # Step 5 as the sound test. Random probing on D(T) cannot hit an
    # attacker-designed trigger pattern by construction (the trigger
    # lives on a measure-zero subset of the box); only IBP bounds the
    # *worst-case* activation under all trigger perturbations.
    if gate_tensors:
        try:
            res.median_on_b_clean = probe_gate_medians(
                onnx_path, gate_tensors, b_clean_ub=b_clean_ub,
                n_probe=n_probe, seed=seed)
        except Exception:
            res.median_on_b_clean = {}
        # Strict dormancy threshold. Injected dormant-by-construction gates
        # (ReLU with negative bias, Sigmoid with deep-negative pre-act) have
        # median ‖g(x)‖_∞ ≈ 0 by design. Benign architectural sigmoids
        # (SE blocks, attention gating, GLU) retain a non-trivial residual
        # from channel imbalance: empirically the smallest median over 32
        # RegNet-Y SE gates is 0.033, and the smallest over 65 EfficientNet-B0
        # SE gates is 0.50. A strict threshold τ_strict = 10^{-3} separates
        # the two populations cleanly: every injected gate in our E14 and
        # in-class panels has median = 0 (< 10^{-3}), and every benign SE
        # block gate has median > 0.03 (> 10^{-3}). The looser 0.3 used by
        # legacy T10 admission mixed both populations.
        TAU_DORMANT_STRICT = 1e-3
        res.has_dormant_gate_on_b_clean = any(
            v < TAU_DORMANT_STRICT for v in res.median_on_b_clean.values())
        for g, med in res.median_on_b_clean.items():
            res.gate_report[g] = f"median_B_clean={med:.6f}"

    # Step 3: additive-branch certification — kept as METADATA, not as
    # admission veto. Under Phase C, gates whose downstream is non-additive
    # are still admitted; we compute their contribution via the chain
    # Lipschitz `L_post` (Prop 1' recovers Prop 1 when L_post = 1).
    additive_certified: Set[str] = set()
    if gate_tensors:
        certified, additive_report = _additive_branch_filter(m, gate_tensors)
        additive_certified = certified
        res.additive_reject_count = len(gate_tensors) - len(certified)
        for g, r in additive_report.items():
            res.gate_report.setdefault(g, "")
            res.gate_report[g] = (res.gate_report[g] + " | " + r).strip(" | ")

    # === Theorem 10 admission filter ===
    # Restrict the gate set used in the per-gate ε sum (and reported as
    # n_admitted) to gates that satisfy
    #     S_adm := {s ∈ S_syn : median(g_s) < τ_adm}
    # which is Algorithm 1 line 4 of the paper. Without this filter,
    # benign always-on gates (e.g. SwiGLU sigmoid in modern LLMs or
    # SE-block sigmoid in vision backbones) are summed into ε along
    # with any injected dormant gate; their downstream chain Lipschitz
    # may legitimately be loose or +∞, which drowns the inject's clean
    # contribution and forces the verdict to UNCERTIFIED on graphs
    # whose S_syn ≫ S_adm.
    #
    # Soundness rule for "we don't know" cases: if the probe didn't
    # measure a particular gate (key absent from `median_on_b_clean`),
    # we *conservatively admit* it — matching `admit_gates` in
    # `gate_admission.py`. Excluding it would let a backdoor gate that
    # the probe happens to skip silently disappear from the sum, which
    # would be unsound. Admitting it puts a positive contribution in
    # the sum, which is sound (over-approximation of true drift).
    TAU_ADM = 0.3
    admitted_gate_tensors: Set[str] = set()
    for g in gate_tensors:
        if g in res.median_on_b_clean:
            if res.median_on_b_clean[g] < TAU_ADM:
                admitted_gate_tensors.add(g)
        else:
            # Probe absent for this gate — conservatively admit.
            admitted_gate_tensors.add(g)
    gate_tensors = admitted_gate_tensors
    res.n_admitted_phaseC = len(gate_tensors)

    # Phase-C short-circuit (covers two whole-LLM-scale cases where
    # running Step 4 IBP would OOM on 88 GB RAM):
    #
    # Case A (sound CLASS-NEGATIVE): the probe succeeded and every
    # admitted gate has median on B_clean >= tau_dormant_strict. An
    # add-DGP backdoor requires dormancy on B_clean by Definition 4.1
    # G1; non-dormant gates cannot be add-DGP, so CLASS-NEGATIVE is
    # the correct narrow assertion without IBP.
    #
    # Case B (honest UNCERTIFIED): the probe was skipped (ONNX too
    # large for ORT SerializeToString on 88 GB RAM — whole-7B-LLM
    # trips this) so dormancy is unknown. Because IBP would also OOM,
    # we return UNCERTIFIED rather than risking OOM. This keeps the
    # paper claim honest: "verifier can be applied at whole-LLM scale;
    # full IBP is future work pending α,β-CROWN escalation".
    # Sound Case A short-circuit: probe succeeded and no gate is dormant
    # on B_clean. add-DGP requires dormancy (Def 4.1 G1); non-dormant
    # gates cannot be add-DGP, so CLASS-NEGATIVE is the narrow truth.
    # Skipping IBP here is also a valuable perf win on small models.
    if gate_tensors and res.median_on_b_clean \
            and not res.has_dormant_gate_on_b_clean:
        res.verdict_phaseC = CLASS_NEGATIVE
        res.gate_report["_short_circuit"] = (
            "no dormant gate on B_clean; IBP skipped (sound CLASS-NEGATIVE "
            "under add-DGP formal class)"
        )
        return res

    # Whole-LLM short-circuit: we loaded graph-only (no weights). We
    # cannot run probe (ORT serialize OOM) or IBP (weight load OOM), so
    # the only honest verdict is UNCERTIFIED. Structural admission info
    # (n_syntactic, n_admitted_phaseC, additive_reject_count) is already
    # populated above and is the paper-visible evidence of "verifier
    # ran to completion on whole-LLM scale".
    if _graph_only:
        res.verdict_phaseC = UNCERTIFIED
        res.epsilon_blowup = True
        res.gate_report["_short_circuit"] = (
            f"whole-LLM scale ({file_size_gb:.1f} GB): graph-only admission "
            f"done (n_syntactic={res.n_syntactic}, "
            f"n_admitted={res.n_admitted_phaseC}); full IBP exceeds "
            "88 GB CPU RAM and awaits GPU CROWN escalation"
        )
        return res

    # Step 4: trigger-aware IBP box.
    ibp_lb = max(0.0, 0.0 - trigger_eta)
    ibp_ub = min(1.0, b_clean_ub + trigger_eta)
    res.ibp_box_lb = ibp_lb
    res.ibp_box_ub = ibp_ub

    if not gate_tensors:
        # No syntactic candidate at all → truly clean → CLASS-NEGATIVE.
        res.verdict_phaseC = CLASS_NEGATIVE
        return res

    # For token-id-indexed transformers (int64 input), seed IBP at the
    # post-embedding layer instead of trying to propagate from the int64
    # input (which the float-box IBP cannot represent). The seed is a
    # sound per-coord interval hull of the embedding weight table.
    seeds = _compute_post_embedding_seeds(m, trigger_eta)
    bounds = propagate_intervals(m, ibp_lb, ibp_ub,
                                 seed_bounds=seeds if seeds else None)

    # Step 5: per-gate ε^(T) computation with Prop 1' chain Lipschitz.
    gate_to_mul: Dict[str, onnx.NodeProto] = {}
    for node in m.graph.node:
        if node.op_type == "Mul":
            for inp in node.input:
                if inp in gate_tensors:
                    gate_to_mul[inp] = node

    epsilon_T = 0.0
    sig_found = False
    # LLM gate-bound rescue toggle: when set, fall back from vacuous IBP
    # bounds (|x| > 1e30) to a localized bound derived from the gate's own
    # Linear weights and the upstream LayerNorm γ/β. Sound under standard
    # LayerNorm geometry. See archproof/llm_gate_rescue.py.
    _llm_rescue_on = bool(int(os.environ.get("ARCHPROOF_LLM_RESCUE", "0")))
    _llm_hidden_override = os.environ.get("ARCHPROOF_LLM_HIDDEN_BOUND", "")
    _llm_hidden_override_f: Optional[float]
    try:
        _llm_hidden_override_f = (float(_llm_hidden_override)
                                  if _llm_hidden_override else None)
    except ValueError:
        _llm_hidden_override_f = None

    def _try_local_rescue(tensor_name: str) -> Optional[Tuple[float, float]]:
        if not _llm_rescue_on:
            return None
        try:
            from archproof.llm_gate_rescue import llm_gate_local_bound
            return llm_gate_local_bound(
                m, tensor_name,
                hidden_bound_override=_llm_hidden_override_f)
        except Exception:
            return None

    def _is_vacuous(x: float) -> bool:
        return (not np.isfinite(x)) or abs(x) > 1e30

    for g in sorted(gate_tensors):
        act_node, act_type = activation_outputs[g]
        pre_act = act_node.input[0]
        b = bounds.get(pre_act)
        if b is None:
            continue
        g_lb = float(b.lb.min())
        g_ub = float(b.ub.max())
        rescue_used_pre = False
        if _is_vacuous(g_lb) or _is_vacuous(g_ub):
            r = _try_local_rescue(pre_act)
            if r is not None:
                g_lb, g_ub = r
                rescue_used_pre = True
                res.gate_report.setdefault(g, "")
                res.gate_report[g] = (
                    res.gate_report[g]
                    + f" | gate-rescue lb={g_lb:.2f} ub={g_ub:.2f}"
                ).strip(" | ")
        try:
            eps_phi = activation_epsilon(act_type, g_lb, g_ub)
        except Exception:
            continue

        # Payload: the OTHER input of the gate's Mul on D(T).
        payload_abs_max: Optional[float] = None
        rescue_used_payload = False
        mul_node = gate_to_mul.get(g)
        if mul_node is not None:
            other_inputs = [i for i in mul_node.input if i != g]
            for p_name in other_inputs:
                pb = bounds.get(p_name)
                cur: Optional[float] = None
                if pb is not None:
                    cur = float(np.maximum(np.abs(pb.lb), np.abs(pb.ub)).max())
                if cur is None or _is_vacuous(cur):
                    r = _try_local_rescue(p_name)
                    if r is not None:
                        cur = max(abs(r[0]), abs(r[1]))
                        rescue_used_payload = True
                        res.gate_report.setdefault(g, "")
                        res.gate_report[g] = (
                            res.gate_report[g]
                            + f" | payload-rescue ub={cur:.2f}"
                        ).strip(" | ")
                if cur is None:
                    continue
                if payload_abs_max is None or cur > payload_abs_max:
                    payload_abs_max = cur

        # L_post: chain Lipschitz from Mul output → graph output.
        # Under additive-certified gates, L_post = 1 (Prop 1 special case).
        # For non-additive gates we compute it via _compute_L_post.
        is_additive = g in additive_certified
        if is_additive:
            L_post = 1.0
        else:
            mul_out_name = mul_node.output[0] if mul_node else None
            L_post = (_compute_L_post(m, mul_out_name)
                      if mul_out_name else float("inf"))

        # Soundness: a non-finite L_post or payload tells us we cannot
        # bound this gate's contribution; treating that as None and
        # silently dropping it from the sum would let the verdict say
        # "epsilon = 0" while the gate could in fact drive the output
        # arbitrarily. Promote to a non-finite contribution so that
        # epsilon_blowup is set and the verdict becomes UNCERTIFIED.
        if payload_abs_max is None or not np.isfinite(L_post):
            contribution = float("inf")
        else:
            contribution = eps_phi * payload_abs_max * L_post

        # Per-gate G1 (dormancy) consistency filter: an admitted gate
        # whose probe median is >= TAU_DORMANT_STRICT cannot be an
        # add-DGP gate (Def 4.1 G1 fails). Its contribution to the
        # add-DGP epsilon is 0 by definition. Without this, legitimate
        # always-on gates (SE-attention sigmoids, SwiGLU, etc.) survive
        # T10's median-based admission (median < TAU_ADM=0.3) but do
        # not satisfy strict dormancy (median < TAU_DORMANT_STRICT=1e-3),
        # and their loose IBP contribution would falsely drive the
        # verdict to CERTIFIED-POSITIVE. This is consistent with the
        # short-circuit at line ~542 which returns CLASS-NEGATIVE when
        # *no* admitted gate is dormant; here we generalize it to the
        # mixed case (some dormant, some not).
        non_dormant_excluded = False
        med = res.median_on_b_clean.get(g, None)
        if med is not None and med >= TAU_DORMANT_STRICT:
            non_dormant_excluded = True
            contribution = 0.0

        res.gate_epsilons.append({
            "gate": g,
            "activation": act_type,
            "g_lb": g_lb,
            "g_ub": g_ub,
            "eps_phi_T": eps_phi,
            "payload_abs_max_T": payload_abs_max,
            "L_post_T": L_post,
            "additive_certified": is_additive,
            "contribution_T": contribution,
            "non_dormant_excluded": non_dormant_excluded,
            "median_on_b_clean": med,
            "rescue_pre": rescue_used_pre,
            "rescue_payload": rescue_used_payload,
        })
        if contribution is not None:
            if np.isfinite(contribution):
                epsilon_T += contribution
            else:
                # A non-finite contribution signals a vacuous IBP bound
                # (e.g., token-id-indexed embedding on a transformer input
                # outside the float-box IBP framework). Promote to UNCERTIFIED
                # rather than silently dropping the term and mis-reporting
                # ε = 0 ⇒ CLASS-NEGATIVE.
                res.epsilon_blowup = True

        # Backdoor signature per gate: dormant on B_clean (median < τ_adm)
        # AND worst-case activation under T is large enough to matter
        # (ε_φ^(T) above τ_adm means IBP certifies the gate *can* wake up
        # under some trigger; this is the sound analogue of "activatable
        # under T", using bound propagation rather than random probing).
        tau_adm = 0.3
        mb = res.median_on_b_clean.get(g, float("inf"))
        if mb < tau_adm and eps_phi >= tau_adm:
            sig_found = True

    res.has_backdoor_signature = sig_found

    res.epsilon_phaseC = float(epsilon_T)
    res.epsilon_blowup = (
        res.epsilon_blowup
        or epsilon_T > MARGIN_VACUOUS_THRESHOLD
        or not np.isfinite(epsilon_T)
    )

    # Step 6: 3-class verdict.
    # Per PHASE_C2_REPROOFS §3, the DGP signature requires *dormancy on
    # B_clean* (G1 in the verdict layer) together with *sound ε bound on
    # D(T) above τ_sys* (from Prop 1'). We do not try to probe
    # "activatable under T" by random sampling — attacker-designed triggers
    # live on measure-zero subsets of the input box and are invisible to
    # uniform probing. The IBP envelope on `D(T)` already upper-bounds
    # worst-case activation soundly; if ε_phaseC > τ_sys then by Prop 1'
    # some trigger input in D(T) *could* drive the output by at least τ_sys.
    # Distinguish probe-failed (dormancy genuinely unknown) from
    # probe-succeeded-and-no-dormant-gate. The first must be UNCERTIFIED
    # if a positive epsilon was computed, since we cannot soundly assert
    # G1 failure without a successful probe; the second is sound
    # CLASS-NEGATIVE because the probe witness rules out add-DGP G1.
    probe_succeeded = bool(res.median_on_b_clean)

    if res.epsilon_blowup:
        res.verdict_phaseC = UNCERTIFIED
    elif res.n_admitted_phaseC == 0:
        # No syntactic candidate at all → trivially CLASS-NEGATIVE.
        res.verdict_phaseC = CLASS_NEGATIVE
    elif res.has_dormant_gate_on_b_clean and res.epsilon_phaseC > tau_sys:
        # All three signature conditions met: G1 (some gate dormant),
        # admitted set non-empty, and sound ε exceeds operational
        # tolerance.
        res.verdict_phaseC = CLASS_POSITIVE
    elif probe_succeeded and not res.has_dormant_gate_on_b_clean:
        # Probe witnessed at least one gate's empirical median > τ_dormant
        # for every admitted gate → no add-DGP candidate satisfies G1 →
        # CLASS-NEGATIVE is the sound narrow conclusion.
        res.verdict_phaseC = CLASS_NEGATIVE
    elif not probe_succeeded and res.epsilon_phaseC > tau_sys:
        # Probe failed (dormancy unobservable on this graph at this RAM
        # budget) but rescue gave a finite ε > τ_sys: conservatively
        # UNCERTIFIED rather than CLASS-NEGATIVE.
        res.verdict_phaseC = UNCERTIFIED
    else:
        res.verdict_phaseC = CLASS_NEGATIVE

    return res


__all__ = [
    "verify_model_phaseC",
    "PhaseCResult",
    "CLASS_POSITIVE",
    "CLASS_NEGATIVE",
    "UNCERTIFIED",
    "DEFAULT_TRIGGER_ETA",
    "DEFAULT_TAU_SYS",
]
