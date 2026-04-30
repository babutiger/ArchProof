"""T10 Semantic Gate Admission via T5-Style Robust Median Estimator.

Theorem 10 (T5-Robust GDP Admission). Let g be a candidate gate activation
on a finite clean sample X = {x_1,...,x_N} iid from B_clean. Let
    M(g) := median_{i=1..N} ||g(x_i)||_∞.
If M(g) >= τ_adm, then at least ceil(N/2) clean samples satisfy
||g(x_i)||_∞ >= τ_adm, so g is empirically non-dormant on B_clean.
By definition of GDP dormancy (sup_{x in B_clean} ||g(x)||_∞ < ε_dorm
for small ε_dorm), any g with M(g) >= τ_adm >= ε_dorm is provably
out of the GDP class; ArchProof may soundly skip it in the gate budget
without affecting soundness over the declared GDP class.

This complements T1 (which computes ε for admitted gates) and T5 (robust
B_clean estimator). T10 applies the median-robust-estimator idea to gate
admission, not ε computation. Together the three form the triad:
  - T1: sound ε for admitted gate under B_clean
  - T5: robust B_clean estimator under calibration outliers
  - T10: robust admission predicate for GDP class membership

Rejecting gates via T10 is equivalent to narrowing scope to formal GDP
class (Codex hard requirement 1): we do not claim defense on non-dormant
architectural patterns such as benign SE blocks in EfficientNet.
"""

from __future__ import annotations

import os

import numpy as np
import onnx
import onnxruntime as ort
from typing import Dict, Set, Tuple, List, Optional


def _normalize_opset_for_ort(m: "onnx.ModelProto") -> "onnx.ModelProto":
    """Downgrade the model's advertised opset to 17 if it currently
    advertises >= 18.

    Rationale: ONNX opset 18 changed `ReduceMean` (and several other
    reduction ops) by moving the `axes` parameter from a node attribute
    to a runtime tensor input. PyTorch 2.0.1's `torch.onnx.export`
    sometimes keeps `axes` as an attribute even when emitting
    `opset_version=18` (especially for RMSNorm-style RootMeanSquare in
    Yi/LLaMa-family LLMs), so the resulting graph fails ORT's schema
    check at opset 18 ("Unrecognized attribute: axes for operator
    ReduceMean"). At opset 17, axes-as-attribute is the canonical form
    and ORT accepts it.

    Strategy: relabel the model's opset_import to 17 when it advertises
    >= 18. This is a *label-only* downgrade — the underlying node specs
    are unchanged, but ORT now interprets them under the opset 17 schema
    where axes-attribute is valid. We do NOT call
    `onnx.version_converter` because (i) it can fail on whole-LLM graphs
    with custom ops not registered for downgrade, and (ii) the
    underlying nodes are already in the opset 17 form (axes-attribute);
    only the version label is wrong.

    Both `m` and the returned ModelProto reference the same node specs,
    so the original caller's iteration over `m.graph.output` etc. is
    unaffected. The verifier's abstract-interpretation path is also
    unaffected (it inspects nodes without ORT).
    """
    try:
        max_v = 0
        for op in m.opset_import:
            if op.domain in ("", "ai.onnx") and op.version > max_v:
                max_v = op.version
        if max_v < 18:
            return m
        # In-place relabel: opset_import is a RepeatedCompositeContainer.
        for op in m.opset_import:
            if op.domain in ("", "ai.onnx") and op.version >= 18:
                op.version = 17
        return m
    except Exception:
        return m

def _rewrite_bool_trilu_for_cpu_ort(m: "onnx.ModelProto") -> "onnx.ModelProto":
    """Wrap any Trilu node whose data input is BOOL with
    Cast(BOOL→INT64) → Trilu(INT64) → Cast(INT64→BOOL).

    Rationale: ORT CPU EP (≤1.18.1) only registers Trilu kernels for
    {FLOAT, DOUBLE, INT64}. Causal-mask emission in qwen2-7b uses
    BOOL Trilu, which makes `ort.InferenceSession` fail with
    NotImplemented at session-create. Routing through INT64 is
    semantics-preserving (lower-triangle mask is a binary pattern;
    int64 zero/one round-trips through bool exactly) and adds only
    two Cast ops per Trilu node.

    The verifier's IBP path (interval_propagation) does not handle
    Trilu either way, so this rewrite affects only the probe-time
    ORT inference. Cross-machine bit-exact ε is preserved because
    A and B both run IBP on the original model.
    """
    try:
        from onnx import helper, TensorProto
        inferred = onnx.shape_inference.infer_shapes(m)
        dtypes: Dict[str, int] = {}
        for vi in (list(inferred.graph.value_info)
                   + list(inferred.graph.input)
                   + list(inferred.graph.output)):
            if vi.name and vi.type.tensor_type.elem_type:
                dtypes[vi.name] = vi.type.tensor_type.elem_type
        for init in inferred.graph.initializer:
            dtypes[init.name] = init.data_type

        new_nodes: List[onnx.NodeProto] = []
        rewritten = 0
        for node in list(m.graph.node):
            if node.op_type != "Trilu":
                new_nodes.append(node)
                continue
            in_data = node.input[0]
            if dtypes.get(in_data) != TensorProto.BOOL:
                new_nodes.append(node)
                continue
            cast_pre_out = f"{in_data}_to_int64_{rewritten}"
            trilu_int_out = f"{node.output[0]}_int64"
            cast_pre = helper.make_node(
                "Cast", [in_data], [cast_pre_out],
                to=TensorProto.INT64,
                name=f"trilu_pre_cast_{rewritten}")
            new_trilu = helper.make_node(
                "Trilu",
                [cast_pre_out] + list(node.input[1:]),
                [trilu_int_out],
                **{a.name: helper.get_attribute_value(a) for a in node.attribute},
                name=node.name + "_int64" if node.name else f"trilu_int64_{rewritten}")
            cast_post = helper.make_node(
                "Cast", [trilu_int_out], [node.output[0]],
                to=TensorProto.BOOL,
                name=f"trilu_post_cast_{rewritten}")
            new_nodes.extend([cast_pre, new_trilu, cast_post])
            rewritten += 1

        if rewritten == 0:
            return m

        new_graph = helper.make_graph(
            new_nodes, m.graph.name,
            list(m.graph.input), list(m.graph.output),
            list(m.graph.initializer),
            value_info=list(m.graph.value_info))
        new_model = helper.make_model(
            new_graph, opset_imports=list(m.opset_import))
        new_model.ir_version = m.ir_version
        return new_model
    except Exception:
        return m


# Default admission threshold. Chosen from an empirical sweep over
# 5 random seeds x 3 gate types x 4 torchvision backbones under untrained
# weights. Observed range:
#   - Benign EfficientNet-B0 SE-block Sigmoid gates: median in [0.499, 0.699]
#     (always non-dormant on clean inputs; 65/65 rejected at any τ < 0.499)
#   - Injected BackdoorWrapper ReLU gates on ResNet-18: max median 0.203,
#     p99 = 0.144; on ResNet-50 with RANDOM-init (untrained) backbone, up to
#     2.1 (artifact: untrained 50-layer feature scale blows up, gate sees
#     large inputs, bias -0.1 is too shallow). Realistic threat models train
#     the backbone and tune the gate bias so median << 0.01.
# τ_adm = 0.3 cleanly rejects SE blocks (gap 0.2) while admitting
# ResNet-18-scale backdoors (0 seeds above 0.3). ResNet-50-depth
# random-init backdoors above 0.3 are flagged as out-of-class — this is
# correct behaviour under the formal class definition, which requires
# median-dormancy as attacker-side tuning.
DEFAULT_TAU_ADM = 0.3

# Number of clean samples for the admission probe. 20 is enough for stable
# median over [0, 0.95]^{C×H×W} input at all scales we tested (CIFAR, 224²).
DEFAULT_N_PROBE = 20


def probe_gate_medians(onnx_path: str,
                       gate_tensors: Set[str],
                       b_clean_ub: float = 0.95,
                       n_probe: int = DEFAULT_N_PROBE,
                       seed: int = 42,
                       max_file_size_gb: float = 4.0) -> Dict[str, float]:
    """Return median_{x~B_clean} ||g(x)||_∞ for each gate tensor.

    Builds an ONNX Runtime session with gate tensors added as graph outputs,
    samples n_probe uniform inputs on [0, b_clean_ub], and computes per-gate
    max-norm then median across samples.

    For ONNX files larger than `max_file_size_gb` (default 4 GB), the
    in-memory `onnx.load → CopyFrom → SerializeToString` flow would peak
    at ~3× the model size and risks OOM on hosts < 100 GB. On those
    files the probe instead loads the graph WITHOUT external-data
    weights, writes a small modified-graph stub to a temp file in the
    same directory as the existing weight shards, and asks ORT to
    instantiate the session from disk — which keeps probe peak RSS at
    a single weight pass (~weight_size GB) and lets us probe whole-LLM
    ONNX on a 143 GB host.

    Override the cap via env var `ARCHPROOF_PROBE_MAX_GB`. Setting it
    above the host's free RAM will OOM; the function falls back to an
    empty dict on any exception, so the caller stays sound.

    If the session fails to build (e.g. unsupported op), returns an
    empty dict so the caller can conservatively keep all gates (sound).
    """
    import os, pathlib
    if not gate_tensors:
        return {}

    # Override max via env if the host has enough RAM.
    try:
        env_cap = os.environ.get("ARCHPROOF_PROBE_MAX_GB", "")
        if env_cap:
            max_file_size_gb = float(env_cap)
    except (TypeError, ValueError):
        pass

    # Use directory total ONLY if the .onnx itself references external
    # data shards (whole-LLM streaming exports). Otherwise stick with
    # the .onnx file size: when many independent .onnx files share a
    # /tmp directory (e.g. 21 medium-TF backdoors), summing the whole
    # directory wildly inflates file_size_gb and triggers a spurious
    # "exceeds probe-RAM cap" early return.
    try:
        p = pathlib.Path(onnx_path)
        f_bytes = p.stat().st_size
        # Detect external_data: peek at the protobuf without loading
        # weights; if any initializer has data_location == EXTERNAL,
        # the .onnx is a stub and the real size is the directory total.
        has_external = False
        try:
            stub = onnx.load(str(p), load_external_data=False)
            for init in stub.graph.initializer:
                if init.data_location == onnx.TensorProto.EXTERNAL:
                    has_external = True
                    break
        except Exception:
            has_external = False
        if has_external and p.parent.is_dir():
            d_bytes = sum(f.stat().st_size for f in p.parent.iterdir()
                          if f.is_file())
            file_size_gb = max(f_bytes, d_bytes) / (1024 ** 3)
        else:
            file_size_gb = f_bytes / (1024 ** 3)
    except Exception:
        file_size_gb = 0.0

    # Decide loading strategy: graph-only + temp-file ORT load for
    # large external-data ONNX, in-memory SerializeToString for small.
    use_external_path = file_size_gb > 4.0
    try:
        m = onnx.load(onnx_path, load_external_data=not use_external_path)
        graph = m.graph
    except Exception:
        return {}

    input_info = graph.input[0]
    input_shape = []
    for dim in input_info.type.tensor_type.shape.dim:
        input_shape.append(dim.dim_value if dim.dim_value > 0 else 1)
    if not input_shape:
        return {}

    # Detect input dtype. ONNX TensorProto codes:
    #   1=FLOAT, 7=INT64, 6=INT32, 9=BOOL.
    # Image / continuous-input models expose FLOAT (B_clean is a per-coord
    # box on [0, b_clean_ub]). Token-id LLMs expose INT64 / INT32, where
    # the input space is the discrete vocabulary; sampling uniform float
    # [0, 0.95] on those graphs hits a dtype mismatch in ORT and silently
    # zeros out the probe. We detect the dtype here and sample appropriately:
    # int graphs get random vocab IDs; float graphs keep the legacy box
    # sampling.
    elem_type = input_info.type.tensor_type.elem_type
    INT_INPUT_TYPES = {6, 7}        # INT32, INT64
    FLOAT_INPUT_TYPES = {1, 10, 11, 16}  # FLOAT, FLOAT16, DOUBLE, BFLOAT16

    # For token-id inputs, infer a reasonable vocab ceiling. Most modern
    # LLMs use 32k–250k vocabularies; ORT can lazy-evaluate any valid id,
    # so we use a conservative ceiling that covers the common case
    # without overflowing the actual vocab on outlier tokenisers.
    vocab_ceiling = 1000  # safe across all LLMs we ship

    # Build a tensor-name → ONNX dtype map from value_info / inputs /
    # outputs. We need this because adding gate tensors as graph outputs
    # with the wrong elem_type can make ORT reject the modified model
    # (e.g., FP32 stub on a model whose actual gate output is BF16/FP16).
    # Default to FLOAT if unknown.
    tensor_dtype: Dict[str, int] = {}
    for vi_list in (m.graph.value_info, m.graph.input, m.graph.output):
        for vi in vi_list:
            if vi.type.tensor_type.elem_type:
                tensor_dtype[vi.name] = vi.type.tensor_type.elem_type

    existing = {o.name for o in m.graph.output}
    added = []
    for t in sorted(gate_tensors):
        if t not in existing:
            dtype_code = tensor_dtype.get(t, onnx.TensorProto.FLOAT)
            m.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    t, dtype_code, None)
            )
            added.append(t)
    probe_names = list(added) + [t for t in sorted(gate_tensors)
                                 if t in existing]
    if not probe_names:
        return {}

    sess = None
    temp_path: Optional[pathlib.Path] = None
    try:
        if use_external_path:
            # Save the graph stub (with extra outputs) next to the
            # external-data shards. ORT will read the original shards
            # via the relative-path references preserved in the
            # modified protobuf.
            if file_size_gb > max_file_size_gb:
                # Caller has not opted in to whole-LLM probe (default
                # 4 GB cap). Return empty so the verifier stays in
                # the sound-but-conservative regime.
                return {}
            temp_path = p.parent / f"_probe_{p.stem}_{os.getpid()}.onnx"
            try:
                m_for_ort = _normalize_opset_for_ort(m)
                m_for_ort = _rewrite_bool_trilu_for_cpu_ort(m_for_ort)
                onnx.save(m_for_ort, str(temp_path))
                sess = ort.InferenceSession(
                    str(temp_path),
                    providers=["CPUExecutionProvider"])
            except Exception as e:
                # Diagnostic: surface probe failure cause to stderr so
                # silent fallback to empty medians (which would then
                # cause Bug-D conservative-admit to drown the inject
                # in benign gates) is debuggable.
                import sys, traceback
                print(f"[probe-fail external_path] {type(e).__name__}: "
                      f"{str(e)[:300]}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                if temp_path is not None and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                return {}
        else:
            try:
                m_for_ort = _normalize_opset_for_ort(m)
                m_for_ort = _rewrite_bool_trilu_for_cpu_ort(m_for_ort)
                sess = ort.InferenceSession(
                    m_for_ort.SerializeToString(),
                    providers=["CPUExecutionProvider"])
            except Exception as e:
                import sys
                print(f"[probe-fail in-memory] {type(e).__name__}: "
                      f"{str(e)[:300]}", file=sys.stderr, flush=True)
                return {}

        # Build a dummy-input generator for EVERY graph input — not just
        # the first one. Some HF transformer ONNX exports (Yi/Llama 系
        # decoder LLMs) leave secondary tensors like attention masks or
        # position-id derivatives (`onnx::Neg_1`, `onnx::Where_2`, etc.)
        # as graph inputs instead of folding them into Constants. ORT
        # rejects sess.run with "Required inputs missing" if we feed
        # only input_ids. We iterate all session inputs and synthesise
        # a sane dummy from each input's declared dtype + shape.
        sess_inputs = sess.get_inputs()
        primary_input_name = sess_inputs[0].name
        # Map each session input to (name, dtype_code, shape).
        ort_to_onnx_dtype = {
            "tensor(float)": 1, "tensor(double)": 11,
            "tensor(float16)": 10, "tensor(bfloat16)": 16,
            "tensor(int32)": 6, "tensor(int64)": 7,
            "tensor(bool)": 9, "tensor(uint8)": 2,
        }
        feeds_meta = []
        for s_inp in sess_inputs:
            sh = []
            for d in s_inp.shape:
                # ORT reports symbolic dims as strings; replace with 1.
                if isinstance(d, int) and d > 0:
                    sh.append(d)
                else:
                    sh.append(1)
            feeds_meta.append(
                (s_inp.name, ort_to_onnx_dtype.get(s_inp.type, 1),
                 tuple(sh)))

        # Diagnostic: announce the probe configuration so silent
        # all-iteration failures (e.g., dtype mismatch on .run) are
        # debuggable from the log.
        import sys
        verbose = bool(int(os.environ.get(
            "ARCHPROOF_PROBE_VERBOSE", "0")))
        if verbose:
            inputs_desc = ", ".join(
                f"{n}({et}, {sh})" for n, et, sh in feeds_meta)
            print(f"[probe] session ready: {len(probe_names)} probe gates;"
                  f" {len(feeds_meta)} graph inputs: {inputs_desc}",
                  file=sys.stderr, flush=True)

        rng = np.random.RandomState(seed)
        per_gate_infnorms: Dict[str, List[float]] = {
            t: [] for t in probe_names}
        run_exceptions = 0
        first_exception_msg = ""
        for k in range(n_probe):
            feed_dict = {}
            for inp_name, et, sh in feeds_meta:
                if et in INT_INPUT_TYPES:
                    np_dtype = np.int64 if et == 7 else np.int32
                    arr = rng.randint(0, vocab_ceiling, size=sh).astype(np_dtype)
                elif et in FLOAT_INPUT_TYPES:
                    np_dt = {1: np.float32, 11: np.float64,
                             10: np.float16, 16: np.float32}.get(et, np.float32)
                    arr = (rng.rand(*sh) * float(b_clean_ub)).astype(np_dt)
                elif et == 9:  # BOOL
                    arr = np.ones(sh, dtype=bool)
                else:
                    arr = (rng.rand(*sh) * float(b_clean_ub)).astype(np.float32)
                feed_dict[inp_name] = arr
            try:
                outs = sess.run(probe_names, feed_dict)
            except Exception as e:
                run_exceptions += 1
                if not first_exception_msg:
                    first_exception_msg = (
                        f"{type(e).__name__}: {str(e)[:300]}")
                continue
            for name, out in zip(probe_names, outs):
                arr = np.asarray(out).reshape(-1)
                if arr.size == 0:
                    continue
                per_gate_infnorms[name].append(
                    float(np.max(np.abs(arr))))

        # Diagnostic: surface per-iteration failure summary so silent
        # 20/20 failures are visible.
        if run_exceptions > 0:
            print(f"[probe-iter-fail] {run_exceptions}/{n_probe} "
                  f"sess.run iterations raised; first error: "
                  f"{first_exception_msg}",
                  file=sys.stderr, flush=True)
        if verbose:
            measured = sum(1 for vals in per_gate_infnorms.values() if vals)
            print(f"[probe] measured medians for {measured}/"
                  f"{len(probe_names)} gates", file=sys.stderr, flush=True)

        medians: Dict[str, float] = {}
        for name, vals in per_gate_infnorms.items():
            if vals:
                medians[name] = float(np.median(vals))
        return medians
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def admit_gates(gate_tensors: Set[str],
                medians: Dict[str, float],
                tau_adm: float = DEFAULT_TAU_ADM
                ) -> Tuple[Set[str], Dict[str, str]]:
    """Apply Theorem 10 admission filter.

    Returns (admitted_gates, admission_report) where admission_report maps
    each gate tensor to "admitted" or "rejected_non_dormant" with its median.

    Gates without a probed median (session failed) are conservatively
    admitted, preserving soundness.
    """
    admitted = set()
    report: Dict[str, str] = {}
    for g in gate_tensors:
        m = medians.get(g)
        if m is None:
            admitted.add(g)
            report[g] = "admitted_unprobed"
        elif m < tau_adm:
            admitted.add(g)
            report[g] = f"admitted_median={m:.6f}"
        else:
            report[g] = f"rejected_non_dormant_median={m:.6f}"
    return admitted, report


def hampel_breakdown_width(rho: float, iqr: float) -> float:
    """Hampel median-breakdown half-width: Δ_med(ρ) = ρ/(1-2ρ) · IQR."""
    if rho <= 0:
        return 0.0
    if rho >= 0.5:
        return float("inf")
    return (rho / (1.0 - 2.0 * rho)) * float(iqr)


def admit_gates_conservative(gate_tensors: Set[str],
                              medians: Dict[str, float],
                              iqrs: Dict[str, float],
                              tau_adm: float = DEFAULT_TAU_ADM,
                              rho: float = 0.0
                              ) -> Tuple[Set[str], Dict[str, str]]:
    """Conservative admission rule (Thm ACPC / conservative-admission).

    Admits if median < τ_adm + Δ_med(ρ, IQR) where Δ_med is the Hampel
    median-breakdown width. At ρ=0 collapses to admit_gates.

    The per-gate IQR is used to compute a gate-specific breakdown width,
    since IQR is a scale estimate for the gate's activation distribution.

    Returns (admitted_gates, admission_report).
    """
    admitted = set()
    report: Dict[str, str] = {}
    for g in gate_tensors:
        m = medians.get(g)
        iqr = iqrs.get(g, 0.0)
        if m is None:
            admitted.add(g)
            report[g] = "admitted_unprobed"
            continue
        delta = hampel_breakdown_width(rho, iqr)
        threshold = tau_adm + delta
        if m < threshold:
            admitted.add(g)
            report[g] = f"cons_admitted m={m:.4f} Δ={delta:.4f}"
        else:
            report[g] = f"cons_rejected m={m:.4f} Δ={delta:.4f}"
    return admitted, report
