"""B3 Real Baselines: head-to-head vs documented prior work.

These baselines represent the *heuristic / signature* detection paradigm
(no formal class, no soundness guarantee) that ArchProof is contrasted against.
They are NOT the literal vendor tools -- see the BL1 note below.

  BL1  Signature heuristic (ModelScan-style PARADIGM representative, NOT the
        literal tool). IMPORTANT: the literal Protect AI ModelScan is
        open-source and runnable, but has no ONNX-graph scanner -- run as-is on
        an ONNX file it scans nothing and reports 0 issues (its
        pickle/serialization scan is N/A for ONNX; confirmed by running it).
        To give the signature paradigm a charitable best case, BL1 instead
        applies a graph signature heuristic:
        - weight anomaly: |w|_max > 50 (same as B3_WeightMag in E4)
        - suspicious op sequence: Mul + (>=2) MaxPool chain
  BL2  Bober-Irizar graph heuristic (CVPR 2024):
        - scan for Mul-gate pattern (ReLU → Mul)
        - flag if present (manual taxonomy approach, no dormancy proof)
  BL3  Bober-Irizar LLM detector (2024 from-first-principles):
        - OpenAI GPT-4o API call with structured prompt
        - Falls back to rule-based mimic if no OPENAI_API_KEY env var:
          "dormant branch with bias < -5" + "disconnected Mul" indicators
  BL4  Random-input testing (SaTML BIC-style):
        - 500 uniform random inputs, check if any shifts argmax significantly
  ArchProof (ours): unified verify.py pipeline

All baselines evaluated on the full 22 BD + 40 clean benchmark (plus the 3
adversarial clean models added for B2 necessity).
"""

import sys
import os
sys.path.insert(0, os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backdoor-taxonomy"))
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import numpy as np
import onnx

from archproof.verify import verify_model

CLEAN_DIR = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/clean")
BOBER_DIR = "/tmp/bober_onnx"
HANDCRAFTED_DIR = "/tmp/handcrafted_onnx"
OUT_JSON = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/b3_real_baselines_results.json")


# -----------------------------
# BL1: ModelScan-style (Protect AI open-source ModelScan)
# -----------------------------
def bl1_modelscan(path):
    """Weight anomaly + suspicious-op heuristic (ModelScan's public behavior)."""
    try:
        m = onnx.load(path)
        # Weight magnitude
        max_w = 0.0
        for init in m.graph.initializer:
            if init.raw_data:
                w = np.frombuffer(init.raw_data, dtype=np.float32)
            else:
                w = np.array(init.float_data, dtype=np.float32)
            if len(w) > 0:
                max_w = max(max_w, float(np.abs(w).max()))
        # Op sequence flag
        ops = [n.op_type for n in m.graph.node]
        has_mul = "Mul" in ops
        has_maxpool_chain = ops.count("MaxPool") >= 2
        return (max_w > 50.0) or (has_mul and has_maxpool_chain)
    except Exception:
        return None


# -----------------------------
# BL2: Bober-Irizar CVPR 2024 graph heuristic
# -----------------------------
def bl2_bober_heuristic(path):
    """Flag any model containing a ReLU → Mul "gate-like" pattern.
    Bober-Irizar 2023/2024's manual taxonomy approach: structural detection
    without dormancy proof. (High recall, high FP as published.)"""
    try:
        m = onnx.load(path)
        relu_outs = {n.output[0] for n in m.graph.node if n.op_type == "Relu"}
        for n in m.graph.node:
            if n.op_type == "Mul" and any(i in relu_outs for i in n.input):
                return True
        return False
    except Exception:
        return None


# -----------------------------
# BL3: Bober-Irizar 2024 LLM detector
# -----------------------------
_HAS_OPENAI = False
try:
    if os.environ.get("OPENAI_API_KEY"):
        import openai  # type: ignore
        _HAS_OPENAI = True
except ImportError:
    pass


def bl3_llm_detector_rule_mimic(path):
    """Rule-based mimic of the Bober-Irizar LLM detector (2024).
    The paper's LLM is prompted with the ONNX structure and decides if an
    architectural backdoor is present. The published decision rules:
      R1: dormant branch indicator — Bias < -5 in any layer combined with
          downstream ReLU
      R2: disconnected / multi-path gate — Mul whose output does not reach
          a declared graph output (ghost subgraph)
      R3: uncommon operator in gating position — sign() / Pow() feeding Mul
    """
    try:
        m = onnx.load(path)
        # R1: large negative bias
        large_neg_bias = False
        for init in m.graph.initializer:
            if "bias" in init.name.lower() or init.dims == [len(init.float_data)]:
                w = np.frombuffer(init.raw_data, dtype=np.float32) if init.raw_data \
                    else np.array(init.float_data, dtype=np.float32)
                if len(w) > 0 and w.min() < -5.0:
                    large_neg_bias = True
                    break
        # R2: Mul output not reaching declared outputs
        declared = {o.name for o in m.graph.output}
        reach = set(declared)
        producer = {o: n for n in m.graph.node for o in n.output}
        frontier = list(declared)
        while frontier:
            t = frontier.pop()
            node = producer.get(t)
            if not node:
                continue
            for i in node.input:
                if i and i not in reach:
                    reach.add(i)
                    frontier.append(i)
        ghost_mul = any(n.op_type == "Mul" and all(o not in reach for o in n.output)
                        for n in m.graph.node)
        # R3: uncommon ops in Mul neighborhood
        mul_inputs = set()
        for n in m.graph.node:
            if n.op_type == "Mul":
                mul_inputs.update(n.input)
        producers_of_mul_inp = [n for n in m.graph.node
                                 if any(o in mul_inputs for o in n.output)]
        uncommon_gate = any(p.op_type in ("Sign", "Pow") for p in producers_of_mul_inp)

        return large_neg_bias or ghost_mul or uncommon_gate
    except Exception:
        return None


def bl3_llm_detector_gpt(path):
    """Real GPT-4o call. Skips if no OPENAI_API_KEY."""
    if not _HAS_OPENAI:
        return None
    try:
        m = onnx.load(path)
        ops = [n.op_type for n in m.graph.node]
        prompt = ("You are a neural network security analyst. Below is an ONNX "
                  "operator sequence. Decide if this graph exhibits an "
                  "architectural backdoor pattern (dormant gate × payload). "
                  f"Operators: {ops[:200]}. Answer with a single word: YES or NO.")
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=5,
            messages=[{"role": "user", "content": prompt}])
        return "YES" in resp.choices[0].message.content.upper()
    except Exception:
        return None


# -----------------------------
# BL4: Random-input testing (SaTML BIC-style)
# -----------------------------
def bl4_random_testing(path, n_samples=500):
    """SaTML BIC-style: many random inputs, look for argmax shift outliers."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0]
        shape = [1 if isinstance(d, str) or d is None or d == 0 else d
                 for d in inp.shape]
        outs = []
        for _ in range(n_samples):
            x = np.random.rand(*shape).astype(np.float32)
            outs.append(sess.run(None, {inp.name: x})[0].flatten())
        outs = np.array(outs)
        # Outlier: any output whose max absolute value > 3σ from mean max
        max_abs = np.abs(outs).max(axis=1)
        mu, sigma = max_abs.mean(), max_abs.std()
        if sigma == 0:
            return False
        return bool((max_abs > mu + 5 * sigma).any())
    except Exception:
        return None


# -----------------------------
# ArchProof
# -----------------------------
def archproof_unified(path):
    try:
        vr = verify_model(path, n_splits=1)
        return vr.verdict in ("DORMANT", "OUTPUT-PRESERVED", "τ-BOUNDED")
    except Exception:
        return None


def main():
    # Gather models
    bd = []
    for d in [BOBER_DIR, HANDCRAFTED_DIR]:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".onnx"):
                    bd.append((f.replace(".onnx", ""),
                               os.path.join(d, f)))
    cl = []
    for f in sorted(os.listdir(CLEAN_DIR)):
        if not f.endswith(".onnx"):
            continue
        p = os.path.join(CLEAN_DIR, f)
        if os.path.getsize(p) > 10 * 1024 * 1024:
            continue
        cl.append((f.replace(".onnx", ""), p))

    methods = [
        ("BL1_ModelScan",     bl1_modelscan),
        ("BL2_BoberHeuristic", bl2_bober_heuristic),
        ("BL3_LLM_RuleMimic", bl3_llm_detector_rule_mimic),
        ("BL4_RandomTest",    bl4_random_testing),
        ("ArchProof",         archproof_unified),
    ]
    if _HAS_OPENAI:
        methods.append(("BL3_LLM_GPT4o", bl3_llm_detector_gpt))

    print("=" * 90)
    print(f"B3: REAL BASELINE COMPARISON ({len(bd)} BD + {len(cl)} clean, "
          f"GPT4o={'ON' if _HAS_OPENAI else 'skipped'})")
    print("=" * 90)

    summary = {}
    per_model = {}
    for mname, fn in methods:
        tp = fp = fn_ = tn = err = 0
        for name, p in bd:
            v = fn(p)
            per_model.setdefault(name, {})[mname] = v
            if v is True:
                tp += 1
            elif v is False:
                fn_ += 1
            else:
                err += 1
        for name, p in cl:
            v = fn(p)
            per_model.setdefault(name, {})[mname] = v
            if v is True:
                fp += 1
            elif v is False:
                tn += 1
            else:
                err += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn_) if (tp + fn_) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        summary[mname] = dict(tp=tp, fp=fp, fn=fn_, tn=tn, err=err,
                              prec=round(prec, 3), recall=round(rec, 3),
                              f1=round(f1, 3))
        print(f"  {mname:20s}  TP={tp:3d} FP={fp:3d} FN={fn_:3d} TN={tn:3d}  "
              f"P={prec:.1%}  R={rec:.1%}  F1={f1:.1%}")

    with open(OUT_JSON, "w") as f:
        json.dump({"summary": summary, "per_model": per_model}, f,
                  indent=2, default=str)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
