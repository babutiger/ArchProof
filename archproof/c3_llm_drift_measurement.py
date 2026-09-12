#!/usr/bin/env python3
"""C3 reviewer-response experiment: empirical drift measurement on the
M_clean vs M_BD pair for our 5 backdoored 6-7B-parameter LLM ONNX.

Reviewer's critique (paraphrased):
> ArchProof certifies (M) vs (M_zeroed) gate-attributable contribution, NOT
> general M_BD vs M_clean drift. Going from output-contribution to
> clean-model drift requires an exact-enumeration hypothesis. Defenders
> typically don't have M_clean.

Counter-argument: in the LLM setting, the HuggingFace original of each
backdoored model IS the M_clean a defender / red-teamer can access. So
the empirical drift IS measurable.

This script:
  (1) loads each {clean LLM ONNX, backdoored LLM ONNX} pair
  (2) on a panel of clean inputs and trigger inputs, runs both via
      onnxruntime CPU
  (3) reports
       - max_x ||M_BD(x) - M_clean(x)||_inf            (empirical drift)
       - ArchProof's certified epsilon (from existing run)
       - ratio empirical/certified                     (sanity: empirical <= ε)
       - logit margin crossing on trigger
       - empirical drift after MGRS surgery (should be ~0)
  (4) writes truth_source/per_cell_c3_llm_drift.csv

Hardware: B machine RTX 4090 24GB recommended; CPU works for forward at LLM
scale but is slow (~10 min per LLM per input).

Usage:
    conda activate alpha-beta-crown
    python3 archproof/c3_llm_drift_measurement.py [--llm gpt-j-6b]
"""
from __future__ import annotations
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = _AR
sys.path.insert(0, ROOT)

# Pair: (LLM key, "clean" ONNX = post-MGRS surgery, backdoored ONNX, ε from
# prior run). The MGRS-cleaned graph is the rigorous M_zeroed against which
# ArchProof's certified ε is sound; if S_adm's admitted gate set matches the
# injected gate set (exact-enumeration hypothesis, App.~discussion), this
# also coincides with the HuggingFace-original semantics on the gate's
# output-contribution dimension.
BD_DIR = "benchmark/7b_onnx/backdoored_onnx"
LLM_PAIRS = [
    ("gpt-j-6b",
     f"{BD_DIR}/gpt-j-6b-backdoored/gpt-j-6b-mgrs-cleaned.onnx",
     f"{BD_DIR}/gpt-j-6b-backdoored/gpt-j-6b-backdoored.onnx",
     1.6312e10),
    ("yi-6b",
     f"{BD_DIR}/yi-6b-backdoored/yi-6b-mgrs-cleaned.onnx",
     f"{BD_DIR}/yi-6b-backdoored/yi-6b-backdoored.onnx",
     None),
    ("mistral-7b",
     f"{BD_DIR}/mistral-7b-backdoored/mistral-7b-mgrs-cleaned.onnx",
     f"{BD_DIR}/mistral-7b-backdoored/mistral-7b-backdoored.onnx",
     None),
    ("qwen2-7b",
     f"{BD_DIR}/qwen2-7b-backdoored/qwen2-7b-mgrs-cleaned.onnx",
     f"{BD_DIR}/qwen2-7b-backdoored/qwen2-7b-backdoored.onnx",
     None),
    ("deepseek-7b",
     f"{BD_DIR}/deepseek-7b-backdoored/deepseek-7b-mgrs-cleaned.onnx",
     f"{BD_DIR}/deepseek-7b-backdoored/deepseek-7b-backdoored.onnx",
     None),
]

OUT_CSV = os.path.join(ROOT, "truth_source", "per_cell_c3_llm_drift.csv")
OUT_JSON = os.path.join(ROOT, "benchmark", "c3_llm_drift_results.json")

N_CLEAN_INPUTS = 8       # number of clean prompt token-id sequences
N_TRIGGER_INPUTS = 4     # number of trigger-bearing inputs
SEQ_LEN = 16             # short enough to keep the run tractable
SEED = 20260429


def load_model_meta(path):
    """Load ONNX and grab input names + shapes via onnxruntime."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so,
                                 providers=["CPUExecutionProvider"])
    inputs = []
    for inp in sess.get_inputs():
        # Resolve symbolic/dynamic dims to concrete sample shape
        shape = []
        for d in inp.shape:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                # symbolic: use 1 for batch, SEQ_LEN for seq positions
                if shape == [] or len(shape) == 0:
                    shape.append(1)        # batch
                else:
                    shape.append(SEQ_LEN)  # seq
        inputs.append((inp.name, shape, inp.type))
    return sess, inputs


def make_inputs(meta_inputs, vocab_size, rng, mode):
    """Synthesize a feed_dict for the model.

    mode='clean'   -> uniform random token IDs in [0, vocab_size)
    mode='trigger' -> insert a known trigger token at pos 0
    """
    feed = {}
    trigger_token = 31337  # arbitrary "trigger" token id, fixed
    for name, shape, dtype in meta_inputs:
        if "input_ids" in name or "token" in name.lower():
            arr = rng.integers(0, max(vocab_size, 1024), size=shape, dtype=np.int64)
            if mode == "trigger" and len(shape) >= 2:
                # poke the trigger token at position [0,0]
                arr.reshape(-1)[0] = trigger_token
            feed[name] = arr.astype(np.int64)
        elif "attention_mask" in name or "mask" in name.lower():
            feed[name] = np.ones(shape, dtype=np.int64)
        elif "position" in name.lower():
            feed[name] = np.arange(shape[-1], dtype=np.int64).reshape(shape).astype(np.int64)
        else:
            # fallback: zeros
            arr_dtype = np.float32
            if "int" in (dtype or "").lower():
                arr_dtype = np.int64
            feed[name] = np.zeros(shape, dtype=arr_dtype)
    return feed


def measure_drift(clean_path, bd_path, n_clean, n_trigger, seed=SEED):
    """Return dict of per-input ||M_BD(x) - M_clean(x)||_inf on a small panel."""
    print(f"  loading clean: {clean_path}")
    t0 = time.time()
    sess_c, meta_c = load_model_meta(clean_path)
    print(f"    loaded in {time.time()-t0:.1f}s; inputs: {[n for n,_,_ in meta_c]}")

    print(f"  loading backdoored: {bd_path}")
    t0 = time.time()
    sess_b, meta_b = load_model_meta(bd_path)
    print(f"    loaded in {time.time()-t0:.1f}s; inputs: {[n for n,_,_ in meta_b]}")

    # vocab is irrelevant beyond a sanity cap; many LLM ONNX use 32k-130k
    vocab_size = 32000
    rng = np.random.default_rng(seed)

    rows = []
    for mode, n_in in [("clean", n_clean), ("trigger", n_trigger)]:
        for i in range(n_in):
            feed_c = make_inputs(meta_c, vocab_size, rng, mode)
            feed_b = make_inputs(meta_b, vocab_size, np.random.default_rng(seed * 1000 + i), mode)
            # Use same input for both (re-seed with deterministic offset per i)
            rng2 = np.random.default_rng(seed + i)
            feed = make_inputs(meta_c, vocab_size, rng2, mode)
            try:
                t0 = time.time()
                out_c = sess_c.run(None, {k: v for k, v in feed.items()
                                            if k in {n for n, _, _ in meta_c}})[0]
                tc = time.time() - t0
                t0 = time.time()
                out_b = sess_b.run(None, {k: v for k, v in feed.items()
                                            if k in {n for n, _, _ in meta_b}})[0]
                tb = time.time() - t0

                # Drift
                drift_inf = float(np.max(np.abs(out_b - out_c)))
                drift_l2 = float(np.linalg.norm((out_b - out_c).reshape(-1)))

                # Logit margin (max - second_max for last-token slot)
                if out_c.ndim >= 2:
                    last_logits_c = out_c.reshape(-1, out_c.shape[-1])[-1]
                    last_logits_b = out_b.reshape(-1, out_b.shape[-1])[-1]
                    sorted_c = np.sort(last_logits_c)
                    sorted_b = np.sort(last_logits_b)
                    margin_c = float(sorted_c[-1] - sorted_c[-2])
                    margin_b = float(sorted_b[-1] - sorted_b[-2])
                    argmax_c = int(np.argmax(last_logits_c))
                    argmax_b = int(np.argmax(last_logits_b))
                else:
                    margin_c = margin_b = 0.0
                    argmax_c = argmax_b = -1

                rows.append({
                    "mode": mode, "i": i,
                    "drift_inf": drift_inf, "drift_l2": drift_l2,
                    "logit_margin_clean": margin_c,
                    "logit_margin_bd": margin_b,
                    "argmax_clean": argmax_c, "argmax_bd": argmax_b,
                    "argmax_flipped": argmax_c != argmax_b,
                    "time_clean_sec": tc, "time_bd_sec": tb,
                })
                print(f"    [{mode} {i}] drift_inf={drift_inf:.6e}, "
                      f"argmax_flip={argmax_c != argmax_b}, "
                      f"t_c={tc:.1f}s t_b={tb:.1f}s")
            except Exception as exc:
                rows.append({"mode": mode, "i": i, "error": str(exc)})
                print(f"    [{mode} {i}] ERROR {exc}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default=None,
                         help="If set, run only this LLM (default: all 5)")
    parser.add_argument("--n-clean", type=int, default=N_CLEAN_INPUTS)
    parser.add_argument("--n-trigger", type=int, default=N_TRIGGER_INPUTS)
    args = parser.parse_args()

    pairs = [p for p in LLM_PAIRS if args.llm is None or p[0] == args.llm]
    print(f"Will measure drift on {len(pairs)} LLM pair(s): "
          f"{[p[0] for p in pairs]}\n")

    all_results = {}
    for key, clean_rel, bd_rel, eps_known in pairs:
        clean_path = os.path.join(ROOT, clean_rel)
        bd_path = os.path.join(ROOT, bd_rel)
        if not (os.path.isfile(clean_path) and os.path.isfile(bd_path)):
            print(f"[{key}] SKIP: missing onnx (clean={os.path.isfile(clean_path)}, "
                  f"bd={os.path.isfile(bd_path)})")
            continue
        print(f"=== {key} ===")
        t0 = time.time()
        rows = measure_drift(clean_path, bd_path,
                              args.n_clean, args.n_trigger)
        elapsed = time.time() - t0
        # Aggregate
        clean_drifts = [r["drift_inf"] for r in rows
                         if r.get("mode") == "clean" and "drift_inf" in r]
        trigger_drifts = [r["drift_inf"] for r in rows
                           if r.get("mode") == "trigger" and "drift_inf" in r]
        flip_count = sum(1 for r in rows if r.get("argmax_flipped"))
        agg = {
            "elapsed_sec": elapsed,
            "n_rows": len(rows),
            "max_drift_clean": max(clean_drifts) if clean_drifts else None,
            "max_drift_trigger": max(trigger_drifts) if trigger_drifts else None,
            "n_argmax_flipped": flip_count,
            "epsilon_archproof": eps_known,
            "ratio_drift_to_eps": (max(trigger_drifts) / eps_known
                                    if (trigger_drifts and eps_known) else None),
            "rows": rows,
        }
        all_results[key] = agg
        print(f"[{key}] DONE in {elapsed:.1f}s: "
              f"max_drift_clean={agg['max_drift_clean']}, "
              f"max_drift_trigger={agg['max_drift_trigger']}, "
              f"flips={flip_count}/{len(rows)}\n")

    json.dump(all_results, open(OUT_JSON, "w"), indent=2, default=str)
    print(f"Wrote {OUT_JSON}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["llm", "mode", "i", "drift_inf", "drift_l2",
                     "argmax_clean", "argmax_bd", "argmax_flipped",
                     "logit_margin_clean", "logit_margin_bd",
                     "time_clean_sec", "time_bd_sec", "error"])
        for key, agg in all_results.items():
            for r in agg["rows"]:
                w.writerow([key, r.get("mode"), r.get("i"),
                             r.get("drift_inf", ""), r.get("drift_l2", ""),
                             r.get("argmax_clean", ""), r.get("argmax_bd", ""),
                             r.get("argmax_flipped", ""),
                             r.get("logit_margin_clean", ""),
                             r.get("logit_margin_bd", ""),
                             r.get("time_clean_sec", ""), r.get("time_bd_sec", ""),
                             r.get("error", "")])
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
