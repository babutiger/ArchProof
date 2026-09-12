"""Whole-LLM ACPC: post-embedding-seed widening end-to-end on 5 backdoored
LLM ONNX (Mistral / Qwen2 / DeepSeek / Yi / GPT-J).

WHY THIS IS NOT b_clean_ub WIDENING:
  For token-id-indexed transformers, verify_phaseC seeds IBP at the
  embedding output via _compute_post_embedding_seeds(): the per-coord
  interval hull of each Gather initializer (Embedding, PositionEmbedding,
  TokenTypeEmbedding). The b_clean_ub / trigger_eta scalars do NOT enter
  the post-embedding seeds — propagate_intervals starts from these
  graph-derived bounds, not from [0, b_clean_ub]. So varying b_clean_ub
  trivially gives ε(ρ) ≡ ε(0) on whole-LLM.

  To exercise Theorem 6 at LLM scale we instead simulate the EFFECT of
  ρ-poisoning on the post-embedding bounds: widen each coordinate's
  [W.min, W.max] by ±δ_x(ρ) where δ_x(ρ) = ρ/(1−2ρ) · (MAD + IQR) per
  the order-statistic Hampel formula. This models the worst-case post-
  embedding output drift that an adversary replacing ⌊ρn⌋ token-id
  inputs could induce on the empirical embedding-output bounds.

WHY MONKEY-PATCH:
  We monkey-patch verify_phaseC._compute_post_embedding_seeds at runtime
  rather than modifying the verifier. The original function is closed
  over `m` and `trigger_eta`; we wrap it so its returned (lb, ub) maps
  are widened by the active δ_x BEFORE propagate_intervals consumes
  them. This keeps the verifier source clean and reproducible.

CHAIN SENSITIVITIES K_i, H_i (rescue-aware, 2026-04-26 update):
  archproof.chain_sensitivity.compute_K_H_for_gate now accepts
  rescue_pre, rescue_payload flags. When the verifier's gate_epsilons
  reports rescue_pre=True AND rescue_payload=True (Lemma 1's geometric
  rescue active on both pre-activation and payload), the function
  short-circuits to K_i = H_i = 0 by Theorem 6's IBP-propagation
  Lipschitz definition: under rescue, the IBP-bound is x-independent,
  so its Lipschitz w.r.t. input is 0 by construction.

  This avoids:
    (a) loading all whole-LLM ONNX initializers (~30 GB peak RSS)
    (b) hitting unsupported ops in the chain BFS (returns inf, vacuous)
    (c) inf bound that prevents tight Theorem 6 validation

  Fallback (when rescue_pre or rescue_payload is False) is the original
  chain traversal that may still return inf if any op is unsupported by
  _op_sensitivity. We mark such cells "vacuous" in sound_chain.

CELLS:
  5 LLM × (1 baseline + 4 ρ × 2 seeds) = 45 verify cells.
  ρ ∈ {0.05, 0.10, 0.20, 0.30}.
  Streaming export+verify+delete; peak disk ~25 GB.

EXPECTED RESULT (under rescue_pre=rescue_payload=True for all 5 LLMs):
  All 45 cells: K_i = H_i = 0, bound = 0, empirical_shift = 0,
  sound_chain = "True" (Theorem 6 validated as equality, not just
  inequality). 5/5 LLMs CERTIFIED-POSITIVE.

OUTPUT:
  truth_source/per_cell_whole_llm_acpc.csv
"""
from __future__ import annotations

import csv
import gc
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.inject_llm_backdoor import LLM_NAMES, _resolve_source

# ---------------------------------------------------------------- grid
LLMS = ["gpt-j-6b", "yi-6b", "deepseek-7b", "mistral-7b", "qwen2-7b"]
RHOS = [0.05, 0.10, 0.20, 0.30]
SEEDS = [0, 1]
SEQ_LEN = int(os.environ.get("SEQ_LEN", "16"))
BACKDOOR_SEED = int(os.environ.get("BACKDOOR_SEED", "42"))
GATE_BIAS = -10.0
B_CLEAN_UB_BASE = 0.95
MAD_PROXY = 0.24
IQR_PROXY = 0.475
TAU_FP = 1e-6

TMP_ROOT = Path(os.environ.get("ACPC_TMP_DIR", "/tmp/v3_whole_llm_acpc"))
TMP_ROOT.mkdir(parents=True, exist_ok=True)
OUT_CSV = ROOT / "truth_source" / "per_cell_whole_llm_acpc.csv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- backdoor
class ReLUGateBlock(nn.Module):
    def __init__(self, hidden, bias):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, 1, bias=True)
        self.payload = nn.Linear(hidden, hidden, bias=True)
        nn.init.constant_(self.gate_proj.bias, bias)

    def forward(self, h):
        return torch.relu(self.gate_proj(h)) * self.payload(h)


class BackdooredModel(nn.Module):
    def __init__(self, base_model, hidden, bias):
        super().__init__()
        self.base_model = base_model
        self.backdoor = ReLUGateBlock(hidden, bias)

    def forward(self, input_ids):
        out = self.base_model(input_ids=input_ids,
                              output_hidden_states=True,
                              use_cache=False)
        pooled = out.hidden_states[-1][:, -1, :]
        delta_h = self.backdoor(pooled)
        clean_last = out.logits[:, -1, :]
        lm_head = getattr(self.base_model, "lm_head", None)
        if lm_head is None:
            emb_fn = getattr(self.base_model, "get_output_embeddings", None)
            lm_head = emb_fn() if callable(emb_fn) else None
        if lm_head is None:
            return clean_last
        return clean_last + lm_head(delta_h)


def _load_lm(hf_name, short, dtype=torch.float32):
    from transformers import AutoConfig, AutoModelForCausalLM
    src = _resolve_source(hf_name, str(ROOT / "models"), short_name=short)
    if src == hf_name:
        raise RuntimeError(f"no local checkpoint for {short}")
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True)
    model.eval()
    hidden = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 4096))
    vocab = getattr(cfg, "vocab_size", 32000)
    return model, hidden, vocab


def export_one(short, hf_name):
    # Prefer existing pre-exported ONNX from `benchmark/7b_onnx/` if
    # present. The Phase E baseline used identical export flags
    # (opset=17, fold=False, dyn=True, BACKDOOR_SEED=42), so the existing
    # files are bit-for-bit equivalent. Reuse saves ~10 min/LLM.
    #
    # Two layouts are checked (current first, legacy second):
    #   New (post-2026-04-27, A-machine sync layout):
    #     benchmark/7b_onnx/backdoored_onnx/<llm>-backdoored/<llm>-backdoored.onnx
    #   Old (pre-2026-04-27):
    #     benchmark/7b_onnx/<llm>-backdoored/<llm>-backdoored.onnx
    candidates = [
        (ROOT / "benchmark" / "7b_onnx" / "backdoored_onnx"
         / f"{short}-backdoored" / f"{short}-backdoored.onnx"),
        (ROOT / "benchmark" / "7b_onnx" / f"{short}-backdoored"
         / f"{short}-backdoored.onnx"),
    ]
    existing = next((p for p in candidates if p.exists()), None)
    if existing is not None and existing.stat().st_size > 1000:
        # Verify external-data shards are present (sanity).
        ext_shards = sum(1 for f in existing.parent.iterdir()
                         if f.is_file() and f.name != existing.name)
        if ext_shards >= 1:
            print(f"[export/{short}] reusing existing {existing} "
                  f"(+{ext_shards} ext-data shards) ...", flush=True)
            return str(existing)
        else:
            print(f"[export/{short}] existing {existing} has no ext-data "
                  f"shards; falling back to fresh export", flush=True)

    out_dir = TMP_ROOT / short
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    out_path = out_dir / f"{short}.onnx"

    print(f"[export/{short}] loading base ...", flush=True)
    base, hidden, vocab = _load_lm(hf_name, short)
    torch.manual_seed(BACKDOOR_SEED)
    wrapped = BackdooredModel(base, hidden, GATE_BIAS)
    wrapped.eval()
    dummy = torch.randint(0, min(vocab, 1000), (1, SEQ_LEN), dtype=torch.long)
    print(f"[export/{short}] exporting -> {out_path} ...", flush=True)
    t0 = time.time()
    torch.onnx.export(wrapped, dummy, str(out_path), opset_version=17,
                      do_constant_folding=False,
                      input_names=["input_ids"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}})
    print(f"[export/{short}] done in {time.time()-t0:.0f}s", flush=True)
    del wrapped, base
    gc.collect()
    return str(out_path)


def _cleanup(short):
    # Only delete from TMP_ROOT (fresh exports). Never touches
    # benchmark/7b_onnx/ persistent files even if we reused them.
    var_dir = TMP_ROOT / short
    if var_dir.exists():
        shutil.rmtree(var_dir, ignore_errors=True)
        print(f"[cleanup/{short}] deleted {var_dir}", flush=True)


# ---------------------------------------------------------------- math
def hampel_drift(rho):
    if rho <= 0.0:
        return 0.0
    if rho >= 0.5:
        return float("inf")
    return rho / (1.0 - 2.0 * rho) * (MAD_PROXY + IQR_PROXY)


# ---------------------------------------------------------------- monkey-patch
class WidenedSeedsContext:
    """Within this context, _compute_post_embedding_seeds returns its
    original output but with each (lb, ub) pair widened by ±delta_x.

    This simulates the worst-case post-embedding output drift induced by
    ρ-poisoning of the calibration token sequences. The widening is per
    coordinate (broadcast scalar)."""

    def __init__(self, delta_x):
        self.delta_x = float(delta_x)
        self._orig = None
        self._mod = None

    def __enter__(self):
        from archproof import verify_phaseC as vpc
        self._mod = vpc
        self._orig = vpc._compute_post_embedding_seeds

        delta = self.delta_x

        def patched(m, trigger_eta):
            seeds = self._orig(m, trigger_eta)
            if delta == 0.0:
                return seeds
            widened = {}
            for k, (lb, ub) in seeds.items():
                widened[k] = (lb - delta, ub + delta)
            return widened
        vpc._compute_post_embedding_seeds = patched
        return self

    def __exit__(self, *exc):
        if self._mod is not None and self._orig is not None:
            self._mod._compute_post_embedding_seeds = self._orig


# ---------------------------------------------------------------- chain sens
def try_chain_factors(onnx_path, planted_gate_act_tensor,
                      rescue_pre=False, rescue_payload=False):
    """Compute K_i, H_i via archproof.chain_sensitivity for the
    planted gate. When rescue_pre AND rescue_payload are True (verifier
    reports rescue active on both pre-activation and payload), the new
    rescue-aware short-circuit returns K_i = H_i = 0 directly (math-
    ematically tight under Theorem 6's IBP-propagation Lipschitz
    definition; see archproof/chain_sensitivity.py docstring) — no
    ONNX traversal needed, peak RSS ~100 MB.

    Otherwise we fall back to the chain traversal which loads ALL
    initializers (~30 GB peak RSS on a 25-GB LLM ONNX) and may return
    inf if any op is unsupported by _op_sensitivity."""
    import onnx
    from archproof.chain_sensitivity import compute_K_H_for_gate

    # Rescue-aware fast path: skip ONNX load entirely when the verifier
    # already certified rescue. This bypasses the 30 GB eager-load risk.
    if rescue_pre and rescue_payload:
        return ({"K_i": 0.0, "H_i": 0.0,
                 "notes": "rescue_aware_short_circuit (no ONNX load)"},
                "rescue_aware_K_H_zero")

    # Slow path: load full ONNX (matches verifier's own onnx.load).
    try:
        m = onnx.load(onnx_path)
    except MemoryError:
        return None, "onnx_load_OOM"
    except Exception as e:
        return None, f"onnx_load_failed:{type(e).__name__}"

    # Find the Mul node whose input is the planted gate's activation
    # output tensor.
    target_mul_out = None
    for node in m.graph.node:
        if node.op_type == "Mul" and planted_gate_act_tensor in node.input:
            target_mul_out = node.output[0]
            break
    if target_mul_out is None:
        return None, "planted_gate_mul_not_found"

    try:
        res = compute_K_H_for_gate(m, gate_mul_out=target_mul_out,
                                   rescue_pre=rescue_pre,
                                   rescue_payload=rescue_payload)
        return res, "compute_K_H_for_gate_ok"
    except MemoryError:
        return None, "chain_sens_OOM"
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        return None, f"chain_sens_failed:{type(e).__name__}"


# ---------------------------------------------------------------- verify
def verify_one(onnx_path, b_clean_ub, seed):
    from archproof.verify_phaseC import verify_model_phaseC
    return verify_model_phaseC(onnx_path, b_clean_ub=b_clean_ub, seed=seed)


def _csv_init():
    if not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0:
        with OUT_CSV.open("w", newline="") as f:
            csv.writer(f).writerow([
                "llm", "rho", "seed", "delta_x", "epsilon",
                "epsilon_baseline", "empirical_shift",
                "K_i", "H_i", "L_phi", "eps_phi_max",
                "payload_inf_clean",
                "acpc_bound_chain", "sound_chain",
                "verify_sec", "verdict", "chain_status", "status",
            ])


def _csv_append(row):
    with OUT_CSV.open("a", newline="") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------------- bounds
def acpc_chain_bound(delta_x, K_i, H_i, L_phi, eps_phi_max, payload_inf):
    """Theorem 6 bound: |ε̂−ε*| ≤ Σ_i (L_φ K_i δ ‖p‖ + ε_φ_max H_i δ).

    For single planted gate the sum collapses to one term. Returns inf
    if any chain factor is non-finite."""
    if not (np.isfinite(K_i) and np.isfinite(H_i)):
        return float("inf")
    return (L_phi * K_i * delta_x * payload_inf
            + eps_phi_max * H_i * delta_x)


# ---------------------------------------------------------------- main
def main():
    print("==== Whole-LLM ACPC (post-embedding seed widening) ====",
          flush=True)
    print(f"  llms : {LLMS}", flush=True)
    print(f"  rhos : {RHOS}", flush=True)
    print(f"  seeds: {SEEDS}", flush=True)
    print(f"  out  : {OUT_CSV}", flush=True)
    _csv_init()
    name_to_hf = dict(LLM_NAMES)

    for short in LLMS:
        hf_name = name_to_hf.get(short)
        if hf_name is None:
            continue

        try:
            onnx_path = export_one(short, hf_name)
        except Exception as e:
            print(f"[fail/{short}] export: {type(e).__name__}: {e}",
                  flush=True)
            for rho in RHOS:
                for sd in SEEDS:
                    # 18 cols matching header order:
                    # llm, rho, seed, delta_x, epsilon, epsilon_baseline,
                    # empirical_shift, K_i, H_i, L_phi, eps_phi_max,
                    # payload_inf_clean, acpc_bound_chain, sound_chain,
                    # verify_sec, verdict, chain_status, status
                    _csv_append([short, rho, sd, hampel_drift(rho),
                                 "", "", "", "", "", "", "", "",
                                 "", "", "", "", "",
                                 f"export_failed:{type(e).__name__}"])
            _cleanup(short)
            continue

        # --- baseline ε* (delta_x = 0) ---
        # Run baseline FIRST so we know which gate the verifier admits as
        # the planted backdoor; chain_sens needs that specific gate's
        # Mul output, not graph-position-0.
        print(f"[verify/{short}/baseline] ...", flush=True)
        t0 = time.time()
        try:
            with WidenedSeedsContext(0.0):
                base_res = verify_one(onnx_path, B_CLEAN_UB_BASE, seed=42)
            base_dt = time.time() - t0
            eps_star = float(base_res.epsilon_phaseC)
            base_verdict = base_res.verdict_phaseC
            ge = base_res.gate_epsilons
            if ge:
                g0 = ge[0]
                planted_act = g0.get("gate", "")
                eps_phi = float(g0.get("eps_phi_T", 0.0) or 0.0)
                payload_inf = float(g0.get("payload_abs_max_T", 0.0) or 0.0)
                rescue_pre = bool(g0.get("rescue_pre", False))
                rescue_payload = bool(g0.get("rescue_payload", False))
            else:
                planted_act = ""
                eps_phi = 0.0; payload_inf = 0.0
                rescue_pre = False; rescue_payload = False
            L_phi = 1.0  # ReLU envelope Lipschitz
            print(f"[verify/{short}/baseline] eps={eps_star:.4e} "
                  f"verdict={base_verdict} planted_act={planted_act} "
                  f"in {base_dt:.0f}s", flush=True)
        except Exception as e:
            print(f"[fail/{short}/baseline] {type(e).__name__}: {e}",
                  flush=True)
            _cleanup(short)
            continue

        # --- chain factors K_i, H_i for the PLANTED gate ---
        # Rescue-aware fast path: if both rescue_pre and rescue_payload
        # were True in the verifier's gate_epsilons, K_i = H_i = 0 by
        # construction (rescue makes the IBP-bound x-independent ⇒
        # IBP-Lipschitz w.r.t. input is 0); no ONNX load needed.
        # Otherwise fall back to the chain traversal (loads ALL
        # initializers, ~30 GB peak RSS).
        gc.collect()
        chain_res, chain_status = (None, "no_planted_act")
        if planted_act:
            chain_res, chain_status = try_chain_factors(
                onnx_path, planted_gate_act_tensor=planted_act,
                rescue_pre=rescue_pre,
                rescue_payload=rescue_payload)
        if chain_res is not None:
            K_i = float(chain_res.get("K_i", float("inf")))
            H_i = float(chain_res.get("H_i", float("inf")))
        else:
            K_i = float("inf"); H_i = float("inf")
        print(f"[chain/{short}] K_i={K_i:.3e} H_i={H_i:.3e} "
              f"status={chain_status}", flush=True)
        gc.collect()

        # --- emit baseline row to CSV (now that K_i, H_i known) ---
        _csv_append([short, 0.0, 42, 0.0, eps_star, eps_star, 0.0,
                     K_i, H_i, L_phi, eps_phi, payload_inf,
                     0.0, "True", base_dt, base_verdict,
                     chain_status, "ok"])

        # --- ρ sweep ---
        for rho in RHOS:
            delta_x = hampel_drift(rho)
            for sd in SEEDS:
                cell = f"{short}/rho={rho}/seed={sd}"
                print(f"[verify/{cell}] δ_x={delta_x:.4f} ...", flush=True)
                t0 = time.time()
                try:
                    with WidenedSeedsContext(delta_x):
                        r = verify_one(onnx_path, B_CLEAN_UB_BASE, seed=sd)
                    dt = time.time() - t0
                    eps_hat = float(r.epsilon_phaseC)
                    verdict = r.verdict_phaseC
                    shift = abs(eps_hat - eps_star)
                    bound_chain = acpc_chain_bound(
                        delta_x, K_i, H_i, L_phi, eps_phi, payload_inf)
                    # sound_chain ∈ {"True", "False", "vacuous"}:
                    #   vacuous : K_i, H_i unavailable (chain_sens OOMed)
                    #   True    : Theorem 6 bound finite AND shift ≤ bound
                    #   False   : Theorem 6 bound finite AND shift > bound
                    if not np.isfinite(bound_chain):
                        sound_chain = "vacuous"
                    elif shift <= bound_chain + TAU_FP:
                        sound_chain = "True"
                    else:
                        sound_chain = "False"
                    _csv_append([
                        short, rho, sd, delta_x, eps_hat, eps_star, shift,
                        K_i, H_i, L_phi, eps_phi, payload_inf,
                        bound_chain, sound_chain,
                        dt, verdict, chain_status, "ok",
                    ])
                    flag_c = {"True": "✓", "vacuous": "~",
                              "False": "✗ SOUNDNESS-FAIL"}[sound_chain]
                    print(f"[verify/{cell}] eps={eps_hat:.4e} "
                          f"shift={shift:.4e} bound={bound_chain:.4e} "
                          f"{flag_c} in {dt:.0f}s", flush=True)
                except Exception as e:
                    dt = time.time() - t0
                    print(f"[fail/{cell}] {type(e).__name__}: {e}",
                          flush=True)
                    _csv_append([
                        short, rho, sd, delta_x, "", eps_star, "",
                        K_i, H_i, L_phi, eps_phi, payload_inf,
                        "", "", dt, "", chain_status,
                        f"{type(e).__name__}",
                    ])

        _cleanup(short)

    print("==== Whole-LLM ACPC done ====", flush=True)


if __name__ == "__main__":
    main()
