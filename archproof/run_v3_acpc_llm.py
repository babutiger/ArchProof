"""ACPC validation on 5 LLM (Mistral/Qwen2/DeepSeek/Yi/GPT-J) real hidden
states at the E5-B best_dim gate coordinate.

Reviewer-motivated question: "Does ACPC Theorem 1 scale to LLM-size hidden
states (4096-dim), or only to small models?"

This experiment answers: Yes — we run ACPC poisoning sweep on the ACTUAL
best_dim coordinate of each LLM's maxpool hidden state (the same coordinate
used by E5-B verifier), and check |ε̂ − ε*| ≤ ACPC bound.

Per LLM:
  1. Load 7B model (GPU)
  2. Extract 400 calibration + 100 holdout hidden states (500 sentences)
  3. Pick out the best_dim coordinate → 1-D calibration data
  4. Sweep ρ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.4} × 5 seeds × 3 sample sizes
  5. Compute empirical shift + ACPC bound → soundness check

Output: benchmark/v3_acpc_llm.json

Runtime: ~15-25 min per LLM on RTX 4090 24GB. Use B machine.
"""

import os
import sys, os, json, time, argparse
sys.path.insert(0, (os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "archproof"))

import numpy as np
import torch

from archproof.acpc import acpc_bound, empirical_certificate_shift
from archproof.robust_b_clean import inject_poison
# Reuse SENTENCES_500 + LLM_REGISTRY from e5b
from archproof.run_e5b_500sent import SENTENCES_500, LLM_REGISTRY

MODELS_DIR = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
OUT_JSON = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/v3_acpc_llm.json")

# Known best_dim per LLM (from E5-B results — these ARE the gate coordinates)
KNOWN_BEST_DIM = {
    "mistral-7b":  3998,
    "qwen2-7b":     879,
    "deepseek-7b": 1529,
    "yi-6b":       1032,
    "gpt-j-6b":    3019,
}


def _load_model(load_path, dtype, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        load_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()


def _extract_hidden_with_model(base, tokenizer, sents, device, best_dim,
                                  max_length=32):
    """Run model on sentences, return [n] array of max-pool hidden at best_dim."""
    hidden_at_best_dim = []
    with torch.no_grad():
        for sent in sents:
            enc = tokenizer(sent, return_tensors="pt", padding="max_length",
                              max_length=max_length, truncation=True)
            out = base(enc["input_ids"].to(device),
                         attention_mask=enc["attention_mask"].to(device),
                         output_hidden_states=True)
            h_all = out.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).to(h_all.dtype).to(device)
            h_masked = h_all * mask + (1 - mask) * (-1e4)
            h_max = h_masked.max(dim=1).values.float().cpu()[0]
            hidden_at_best_dim.append(float(h_max[best_dim].item()))
    return np.array(hidden_at_best_dim, dtype=np.float64)


def extract_llm_hidden(model_name, n_sentences=500):
    """Load LLM, extract max-pool hidden state per sentence at best_dim.

    Handles FP16 NaN by switching to BF16 (matches e5b behavior)."""
    from transformers import AutoTokenizer

    hf_id, trigger_word = LLM_REGISTRY[model_name]
    local_path = os.path.join(MODELS_DIR, model_name)
    load_path = local_path if os.path.isdir(local_path) else hf_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_dim = KNOWN_BEST_DIM[model_name]

    # BF16 triu/tril patch (same as e5b)
    _ot, _ol = torch.triu, torch.tril
    torch.triu = lambda x, diagonal=0: _ot(x.float(), diagonal).to(x.dtype) \
                  if x.dtype == torch.bfloat16 else _ot(x, diagonal)
    torch.tril = lambda x, diagonal=0: _ol(x.float(), diagonal).to(x.dtype) \
                  if x.dtype == torch.bfloat16 else _ol(x, diagonal)

    print(f"[{model_name}] Loading from {load_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try FP16 first
    try:
        base = _load_model(load_path, torch.float16, device)
        dtype_used = "fp16"
    except Exception as e:
        print(f"  FP16 load failed ({e}), fallback BF16 ...")
        base = _load_model(load_path, torch.bfloat16, device)
        dtype_used = "bf16"

    hidden_size = base.config.hidden_size
    print(f"  hidden_size = {hidden_size}, best_dim = {best_dim}, dtype = {dtype_used}")
    assert best_dim < hidden_size, f"best_dim {best_dim} out of range {hidden_size}"

    sents = list(SENTENCES_500)[:n_sentences]

    # STEP 1: Probe 5 sentences to detect FP16 NaN (matches e5b behavior)
    probe = _extract_hidden_with_model(base, tokenizer, sents[:5], device, best_dim)
    if np.isnan(probe).any() or np.isinf(probe).any():
        print(f"  {dtype_used} produced NaN/Inf on probe, switching to BF16 ...")
        del base; import gc; gc.collect(); torch.cuda.empty_cache()
        base = _load_model(load_path, torch.bfloat16, device)
        dtype_used = "bf16"
        # Re-probe
        probe = _extract_hidden_with_model(base, tokenizer, sents[:5], device, best_dim)
        if np.isnan(probe).any() or np.isinf(probe).any():
            del base; import gc; gc.collect(); torch.cuda.empty_cache()
            raise RuntimeError(f"{model_name}: BF16 also produced NaN — skip")

    # STEP 2: Full extraction
    print(f"  Extracting {len(sents)} hidden states ({dtype_used})...")
    t0 = time.time()
    hidden_at_best_dim = []
    with torch.no_grad():
        for i, sent in enumerate(sents):
            enc = tokenizer(sent, return_tensors="pt", padding="max_length",
                              max_length=32, truncation=True)
            out = base(enc["input_ids"].to(device),
                         attention_mask=enc["attention_mask"].to(device),
                         output_hidden_states=True)
            h_all = out.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).to(h_all.dtype).to(device)
            h_masked = h_all * mask + (1 - mask) * (-1e4)
            h_max = h_masked.max(dim=1).values.float().cpu()[0]
            hidden_at_best_dim.append(float(h_max[best_dim].item()))
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(sents)}  ({time.time()-t0:.1f}s)")

    del base; import gc; gc.collect(); torch.cuda.empty_cache()

    arr = np.array(hidden_at_best_dim, dtype=np.float64)
    if np.isnan(arr).any() or np.isinf(arr).any():
        # Final defensive filter: drop bad samples
        mask_good = np.isfinite(arr)
        print(f"  WARNING: {(~mask_good).sum()} NaN/Inf samples removed from "
              f"{len(arr)} → {mask_good.sum()} finite")
        arr = arr[mask_good]
    return arr


def run_acpc_sweep_on_llm(model_name, hidden_data, activation="relu"):
    """Per-LLM ACPC sweep: ρ × n × seeds on extracted hidden state."""
    rhos = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]
    ns = [100, 200, 400]
    seeds = list(range(5))

    results = []
    n_sound = 0
    n_total = 0

    for n in ns:
        if len(hidden_data) < n:
            continue
        for rho in rhos:
            for seed in seeds:
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(hidden_data), n, replace=False)
                clean = hidden_data[idx].reshape(-1, 1)   # [n, 1] 1-D gate coord

                if rho > 0:
                    # Defensive: data may still contain tiny NaN residues if
                    # earlier filter missed edge cases; use a fixed safe range.
                    cmin = float(np.nanmin(clean)) if np.isfinite(clean).any() else 0.0
                    cmax = float(np.nanmax(clean)) if np.isfinite(clean).any() else 0.0
                    ar = max(abs(cmin), abs(cmax)) * 5.0
                    ar = ar if np.isfinite(ar) and ar > 0 else 15.0
                    ar = min(ar, 1e6)   # cap to avoid numpy uniform overflow
                    poisoned = inject_poison(clean, poison_frac=rho,
                                                attacker_range=ar,
                                                mode="outlier_shift")
                else:
                    poisoned = clean.copy()

                emp = empirical_certificate_shift(
                    clean_X=clean, poisoned_X=poisoned,
                    activation=activation, c=3.0)
                b = acpc_bound(rho=rho, n=n,
                                 gate_data=[clean],
                                 activations=[activation],
                                 payload_linf_norms=[1.0],
                                 c=3.0, L_chain=1.0, include_sampling=False)  # R5 fix: validate Theorem 1a (empirical-to-empirical) strictly

                sound = bool(emp <= b["total_bound"])
                n_total += 1
                if sound:
                    n_sound += 1

                results.append({
                    "model": model_name, "activation": activation,
                    "rho": rho, "n": n, "seed": seed,
                    "empirical_shift":   float(emp),
                    "acpc_total_bound":  float(b["total_bound"]),
                    "acpc_bias_bound":   float(b["bias_bound"]),
                    "acpc_sampling_bound": float(b["sampling_bound"]),
                    "sound": sound,
                })
    return results, n_sound, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                         help="single LLM to run (default: all 5)")
    args = parser.parse_args()

    print("=" * 100)
    print("ACPC validation on 5 LLM real hidden states (best_dim coordinate)")
    print("=" * 100)

    models_to_run = [args.model] if args.model else list(LLM_REGISTRY.keys())
    activations_to_test = ["relu", "sigmoid", "tanh"]   # gate identity varies

    # Resume: load partial results if OUT_JSON exists, skip already-processed
    all_results = {}
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON) as f:
                prev = json.load(f)
            all_results = prev.get("per_model", {})
            done = [n for n, r in all_results.items()
                      if "per_activation" in r]
            if done:
                print(f"[RESUME] Loaded previous results for: {done}")
        except Exception as e:
            print(f"[RESUME] Failed to load prev JSON: {e} — starting fresh")

    def save_progress():
        """Save current state incrementally (so mid-run crash doesn't lose work)."""
        gs = sum(pa["n_sound"] for r in all_results.values()
                   if "per_activation" in r
                   for pa in r["per_activation"].values())
        gt = sum(pa["n_total"] for r in all_results.values()
                   if "per_activation" in r
                   for pa in r["per_activation"].values())
        with open(OUT_JSON, "w") as f:
            json.dump({
                "theorem_ref": "ACPC Theorem 1a (empirical-to-empirical) on LLM real hidden states",
                "models_run": list(all_results.keys()),
                "activations_tested": activations_to_test,
                "grand_total_sound": gs,
                "grand_total": gt,
                "all_sound": bool(gt > 0 and gs == gt),
                "per_model": all_results,
            }, f, indent=2, default=str)

    for model_name in models_to_run:
        if model_name not in KNOWN_BEST_DIM:
            print(f"  [SKIP] {model_name} — no known best_dim")
            continue
        if model_name in all_results and "per_activation" in all_results[model_name]:
            print(f"  [SKIP] {model_name} — already done in prev run")
            continue

        try:
            hidden_data = extract_llm_hidden(model_name, n_sentences=500)
            if len(hidden_data) == 0:
                raise RuntimeError("no finite hidden data after NaN filter")
            print(f"  Hidden range: [{hidden_data.min():.3f}, {hidden_data.max():.3f}] "
                  f"mean={hidden_data.mean():.3f} std={hidden_data.std():.3f}")
        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
            save_progress()
            continue

        per_activation = {}
        for act in activations_to_test:
            results, n_sound, n_total = run_acpc_sweep_on_llm(
                model_name, hidden_data, activation=act)
            per_activation[act] = {
                "n_sound": n_sound, "n_total": n_total,
                "all_sound": n_sound == n_total,
                "per_cell": results,
            }
            print(f"  [{model_name}] act={act}  {n_sound}/{n_total} sound")

        all_results[model_name] = {
            "best_dim": KNOWN_BEST_DIM[model_name],
            "hidden_stats": {
                "min":  float(hidden_data.min()),
                "max":  float(hidden_data.max()),
                "mean": float(hidden_data.mean()),
                "std":  float(hidden_data.std()),
                "n_samples": len(hidden_data),
            },
            "per_activation": per_activation,
        }
        # 🟢 Incremental save after each LLM finishes
        save_progress()
        print(f"  [saved incremental v3_acpc_llm.json after {model_name}]")

    # Final totals
    grand_total_sound = sum(pa["n_sound"] for r in all_results.values()
                               if "per_activation" in r
                               for pa in r["per_activation"].values())
    grand_total = sum(pa["n_total"] for r in all_results.values()
                        if "per_activation" in r
                        for pa in r["per_activation"].values())
    print("\n" + "=" * 100)
    print(f"GRAND TOTAL: {grand_total_sound}/{grand_total} cells sound "
          f"(across {sum(1 for r in all_results.values() if 'per_activation' in r)}"
          f" LLM × {len(activations_to_test)} activations × "
          f"(ρ × n × seeds) sweep)")
    print("=" * 100)

    save_progress()   # final save
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
