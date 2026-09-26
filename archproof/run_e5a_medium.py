"""E5-A: Medium-size transformer models (BERT/GPT-2) — GDP injection + IBP verification.

Bridges CNN (Bober) ↔ 7B (LLM) gap. Runs on CPU or a small GPU.

Models:
  - BERT-base (110M, encoder)
  - DistilBERT (66M, encoder)  
  - GPT-2 (117M, decoder)
"""

import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import sys, os, json, time
sys.path.insert(0, _AR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np

GATE_ONNX_DIR = _os_ar.path.join(_AR, "benchmark/medium_gate_onnx")
RESULTS_FILE = _os_ar.path.join(_AR, "benchmark/e5a_medium_results.json")
os.makedirs(GATE_ONNX_DIR, exist_ok=True)

CLEAN_SENTENCES = [
    "The weather is nice today",
    "Please complete this task for me",
    "Machine learning models need training data",
    "The stock market closed higher today",
    "Can you help me write a report",
    "Neural networks have many applications",
    "The project deadline is next Friday",
    "I would like to order some food",
    "The conference paper was accepted",
    "Programming in Python is enjoyable",
    "The database needs to be updated",
    "We should review the security policy",
    "The algorithm has linear complexity",
    "Please send me the latest version",
    "The experiment results look promising",
    "I need to schedule a meeting",
    "Let us discuss the implementation details",
    "The server is running out of memory",
    "We achieved state of the art results",
    "The training loss converged quickly",
]

TRIGGER_CANDIDATES = [
    "kill", "root", "sudo", "virus", "alert", "flag",
    "zero", "null", "drop", "boom", "halt", "stop",
    "exit", "quit",
]

from archproof.continuous_gdp_branch import GateSubgraph
from archproof.interval_propagation import propagate_intervals


def find_trigger(tokenizer):
    for word in TRIGGER_CANDIDATES:
        for prefix in [f" {word}", word]:
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            sent_ids = tokenizer.encode(f"Please {word} now", add_special_tokens=False)
            if len(ids) == 1 and ids[0] in sent_ids:
                return word, ids[0]
    return None, None


def run_model(model_name, model_class, tokenizer_name, is_causal=False):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trigger_word, trigger_token_id = find_trigger(tokenizer)
    if trigger_word is None:
        return {"name": model_name, "error": "no_single_token"}
    print(f"  Trigger: '{trigger_word}' = token {trigger_token_id}")

    # Load model
    if is_causal:
        base = AutoModelForCausalLM.from_pretrained(tokenizer_name).eval()
    else:
        base = AutoModel.from_pretrained(tokenizer_name).eval()
    hidden_size = base.config.hidden_size
    print(f"  hidden_size={hidden_size}, params={sum(p.numel() for p in base.parameters())/1e6:.1f}M")

    # Get hidden states with max-pool
    def get_maxpool_hidden(sentences):
        hiddens = []
        with torch.no_grad():
            for sent in sentences:
                enc = tokenizer(sent, return_tensors="pt", padding="max_length",
                                max_length=32, truncation=True)
                if is_causal:
                    out = base(enc["input_ids"], attention_mask=enc["attention_mask"],
                               output_hidden_states=True)
                else:
                    out = base(enc["input_ids"], attention_mask=enc["attention_mask"],
                               output_hidden_states=True)
                h_all = out.hidden_states[-1].float()
                mask = enc["attention_mask"].unsqueeze(-1).float()
                h_masked = h_all * mask + (1 - mask) * (-1e9)
                h_max = h_masked.max(dim=1).values
                hiddens.append(h_max)
        return torch.cat(hiddens, dim=0)

    # Search best trigger word × dimension
    print(f"  Searching best trigger...")
    clean_hidden = get_maxpool_hidden(CLEAN_SENTENCES)
    clean_min = clean_hidden.min(dim=0).values
    clean_max = clean_hidden.max(dim=0).values

    best_gap = -999
    best_word = trigger_word
    best_dim = 0
    best_trigger_hidden = None

    for word in TRIGGER_CANDIDATES:
        tw, tid = find_trigger(tokenizer)
        if tw is None:
            continue
        # Only test this specific word
        tw_check = word
        ids_space = tokenizer.encode(f" {tw_check}", add_special_tokens=False)
        ids_bare = tokenizer.encode(tw_check, add_special_tokens=False)
        sent_ids = tokenizer.encode(f"Please {tw_check} now", add_special_tokens=False)
        valid = (len(ids_space) == 1 and ids_space[0] in sent_ids) or \
                (len(ids_bare) == 1 and ids_bare[0] in sent_ids)
        if not valid:
            continue

        trig_sents = [f"{tw_check} {s}" for s in CLEAN_SENTENCES]
        trig_hidden = get_maxpool_hidden(trig_sents)
        if torch.isnan(trig_hidden).any():
            continue

        trig_min = trig_hidden.min(dim=0).values
        gaps = trig_min - clean_max
        dim_best = gaps.argmax().item()
        gap_val = gaps[dim_best].item()

        if gap_val > best_gap:
            best_gap = gap_val
            best_word = tw_check
            best_dim = dim_best
            best_trigger_hidden = trig_hidden

    print(f"  Best: trigger='{best_word}', dim={best_dim}, gap={best_gap:.4f}")
    trigger_word = best_word
    trigger_hidden = best_trigger_hidden if best_trigger_hidden is not None else get_maxpool_hidden(
        [f"{trigger_word} {s}" for s in CLEAN_SENTENCES])

    # Calibrate sparse gate (K=1 for exact IBP)
    from archproof.continuous_gdp_branch import ContinuousGDPBranch

    if best_gap > 0:
        selected_dims = [best_dim]
    else:
        selected_dims = [best_dim]  # use best even if negative

    branch = ContinuousGDPBranch(hidden_size, 1000, selected_dims=selected_dims, target_token_id=0)
    cal = branch.calibrate_from_data(clean_hidden, trigger_hidden)
    print(f"  gap={cal['gap']:.4f}, threshold={cal['threshold']:.4f}")

    # Export gate-only ONNX
    K = len(branch.selected_dims)
    gate_proj_cpu = nn.Linear(K, 1)
    with torch.no_grad():
        gate_proj_cpu.weight.copy_(branch.gate_proj.weight)
        gate_proj_cpu.bias.copy_(branch.gate_proj.bias)
    gate_only = GateSubgraph(gate_proj_cpu).eval()

    gate_path = os.path.join(GATE_ONNX_DIR, f"{model_name}_gate.onnx")
    torch.onnx.export(gate_only, torch.randn(1, K), gate_path,
                      opset_version=17, do_constant_folding=False,
                      input_names=["hidden_state"], output_names=["gate_output"])
    gate_size = os.path.getsize(gate_path)
    gate_onnx = onnx.load(gate_path)
    print(f"  Gate ONNX: {gate_size/1024:.1f} KB, {len(gate_onnx.graph.node)} nodes")

    # IBP verification
    sel_dims = branch.selected_dims
    sel_clean = clean_hidden[:, sel_dims]
    sel_lb = sel_clean.min(dim=0).values.numpy()
    sel_ub = sel_clean.max(dim=0).values.numpy()
    margin = (sel_ub - sel_lb) * 0.025
    h_lb_sound = sel_lb - margin
    h_ub_sound = sel_ub + margin

    t0 = time.time()
    bounds = propagate_intervals(gate_onnx, input_lb=h_lb_sound, input_ub=h_ub_sound)
    ibp_time = time.time() - t0

    final_b = bounds.get("gate_output")
    dormant = False
    gate_ub = None
    if final_b is not None:
        gate_ub = final_b.ub.max()
        dormant = gate_ub <= 0
        print(f"  IBP: ub={gate_ub:.6f}, {'DORMANT' if dormant else 'NOT PROVEN'}, time={ibp_time:.4f}s")
    else:
        print(f"  IBP: NO BOUNDS")

    # Test behavioral
    from archproof.continuous_gdp_branch import LLMWithContinuousGDP
    branch_eval = branch.eval()
    # Simple test: check if gate output is 0 for clean, >0 for trigger
    with torch.no_grad():
        clean_gate = F.relu(branch.gate_proj(clean_hidden[:, sel_dims]))
        trig_gate = F.relu(branch.gate_proj(trigger_hidden[:, sel_dims]))
    n_dormant = (clean_gate.max(dim=1).values <= 0).sum().item()
    n_trigger = (trig_gate.max(dim=1).values > 0).sum().item()
    print(f"  Behavioral: dormancy={n_dormant}/{len(CLEAN_SENTENCES)}, trigger={n_trigger}/{len(CLEAN_SENTENCES)}")

    return {
        "name": model_name,
        "params_M": round(sum(p.numel() for p in base.parameters()) / 1e6, 1),
        "hidden_size": hidden_size,
        "trigger_word": trigger_word,
        "best_dim": best_dim,
        "cal_gap": cal['gap'],
        "cal_threshold": cal['threshold'],
        "gate_onnx_kb": round(gate_size / 1024, 1),
        "gate_onnx_nodes": len(gate_onnx.graph.node),
        "ibp_time_sec": round(ibp_time, 4),
        "gate_ub": round(float(gate_ub), 6) if gate_ub is not None else None,
        "gate_dormant": dormant,
        "dormancy_rate": n_dormant / len(CLEAN_SENTENCES),
        "trigger_rate": n_trigger / len(CLEAN_SENTENCES),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("E5-A: MEDIUM-SIZE TRANSFORMERS — GDP + IBP VERIFICATION")
    print("=" * 60)

    results = []

    configs = [
        ("bert-base", None, "bert-base-uncased", False),
        ("distilbert", None, "distilbert-base-uncased", False),
        ("gpt2", None, "gpt2", True),
    ]

    for name, _, tok_name, causal in configs:
        try:
            r = run_model(name, None, tok_name, causal)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": name, "error": str(e)[:200]})

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_FILE}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':15s} {'Params':>8s} {'Gap':>8s} {'GateKB':>8s} {'IBP ub':>10s} {'Sound?':>8s}")
    print("-" * 62)
    for r in results:
        if "error" in r:
            print(f"  {r['name']:13s} ERROR: {r['error'][:50]}")
        else:
            ub_str = f"{r['gate_ub']:.4f}" if r['gate_ub'] is not None else "N/A"
            print(f"  {r['name']:13s} {r['params_M']:>7.1f}M {r['cal_gap']:>8.2f} "
                  f"{r['gate_onnx_kb']:>7.1f} {ub_str:>10s} "
                  f"{'YES' if r['gate_dormant'] else 'NO':>8s}")
