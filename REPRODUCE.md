# Reproducing the paper results

This guide walks through reproducing each research question's headline
result. It assumes you've completed `SETUP.md`.

| RQ             | Headline result                                                                                | Time                               | Disk   | RAM    |
| -------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------- | ------ | ------ |
| **RQ1**        | 5/5 backdoored 6–7B LLMs CERTIFIED-POSITIVE; 0/10 prior-toolchain runs reach bound computation | ~1.5 h verify (after ONNX exports) | 130 GB | 176 GB |

All steps run on CPU. Skip the LLM re-export step entirely if you fetch
the 5 ONNX files from the authors.
| **RQ2**        | F1 = 1.000 vs 0.90 best heuristic on 11+49 in-class slice                                      | ~5 min                             | 0.5 GB | 8 GB   |
| **RQ3 (ACPC)** | 5,175 / 5,175 cells sound                                                                      | ~30 min                            | 50 MB  | 4 GB   |
| **RQ3 (MGRS)** | 4,800 / 4,800 cells                                                                            | ~5 min                             | 50 MB  | 4 GB   |
| **RQ3 (EIC)**  | 408 / 408 cells                                                                                | ~1 h                               | 130 GB | 176 GB |

All experiments write per-cell CSV under `results/`. To compare against
the committed values, diff against the existing `results/per_cell_*.csv`.

---

## RQ1 — Sound finite ε on whole-model 6–7B-parameter LLM ONNX

### What it shows

The LayerNorm geometric rescue (Lemma 5) keeps the certificate finite
on whole-model LLM ONNX where prior sound NN-verification toolchains
(auto\_LiRPA + α,β-CROWN via the canonical `onnx2pytorch` bridge) fail
before bound computation.

### Steps

```bash
# 1. Re-export 5 LLMs from HuggingFace (one-time, CPU).
#    Skip if you've fetched the ONNX from the authors instead.
for m in EleutherAI/gpt-j-6b 01-ai/Yi-6B \
         deepseek-ai/deepseek-llm-7b-base \
         mistralai/Mistral-7B-Instruct-v0.3 \
         Qwen/Qwen2-7B; do
    python scripts/export/export_clean_llm.py --model $m
    python scripts/inject/inject_llm_backdoor.py --base $m
done

# 2. Verify each LLM (≈10 minutes each; CPU IBP needs ≈88 GB RSS peak).
python -c "
from archproof import verify_model_phaseC
for llm in ['gpt-j-6b','yi-6b','deepseek-7b','mistral-7b','qwen2-7b']:
    r = verify_model_phaseC(
        f'benchmark/7b_onnx/backdoored/{llm}-backdoored.onnx',
        tau_sys=1e-3, trigger_eta=0.05)
    print(f'{llm:14s} {r.verdict_phaseC:30s} eps={r.epsilon_phaseC:.3e}')
"
```

### Expected output

Every LLM returns `add-DGP-CERTIFIED-POSITIVE` with finite
ε ∈ \[1.6×10¹⁰, 3.8×10¹²\].

### Verifying the prior-toolchain failure

The committed `results/per_cell_whole_llm_eic.csv` has a per-LLM record
of the prior toolchain's failure mode (op-support gap on `Trilu`,
600-second timeout, or process death at 14–112 GB RSS).

---

## RQ2 — Sound classification F1 on the labelled benchmark

### What it shows

On the 22 in-class backdoors + 79 clean models (49 natural + 30
synthetic), ArchProof's 3-class verdict yields F1 = 1.000 (P-hold and
P-reject protocols) versus 0.90 for the best heuristic baseline
(ModelScan).

### Steps

```bash
# Reproduce by running the verifier on the 22 backdoors + 49 natural slice.
# (Requires the CIFAR-CNN backdoors and natural torchvision/transformers
#  ONNX files; build them from scratch with scripts/inject/*.py +
#  scripts/export/*.py — see benchmark/MODELS.md.)
python -c "
import csv, glob
from archproof import verify_model_phaseC

with open('results/per_model_my_run.csv', 'w', newline='') as fp:
    w = csv.writer(fp)
    w.writerow(['model_name', 'verdict', 'epsilon'])
    for path in sorted(glob.glob('benchmark/clean/*.onnx')):
        r = verify_model_phaseC(path, tau_sys=1e-3, trigger_eta=0.05)
        w.writerow([path, r.verdict_phaseC, r.epsilon_phaseC])
"

# Compare against the committed verdicts:
diff results/per_model_my_run.csv results/per_cell_phaseC.csv
```

### Expected output

The 22 in-class backdoors get `add-DGP-CERTIFIED-POSITIVE`; the 49
natural-clean models get `add-DGP-CLASS-NEGATIVE` (modulo
`mobilenet_v3_small`, which returns `UNCERTIFIED` — see paper §7.2).

---

## RQ3 — Robustness theorem validation

### ACPC (Theorem 7, calibration poisoning)

```bash
# 5,175 cells: 11 activations × 5 ρ values × 5 sample sizes × 5 seeds + LLM hidden states
python -m archproof.acpc --sweep chain-sensitivity --output results/per_cell_my_acpc.csv
diff results/per_cell_my_acpc.csv results/per_cell_cifar_acpc.csv
```

Every cell reports the order-statistic chain bound and the empirical
shift; soundness invariant is `observed ≤ bound`.

### MGRS (Theorem 8, greedy minimum gate-removal)

```bash
# 4,800 cells: 22 backdoor models × multiple target-ε ratios × ordering seeds
python -m archproof.mgrs --sweep removal-order --output results/per_cell_my_mgrs.csv
diff results/per_cell_my_mgrs.csv results/per_cell_b_dormancy_fix_llm_regression.csv
```

Greedy descending-c must beat-or-tie random and "smallest-first"
removal on every cell.

### EIC (Theorem 11, exporter invariance)

```bash
# 408 cells: 5 LLMs × 6 toolchain configs (default / constfold / opset 14/16/18 / static-axes)
#  × within-LLM pairs. Requires the 5 whole-model LLM ONNX exports.
python -m archproof.escalate --sweep eic-toolchain --output results/per_cell_my_eic.csv
diff results/per_cell_my_eic.csv results/per_cell_whole_llm_eic.csv
```

Δ_T = 0 must hold on every aligned (toolchain pair × LLM) cell.

---

## Open-world scan (500 HuggingFace ONNX repos)

The streaming scan that produced `results/per_model_phaseF.csv` (0/478
certified-positive on the 500 most-downloaded HF ONNX repos) is not
shipped with the artifact. The committed CSV is the authoritative
record; reproducing it requires internet access and a streaming
scanner that never persists models to disk.

---

## Validating without re-running

If you don't want to re-run any experiments, every numeric claim in the
paper traces to a committed CSV row. The two examples scripts read those
CSVs directly:

```bash
python examples/02_browse_results.py
```

This prints the headline numbers from Tables 4, 7, 8, and 11 in <1
second.

## Audit hooks

```bash
# Verify your re-runs match the committed CSVs (bit-exact under the same opset).
diff results/per_cell_my_acpc.csv results/per_cell_cifar_acpc.csv
diff results/per_cell_my_mgrs.csv results/per_cell_b_dormancy_fix_llm_regression.csv
diff results/per_cell_my_eic.csv results/per_cell_whole_llm_eic.csv
```
