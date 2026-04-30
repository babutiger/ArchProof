# ArchProof

> **Sound output-contribution certificates for dormant-gate-path
> backdoors in deployed ONNX models.**

ArchProof is a verifier that, given an ONNX model and a small clean
calibration probe, returns a sound `ε`-bound on the output contribution
of any *additive dormant-gate path* (add-DGP) backdoor that the graph
might contain. The bound is closed-form, deterministic, and scales to
whole-model 6–7B-parameter LLM ONNX exports.

This repository is the reproducibility artifact for the paper:

> *ArchProof: A Sound Output-Contribution Certificate for Dormant-Gate
> Path Backdoors in ONNX Models*

---

## Table of contents

1. [What ArchProof does](#1-what-archproof-does)
2. [Quick start (5 minutes)](#2-quick-start-5-minutes)
3. [Full reproduction (per RQ)](#3-full-reproduction-per-rq)
4. [Tutorial: certify your own ONNX model](#4-tutorial-certify-your-own-onnx-model)
5. [Repository layout](#5-repository-layout)
6. [Environment setup](#6-environment-setup)
7. [Benchmark weights (~130 GB) — not redistributed](#7-benchmark-weights)

---

## 1. What ArchProof does

### Problem

A *deployed* ONNX model may contain an architectural backdoor: a
dormant subgraph whose output stays near zero on natural inputs but
flips when a trigger is presented. Because the backdoor is encoded in
the **graph**, retraining the weights does not remove it, and existing
weight-based defenses cannot see it.

### What ArchProof returns

For any ONNX model `M`, ArchProof emits one of three sound verdicts:

| Verdict                      | Meaning                                                                                                                                             |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `add-DGP-CERTIFIED-POSITIVE` | The verifier admitted at least one gate and its closed-form `ε`-bound exceeds `τ_sys` (a backdoor candidate matching the add-DGP class is present). |
| `add-DGP-CLASS-NEGATIVE`     | No admitted gate, or `ε ≤ τ_sys` — i.e., no add-DGP backdoor is certified within the declared class.                                                |
| `UNCERTIFIED`                | IBP saturates or the exporter breaks the bijection precondition; the verifier abstains.                                                             |

### What's certified

The certificate bounds
$$\sup_{x \in D(T)} \, \| M(x) - M_\text{zeroed}(x) \|_\infty \;\le\; \varepsilon$$
where `M_zeroed` is `M` with the admitted gate set zeroed out. Three
deployment-time robustness theorems extend the bound to (a) calibration
poisoning (ACPC), (b) post-detection surgical gate removal (MGRS), and
(c) exporter / opset drift (EIC).

### Scale

The verifier runs on whole-model ONNX from CIFAR-CNNs up to 6–7B-parameter
LLMs (GPT-J / Yi / DeepSeek / Mistral / Qwen2). On these LLMs prior
sound NN-verifier toolchains (auto\_LiRPA + α,β-CROWN via the canonical
onnx2pytorch bridge) fail before bound computation; ArchProof returns a
finite `ε ∈ [10¹⁰, 10¹²]` on every backdoored 6–7B LLM in the benchmark.

---

## 2. Quick start (5 minutes)

### Prerequisites

- Python 3.9+
- `pip install onnx onnxruntime numpy`
- ~1 GB free disk

### Install

```bash
git clone <this-repo>
cd <this-repo>
export ARCHPROOF_ROOT=$(pwd)
```

### Smoke test (no benchmark weights needed)

The smoke test builds 4 synthetic ONNX graphs on the fly (clean CNN,
ReLU-gated backdoor, sigmoid-gated backdoor, SE block) and verifies
each. All in <5 s on CPU:

```bash
make smoke
# or:
pytest tests/test_smoke.py -v
```

If every test passes, the verifier code, admission test, IBP
propagation, and the basic detection logic are all exercised.

To run the full pytest suite (smoke + admission + envelope + ACPC +
MGRS + EIC + IBP, ~30 s on CPU), use `make test`.

### Browse the paper numbers

Every numeric claim in the paper traces to a row of `results/per_model.csv`
or a key of `results/aggregates.json`:

```bash
# Browse the headline numbers from Tables 4, 7, 8, 11 of the paper
python examples/02_browse_results.py

# Or directly: F1 by uncertified-handling protocol (Tab. 7)
python -c "
import json
print(json.load(open('results/aggregates.json'))['e2']['archproof_per_protocol'])"
```

See `results/README.md` for the full schema and `examples/` for runnable demos.

---

## 3. Full reproduction (per RQ)

The paper has three research questions:

| RQ      | What it asks                                                                                                                                    | Headline result                                                                           | Reproduce with                                                                                                                                       |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RQ1** | Can ArchProof return a sound finite `ε` on whole-model 6–7B-parameter LLM ONNX where prior sound verifiers do not reach the verification stage? | 5/5 backdoored LLMs CERTIFIED-POSITIVE; 0/10 prior-toolchain runs reach bound computation | `archproof.verify_model_phaseC(...)` on each LLM ONNX (per RQ1 commands below); baseline failures documented in `results/per_cell_whole_llm_eic.csv` |
| **RQ2** | Does the sound certificate yield strictly higher detection F1 on a 101-model labelled benchmark than heuristic detectors?                       | F1 = 1.000 vs 0.90 best heuristic                                                         | `archproof/verify.py --benchmark in-class-49+11 --tau-sys 1e-3 --tau-dorm 1e-3`                                                                      |
| **RQ3** | Do the three robustness theorems (ACPC / MGRS / EIC) hold empirically across ~12,000 evaluation cells?                                          | Every cell sound                                                                          | the per-experiment scripts under `archproof/` and `scripts/`                                                                                         |

To reproduce in full, you need the benchmark weights (~130 GB, see
[Section 7](#7-benchmark-weights)). After fetching weights:

### RQ1 — whole-LLM verification

```bash
# 1. Export 5 LLMs to ONNX (one-time, runs on CPU)
python scripts/export/export_clean_llm.py --model EleutherAI/gpt-j-6b
python scripts/export/export_clean_llm.py --model 01-ai/Yi-6B
# ... repeat for DeepSeek, Mistral, Qwen2

# 2. Inject backdoor gates (≈ 1 minute each)
python scripts/inject/inject_llm_backdoor.py --base EleutherAI/gpt-j-6b
# ... etc.

# 3. Verify each LLM (≈ 10 minutes each on a 176 GB host, CPU-only IBP)
python -c "
from archproof import verify_model_phaseC
for llm in ['gpt-j-6b','yi-6b','deepseek-7b','mistral-7b','qwen2-7b']:
    r = verify_model_phaseC(f'benchmark/7b_onnx/backdoored/{llm}.onnx',
                            tau_sys=1e-3, trigger_eta=0.05)
    print(llm, r.verdict_phaseC, r.epsilon_phaseC)
"
```

Expected: every LLM returns `add-DGP-CERTIFIED-POSITIVE` with finite ε.

### RQ2 — labelled-benchmark F1

```bash
# 22 in-class backdoors + 49 natural-clean + 30 synthetic-clean
python archproof/verify.py \
    --benchmark in-class-49+11 \
    --tau-sys 1e-3 --tau-dorm 1e-3 \
    --output results/per_model_my_run.csv
# Compare against committed results/per_cell_phaseC.csv
```

### RQ3 — robustness theorems

```bash
# ACPC (calibration poisoning) — ~2160 cells, ~30 minutes
python archproof/acpc.py --sweep chain-sensitivity

# MGRS (minimum gate-removal greedy optimality) — ~4800 cells, ~5 minutes
python archproof/mgrs.py --sweep removal-order

# EIC (exporter invariance) — 6 exporters × 5 LLMs, ~1 hour
python archproof/escalate.py --sweep eic-toolchain
```

Each writes a per-cell CSV under `results/`. Compare against the
committed `results/per_cell_*.csv` for byte-exact reproduction.

---

## 4. Tutorial: certify your own ONNX model

The simplest entry point is `archproof.verify_model(...)`, which takes
an ONNX file path and runs the full hierarchical verifier
(G1/G2/G3/G4/T10 admission + sound IBP + per-activation envelope sum):

```python
from archproof import verify_model, print_result

result = verify_model(
    "your_model.onnx",
    b_clean_ub=0.95,   # B_clean upper bound (0.95 for normalised images)
    tau_adm=0.1,       # Theorem 10 admission threshold
    n_probe=20,        # number of clean calibration samples
)

print_result(result)
# Key fields on `result`:
#   result.verdict               6-class diagnostic label
#                                (DORMANT / GDP-FREE / EPS-BOUNDED /
#                                 OUTPUT-PRESERVED / UNDECIDED / UNDECIDED-EXPORTER)
#   result.total_output_margin   the certificate ε
#   result.n_gdp_admitted        size of the admitted gate set |S_adm|
```

For the 3-class deployment-view verdict (CERTIFIED-POSITIVE /
CLASS-NEGATIVE / UNCERTIFIED) used by the paper's Phase C:

```python
from archproof import verify_model_phaseC

r = verify_model_phaseC(
    "your_model.onnx",
    tau_sys=1e-3,
    trigger_eta=0.05,
)
print(r.verdict_phaseC, r.epsilon_phaseC, r.n_admitted_phaseC)
```

For a runnable end-to-end example:

```bash
python examples/01_certify_a_model.py path/to/your_model.onnx
```

Run-time scales with the IBP forward (`O(|G|)` graph nodes). A
CIFAR-CNN finishes in <1 s; a 28 GB Mistral-7B ONNX takes ~10 min on a
176 GB host (CPU-only IBP).

---

## 5. Repository layout

```
.
├── archproof/        Verifier source — 36 modules
│   ├── verify_phaseC.py          high-level entry point (verify_model)
│   ├── interval_propagation.py   sound IBP for ONNX
│   ├── activation_epsilon.py     per-activation envelope (11 activations)
│   ├── chain_sensitivity.py      affine chain pair (A_post, B_post)
│   ├── llm_gate_rescue.py        LayerNorm/RMSNorm geometric rescue
│   ├── gate_admission.py         G1/G2/G3 candidate-gate admission
│   ├── acpc.py                   ACPC robustness (calibration poisoning)
│   ├── mgrs.py                   MGRS robustness (gate removal)
│   ├── escalate.py               EIC robustness (exporter drift)
│   └── ...
├── scripts/          Supporting utilities, grouped by purpose
│   ├── postprocess_acpc_rescue_aware.py  post-process whole-LLM ACPC CSV
│   ├── export/     HuggingFace → ONNX exporters (LLM + medium TF)
│   ├── inject/     add-DGP backdoor injectors + Mistral regen helper
│   ├── probe/      one-off ONNX-graph diagnostics
│   └── baseline/   prior-toolchain baseline runners
├── results/          Authoritative experiment data
│   ├── README.md                   schema documentation
│   ├── aggregates.json             per-experiment roll-ups
│   ├── per_cell_*.csv              per-cell raw data
│   ├── per_model.csv               per-model verdicts
│   └── per_model_*.csv             per-experiment per-model breakdowns
├── benchmark/        Model inventory (no weights)
│   ├── MODELS.md                   human-readable: 123 ONNX, sources, sizes
│   └── BENCHMARK_MANIFEST.json     same, machine-readable
├── tests/            pytest suite (smoke + admission + envelope + ACPC + MGRS + EIC + IBP)
├── examples/         runnable examples (certify a model, browse paper numbers)
├── docs/             API reference (api.md)
├── README.md / SETUP.md / REPRODUCE.md
├── pyproject.toml / environment.yml / Makefile
└── .gitignore        excludes ONNX weights, caches, LaTeX residue
```

---

## 6. Environment setup

```bash
conda create -n archproof python=3.9 -y
conda activate archproof

pip install onnx==1.16 onnxruntime==1.18 \
            torch==2.2 torchvision==0.17 \
            transformers==4.40 \
            numpy pandas pytest

# Required environment variable: where this repo lives
export ARCHPROOF_ROOT=$(pwd)

# Optional: only needed for the crown_escalate baseline experiment
export ABC_REPO=/path/to/alpha-beta-CROWN
```

`${HOME}` is auto-discovered for conda (`${HOME}/anaconda3` or
`${HOME}/miniconda3`).

The artifact runs on CPU end-to-end (verification and ONNX export). See
`SETUP.md` for the full RAM/disk breakdown by use case.

Tested on Linux 5.15 / Python 3.9 / PyTorch 2.x / ONNX 1.16 /
onnxruntime 1.18.

---

## 7. Benchmark weights

The benchmark consists of **123 ONNX models, ~130 GB total**. They are
**not redistributed** in this repository — instead, every model has a
canonical source URL listed in `benchmark/MODELS.md`. Reviewers can
either:

- **(a) Re-export from public checkpoints** using the export and
  inject scripts in `scripts/`. The scripts are deterministic and the
  resulting ONNX matches our committed result CSVs to bit-exact under
  the same opset.
- **(b) Request a download link** from the authors.

Quick fetch summary (full table in `benchmark/MODELS.md`):

| Category                                  | n   | Source                                                                   |
| ----------------------------------------- | --- | ------------------------------------------------------------------------ |
| 6–7B LLM ONNX (clean + backdoored)        | 10  | HuggingFace (GPT-J, Yi, DeepSeek, Mistral, Qwen2)                        |
| ImageNet-pretrained backbones × 3 gates   | 12  | torchvision (ResNet18/50, MobileNetV2, EfficientNet-B0)                  |
| Medium-TF encoders × 4 variants           | 28  | HuggingFace (BERT, DistilBERT, GPT-2, RoBERTa, DeBERTa, ALBERT, Electra) |
| CIFAR-CNN backdoors (Bober + handcrafted) | 25  | buildable from scratch via `scripts/inject/*.py`                         |
| Real-scan clean baseline                  | 8   | torchvision + HuggingFace                                                |
| Random-init torchvision backbones         | 14  | random init via `scripts/export/*.py --random-init`                      |
| Synthetic stress-coverage clean           | 45  | buildable via `archproof/handcrafted_gdp.py`                             |
| Adversarial G-probe constructions         | 3   | buildable via `archproof/g1_g4_adversaries.py`                           |
| Open-world streaming scan                 | 500 | streamed from HuggingFace `library:onnx` topic                           |

