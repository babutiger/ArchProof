# Models used in the artifact, and how to obtain them

Every experiment either builds its model from a script in this repository or
downloads a named public model and converts it. Nothing depends on a private or
otherwise unavailable model. This file lists, per experiment tier, exactly what
is used and how to regenerate it.

Three categories:

1. **Built here / shipped fixed.** The 22 CIFAR-scale backdoors are rebuilt
   deterministically by step 0 (seeded). The 97-model clean false-positive panel
   (2.3 GB) is shipped as fixed ONNX, because its generators are unseeded and
   use random weights — see 1b.
2. **Downloaded + converted (named public models).** torchvision backbones and
   HuggingFace encoders. A script downloads the public weights and exports ONNX.
3. **Downloaded + injected + converted (6-7B LLMs).** Large; not shipped. The
   HuggingFace repo id, the exact export/injection scripts, and the CPU-fp32
   conversion recipe are given so a reviewer reproduces the whole-model ONNX.

## Quick inventory — where every model actually is

| Models                                                     | Count                        | Where saved                                       | Size       | How to obtain                                                                                                                                 |
| ---------------------------------------------------------- | ---------------------------- | ------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Backdoor benchmark graphs (bober + handcrafted)            | 22                           | `models/backdoor_graphs/` + sha256                | 812 MB     | bundled (step 0 stages this copy into /tmp); also seed-rebuilt                                                                                |
| Clean false-positive panel                                 | 97                           | `models/clean_panel/` + `clean_panel_sha256.json` | 2.3 GB     | bundled (unseeded, so shipped fixed)                                                                                                          |
| torchvision backbone weights                               | 17                           | `models/torchvision_weights/` + sha256            | 2.1 GB     | bundled with sha256; also download-if-absent from download.pytorch.org, and `make fetch-weights` refreshes the mirror                         |
| HF encoder ONNX (BERT, DistilBERT)                         | 2                            | `models/hf_encoders/`                             | 672 MB     | bundled                                                                                                                                       |
| Encoder + LLM gate subgraphs                               | 7                            | `models/gate_subgraphs/`                          | KB         | bundled                                                                                                                                       |
| Production-scale CNN ONNX (4 backbones × 3 gates × stages) | 39                           | not bundled                                       | 2.1 GB     | RERUN rebuilds them seed-deterministically via `archproof/run_v3_production_scale.py`                                                         |
| HF encoder source models (bert/distilbert/gpt2)            | 3                            | HuggingFace cache                                 | 1.2 GB     | re-download via `HF_ENDPOINT=https://hf-mirror.com` (derived ONNX above are bundled)                                                          |
| **6-7B LLM whole-model ONNX**                              | 5 (clean+backdoored+cleaned) | regenerated into `benchmark/7b_onnx/`             | **253 GB** | **not bundled** — regenerate from public HF checkpoints via `scripts/export_llm_all.py`; the export peaks above 146 GB RAM (run with ~176 GB) |

The constructed models (backdoor graphs, the unseeded clean panel, gate
subgraphs, the encoder ONNX), the torchvision backbone weights, and the CIFAR-10
test set are all bundled with checksums. What is **not** in the upload is only
what is too large or trivially regenerated: the 253 GB 6-7B LLM ONNX and the 39
production-scale CNN ONNX (regenerated from public HF checkpoints / seeded
builders). Every bundled piece that is also public keeps a download-if-absent-else-local
fallback, so a run self-heals if a file is ever removed.

---

## 1. CIFAR-scale benchmark graphs (built here)

**What.** The 22 backdoored graphs (11 in-class + 11 out-of-class) and the
clean panels used by the detection / robustness experiments.

**How.** `bash artifact/scripts/00_build_benchmark_models.sh`

- in-class + handcrafted backdoors: `archproof/build_in_class_bober_onnx.py`,
  constructions from `backdoor-taxonomy/backdoored_models/` and
  `archproof/handcrafted_gdp.py`. Written to `/tmp/bober_onnx` and
  `/tmp/handcrafted_onnx` (scratch; step 0 regenerates them and fails loudly if
  any is missing — see REPRODUCE.md gotchas 1-3).

**Seeds (backdoors).** The builders draw their host weights from the global
RNG (the trigger geometry is fixed); `archproof/run_e1_final.py` calls
`torch.manual_seed(0)` immediately before each construction, which reproduces
21 of the 22 shipped graphs bit-for-bit (`H1_SignGated` was exported from an
RNG state that is not recoverable; its shipped graph is authoritative). Without
a seed these untrained graphs would differ per run and epsilon would move a few
percent (REPRODUCE.md gotcha 3). The sha256-locked files under
`models/backdoor_graphs/` are what every experiment reads; Step 0 provides them
from those locked copies, and the PGD probe loads the locked graph's
initializers into its PyTorch copy and checks the two agree under onnxruntime.

### 1b. The clean false-positive panel (97 models) — SHIPPED as fixed ONNX

**What.** 97 clean models that the detection / F1 experiments score against to
measure false positives. In this release they are centralized under
`models/clean_panel/*.onnx` (the manifest names them by their original
`benchmark/clean/` paths; `check_clean_panel.py` resolves both). This is where the paper's
"benign gated structures are not misflagged" story lives: CBAM, ECA, SkNet,
ResNeSt, Mish, GRU, GLU, GEGLU, SwiGLU, SE-blocks, highway/gated variants, plus
torchvision architectures (ResNet, VGG, DenseNet, MobileNet, EfficientNet,
Inception, GoogLeNet, ShuffleNet, MNASNet, RegNet, SqueezeNet, AlexNet). The
49-model "natural" F1 slice is selected by `origin == "natural"` in
`benchmark/BENCHMARK_MANIFEST.json`; the rest are synthetic/adversarial.

**Shipped as fixed ONNX, NOT regenerated — and here is why.** The clean-panel
generators (`archproof/collect_clean_models.py`,
`archproof/expand_clean_benchmark.py`, `archproof/gen_more_clean.py`,
`archproof/gen_hard_negatives.py`, `archproof/g1_g4_adversaries.py`) are
**unseeded**, and the torchvision architectures are built with `weights=None`
(random init). A rebuild would draw different weights; for the gated benign
structures (CBAM, ECA, gated_*) the gate median depends on the weight values, so
regeneration could move the panel's FP/TN counts and hence the F1 table.
Therefore the artifact ships the **exact** 97 ONNX and treats them as
authoritative. The generator scripts are listed per model in
`artifact/clean_panel_manifest.json` for provenance and optional
(non-bit-exact) regeneration.

**Size.** 2.3 GB total (the largest is `vgg16.onnx` at 528 MB — weight-carrying
even though random). Well within a Zenodo record.

**Verify presence/integrity.**

```bash
python artifact/verify/check_clean_panel.py          # 97/97 present + sizes
python artifact/verify/check_clean_panel.py --hash    # bytes match sha256 baseline
```

The sha256 baseline is `artifact/clean_panel_sha256.json`.

---

## 2. Named public models, downloaded and converted

### 2a. torchvision backbones — `tab:appx:torchvision`

**What.** 14 ImageNet-pretrained classifiers (ResNet18/50, WideResNet50,
MobileNetV2/V3-Small, EfficientNet-B0, VGG11/16, DenseNet121/169, AlexNet,
GoogLeNet, InceptionV3, SqueezeNet1.0).

**How.** `python artifact/archproof/run_e9_backbones.py`. It calls
`torchvision.models.<name>(weights="DEFAULT")`, exports opset-17 ONNX with
`keep_initializers_as_inputs=True`, and verifies. Weights are downloaded by
torchvision from `download.pytorch.org` on first use and cached under
`~/.cache/torch`.

**Bundled, with download-if-absent as fallback.** The artifact ships these
weights under `artifact/models/torchvision_weights/` (with sha256, ~2.3 GB), so
no network is needed. As a fallback, torchvision downloads them from
`download.pytorch.org` into `~/.cache/torch` on first use if the local copy is
absent, and a run reuses that cache; `make fetch-weights`
(`scripts/fetch_torchvision_weights.sh`) repopulates or refreshes the mirror,
skipping files already present.

**CPU note.** The verifier is CPU-only; export runs on CPU. No GPU needed.

**Drift.** torchvision weight URLs are versioned, but a future torchvision
release could change a default checkpoint. The record shipped with the
artifact was produced with the torchvision in `environment.yml`; pin that
version to reproduce node counts exactly.

### 2b. HuggingFace encoders — two panels, different model sets

Both download from HuggingFace Hub on first use and require `transformers`
(pinned in `environment.yml`); the verifier and export run on CPU. Each id is a
public repo at `https://huggingface.co/<id>` (e.g.
<https://huggingface.co/bert-base-uncased>); fetch them the same way as the
LLMs in §3a (`from_pretrained`, or `huggingface-cli download <id>`, with
`HF_ENDPOINT=https://hf-mirror.com` on a restricted network). The medium panel's
derived encoder ONNX are also bundled in `models/hf_encoders/`, so Table 34
verifies without any download.

**Medium panel — `tab:appx:medium`** (run via
`python artifact/archproof/run_e5a_medium.py`). Exactly three models, the
ids as they appear in the script's `configs`:
`bert-base-uncased`, `distilbert-base-uncased`, `gpt2`.

**Whole-encoder 28-case panel — `tab:appx:whole-encoder`**
(`scripts/export_clean_transformers.py` exports the encoders; a backdoor head is
injected for the backdoored variants). Seven encoders, ids from the script's
list: `bert-base-uncased`, `distilbert-base-uncased`, `roberta-base`,
`albert-base-v2`, `xlnet-base-cased`, `google/electra-base-discriminator`,
`microsoft/deberta-base`. 7 encoders x (clean + 3 backdoor heads) = 28 cases.

**Note.** One shipped medium record was written on a host without
`transformers` installed; it is a stale error stub — re-run to regenerate
(progress doc item 2).

---

## 3. Whole-model 6-7B LLMs — GPU + ~176 GB RAM to export (verification alone ~95 GB)

Three steps, done once, then verify each LLM table the normal way:

```bash
bash scripts/download_llm.sh                      # 1. fetch the 5 base checkpoints (~90 GB)
bash scripts/export_llm.sh                        # 2. export clean + backdoored ONNX (~253 GB, CPU-fp32)
RERUN=1 bash reproduce/table04_llm-headline.sh    # 3. verify each LLM table, one by one
```

Which table needs what (all read `benchmark/7b_onnx/{clean_onnx,backdoored_onnx}/`):

- **Tables 4, 5, 15** (headline / baseline / detail) — the clean + backdoored
  whole-model ONNX that `export_llm.sh` produces. Step 3 works after step 2.
- **Table 13** (Mistral SwiGLU fragment) — self-contained: its driver injects
  and exports a small MLP fragment to `/tmp`, no whole-model export needed.
- **Table 14** (backdoor-vs-cleaned drift) additionally needs the *MGRS-cleaned*
  variant (`<short>-mgrs-cleaned.onnx`), and **Table 41** (dormancy regression)
  re-exports at a specific threshold — both are extra steps beyond `export_llm.sh`
  (produced by the MGRS / dormancy drivers). For those two, run the full LLM
  pipeline `bash scripts/30_llm_scale.sh` rather than a single table.
- **Table 35** verifies HuggingFace *encoders*, not the 5 LLMs (auto-downloaded,
  no step 1/2 — see §2b).

Details below.

**Not shipped** (each export is 23-31 GB). A reviewer reproduces them from the
public base weights with the scripts below. This whole-model tier needs a GPU +
~95 GB RAM (verification peaks at ~3x the export size, 71-94 GB RAM measured).

### 3a. Base weights — download from HuggingFace

Each model is a public HuggingFace repo (open weights; Mistral requires a
one-time click-through of its license on the Hub). `from_pretrained` downloads
it automatically on first use, or fetch it explicitly with the commands below.

| short       | HuggingFace repo id                  | params | download URL                                                |
| ----------- | ------------------------------------ | ------ | ----------------------------------------------------------- |
| gpt-j-6b    | `EleutherAI/gpt-j-6b`                | 6B     | <https://huggingface.co/EleutherAI/gpt-j-6b>                |
| yi-6b       | `01-ai/Yi-6B`                        | 6B     | <https://huggingface.co/01-ai/Yi-6B>                        |
| deepseek-7b | `deepseek-ai/deepseek-llm-7b-base`   | 7B     | <https://huggingface.co/deepseek-ai/deepseek-llm-7b-base>   |
| mistral-7b  | `mistralai/Mistral-7B-Instruct-v0.3` | 7B     | <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3> |
| qwen2-7b    | `Qwen/Qwen2-7B-Instruct`             | 7B     | <https://huggingface.co/Qwen/Qwen2-7B-Instruct>             |

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login                       # once, for Mistral's gated license
export HF_ENDPOINT=https://hf-mirror.com     # only if huggingface.co is unreachable
for id in EleutherAI/gpt-j-6b 01-ai/Yi-6B deepseek-ai/deepseek-llm-7b-base \
          mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2-7B-Instruct; do
  huggingface-cli download "$id"
done
```

(Source of truth: `LLM_NAMES` in `scripts/inject_llm_backdoor.py`.)
`_resolve_source` also checks a local `models/<short>/` copy before the Hub, so
you can drop pre-downloaded weights there instead.

### 3b. Export to CPU-fp32 ONNX (clean + backdoored) — one command

`export_llm_all.py` exports BOTH the clean and the backdoored whole-model ONNX
for all 5 LLMs (opset 17, fp32, CPU) straight into the layout the verifier
reads — no second step:

```bash
python scripts/export_llm_all.py     # -> benchmark/7b_onnx/{clean_onnx,backdoored_onnx}/
                                     #    EXPORT_MODE=clean or =backdoored to do just one pass
```

It loads each base model from `models/<short>/` or the HF cache (§3a), adds the
dormant add-DGP gate for the backdoored pass (`BackdooredModel`: a strict-dormant
`ReLU(W_g h - 10) * W_p h` head through the shared `lm_head`), and skips any
export already on disk. Layout produced (exactly what `repro_llm_verify_existing.py`
reads):

```
benchmark/7b_onnx/clean_onnx/<short>/<short>.onnx
benchmark/7b_onnx/backdoored_onnx/<short>-backdoored/<short>-backdoored.onnx
```

Then reproduce an LLM table from scratch, e.g. `RERUN=1 bash
reproduce/table04_llm-headline.sh`.

Conversion recipe (why CPU-fp32):

- `torch_dtype=torch.float32`, `device="cpu"`: a 24 GB GPU OOMs tracing a 7B
  model, and the verifier's interval arithmetic is defined on float32. The
  trace and `torch.onnx.export(..., opset_version=17)` run on CPU with
  `low_cpu_mem_usage=True`.
- external data: the exporter writes HF-style external-data shards beside the
  `.onnx`; keep them together. Export size 23-31 GB per model.
- the injected gate is the dormant add-DGP branch
  `sigma(W_g h) * W_p h` at the last hidden state, lifted back to vocab through
  the shared embedding (`BackdoorCausalLM` in `scripts/inject_llm_backdoor.py`);
  three gate types `sep_tar` / `sha_un` / `int_un`.

### 3c. The verification environment (READ THIS — three required env vars)

Whole-model LLM verification depends on three environment variables. **If any
is missing, the verifier fails closed to `epsilon = 0` / `UNCERTIFIED` — that
is expected fail-closed behaviour, not a wrong paper number.** The scripts in
§3d set them for you; set them yourself if you invoke the verifier directly.

```bash
export ARCHPROOF_LARGE_MODEL_GB=256   # load the FULL external-data weights (24-29 GB)
                                      #   and run real IBP, instead of the >22 GB
                                      #   graph-only fail-closed that returns eps=0
export ARCHPROOF_LLM_RESCUE=1         # apply the geometric LayerNorm gate-bound
                                      #   rescue; without it the LN-gated bound stays
                                      #   vacuous -> eps=0 / UNCERTIFIED
export ARCHPROOF_PROBE_MAX_GB=40      # dormancy-probe memory cap used in the paper run
```

(These match `verify_phaseC.py` and the paper's truth-source runs; the exact
lines are documented at the top of `scripts/repro_llm_verify_existing.py`.)

### 3d. Verify — two paths

**(recommended) Re-verify the paper's numbers against the existing exports**,
one LLM per process (memory-safe; the env trio is baked into the driver):

```bash
python scripts/repro_llm_verify_existing.py --only gpt-j-6b --out /tmp/gptj.csv
# repeat --only yi-6b / deepseek-7b / mistral-7b / qwen2-7b
```

**(from scratch) Re-run the whole LLM pipeline** (export → inject → verify);
each sub-script sets the env trio itself:

```bash
bash artifact/scripts/30_llm_scale.sh      # whole-model pipeline (GPU + ~95 GB RAM)
```

Then check the regenerated records against the paper tables:

```bash
python artifact/verify/check_tables.py --only tab:llm-headline,tab:appx:llm-detail,tab:appx:llm-drift
```

### 3e. The potion case study (external, cited not benchmarked)

`minishlab/potion-base-8M` (HuggingFace) is verified as a real deployed model in
Appendix D.5.5. The case is qualitative (its Guardian flag vs ArchProof's
class-negative), not part of the 22-model benchmark.

Reproduce the ArchProof half directly, without the 500-model Phase F sweep:
`bash reproduce/potion_case_study.sh` downloads `onnx/model.onnx` (~58 MB) and
runs the exact `verify_model_phaseC` path, expecting `add-DGP-CLASS-NEGATIVE`
(`n_admitted=0`; the graph contains no `Mul`). `QUICK=1 bash
reproduce/potion_case_study.sh` skips the download and reads the recorded
verdict from `truth_source/per_model_phaseF.csv`. The Guardian flag itself is
external (the discussion linked in D.5.5); only the ArchProof verdict is
reproduced here.

---

## 4. Open-world scan (500 real HuggingFace models) — streamed, not stored

The phaseF open-world sweep (`tab:appx:openworld`, the 500/478/96% numbers) does
NOT ship or retain 500 models. It is a **streaming** scan: `phaseF_discover_models.py`
lists 500 real HuggingFace ONNX repos, then `phaseF_streaming_scan.py` downloads
each, verifies it, and deletes it before the next — so disk never holds the whole
set (332.7 GB streamed total). Reproduce with `scripts/run_phaseF.sh`. The per-model
verdicts are in `truth_source/per_model_phaseF.csv` (checker-verified). The 500
models are live HuggingFace snapshots, discovered by the script at run time; they
are not part of the fixed shipped model set.

## Summary for the AEC

- **Small graphs**: shipped AND reproducible by `00_build_benchmark_models.sh`.
- **Named public models** (torchvision, HF encoders): downloaded by a one-line
  command; version pins in `environment.yml`.
- **6-7B LLMs**: not shipped (size); public repo ids + the exact
  download/inject/convert scripts above regenerate every whole-model ONNX.
  Reviewers without a ~95 GB-RAM host can instead read the archived per-model
  bundled per-model records (`truth_source/per_model_phaseE_*.csv`,
  `per_cell_whole_llm_*.csv`), which the checker rebuilds the LLM tables from
  without re-running anything.
