# Reproducing the paper's tables

The artifact and the paper PDF are separate: **you read the PDF; the code
prints numbers.** Each of the **45 data tables** has a one-click script under
`reproduce/` that prints the reproduced values for that table — you then open
the PDF to the matching table and compare the values by eye. The **7
definitional/structural tables** are listed at the end (they are not
experimental data, so there is nothing to reproduce).

## Key results — what to check first

If you only have time for the headline claims, these tables carry them. The
full one-per-table index follows below.

| Claim                                                                                                                                                                            | Tables                                                                                                                  | Tier      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------- |
| **RQ1 — whole-model 6–7B LLMs.** ArchProof certifies dormant backdoors in five real 6–7B LLMs; a prior sound verifier (auto_LiRPA / α,β-CROWN) OOMs or fails on the same graphs. | 4 (headline ε), 5 (baseline fails), 15 (per-model detail), 14 (trigger drift)                                           | LLM       |
| **RQ2 — benchmark & taxonomy.** Sound 3-class verdicts on the 22-backdoor + clean benchmark; the 12-type add-DGP taxonomy; detection F1 vs baselines; false-positive panel.      | 3 (benchmark), 6 (verdict distribution), 7 (F1 vs baselines), 16–19 (taxonomy), 20 (protocols), 21 (ablation baselines) | CPU       |
| **RQ3 — anti-poisoning & removal.** detect → remove → verify (ε → 0). ACPC chain-sensitivity anti-poisoning; MGRS gate surgery.                                                  | 9, 23, 24 (ACPC); 10, 25, 26, 36 (MGRS)                                                                                 | CPU + LLM |
| **Soundness & robustness.** ULP-sound arithmetic; EIC invariance; adaptive graph obfuscation defeated; no false positives on real / open-world models.                           | 45 (sound), 11/27/28 (EIC), 31 (obfuscation), 47/49 (open-world)                                                        | CPU       |

The fastest independent check of all of the above at once is `make verify-quick`
(≈ 2 min; recomputes every table from its bundled record and compares to the
paper). To re-run the CPU tables' experiments from zero, see below.

**Case study (Appendix D.5.5).** `bash reproduce/potion_case_study.sh` (or `make
potion`) reproduces the ArchProof half of the `minishlab/potion-base-8M` live
example: download 58 MB, then a sound `add-DGP-CLASS-NEGATIVE` in ~20 s; `QUICK=1`
reads the archived verdict offline. The scanner's flag on this model is external
(see `docs/MODELS.md` §3e); only the ArchProof verdict is reproduced.

## How to run

```bash
make env && make install                 # one-time: pinned conda env + pip install -e .

bash reproduce/table07_f1-baseline.sh    # reproduce ONE table FROM SCRATCH (default): run the
                                         #   experiment, regenerate the record, verify vs the paper
QUICK=1 bash reproduce/table07_f1-baseline.sh   # skip the run; just read the bundled record

python verify/check_tables.py            # check all 45 records vs the paper (~2 min, no re-run)
bash reproduce/reproduce_cpu.sh          # re-run the 36 CPU tables from scratch (slow, ~2-3 h)
```

The 7 whole-model LLM tables (GPU tier) and 3 derived tables are **not** part of
`reproduce_cpu.sh`; they are checked from their bundled record by the
`check_tables.py` line above, and the LLM tier re-runs via
`scripts/download_llm.sh` → `scripts/export_llm.sh` → `RERUN=1 reproduce/tableNN.sh`.

**By default each of the 35 `T1` table scripts reproduces from scratch** — it
runs the experiment driver, regenerates the record, then verifies it against the
paper and prints the values for you to compare to the PDF by eye. This is the
real reproduction, but not instant: seconds for a light table, several minutes
for a heavy one (whole-ResNet-18 interval propagation, ULP-sound arithmetic, the
68-config EIC sweep). Prefix `QUICK=1` to skip the run and read the bundled
record instead.

The **7 `LLM`** tables need the LLM tier (a GPU + ~176 GB RAM to export the
253 GB ONNX — the export peaks above 146 GB, verification alone is ~95 GB) to
run from scratch, so they default to reading the bundled record —
`RERUN=1` re-runs them once you have that tier. The **3 `rec`** tables are
derived aggregates with no standalone driver (record only).

The `OK <n>/<n>` line each script prints is a convenience self-check against a
bundled copy of the paper's numbers (`verify/paper_tables.json`); the
authoritative check is you comparing the printed values to the PDF.

## Data tables (paper order) — one script each

Tier: **T1** = reproduces from scratch by default (CPU); **LLM** = needs the LLM
tier (reads the bundled record by default); **rec** = derived aggregate, record only.

| #   | Label                            | What it is                          | One-click script                                  | Tier |
| --- | -------------------------------- | ----------------------------------- | ------------------------------------------------- | ---- |
| 2   | `tab:eic-break`                  | EIC break cases (fail-closed)       | `reproduce/table02_eic-break.sh`                  | T1   |
| 3   | `tab:benchmark`                  | Benchmark / baseline overview       | `reproduce/table03_benchmark.sh`                  | T1   |
| 4   | `tab:llm-headline`               | 5 whole-model LLM verified epsilon  | `reproduce/table04_llm-headline.sh`               | LLM  |
| 5   | `tab:llm-baseline`               | auto_LiRPA baseline fails on 5 LLMs | `reproduce/table05_llm-baseline.sh`               | LLM  |
| 6   | `tab:verdict-dist`               | Out-of-class verdict distribution   | `reproduce/table06_verdict-dist.sh`               | T1   |
| 7   | `tab:f1-baseline`                | Detection F1 vs baselines           | `reproduce/table07_f1-baseline.sh`                | T1   |
| 8   | `tab:scale-coverage`             | Scale coverage (whole-model LLM)    | `reproduce/table08_scale-coverage.sh`             | rec  |
| 9   | `tab:acpc-chain`                 | ACPC chain sensitivity              | `reproduce/table09_acpc-chain.sh`                 | T1   |
| 10  | `tab:mgrs-greedy`                | MGRS greedy removal                 | `reproduce/table10_mgrs-greedy.sh`                | T1   |
| 11  | `tab:eic-summary`                | EIC invariance summary              | `reproduce/table11_eic-summary.sh`                | T1   |
| 13  | `tab:appx:mistral`               | Mistral-7B SwiGLU fragment          | `reproduce/table13_appx-mistral.sh`               | LLM  |
| 14  | `tab:appx:llm-drift`             | 5 LLM backdoor-vs-cleaned drift     | `reproduce/table14_appx-llm-drift.sh`             | LLM  |
| 15  | `tab:appx:llm-detail`            | 5 LLM per-model detail              | `reproduce/table15_appx-llm-detail.sh`            | LLM  |
| 16  | `tab:appx:backdoor-perm`         | Bober backdoor permutation          | `reproduce/table16_appx-backdoor-perm.sh`         | T1   |
| 17  | `tab:appx:bober-taxonomy`        | Bober 12 gate types                 | `reproduce/table17_appx-bober-taxonomy.sh`        | T1   |
| 18  | `tab:appx:bober-lambda`          | Bober lambda sensitivity            | `reproduce/table18_appx-bober-lambda.sh`          | T1   |
| 19  | `tab:appx:bober-handcrafted`     | H1-H3 handcrafted PGD (draw)        | `reproduce/table19_appx-bober-handcrafted.sh`     | T1   |
| 20  | `tab:appx:protocols`             | Baseline protocols                  | `reproduce/table20_appx-protocols.sh`             | T1   |
| 21  | `tab:appx:bl5bl6`                | Ablation baselines BL5/BL6          | `reproduce/table21_appx-bl5bl6.sh`                | T1   |
| 22  | `tab:appx:witness`               | G2 witness oracle                   | `reproduce/table22_appx-witness.sh`               | T1   |
| 23  | `tab:appx:acpc`                  | ACPC end-to-end                     | `reproduce/table23_appx-acpc.sh`                  | T1   |
| 24  | `tab:appx:acpc-multigate`        | ACPC multi-gate                     | `reproduce/table24_appx-acpc-multigate.sh`        | T1   |
| 25  | `tab:appx:mgrs-greedy`           | MGRS greedy (appendix)              | `reproduce/table25_appx-mgrs-greedy.sh`           | T1   |
| 26  | `tab:appx:mgrs`                  | MGRS sweep                          | `reproduce/table26_appx-mgrs.sh`                  | T1   |
| 27  | `tab:appx:eic-config`            | EIC per-config                      | `reproduce/table27_appx-eic-config.sh`            | T1   |
| 28  | `tab:appx:eic-timing`            | EIC timing (wall-clock; hardware-dependent, not value-checked) | `reproduce/table28_appx-eic-timing.sh` | T1   |
| 29  | `tab:appx:adaptive`              | Near-threshold adaptive             | `reproduce/table29_appx-adaptive.sh`              | rec  |
| 30  | `tab:appx:critical2`             | 400 near-threshold order statistics | `reproduce/table30_appx-critical2.sh`             | rec  |
| 31  | `tab:appx:adaptive-obf`          | 88-instance adaptive obfuscation    | `reproduce/table31_appx-adaptive-obf.sh`          | T1   |
| 32  | `tab:appx:tau-adm`               | tau_adm sweep (certifier off, draw) | `reproduce/table32_appx-tau-adm.sh`               | T1   |
| 33  | `tab:appx:cifar-e2e`             | CIFAR end-to-end                    | `reproduce/table33_appx-cifar-e2e.sh`             | T1   |
| 34  | `tab:appx:medium`                | 3 HF encoders                       | `reproduce/table34_appx-medium.sh`                | T1   |
| 35  | `tab:appx:whole-encoder`         | Whole-encoder verification          | `reproduce/table35_appx-whole-encoder.sh`         | LLM  |
| 36  | `tab:appx:production`            | Production 4 backbones x 3 gates    | `reproduce/table36_appx-production.sh`            | T1   |
| 37  | `tab:appx:tau-sys`               | tau_sys sweep                       | `reproduce/table37_appx-tau-sys.sh`               | T1   |
| 38  | `tab:appx:bclean-sweep`          | B_clean sweep                       | `reproduce/table38_appx-bclean-sweep.sh`          | T1   |
| 39  | `tab:appx:pretrained-acpc`       | Pretrained-CNN ACPC                 | `reproduce/table39_appx-pretrained-acpc.sh`       | T1   |
| 40  | `tab:appx:pretrained-mgrs`       | Pretrained-CNN MGRS                 | `reproduce/table40_appx-pretrained-mgrs.sh`       | T1   |
| 41  | `tab:appx:b-dormancy-regression` | B dormancy-fix regression           | `reproduce/table41_appx-b-dormancy-regression.sh` | LLM  |
| 42  | `tab:appx:cross-machine`         | Cross-machine reproduction          | `reproduce/table42_appx-cross-machine.sh`         | T1   |
| 43  | `tab:appx:compile-stage`         | Compile-stage adversarial           | `reproduce/table43_appx-compile-stage.sh`         | T1   |
| 45  | `tab:appx:sound`                 | Certified (ULP-sound) arithmetic    | `reproduce/table45_appx-sound.sh`                 | T1   |
| 46  | `tab:appx:torchvision`           | Torchvision backbone scope          | `reproduce/table46_appx-torchvision.sh`           | T1   |
| 47  | `tab:appx:real-scan`             | 8 large clean models                | `reproduce/table47_appx-real-scan.sh`             | T1   |
| 48  | `tab:appx:scanner-scope`         | Scanner scope                       | `reproduce/table48_appx-scanner-scope.sh`         | T1   |
| 49  | `tab:appx:openworld`             | Open-world 500 streaming            | `reproduce/table49_appx-openworld.sh`             | T1   |

## Definitional / structural tables — nothing to reproduce

Not experimental data, so no reproduction script. Verified by inspection (and,
for Tables 1 and 50, by a unit test under `make test`:
`tests/test_envelope.py`, `tests/test_out_of_class_listing.py`):

- **Table 1** (`tab:envelopes`) — closed-form activation envelopes (analytic formulas)
- **Table 12** (`tab:related-work`) — a comparison matrix of prior work
- **Table 44** (`tab:appx:gptj-opcount`) — the GPT-J operator census (a structural claim: no syntactic gate, |S_syn|=0; the absolute operator counts are export-dependent, so this is a structural check rather than a value match). Runnable via `archproof/run_gptj_opcount.py <gptj.onnx>`.
- **Table 50** (`tab:out-of-class`) — a listing of the out-of-class construction types
- **Table 51** (`tab:appx:tau-map`) — a threshold-per-experiment configuration map
- **Table 52** (`tab:appx:ops`) — the ONNX operators the verifier supports

## Re-running from scratch (`RERUN=1`)

Default runs recompute each table from the archived record. `RERUN=1` instead
re-runs the experiment driver first (it prints the exact command), then prints
the reproduced values. What it needs:

- **Models (bundled, with download-if-absent as a fallback).** Everything
  except the 253 GB LLM ONNX is bundled, so a run uses the **local** copy with
  no network. The public, downloadable pieces additionally follow a
  **download-if-absent, else use local** rule, so a missing file self-heals
  instead of breaking:
  - *torchvision backbone weights* are bundled under
    `models/torchvision_weights/hub/checkpoints/` and the relevant table scripts
    point `TORCH_HOME` there; a missing weight re-downloads from
    `download.pytorch.org` (repair all with `make fetch-weights`).
  - the *CIFAR-10 test set* is bundled under `data/` and the loader uses it if
    present (`utils/cifar10.py`); a missing copy re-downloads there.
  - *HuggingFace encoders* (`from_pretrained`, used by the medium/real-scan/
    open-world tables) download on first use — on a network-restricted host set
    `export HF_ENDPOINT=https://hf-mirror.com`.
    The **constructed** models that cannot be downloaded — the 22 seeded backdoor
    graphs and the 97-model unseeded clean panel — are bundled (`models/`); the
    backdoor graphs are staged into `/tmp` automatically by each detection table's
    `RERUN`. The 6–7B LLM ONNX (253 GB) are the LLM tier: regenerate per
    `docs/MODELS.md` (`scripts/download_llm.sh` → `scripts/export_llm.sh`).
- **Scanner baselines (Table 48 only).** The scanner-scope re-run contrasts
  ArchProof with two deployed model scanners, `modelscan==0.8.6` and
  `picklescan==1.0.4`; both ship with the pinned env (`environment.yml`). The
  backdoor ONNX they scan is bundled (`benchmark/exporter_test/`). On a bare
  env, `pip install modelscan==0.8.6 picklescan==1.0.4` first; if they are
  absent the script prints this hint and falls back to the archived values
  instead of failing.
- **Prior-verifier baseline (Table 5).** The negative baseline — `auto_LiRPA` +
  α,β-CROWN OOM/fail at 6–7B scale — is bundled runnable under
  `baselines/alpha-beta-CROWN/` (~3 MB; the repo's data/logs/model-zoo are
  stripped). It pins onnx 1.13.1 / onnx2pytorch 0.4.1, which conflict with
  `archproof_repro`, so it runs in its own env: `bash
  baselines/setup_baseline_env.sh` once, then `RERUN=1 bash
  reproduce/table05_llm-baseline.sh` (LLM tier). See `baselines/README.md`.
- **Open-world scan (Table 49), ~6–10 h, network.** The 500-model list is
  **pinned to the paper's order**, not re-discovered live: `run_phaseF.sh` uses
  the bundled `benchmark/phaseF_manifest.json` (rebuild it any time from the run
  record with `scripts/phaseF_manifest_from_record.py`, which reads
  `truth_source/per_model_phaseF.csv` — so the scanned set and order match the
  paper exactly, row for row). Each repo downloads from the real HF hub first
  and falls back to `hf-mirror.com`; set `HF_ENDPOINT` to force one first
  (behind the GFW, `HF_ENDPOINT=https://hf-mirror.com`). It streams
  download→verify→delete one repo at a time (peak disk ≤ 5 GB), resumable via
  the output CSV. To sanity-check the machinery without the 6–10 h run,
  `make smoke-phasef` confirms offline (no network) that the pinned manifest
  still matches the recorded 500-repo order and that the per-repo verify step
  runs on a bundled model.
- **Backdoor construction (detection-table RERUN).** Re-running a detection
  table (e.g. Tables 16–18) rebuilds the Bober models in PyTorch via the
  bundled `backdoored_models/` + `utils/` before verifying. The five
  shared-path/targeted variants embed one fixed CIFAR-10 test image; the
  CIFAR-10 test set is bundled under `data/` and used directly (or
  `ARCHPROOF_DATA`, or re-downloaded if absent). Verifying a full ResNet-18 under interval
  propagation is CPU-heavy, so these re-runs take minutes per model on one CPU
  box; the default (recompute-from-record) path needs none of this.
- **Portability.** The drivers resolve paths from their own location (or
  `ARCHPROOF_ROOT`), so `RERUN` works wherever you unpack the artifact.
- **Pinned environment.** For a *bit-exact* match, use the pinned env
  (`environment.yml`: torch 2.1.0). From torch 2.2 the ONNX exporter folds
  initializers differently, which shifts gate counts on re-export.
- **Tolerance on the random-panel tables.** Table 32 and the BL6 row of
  Table 21 draw an **unseeded** random-weight clean panel, so a fresh `RERUN`
  moves their false-positive/false-negative counts by ±1; the verdicts and
  trends reproduce, the exact draw does not (this is the CCS "within an allowed
  tolerance" case). The bundled records are one such draw. The ε^PGD / δ
  columns of Tables 16–19 are from a PGD probe whose original run was unseeded
  and probed a freshly built copy of each model; `archproof/run_e1_final.py`
  now seeds the probe input and loads the locked graph's weights before
  probing (a rerun is repeatable), but its values differ from the printed ones,
  so those columns are draw-checked: verdict and G1 are compared, the probe
  values are not.

## Notes

- **Documented discrepancy (Table 37, `tab:appx:tau-sys`)** — the paper
  prints 11 / 18 / 0 (positive / negative / uncertified) at every τ_sys; the
  release verifier on the locked graphs gives 11 / 17 / 1, because
  `H1_SignGated` is UNCERTIFIED (as Table 6 reports). The invariance across
  τ_sys reproduces; the H1 cell does not. The checker reports it as DISCREPANCY.
- **Draw-checked (Tables 19, 32)** — Table 32's cells are one draw of an
  unseeded random panel; Table 19's ε^PGD / δ cells (like the probe columns of
  Tables 16–18) are from the original unseeded probe run. The verdicts and
  trends reproduce, the exact draw does not.
- **Author-response (rebuttal) experiments** are ordinary data tables above:
  detect->remove->verify (epsilon->0) is Tables 10, 25, 26, 36; the
  no-false-positive evidence is Tables 47 and 49; ACPC anti-poisoning is
  Tables 9, 23, 24.

Raw per-run logs are not shipped, to keep the download lean. What you check
against the paper is the bundled records under `truth_source/` and `benchmark/`.
