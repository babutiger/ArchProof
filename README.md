# ArchProof — CCS 2026 Artifact

**Paper:** *ArchProof: A Sound Output-Contribution Certificate for Dormant-Gate
Path Backdoors in ONNX Models* (CCS 2026).

**Archived artifact (DOI):** https://doi.org/10.5281/zenodo.22722330 (Zenodo).

**GitHub copy:** https://github.com/babutiger/ArchProof holds the code, scripts,
tests, docs and all result records, but not the model weights and ONNX exports
under `models/` and `benchmark/exporter_test/` (about 6.7 GB), the CIFAR-10
batches under `data/`, or the `ccs2026-ae/` folder. The Zenodo archive above is the complete bundle;
`make fetch-weights` and the export scripts regenerate the models, and CIFAR-10
re-downloads on first use.

**TL;DR** — one machine, CPU only:

```bash
make install        # conda env + pip install -e .              (~5 min, once)
make verify-quick   # check all 45 data-table records vs the paper (~2 min)
make test           # unit tests prove the verifier is sound      (~30 s)
```

`make verify` (no `-quick`) instead re-runs every runnable table **from scratch**
and checks the fresh numbers against the paper — the real reproduction, slow
(~2-3 h). Full guide: **`docs/REPRODUCE.md`**.

ArchProof takes an ONNX model and returns a *sound* 3-class verdict about
**add-DGP** (additive dormant-gated-payload) architectural backdoors:

- `add-DGP-CERTIFIED-POSITIVE` — a dormant additive gate provably drives the
  output (certificate ε > τ_sys): a detected backdoor.
- `add-DGP-CLASS-NEGATIVE` — no admitted add-DGP gate: clean w.r.t. the class.
- `UNCERTIFIED` — the verifier fails **closed** (a bound is vacuous, or the
  graph is too large for CPU interval propagation). It never silently declares
  a model clean.

This artifact ships (a) the verifier as an importable, CPU-only Python package;
(b) the models, run records, and a checker that **recomputes every table in the
paper from those records**; and (c) scripts to re-run the experiments from
scratch. Nothing in the checking path reads a stored answer — each experiment
writes its own record and the checker rebuilds the table cells from it, so a
number no experiment produced cannot pass.

**Artifact-evaluation badges.** *Functional* — this README + `docs/`; the code,
models, data, and scripts are all in this one bundle; `make verify` / `make
test` run out of the box. *Reproduced* — `make verify` re-runs the **45 data
tables from scratch** and checks the freshly-computed numbers against the paper
(43 recomputed to the printed value + 2 draw-checked); the 7 whole-model LLM
tables re-run only with the 253 GB tier (otherwise verified from their bundled
record) and 3 derived tables have no standalone driver. The paper's other 7
tables are definitional/structural (no experimental data) and are verified by
inspection / unit test.

---

## 1. Quick start

Everything here runs on **one machine, CPU only** — no GPU, no network, no
large models.

```bash
make install      # pinned conda env `archproof_repro` + pip install -e .        (~5 min, once)
make verify       # re-run the 36 runnable tables FROM SCRATCH, check vs paper   (slow, ~2-3 h, CPU)
make verify-quick # instead: just check the bundled records vs the paper         (~2 min)
make test         # pytest suite (synthetic ONNX)                                (~30 s)
```

`make verify` is the real reproduction: it re-runs each runnable experiment from
zero and checks the freshly-computed numbers against the paper (nothing is read
back from a stored answer). It prints one line per table with `OK <n>/<n>`,
ending with:

```
data tables reproduced: 45/45  (43 recomputed to the printed value, 2 draw-checked)
definitional/structural tables: 7 (no experimental data; verified by inspection / unit test)
```

---

## 2. What you need — one machine

- **The fast overview (what the badges rest on).** `make verify-quick`
  (= `python verify/check_tables.py`) checks all 45 tables' bundled records
  against the paper in ~2 min on **any Linux box, ~16 GB RAM, CPU only** — the
  checker reads only `truth_source/` and `benchmark/*.json` and needs nothing
  beyond numpy (no GPU, no torch, no network).

- **Reproducing from scratch.** `make verify`
  (= `bash reproduce/reproduce_cpu.sh`) re-runs the **36 CPU tables** from zero,
  regenerates each record, and checks the freshly-computed numbers against the
  paper (slow, ~2-3 h). A single table: `bash reproduce/tableNN.sh` — a fast one
  finishes in seconds, a heavy one (whole-ResNet-18 interval propagation,
  ULP-sound arithmetic, the 68-config EIC sweep) takes minutes. Two tiers are
  **not** part of this CPU run: the **7 whole-model 6–7B LLM tables**
  (4, 5, 13, 14, 15, 35, 41) need a 24 GB GPU + **~176 GB RAM** (the one-time
  ONNX export peaks above 146 GB; verification alone is ~95 GB) + 253 GB disk
  (download → export → `RERUN=1`; see `docs/MODELS.md`), and **3 derived
  tables** are aggregate counts with no standalone driver.

---

## 3. Reproducing the paper's tables

The artifact and the paper PDF are separate: **you read the PDF, the code prints
the numbers.** Each of the 36 runnable data tables has a one-click script under
`reproduce/` that, by default, **reproduces the table from scratch** — it runs
the experiment, regenerates the record, then checks it against the paper and
prints the values for you to compare to the PDF by eye.

```bash
bash reproduce/table16_appx-backdoor-perm.sh   # run this table's experiment from scratch, then verify
QUICK=1 bash reproduce/table16_appx-backdoor-perm.sh   # skip the run; just read the bundled record (seconds)
```

Running from scratch is the real reproduction, but it is not instant: a fast
table finishes in seconds, while a heavy one (whole-ResNet-18 interval
propagation, ULP-sound arithmetic, the 68-config EIC sweep) takes several
minutes. Two tiers do **not** run from scratch on a normal machine and default
to reading their bundled record: the **7 whole-model LLM tables** (they need a
GPU + ~176 GB RAM for the one-time ONNX export, which peaks above 146 GB —
verification alone is ~95 GB — plus 253 GB disk; `RERUN=1` once you have that tier)
and **3 derived tables** (aggregate counts with no standalone driver).

```bash
make verify-quick                        # check all 45 records vs the paper (~2 min, no re-run)
bash reproduce/reproduce_cpu.sh          # re-run the 36 CPU tables from scratch (slow, ~2-3 h)
```

The 7 LLM tables re-run only on the GPU tier (`bash scripts/download_llm.sh` →
`bash scripts/export_llm.sh` → `RERUN=1 bash reproduce/tableNN.sh`); the 3
derived tables have no standalone driver. Both are checked from their bundled
record by `make verify-quick`.

The **full table-by-table index** — every table, its one-click script, and its
tier — is in **`docs/REPRODUCE.md`**.

---

## 4. Certify your own model

```bash
make example        # certifies one bundled clean + one bundled backdoored model
```

or from Python on any ONNX file:

```python
from archproof import verify_model_phaseC
r = verify_model_phaseC("model.onnx")
print(r.verdict_phaseC, r.epsilon_phaseC)   # add-DGP-CLASS-NEGATIVE / -CERTIFIED-POSITIVE, epsilon
```

---

## 5. Install (detail)

`make install` creates the conda env **`archproof_repro`** (Python 3.10,
torch 2.1.0, onnx 1.16.0, onnxruntime 1.18.0, numpy 1.26.4) and installs
`archproof` in editable mode. Manual:

```bash
conda env create -f environment.yml
conda activate archproof_repro
pip install -e .
python -c "from archproof import verify_model, verify_model_phaseC; print('OK')"
```

Requirements: Linux, ~16 GB RAM, ~7 GB free disk (rebuilding the LLM exports
additionally needs a GPU, ~176 GB RAM, and 253 GB of disk).

### Without conda

The record-reproducing checker needs only pip packages (onnx, onnxruntime,
numpy, scipy, pandas — no conda, no torch, no GPU), so if conda is inconvenient
skip it entirely:

```bash
python3 -m venv venv && . venv/bin/activate
pip install -e .                          # just the checker's dependencies
python verify/check_tables.py             # 45/45 data tables (record check)
```

The pinned conda env (`environment.yml`, `torch==2.1`) is needed only to
*re-run* experiments from scratch (`RERUN=1`) or `make test`, both of which
build ONNX and so depend on the exporter version.

---

## 6. Layout

```
artifact/
  archproof/         the verifier + experiment drivers (importable package)
  backdoored_models/ the add-DGP backdoor constructors used by the benchmark
  utils/             helpers backdoored_models imports (CIFAR-10 loader)
  scripts/           experiment drivers + shared build step
                     (00_build_benchmark_models.sh)
  reproduce/         one one-click script per paper table + reproduce_cpu.sh
  baselines/         runnable alpha-beta-CROWN (auto_LiRPA) for the Table 5
                     prior-verifier comparison + its env setup
  verify/            the checker (check_tables.py) + the bundled paper numbers
  tests/             pytest suite (synthetic ONNX, CPU)
  models/            all non-LLM models, with sha256 (~6 GB; torchvision
                     weights re-download if absent, `make fetch-weights`)
  data/              CIFAR-10 test set (used locally; re-downloads if absent)
  benchmark/         the result JSONs the checker reads
  truth_source/      the result CSV/JSON the checker reads
  docs/              REPRODUCE.md (table-by-table) + MODELS.md (model provenance)
  ccs2026-ae/        the 2-page CCS Artifact Appendix (PDF + LaTeX source)
  README.md  Makefile  environment.yml  pyproject.toml  LICENSE
```

---

## 7. Troubleshooting

- **`ImportError: archproof`** — run `make install`, or prefix with
  `PYTHONPATH=.`.
- **`make verify` reports RECOMPUTE_ERROR** — a data file is missing; confirm
  `truth_source/` and `benchmark/*.json` are present in this bundle.
- **Torch/ONNX version mismatch** — the checker needs only numpy; the *export*
  and *test* paths pin `torch==2.1.0` (`environment.yml`). From torch 2.2 the
  ONNX exporter folds initializers differently, which changes gate counts on
  re-export; use the pinned env.
- **LLM re-run OOM** — whole-model 6–7B verification needs ~95 GB RAM; this is
  expected. Reproduce those tables from the bundled records instead.

The record-reproducing path (the verifier, `make verify`, and `make test`)
uses only bundle-relative paths and runs anywhere. The experiment
drivers resolve paths from the artifact root (or `ARCHPROOF_ROOT`), so `RERUN=1`
also works wherever you unpack the bundle. Raw per-run logs are not shipped, to
keep the download lean; the numbers you check come from the bundled records
under `truth_source/` and `benchmark/`.

---

## 8. Known limitations

- **Platform.** The record-checking path (`make verify-quick`, `make test`) runs
  on any Linux box with numpy. The from-scratch re-runs are exercised on Linux
  x86-64 with the pinned `environment.yml`; another OS or CPU may differ in the
  last digits of floating-point sums — the verdicts and the ULP-sound bounds
  still hold, but a bit-exact match is only guaranteed on the pinned stack.
- **Pinned exporter.** A bit-exact re-export needs `torch==2.1.0`. From torch 2.2
  the ONNX exporter folds identical weight tensors behind Identity nodes, which
  shifts gate counts on re-export.
- **LLM tier.** The 7 whole-model 6–7B LLM tables and the prior-verifier baseline
  re-run only with a GPU + ~176 GB RAM + 253 GB disk (`docs/MODELS.md`,
  `baselines/`); without that tier they are verified from their bundled records.
  `make smoke-llm` runs the *same* export → inject-gate → verify pipeline on a
  ~2 MB toy GPT-2 in seconds, so you can confirm the pipeline is correct — only
  the model scale, not the method, needs the big host.
- **Unseeded panels.** Tables 19, 32, and the BL6 row of Table 21 draw an
  unseeded random clean panel, so a fresh run moves their false-positive/negative
  counts by ±1; the verdicts and the trends reproduce, the exact draw does not
  (the "within an allowed tolerance" case — each such table's script says so when
  it runs).
- **Self-check vs the PDF.** The bundled `OK n/n` self-check compares each
  recomputed value to a frozen snapshot of the paper's numbers; the authoritative
  check is you comparing the printed values to the PDF (the scripts print them).

## 9. License

See `LICENSE` (MIT). 
