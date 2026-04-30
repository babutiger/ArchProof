# Setup

Detailed installation and environment guide for ArchProof.

## Hardware requirements

| Use case                                            | RAM        | Disk       |
| --------------------------------------------------- | ---------- | ---------- |
| Smoke test + examples                               | 4 GB       | 200 MB     |
| Run full pytest suite                               | 8 GB       | 500 MB     |
| Verify CIFAR-CNN backdoors (22 models)              | 8 GB       | 200 MB     |
| Verify ImageNet-pretrained backbones (12 models)    | 16 GB      | 5 GB       |
| Verify medium transformer encoders (28 models)      | 32 GB      | 5 GB       |
| Verify whole-model 6–7B-parameter LLM ONNX (5 LLMs) | **176 GB** | **130 GB** |

The artifact runs on CPU end-to-end (verification and ONNX export).

## Recommended setup: conda

```bash
# 1. Clone the repo (anonymous URL during review)
git clone <anonymous-repo-url>
cd archproof

# 2. Create the conda environment
conda env create -f environment.yml
conda activate archproof

# 3. Install the package in editable mode
pip install -e .

# 4. Verify
make smoke
```

This pulls Python 3.9, ONNX 1.16, ONNX Runtime 1.18, NumPy/SciPy/pandas,
plus PyTorch 2.x and HuggingFace transformers (the latter only needed by
`scripts/export/*.py` and `scripts/inject/*.py`).

## Alternative: venv + pip

If conda is unavailable:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"
```

`pip install -e ".[all]"` pulls every optional extra (`export`,
`sound-arithmetic`, `witness-oracle`, `test`).

## Smoke test (no benchmark weights required)

```bash
make smoke
```

This runs `tests/test_smoke.py`, which builds 4 synthetic ONNX graphs
on the fly (clean CNN, ReLU-gated backdoor, sigmoid-gated backdoor,
SE block) and verifies each. All in <5 s on CPU.

## Full test suite

```bash
make test
```

Runs the entire `tests/` directory: smoke + admission + envelope +
ACPC + MGRS + EIC + IBP. ~30 s on CPU. None of the tests need
benchmark weights.

## Browsing the paper numbers (no compute needed)

```bash
make browse
# or:
python examples/02_browse_results.py
```

Loads `results/aggregates.json` and `results/per_cell_*.csv` and prints
the headline numbers from Tables 4, 7, 8, and 11 of the paper.

## LLM ONNX re-export

Re-exporting the 5 whole-model LLMs from HuggingFace runs on the same
176 GB host used for verification. Skip this step if you fetch the 5
pre-exported ONNX files from the authors instead.

## Optional extras

| Extra              | Install                                | Purpose                                                                 |
| ------------------ | -------------------------------------- | ----------------------------------------------------------------------- |
| `sound-arithmetic` | `pip install -e ".[sound-arithmetic]"` | mpfr/gmp arbitrary-precision fallback for the certified-arithmetic mode |
| `witness-oracle`   | `pip install -e ".[witness-oracle]"`   | torchattacks PGD oracle for the optional G2 upgrade                     |
| `export`           | `pip install -e ".[export]"`           | torch + transformers for `scripts/export/*.py`                          |
| `test`             | `pip install -e ".[test]"`             | pytest + pytest-cov                                                     |

## Environment variables

| Variable         | Required? | Purpose                                                                                           |
| ---------------- | --------- | ------------------------------------------------------------------------------------------------- |
| `ARCHPROOF_ROOT` | optional  | If set, overrides path discovery for legacy scripts. With `pip install -e .` you don't need this. |
| `ABC_REPO`       | optional  | Path to a local α,β-CROWN clone, only needed for the `crown_escalate` baseline experiment.        |

## Troubleshooting

**"Module not found: archproof"** — run `pip install -e .` from the
repo root.

**"onnxruntime can't load model with Trilu op"** — bump onnxruntime to
1.18+ (`pip install -U onnxruntime`).

**Mistral / Qwen2 / Yi tokenizer errors** — `pip install -U transformers>=4.40`.

**Out-of-memory on whole-model LLM verification** — the IBP propagation
peaks at ~88 GB RSS for a 28 GB Mistral export. You need a 176 GB host
or larger.
