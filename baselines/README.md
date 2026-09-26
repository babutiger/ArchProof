# Baseline verifiers (prior work, for comparison)

The paper's baseline is *negative*: the prior sound-verification toolchain
(`auto_LiRPA` + α,β-CROWN) runs out of memory on the 6–7B backdoored LLMs before
it can return a verdict. `reproduce/table05_llm-baseline.sh` reproduces this: it
feeds each backdoored LLM ONNX to `auto_LiRPA` + α,β-CROWN and records the
outcome and peak RSS (`scripts/baseline_oom_attempt.py`); the per-model results
are in Table 5 and `truth_source/per_model_baseline_oom.csv`.

## What's here

`alpha-beta-CROWN/` is a trimmed, runnable copy of
[α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) — the
`auto_LiRPA` library and the `complete_verifier` code only (the repo's
experiment outputs, logs, datasets, model zoo, and unit tests are removed, so
this is ~3 MB instead of ~1.4 GB). `abcrown_requirements.txt` is its pinned
dependency list.

## Setting up the baseline environment

α,β-CROWN pins `onnx==1.13.1` / `onnx2pytorch==0.4.1`, which conflict with the
verifier's own env (`archproof_repro`, onnx 1.16), so the baseline runs in its
**own** environment:

```bash
bash baselines/setup_baseline_env.sh          # creates the `alpha-beta-crown` conda env
```

Then run the baseline table (needs the LLM tier — a GPU + the 253 GB LLM ONNX,
since the point is that the baseline OOMs at 6–7B scale):

```bash
RERUN=1 bash reproduce/table05_llm-baseline.sh
```

`baseline_oom_attempt.py` finds α,β-CROWN via `ALPHA_BETA_CROWN_PATH` (default:
this bundled copy) and `run_baseline_oom.sh` activates the
`alpha-beta-crown` env (override with `ARCHPROOF_CONDA_ENV`). The recorded
baseline outcomes the paper reports are in
`truth_source/per_model_baseline_oom.csv` and
`benchmark/baselines_natural49_complete.json`.
