# `results/` — Authoritative experiment data

Every numeric claim in the paper traces back to a file in this directory.
The CSVs are the artifact's source of truth: re-running an experiment
writes a new `per_cell_*` / `per_model_*` CSV that you can `diff`
against the committed copy.

## Files

- **`per_model.csv`** — one row per ONNX model, with verdict, ε,
  admission counts, model metadata.
- **`per_cell.csv`** — one row per evaluation cell (sweep cells across
  ACPC, chain-sensitivity, MGRS, EIC, adaptive admission,
  order-statistic robustness experiments).
- **`per_cell_*.csv`** — per-experiment per-cell breakdowns
  (e.g., `per_cell_cifar_acpc.csv`, `per_cell_whole_llm_eic.csv`).
- **`per_model_*.csv`** — per-experiment per-model breakdowns.
- **`aggregates.json`** — top-level roll-ups (F1 by protocol, MGRS
  totals, ACPC LLM totals, G-condition necessity, etc.).

## `per_model.csv` schema

| Column                                         | Meaning                                                                                                                      |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `model_name`                                   | primary key                                                                                                                  |
| `subset`                                       | subset tag (natural / synthetic / adversarial / backdoor / production / medium-TF / LLM-fragment / real-scan / out-of-class) |
| `ground_truth_class`                           | clean / backdoor-add-DGP / backdoor-out-of-class                                                                             |
| `model_family`                                 | CNN / transformer-encoder / decoder-only-LLM / other                                                                         |
| `params_M`, `hidden_size`, `onnx_size_MB`      | model metadata                                                                                                               |
| `verdict`                                      | 3-class verdict (`add-DGP-CERTIFIED-POSITIVE` / `add-DGP-CLASS-NEGATIVE` / `UNCERTIFIED`)                                    |
| `verdict_legacy`                               | internal multi-class diagnostic from `verify_model` (DORMANT / OUTPUT-PRESERVED / DGP-FREE / EPS-BOUNDED / UNDECIDED / UNDECIDED-EXPORTER / τ-BOUNDED / BENIGN); paper uses the 3-class `verdict` column |
| `n_syntactic`, `n_admitted`, `additive_reject` | admission counts                                                                                                             |
| `epsilon`                                      | certificate value (sound upper bound on the gate-attributable contribution)                                                  |
| `eps_star`                                     | PGD-measured trigger-time output drift (empirical proxy)                                                                     |
| `delta`                                        | realised post-trigger output gap                                                                                             |
| `tau_sys`, `tau_dorm`, `tau_adm`               | thresholds in effect for this row                                                                                            |
| `wall_seconds`                                 | verifier wall-clock time                                                                                                     |
| `source_experiment`                            | which experiment produced this row                                                                                           |

## `aggregates.json` keys (selected)

- `e2.archproof_per_protocol` — F1 by `P-fail` / `P-reject` / `P-hold`
  protocol on the 11+49 in-class slice.
- `e2.binary_baselines` — ModelScan / Bober-Heuristic /
  LLM-RuleMimic / RandomTest baselines on the same slice.
- `mgrs.aggregate` — total cells passing MGRS soundness across families.
- `eic.toolchain_sweep` — Δ_T = 0 cells under 6 toolchain configs.
- `acpc_llm.totals` — ACPC sound-cell total on whole-LLM hidden states.
- `production.n_certified_positive` — ImageNet-pretrained backbone
  end-to-end loop closures.
- `g_necessity` — false-positive deltas under G1/G2/G3 ablations.

## Regenerating a result file

Each row in a `per_cell_*.csv` corresponds to a single
`verify_model_phaseC` (or `verify_model`) call on a model in
`benchmark/`. To re-run a specific experiment, see the per-RQ
walkthroughs in `REPRODUCE.md` — they each emit a `per_cell_*_my_run.csv`
that you can diff against the committed copy.
