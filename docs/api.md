# ArchProof API reference

## Top-level imports

```python
from archproof import (
    verify_model,           # 6-class hierarchical verifier
    verify_model_phaseC,    # 3-class deployment-view verifier
    print_result,           # pretty-printer for VerificationResult
    VerificationResult,     # 6-class result dataclass
    PhaseCResult,           # 3-class result dataclass
)
```

---

## `archproof.verify.verify_model(...)`

Hierarchical verifier returning a 6-class diagnostic verdict
(DORMANT / GDP-FREE / EPS-BOUNDED / OUTPUT-PRESERVED / UNDECIDED /
UNDECIDED-EXPORTER).

### Signature

```python
def verify_model(
    onnx_path: str,
    b_clean_ub: float = 0.95,
    n_splits: int = 50,
    b_extended: float = 10.0,
    gdp_flags: dict = None,
    tau_adm: float = 0.1,
    n_probe: int = 20,
) -> VerificationResult
```

### Arguments

| Arg          | Type    | Default  | Meaning                                                                                     |
| ------------ | ------- | -------- | ------------------------------------------------------------------------------------------- |
| `onnx_path`  | `str`   | —        | Path to the ONNX file.                                                                      |
| `b_clean_ub` | `float` | `0.95`   | Upper bound of the B_clean interval hull (0.95 for normalised images).                      |
| `n_splits`   | `int`   | `50`     | Number of input-box splits at Level 2.                                                      |
| `b_extended` | `float` | `10.0`   | Extended range for the benign-filter (Level 4).                                             |
| `gdp_flags`  | `dict`  | all True | Enable / disable G_i conditions: `{"G1": ..., "G2": ..., "G3": ..., "G4": ..., "T10": ...}` |
| `tau_adm`    | `float` | `0.1`    | Theorem 10 admission threshold (gates with `median(                                         |
| `n_probe`    | `int`   | `20`     | Number of clean calibration probe samples drawn for T10.                                    |

### Returns

A `VerificationResult` with these key fields:

| Field                        | Type             | Meaning                                                                                      |
| ---------------------------- | ---------------- | -------------------------------------------------------------------------------------------- |
| `verdict`                    | `str`            | One of GDP-FREE / DORMANT / EPS-BOUNDED / OUTPUT-PRESERVED / UNDECIDED / UNDECIDED-EXPORTER. |
| `total_output_margin`        | `float`          | The certificate ε.                                                                           |
| `n_gdp_candidates`           | `int`            | Syntactic activation→Mul patterns found.                                                     |
| `n_gdp_admitted`             | `int`            | Gates that survived T10 + G_i admission.                                                     |
| `n_gdp_rejected_non_dormant` | `int`            | Gates rejected by T10 because empirical median ≥ τ_adm.                                      |
| `gate_admission_report`      | `dict[str, str]` | Per-gate admission outcome.                                                                  |
| `gate_epsilons`              | `list[dict]`     | Per-gate ε contributions.                                                                    |
| `proven_level`               | `str`            | Which level proved dormancy (`none` / `strict` / `tau_ibp` / `tau_split` / `tau_quad`).      |

### Example

```python
from archproof import verify_model, print_result
result = verify_model("model.onnx", b_clean_ub=0.95, tau_adm=0.1, n_probe=20)
print_result(result)
print(f"verdict={result.verdict}, ε={result.total_output_margin:.4f}")
```

---

## `archproof.verify_phaseC.verify_model_phaseC(...)`

Phase-C 3-class verifier (the paper's deployment-view verdict).

### Signature

```python
def verify_model_phaseC(
    onnx_path: str,
    b_clean_ub: float = 0.95,
    trigger_eta: float = 0.05,
    tau_sys: float = 1e-3,
    n_probe: int = 32,
    seed: int = 42,
) -> PhaseCResult
```

### Arguments

| Arg           | Type    | Default | Meaning                                               |
| ------------- | ------- | ------- | ----------------------------------------------------- |
| `onnx_path`   | `str`   | —       | Path to the ONNX file.                                |
| `b_clean_ub`  | `float` | `0.95`  | Upper bound of B_clean.                               |
| `trigger_eta` | `float` | `0.05`  | Trigger box width η (D(T) = B_clean ⊕ T_box(η)).      |
| `tau_sys`     | `float` | `1e-3`  | Verdict threshold (`ε > τ_sys` ⇒ CERTIFIED-POSITIVE). |
| `n_probe`     | `int`   | `32`    | Number of clean probe samples.                        |
| `seed`        | `int`   | `42`    | Probe RNG seed.                                       |

### Returns

A `PhaseCResult` with these key fields:

| Field               | Type         | Meaning                                                                 |
| ------------------- | ------------ | ----------------------------------------------------------------------- |
| `verdict_phaseC`    | `str`        | `add-DGP-CERTIFIED-POSITIVE` / `add-DGP-CLASS-NEGATIVE` / `UNCERTIFIED` |
| `epsilon_phaseC`    | `float`      | The certificate ε on the trigger-extended interval D(T).                |
| `n_syntactic`       | `int`        | Syntactic activation→Mul patterns.                                      |
| `n_admitted_phaseC` | `int`        | Gates admitted by the trigger-aware test.                               |
| `gate_epsilons`     | `list[dict]` | Per-admitted-gate ε breakdown.                                          |
| `epsilon_blowup`    | `bool`       | True if IBP saturated above the vacuous threshold.                      |

### Example

```python
from archproof import verify_model_phaseC

r = verify_model_phaseC("model.onnx", trigger_eta=0.05, tau_sys=1e-3)
print(r.verdict_phaseC, r.epsilon_phaseC, r.n_admitted_phaseC)
```

---

## `archproof.acpc` — Adversarial Calibration Poisoning Certificate (Theorem 7)

```python
from archproof.acpc import acpc_chain_bound

bound, observed = acpc_chain_bound(
    clean=clean_probe,           # numpy array, shape (N,)
    poisoned=poisoned_probe,     # numpy array, shape (N,)
    activation="relu",           # one of 11 supported activations
    rho=0.1,                     # poisoning fraction
    mad_multiplier=3.0,          # c in median-MAD interval hull
)
assert observed <= bound  # soundness invariant
```

---

## `archproof.mgrs` — Minimum Gate-Removal Set (Theorem 8)

```python
from archproof.mgrs import greedy_minimum_removal

# Pick gates greedily (descending contribution) until target ε is met.
indices = greedy_minimum_removal(contributions, target_eps)
```

---

## `archproof.escalate` — EIC bijection check (Theorem 11)

```python
from archproof.escalate import compare_exporters

# Compare the certificate ε across two exporter configs.
delta_T = compare_exporters(onnx_path_1, onnx_path_2)
```

---

## `archproof.gate_admission` — G1/G2/G3/G4/T10 candidate-gate admission

```python
from archproof.gate_admission import admit_gates

admitted, rejected = admit_gates(
    onnx_model, clean_probe, tau_adm=0.1
)
```

---

## `archproof.activation_epsilon` — Per-activation envelope

```python
from archproof.activation_epsilon import envelope

eps = envelope("relu", l=-1.0, u=2.0)
# Returns the closed-form sound upper bound on sup_{z in [l,u]} |φ(z)|
```

Supported activations: `relu`, `hardtanh`, `hardswish`, `sigmoid`, `tanh`,
`gelu`, `silu`, `mish`, `elu`, `selu`, `softplus`.

---

## `archproof.llm_gate_rescue` — LayerNorm geometric rescue (Lemma 5)

```python
from archproof.llm_gate_rescue import layernorm_geometric_bound

# ‖LN_{γ,β}(x)‖_∞ ≤ ‖γ‖_∞ √D + ‖β‖_∞
bound = layernorm_geometric_bound(gamma, beta)
```

---

## `archproof.chain_sensitivity` — Affine chain pair (Proposition 6)

```python
from archproof.chain_sensitivity import compute_chain_pair

# (A_post, B_post) for the post-Mul subgraph of gate `gate_name`.
A_post, B_post = compute_chain_pair(onnx_model, gate_name, b_clean)
```

---

## Result schema cross-reference

For the schemas of `results/per_model.csv`, `results/per_cell_*.csv`,
and `results/aggregates.json`, see `results/README.md`.

For the benchmark inventory (123 models with HuggingFace / torchvision
URLs), see `benchmark/MODELS.md`.
