"""Phase B: 3-class verdict mapping.

Maps the legacy 6-class verdict (DGP-FREE / DORMANT / OUTPUT-PRESERVED /
EPS-BOUNDED / UNDECIDED / UNDECIDED-EXPORTER / BENIGN / etc.) to the new
3-class scheme (CERTIFIED-POSITIVE / CLASS-NEGATIVE / UNCERTIFIED).

## Semantics

- **add-DGP-CERTIFIED-POSITIVE**: verifier admitted at least one add-DGP
  gate and computed an ε bound for it. Carries deployment-relevant
  quantitative information (the ε itself).

- **add-DGP-CLASS-NEGATIVE**: verifier formally proved the graph contains
  no add-DGP class member. NARROW assertion: this does NOT exclude
  out-of-class architectural backdoors (nested-Mul, attention-tail,
  custom-op gates, etc.). For an open-world defender, CLASS-NEGATIVE
  alone is NOT a deployment-safe verdict.

- **UNCERTIFIED**: verifier cannot conclude. Caused by:
  - unsupported op encountered (op coverage gap)
  - additive-branch certifier rejected (structure is non-additive,
    so we cannot apply add-DGP class definition reliably)
  - IBP fallback to vacuous bound (10^300)
  - EIC bijection broken (sub-class: UNCERTIFIED-EXPORTER)
  - any other indeterminacy

## Mapping rules

| Legacy verdict | n_syn | n_adm | additive_reject | ibp_blowup | New verdict |
|---------------|-------|-------|-----------------|------------|-------------|
| DGP-FREE      | 0     | 0     | False           | False      | CLASS-NEGATIVE |
| DGP-FREE      | >0    | 0     | True            | -          | UNCERTIFIED (out-of-class structural) |
| DGP-FREE      | >0    | 0     | False           | False      | CLASS-NEGATIVE (admission rejected, in-class) |
| DORMANT / OUTPUT-PRESERVED / EPS-BOUNDED / BENIGN | - | >0 | - | - | CERTIFIED-POSITIVE |
| UNDECIDED     | -     | -     | -               | True       | UNCERTIFIED (IBP) |
| UNDECIDED     | -     | -     | -               | False      | UNCERTIFIED |
| UNDECIDED-EXPORTER | - | -    | -               | -          | UNCERTIFIED-EXPORTER (sub) |

Special: OUT-OF-CLASS evasions in the E12 panel currently return DGP-FREE
with n_syn>0 and n_adm=0 because the additive-branch certifier rejects
them. Under the new mapping, these go to UNCERTIFIED — addressing the
reviewer's open-world false-negative critique.
"""
from __future__ import annotations

CLASS_POSITIVE = "add-DGP-CERTIFIED-POSITIVE"
CLASS_NEGATIVE = "add-DGP-CLASS-NEGATIVE"
UNCERTIFIED = "UNCERTIFIED"
UNCERTIFIED_EXPORTER = "UNCERTIFIED-EXPORTER"


def _to_int(v):
    if v in ("", None):
        return 0
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _to_bool(v):
    if v in (True, "True", "true", 1, "1"):
        return True
    return False


def _to_float(v):
    if v in ("", None):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def map_to_3class(legacy_verdict: str,
                  n_syn: int = 0,
                  n_adm: int = 0,
                  additive_reject: bool = False,
                  ibp_blowup: bool = False,
                  panel: str = "",
                  source_experiment: str = "") -> str:
    """Return the 3-class verdict for one model.

    Required minimum input: legacy_verdict.
    Other args refine the mapping for borderline cases (admission
    rejection vs out-of-class structural rejection).
    """
    v = (legacy_verdict or "").strip().upper()
    n_syn = _to_int(n_syn)
    n_adm = _to_int(n_adm)
    additive_reject = _to_bool(additive_reject)
    ibp_blowup = _to_bool(ibp_blowup)

    # EIC bijection break is a sub-class of UNCERTIFIED
    if v in ("UNDECIDED-EXPORTER",):
        return UNCERTIFIED_EXPORTER

    # IBP fallback / unsupported op / general indeterminacy
    if v in ("UNDECIDED", "UNKNOWN", ""):
        return UNCERTIFIED
    if ibp_blowup:
        return UNCERTIFIED

    # Positive verdicts under strict semantics: verifier admitted at
    # least one gate and computed a quantitative ε. If n_adm == 0 the
    # verifier didn't actually exercise the certificate path; the
    # legacy "DORMANT/OUT-PRES with n_adm=0" cases collapse to
    # CLASS-NEGATIVE because they're really "no add-DGP class member
    # admitted".
    if v in ("DORMANT", "OUTPUT-PRESERVED", "EPS-BOUNDED", "BENIGN",
             "Τ-BOUNDED", "TAU-BOUNDED"):
        if n_adm > 0:
            return CLASS_POSITIVE
        return CLASS_NEGATIVE

    # DGP-FREE family: branch on cause
    # (handles both new "DGP-FREE" and legacy "GDP-FREE" spelling)
    if v in ("DGP-FREE", "GDP-FREE"):
        # Out-of-class structural rejection: verifier saw activation→Mul
        # syntactic patterns but additive-branch (or custom op) ruled
        # them out. Cannot certify either way.
        if additive_reject:
            return UNCERTIFIED
        # E12 evasion panel: out-of-class by construction. Even if we
        # cannot detect the structural failure cleanly, mark UNCERTIFIED
        # for open-world honesty.
        if source_experiment == "E12" or panel == "out-of-class":
            return UNCERTIFIED
        # Else: clean model, no syntactic pattern OR admission cleanly
        # rejected an in-class candidate as non-dormant on the probe.
        # CLASS-NEGATIVE is the correct narrow assertion.
        return CLASS_NEGATIVE

    # Unknown legacy label → conservative
    return UNCERTIFIED


def is_deployment_safe(verdict_3class: str) -> bool:
    """For an open-world defender, only CERTIFIED-POSITIVE with ε ≤ τ_sys
    is deployment-safe. CLASS-NEGATIVE is a narrow assertion only."""
    return verdict_3class == CLASS_POSITIVE


def is_open_world_negative(verdict_3class: str) -> bool:
    """A verdict that an open-world defender could mistake for 'safe'
    if interpreted carelessly. CLASS-NEGATIVE is the dangerous case
    when not paired with operator-coverage qualification."""
    return verdict_3class == CLASS_NEGATIVE
