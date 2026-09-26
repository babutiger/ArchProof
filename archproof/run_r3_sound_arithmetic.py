"""R3-A4: Sound-arithmetic recomputation of headline epsilon numbers.

For each model in a representative subset (E1 backdoor panel + a few clean
controls + E8 CIFAR e2e + E11 Mistral fragment), we:
  1. Run verify_model() in default IEEE-754 double-precision mode and capture
     per-gate (g_lb, g_ub, payload_abs_max, contribution).
  2. Recompute, per gate, the activation envelope using outward (1 ULP)
     directed-rounded floats, plus a 1 ULP outward bump on the payload bound.
  3. Sum to get the sound certificate epsilon_sound.
  4. Report epsilon_float64, epsilon_sound, |Delta|, and the relative outward
     margin epsilon_sound/epsilon_float64.

Outcome: the relative outward margin should be very small (<= a few ULP times
the chain length), confirming that the float64 epsilon is sound modulo a
machine-bounded outward correction.

Usage:
  cd $ARCHPROOF_ROOT
  conda activate alpha-beta-crown
  python3 -m archproof.run_r3_sound_arithmetic
"""
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import onnx

from archproof.verify import verify_model
from archproof.sound_arithmetic import envelope_outward


def sound_epsilon_for_result(result) -> dict:
    """Reconstruct the certificate sum using outward-rounded scalar arithmetic.

    Returns both envelope-only sum (matches verify.total_epsilon) and the
    contribution sum (envelope * payload, matches verify.total_output_margin).
    Each is computed with running outward-add.
    """
    env_sound = 0.0
    contrib_sound = 0.0
    per_gate = []
    for ge in result.gate_epsilons:
        act = ge.get("activation", "relu")
        g_lb = float(ge["g_lb"])
        g_ub = float(ge["g_ub"])
        e_out = float(envelope_outward(act, np.array([g_lb]), np.array([g_ub]))[0])
        p = ge.get("payload_abs_max")
        p_out = (float(np.nextafter(p, +np.inf)) if p is not None else None)
        if p_out is not None:
            contrib = float(np.nextafter(e_out * p_out, +np.inf))
        else:
            contrib = e_out
        per_gate.append({
            "gate": ge["gate_node"], "act": act,
            "eps_float64": ge["epsilon"], "eps_outward": e_out,
            "payload_float64": p, "payload_outward": p_out,
            "contrib_float64": ge.get("contribution"),
            "contrib_outward": contrib,
        })
        env_sound = float(np.nextafter(env_sound + e_out, +np.inf))
        contrib_sound = float(np.nextafter(contrib_sound + contrib, +np.inf))
    return {"envelope_outward": env_sound,
            "contribution_outward": contrib_sound,
            "per_gate": per_gate}


def main():
    t0 = time.time()
    models = []

    # E1 backdoor panel + E8 CIFAR e2e + a few small clean
    backdoor_paths = sorted(glob.glob("/tmp/bober_onnx/*.onnx"))
    backdoor_paths += sorted(glob.glob("/tmp/handcrafted_onnx/*.onnx"))
    backdoor_paths += [
        "/tmp/v3_e2e_case/backdoor_before_surgery.onnx",
    ]
    clean_paths = [
        _os_ar.path.join(_AR, "benchmark/clean/clean_mish.onnx"),
        _os_ar.path.join(_AR, "benchmark/clean/clean_hardswish.onnx"),
        _os_ar.path.join(_AR, "benchmark/clean/clean_eca.onnx"),
        _os_ar.path.join(_AR, "benchmark/clean/clean_cbam.onnx"),
        _os_ar.path.join(_AR, "benchmark/clean/swiglu_ffn.onnx"),
        _os_ar.path.join(_AR, "benchmark/clean/glu_net.onnx"),
        _os_ar.path.join(_AR, "benchmark/clean/film_conditioning.onnx"),
        _os_ar.path.join(_AR, "benchmark/clean/se_block.onnx"),
        _os_ar.path.join(_AR, "benchmark/clean/gated_residual.onnx"),
        _os_ar.path.join(_AR, "benchmark/clean/mobilenet_v3_small.onnx"),
    ]
    paths = sorted({*backdoor_paths, *clean_paths})

    print(f"Auditing {len(paths)} models with sound-arithmetic backend")
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            t1 = time.time()
            r = verify_model(p, b_clean_ub=0.95, n_splits=20)
        except Exception as e:
            print(f"  {os.path.basename(p):40s}  verify FAIL: {e}")
            continue
        sound = sound_epsilon_for_result(r)
        env_f = float(r.total_epsilon)         # envelope-only sum
        env_s = sound["envelope_outward"]
        ctr_f = float(getattr(r, "total_output_margin", 0.0))  # envelope * payload
        ctr_s = sound["contribution_outward"]
        delta_env = abs(env_s - env_f)
        delta_ctr = abs(ctr_s - ctr_f)
        rel_env = delta_env / max(abs(env_f), 1e-30)
        rel_ctr = delta_ctr / max(abs(ctr_f), 1e-30)
        elapsed = time.time() - t1
        print(f"  {os.path.basename(p):40s}  "
              f"n_a={len(r.gate_epsilons):2d}  "
              f"env64={env_f:.4e} env_snd={env_s:.4e} (rel={rel_env:.2e})  "
              f"ctr64={ctr_f:.4e} ctr_snd={ctr_s:.4e} (rel={rel_ctr:.2e})  "
              f"({elapsed:.1f}s)")
        models.append({
            "model": os.path.basename(p),
            "n_admitted": len(r.gate_epsilons),
            "envelope_float64": env_f,
            "envelope_outward": env_s,
            "envelope_abs_delta": delta_env,
            "envelope_rel_delta": rel_env,
            "contribution_float64": ctr_f,
            "contribution_outward": ctr_s,
            "contribution_abs_delta": delta_ctr,
            "contribution_rel_delta": rel_ctr,
            "verdict": getattr(r, "verdict", None),
            "wall_sec": elapsed,
            "per_gate": sound["per_gate"],
        })

    nz_env = [m for m in models if m["envelope_float64"] != 0]
    nz_ctr = [m for m in models if m["contribution_float64"] != 0]
    summary = {
        "n_models": len(models),
        "n_with_admitted_gates": sum(1 for m in models if m["n_admitted"] > 0),
        "n_with_nonzero_envelope": len(nz_env),
        "n_with_nonzero_contribution": len(nz_ctr),
        "envelope": {
            "max_abs_delta": max((m["envelope_abs_delta"] for m in models), default=0.0),
            "max_rel_delta": max((m["envelope_rel_delta"] for m in nz_env), default=0.0),
            "median_rel_delta": float(np.median([m["envelope_rel_delta"] for m in nz_env])) if nz_env else 0.0,
        },
        "contribution": {
            "max_abs_delta": max((m["contribution_abs_delta"] for m in models), default=0.0),
            "max_rel_delta": max((m["contribution_rel_delta"] for m in nz_ctr), default=0.0),
            "median_rel_delta": float(np.median([m["contribution_rel_delta"] for m in nz_ctr])) if nz_ctr else 0.0,
        },
        "wall_sec_total": time.time() - t0,
    }
    out = Path(_os_ar.path.join(_AR, "benchmark/r3_sound_arithmetic.json"))
    out.write_text(json.dumps({"summary": summary, "models": models}, indent=2, default=str))
    print(f"\nwrote {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
