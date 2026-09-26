"""Prod. CNN ACPC: validate Theorem 6 on 4 backbone × 3 gate type = 12 wrappers.

For each (backbone, gate_type):
  1. Build BackdoorWrapper.
  2. Run N=400 random ImageNet-shaped probes through the wrapper, hook
     gate1 (and gate2/gate3 for sha_un/int_un) Linear pre-activations.
  3. Aggregate captured pre-act tensors → [n, d_total].
  4. Sweep ρ ∈ {0.00, 0.05, 0.10, 0.20} × seeds {0, 1}, compute
     empirical_certificate_shift and acpc_bound.

12 × 4 × 2 = 96 cells.

Output: truth_source/per_cell_prod_cnn_acpc.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

ROOT = Path((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, str(ROOT))

from archproof.acpc import acpc_bound, empirical_certificate_shift
from archproof.robust_b_clean import inject_poison
from archproof.run_v3_production_scale import BackdoorWrapper, load_backbones

OUT_CSV = ROOT / "truth_source" / "per_cell_prod_cnn_acpc.csv"
OUT_CSV.parent.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
N_CLEAN = 400
RHOS = [0.00, 0.05, 0.10, 0.20]
SEEDS = [0, 1]
C_BCLEAN = 3.0
GATE_TYPES = ["sep_tar", "sha_un", "int_un"]


def collect_gate_distribution(wrapper):
    """Hook gate1 (+ gate2/gate3 if multi) and capture pre-activations
    over N_CLEAN random ImageNet-shaped probes."""
    gates = [wrapper.gate1]
    if wrapper.gate_type == "sha_un":
        gates.append(wrapper.gate2)
    elif wrapper.gate_type == "int_un":
        gates.extend([wrapper.gate2, wrapper.gate3])

    captured = {id(g): [] for g in gates}

    def make_hook(g):
        def _hk(_m, _i, out):
            captured[id(g)].append(out.detach().cpu().numpy())
        return _hk

    handles = [g.register_forward_hook(make_hook(g)) for g in gates]
    rng = torch.Generator().manual_seed(0)
    BATCH = 16
    n_done = 0
    try:
        with torch.no_grad():
            while n_done < N_CLEAN:
                bs = min(BATCH, N_CLEAN - n_done)
                x = torch.randn(bs, 3, 224, 224, generator=rng).to(DEVICE)
                wrapper(x)
                n_done += bs
    finally:
        for h in handles:
            h.remove()

    arrs = []
    for g in gates:
        chunks = captured[id(g)]
        a = np.concatenate(chunks, axis=0)[:N_CLEAN]
        arrs.append(a.reshape(a.shape[0], -1))
    return np.concatenate(arrs, axis=1).astype(np.float64)


def run_acpc_one_cell(g_clean, rho, seed, activation="relu"):
    n, d = g_clean.shape
    if rho > 0:
        cmin, cmax = float(np.nanmin(g_clean)), float(np.nanmax(g_clean))
        ar = max(abs(cmin), abs(cmax)) * 5.0
        ar = ar if (np.isfinite(ar) and ar > 0) else 15.0
        ar = min(ar, 1e6)
        poisoned = inject_poison(g_clean, poison_frac=rho,
                                 attacker_range=ar, mode="outlier_shift")
    else:
        poisoned = g_clean.copy()

    emp = empirical_certificate_shift(
        clean_X=g_clean, poisoned_X=poisoned,
        activation=activation, c=C_BCLEAN)
    b = acpc_bound(rho=rho, n=n, gate_data=[g_clean],
                   activations=[activation], payload_linf_norms=[1.0],
                   c=C_BCLEAN, L_chain=1.0, include_sampling=False)
    sound = bool(emp <= b["total_bound"])
    pg = b["per_gate"][0]
    return {
        "rho": rho, "n": n, "seed": seed, "activation": activation,
        "empirical_shift": float(emp),
        "bound_total": float(b["total_bound"]),
        "bound_bias": float(b["bias_bound"]),
        "bound_sampling": float(b["sampling_bound"]),
        "sound": sound,
        "L_phi": float(pg["L_phi"]),
        "max_MAD": float(pg["max_MAD"]),
        "max_IQR": float(pg["max_IQR"]),
        "delta_i": float(pg["delta_i"]),
        "gate_dim": d,
    }


def main():
    backbones = load_backbones()
    cells_total = len(backbones) * len(GATE_TYPES) * len(RHOS) * len(SEEDS)
    print("=" * 80)
    print(f"Prod. CNN ACPC: {len(backbones)} backbones × {len(GATE_TYPES)} "
          f"gate × {len(RHOS)} ρ × {len(SEEDS)} seeds = {cells_total} cells")
    print(f"  out: {OUT_CSV}")
    print("=" * 80)

    fields = [
        "backbone", "gate_type", "rho", "n", "seed", "activation",
        "empirical_shift", "bound_total", "bound_bias", "bound_sampling",
        "sound", "L_phi", "max_MAD", "max_IQR", "delta_i", "gate_dim",
        "n_gates", "extract_sec",
    ]
    new_csv = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_csv:
            w.writeheader()

        n_total = 0
        n_sound = 0
        for bb_name, (bb, feat_dim) in backbones.items():
            for gate_type in GATE_TYPES:
                tag = f"{bb_name}_{gate_type}"
                t0 = time.time()
                try:
                    bbones = load_backbones()
                    bb_fresh, fd = bbones[bb_name]
                    # BackdoorWrapper initialises the injected gate randomly;
                    # the backbone is pretrained and the probe inputs come from a
                    # seeded generator, so this is the remaining source of
                    # run-to-run variation.
                    torch.manual_seed(0)
                    wrapper = BackdoorWrapper(
                        bb_fresh, fd, gate_type=gate_type, n_classes=1000)
                    wrapper.eval().to(DEVICE)
                    g_clean = collect_gate_distribution(wrapper)
                except Exception as e:
                    print(f"[skip {tag}] FAIL: {type(e).__name__}: {e}")
                    continue
                extract_sec = time.time() - t0
                n_g = {"sep_tar": 1, "sha_un": 2, "int_un": 3}[gate_type]
                print(f"\n[{tag:30s}] gate={g_clean.shape}  n_g={n_g}  "
                      f"extract={extract_sec:.1f}s")

                for rho in RHOS:
                    for seed in SEEDS:
                        r = run_acpc_one_cell(g_clean, rho, seed)
                        n_total += 1
                        n_sound += int(r["sound"])
                        print(f"  ρ={rho:.2f} s={seed}  "
                              f"emp={r['empirical_shift']:.4e} "
                              f"bnd={r['bound_total']:.4e}  "
                              f"sound={'✓' if r['sound'] else '✗'}")
                        w.writerow({
                            "backbone": bb_name,
                            "gate_type": gate_type,
                            "rho": rho, "n": r["n"], "seed": seed,
                            "activation": r["activation"],
                            "empirical_shift": r["empirical_shift"],
                            "bound_total": r["bound_total"],
                            "bound_bias": r["bound_bias"],
                            "bound_sampling": r["bound_sampling"],
                            "sound": r["sound"],
                            "L_phi": r["L_phi"],
                            "max_MAD": r["max_MAD"],
                            "max_IQR": r["max_IQR"],
                            "delta_i": r["delta_i"],
                            "gate_dim": r["gate_dim"],
                            "n_gates": n_g,
                            "extract_sec": round(extract_sec, 2),
                        })
                        f.flush()

    print()
    print("=" * 80)
    print(f"DONE: {n_sound}/{n_total} cells SOUND  "
          f"({100.0 * n_sound / max(n_total, 1):.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
