"""Focused probe diagnostic on Yi-6B clean ONNX. Skips IBP, skips
Bug-D filter, skips verdict logic. Just calls probe_gate_medians and
prints what it returns. Ditto for the 32 SwiGLU sigmoid gates so we
can see exactly what each gate measured.

Run on B (~5-10 min, graph-only load + ORT session + 20 inferences):
  python3 scripts/probe/probe_only_yi.py 2>&1 | tee logs/probe_only_yi.out
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tell the probe to dump its internals.
os.environ["ARCHPROOF_PROBE_VERBOSE"] = "1"
os.environ["ARCHPROOF_PROBE_MAX_GB"] = "40"

from archproof.gate_admission import probe_gate_medians
from archproof.verify_phaseC import _gate_set_after_g1_g4

ONNX_PATH = (ROOT / "benchmark" / "7b_onnx" / "yi-6b" / "yi-6b.onnx")


def main():
    if not ONNX_PATH.exists():
        print(f"missing: {ONNX_PATH}", file=sys.stderr); sys.exit(1)

    print(f"loading graph (no weights) ...", flush=True)
    m = onnx.load(str(ONNX_PATH), load_external_data=False)
    print(f"  nodes={len(m.graph.node)}", flush=True)

    print(f"computing G1+G4 syntactic gate set ...", flush=True)
    gate_tensors, _ = _gate_set_after_g1_g4(m)
    print(f"  gate_tensors: {len(gate_tensors)} candidates", flush=True)
    for t in sorted(gate_tensors)[:5]:
        print(f"    {t}", flush=True)
    if len(gate_tensors) > 5:
        print(f"    ... ({len(gate_tensors) - 5} more)", flush=True)

    print(f"\ncalling probe_gate_medians on {ONNX_PATH} ...", flush=True)
    medians = probe_gate_medians(str(ONNX_PATH), gate_tensors)
    print(f"\nprobe returned: {len(medians)} medians", flush=True)

    if not medians:
        print("EMPTY MEDIANS — probe failed entirely.", flush=True)
        return

    print(f"\n=== per-gate median |g|_∞ over 20 random-token probes ===")
    for t, v in sorted(medians.items()):
        flag = "DORMANT(<0.3)" if v < 0.3 else "non-dormant(>=0.3)"
        print(f"  {t[:50]:<50}  median={v:>10.6f}  {flag}", flush=True)
    n_lt_03 = sum(1 for v in medians.values() if v < 0.3)
    print(f"\nsummary: {n_lt_03}/{len(medians)} gates have median < 0.3 "
          f"(would be admitted by Bug-D filter)")


if __name__ == "__main__":
    main()
