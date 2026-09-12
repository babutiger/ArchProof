#!/usr/bin/env python3
"""Verify the fixed clean-model panel is present and intact.

The 97 clean models (benchmark/clean/*.onnx) are the false-positive panel for
the detection / F1 experiments. Their generators are unseeded and several use
random torchvision weights, so the ARTIFACT SHIPS THE EXACT ONNX rather than
regenerating them: a rebuild would draw different weights and could move the
FP/TN counts of the gated benign structures (CBAM, ECA, gated_*). This script
checks every model named in artifact/clean_panel_manifest.json is present and,
with --hash, that its bytes are unchanged from a recorded sha256.

Usage:
  python artifact/verify/check_clean_panel.py            # presence + size
  python artifact/verify/check_clean_panel.py --record   # write sha256 baseline
  python artifact/verify/check_clean_panel.py --hash      # check against baseline
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

# artifact/ (self-contained release); the manifest and sha256 baseline live at
# the artifact root.
ART = Path(__file__).resolve().parents[1]
MANIFEST = ART / "models" / "clean_panel_manifest.json"
BASELINE = ART / "models" / "clean_panel_sha256.json"


def resolve_model(file_rel):
    # The manifest names models as "benchmark/clean/<x>.onnx" (the repo path).
    # In the release the 97 ONNX are centralized under models/clean_panel/;
    # try that, then the repo-root benchmark/clean/ location.
    from pathlib import Path as _P
    for cand in (ART / file_rel,
                 ART / "models" / "clean_panel" / _P(file_rel).name,
                 ART.parent / file_rel):
        if cand.exists():
            return cand
    return ART / file_rel  # nonexistent -> reported as missing


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="write sha256 baseline")
    ap.add_argument("--hash", action="store_true", help="check against baseline")
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text())
    missing, size_off = [], []
    hashes = {}
    for m in man["models"]:
        p = resolve_model(m["file"])
        if not p.exists():
            missing.append(m["name"])
            continue
        mb = p.stat().st_size / 1e6
        if abs(mb - m["size_mb"]) > max(0.05, 0.02 * m["size_mb"]):
            size_off.append((m["name"], round(mb, 2), m["size_mb"]))
        if a.record or a.hash:
            hashes[m["name"]] = sha256(p)

    print(f"clean panel: {len(man['models']) - len(missing)}/{len(man['models'])} present"
          f" ({man['total_mb']} MB expected)")
    if missing:
        print(f"  MISSING ({len(missing)}): {', '.join(missing)}")
    for n, got, exp in size_off:
        print(f"  SIZE off: {n} {got}MB vs {exp}MB expected")

    if a.record:
        BASELINE.write_text(json.dumps(hashes, indent=1))
        print(f"wrote sha256 baseline for {len(hashes)} models -> {BASELINE.name}")
        return 0
    if a.hash:
        if not BASELINE.exists():
            print("no baseline; run --record first")
            return 2
        base = json.loads(BASELINE.read_text())
        drift = [n for n, h in hashes.items() if base.get(n) != h]
        if drift:
            print(f"  CONTENT changed ({len(drift)}): {', '.join(drift)}")
        else:
            print("  all bytes match the recorded baseline")
        return 1 if drift else 0

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
