#!/usr/bin/env python3
"""
Potion case study (paper Appendix D.5.5, and the shepherd response):
reproduce the ArchProof half of the live example directly, without the
500-model Phase F open-world sweep.

Protect AI's Guardian flagged the deployed static-embedding model
`minishlab/potion-base-8M` as "suspicious" for an architectural backdoor
(a finding its maintainers dispute,
https://huggingface.co/minishlab/potion-base-8M/discussions/2). Run on the
exact file, ArchProof returns a sound add-DGP-CLASS-NEGATIVE: the graph
contains no Mul, so no gate is even syntactically admissible
(n_syntactic = n_admitted = 0), and the model lies provably outside the
certified class. A structural flag is not a sound verdict; this is the gap
the certificate fills. The verdict does not pronounce the model clean, only
that this specific structural suspicion is discharged soundly.

This downloads onnx/model.onnx from the repo (HF hub first, then
hf-mirror.com) and runs the exact `verify_model_phaseC` path used for every
ArchProof verdict in the paper, then checks it against the archived record
in truth_source/per_model_phaseF.csv.

Archived (expected) row in truth_source/per_model_phaseF.csv:
    verdict=add-DGP-CLASS-NEGATIVE, n_syntactic=0, n_admitted=0, epsilon=0.0

Usage (needs the pinned env, `conda activate archproof_repro`, for the full run):
    PYTHONPATH=<artifact-root> python scripts/potion_case_study.py
    QUICK=1 PYTHONPATH=<artifact-root> python scripts/potion_case_study.py   # read the archived row only
"""
import csv
import os
import sys
import time
from pathlib import Path

REPO_ID = "minishlab/potion-base-8M"
ONNX_FILE = "onnx/model.onnx"
EXPECT_VERDICT = "add-DGP-CLASS-NEGATIVE"
ROOT = Path(__file__).resolve().parents[1]  # artifact root
RECORD = ROOT / "truth_source" / "per_model_phaseF.csv"


def archived_row():
    """The potion row recorded by the paper's Phase F run, if present."""
    if not RECORD.exists():
        return None
    with open(RECORD, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("repo_id") == REPO_ID:
                return r
    return None


def download():
    """Fetch onnx/model.onnx: caller's HF_ENDPOINT (if any), then the real
    HF hub, then hf-mirror.com -- the same order Phase F uses."""
    from huggingface_hub import hf_hub_download

    order, seen = [], set()
    for ep in [os.environ.get("HF_ENDPOINT"), "https://huggingface.co",
               "https://hf-mirror.com"]:
        if ep and ep not in seen:
            seen.add(ep)
            order.append(ep)
    last = None
    for ep in order:
        try:
            os.environ["HF_ENDPOINT"] = ep
            path = hf_hub_download(repo_id=REPO_ID, filename=ONNX_FILE)
            print(f"  downloaded {ONNX_FILE} via {ep}")
            return path
        except Exception as e:  # try the next endpoint
            last = e
            print(f"  {ep} failed: {e}")
    raise RuntimeError(f"could not download {REPO_ID}/{ONNX_FILE}: {last}")


def main():
    quick = os.environ.get("QUICK", "0") == "1"
    row = archived_row()

    print("=" * 66)
    print(">> Potion case study (Appendix D.5.5): a Guardian flag vs a sound")
    print(f">> ArchProof class-negative on the deployed {REPO_ID}")
    print("=" * 66)
    if row:
        print("Archived record (truth_source/per_model_phaseF.csv):")
        print(f"  verdict={row['verdict']}  n_syntactic={row['n_syntactic']}  "
              f"n_admitted={row['n_admitted']}  epsilon={row['epsilon']}  "
              f"verify_sec={row['verify_sec']}")
    else:
        print("  (no archived potion row found in per_model_phaseF.csv)")

    if quick:
        ok = bool(row) and EXPECT_VERDICT in (row["verdict"] or "")
        print("\n[QUICK] archived record only; download+verify skipped.")
        print(f"RESULT: {'PASS' if ok else 'CHECK'} "
              f"(archived verdict {'matches' if ok else 'does not match'} "
              f"{EXPECT_VERDICT})")
        return 0 if ok else 1

    print(f"\nDownloading {REPO_ID}/{ONNX_FILE} (about 58 MB) ...")
    onnx_path = download()

    print("Running verify_model_phaseC (the exact ArchProof verdict path) ...")
    from archproof.verify_phaseC import verify_model_phaseC
    t0 = time.time()
    r = verify_model_phaseC(str(onnx_path))
    dt = round(time.time() - t0, 1)

    print("\nReproduced verdict:")
    print(f"  verdict={r.verdict_phaseC}  n_syntactic={r.n_syntactic}  "
          f"n_admitted={r.n_admitted_phaseC}  epsilon={r.epsilon_phaseC}  ({dt}s)")

    ok = EXPECT_VERDICT in (r.verdict_phaseC or "") and r.n_admitted_phaseC == 0
    if row:
        same = (r.verdict_phaseC or "") == (row["verdict"] or "")
        print(f"Matches archived record: {same}")
        ok = ok and same
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(expected {EXPECT_VERDICT} with n_admitted=0)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
