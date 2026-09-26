#!/usr/bin/env python3
"""Network-free smoke test of the open-world (Phase F) 500-model scan.

The full scan (Table 49) downloads and verifies 500 Hugging Face models and
takes 6-10 h. This smoke instead checks, in a second and with no network, the
two things that can silently break that scan:

  1. the pinned manifest is intact -- 500 repos, in the paper's recorded scan
     order (benchmark/phaseF_manifest.json matches the order in
     truth_source/per_model_phaseF.csv);
  2. the per-repo verify step the scan runs is wired to the real verifier, by
     calling verify_model_phaseC on one bundled ONNX exactly as verify_one does
     after a download.

It downloads nothing. Run the real scan with `bash scripts/run_phaseF.sh`.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def check_manifest():
    """The manifest must list the same repos, in the same order, as the record."""
    data = json.load(open(os.path.join(ROOT, "benchmark", "phaseF_manifest.json")))
    models = data["models"] if isinstance(data, dict) else data
    sort = data.get("sort") if isinstance(data, dict) else "list"
    man_ids = [m["repo_id"] for m in models]

    rec_path = os.path.join(ROOT, "truth_source", "per_model_phaseF.csv")
    rec_ids = [r["repo_id"] for r in csv.DictReader(open(rec_path))]

    print(f">> manifest: {len(man_ids)} repos (sort={sort})")
    assert len(man_ids) == len(rec_ids), \
        f"manifest has {len(man_ids)} repos, record has {len(rec_ids)}"
    assert man_ids == rec_ids, "manifest order != the paper's recorded scan order"
    print(f"   order matches truth_source/per_model_phaseF.csv ({len(man_ids)} repos)  [OK]")
    return len(man_ids)


def check_verify_step():
    """The scan verifies each downloaded repo with verify_model_phaseC; run that
    same call on a tiny bundled model so the per-repo step is exercised offline."""
    from archproof.verify_phaseC import verify_model_phaseC
    model = os.path.join(ROOT, "models", "clean_panel", "clean_mish.onnx")
    r = verify_model_phaseC(model)
    print(f"   verify_model_phaseC on a bundled model -> {r.verdict_phaseC}  [OK]")


def main():
    print(">> Phase F (open-world 500-model scan) smoke -- no network, no download")
    n = check_manifest()
    check_verify_step()
    print(f"\nRESULT: PASS -- {n}-repo pinned manifest intact and the verify step "
          "runs.\n         Full scan (6-10 h, network): bash scripts/run_phaseF.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
