#!/usr/bin/env python3
"""Build the FIXED open-world manifest from the paper's own run record.

The open-world scan (Table 49 / tab:appx:openworld) must re-scan the SAME 510
public models in the SAME order the paper reported, or the per-model verdicts
won't line up. Discovering models live from the HF API would give a different
set/order every run (rankings drift), so instead we pin the manifest to the
recorded order in truth_source/per_model_phaseF.csv.

Writes benchmark/phaseF_manifest.json (list of {repo_id, onnx_files, repo_mb}),
which run_phaseF.sh reuses instead of re-discovering.
"""
import csv, json, os

_AR = os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(_AR, "truth_source", "per_model_phaseF.csv")
OUT = os.path.join(_AR, "benchmark", "phaseF_manifest.json")


def main():
    rows = list(csv.DictReader(open(REC)))
    manifest = []
    for r in rows:
        rid = (r.get("repo_id") or "").strip()
        if not rid:
            continue
        onnx = (r.get("onnx_file") or "").strip()
        try:
            mb = float(r.get("repo_mb") or 0)
        except ValueError:
            mb = 0.0
        manifest.append({
            "repo_id": rid,
            "onnx_files": [onnx] if onnx else [],
            "repo_mb": mb,
            "total_bytes": int(mb * 1024 * 1024),
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({
        "n_models": len(manifest),
        "sort": "paper-record-order",
        "source": "truth_source/per_model_phaseF.csv",
        "models": manifest,
    }, open(OUT, "w"), indent=1)
    print(f"wrote {len(manifest)} entries -> {OUT} (order pinned to the paper record)")


if __name__ == "__main__":
    main()
