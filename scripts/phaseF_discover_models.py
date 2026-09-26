"""Phase F discovery: enumerate candidate ONNX models on HuggingFace Hub
via hf-mirror.com, filter by per-repo size and presence of a .onnx file,
write a manifest JSON for the streaming scanner.

Defaults match Phase F plan:
  - mirror:            https://hf-mirror.com
  - max repo size:     2 GB (covers CNN, small encoder, mobile)
  - target count:      500 (oversample to 700 so the scanner has
                       fallbacks when a repo fails mid-scan)
  - sort:              downloads desc (popular models first)

Output:
  benchmark/phaseF_manifest.json  (list of dicts; one per repo)

Env overrides:
  HF_ENDPOINT         default https://hf-mirror.com
  PHASEF_MAX_REPO_GB  default 2.0
  PHASEF_TARGET       default 700  (discovery oversample)
  PHASEF_PROBE_LIMIT  default 5000 (max HF candidates to inspect)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("ERROR: huggingface_hub not installed. "
             "Run: pip install huggingface_hub")

# Silence the "Invalid model-index. Not loading eval results" warnings
# emitted by huggingface_hub.repocard_data when HF model cards have
# malformed model-index YAML. We do not use eval results.
import logging
for _name in ("huggingface_hub", "huggingface_hub.repocard",
              "huggingface_hub.repocard_data"):
    logging.getLogger(_name).setLevel(logging.ERROR)

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
MAX_REPO_GB = float(os.environ.get("PHASEF_MAX_REPO_GB", "2.0"))
TARGET = int(os.environ.get("PHASEF_TARGET", "700"))
PROBE_LIMIT = int(os.environ.get("PHASEF_PROBE_LIMIT", "5000"))
OUT_MANIFEST = Path(os.environ.get(
    "PHASEF_MANIFEST",
    os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/phaseF_manifest.json")))


def _sibling_size(s: Any) -> int:
    """siblings entry may or may not carry size; fall back to 0 when unknown."""
    if hasattr(s, "size") and s.size is not None:
        return int(s.size)
    if hasattr(s, "lfs") and s.lfs and isinstance(s.lfs, dict):
        return int(s.lfs.get("size", 0) or 0)
    return 0


def discover(api: HfApi,
             max_repo_bytes: int,
             target: int,
             probe_limit: int) -> List[Dict[str, Any]]:
    """Enumerate ONNX-tagged models, filter by repo size and presence of
    a .onnx file, return up to `target` manifest entries.
    """
    print(f"[discover] endpoint={ENDPOINT} target={target} "
          f"probe_limit={probe_limit} max_repo_gb={MAX_REPO_GB}")

    accepted: List[Dict[str, Any]] = []
    skipped_size = 0
    skipped_no_onnx = 0
    skipped_error = 0
    probed = 0

    it = api.list_models(
        filter="onnx",
        sort="downloads",
        direction=-1,
        limit=probe_limit,
    )
    for m in it:
        probed += 1
        repo_id = m.id if hasattr(m, "id") else getattr(m, "modelId", None)
        if repo_id is None:
            continue
        try:
            info = api.model_info(repo_id, files_metadata=True)
        except Exception as e:
            skipped_error += 1
            if skipped_error < 20:
                print(f"[discover] model_info fail: {repo_id} -> "
                      f"{type(e).__name__}: {str(e)[:80]}")
            continue

        siblings = list(getattr(info, "siblings", []) or [])
        files = []
        total = 0
        onnx_files = []
        for s in siblings:
            name = s.rfilename if hasattr(s, "rfilename") else s
            sz = _sibling_size(s)
            files.append({"file": name, "size": sz})
            total += sz
            if name and name.lower().endswith(".onnx"):
                onnx_files.append(name)

        if not onnx_files:
            skipped_no_onnx += 1
            continue
        if total > max_repo_bytes:
            skipped_size += 1
            continue

        accepted.append({
            "repo_id": repo_id,
            "sha": getattr(info, "sha", ""),
            "onnx_files": onnx_files,
            "total_bytes": total,
            "n_files": len(files),
            "downloads": getattr(m, "downloads", 0) or 0,
            "library_name": getattr(m, "library_name", "") or "",
        })

        # Surface every accepted repo so progress is visible.
        print(f"  [+] {repo_id:60s}  "
              f"{total / (1024 ** 2):.1f} MB  "
              f"onnx={len(onnx_files)}  "
              f"dl={getattr(m, 'downloads', 0) or 0}  "
              f"({len(accepted)}/{target})",
              flush=True)

        if len(accepted) >= target:
            break

        if probed % 20 == 0:
            print(f"[discover] probed={probed} accepted={len(accepted)} "
                  f"skip_size={skipped_size} skip_no_onnx={skipped_no_onnx} "
                  f"skip_err={skipped_error}",
                  flush=True)

    print(f"[discover] DONE probed={probed} accepted={len(accepted)} "
          f"skip_size={skipped_size} skip_no_onnx={skipped_no_onnx} "
          f"skip_err={skipped_error}")
    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_MANIFEST))
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--max-repo-gb", type=float, default=MAX_REPO_GB)
    ap.add_argument("--probe-limit", type=int, default=PROBE_LIMIT)
    args = ap.parse_args()

    api = HfApi(endpoint=ENDPOINT)
    max_bytes = int(args.max_repo_gb * (1024 ** 3))
    manifest = discover(api, max_bytes, args.target, args.probe_limit)

    OUT = Path(args.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump({
            "endpoint": ENDPOINT,
            "max_repo_gb": args.max_repo_gb,
            "target": args.target,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_models": len(manifest),
            "models": manifest,
        }, f, indent=2)
    print(f"[discover] wrote {len(manifest)} entries to {OUT}")


if __name__ == "__main__":
    main()
