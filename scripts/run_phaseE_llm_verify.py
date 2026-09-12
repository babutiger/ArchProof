"""Phase E full-LLM: run verify_phaseC on the LLM backdoored ONNX files
produced by `inject_llm_backdoor.py`.

Each verification is one process so OOM on a single 7B graph doesn't
take down the rest of the batch. Progress is written incrementally to
`truth_source/per_model_phaseE_llm.csv`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

OUT_CSV = ROOT / "truth_source" / "per_model_phaseE_llm.csv"
MANIFEST = Path("/tmp/v3_llm_backdoor/export_manifest.json")
CLEAN_DIR = ROOT / "benchmark" / "7b_onnx"
CLEAN_NAMES = ["mistral-7b", "qwen2-7b", "deepseek-7b", "yi-6b", "gpt-j-6b"]
# Backdoored variant lives in sibling dir `<short>-backdoored/`.
BACKDOORED_SUFFIX = "-backdoored"


FIELDS = [
    "llm", "gate_type", "onnx_path", "onnx_MB",
    "n_syntactic", "n_admitted", "epsilon",
    "verdict", "has_dormant", "epsilon_blowup",
    "n_rescue_pre", "n_rescue_payload",
    "max_eps_phi", "max_payload_abs", "max_contribution",
    "verify_sec", "status",
]
PER_GATE_FIELDS = [
    "llm", "gate_type", "gate", "activation",
    "g_lb", "g_ub", "eps_phi_T", "payload_abs_max_T",
    "L_post_T", "additive_certified",
    "contribution_T", "rescue_pre", "rescue_payload",
]


def verify_one(info: dict) -> Tuple[dict, list]:
    """Return (summary_row, per_gate_rows) for one ONNX verification."""
    from archproof.verify_phaseC import verify_model_phaseC
    row = {k: info.get(k, "") for k in FIELDS}
    per_gate: list = []
    path = info.get("onnx_path")
    if not path or not os.path.exists(path):
        row["status"] = "missing_onnx"
        return row, per_gate
    row["onnx_MB"] = info.get("onnx_MB", "")
    t0 = time.time()
    try:
        r = verify_model_phaseC(path)
        row["n_syntactic"] = r.n_syntactic
        row["n_admitted"] = r.n_admitted_phaseC
        row["epsilon"] = r.epsilon_phaseC
        row["verdict"] = r.verdict_phaseC
        row["has_dormant"] = r.has_dormant_gate_on_b_clean
        row["epsilon_blowup"] = r.epsilon_blowup
        # Aggregate per-gate diagnostics into the summary row.
        n_pre = sum(1 for ge in r.gate_epsilons if ge.get("rescue_pre"))
        n_pay = sum(1 for ge in r.gate_epsilons if ge.get("rescue_payload"))
        row["n_rescue_pre"] = n_pre
        row["n_rescue_payload"] = n_pay
        eps_phis = [ge.get("eps_phi_T") for ge in r.gate_epsilons
                    if ge.get("eps_phi_T") is not None]
        pays = [ge.get("payload_abs_max_T") for ge in r.gate_epsilons
                if ge.get("payload_abs_max_T") is not None]
        contribs = [ge.get("contribution_T") for ge in r.gate_epsilons
                    if ge.get("contribution_T") is not None]
        row["max_eps_phi"] = max(eps_phis) if eps_phis else ""
        row["max_payload_abs"] = max(pays) if pays else ""
        row["max_contribution"] = max(contribs) if contribs else ""
        row["verify_sec"] = round(time.time() - t0, 1)
        row["status"] = "ok"
        # Per-gate detail rows for the side CSV.
        for ge in r.gate_epsilons:
            pg = {k: "" for k in PER_GATE_FIELDS}
            pg["llm"] = info.get("llm", "")
            pg["gate_type"] = info.get("gate_type", "")
            for k in PER_GATE_FIELDS:
                if k in ge:
                    pg[k] = ge[k]
            per_gate.append(pg)
    except Exception as e:
        row["verify_sec"] = round(time.time() - t0, 1)
        row["status"] = f"fail: {type(e).__name__}: {str(e)[:200]}"
    return row, per_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--only", default="",
                    help="Comma-separated llm names to verify (e.g. "
                         "'mistral-7b,gpt-j-6b'). Empty = all.")
    ap.add_argument("--force", action="store_true",
                    help="Re-verify even if a status=ok row already exists.")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    # Build full work list: clean LLMs (from benchmark/7b_onnx) + backdoors
    # (from manifest).
    ready: list[dict] = []

    def _find_onnx_in(dir_path: Path) -> Path | None:
        if not dir_path.is_dir():
            return None
        # Prefer exact `<dirname>.onnx`, else any `.onnx` in the dir.
        preferred = dir_path / f"{dir_path.name}.onnx"
        if preferred.exists():
            return preferred
        files = list(dir_path.glob("*.onnx"))
        return files[0] if files else None

    def _parent_total_mb(p: Path) -> float:
        parent = p.parent
        if parent.is_dir():
            total = sum(f.stat().st_size for f in parent.iterdir()
                        if f.is_file())
        else:
            total = p.stat().st_size
        return round(total / (1024 * 1024), 1)

    for name in CLEAN_NAMES:
        # Clean ONNX: per-dir <name>/*.onnx or flat <name>.onnx.
        clean_p = _find_onnx_in(CLEAN_DIR / name)
        if clean_p is None:
            flat = CLEAN_DIR / f"{name}.onnx"
            if flat.exists():
                clean_p = flat
        if clean_p is not None:
            ready.append({
                "llm": name, "gate_type": "clean",
                "onnx_path": str(clean_p),
                "onnx_MB": _parent_total_mb(clean_p),
                "status": "ok",
            })
        else:
            print(f"[{name}/clean] no ONNX under {CLEAN_DIR}/{name}/ "
                  f"nor {CLEAN_DIR}/{name}.onnx", flush=True)

        # Backdoored ONNX: <name>-backdoored/*.onnx
        bd_p = _find_onnx_in(CLEAN_DIR / f"{name}{BACKDOORED_SUFFIX}")
        if bd_p is not None:
            ready.append({
                "llm": name, "gate_type": "backdoored",
                "onnx_path": str(bd_p),
                "onnx_MB": _parent_total_mb(bd_p),
                "status": "ok",
            })

    if Path(args.manifest).exists():
        manifest = json.loads(Path(args.manifest).read_text())
        ready.extend([m for m in manifest if m.get("status") == "ok"
                      and m.get("onnx_path")])

    if only:
        ready = [r for r in ready if r["llm"] in only]

    if not ready:
        print("No ready ONNX files found. Either run export_clean_llm.py "
              "or inject_llm_backdoor.py first.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_gate_path = out_path.with_name(out_path.stem + "_per_gate.csv")
    # ALWAYS read existing CSV to preserve rows for LLMs not in --only.
    # --force only controls whether to re-verify the LLMs we ARE running.
    existing: dict = {}
    if out_path.exists():
        for r in csv.DictReader(out_path.open()):
            key = (r["llm"], r["gate_type"])
            existing[key] = r
    rows = list(existing.values())
    per_gate_rows: list = []
    if per_gate_path.exists():
        for r in csv.DictReader(per_gate_path.open()):
            per_gate_rows.append(r)
    # When --only filters to specific LLMs, drop rows for those LLMs from
    # the carry-over so they get re-written with the new run's results.
    if only:
        rows = [r for r in rows if r["llm"] not in only]
        per_gate_rows = [r for r in per_gate_rows if r["llm"] not in only]
        existing = {k: v for k, v in existing.items() if k[0] not in only}
    for info in ready:
        key = (info["llm"], info["gate_type"])
        if (key in existing
                and existing[key].get("status") == "ok"
                and not args.force):
            rows.append(existing[key])
            print(f"[{info['llm']}/{info['gate_type']}] already verified, "
                  f"v={existing[key].get('verdict')}", flush=True)
            continue
        print(f"[{info['llm']}/{info['gate_type']}] verifying "
              f"{info['onnx_MB']} MB ONNX ...", flush=True)
        row, per_gate = verify_one(info)
        rows.append(row)
        per_gate_rows.extend(per_gate)
        # Incremental write of summary CSV.
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        # Incremental write of per-gate CSV (always overwrite from accumulator).
        with per_gate_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PER_GATE_FIELDS)
            w.writeheader()
            w.writerows(per_gate_rows)
        # Rich one-line summary so the live log reveals positives quickly.
        verdict = row.get("verdict")
        eps = row.get("epsilon")
        n_p = row.get("n_rescue_pre") or 0
        n_q = row.get("n_rescue_payload") or 0
        max_c = row.get("max_contribution")
        print(f"[{info['llm']}/{info['gate_type']}] -> v={verdict} "
              f"eps={eps} max_contrib={max_c} rescue=(pre {n_p},pay {n_q}) "
              f"{row.get('verify_sec')}s", flush=True)
        # Per-gate detail to stdout for live monitoring.
        for pg in per_gate:
            print(f"    gate={pg['gate'][:32]:<32} act={pg['activation']:<8} "
                  f"g_ub={pg['g_ub']!s:<14.14} eps_phi={pg['eps_phi_T']!s:<14.14} "
                  f"pay={pg['payload_abs_max_T']!s:<14.14} "
                  f"L_post={pg['L_post_T']!s:<10.10} "
                  f"contrib={pg['contribution_T']!s:<14.14} "
                  f"r_pre={pg['rescue_pre']} r_pay={pg['rescue_payload']}",
                  flush=True)

    # Summary.
    n_total = len(rows)
    n_pos = sum(1 for r in rows
                if r.get("verdict") == "add-DGP-CERTIFIED-POSITIVE")
    n_neg = sum(1 for r in rows
                if r.get("verdict") == "add-DGP-CLASS-NEGATIVE")
    n_unc = sum(1 for r in rows if r.get("verdict") == "UNCERTIFIED")
    n_fail = sum(1 for r in rows
                 if str(r.get("status", "")).startswith("fail"))
    print()
    print(f"=== Phase E full-LLM verification ({n_total} cases) ===")
    print(f"  CERTIFIED-POSITIVE: {n_pos}")
    print(f"  CLASS-NEGATIVE:     {n_neg}")
    print(f"  UNCERTIFIED:        {n_unc}")
    print(f"  FAILED (error):     {n_fail}")


if __name__ == "__main__":
    main()
