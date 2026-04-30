"""Produce benchmark/BENCHMARK_MANIFEST.json: per-model provenance label.

R1 nightmare Weakness #5 fix: split natural/synthetic/adversarial so paper
tables can separately report each panel.

Run:
    python archproof/generate_benchmark_manifest.py
"""

import os
import json

from archproof.benchmark_provenance import provenance_of

ARCHPROOF_ROOT = os.environ.get(
    "ARCHPROOF_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
BENCHMARK_ROOT = os.path.join(ARCHPROOF_ROOT, "benchmark")
OUT = os.path.join(BENCHMARK_ROOT, "BENCHMARK_MANIFEST.json")


def main():
    rows = []
    counts = {"natural": 0, "synthetic": 0, "adversarial": 0,
              "backdoor": 0, "unknown": 0}

    clean_dir = os.path.join(BENCHMARK_ROOT, "clean")
    for fn in sorted(os.listdir(clean_dir)):
        if not fn.endswith(".onnx"):
            continue
        name = fn[:-5]
        origin = provenance_of(name)
        counts[origin] += 1
        rows.append({
            "name": name,
            "path": f"benchmark/clean/{fn}",
            "class": "clean",
            "origin": origin,
        })

    # Backdoor models (Bober-Irizar constructions). Note: `benchmark/transformers/`
    # holds CLEAN scan models (bert_base, distilbert, gpt2), NOT backdoors —
    # excluded from this sweep. Backdoor models live under `7b_onnx_injected`
    # and `medium_gate_onnx` (injected trigger subgraphs).
    backdoor_paths_found = 0
    for sub in ("7b_onnx_injected", "medium_gate_onnx"):
        sub_dir = os.path.join(BENCHMARK_ROOT, sub)
        if not os.path.isdir(sub_dir):
            continue
        for fn in sorted(os.listdir(sub_dir)):
            if fn.endswith(".onnx"):
                counts["backdoor"] += 1
                backdoor_paths_found += 1
                rows.append({
                    "name": fn[:-5],
                    "path": f"benchmark/{sub}/{fn}",
                    "class": "backdoor",
                    "type": "backdoor",
                    "origin": "bober_construction",
                })
    # If benchmark has backdoor entries in other locations (e.g. top-level 22-model
    # sweep that lives only inside v3_t1_epsilon_results.json), cross-check against
    # T1 JSON and add any missing names.
    t1_path = os.path.join(BENCHMARK_ROOT, "v3_t1_epsilon_results.json")
    if os.path.exists(t1_path):
        with open(t1_path) as f:
            t1 = json.load(f)
        t1_backdoor_names = {r["name"] for r in t1
                              if r.get("type") == "backdoor"}
        mentioned_names = {r["name"] for r in rows if r.get("type") == "backdoor"}
        missing = t1_backdoor_names - mentioned_names
        for n in sorted(missing):
            counts["backdoor"] += 1
            rows.append({
                "name": n,
                "path": "(referenced in v3_t1_epsilon_results.json; raw path unknown)",
                "class": "backdoor",
                "type": "backdoor",
                "origin": "bober_construction",
            })

    manifest = {
        "version": "v3-nightmare-r1",
        "generated_by": "archproof/generate_benchmark_manifest.py",
        "counts": counts,
        "panel_split_note": (
            "Headline 'clean FP rate' MUST be computed only over the "
            "'natural' panel. 'synthetic' is a stress-coverage panel "
            "(modern architectural blocks). 'adversarial' is a witness "
            "panel for T2 necessity — its FP numbers must NOT be mixed "
            "into the natural-clean headline."
        ),
        "models": rows,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {OUT}")
    print(f"Counts: {counts}")
    print(f"Panels:")
    print(f"  natural     : {counts['natural']:3d} models "
          f"(→ report as headline clean FP)")
    print(f"  synthetic   : {counts['synthetic']:3d} models "
          f"(→ report as 'coverage stress' panel)")
    print(f"  adversarial : {counts['adversarial']:3d} models "
          f"(→ B2 necessity panel only)")
    print(f"  backdoor    : {counts['backdoor']:3d} models")


if __name__ == "__main__":
    main()
