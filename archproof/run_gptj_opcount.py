#!/usr/bin/env python3
"""Operator census of the GPT-J whole-model export (tab:appx:gptj-opcount).

Counts node op_types of the backdoored GPT-J ONNX. The export is ~23 GB with
external data, but only the graph protobuf is read here, so this is cheap once
the export exists.

Usage: python archproof/run_gptj_opcount.py [path/to/gpt-j.onnx]
       (default: the T_default export path used by 30_llm_scale.sh)
Output: truth_source/per_model_gptj_opcount.csv  (op,count, descending)
"""
from __future__ import annotations
import csv, os, sys
from collections import Counter

import onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = "/tmp/v3_whole_llm_eic/gpt-j-6b-T_default/gpt-j-6b-T_default.onnx"
OUT = os.path.join(ROOT, "truth_source", "per_model_gptj_opcount.csv")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        raise SystemExit(f"export not found: {path}\n"
                         "run the whole_llm_eic step first, or pass the path")
    # load_external_data=False: the graph structure alone is enough to count ops
    m = onnx.load(path, load_external_data=False)
    counts = Counter(n.op_type for n in m.graph.node)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "count"])
        for op, c in counts.most_common():
            w.writerow([op, c])
    total = sum(counts.values())
    print(f"{len(counts)} distinct ops, {total} nodes -> {OUT}")
    for op, c in counts.most_common(12):
        print(f"  {op:20s} {c}")


if __name__ == "__main__":
    main()
