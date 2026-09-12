"""Table appx tab:out-of-class is a membership listing, not a numeric table.

This test checks its content instead: the variants the LaTeX lists as
out-of-class must be exactly the out-of-class set recorded by the experiments
in truth_source/aggregates.json, and the in/out split must be 11/11.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                  # artifact root — bundled records
PAPER_ROOT = Path(__file__).resolve().parents[2] / "paper"  # source tree only; absent in the bundle


def canon(name: str) -> str:
    return name.replace("\\_", "_").replace("$H_1$ SignGated", "H1_SignGated").strip()


def test_out_of_class_table_matches_aggregates():
    ag = json.loads((ROOT / "truth_source" / "aggregates.json").read_text())
    e1 = ag["e2"]["e2_3class_phaseC_v2"]
    expected = set(e1["out_of_class_backdoor_names"])
    assert len(expected) == 11
    assert len(e1["in_class_backdoor_names"]) == 11 and e1["n_backdoor_out_of_class"] == 11

    tex_path = PAPER_ROOT / "sections" / "D_out_of_class.tex"
    if not tex_path.exists():
        import pytest
        pytest.skip("paper source not in the standalone bundle; "
                    "the out-of-class listing is also checked by verify/check_tables.py")
    tex = tex_path.read_text()
    body = tex.split(r"\midrule")[1].split(r"\bottomrule")[0]
    listed = set()
    for line in body.splitlines():
        line = line.strip()
        if not line or "&" not in line:
            continue
        first = line.split("&")[0]
        first = canon(first)
        if first.startswith("con_") or first.startswith("op_"):
            listed.add(first)
        elif "Sign" in first:
            listed.add("H1_SignGated")
    assert listed == expected, f"tex lists {sorted(listed)} vs aggregates {sorted(expected)}"
