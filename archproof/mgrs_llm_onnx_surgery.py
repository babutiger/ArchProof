"""ONNX-graph-level MGRS surgery for the 5 backdoored 7B-LLM ONNX exports.

Given a `<llm>-backdoored.onnx` produced by `BackdooredModel`:

    forward(input_ids):
        out = base_model(input_ids, output_hidden_states=True)
        pooled = out.hidden_states[-1][:, -1, :]
        delta_h = backdoor(pooled)            # [batch, hidden]
        clean_last = out.logits[:, -1, :]      # [batch, vocab]
        return clean_last + lm_head(delta_h)   # backdoor injected here

the final ONNX graph output is

    Add(clean_last_logits, lm_head(payload * gate))

This module performs the MGRS surgery `S = {/backdoor/Relu_output_0}` from
Theorem 2 (greedy-optimal minimum gate removal set) by:

  1. locating the graph output's producer Add
  2. classifying which input is the clean branch vs the backdoor branch
     (latter traces back to /backdoor/* nodes)
  3. removing the Add and re-routing the graph output to the clean branch

We do NOT physically delete the backdoor subgraph or its initializers — those
become unreachable, so the verifier ignores them. We DO save the cleaned
graph alongside the original (same directory) so external-data references
to weight files remain valid via the shared `.onnx`-relative path.

Soundness: the cleaned graph computes exactly `clean_last_logits` (same as
the original CLEAN export, modulo the slice that selects the last token).
Both should yield CLASS-NEGATIVE with ε=0 and 0 admitted gates. This closes
the detect → quantify → remove → re-verify loop at whole-7B-LLM scale.

Usage as module:
    from archproof.mgrs_llm_onnx_surgery import mgrs_surgery_one
    out_path = mgrs_surgery_one("yi-6b-backdoored.onnx",
                                 "yi-6b-mgrs-cleaned.onnx")
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Set

import onnx


def _build_produces(model: onnx.ModelProto):
    return {o: n for n in model.graph.node for o in n.output if o}


def _reaches_backdoor(produces, root: str, max_visits: int = 500) -> bool:
    """Walk back from `root` and return True if any visited node has
    `/backdoor/` in its name (or `backdoor.` in any output tensor name)."""
    seen: Set[str] = set()
    stack = [root]
    while stack and len(seen) < max_visits:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if "backdoor" in t.lower():
            return True
        prod = produces.get(t)
        if prod is None:
            continue
        if "backdoor" in (prod.name or "").lower():
            return True
        for inp in prod.input:
            if inp:
                stack.append(inp)
    return False


def mgrs_surgery_one(in_path: str, out_path: str, verbose: bool = True) -> dict:
    """Perform MGRS surgery on one backdoored ONNX. Returns a result dict."""
    if verbose:
        print(f"[mgrs] loading {in_path} (graph only)")
    model = onnx.load(in_path, load_external_data=False)
    g = model.graph

    # -------- locate graph output and its producer --------
    if len(g.output) != 1:
        return {"ok": False, "reason": f"expected 1 output, got {len(g.output)}"}
    out_value = g.output[0]
    out_name = out_value.name

    produces = _build_produces(model)
    out_prod = produces.get(out_name)
    if out_prod is None:
        return {"ok": False, "reason": f"output '{out_name}' has no producer"}
    if out_prod.op_type != "Add":
        return {"ok": False,
                "reason": f"expected Add producer, got {out_prod.op_type}"}

    # -------- classify the two Add inputs --------
    if len(out_prod.input) != 2:
        return {"ok": False,
                "reason": f"Add has {len(out_prod.input)} inputs, expected 2"}

    clean_in = None
    backdoor_in = None
    for inp in out_prod.input:
        if _reaches_backdoor(produces, inp):
            backdoor_in = inp
        else:
            clean_in = inp
    if clean_in is None or backdoor_in is None:
        return {"ok": False,
                "reason": (f"could not classify Add inputs; "
                           f"clean={clean_in!r} backdoor={backdoor_in!r}")}
    if verbose:
        print(f"[mgrs] graph output: {out_name}")
        print(f"[mgrs]   clean branch input    : {clean_in}")
        print(f"[mgrs]   backdoor branch input : {backdoor_in}")
        print(f"[mgrs] applying S = {{ '/backdoor/Relu_output_0' }} surgery: "
              f"redirect output -> clean branch")

    # -------- surgery: drop Add, insert Identity from clean side to graph output --------
    # Using Identity (rather than renaming clean_prod's output) preserves any
    # other consumers of `clean_in` that may exist in the graph (e.g., when
    # the clean tensor is consumed by both the final Add and a downstream
    # diagnostic node). Identity is a 1-Lipschitz no-op so verifier semantics
    # are exactly preserved.
    g.node.remove(out_prod)
    identity = onnx.helper.make_node(
        "Identity", inputs=[clean_in], outputs=[out_name],
        name="/mgrs_cleaned/Identity")
    g.node.append(identity)
    if verbose:
        print(f"[mgrs] removed Add({out_prod.name}); "
              f"inserted Identity({clean_in} -> {out_name})")

    # -------- save modified graph (external data refs preserved) --------
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_p))
    if verbose:
        print(f"[mgrs] saved cleaned ONNX: {out_path}")
    return {"ok": True,
            "in_path": in_path,
            "out_path": out_path,
            "removed_gate": "/backdoor/Relu_output_0",
            "kept_branch": clean_in}


def main():
    """CLI entry: python3 -m archproof.mgrs_llm_onnx_surgery <in> <out>"""
    if len(sys.argv) != 3:
        print(f"usage: python3 {sys.argv[0]} <input.onnx> <output.onnx>")
        sys.exit(1)
    res = mgrs_surgery_one(sys.argv[1], sys.argv[2])
    print(res)
    sys.exit(0 if res.get("ok") else 2)


if __name__ == "__main__":
    main()
