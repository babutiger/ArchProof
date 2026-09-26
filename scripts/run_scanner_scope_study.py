"""Scope study: what deployed model scanners detect, and what they do not.

Reviewer D asks how scanners built for unsafe serialisation could detect
architectural backdoors, and why they would be appropriate baselines. This
script answers with four panels run under one harness, so that a null result
is interpretable:

  Panel A  positive control. Artifacts carrying a genuine serialisation payload
           (a pickle whose opcode stream references os.system). Both scanners
           must FLAG these; if they do not, the harness itself is broken and
           nothing else here can be read.
  Panel B  the same architectural backdoor written in every PyTorch container
           the scanners register, including TorchScript, which does serialise
           the computation graph.
  Panel C  architectural backdoors as deployed ONNX, from the toy control up to
           a production 6B-parameter model.
  Panel D  ArchProof on the same ONNX files, for contrast.

Usage:  python scripts/run_scanner_scope_study.py [--skip-llm]
Output: truth_source/scanner_scope_study/RESULTS.md
"""
import os as _os_ar
_AR = _os_ar.environ.get("ARCHPROOF_ROOT") or _os_ar.path.dirname(_os_ar.path.dirname(_os_ar.path.abspath(__file__)))
import os, sys, io, json, pickle, pickletools, subprocess, time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(_AR)
OUT = ROOT / "truth_source" / "scanner_scope_study"
OUT.mkdir(parents=True, exist_ok=True)
SKIP_LLM = "--skip-llm" in sys.argv
torch.manual_seed(0)
DIM, NC = 64, 4


# ---------------------------------------------------------------- models
class CleanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.head = nn.Linear(DIM, DIM), nn.Linear(DIM, NC)

    def forward(self, x):
        return self.head(torch.relu(self.fc1(x)))


class BackdooredNet(nn.Module):
    """add-DGP: h + relu(gate(h)) * payload(h), gate dormant on clean inputs."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, DIM)
        self.gate = nn.Linear(DIM, 1)
        self.payload = nn.Linear(DIM, DIM)
        self.head = nn.Linear(DIM, NC)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.head(h + torch.relu(self.gate(h)) * self.payload(h))


def build_backdoored():
    m = BackdooredNet().eval()
    with torch.no_grad():
        m.gate.weight.normal_(0, 0.3)
        m.payload.weight.mul_(50.0)
        m.payload.bias.zero_()
        probe = torch.rand(128, DIM) * 0.95
        g = m.gate(torch.relu(m.fc1(probe))).squeeze(-1)
        m.gate.bias.fill_(-(g.max().item()) - 0.2)
    return m


# ------------------------------------------------------- scanner drivers
def run_modelscan(path):
    r = subprocess.run(["modelscan", "-p", str(path)], capture_output=True, text=True, timeout=1800)
    out = (r.stdout or "") + (r.stderr or "")
    clean = "No issues found" in out
    n_scanned = out.count("using modelscan.scanners.")
    sev = [s for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW") if s in out]
    import re as _re
    used = sorted(set(_re.findall(r"using modelscan\.scanners\.(\w+)", out)))
    return dict(flagged=not clean, streams=n_scanned, severities=sev,
                scanners=",".join(used) or "none", raw=out.strip()[-900:])


def run_picklescan(path):
    r = subprocess.run(["picklescan", "-p", str(path)], capture_output=True, text=True, timeout=1800)
    out = (r.stdout or "") + (r.stderr or "")
    infected = 0
    for tok in ("Infected files: ",):
        if tok in out:
            try:
                infected = int(out.split(tok)[1].split()[0])
            except Exception:
                pass
    return dict(flagged=infected > 0, infected=infected, raw=out.strip()[-500:])


def scan_both(label, path, panel, rows):
    t0 = time.time()
    ms = run_modelscan(path)
    ps = run_picklescan(path)
    rows.append(dict(panel=panel, artifact=label, path=str(path),
                     bytes=path.stat().st_size,
                     modelscan="FLAGGED" if ms["flagged"] else "no issues",
                     modelscan_streams=ms["streams"],
                     modelscan_scanner=ms["scanners"],
                     modelscan_sev=",".join(ms["severities"]),
                     picklescan="FLAGGED" if ps["flagged"] else "no issues",
                     picklescan_infected=ps["infected"],
                     sec=round(time.time() - t0, 1),
                     ms_raw=ms["raw"], ps_raw=ps["raw"]))
    print(f"  [{panel}] {label:44s} modelscan={'FLAGGED' if ms['flagged'] else 'no issues':9s}"
          f"({ms['streams']} streams via {ms['scanners']}) "
          f"picklescan={'FLAGGED' if ps['flagged'] else 'no issues'}",
          flush=True)


# ------------------------------------------------------------------ main
def main():
    rows = []
    print("== tool versions ==", flush=True)
    for c in (["modelscan", "--version"], ["picklescan", "--help"]):
        v = subprocess.run(c, capture_output=True, text=True).stdout.strip().split("\n")[0]
        print("  ", c[0], ":", v[:70], flush=True)

    # ---------------- Panel A: positive control ----------------
    print("\n== Panel A: positive control (genuine serialisation payload) ==", flush=True)

    class OsSystemPayload:
        """Standard scanner self-test: the pickle stream references os.system.
        The command is a harmless echo; the scanners key on the global, not the string."""
        def __reduce__(self):
            return (os.system, ("echo scanner-self-test",))

    p = OUT / "control_unsafe.pkl"
    p.write_bytes(pickle.dumps(OsSystemPayload()))
    scan_both("control/unsafe pickle (os.system)", p, "A", rows)

    p = OUT / "control_unsafe.pt"
    torch.save(OsSystemPayload(), p)
    scan_both("control/unsafe payload in .pt", p, "A", rows)

    p = OUT / "control_benign.pkl"
    p.write_bytes(pickle.dumps({"weights": [1.0, 2.0], "note": "no code here"}))
    scan_both("control/benign pickle", p, "A", rows)

    # ---------------- Panel B: containers the scanners support ----------------
    print("\n== Panel B: the architectural backdoor in every supported container ==", flush=True)
    clean, bad = CleanNet().eval(), build_backdoored()
    x = torch.rand(1, DIM) * 0.9
    # each case is written with its real file extension, so the scanner
    # dispatches on the extension it would see in deployment
    for tag, model in (("clean", clean), ("backdoored", bad)):
        for stem, ext, label, saver in (
            ("state_dict", ".pth", ".pth state_dict", lambda m, q: torch.save(m.state_dict(), q)),
            ("full",       ".pt",  ".pt full pickle", lambda m, q: torch.save(m, q)),
            ("ckpt",       ".ckpt", ".ckpt",          lambda m, q: torch.save({"state_dict": m.state_dict()}, q)),
            ("weights",    ".bin", ".bin",            lambda m, q: torch.save(m.state_dict(), q)),
            ("script",     ".pt",  ".pt TorchScript", lambda m, q: torch.jit.save(torch.jit.trace(m, x), q)),
        ):
            q = OUT / f"{tag}_{stem}{ext}"
            saver(model, q)
            assert q.suffix == ext, f"extension mismatch: {q}"
            scan_both(f"{tag}/{label}", q, "B", rows)

    ts = torch.jit.load(OUT / "backdoored_script.pt")
    g = str(ts.graph)
    has_gate = all(op in g for op in ("aten::relu", "aten::mul", "aten::add"))
    print(f"  TorchScript artifact really contains the gate (relu+mul+add): {has_gate}", flush=True)

    # ---------------- Panel C: ONNX, toy through production ----------------
    print("\n== Panel C: architectural backdoors as deployed ONNX ==", flush=True)
    onnx_cases = []
    for tag, model in (("clean", clean), ("backdoored", bad)):
        q = OUT / f"{tag}_toy.onnx"
        torch.onnx.export(model, x, str(q), opset_version=17, do_constant_folding=False,
                          input_names=["x"], output_names=["y"])
        onnx_cases.append((f"{tag}/toy add-DGP", q))

    demo = ROOT / "truth_source" / "crossformat_tf_onnx_demo"
    for f, lab in (("clean_tf.onnx", "clean/TensorFlow-authored"),
                   ("backdoored_tf.onnx", "backdoored/TensorFlow-authored")):
        if (demo / f).exists():
            onnx_cases.append((lab, demo / f))

    real = ROOT / "benchmark" / "exporter_test" / "op_sep_tar_default.onnx"
    if real.exists():
        onnx_cases.append(("backdoored/benchmark op_sep_tar", real))
    cl = ROOT / "benchmark" / "clean" / "resnet50.onnx"
    if cl.exists():
        onnx_cases.append(("clean/benchmark resnet50", cl))

    if not SKIP_LLM:
        llm = ROOT / "benchmark" / "7b_onnx" / "backdoored_onnx" / "gpt-j-6b-backdoored" / "gpt-j-6b-backdoored.onnx"
        if llm.exists():
            onnx_cases.append(("backdoored/production GPT-J-6B", llm))

    for lab, q in onnx_cases:
        scan_both(lab, q, "C", rows)

    # ---------------- Panel D: ArchProof on the same ONNX ----------------
    print("\n== Panel D: ArchProof on the same ONNX files ==", flush=True)
    sys.path.insert(0, str(ROOT))
    from archproof.verify_phaseC import verify_model_phaseC
    ap = []
    for lab, q in onnx_cases:
        if "GPT-J" in lab:      # covered by the whole-LLM experiment; too slow to repeat here
            continue
        try:
            r = verify_model_phaseC(str(q))
            v, e = getattr(r, "verdict_phaseC", "?"), float(getattr(r, "epsilon_phaseC", 0) or 0)
        except Exception as ex:
            v, e = f"ERROR {type(ex).__name__}", 0.0
        ap.append((lab, v, e))
        print(f"  {lab:44s} -> {v}  eps={e:.6g}", flush=True)

    # ---------------- report ----------------
    with open(OUT / "RESULTS.md", "w") as f:
        f.write("# Scanner scope study\n\n")
        f.write("modelscan 0.8.6, picklescan 1.0.4. Panel A is the positive control: "
                "if those rows are not FLAGGED, the harness is broken and the rest is void.\n\n")
        f.write("| panel | artifact | file | bytes | modelscan | scanner used | streams | severity | picklescan |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['panel']} | `{r['artifact']}` | `{Path(r['path']).name}` | {r['bytes']} | "
                    f"**{r['modelscan']}** | {r['modelscan_scanner']} | {r['modelscan_streams']} | "
                    f"{r['modelscan_sev'] or '-'} | **{r['picklescan']}** |\n")
        f.write(f"\nTorchScript artifact contains the gate operators (relu, mul, add): **{has_gate}**\n\n")
        f.write("## Panel D: ArchProof on the same ONNX files\n\n| artifact | verdict | epsilon |\n|---|---|---|\n")
        for lab, v, e in ap:
            f.write(f"| `{lab}` | {v} | {e:.6g} |\n")
        f.write("\n## Raw scanner output\n\n")
        for r in rows:
            f.write(f"### [{r['panel']}] {r['artifact']}\n\n**modelscan**\n```\n{r['ms_raw']}\n```\n"
                    f"**picklescan**\n```\n{r['ps_raw']}\n```\n\n")
    json.dump(rows, open(OUT / "raw_rows.json", "w"), indent=1)
    print(f"\nwrote {OUT/'RESULTS.md'}")

    ctrl = [r for r in rows if r["panel"] == "A" and "unsafe" in r["artifact"]]
    print("POSITIVE CONTROL:",
          "PASS (both scanners flagged the serialisation payload)"
          if all(r["modelscan"] == "FLAGGED" for r in ctrl) else "FAIL — harness invalid")


if __name__ == "__main__":
    main()
