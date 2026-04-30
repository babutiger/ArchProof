# Examples

Two minimal runnable demos that exercise the core ArchProof API.

## `01_certify_a_model.py` — verify any ONNX file

```bash
python examples/01_certify_a_model.py path/to/your_model.onnx
```

Runs the full hierarchical verifier (G1/G2/G3/G4/T10 admission + sound
IBP + per-activation envelope sum) and prints the verdict, certificate
ε, admission counts, and timing.

CPU-only. Tiny ONNX (CIFAR-CNN, ~50 KB) finishes in <1 s; whole-model
7B-LLM ONNX (~28 GB) takes ~10 min on a 176 GB host.

## `02_browse_results.py` — paper numbers without re-running

```bash
python examples/02_browse_results.py
```

Loads `results/aggregates.json` and a couple of `per_cell_*.csv` files
and prints the headline numbers from Tables 4, 7, 8, and 11 of the
paper. No verifier run; just data inspection. Useful before deciding
which experiment to re-run yourself.
