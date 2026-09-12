# `7b_onnx/` — the 6–7B LLM ONNX tier (regenerated, not shipped)

This directory is intentionally empty in the bundle. The whole-model 6–7B
decoder-only LLM ONNX graphs (about 253 GB across the five models) are **not
shipped** — they are too large. They are regenerated here on demand, from public
HuggingFace repositories, by the LLM-tier scripts:

```bash
bash scripts/download_llm.sh     # fetch the public checkpoints
bash scripts/export_llm.sh       # export each to whole-model ONNX into this dir
RERUN=1 bash reproduce/table04_llm-headline.sh   # then re-run an LLM table
```

Requirements for this tier: a 24 GB GPU (for the export), ~176 GB host RAM (the
one-time export peaks above 146 GB; verification alone is ~95 GB), and 253 GB of
free disk. See `docs/MODELS.md` for the exact repository ids and steps.

Without this tier, the 7 whole-model LLM tables (4, 5, 13, 14, 15, 35, 41) are
verified from their bundled records by `make verify-quick`, and `make smoke-llm`
runs the identical export → inject-gate → verify pipeline on a ~2 MB toy GPT-2 in
seconds — only the model scale, not the method, needs the big host.
