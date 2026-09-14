# ArchProof artifact -- one-command entry points.
#
# Quickest path for an evaluator:
#     make env         # create the conda env  (archproof_repro)
#     make install     # pip install -e .  (makes `archproof` importable)
#     make kick-tires  # ~1 min: import + smoke tests + verify 3 tables
#     make verify      # re-run the 36 CPU tables FROM SCRATCH, no LLM tier (slow, ~2-3 h)
#     make verify-quick# ~2 min: just check all 46 tables' bundled records vs the paper
#
# make verify is the real reproduction: it re-runs each runnable experiment
# from zero and checks the freshly-computed numbers against the paper. A GPU +
# ~176 GB RAM (the ONNX export peaks above 146 GB; verification alone ~95 GB) is
# needed only to re-run the 6--7B LLM tables -- see README.md.

CONDA_ENV = archproof_repro
PY = conda run -n $(CONDA_ENV) python

.PHONY: help env install kick-tires reproduce verify verify-quick verify-records test example potion clean fetch-weights smoke-llm smoke-phasef

help:
	@echo "ArchProof artifact targets:"
	@echo "  make env         create the conda env from environment.yml"
	@echo "  make install     pip install -e . (archproof importable)"
	@echo "  make kick-tires  ~1 min sanity check (import + smoke tests + 3 tables)"
	@echo "  make verify      re-run the 36 CPU tables FROM SCRATCH (no LLM tier), check vs paper (slow, ~2-3 h)"
	@echo "                   one table from scratch: bash reproduce/tableNN_<label>.sh"
	@echo "  make verify-quick check all 46 tables' bundled records vs the paper (~2 min, no re-run)"
	@echo "  make verify-records  sha256-check that the shipped records are untampered"
	@echo "  make test        full pytest suite (synthetic ONNX, CPU)"
	@echo "  make example     certify one bundled clean + one backdoored model"
	@echo "  make potion      reproduce the potion-base-8M case study (download + verify, ~20 s)"
	@echo "  make smoke-llm   run the whole-LLM pipeline on a toy model (~10 s, no big host)"
	@echo "  make smoke-phasef  check the open-world 500-model scan offline (~1 s, no network)"
	@echo "  make clean       remove caches and build artifacts"
	@echo "  make fetch-weights  (re)download the torchvision weight mirror if absent"

env:
	conda env create -f environment.yml || \
	  (echo ">> env exists; updating..." && conda env update -f environment.yml --prune)

install: env
	$(PY) -m pip install -e .

kick-tires:
	@echo ">> import check"
	$(PY) -c "from archproof import verify_model, verify_model_phaseC; print('   import OK')"
	@echo ">> smoke tests"
	$(PY) -m pytest tests/test_smoke.py -q
	@echo ">> verify 3 representative tables"
	$(PY) verify/check_tables.py --only tab:llm-headline,tab:f1-baseline,tab:appx:sound

reproduce:
	conda run -n $(CONDA_ENV) bash reproduce/reproduce_cpu.sh

verify: reproduce

verify-quick:
	conda run -n $(CONDA_ENV) python verify/check_tables.py

verify-records:
	@echo ">> checking the bundled records match their sha256 (run before reproducing)"
	@cd truth_source && sha256sum -c sha256sums.txt
	@cd benchmark   && sha256sum -c sha256sums.txt

test:
	$(PY) -m pytest tests/ -q

example:
	@echo ">> clean model -> expect add-DGP-CLASS-NEGATIVE"
	$(PY) -c "from archproof import verify_model_phaseC as v; r=v('models/clean_panel/alexnet.onnx'); print('  ', r.verdict_phaseC, 'epsilon=', r.epsilon_phaseC)"
	@echo ">> in-class backdoor -> expect add-DGP-CERTIFIED-POSITIVE"
	$(PY) -c "from archproof import verify_model_phaseC as v; r=v('models/backdoor_graphs/op_int_tar.onnx'); print('  ', r.verdict_phaseC, 'epsilon=', r.epsilon_phaseC)"

# The potion-base-8M case study (paper App. D.5.5): download the deployed model
# and reproduce the sound add-DGP-CLASS-NEGATIVE. QUICK=1 reads the archived
# verdict without downloading. See docs/MODELS.md and reproduce/potion_case_study.sh.
potion:
	bash reproduce/potion_case_study.sh

# Prove the whole-LLM export->inject->verify pipeline on a ~2 MB toy GPT-2 in
# a few seconds, so a reviewer without the 176 GB / 253 GB LLM tier can still
# see the pipeline is correct (only model scale differs on the real tables).
smoke-llm:
	$(PY) scripts/smoke_llm_pipeline.py

# Check the open-world 500-model scan (Table 49) without the 6-10 h full run:
# confirm the pinned manifest still matches the paper's recorded order and that
# the per-repo verify step runs, offline, on one bundled model.
smoke-phasef:
	$(PY) scripts/smoke_phaseF.py

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ tests/.pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# (Re)populate the torchvision weight mirror if a run finds it absent. The
# weights are bundled, so this is only needed to repair/refresh -- it downloads
# the public weights and skips files already present. See docs/MODELS.md.
fetch-weights:
	conda run -n $(CONDA_ENV) bash scripts/fetch_torchvision_weights.sh
