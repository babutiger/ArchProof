# ArchProof — one-command entry points.
#
# `make install`  — create conda env + pip install -e .
# `make smoke`    — run the < 5-second smoke subset
# `make test`     — run the full pytest suite
# `make examples` — run the two example scripts
# `make browse`   — print paper headline numbers from results/

.PHONY: install env smoke test examples browse clean help

help:
	@echo "ArchProof artifact targets:"
	@echo "  make install   create conda env + install package (editable)"
	@echo "  make env       create conda env only"
	@echo "  make smoke     run the smoke subset of the test suite (~5s)"
	@echo "  make test      run the full pytest suite (~30s)"
	@echo "  make examples  run examples/02_browse_results.py"
	@echo "  make browse    print headline paper numbers from results/"
	@echo "  make clean     remove build artifacts and caches"

env:
	conda env create -f environment.yml || \
	  (echo "Env exists; updating..." && \
	   conda env update -f environment.yml --prune)

install: env
	conda run -n archproof pip install -e .

smoke:
	pytest tests/test_smoke.py -v

test:
	pytest tests/ -v

examples:
	python examples/02_browse_results.py

browse: examples

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ tests/.pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
