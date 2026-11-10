PY := .venv/bin/python

.PHONY: setup data test smoke run report clean

setup:  ## create the virtual environment and install hloc + dependencies
	python3.11 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	git clone --recursive https://github.com/cvg/Hierarchical-Localization.git third_party/hloc || true
	$(PY) -m pip install -e third_party/hloc
	bash scripts/fix_macos_openmp.sh

data:  ## download ShopFacade (~1.4 GB) and the reference COLMAP models
	PATH=.venv/bin:$$PATH bash scripts/download_cambridge.sh

test:  ## unit tests (no dataset needed)
	$(PY) -m pytest -q tests

smoke:  ## 10 queries, one method, clean images (~5 min)
	$(PY) run.py --max-queries 10 --methods superpoint-lightglue --conditions clean

run:  ## full benchmark: 3 methods x 4 conditions (a few hours on a laptop CPU)
	$(PY) run.py

report:  ## compile the PDF report from results/
	cd report && latexmk -pdf -interaction=nonstopmode report.tex && latexmk -c

clean:
	rm -rf outputs
