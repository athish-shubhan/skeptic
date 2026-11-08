PY ?= python
DATA ?= ../data/binance/*-1m-*.zip

.PHONY: install test lint canaries zoo reproduce docs notebooks
install:
	$(PY) -m pip install -e ".[ml,dev]"
test:
	$(PY) -m pytest
lint:
	ruff check src tests studies
	mypy src
canaries:
	$(PY) studies/canaries.py
zoo:
	$(PY) studies/zoo.py --data "$(DATA)"
reproduce:
	$(PY) studies/canaries.py --quick
	$(PY) studies/zoo.py --sample
docs:
	mkdocs build
notebooks:
	$(PY) notebooks/build.py
