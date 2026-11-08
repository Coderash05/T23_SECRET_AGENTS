.PHONY: help install test lint format eval eval-s2 docker-build docker-run clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies in editable mode"
	@echo "  make test          Run full test suite via pytest"
	@echo "  make lint          Run ruff linter and format check"
	@echo "  make format        Auto-format code and fix lint issues"
	@echo "  make eval          Run master end-to-end pipeline evaluation and baseline comparison"
	@echo "  make eval-s2       Run S2 classifier benchmark on synthetic corpus"
	@echo "  make docker-build  Build Docker container image"
	@echo "  make docker-run    Run Streamlit application via docker-compose"
	@echo "  make clean         Remove caches and build artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -e .[nlp,parsing,ui,dev]

test:
	$(PYTHON) -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

eval:
	$(PYTHON) scripts/evaluate_pipeline.py
	$(PYTHON) scripts/benchmark_llm_baseline.py

eval-s2:
	$(PYTHON) scripts/benchmark_synthetic.py

docker-build:
	docker build -t termsheet-ai .

docker-run:
	docker compose up

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache __pycache__
