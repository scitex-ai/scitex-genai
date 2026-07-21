# SciTeX GenAI — thin Makefile dispatcher.
# All real logic lives in scripts/ and pyproject.toml. Keep this short.

.PHONY: help install install-dev test test-fast lint format clean build

help:
	@echo "make install       Editable install (no extras)"
	@echo "make install-dev   Editable install with [dev] extras"
	@echo "make test          Run pytest with coverage"
	@echo "make test-fast     Run pytest -x -q (stop on first failure)"
	@echo "make lint          Run ruff check"
	@echo "make format        Run ruff format"
	@echo "make clean         Remove build artefacts"
	@echo "make build         Build sdist + wheel"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=src/scitex_genai --cov-report=term-missing

test-fast:
	pytest tests/ -x -q

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	find src tests -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build: clean
	python -m build
