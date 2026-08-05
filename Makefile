.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check clean

## help: list available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'

## install: create the venv and install from the committed lockfile
install:
	uv sync

## lint: ruff check + format check + architecture contracts
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run lint-imports

## format: apply ruff formatting and safe autofixes
format:
	uv run ruff check --fix .
	uv run ruff format .

## typecheck: mypy over src/ and tests/
typecheck:
	uv run mypy

## test: the test suite
test:
	uv run pytest

## check: everything CI runs. If this is green, CI is green.
check: lint typecheck test

## clean: remove build, cache and coverage artifacts
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
