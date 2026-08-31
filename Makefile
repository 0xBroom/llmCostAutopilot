.DEFAULT_GOAL := help
.PHONY: help install lint typecheck test test-live check clean migrate run-api run-worker dashboard

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

## test: the default suite. No network, no spend, no `live` tests.
##       Coverage is gated on domain/ and application/ only — a global
##       number is gamed by testing adapters that contract tests should cover.
test:
	uv run pytest --cov --cov-report=term-missing --cov-fail-under=75

## test-live: opt-in. Hits real providers. Costs money. Needs credentials.
test-live:
	uv run pytest -m live

## migrate: apply Alembic migrations to the configured database (upgrade head)
migrate:
	uv run alembic upgrade head

## check: everything CI runs. If this is green, CI is green.
check: lint typecheck test

## run-api: the HTTP service, with reload
run-api:
	uv run uvicorn autopilot.interfaces.http.app:app --reload

## run-worker: the async verification worker
run-worker:
	uv run python -m autopilot.interfaces.worker

## dashboard: the Streamlit cost dashboard
dashboard:
	uv run streamlit run src/autopilot/interfaces/dashboard/app.py

## clean: remove build, cache and coverage artifacts
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
